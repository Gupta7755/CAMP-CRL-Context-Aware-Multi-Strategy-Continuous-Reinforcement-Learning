"""Command-line entry points.

  python -m camp_crl.cli run --controller camp_rule --scenario balanced --seed 1
  python -m camp_crl.cli tune-fixed --scenarios balanced --seeds 1 2 3
  python -m camp_crl.cli compare --scenario ns_dominant --seeds 1 2 3
  python -m camp_crl.cli evaluate --splits train validation --seeds 1 2 3 4 5 6 7 8 9 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from camp_crl.config import AppConfig, load_config
from camp_crl.control.controllers import CONTROLLER_NAMES
from camp_crl.experiments import ExperimentParams, ExperimentStore, merge_into_evaluation, run_experiment
from camp_crl.metrics import improvement_pct, mean_std
from camp_crl.runner import run_episode
from camp_crl.sim.network import ROOT
from camp_crl.tuning import TUNED_FIXED_TIME_PATH, save_tuned, tune_fixed_time, with_tuned_fixed_time

COMPARE_METRICS = (
    ("delay_mean_s", True), ("delay_p95_s", True), ("queue_mean_veh", True), ("queue_max_veh", True),
    ("throughput_completed", False), ("unfinished", True), ("travel_time_mean_s", True),
    ("max_wait_s", True), ("approach_delay_std_s", True), ("phase_switches", True),
    ("safety_interventions_total", True), ("teleports", True),
)


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def _load(args: argparse.Namespace) -> AppConfig:
    return with_tuned_fixed_time(load_config(args.config), args.tuned)


def cmd_run(args: argparse.Namespace) -> int:
    cfg = _load(args)
    if args.gui:
        cfg = cfg.model_copy(update={"simulation": cfg.simulation.model_copy(update={"gui": True})})
    res = run_episode(cfg, args.controller, args.scenario, args.seed)
    print(json.dumps(res.metrics, indent=2))
    print(f"outputs: {res.run_dir}")
    return 0


def cmd_tune_fixed(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    scenarios = args.scenarios or [n for n, s in cfg.scenarios.items() if s.split == "train"]
    result = tune_fixed_time(cfg, scenarios, args.seeds, args.grid, workers=args.workers)
    path = save_tuned(result, args.tuned)
    print(f"{'green NS/EW':<14}{'rel. regret':>12}{'raw delay':>11}  rejected")
    for c in sorted(result["candidates"], key=lambda c: (c["rejected"], c["relative_regret"] or 1e9)):
        print(f"{str(c['green_s']):<14}{_fmt(c['relative_regret']):>12}{_fmt(c['delay_mean_s']):>11}  {c['rejected']}")
    print(f"selected green_s={result['green_s']} (tuned on {scenarios}, seeds {args.seeds}); saved {path}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    cfg = _load(args)
    split = cfg.scenarios[args.scenario].split
    if split == "test" and not args.allow_test:
        print(f"scenario '{args.scenario}' is a held-out test scenario; pass --allow-test only for final evaluation.")
        return 2
    controllers = list(args.controllers or CONTROLLER_NAMES)
    if "fixed_time_tuned" in controllers and cfg.fixed_time_tuned is None:
        print("note: skipping fixed_time_tuned (no tuned plan; run tune-fixed first)")
        controllers.remove("fixed_time_tuned")
    baseline = args.baseline or ("fixed_time_tuned" if "fixed_time_tuned" in controllers else "fixed_time")
    if baseline not in controllers:
        print(f"baseline '{baseline}' is not among the controllers {controllers}")
        return 2
    results: dict[str, list[dict]] = {c: [] for c in controllers}
    for seed in args.seeds:
        for c in controllers:
            results[c].append(run_episode(cfg, c, args.scenario, seed).metrics)
    out = ROOT / cfg.output_dir / f"compare_{args.scenario}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\nScenario '{args.scenario}' (split={split}), seeds={args.seeds}, warm-up={cfg.simulation.warmup_s}s, "
          f"duration={cfg.simulation.duration_s}s. Cells: mean +/- sample std over seeds.")
    if cfg.fixed_time_tuned is not None:
        print(f"fixed_time green={list(cfg.fixed_time.green_s)} (untuned), "
              f"fixed_time_tuned green={list(cfg.fixed_time_tuned.green_s)}")
    w = 19
    print(f"{'metric':<28}" + "".join(f"{c:>{w}}" for c in controllers))
    for key, _ in COMPARE_METRICS:
        cells = []
        for c in controllers:
            m, s, _ = mean_std([r[key] for r in results[c]])
            cells.append(f"{_fmt(m)} +/- {_fmt(s)}")
        print(f"{key:<28}" + "".join(f"{x:>{w}}" for x in cells))

    others = [c for c in controllers if c != baseline]
    print(f"\nPaired change vs {baseline} (mean over seeds of per-seed %; > 0 = better than baseline)")
    print(f"{'metric':<28}" + "".join(f"{c:>{w}}" for c in others))
    for key, lower in COMPARE_METRICS:
        cells = []
        for c in others:
            paired = [improvement_pct(b[key], k[key], lower) for b, k in zip(results[baseline], results[c])
                      if b[key] is not None and k[key] is not None]
            cells.append(_fmt(mean_std(paired)[0]))
        print(f"{key:<28}" + "".join(f"{x:>{w}}" for x in cells))
    print("Exploratory only: few seeds, untuned CAMP defaults, no significance test.")
    print(f"raw per-seed metrics: {out}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    cfg = _load(args)
    if "test" in args.splits and not args.allow_test:
        print("the test split is held out; pass --allow-test only for the final evaluation.")
        return 2
    scenarios = [n for n, s in cfg.scenarios.items() if s.split in args.splits]
    controllers = [c for c in CONTROLLER_NAMES if c != "fixed_time_tuned" or cfg.fixed_time_tuned is not None]
    baseline = "fixed_time_tuned" if "fixed_time_tuned" in controllers else "fixed_time"
    store = ExperimentStore(ROOT / cfg.output_dir)
    kind = "evaluation"
    if args.replace_in:  # re-run only some controllers on the base evaluation's scenarios and seeds
        if not args.controllers:
            print("--replace-in needs --controllers")
            return 2
        base = store.manifest(args.replace_in)["params"]
        scenarios, args.seeds, controllers = base["scenarios"], base["seeds"], args.controllers
        baseline, kind = controllers[0], "comparison"
    params = ExperimentParams(controllers, scenarios, args.seeds, baseline, kind=kind, workers=args.workers)
    handle = store.create(params)
    print(f"experiment {handle.id}: {len(controllers)} controllers x {len(scenarios)} scenarios x "
          f"{len(args.seeds)} seeds, {args.workers} workers")
    last = [0]

    def progress(m: dict) -> None:
        pct = int(100 * m["completed_runs"] / m["total_runs"])
        if pct >= last[0] + 10:
            last[0] = pct - pct % 10
            print(f"  {m['completed_runs']}/{m['total_runs']} runs ({m['failed_runs']} failed)", flush=True)

    summary = run_experiment(cfg, params, store, handle, progress)
    if args.replace_in:
        new_rows = store.read(handle.id, "results.json")
        merged = merge_into_evaluation(store, args.replace_in, new_rows, {k: v.split for k, v in cfg.scenarios.items()},
                                       f"Rows for {', '.join(controllers)} re-run in {handle.id} and replaced in "
                                       f"{args.replace_in}.")
        summary, handle = store.read(merged.id, "summary.json"), merged
        controllers = summary["controllers"]
    w = 24
    print()
    print(f"Mean delay per vehicle (s): mean [95% bootstrap CI] over {len(args.seeds)} seeds")
    print(f"{'scenario':<24}{'split':<12}" + "".join(f"{c:>{w}}" for c in controllers))
    for sc, e in summary["scenarios"].items():
        cells = []
        for c in controllers:
            d = e["controllers"][c]["delay_mean_s"]
            ci = d["ci95"]
            cells.append(f"{_fmt(d['mean'])} [{_fmt(ci[0]) if ci else 'n/a'}, {_fmt(ci[1]) if ci else 'n/a'}]")
        print(f"{sc:<24}{e['split']:<12}" + "".join(f"{x:>{w}}" for x in cells))
    print()
    print("By split (unweighted mean of scenario means), delay_mean_s:")
    for split, e in summary["by_split"].items():
        print(f"  {split:<12}" + "".join(f"{c}={_fmt(e[c]['delay_mean_s'])}  " for c in controllers))
    print()
    print(f"saved: {handle.dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="camp_crl")
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    p.add_argument("--tuned", type=Path, default=TUNED_FIXED_TIME_PATH, help="tuned Fixed-Time plan file")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run one episode")
    r.add_argument("--controller", choices=CONTROLLER_NAMES, required=True)
    r.add_argument("--scenario", required=True)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--gui", action="store_true", help="show the run in the SUMO GUI window")
    r.set_defaults(func=cmd_run)
    t = sub.add_parser("tune-fixed", help="grid-search the Fixed-Time split on training scenarios")
    t.add_argument("--scenarios", nargs="+", help="default: all train scenarios")
    t.add_argument("--workers", type=int, default=12)
    t.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    t.add_argument("--grid", type=float, nargs="+", default=[15, 25, 35, 45])
    t.set_defaults(func=cmd_tune_fixed)
    c = sub.add_parser("compare", help="run controllers over several seeds")
    c.add_argument("--scenario", required=True)
    c.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    c.add_argument("--controllers", nargs="+", choices=CONTROLLER_NAMES)
    c.add_argument("--baseline", choices=CONTROLLER_NAMES)
    c.add_argument("--allow-test", action="store_true", help="permit held-out test scenarios")
    c.set_defaults(func=cmd_compare)
    e = sub.add_parser("evaluate", help="controllers x scenarios x seeds with bootstrap CIs")
    e.add_argument("--splits", nargs="+", choices=["train", "validation", "test"], default=["train", "validation"])
    e.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    e.add_argument("--workers", type=int, default=12)
    e.add_argument("--allow-test", action="store_true", help="permit the held-out test split")
    e.add_argument("--controllers", nargs="+", choices=CONTROLLER_NAMES, help="with --replace-in: controllers to re-run")
    e.add_argument("--replace-in", help="base evaluation id whose rows for --controllers are replaced")
    e.set_defaults(func=cmd_evaluate)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
