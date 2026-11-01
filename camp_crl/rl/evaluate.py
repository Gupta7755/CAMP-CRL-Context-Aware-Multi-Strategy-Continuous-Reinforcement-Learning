"""Independent evaluation of trained RL policies, merged with the Phase-1 baseline evaluation.

  python -m camp_crl.rl.evaluate --runs runs/rl/<id1> runs/rl/<id2> ... --baseline-experiment <exp id>

For each trained run and each fusion mode, the policy is rolled out on every scenario of the baseline
experiment with the same SUMO seeds, so results pair with the baselines seed by seed. Controller names:
  "<algo>_<obs>"            pure RL policy (mode rl)
  "camp_crl_<mode>"         CAMP-state TD3 fused with the CAMP prior (modes camp / fixed / gated)
Several training seeds of one controller are averaged per (scenario, SUMO seed) before pairing; the
spread across training seeds is reported separately. The merged experiment is saved with kind
"evaluation" so the dashboard shows it. Nothing here is used for training or model selection.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from camp_crl.config import AppConfig
from camp_crl.experiments import ExperimentParams, ExperimentStore, _write_csv, aggregate
from camp_crl.sim.network import ROOT

log = logging.getLogger(__name__)
FUSION_MODES = ("camp", "fixed", "gated")


def rollout(app_data: dict, rl_data: dict, run_dir: str, mode: str, scenario: str, sumo_seed: int) -> dict:
    """One full episode with a trained policy (or the CAMP prior); returns Phase-1 style metrics."""
    import torch
    from stable_baselines3 import DDPG, SAC, TD3

    from camp_crl.control.controllers import CampRuleController
    from camp_crl.rl.config import RLConfig
    from camp_crl.rl.env import TrafficSignalEnv
    from camp_crl.rl.fusion import FusedPolicy, ObsStats

    torch.set_num_threads(1)
    app, rl = AppConfig.model_validate(app_data), RLConfig.model_validate(rl_data)
    run = Path(run_dir)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    algo_cls = {"td3": TD3, "ddpg": DDPG, "sac": SAC}[manifest["algo"]]
    model = algo_cls.load(run / "best_model.zip", device="cpu")
    stats = ObsStats.load(run / "obs_stats.json") if (run / "obs_stats.json").is_file() else None
    policy = FusedPolicy(mode, model, stats)
    rule = CampRuleController(app)
    name = f"{manifest['algo']}_{manifest['obs']}" if mode == "rl" else f"camp_crl_{mode}"
    env = TrafficSignalEnv(app, rl, scenarios=[scenario], obs_mode=manifest["obs"], run_label=name)
    try:
        obs, _ = env.reset(seed=sumo_seed, options={"scenario": scenario, "sumo_seed": sumo_seed})
        done, info = False, {}
        while not done:
            obs, _, term, trunc, info = env.step(policy.act(env, obs, rule))
            done = term or trunc
    finally:
        env.close()
    m = dict(info["episode_metrics"])
    alphas = [r["alpha"] for r in policy.log]
    m.update(controller=name, training_seed=manifest["seed"], run_id=manifest["id"], fusion_mode=mode,
             alpha_mean=float(np.mean(alphas)) if alphas else None,
             alpha_p10=float(np.percentile(alphas, 10)) if alphas else None,
             alpha_p90=float(np.percentile(alphas, 90)) if alphas else None, decisions=len(alphas))
    return m


def average_over_training_seeds(rows: list[dict]) -> tuple[list[dict], dict]:
    """Mean of numeric metrics per (controller, scenario, SUMO seed); plus per-controller seed spread."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["controller"], r["scenario"], r["seed"])].append(r)
    out = []
    for (c, sc, seed), rs in groups.items():
        merged = {"controller": c, "scenario": sc, "seed": seed, "split": rs[0]["split"], "n_training_seeds": len(rs)}
        for k, v in rs[0].items():
            if k in merged or not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            vals = [x[k] for x in rs if isinstance(x.get(k), (int, float))]
            merged[k] = float(np.mean(vals)) if vals else None
        out.append(merged)
    spread: dict = {}
    by_ctrl: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("delay_mean_s") is not None:
            by_ctrl[r["controller"]][r["training_seed"]].append(r["delay_mean_s"])
    for c, per_seed in by_ctrl.items():
        means = {s: float(np.mean(v)) for s, v in per_seed.items()}
        spread[c] = {"delay_mean_s_by_training_seed": means,
                     "std_across_training_seeds": float(np.std(list(means.values()), ddof=1)) if len(means) > 1 else None}
    return out, spread


def evaluate(runs: list[Path], baseline_exp: str, workers: int = 12, modes_for_camp_state_td3=FUSION_MODES,
             out_root: Path | None = None) -> Path:
    store = ExperimentStore(out_root or ROOT / "runs")
    base_manifest = store.manifest(baseline_exp)
    base_results = store.read(baseline_exp, "results.json")
    cfg_data = json.loads(next((store.path(baseline_exp) / "runs").glob("*/config.json")).read_text())["config"]
    app = AppConfig.model_validate(cfg_data)
    scenarios, seeds = base_manifest["params"]["scenarios"], base_manifest["params"]["seeds"]
    jobs = []
    camp_prior_done = False  # the CAMP-prior-only mode ignores the model: run it once, not per training seed
    for run in runs:
        m = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        if m.get("status") != "completed":
            raise ValueError(f"{run} did not complete training")
        modes = ["rl"] + (list(modes_for_camp_state_td3) if (m["algo"], m["obs"]) == ("td3", "camp") else [])
        if "camp" in modes and camp_prior_done:
            modes.remove("camp")
        camp_prior_done |= "camp" in modes
        for mode in modes:
            for sc in scenarios:
                for seed in seeds:
                    jobs.append((cfg_data, m["rl_config"], str(run), mode, sc, seed))
    log.info("%d RL rollouts on %d scenarios x %d seeds", len(jobs), len(scenarios), len(seeds))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rl_rows = list(pool.map(rollout, *zip(*jobs)))
    rl_avg, spread = average_over_training_seeds(rl_rows)
    baseline = base_manifest["params"]["baseline"]
    merged = base_results + rl_avg
    params = ExperimentParams(sorted({r["controller"] for r in merged}, key=lambda c: (c.startswith(("td3", "ddpg", "camp_crl")), c)),
                              scenarios, seeds, baseline, kind="evaluation", workers=workers)
    handle = store.create(params)
    (handle.dir / "rl_rollouts.json").write_text(json.dumps(rl_rows, indent=2), encoding="utf-8")
    (handle.dir / "results.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
    _write_csv(handle.dir / "results.csv", merged)
    summary = aggregate(merged, baseline, {k: v.split for k, v in app.scenarios.items()})
    summary["training_seed_spread"] = spread
    alpha: dict = defaultdict(lambda: defaultdict(list))
    for r in rl_rows:
        if r["controller"].startswith("camp_crl_") and r.get("alpha_mean") is not None:
            alpha[r["controller"]][r["scenario"]].append(r["alpha_mean"])
    summary["fusion_alpha"] = {c: {sc: float(np.mean(v)) for sc, v in per.items()} for c, per in alpha.items()}
    summary["rl_runs"] = [r.name for r in runs]
    summary["baseline_experiment"] = baseline_exp
    summary["notes"].append("RL rows: metrics averaged over training seeds per (scenario, SUMO seed) before pairing.")
    (handle.dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    store.update(handle, status="completed", completed_runs=len(merged), total_runs=len(merged))
    return handle.dir


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("camp_crl.sim.traci_adapter", "camp_crl.runner"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", type=Path, required=True)
    p.add_argument("--baseline-experiment", required=True)
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args()
    print(f"saved: {evaluate(args.runs, args.baseline_experiment, args.workers)}")


if __name__ == "__main__":
    main()
