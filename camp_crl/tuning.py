"""Grid search of the Fixed-Time green split on *training* scenarios only.

Objective: relative regret = mean over scenarios of (plan's mean delay on the scenario / best plan's
mean delay on that scenario). Each scenario weighs equally, so high-delay scenarios (e.g. bottlenecks,
~130 s) cannot dominate low-delay ones (~10 s) as they did with the earlier raw-mean objective.
Plans with any teleport or with more unfinished trips than `max_unfinished_ratio` of measured trips
are rejected, so a plan cannot look good by leaving vehicles stuck in the network.
"""

from __future__ import annotations

import itertools
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

from camp_crl.config import AppConfig
from camp_crl.metrics import mean_std
from camp_crl.experiments import _run_one
from camp_crl.sim.network import ROOT

log = logging.getLogger(__name__)
TUNED_FIXED_TIME_PATH = ROOT / "configs" / "tuned" / "fixed_time.json"


def tune_fixed_time(cfg: AppConfig, scenarios: list[str], seeds: list[int], grid: list[float],
                    output_dir: Path | None = None, max_unfinished_ratio: float = 0.1, workers: int = 1) -> dict:
    for name in scenarios:
        if name not in cfg.scenarios:
            raise ValueError(f"unknown scenario '{name}'")
        if cfg.scenarios[name].split != "train":
            raise ValueError(f"scenario '{name}' has split '{cfg.scenarios[name].split}'; tune only on train scenarios")
    t = cfg.timing
    bad = [g for g in grid if not t.min_green_s <= g <= t.max_green_s]
    if bad:
        raise ValueError(f"grid values {bad} outside [{t.min_green_s}, {t.max_green_s}]")
    out_dir = output_dir or ROOT / cfg.output_dir / "tuning"
    plans = list(itertools.product(grid, grid))
    futures = {}
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        for plan in plans:
            data = cfg.model_dump(mode="json")
            data["fixed_time"]["green_s"] = list(plan)
            for sc in scenarios:
                for seed in seeds:
                    futures[(plan, sc, seed)] = pool.submit(_run_one, data, "fixed_time", sc, seed, str(out_dir))
        metrics = {k: f.result() for k, f in futures.items()}
    per_sc = {(plan, sc): mean_std([metrics[(plan, sc, s)]["delay_mean_s"] for s in seeds])[0]
              for plan in plans for sc in scenarios}
    best_sc = {sc: min(v for (p, s2), v in per_sc.items() if s2 == sc and v is not None) for sc in scenarios}
    candidates = []
    for plan in plans:
        g_ns, g_ew = plan
        runs = [metrics[(plan, sc, seed)] for sc in scenarios for seed in seeds]
        raw, std, _ = mean_std([r["delay_mean_s"] for r in runs])
        ratios = [per_sc[(plan, sc)] / best_sc[sc] for sc in scenarios if per_sc[(plan, sc)] is not None]
        regret = sum(ratios) / len(ratios) if len(ratios) == len(scenarios) else None
        rejected = any(r["teleports"] > 0 or r["unfinished"] > max_unfinished_ratio * max(r["trips_measured"], 1)
                       for r in runs)
        candidates.append({"green_s": [g_ns, g_ew], "relative_regret": regret, "delay_mean_s": raw,
                           "delay_std_s": std, "per_scenario_delay_s": {sc: per_sc[(plan, sc)] for sc in scenarios},
                           "rejected": rejected})
        log.info("tune green=(%s, %s) regret=%.3f delay=%.2f rejected=%s", g_ns, g_ew,
                 regret or float("nan"), raw or float("nan"), rejected)
    valid = [c for c in candidates if not c["rejected"] and c["relative_regret"] is not None]
    if not valid:
        raise RuntimeError("no valid Fixed-Time plan in the grid")
    best = min(valid, key=lambda c: c["relative_regret"])
    return {
        "green_s": best["green_s"],
        "objective": "relative regret: mean over scenarios of plan delay / best plan delay; reject teleports "
                     f"or unfinished > {max_unfinished_ratio:.0%}",
        "scenarios": scenarios, "seeds": seeds, "grid": grid,
        "simulation": cfg.simulation.model_dump(), "timing": t.model_dump(),
        "created": datetime.now().isoformat(timespec="seconds"),
        "candidates": candidates,
    }


def save_tuned(result: dict, path: Path = TUNED_FIXED_TIME_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return path


def with_tuned_fixed_time(cfg: AppConfig, path: Path = TUNED_FIXED_TIME_PATH) -> AppConfig:
    """Return cfg with `fixed_time_tuned` loaded from `path` (validated), if the file exists."""
    if cfg.fixed_time_tuned is not None or not path.is_file():
        return cfg
    green = json.loads(path.read_text(encoding="utf-8"))["green_s"]
    return AppConfig.model_validate({**cfg.model_dump(), "fixed_time_tuned": {"green_s": green}})
