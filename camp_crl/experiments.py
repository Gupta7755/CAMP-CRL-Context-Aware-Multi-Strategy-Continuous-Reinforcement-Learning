"""Experiment registry and multi-seed evaluation with bootstrap confidence intervals.

An experiment = controllers x scenarios x seeds, run in parallel SUMO processes. It is stored in
<output_dir>/experiments/<id>/:
  manifest.json   parameters, status, progress, timestamps
  results.json    per-run metrics (one entry per controller/scenario/seed)
  results.csv     same, flattened scalar metrics
  summary.json    aggregates (see `aggregate`)

Statistics: per (scenario, controller, metric) the mean, sample std and a percentile-bootstrap 95 %
CI of the mean over seeds. Versus the baseline, paired per-seed differences (candidate - baseline)
with a bootstrap 95 % CI. "CI excludes 0" is reported instead of a significance claim; with
10 seeds and many metrics, multiple-comparison effects are not corrected for.
"""

from __future__ import annotations

import csv
import json
import logging
import threading
import uuid
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from camp_crl.config import AppConfig
from camp_crl.metrics import improvement_pct

log = logging.getLogger(__name__)

# (metric, lower_is_better)
SUMMARY_METRICS: tuple[tuple[str, bool], ...] = (
    ("delay_mean_s", True), ("delay_median_s", True), ("delay_p95_s", True),
    ("queue_mean_veh", True), ("queue_max_veh", True), ("throughput_completed", False),
    ("unfinished", True), ("travel_time_mean_s", True), ("stops_mean", True), ("max_wait_s", True),
    ("approach_delay_std_s", True), ("spillback_steps", True), ("downstream_blocking_steps", True),
    ("phase_switches", True), ("safety_interventions_total", True), ("teleports", True),
)


def bootstrap_ci(values: list[float], n_boot: int = 10_000, seed: int = 0, level: float = 0.95
                 ) -> tuple[float, float] | None:
    """Percentile bootstrap CI of the mean. None for fewer than 2 values."""
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if v.size < 2:
        return None
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def _describe(values: list[float | None]) -> dict:
    v = [x for x in values if x is not None]
    if not v:
        return {"mean": None, "std": None, "n": 0, "ci95": None}
    return {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else None,
            "n": len(v), "ci95": bootstrap_ci(v)}


def aggregate(results: list[dict], baseline: str, splits: dict[str, str]) -> dict:
    """Summaries per scenario, paired comparisons vs baseline, and per-split means."""
    scenarios = sorted({r["scenario"] for r in results})
    controllers = list(dict.fromkeys(r["controller"] for r in results))
    per_scenario: dict = {}
    for sc in scenarios:
        rows = [r for r in results if r["scenario"] == sc]
        entry = {"split": splits.get(sc, "?"), "controllers": {}, "paired_vs_baseline": {}}
        for c in controllers:
            cr = [r for r in rows if r["controller"] == c]
            entry["controllers"][c] = {m: _describe([r.get(m) for r in cr]) for m, _ in SUMMARY_METRICS}
        base = {r["seed"]: r for r in rows if r["controller"] == baseline}
        for c in controllers:
            if c == baseline:
                continue
            cand = {r["seed"]: r for r in rows if r["controller"] == c}
            seeds = sorted(set(base) & set(cand))
            out = {}
            for m, lower in SUMMARY_METRICS:
                pairs = [(base[s].get(m), cand[s].get(m)) for s in seeds]
                pairs = [(b, k) for b, k in pairs if b is not None and k is not None]
                diffs = [k - b for b, k in pairs]
                ci = bootstrap_ci(diffs)
                bmean = float(np.mean([b for b, _ in pairs])) if pairs else None
                kmean = float(np.mean([k for _, k in pairs])) if pairs else None
                out[m] = {
                    "mean_diff": float(np.mean(diffs)) if diffs else None,
                    "ci95": ci,
                    "ci_excludes_zero": bool(ci is not None and (ci[0] > 0 or ci[1] < 0)),
                    "improvement_pct_of_means": (improvement_pct(bmean, kmean, lower)
                                                 if bmean is not None and kmean is not None else None),
                    "lower_is_better": lower,
                    "n_pairs": len(pairs),
                }
            entry["paired_vs_baseline"][c] = out
        per_scenario[sc] = entry
    by_split: dict = {}
    for split in sorted({splits.get(s, "?") for s in scenarios}):
        scs = [s for s in scenarios if splits.get(s) == split]
        by_split[split] = {
            c: {m: _mean_or_none([per_scenario[s]["controllers"][c][m]["mean"] for s in scs])
                for m, _ in SUMMARY_METRICS}
            for c in controllers
        }
        by_split[split]["_scenarios"] = scs
    return {"baseline": baseline, "controllers": controllers, "scenarios": per_scenario, "by_split": by_split,
            "metrics": [{"name": m, "lower_is_better": lo} for m, lo in SUMMARY_METRICS],
            "notes": ["Means over seeds; CI = percentile bootstrap (10k) of the mean.",
                      "Paired diff = candidate - baseline per seed; 'ci_excludes_zero' is not a corrected "
                      "significance test.",
                      "by_split = unweighted mean of per-scenario means."]}


def _mean_or_none(values: list[float | None]) -> float | None:
    v = [x for x in values if x is not None]
    return float(np.mean(v)) if v else None


def _run_one(cfg_data: dict, controller: str, scenario: str, seed: int, out_dir: str) -> dict:
    """Worker entry point (top-level so it can be pickled into a subprocess)."""
    from camp_crl.runner import run_episode

    cfg = AppConfig.model_validate(cfg_data)
    return run_episode(cfg, controller, scenario, seed, Path(out_dir)).metrics


@dataclass
class ExperimentParams:
    controllers: list[str]
    scenarios: list[str]
    seeds: list[int]
    baseline: str
    kind: str = "comparison"  # comparison | evaluation
    workers: int = 4


@dataclass
class ExperimentHandle:
    id: str
    dir: Path
    cancel: threading.Event = field(default_factory=threading.Event)


class ExperimentStore:
    def __init__(self, root: Path) -> None:
        self.root = root / "experiments"
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, exp_id: str) -> Path:
        if not exp_id.replace("-", "").isalnum():  # ids are generated; never accept paths
            raise KeyError(exp_id)
        p = self.root / exp_id
        if not (p / "manifest.json").is_file():
            raise KeyError(exp_id)
        return p

    def manifest(self, exp_id: str) -> dict:
        return json.loads((self.path(exp_id) / "manifest.json").read_text(encoding="utf-8"))

    def list(self) -> list[dict]:
        out = []
        for d in self.root.iterdir():
            m = d / "manifest.json"
            if m.is_file():
                try:
                    out.append(json.loads(m.read_text(encoding="utf-8")))
                except json.JSONDecodeError:
                    log.warning("unreadable manifest %s", m)
        return sorted(out, key=lambda m: m["created"], reverse=True)

    def read(self, exp_id: str, name: str) -> dict | list | None:
        if name not in ("results.json", "summary.json"):
            raise KeyError(name)
        f = self.path(exp_id) / name
        return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else None

    def csv_path(self, exp_id: str) -> Path | None:
        f = self.path(exp_id) / "results.csv"
        return f if f.is_file() else None

    def create(self, params: ExperimentParams) -> ExperimentHandle:
        exp_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        d = self.root / exp_id
        d.mkdir()
        total = len(params.controllers) * len(params.scenarios) * len(params.seeds)
        self._write_manifest(d, {"id": exp_id, "created": datetime.now().isoformat(timespec="seconds"),
                                 "status": "queued", "params": params.__dict__, "total_runs": total,
                                 "completed_runs": 0, "failed_runs": 0, "error": None, "finished": None})
        return ExperimentHandle(exp_id, d)

    @staticmethod
    def _write_manifest(d: Path, manifest: dict) -> None:
        tmp = d / "manifest.json.tmp"
        tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        tmp.replace(d / "manifest.json")

    def update(self, handle: ExperimentHandle, **changes) -> dict:
        m = json.loads((handle.dir / "manifest.json").read_text(encoding="utf-8"))
        m.update(changes)
        self._write_manifest(handle.dir, m)
        return m


def run_experiment(cfg: AppConfig, params: ExperimentParams, store: ExperimentStore, handle: ExperimentHandle,
                   on_progress: Callable[[dict], None] | None = None) -> dict:
    """Run all combinations in parallel processes; returns the summary. Honors handle.cancel."""
    jobs = [(c, s, seed) for s in params.scenarios for seed in params.seeds for c in params.controllers]
    runs_dir = handle.dir / "runs"
    cfg_data = cfg.model_dump(mode="json")
    results: list[dict] = []
    failed = 0
    store.update(handle, status="running")
    try:
        with ProcessPoolExecutor(max_workers=max(1, params.workers)) as pool:
            pending = {pool.submit(_run_one, cfg_data, c, s, seed, str(runs_dir)): (c, s, seed)
                       for c, s, seed in jobs}
            while pending:
                if handle.cancel.is_set():
                    for f in pending:
                        f.cancel()
                    break
                done, _ = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                for f in done:
                    c, s, seed = pending.pop(f)
                    try:
                        results.append(f.result())
                    except Exception as exc:  # keep going; record the failure
                        failed += 1
                        log.error("run failed controller=%s scenario=%s seed=%s: %s", c, s, seed, exc)
                    m = store.update(handle, completed_runs=len(results), failed_runs=failed)
                    if on_progress:
                        on_progress(m)
    except Exception as exc:
        store.update(handle, status="failed", error=str(exc), finished=datetime.now().isoformat(timespec="seconds"))
        raise
    results.sort(key=lambda r: (r["scenario"], r["controller"], r["seed"]))
    (handle.dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    _write_csv(handle.dir / "results.csv", results)
    summary = aggregate(results, params.baseline, {k: v.split for k, v in cfg.scenarios.items()}) if results else {}
    (handle.dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    status = "cancelled" if handle.cancel.is_set() else ("completed" if not failed else "completed_with_failures")
    store.update(handle, status=status, finished=datetime.now().isoformat(timespec="seconds"))
    return summary


def merge_into_evaluation(store: ExperimentStore, base_id: str, new_results: list[dict], splits: dict[str, str],
                          note: str) -> ExperimentHandle:
    """New 'evaluation' experiment = base results with the controllers in `new_results` replaced."""
    base = store.manifest(base_id)
    replaced = {r["controller"] for r in new_results}
    merged = [r for r in store.read(base_id, "results.json") if r["controller"] not in replaced] + new_results
    merged.sort(key=lambda r: (r["scenario"], r["controller"], r["seed"]))
    p = base["params"]
    params = ExperimentParams(list(dict.fromkeys(p["controllers"] + sorted(replaced))), p["scenarios"], p["seeds"],
                              p["baseline"], kind="evaluation", workers=p["workers"])
    handle = store.create(params)
    (handle.dir / "results.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
    _write_csv(handle.dir / "results.csv", merged)
    summary = aggregate(merged, p["baseline"], splits)
    summary["notes"].append(note)
    summary["derived_from"] = base_id
    (handle.dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    store.update(handle, status="completed", completed_runs=len(merged), total_runs=len(merged),
                 finished=datetime.now().isoformat(timespec="seconds"))
    return handle


def _write_csv(path: Path, results: list[dict]) -> None:
    keys: list[str] = []
    for r in results:
        for k, v in r.items():
            if not isinstance(v, (dict, list)) and k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
