"""Episode metrics from SUMO tripinfo output and per-step logs.

Measurement window: only trips that *depart* at or after warm-up are counted.
- delay = SUMO `timeLoss` (s): travel time lost versus driving at desired speed.
- throughput = completed trips (departed after warm-up and arrived before the end).
- unfinished = trips that departed after warm-up but were still in the network at the end;
  their delay is NOT in the delay statistics, so always read delay together with this count.
- queue = total halting vehicles over the four approaches, sampled each step after warm-up.
- spillback_steps / downstream_blocking_steps (added by the runner): steps after warm-up where an
  approach queue reaches 90 % of its length / vehicles fill >= 50 % of a through-exit's storage.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from camp_crl.config import APPROACHES


@dataclass(frozen=True)
class Trip:
    vid: str
    approach: str
    depart: float
    arrival: float  # -1 when unfinished
    duration: float
    time_loss: float
    waiting_time: float
    waiting_count: int

    @property
    def finished(self) -> bool:
        return self.arrival >= 0


def parse_tripinfo(path: Path) -> list[Trip]:
    if not path.is_file():
        raise FileNotFoundError(f"tripinfo file missing: {path}")
    trips = []
    for el in ET.parse(path).getroot().iter("tripinfo"):
        lane = el.get("departLane", "")
        trips.append(Trip(
            vid=el.get("id", ""),
            approach=lane.split("2")[0] if "2C_" in lane else "?",
            depart=float(el.get("depart", "nan")),
            arrival=float(el.get("arrival", "-1")),
            duration=float(el.get("duration", "nan")),
            time_loss=float(el.get("timeLoss", "nan")),
            waiting_time=float(el.get("waitingTime", "nan")),
            waiting_count=int(el.get("waitingCount", "0")),
        ))
    return trips


def _stat(values: list[float], fn) -> float | None:
    return float(fn(values)) if values else None


def summarize(trips: list[Trip], step_queues: list[float], warmup_s: float,
              phase_switches: int, interventions: dict[str, int], teleports: int) -> dict:
    """Aggregate one episode. Missing values are None, never 0, so they cannot be mistaken for results."""
    measured = [t for t in trips if t.depart >= warmup_s]
    done = [t for t in measured if t.finished]
    delays = [t.time_loss for t in done]
    per_app_delay = {a: _stat([t.time_loss for t in done if t.approach == a], np.mean) for a in APPROACHES}
    app_means = [v for v in per_app_delay.values() if v is not None]
    return {
        "trips_measured": len(measured),
        "throughput_completed": len(done),
        "unfinished": len(measured) - len(done),
        "delay_mean_s": _stat(delays, np.mean),
        "delay_median_s": _stat(delays, np.median),
        "delay_p95_s": _stat(delays, lambda v: np.percentile(v, 95)),
        "travel_time_mean_s": _stat([t.duration for t in done], np.mean),
        "stops_mean": _stat([float(t.waiting_count) for t in done], np.mean),
        "max_wait_s": _stat([t.waiting_time for t in measured], max),
        "queue_mean_veh": _stat(step_queues, np.mean),
        "queue_max_veh": _stat(step_queues, max),
        "delay_by_approach_s": per_app_delay,
        "approach_delay_std_s": _stat(app_means, np.std) if len(app_means) > 1 else None,
        "phase_switches": phase_switches,
        "safety_interventions": interventions,
        "safety_interventions_total": sum(interventions.values()),
        "teleports": teleports,
    }


def improvement_pct(baseline: float, candidate: float, lower_is_better: bool = True,
                    epsilon: float = 1e-6, min_abs_baseline: float = 1e-3) -> float | None:
    """100 * (baseline - candidate) / max(|baseline|, epsilon) for lower-is-better metrics.

    Returns None when the baseline is near zero (|baseline| < min_abs_baseline), where a
    percentage would be misleading, or when either input is not finite.
    """
    if not (math.isfinite(baseline) and math.isfinite(candidate)) or abs(baseline) < min_abs_baseline:
        return None
    num = (baseline - candidate) if lower_is_better else (candidate - baseline)
    return 100.0 * num / max(abs(baseline), epsilon)


def mean_std(values: list[float | None]) -> tuple[float | None, float | None, int]:
    """Mean and sample std (ddof=1) over the non-None values, plus how many were used."""
    v = [x for x in values if x is not None]
    if not v:
        return None, None, 0
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else None), len(v)
