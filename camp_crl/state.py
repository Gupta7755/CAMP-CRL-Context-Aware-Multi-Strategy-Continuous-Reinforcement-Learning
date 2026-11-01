"""Traffic-state schema and TraCI collector.

Definitions (per approach = one incoming edge of two lanes, ~190 m long):
- vehicle_count: vehicles on the approach edge this step.
- queue_veh: halting vehicles (speed < 0.1 m/s, SUMO's halting threshold) on the edge.
- arrival_veh_per_h: vehicles newly seen on the edge, EWMA-smoothed, expressed per hour.
- arrivals_step: raw count of vehicles newly seen on the edge this step (for change detection).
- mean_speed_mps: SUMO edge mean speed; SUMO reports the speed limit when the edge is empty.
- occupancy_pct: SUMO edge occupancy x 100. SUMO 1.28 returns a fraction in [0, 1]
  (verified empirically), which is converted to percent here.
- max_wait_s / mean_wait_s: accumulated stopped time of vehicles currently on the edge.
- spillback_ratio: estimated queue length / edge storage length, where queue length
  = queue_veh * 7.5 m (vehicle 5 m + gap 2.5 m) / lanes. 1.0 means the queue reaches the upstream end.
- downstream_occupancy_pct: occupancy of the approach's through-movement exit edge.
- downstream_count: vehicles on that through-movement exit edge.
- downstream_fill_ratio: all vehicles on that exit x 7.5 m / (lanes x exit length), clipped to 1.
  ~0.03 in free flow, -> 1 when the exit is packed. Occupancy (~24 %) and halting counts (crawling
  vehicles exceed 0.1 m/s) both stayed low in a measured 0.3 m/s spillback case.
- downstream_congestion: 1 - (exit mean speed / free-flow speed 13.89 m/s), 0 when the exit is empty.
  A slowed exit blocks the junction before it fills (vehicles stop inside the junction waiting to
  enter), so this is the overflow signal used by CAMP; the fill ratio is used for the blocking metric.
- stopline_count: vehicles within `detection_zone_m` of the stop line (virtual presence
  detector used by the actuated baseline).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import numpy as np

from camp_crl.config import APPROACHES, SensorModel
from camp_crl.sim.network import THROUGH_EXIT

VEHICLE_SPACE_M = 7.5
FREE_FLOW_MPS = 13.89  # network speed limit


@dataclass(frozen=True)
class ApproachState:
    vehicle_count: int
    queue_veh: float
    arrival_veh_per_h: float
    mean_speed_mps: float
    occupancy_pct: float
    max_wait_s: float
    mean_wait_s: float
    spillback_ratio: float
    downstream_occupancy_pct: float
    downstream_count: int = 0
    stopline_count: int = 0
    arrivals_step: int = 0
    downstream_fill_ratio: float = 0.0
    downstream_congestion: float = 0.0


@dataclass(frozen=True)
class TrafficState:
    """Snapshot at simulation time `time_s`. `source` records where the values came from."""

    time_s: float
    approaches: dict[str, ApproachState]
    source: str = "sumo_ground_truth"
    stale: tuple[str, ...] = ()  # approaches whose values are repeated from an earlier step

    def to_dict(self) -> dict:
        return {"time_s": self.time_s, "source": self.source, "stale": list(self.stale),
                "approaches": {k: asdict(v) for k, v in self.approaches.items()}}


class StateCollector:
    """Reads per-approach state from a live TraCI connection."""

    def __init__(self, conn, arrival_alpha: float = 0.2, detection_zone_m: float = 40.0) -> None:
        self.conn = conn
        self.detection_zone_m = detection_zone_m
        self.alpha = arrival_alpha
        self._prev_ids: dict[str, set[str]] = {a: set() for a in APPROACHES}
        self._arrival_ewma: dict[str, float] = {a: 0.0 for a in APPROACHES}
        self._last_time: float | None = None
        self._storage_m: dict[str, float] = {}
        self._lanes: dict[str, int] = {}
        for a in APPROACHES:
            edge = f"{a}2C"
            n = conn.edge.getLaneNumber(edge)
            self._lanes[a] = n
            self._storage_m[a] = conn.lane.getLength(f"{edge}_0")
        self._exit_storage: dict[str, float] = {
            a: conn.edge.getLaneNumber(THROUGH_EXIT[a]) * conn.lane.getLength(f"{THROUGH_EXIT[a]}_0") for a in APPROACHES}

    def collect(self) -> TrafficState:
        c = self.conn
        t = c.simulation.getTime()
        dt = 0.0 if self._last_time is None else t - self._last_time
        self._last_time = t
        out: dict[str, ApproachState] = {}
        for a in APPROACHES:
            edge = f"{a}2C"
            ids = set(c.edge.getLastStepVehicleIDs(edge))
            new = len(ids - self._prev_ids[a])
            self._prev_ids[a] = ids
            if dt > 0:
                inst = new / dt * 3600.0
                self._arrival_ewma[a] = self.alpha * inst + (1 - self.alpha) * self._arrival_ewma[a]
            waits = [c.vehicle.getAccumulatedWaitingTime(v) for v in ids]
            queue = float(c.edge.getLastStepHaltingNumber(edge))
            near = sum(1 for v in ids
                       if self._storage_m[a] - c.vehicle.getLanePosition(v) <= self.detection_zone_m)
            spill = queue * VEHICLE_SPACE_M / self._lanes[a] / self._storage_m[a]
            out[a] = ApproachState(
                vehicle_count=len(ids),
                queue_veh=queue,
                arrival_veh_per_h=self._arrival_ewma[a],
                mean_speed_mps=float(c.edge.getLastStepMeanSpeed(edge)),
                occupancy_pct=100.0 * c.edge.getLastStepOccupancy(edge),
                max_wait_s=max(waits, default=0.0),
                mean_wait_s=(sum(waits) / len(waits)) if waits else 0.0,
                spillback_ratio=min(spill, 1.0),
                downstream_occupancy_pct=100.0 * c.edge.getLastStepOccupancy(THROUGH_EXIT[a]),
                downstream_count=c.edge.getLastStepVehicleNumber(THROUGH_EXIT[a]),
                stopline_count=near,
                arrivals_step=new,
                downstream_fill_ratio=min(c.edge.getLastStepVehicleNumber(THROUGH_EXIT[a]) * VEHICLE_SPACE_M
                                          / self._exit_storage[a], 1.0),
                downstream_congestion=(max(0.0, 1.0 - c.edge.getLastStepMeanSpeed(THROUGH_EXIT[a]) / FREE_FLOW_MPS)
                                       if c.edge.getLastStepVehicleNumber(THROUGH_EXIT[a]) else 0.0),
            )
        return TrafficState(time_s=t, approaches=out)


def is_finite_state(state: ApproachState) -> bool:
    """True when every numeric field is finite (used to detect corrupted observations)."""
    return all(math.isfinite(float(v)) for v in asdict(state).values())


NOISY_FIELDS: tuple[str, ...] = ("vehicle_count", "queue_veh", "arrival_veh_per_h", "occupancy_pct",
                                 "downstream_count", "stopline_count", "downstream_occupancy_pct", "arrivals_step",
                                 "downstream_fill_ratio", "downstream_congestion")


class SensorDegrader:
    """Applies a SensorModel to ground-truth states before controllers see them.

    Seeded independently of SUMO so a given (scenario, seed) always degrades identically.
    """

    def __init__(self, model: SensorModel, seed: int) -> None:
        self.model = model
        self.rng = np.random.default_rng(seed + 7919)
        self._last: dict[str, ApproachState] = {}

    @property
    def active(self) -> bool:
        return self.model.noise_std_frac > 0 or self.model.dropout_prob > 0

    def apply(self, state: TrafficState) -> TrafficState:
        if not self.active:
            return state
        out: dict[str, ApproachState] = {}
        stale: list[str] = []
        for a in APPROACHES:
            if self.rng.random() < self.model.dropout_prob and a in self._last:
                out[a] = self._last[a]
                stale.append(a)
                continue
            st = state.approaches[a]
            changes = {}
            for f in NOISY_FIELDS:
                v = max(0.0, float(getattr(st, f)) * (1.0 + self.rng.normal(0.0, self.model.noise_std_frac)))
                if f in ("downstream_fill_ratio", "downstream_congestion"):
                    v = min(v, 1.0)
                changes[f] = int(round(v)) if isinstance(getattr(st, f), int) else v
            out[a] = replace(st, **changes)
            self._last[a] = out[a]
        return TrafficState(state.time_s, out, source="degraded_sensor", stale=tuple(stale))
