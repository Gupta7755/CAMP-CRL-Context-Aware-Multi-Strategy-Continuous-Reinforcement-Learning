"""Validated configuration schema.

All numeric values are *starting defaults*, not tuned or approved values. Tune only on
training/validation scenarios, never on held-out test scenarios.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

APPROACHES: tuple[str, ...] = ("N", "E", "S", "W")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SimulationConfig(_Strict):
    """SUMO run settings. Times in seconds."""

    step_length_s: float = Field(1.0, gt=0, le=5)
    duration_s: int = Field(1800, gt=0, description="Simulated time including warm-up.")
    warmup_s: int = Field(300, ge=0, description="Trips departing before this are excluded from metrics.")
    time_to_teleport_s: int = Field(300, description="SUMO jam teleport threshold; -1 disables.")
    gui: bool = False

    @model_validator(mode="after")
    def _warmup_shorter_than_run(self) -> "SimulationConfig":
        if self.warmup_s >= self.duration_s:
            raise ValueError("warmup_s must be smaller than duration_s")
        return self


class TimingConfig(_Strict):
    """Signal timing limits enforced by the safety filter. Seconds."""

    min_green_s: float = Field(10.0, gt=0)
    max_green_s: float = Field(60.0, gt=0)
    yellow_s: float = Field(3.0, ge=3.0, description="Yellow change interval (>=3 s).")
    all_red_s: float = Field(2.0, ge=0)
    max_red_s: float = Field(120.0, gt=0, description="Starvation bound: a phase red this long is forced green.")

    @model_validator(mode="after")
    def _consistent(self) -> "TimingConfig":
        if self.min_green_s > self.max_green_s:
            raise ValueError("min_green_s must be <= max_green_s")
        if self.max_red_s < self.max_green_s + self.yellow_s + self.all_red_s:
            raise ValueError("max_red_s must be >= max_green_s + yellow_s + all_red_s")
        return self


class FixedTimeConfig(_Strict):
    """Green time per phase (NS, EW) in seconds; must lie within the timing limits."""

    green_s: tuple[float, float] = (30.0, 30.0)


class ActuatedConfig(_Strict):
    """Gap-out actuation using a virtual presence detector near the stop line.

    After min green, the current phase is extended while vehicles are present in the
    detection zone; it ends once the zone has been empty for `gap_s` and a conflicting
    approach has a vehicle waiting, or when `max_green_s` is reached.
    """

    detection_zone_m: float = Field(40.0, gt=0, description="Zone length upstream of the stop line (m).")
    gap_s: float = Field(3.0, gt=0, description="Empty-zone time that ends the green (s).")
    max_green_s: float = Field(45.0, gt=0, description="Actuated max green; must be <= timing.max_green_s.")


class NormalizationBounds(_Strict):
    """[min, max] used by x_hat = clip((x - min) / (max - min), 0, 1)."""

    queue_veh: tuple[float, float] = (0.0, 40.0)
    arrival_veh_per_h: tuple[float, float] = (0.0, 1200.0)
    max_wait_s: tuple[float, float] = (0.0, 120.0)
    occupancy_pct: tuple[float, float] = (0.0, 50.0)
    spillback_ratio: tuple[float, float] = (0.0, 1.0)


class PressureWeights(_Strict):
    """Weights of P_i = wQ*Q + wA*A + wW*W + wD*D + wR*R. Must sum to 1 so P is in [0, 1]."""

    queue: float = Field(0.35, ge=0)
    arrival: float = Field(0.15, ge=0)
    wait: float = Field(0.20, ge=0)
    density: float = Field(0.15, ge=0)
    spillback: float = Field(0.15, ge=0)

    @model_validator(mode="after")
    def _sum_to_one(self) -> "PressureWeights":
        total = self.queue + self.arrival + self.wait + self.density + self.spillback
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"pressure weights must sum to 1, got {total:.6f}")
        return self


class MomentumConfig(_Strict):
    """M_i = bQ*qg + bA*ag - bV*vg with each growth term scaled into [-1, 1]; b's sum to 1."""

    beta_queue: float = Field(0.5, ge=0)
    beta_arrival: float = Field(0.3, ge=0)
    beta_speed: float = Field(0.2, ge=0)
    ewma_alpha: float = Field(0.2, gt=0, le=1, description="Smoothing for derivatives and arrival rate.")
    queue_rate_scale: float = Field(0.5, gt=0, description="veh/s mapped to growth 1.0")
    arrival_rate_scale: float = Field(60.0, gt=0, description="(veh/h)/s mapped to growth 1.0")
    speed_rate_scale: float = Field(1.0, gt=0, description="(m/s)/s mapped to growth 1.0")

    @model_validator(mode="after")
    def _sum_to_one(self) -> "MomentumConfig":
        total = self.beta_queue + self.beta_arrival + self.beta_speed
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"momentum betas must sum to 1, got {total:.6f}")
        return self


class CampConfig(_Strict):
    bounds: NormalizationBounds = NormalizationBounds()
    pressure: PressureWeights = PressureWeights()
    momentum: MomentumConfig = MomentumConfig()
    risk_horizon_s: float = Field(10.0, ge=0, description="tau in F = clip(P + tau*kM*M, 0, Fmax).")
    risk_k_m: float = Field(0.03, ge=0, description="1/s")
    risk_max: float = Field(1.0, gt=0)
    fairness_ref_wait_s: float = Field(90.0, gt=0, description="W_ref in S = clip(W/W_ref, 0, 1).")
    lambda_momentum: float = Field(0.3, ge=0)
    lambda_risk: float = Field(0.3, ge=0)
    lambda_fairness: float = Field(0.5, ge=0)
    # Explainability signals (reported, not used by the controller in Phase 1).
    trend_threshold_veh_per_s: float = Field(0.05, gt=0, description="|smoothed dQ/dt| above this = RISING/FALLING")
    # Poisson likelihood-ratio CUSUM on per-step arrival counts. Calibrated on synthetic Poisson
    # arrivals (~0.7 false alarms per approach-hour, +550 veh/h surge detected in ~75 s median);
    # not yet validated on SUMO data.
    cusum_baseline_alpha: float = Field(1 / 600, gt=0, le=1, description="Slow EWMA (per s) of the arrival-rate baseline.")
    cusum_ratio: float = Field(2.0, gt=1, description="Alternative rate = max(ratio * baseline, baseline + min_shift).")
    cusum_min_shift_veh_per_h: float = Field(250.0, gt=0)
    cusum_floor_veh_per_h: float = Field(30.0, gt=0, description="Lower bound on the baseline rate.")
    cusum_h: float = Field(6.0, gt=0, description="Alarm threshold on the log-likelihood-ratio statistic.")
    cusum_warmup_s: float = Field(120.0, ge=0, description="No CUSUM accumulation before this time.")
    pattern_low_pressure: float = Field(0.1, ge=0, le=1)
    pattern_dominance_gap: float = Field(0.2, ge=0, le=1)
    pattern_congested_pressure: float = Field(0.55, ge=0, le=1)


class StrategyConfig(_Strict):
    """Rule thresholds (dimensionless, on normalized CAMP terms) and hysteresis."""

    fairness_threshold: float = Field(0.8, ge=0, le=1)
    overflow_congestion: float = Field(0.5, ge=0, le=1, description="Through-exit congestion (1 - speed/free-flow) that triggers OVERFLOW_PREVENTION.")
    congestion_pressure: float = Field(0.55, ge=0, le=1)
    dominance_gap: float = Field(0.2, ge=0, le=1)
    balanced_min_pressure: float = Field(0.2, ge=0, le=1)
    balanced_max_gap: float = Field(0.1, ge=0, le=1)
    confirm_steps: int = Field(3, ge=1, description="Consecutive evaluations a new strategy must persist.")
    min_hold_s: float = Field(15.0, ge=0, description="Minimum time a strategy stays active.")
    switch_margin: dict[str, float] = Field(
        default_factory=lambda: {
            "NORMAL_ADAPTIVE": 0.10,
            "BALANCED_FLOW": 0.05,
            "DIRECTIONAL_PRIORITY": 0.20,
            "CONGESTION_RECOVERY": 0.15,
            "FAIRNESS_RECOVERY": 0.0,
            "OVERFLOW_PREVENTION": 0.10,
        }
    )
    overflow_penalty: float = Field(0.5, ge=0, description="Urgency reduction per unit through-exit congestion.")


def _check_demand(d: dict[str, float]) -> None:
    if set(d) != set(APPROACHES):
        raise ValueError(f"demand must define exactly the approaches {APPROACHES}")
    if any(v < 0 or v > 3600 for v in d.values()):
        raise ValueError("demand must be within [0, 3600] veh/h (Bernoulli per-second insertion)")


class DemandSegment(_Strict):
    """From `start_s` onwards (until the next segment) demand switches to `demand_veh_per_h`."""

    start_s: float = Field(gt=0)
    demand_veh_per_h: dict[str, float]

    @model_validator(mode="after")
    def _valid(self) -> "DemandSegment":
        _check_demand(self.demand_veh_per_h)
        return self


EXIT_EDGES: tuple[str, ...] = ("C2N", "C2E", "C2S", "C2W")


class BottleneckEvent(_Strict):
    """Downstream capacity drop: the exit edge's speed limit is lowered during [start_s, end_s)."""

    edge: str
    speed_mps: float = Field(gt=0, le=13.89)
    start_s: float = Field(ge=0)
    end_s: float

    @model_validator(mode="after")
    def _valid(self) -> "BottleneckEvent":
        if self.edge not in EXIT_EDGES:
            raise ValueError(f"bottleneck edge must be one of {EXIT_EDGES}")
        if self.end_s <= self.start_s:
            raise ValueError("bottleneck end_s must be after start_s")
        return self


class SensorModel(_Strict):
    """Degrades what the *controllers* observe; metrics always use SUMO ground truth.

    Each step, each approach observation is independently dropped with `dropout_prob`
    (the controller then sees the last valid value, flagged stale), otherwise count-like
    fields are multiplied by (1 + N(0, noise_std_frac)) and clipped at 0.
    """

    noise_std_frac: float = Field(0.0, ge=0, le=1)
    dropout_prob: float = Field(0.0, ge=0, lt=1)


class Scenario(_Strict):
    """Demand in vehicles/hour per approach, with turning shares (left, straight, right).

    `demand_veh_per_h` applies from t=0; optional `profile` segments change it later.
    """

    name: str
    split: Literal["train", "validation", "test"] = "train"
    description: str = ""
    demand_veh_per_h: dict[str, float]
    profile: list[DemandSegment] = Field(default_factory=list)
    turn_shares: tuple[float, float, float] = (0.2, 0.6, 0.2)
    bottlenecks: list[BottleneckEvent] = Field(default_factory=list)
    sensor: SensorModel = SensorModel()

    @model_validator(mode="after")
    def _valid(self) -> "Scenario":
        _check_demand(self.demand_veh_per_h)
        starts = [s.start_s for s in self.profile]
        if starts != sorted(starts) or len(set(starts)) != len(starts):
            raise ValueError("profile segments must have strictly increasing start_s")
        if abs(sum(self.turn_shares) - 1.0) > 1e-6 or min(self.turn_shares) < 0:
            raise ValueError("turn_shares must be non-negative and sum to 1")
        return self

    def segments(self, duration_s: float) -> list[tuple[float, float, dict[str, float]]]:
        """(begin, end, demand) pieces covering [0, duration_s)."""
        points = [(0.0, self.demand_veh_per_h)] + [(s.start_s, s.demand_veh_per_h) for s in self.profile]
        points = [p for p in points if p[0] < duration_s]
        ends = [p[0] for p in points[1:]] + [float(duration_s)]
        return [(b, e, d) for (b, d), e in zip(points, ends)]


class AppConfig(_Strict):
    simulation: SimulationConfig = SimulationConfig()
    timing: TimingConfig = TimingConfig()
    fixed_time: FixedTimeConfig = FixedTimeConfig()
    fixed_time_tuned: FixedTimeConfig | None = Field(
        None, description="Green split from `tune-fixed` on training scenarios; None until tuned.")
    actuated: ActuatedConfig = ActuatedConfig()
    camp: CampConfig = CampConfig()
    strategy: StrategyConfig = StrategyConfig()
    scenarios: dict[str, Scenario]
    output_dir: str = "runs"

    @model_validator(mode="after")
    def _cross_checks(self) -> "AppConfig":
        plans = [("fixed_time", self.fixed_time)]
        if self.fixed_time_tuned is not None:
            plans.append(("fixed_time_tuned", self.fixed_time_tuned))
        for label, plan in plans:
            for g in plan.green_s:
                if not self.timing.min_green_s <= g <= self.timing.max_green_s:
                    raise ValueError(f"{label} green {g}s outside [{self.timing.min_green_s}, {self.timing.max_green_s}]")
        if not self.timing.min_green_s <= self.actuated.max_green_s <= self.timing.max_green_s:
            raise ValueError("actuated.max_green_s must lie within [timing.min_green_s, timing.max_green_s]")
        for name, sc in self.scenarios.items():
            if sc.name != name:
                raise ValueError(f"scenario key '{name}' must equal its name '{sc.name}'")
        return self


def load_config(path: str | Path) -> AppConfig:
    """Load and validate a YAML config file; raises a clear error if missing or invalid."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"config file not found: {p.resolve()}")
    with p.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return AppConfig.model_validate(data)
