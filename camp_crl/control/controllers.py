"""Signal controllers. Each returns a *requested* phase; the SafetyFilter decides what is applied."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from camp_crl.camp.features import CampContext, CampEngine
from camp_crl.camp.strategy import AXES, Strategy, StrategySelector
from camp_crl.config import ActuatedConfig, AppConfig
from camp_crl.control.safety import Mode, SignalStatus
from camp_crl.sim.network import PHASE_APPROACHES
from camp_crl.state import TrafficState

PHASE_AXIS = {0: "NS", 1: "EW"}
assert all(set(AXES[PHASE_AXIS[p]]) == set(a) for p, a in PHASE_APPROACHES.items())


@dataclass
class Decision:
    requested_phase: int
    info: dict = field(default_factory=dict)


class Controller(Protocol):
    name: str

    def act(self, state: TrafficState, signal: SignalStatus) -> Decision: ...


class FixedTimeController:
    """Cycles phases with fixed green durations; clearance intervals come from the filter."""

    name = "fixed_time"

    def __init__(self, green_s: tuple[float, ...]) -> None:
        self.green_s = green_s

    def act(self, state: TrafficState, signal: SignalStatus) -> Decision:
        p = signal.green_phase
        if signal.mode is Mode.GREEN and signal.time_in_mode_s >= self.green_s[p]:
            return Decision((p + 1) % len(self.green_s))
        return Decision(signal.target_phase if signal.mode is not Mode.GREEN else p)


class ActuatedController:
    """Gap-out actuated control with a virtual stop-line presence detector (see ActuatedConfig).

    The green rests on the current phase while no conflicting approach has a vehicle.
    """

    name = "actuated"

    def __init__(self, cfg: ActuatedConfig, min_green_s: float) -> None:
        self.cfg = cfg
        self.min_green_s = min_green_s
        self._phase: int | None = None
        self._last_detection_s = 0.0

    def act(self, state: TrafficState, signal: SignalStatus) -> Decision:
        if signal.mode is not Mode.GREEN:
            self._phase = None
            return Decision(signal.target_phase)
        p = signal.green_phase
        other = 1 - p
        if self._phase != p:  # green onset counts as a detection
            self._phase, self._last_detection_s = p, state.time_s
        if any(state.approaches[a].stopline_count > 0 for a in PHASE_APPROACHES[p]):
            self._last_detection_s = state.time_s
        gap = state.time_s - self._last_detection_s
        demand_other = any(state.approaches[a].vehicle_count > 0 for a in PHASE_APPROACHES[other])
        green = signal.time_in_mode_s
        if demand_other and green >= self.cfg.max_green_s:
            return Decision(other, {"decision_reason": f"max-out at {green:.0f}s"})
        if demand_other and green >= self.min_green_s and gap >= self.cfg.gap_s:
            return Decision(other, {"decision_reason": f"gap-out: zone empty {gap:.0f}s"})
        return Decision(p, {"decision_reason": f"extend: gap {gap:.0f}s, other demand={demand_other}"})


def phase_pressures(state: TrafficState) -> dict[int, float]:
    """Max-pressure phase weights: sum over served approaches of (upstream vehicles - vehicles on
    the through-movement exit). Approach-level approximation of Varaiya's movement pressure;
    turning movements' exits are not counted separately."""
    return {p: float(sum(state.approaches[a].vehicle_count - state.approaches[a].downstream_count for a in apps))
            for p, apps in PHASE_APPROACHES.items()}


class MaxPressureController:
    """Requests the phase with the highest pressure; ties keep the current phase."""

    name = "max_pressure"

    def __init__(self, min_green_s: float) -> None:
        self.min_green_s = min_green_s

    def act(self, state: TrafficState, signal: SignalStatus) -> Decision:
        if signal.mode is not Mode.GREEN:
            return Decision(signal.target_phase)
        cur = signal.green_phase
        pr = phase_pressures(state)
        best = max(pr, key=lambda p: (pr[p], p == cur))
        if signal.time_in_mode_s < self.min_green_s:
            return Decision(cur, {"pressures": pr, "decision_reason": "min green not yet elapsed"})
        why = "highest pressure" if best != cur or pr[best] > min(pr.values()) else "tie: keep current"
        return Decision(best, {"pressures": pr, "decision_reason": why})


class CampRuleController:
    """CAMP rule-only controller.

    Axis urgency U_axis = max over its approaches of C_i (with strategy-specific adjustments).
    Request the other phase when U_other - U_current > switch_margin[strategy], or when the
    current axis has no vehicles while the other axis has a queue (gap-out). Min/max green are left to
    the safety filter so the two layers stay independent.
    """

    name = "camp_rule"

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.engine = CampEngine(cfg.camp)
        self.selector = StrategySelector(cfg.strategy)
        self.last_context: CampContext | None = None

    def _urgency(self, ctx: CampContext, axis: str, strategy: Strategy) -> float:
        vals = []
        for a in AXES[axis]:
            c = ctx.approaches[a]
            u = c.composite
            if strategy is Strategy.FAIRNESS_RECOVERY:
                u += self.cfg.camp.lambda_fairness * c.fairness  # double fairness weight
            elif strategy is Strategy.OVERFLOW_PREVENTION:
                u -= self.cfg.strategy.overflow_penalty * c.downstream_congestion
            vals.append(u)
        return max(vals)

    def act(self, state: TrafficState, signal: SignalStatus) -> Decision:
        ctx = self.engine.update(state)
        self.last_context = ctx
        sd = self.selector.update(ctx)
        if signal.mode is not Mode.GREEN:
            return Decision(signal.target_phase, {"strategy": sd.strategy.value, "strategy_reason": sd.reason})
        cur = signal.green_phase
        other = 1 - cur
        u_cur = self._urgency(ctx, PHASE_AXIS[cur], sd.strategy)
        u_oth = self._urgency(ctx, PHASE_AXIS[other], sd.strategy)
        margin = self.cfg.strategy.switch_margin.get(sd.strategy.value, 0.1)
        n_cur = sum(state.approaches[a].vehicle_count for a in PHASE_APPROACHES[cur])
        q_oth = sum(state.approaches[a].queue_veh for a in PHASE_APPROACHES[other])
        if signal.time_in_mode_s < self.cfg.timing.min_green_s:
            req, why = cur, "min green not yet elapsed"
        elif u_oth - u_cur > margin:
            req, why = other, f"U_other {u_oth:.2f} - U_current {u_cur:.2f} > margin {margin:.2f}"
        elif n_cur == 0 and q_oth > 0:
            req, why = other, "gap-out: no vehicles on current axis, other axis queued"
        else:
            req, why = cur, f"hold: U_current {u_cur:.2f}, U_other {u_oth:.2f}, margin {margin:.2f}"
        return Decision(req, {
            "strategy": sd.strategy.value,
            "strategy_reason": sd.reason,
            "u_current": round(u_cur, 4),
            "u_other": round(u_oth, 4),
            "decision_reason": why,
        })


CONTROLLER_NAMES = ("fixed_time", "fixed_time_tuned", "actuated", "max_pressure", "camp_rule")


def make_controller(name: str, cfg: AppConfig) -> Controller:
    if name == "fixed_time":
        return FixedTimeController(cfg.fixed_time.green_s)
    if name == "fixed_time_tuned":
        if cfg.fixed_time_tuned is None:
            raise ValueError("fixed_time_tuned has no plan: run `python -m camp_crl.cli tune-fixed` first")
        ctl = FixedTimeController(cfg.fixed_time_tuned.green_s)
        ctl.name = "fixed_time_tuned"
        return ctl
    if name == "actuated":
        return ActuatedController(cfg.actuated, cfg.timing.min_green_s)
    if name == "max_pressure":
        return MaxPressureController(cfg.timing.min_green_s)
    if name == "camp_rule":
        return CampRuleController(cfg)
    raise ValueError(f"unknown controller '{name}'; available: {', '.join(CONTROLLER_NAMES)}")
