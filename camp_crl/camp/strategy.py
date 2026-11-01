"""Transparent rule-based strategy selection with hysteresis.

Rules are evaluated in priority order (first match wins):
1. OVERFLOW_PREVENTION max through-exit congestion (1 - speed/free-flow) >= overflow_congestion
   (first: a green into a blocked exit cannot discharge anyone, so fairness cannot be served there)
2. FAIRNESS_RECOVERY   max S_i >= fairness_threshold
3. CONGESTION_RECOVERY mean P_i >= congestion_pressure
4. DIRECTIONAL_PRIORITY |P_NS - P_EW| >= dominance_gap  (axis P = max of its two approaches)
5. BALANCED_FLOW       mean P_i >= balanced_min_pressure and gap <= balanced_max_gap
6. NORMAL_ADAPTIVE     otherwise

Hysteresis: a new strategy must be the rule outcome for `confirm_steps` consecutive
evaluations AND the current strategy must have been active for `min_hold_s`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from camp_crl.camp.features import CampContext
from camp_crl.config import StrategyConfig


class Strategy(str, Enum):
    NORMAL_ADAPTIVE = "NORMAL_ADAPTIVE"
    DIRECTIONAL_PRIORITY = "DIRECTIONAL_PRIORITY"
    CONGESTION_RECOVERY = "CONGESTION_RECOVERY"
    BALANCED_FLOW = "BALANCED_FLOW"
    FAIRNESS_RECOVERY = "FAIRNESS_RECOVERY"
    OVERFLOW_PREVENTION = "OVERFLOW_PREVENTION"


AXES: dict[str, tuple[str, str]] = {"NS": ("N", "S"), "EW": ("E", "W")}


def axis_pressure(ctx: CampContext) -> dict[str, float]:
    return {ax: max(ctx.approaches[a].pressure for a in apps) for ax, apps in AXES.items()}


def classify(ctx: CampContext, cfg: StrategyConfig) -> tuple[Strategy, str]:
    """Stateless rule outcome and a human-readable reason."""
    ap = ctx.approaches.values()
    max_s = max(c.fairness for c in ap)
    max_down = max(c.downstream_congestion for c in ap)
    mean_p = sum(c.pressure for c in ap) / len(ctx.approaches)
    axp = axis_pressure(ctx)
    gap = abs(axp["NS"] - axp["EW"])
    if max_down >= cfg.overflow_congestion:
        return Strategy.OVERFLOW_PREVENTION, f"exit congestion {max_down:.2f} >= {cfg.overflow_congestion}"
    if max_s >= cfg.fairness_threshold:
        return Strategy.FAIRNESS_RECOVERY, f"max fairness S={max_s:.2f} >= {cfg.fairness_threshold}"
    if mean_p >= cfg.congestion_pressure:
        return Strategy.CONGESTION_RECOVERY, f"mean P={mean_p:.2f} >= {cfg.congestion_pressure}"
    if gap >= cfg.dominance_gap:
        dom = "NS" if axp["NS"] > axp["EW"] else "EW"
        return Strategy.DIRECTIONAL_PRIORITY, f"{dom} dominant, axis gap={gap:.2f} >= {cfg.dominance_gap}"
    if mean_p >= cfg.balanced_min_pressure and gap <= cfg.balanced_max_gap:
        return Strategy.BALANCED_FLOW, f"mean P={mean_p:.2f}, axis gap={gap:.2f} (balanced)"
    return Strategy.NORMAL_ADAPTIVE, f"no rule triggered (mean P={mean_p:.2f}, gap={gap:.2f})"


@dataclass
class StrategyDecision:
    strategy: Strategy
    candidate: Strategy
    reason: str
    changed: bool


class StrategySelector:
    def __init__(self, cfg: StrategyConfig) -> None:
        self.cfg = cfg
        self.current = Strategy.NORMAL_ADAPTIVE
        self._since: float | None = None
        self._candidate: Strategy | None = None
        self._count = 0

    def update(self, ctx: CampContext) -> StrategyDecision:
        cand, reason = classify(ctx, self.cfg)
        if self._since is None:
            self._since = ctx.time_s
        if cand == self.current:
            self._candidate, self._count = None, 0
            return StrategyDecision(self.current, cand, reason, False)
        if cand == self._candidate:
            self._count += 1
        else:
            self._candidate, self._count = cand, 1
        held = ctx.time_s - self._since
        if self._count >= self.cfg.confirm_steps and held >= self.cfg.min_hold_s:
            self.current, self._since = cand, ctx.time_s
            self._candidate, self._count = None, 0
            return StrategyDecision(self.current, cand, reason, True)
        return StrategyDecision(self.current, cand, reason, False)
