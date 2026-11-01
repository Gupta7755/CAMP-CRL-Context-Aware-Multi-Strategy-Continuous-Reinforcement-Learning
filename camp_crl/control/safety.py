"""Independent safety filter: the only component allowed to set the signal state.

Controllers *request* a green phase each tick. The filter owns its own clock and enforces,
regardless of what was requested:
- requested phase must be a legal phase index, otherwise the current phase is held;
- a green lasts at least `min_green_s` and at most `max_green_s`;
- every change goes GREEN -> YELLOW (exactly yellow_s) -> ALL_RED (exactly all_red_s) -> GREEN;
- no phase stays red longer than `max_red_s` (starvation bound);
- requests during YELLOW/ALL_RED are ignored (the transition always completes).
Every override is recorded as an intervention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from camp_crl.config import TimingConfig
from camp_crl.sim.network import SignalPlan

TIME_EPS = 1e-6


class Mode(str, Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    ALL_RED = "ALL_RED"


@dataclass(frozen=True)
class Intervention:
    time_s: float
    kind: str  # invalid_phase | min_green_hold | max_green_force | starvation_force
    requested: object
    applied: int
    detail: str


@dataclass(frozen=True)
class SignalCommand:
    """Signal state to apply for the next interval [t, t + dt)."""

    state: str
    mode: Mode
    green_phase: int  # phase currently green, or the phase being cleared during YELLOW/ALL_RED
    target_phase: int  # phase that will be green after the transition (== green_phase when GREEN)
    changed: bool
    interventions: tuple[Intervention, ...] = ()


@dataclass
class SignalStatus:
    mode: Mode
    green_phase: int
    target_phase: int
    time_in_mode_s: float
    red_time_s: dict[int, float] = field(default_factory=dict)


class SafetyFilter:
    def __init__(self, plan: SignalPlan, timing: TimingConfig, initial_phase: int = 0) -> None:
        if initial_phase not in plan.phases:
            raise ValueError(f"initial phase {initial_phase} not in {plan.phases}")
        self.plan = plan
        self.t = timing
        self.mode = Mode.GREEN
        self.current = initial_phase
        self.target = initial_phase
        self.time_in_mode = 0.0
        self.red_time = {p: 0.0 for p in plan.phases}
        self.now = 0.0
        self.switches = 0
        self.interventions: list[Intervention] = []
        self._last_state: str | None = None

    @property
    def status(self) -> SignalStatus:
        return SignalStatus(self.mode, self.current, self.target, self.time_in_mode, dict(self.red_time))

    def _most_starved(self) -> int:
        others = [p for p in self.plan.phases if p != self.current]
        return max(others, key=lambda p: self.red_time[p])

    def _decide_green(self, requested: object) -> tuple[int, list[Intervention]]:
        ivs: list[Intervention] = []
        req = requested if isinstance(requested, int) and not isinstance(requested, bool) else None
        if req is None or req not in self.plan.phases:
            ivs.append(Intervention(self.now, "invalid_phase", requested, self.current,
                                    f"legal phases are {self.plan.phases}; holding current"))
            req = self.current
        elapsed = self.time_in_mode
        starved = [p for p in self.plan.phases
                   if p != self.current and self.red_time[p] + TIME_EPS >= self.t.max_red_s]
        if elapsed + TIME_EPS >= self.t.min_green_s and starved and req not in starved:
            target = max(starved, key=lambda p: self.red_time[p])
            ivs.append(Intervention(self.now, "starvation_force", requested, target,
                                    f"phase {target} red {self.red_time[target]:.0f}s >= {self.t.max_red_s}s"))
            return target, ivs
        if req != self.current and elapsed + TIME_EPS < self.t.min_green_s:
            ivs.append(Intervention(self.now, "min_green_hold", requested, self.current,
                                    f"green {elapsed:.0f}s < min {self.t.min_green_s}s"))
            return self.current, ivs
        if req == self.current and elapsed + TIME_EPS >= self.t.max_green_s:
            target = self._most_starved()
            ivs.append(Intervention(self.now, "max_green_force", requested, target,
                                    f"green {elapsed:.0f}s reached max {self.t.max_green_s}s"))
            return target, ivs
        return req, ivs

    def step(self, requested: object, dt: float) -> SignalCommand:
        """Advance the filter by one control interval of length dt (seconds)."""
        ivs: list[Intervention] = []
        if self.mode is Mode.YELLOW and self.time_in_mode + TIME_EPS >= self.t.yellow_s:
            self.mode, self.time_in_mode = Mode.ALL_RED, 0.0
        if self.mode is Mode.ALL_RED and self.time_in_mode + TIME_EPS >= self.t.all_red_s:
            self.mode, self.time_in_mode = Mode.GREEN, 0.0
            self.current = self.target
            self.red_time[self.current] = 0.0
            self.switches += 1
        elif self.mode is Mode.GREEN:
            target, ivs = self._decide_green(requested)
            if target != self.current:
                self.mode, self.time_in_mode, self.target = Mode.YELLOW, 0.0, target
        if self.mode is Mode.GREEN:
            state = self.plan.green[self.current]
        elif self.mode is Mode.YELLOW:
            state = self.plan.yellow[self.current]
        else:
            state = self.plan.all_red
        changed = state != self._last_state
        self._last_state = state
        self.interventions.extend(ivs)
        cmd = SignalCommand(state, self.mode, self.current, self.target, changed, tuple(ivs))
        # Advance clocks for the interval this command covers.
        self.time_in_mode += dt
        for p in self.plan.phases:
            if not (self.mode is Mode.GREEN and p == self.current):
                self.red_time[p] += dt
        self.now += dt
        return cmd
