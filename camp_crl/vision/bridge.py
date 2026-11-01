"""Camera -> SUMO demand bridge: vehicles counted by a camera are inserted on that SUMO approach.

SUMO is the digital twin; the signal being controlled is SUMO's. Camera data never reaches real
infrastructure.

Per approach and per simulation step:
  LIVE      the camera is OK and confidence >= min_confidence: insert as many vehicles as the camera's
            arrivals counter advanced since the previous step, and update a trusted arrival-rate EWMA
            (time constant ewma_tau_s);
  FALLBACK  any other state (degraded, stale, disconnected, low confidence): insert Poisson arrivals at
            the trusted EWMA rate if the camera was live for at least min_trusted_s, otherwise at the
            configured default rate. The camera's counter still advances in the background but its
            increments are discarded, so recovery does not dump a backlog of doubtful counts into SUMO.
Turning movements are drawn with the scenario turn shares (left, straight, right); cameras do not
observe turns. All randomness comes from one seeded generator, so a demo replays exactly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from camp_crl.config import APPROACHES
from camp_crl.vision.types import ApproachPerception, Status

log = logging.getLogger(__name__)

TURNS = ("L", "T", "R")


def route_id(approach: str, turn: str) -> str:
    return f"r_{approach}_{turn}"


@dataclass
class BridgeConfig:
    min_confidence: float = 0.5
    ewma_tau_s: float = 120.0
    min_trusted_s: float = 60.0
    default_veh_per_h: dict[str, float] = field(default_factory=lambda: {a: 400.0 for a in APPROACHES})


class CameraDemandBridge:
    def __init__(self, cfg: BridgeConfig, turn_shares: tuple[float, float, float], seed: int) -> None:
        self.cfg, self.turn_shares = cfg, np.asarray(turn_shares)
        self.rng = np.random.default_rng(seed + 32452843)
        self.mode = {a: "fallback" for a in APPROACHES}
        self.rate = {a: 0.0 for a in APPROACHES}
        self.trusted_s = {a: 0.0 for a in APPROACHES}
        self.inserted = {a: 0 for a in APPROACHES}
        self.fallback_s = {a: 0.0 for a in APPROACHES}
        self._last_total: dict[str, int | None] = {a: None for a in APPROACHES}
        self._last_t: float | None = None
        self._vid = 0
        self.transitions: list[dict] = []

    def is_live(self, p: ApproachPerception | None) -> bool:
        return p is not None and p.status is Status.OK and p.confidence >= self.cfg.min_confidence

    def plan(self, t: float, perceptions: dict[str, ApproachPerception]) -> dict[str, int]:
        """Decide how many vehicles to insert per approach at time t (no SUMO access; unit-testable)."""
        dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t
        out: dict[str, int] = {}
        for a in APPROACHES:
            p = perceptions.get(a)
            delta = 0
            if p is not None:
                prev = self._last_total[a]
                delta = 0 if prev is None else max(0, p.arrivals_total - prev)
                self._last_total[a] = p.arrivals_total
            live = self.is_live(p)
            mode = "live" if live else "fallback"
            if mode != self.mode[a]:
                why = "" if live or p is None else (
                    p.reason or f"confidence {p.confidence:.2f} < {self.cfg.min_confidence} "
                                f"(frame age {p.frame_age_s:.1f}s, {p.fps:.1f} fps)")
                self.transitions.append({"time_s": t, "approach": a, "mode": mode, "reason": why or "camera ok"})
                log.info("t=%.0f approach %s -> %s %s", t, a, mode, why)
                self.mode[a] = mode
            if live:
                n = delta
                if dt > 0:
                    k = min(1.0, dt / self.cfg.ewma_tau_s)
                    self.rate[a] = (1 - k) * self.rate[a] + k * (delta / dt * 3600.0)
                    self.trusted_s[a] += dt
            else:
                rate = self.rate[a] if self.trusted_s[a] >= self.cfg.min_trusted_s else self.cfg.default_veh_per_h[a]
                n = int(self.rng.poisson(rate / 3600.0 * dt)) if dt > 0 else 0
                self.fallback_s[a] += dt
            out[a] = n
        return out

    def step(self, conn, t: float, perceptions: dict[str, ApproachPerception]) -> dict[str, int]:
        plan = self.plan(t, perceptions)
        for a, n in plan.items():
            for _ in range(n):
                turn = TURNS[int(self.rng.choice(3, p=self.turn_shares))]
                self._vid += 1
                conn.vehicle.add(f"cam_{a}_{self._vid}", route_id(a, turn), typeID="car",
                                 departLane="best", departSpeed="max")
                self.inserted[a] += 1
        return plan
