"""CAMP context engine: pressure, momentum, short-horizon risk, fairness and composite urgency.

These are proposed engineering heuristics under evaluation, not validated innovations.

Conventions
- Every "hat" term is dimensionless in [0, 1] via `normalize`.
- Growth terms are dimensionless in [-1, 1] via `signed_scale`.
- P in [0, 1] (weights sum to 1), M in [-1, 1] (betas sum to 1), F in [0, F_max], S in [0, 1].
- C_i = P_i + lambda_M*M_i + lambda_F*F_i + lambda_S*S_i is a ranking score, not a probability.

Explainability signals (reported only; Phase-1 control decisions do not use them):
- trend: RISING / FALLING / STABLE from the smoothed queue derivative vs a threshold.
- anomaly: one-sided Poisson likelihood-ratio CUSUM (Page, 1954) on per-step arrival counts n_t
  against a slow EWMA baseline rate l0: with l1 = max(ratio*l0, l0 + min_shift),
  S_t = max(0, S_{t-1} + n_t*ln(l1/l0) - (l1 - l0)*dt/3600); alarm when S_t >= h;
  score = min(S_t/h, 1). Detects sustained arrival surges; not validated on SUMO data.
- pattern: LOW_DEMAND / CONGESTED / NS_DOMINANT / EW_DOMINANT / BALANCED from pressures.
- confidence: share of approaches with a fresh (non-stale) observation. A data-quality
  score, NOT a calibrated probability.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from camp_crl.config import APPROACHES, CampConfig
from camp_crl.state import ApproachState, TrafficState

EPS = 1e-9


def normalize(x: float, lo: float, hi: float, safe_value: float = 0.0) -> float:
    """x_hat = clip((x - lo) / (hi - lo), 0, 1).

    Returns `safe_value` when x is NaN/inf or the range is (near) zero, so a broken
    observation or misconfigured bound never produces NaN downstream.
    """
    if not math.isfinite(x) or not math.isfinite(lo) or not math.isfinite(hi) or abs(hi - lo) < EPS:
        return safe_value
    return min(max((x - lo) / (hi - lo), 0.0), 1.0)


def signed_scale(x: float, scale: float) -> float:
    """clip(x / scale, -1, 1); 0.0 for non-finite x or non-positive scale."""
    if not math.isfinite(x) or scale <= 0:
        return 0.0
    return min(max(x / scale, -1.0), 1.0)


def pressure(hats: dict[str, float], cfg: CampConfig) -> float:
    w = cfg.pressure
    return (w.queue * hats["queue"] + w.arrival * hats["arrival"] + w.wait * hats["wait"]
            + w.density * hats["density"] + w.spillback * hats["spillback"])


def momentum(queue_growth: float, arrival_growth: float, speed_growth: float, cfg: CampConfig) -> float:
    """M = bQ*qg + bA*ag - bV*vg. Rising queues/arrivals and falling speeds increase M."""
    m = cfg.momentum
    return m.beta_queue * queue_growth + m.beta_arrival * arrival_growth - m.beta_speed * speed_growth


def risk(p: float, m: float, cfg: CampConfig) -> float:
    """Heuristic linear extrapolation F = clip(P + tau*kM*M, 0, Fmax). Not a validated forecast."""
    return min(max(p + cfg.risk_horizon_s * cfg.risk_k_m * m, 0.0), cfg.risk_max)


def fairness(max_wait_s: float, cfg: CampConfig) -> float:
    """S = clip(W / W_ref, 0, 1) using the longest accumulated wait on the approach."""
    return normalize(max_wait_s, 0.0, cfg.fairness_ref_wait_s)


def composite(p: float, m: float, f: float, s: float, cfg: CampConfig) -> float:
    return p + cfg.lambda_momentum * m + cfg.lambda_risk * f + cfg.lambda_fairness * s


@dataclass(frozen=True)
class ApproachContext:
    pressure: float
    momentum: float
    risk: float
    fairness: float
    composite: float
    hats: dict[str, float]
    downstream_occupancy_pct: float
    downstream_fill_ratio: float = 0.0
    downstream_congestion: float = 0.0
    trend: str = "STABLE"
    anomaly_score: float = 0.0
    anomaly: bool = False
    stale: bool = False


@dataclass(frozen=True)
class CampContext:
    time_s: float
    approaches: dict[str, ApproachContext]
    max_wait_s: float
    wait_dispersion_s: float  # population std of per-approach mean waits
    confidence: float  # data-quality score in [0, 1], NOT a calibrated probability
    pattern: str = "BALANCED"

    def to_dict(self) -> dict:
        return {
            "time_s": self.time_s,
            "pattern": self.pattern,
            "max_wait_s": self.max_wait_s,
            "wait_dispersion_s": self.wait_dispersion_s,
            "confidence": self.confidence,
            "approaches": {k: asdict(v) for k, v in self.approaches.items()},
        }


class CampEngine:
    """Stateful CAMP feature engine; keeps EWMA-smoothed derivatives per approach."""

    def __init__(self, cfg: CampConfig) -> None:
        self.cfg = cfg
        self._prev: dict[str, ApproachState] = {}
        self._prev_time: float | None = None
        self._d: dict[str, dict[str, float]] = {a: {"q": 0.0, "a": 0.0, "v": 0.0} for a in APPROACHES}
        self._base: dict[str, float | None] = {a: None for a in APPROACHES}
        self._cusum: dict[str, float] = {a: 0.0 for a in APPROACHES}

    def _anomaly(self, a: str, n: float, t: float, dt: float) -> float:
        c = self.cfg
        if not math.isfinite(n) or dt <= 0:
            return self._cusum[a]
        rate = n * 3600.0 / dt
        mu = self._base[a]
        if mu is None:
            self._base[a] = mu = max(rate, c.cusum_floor_veh_per_h)
        if t >= c.cusum_warmup_s:
            l0 = max(mu, c.cusum_floor_veh_per_h)
            l1 = max(l0 * c.cusum_ratio, l0 + c.cusum_min_shift_veh_per_h)
            llr = n * math.log(l1 / l0) - (l1 - l0) * dt / 3600.0
            self._cusum[a] = max(0.0, self._cusum[a] + llr)
        # faster baseline learning during warm-up so it starts near the true level
        alpha = c.cusum_baseline_alpha * dt if t >= c.cusum_warmup_s else max(c.cusum_baseline_alpha * dt, 0.02)
        self._base[a] = min(alpha, 1.0) * rate + (1 - min(alpha, 1.0)) * mu
        return self._cusum[a]

    def _pattern(self, out: dict[str, ApproachContext]) -> str:
        c = self.cfg
        mean_p = sum(x.pressure for x in out.values()) / len(out)
        ns = max(out["N"].pressure, out["S"].pressure)
        ew = max(out["E"].pressure, out["W"].pressure)
        if mean_p >= c.pattern_congested_pressure:
            return "CONGESTED"
        if mean_p < c.pattern_low_pressure and abs(ns - ew) < c.pattern_dominance_gap:
            return "LOW_DEMAND"
        if ns - ew >= c.pattern_dominance_gap:
            return "NS_DOMINANT"
        if ew - ns >= c.pattern_dominance_gap:
            return "EW_DOMINANT"
        return "BALANCED"

    def _hats(self, s: ApproachState) -> dict[str, float]:
        b = self.cfg.bounds
        return {
            "queue": normalize(s.queue_veh, *b.queue_veh),
            "arrival": normalize(s.arrival_veh_per_h, *b.arrival_veh_per_h),
            "wait": normalize(s.max_wait_s, *b.max_wait_s),
            "density": normalize(s.occupancy_pct, *b.occupancy_pct),
            "spillback": normalize(s.spillback_ratio, *b.spillback_ratio),
        }

    def update(self, state: TrafficState) -> CampContext:
        """Compute context for `state`. A stale or repeated timestamp (dt <= 0) leaves the
        smoothed derivatives unchanged instead of dividing by zero."""
        mc = self.cfg.momentum
        dt = None if self._prev_time is None else state.time_s - self._prev_time
        out: dict[str, ApproachContext] = {}
        mean_waits: list[float] = []
        max_wait = 0.0
        for a in APPROACHES:
            s = state.approaches[a]
            prev = self._prev.get(a)
            d = self._d[a]
            if prev is not None and dt is not None and dt > 0:
                for key, cur, old in (("q", s.queue_veh, prev.queue_veh),
                                      ("a", s.arrival_veh_per_h, prev.arrival_veh_per_h),
                                      ("v", s.mean_speed_mps, prev.mean_speed_mps)):
                    raw = (cur - old) / dt
                    if math.isfinite(raw):
                        d[key] = mc.ewma_alpha * raw + (1 - mc.ewma_alpha) * d[key]
            hats = self._hats(s)
            p = pressure(hats, self.cfg)
            m = momentum(signed_scale(d["q"], mc.queue_rate_scale),
                         signed_scale(d["a"], mc.arrival_rate_scale),
                         signed_scale(d["v"], mc.speed_rate_scale), self.cfg)
            f = risk(p, m, self.cfg)
            fs = fairness(s.max_wait_s, self.cfg)
            trend = ("RISING" if d["q"] > self.cfg.trend_threshold_veh_per_s
                     else "FALLING" if d["q"] < -self.cfg.trend_threshold_veh_per_s else "STABLE")
            # stale observations repeat old counts and a repeated timestamp has dt = 0: skip both
            n = math.nan if a in state.stale else float(s.arrivals_step)
            cus = self._anomaly(a, n, state.time_s, dt if dt is not None else 1.0)
            out[a] = ApproachContext(p, m, f, fs, composite(p, m, f, fs, self.cfg), hats,
                                     s.downstream_occupancy_pct, s.downstream_fill_ratio, s.downstream_congestion, trend,
                                     min(cus / self.cfg.cusum_h, 1.0), cus >= self.cfg.cusum_h, a in state.stale)
            mw = s.mean_wait_s if math.isfinite(s.mean_wait_s) else 0.0
            mean_waits.append(mw)
            if math.isfinite(s.max_wait_s):
                max_wait = max(max_wait, s.max_wait_s)
            self._prev[a] = s
        if dt is None or dt > 0:
            self._prev_time = state.time_s
        mu = sum(mean_waits) / len(mean_waits)
        disp = math.sqrt(sum((w - mu) ** 2 for w in mean_waits) / len(mean_waits))
        conf = 1.0 - len(state.stale) / len(APPROACHES)
        return CampContext(state.time_s, out, max_wait, disp, conf, self._pattern(out))
