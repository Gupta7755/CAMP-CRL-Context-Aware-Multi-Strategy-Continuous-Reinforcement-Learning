"""CAMP action prior, adaptive fusion gate and the fused policy (Phase 2, deployment-time fusion).

Both actions share the environment's representation a in [-1, 1]^2 (a[0] > 0 = switch; a[1] = hold length).

CAMP prior  a_CAMP[0] = clip(k * (U_other - U_current - margin[strategy]), -1, 1), or +1 for gap-out
            a_CAMP[1] = -1 (1 s hold: re-evaluate every second, like the Phase-1 CAMP rule controller)
Fusion      a_mix = (1 - alpha) * a_CAMP + alpha * a_RL
Gate        alpha = sigmoid(w . u + b),  u = [1 - confidence, anomaly, novelty, rl_uncertainty], each in [0, 1]
            All w are NEGATIVE so poor data, surges, unfamiliar inputs or critic disagreement REDUCE trust
            in RL; b is chosen so alpha starts low (conservative). w and b are untuned starting values.
Novelty     mean |z| of the observation against training observation statistics, / 3, clipped to [0, 1].
Uncertainty TD3 twin-critic disagreement |Q1 - Q2| / (|Q1| + |Q2| + eps); 0 for single-critic DDPG.

Fusion is applied when a trained policy is deployed. Training-time fusion is not implemented: the
replay buffer would store the agent's action while a different (mixed) action was executed.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from camp_crl.control.controllers import PHASE_AXIS, CampRuleController
from camp_crl.control.safety import Mode
from camp_crl.sim.network import PHASE_APPROACHES

FusionMode = Literal["camp", "rl", "fixed", "gated"]
PRIOR_GAIN = 5.0


@dataclass(frozen=True)
class GateParams:
    w_confidence_deficit: float = -2.0
    w_anomaly: float = -2.0
    w_novelty: float = -2.0
    w_uncertainty: float = -2.0
    bias: float = -1.0
    fixed_alpha: float = 0.5

    def __post_init__(self) -> None:
        for name in ("w_confidence_deficit", "w_anomaly", "w_novelty", "w_uncertainty"):
            if getattr(self, name) > 0:
                raise ValueError(f"{name} must be <= 0 so that worse conditions never increase trust in RL")
        if not 0.0 <= self.fixed_alpha <= 1.0:
            raise ValueError("fixed_alpha must be in [0, 1]")


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def gate_alpha(u: np.ndarray, p: GateParams) -> float:
    w = np.array([p.w_confidence_deficit, p.w_anomaly, p.w_novelty, p.w_uncertainty])
    u = np.clip(np.nan_to_num(np.asarray(u, dtype=float), nan=1.0), 0.0, 1.0)  # missing signal = worst case
    return sigmoid(float(w @ u) + p.bias)


def fuse(a_camp: np.ndarray, a_rl: np.ndarray, alpha: float) -> np.ndarray:
    alpha = min(max(float(alpha), 0.0), 1.0)
    return np.clip((1.0 - alpha) * np.asarray(a_camp, float) + alpha * np.asarray(a_rl, float), -1.0, 1.0)


def camp_prior_action(env, rule: CampRuleController) -> np.ndarray:
    """CAMP rule decision at the current decision point, expressed as an environment action."""
    ctx, ext, ep = env.ext.context, env.ext, env.ep
    if ctx is None or ep is None:
        return np.array([-1.0, -1.0])
    st = ep.safety.status
    cur = st.green_phase if st.mode is Mode.GREEN else st.target_phase
    other = 1 - cur
    strategy = ext.strategy
    u_cur = rule._urgency(ctx, PHASE_AXIS[cur], strategy)
    u_oth = rule._urgency(ctx, PHASE_AXIS[other], strategy)
    margin = env.app.strategy.switch_margin.get(strategy.value, 0.1)
    n_cur = sum(ext.state.approaches[a].vehicle_count for a in PHASE_APPROACHES[cur])
    q_oth = sum(ext.state.approaches[a].queue_veh for a in PHASE_APPROACHES[other])
    if u_oth - u_cur > margin or (n_cur == 0 and q_oth > 0):
        a0 = 1.0 if u_oth - u_cur <= margin else min(1.0, max(PRIOR_GAIN * (u_oth - u_cur - margin), 1e-3))
    else:
        a0 = max(-1.0, min(PRIOR_GAIN * (u_oth - u_cur - margin), 0.0))
    return np.array([a0, -1.0])


@dataclass
class ObsStats:
    """Running mean/variance of training observations (Welford), saved with each RL run."""

    n: int = 0
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None

    def update(self, batch: np.ndarray) -> None:
        for x in np.atleast_2d(np.asarray(batch, dtype=float)):
            if self.mean is None:
                self.mean, self.m2 = np.zeros_like(x), np.zeros_like(x)
            self.n += 1
            d = x - self.mean
            self.mean += d / self.n
            self.m2 += d * (x - self.mean)

    def novelty(self, obs: np.ndarray) -> float:
        if self.mean is None or self.n < 2:
            return 1.0  # no statistics: treat as unfamiliar (conservative)
        std = np.sqrt(self.m2 / (self.n - 1)) + 1e-3
        return float(np.clip(np.mean(np.abs((np.asarray(obs, float) - self.mean) / std)) / 3.0, 0.0, 1.0))

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"n": self.n, "mean": None if self.mean is None else self.mean.tolist(),
                                    "m2": None if self.m2 is None else self.m2.tolist()}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ObsStats":
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(d["n"], None if d["mean"] is None else np.array(d["mean"]), None if d["m2"] is None else np.array(d["m2"]))


def critic_uncertainty(model, obs: np.ndarray, action: np.ndarray) -> float:
    """Twin-critic disagreement for TD3; 0.0 when the model has a single critic."""
    import torch

    critic = getattr(model, "critic", None)
    if critic is None or getattr(critic, "n_critics", 1) < 2:
        return 0.0
    with torch.no_grad():
        o = torch.as_tensor(np.asarray(obs, np.float32)[None])
        a = torch.as_tensor(np.asarray(action, np.float32)[None])
        q1, q2 = (float(q) for q in critic(o, a)[:2])
    return float(min(abs(q1 - q2) / (abs(q1) + abs(q2) + 1e-6), 1.0))


@dataclass
class FusedPolicy:
    """Decides an action at each decision point; logs every component for explainability/ablation."""

    mode: FusionMode
    model: object | None = None  # SB3 model trained on the same observation mode as the env
    obs_stats: ObsStats | None = None
    gate: GateParams = field(default_factory=GateParams)
    log: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.mode != "camp" and self.model is None:
            raise ValueError(f"mode '{self.mode}' needs a trained model")

    def act(self, env, obs: np.ndarray, rule: CampRuleController) -> np.ndarray:
        a_camp = camp_prior_action(env, rule)
        a_rl = np.zeros(2)
        u = np.zeros(4)
        if self.model is not None:
            a_rl, _ = self.model.predict(obs, deterministic=True)
            a_rl = np.asarray(a_rl, float).reshape(2)
        if self.mode == "camp":
            alpha = 0.0
        elif self.mode == "rl":
            alpha = 1.0
        elif self.mode == "fixed":
            alpha = self.gate.fixed_alpha
        else:
            ctx = env.ext.context
            conf = ctx.confidence if ctx is not None else 0.0
            anomaly = max((c.anomaly_score for c in ctx.approaches.values()), default=0.0) if ctx else 1.0
            novelty = self.obs_stats.novelty(obs) if self.obs_stats else 1.0
            unc = critic_uncertainty(self.model, obs, a_rl)
            u = np.array([1.0 - conf, anomaly, novelty, unc])
            alpha = gate_alpha(u, self.gate)
        a = fuse(a_camp, a_rl, alpha)
        self.log.append({"t": env._time(), "alpha": alpha, "a_camp": a_camp.round(3).tolist(),
                         "a_rl": np.round(a_rl, 3).tolist(), "a_mix": a.round(3).tolist(), "u": np.round(u, 3).tolist()})
        return a.astype(np.float32)
