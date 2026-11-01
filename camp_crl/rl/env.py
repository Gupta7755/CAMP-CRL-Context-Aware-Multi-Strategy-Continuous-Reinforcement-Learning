"""Gymnasium environment: one agent decision per green-signal decision point.

Decision points: the signal is GREEN and its minimum green has elapsed. Between decision points
SUMO advances in 1 s steps and the Phase-1 SafetyFilter keeps sole control of the signal state.

Action  a in [-1, 1]^2 (float32)
  a[0] > 0  -> request the other phase (clearance + min green follow, enforced by the filter)
  a[0] <= 0 -> hold the current phase for 1 + (a[1] + 1) / 2 * (max_hold_s - 1) seconds (rounded)
  The filter still forces a change at max green; that counts as a safety intervention.

Observation (float32 in [0, 1])
  raw : per approach (N, E, S, W) queue/40, vehicles/50, arrivals/1200 veh/h, longest wait/120 s,
        occupancy/50 %, exit congestion; then phase one-hot (2) and green elapsed / max green.  27 dims
  camp: per approach P, (M+1)/2, F, S, (C+0.3)/2.4, exit congestion; phase one-hot, green elapsed,
        strategy one-hot (6).  33 dims
  Observations come from the (possibly degraded) sensor view; the reward uses simulator ground truth.

Episodes end only by the time limit: terminated is always False, truncated is True at duration_s.
"""

from __future__ import annotations

import math
import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal

import gymnasium as gym
import numpy as np

from camp_crl.camp.features import CampContext, CampEngine
from camp_crl.camp.strategy import Strategy, StrategySelector
from camp_crl.config import APPROACHES, AppConfig
from camp_crl.control.controllers import Decision
from camp_crl.control.safety import Mode, SignalStatus
from camp_crl.rl.config import RLConfig
from camp_crl.runner import Episode
from camp_crl.state import TrafficState

ObsMode = Literal["raw", "camp"]
STRATEGIES = list(Strategy)
RAW_DIM = 4 * 6 + 3
CAMP_DIM = 4 * 6 + 3 + len(STRATEGIES)


def _clip01(x: float) -> float:
    return 0.0 if not math.isfinite(x) else min(max(x, 0.0), 1.0)


def action_to_request(action: np.ndarray, current_phase: int, max_hold_s: float) -> tuple[int, int]:
    """Map a normalized action to (requested phase, hold seconds). Hold is ignored when switching."""
    a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
    if a.size != 2 or not np.all(np.isfinite(a)):
        raise ValueError(f"action must be 2 finite values, got {action!r}")
    if a[0] > 0:
        return 1 - current_phase, 0
    hold = 1.0 + (a[1] + 1.0) / 2.0 * (max_hold_s - 1.0)
    return current_phase, int(round(hold))


class ExternalController:
    """Phase-1 controller interface driven by the environment; also tracks CAMP context."""

    name = "rl_external"

    def __init__(self, app: AppConfig, track_camp: bool) -> None:
        self.request = 0
        self.state: TrafficState | None = None
        self.context: CampContext | None = None
        self.strategy = Strategy.NORMAL_ADAPTIVE
        self._engine = CampEngine(app.camp) if track_camp else None
        self._selector = StrategySelector(app.strategy) if track_camp else None

    def act(self, state: TrafficState, signal: SignalStatus) -> Decision:
        self.state = state
        if self._engine is not None and self._selector is not None:
            self.context = self._engine.update(state)
            self.strategy = self._selector.update(self.context).strategy
        return Decision(self.request, {"strategy": self.strategy.value if self._engine else ""})


class TrafficSignalEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, app: AppConfig, rl: RLConfig, scenarios: list[str] | None = None, obs_mode: ObsMode = "camp",
                 duration_s: int | None = None, keep_episode_files: bool = False, run_label: str = "rl") -> None:
        super().__init__()
        if scenarios is None:  # default: training scenarios, which must be train-split
            rl.check_against(app)
        self.scenarios = list(scenarios or rl.train_scenarios)
        for s in self.scenarios:
            if s not in app.scenarios:
                raise ValueError(f"unknown scenario '{s}'")
        if duration_s is not None:
            sim = app.simulation.model_copy(update={"duration_s": duration_s,
                                                    # metrics only; the reward ignores warm-up
                                                    "warmup_s": min(app.simulation.warmup_s, duration_s // 5)})
            app = app.model_copy(update={"simulation": sim})
        self.app, self.rl, self.obs_mode = app, rl, obs_mode
        self.keep_episode_files = keep_episode_files
        self.run_label = run_label
        dim = CAMP_DIM if obs_mode == "camp" else RAW_DIM
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self._tmp = Path(tempfile.mkdtemp(prefix="camp_rl_"))
        self.ep: Episode | None = None
        self.ext: ExternalController | None = None
        self._arrived = 0
        self.episode_info: dict[str, Any] = {}

    # ---------- episode lifecycle ----------

    def _close_episode(self) -> dict | None:
        ep, self.ep = self.ep, None
        if ep is None:
            return None
        metrics = None
        try:
            if ep.done:
                metrics = ep.finish().metrics
            else:
                ep.abort()
        finally:
            if not self.keep_episode_files:
                shutil.rmtree(ep.run_dir, ignore_errors=True)
        return metrics

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._close_episode()
        options = options or {}
        scenario = options.get("scenario") or str(self.np_random.choice(self.scenarios))
        sumo_seed = int(options.get("sumo_seed", self.np_random.integers(0, 2**31 - 1)))
        ep = Episode(self.app, "fixed_time", scenario, sumo_seed, self._tmp)
        self.ext = ExternalController(self.app, track_camp=self.obs_mode == "camp")
        ep.controller = self.ext  # the environment, not a Phase-1 controller, decides
        ep.controller_name = self.run_label
        self.ep = ep
        self._arrived = 0
        self.episode_info = {"scenario": scenario, "sumo_seed": sumo_seed}
        self._advance_until_decision(first_request=0)
        return self._obs(), {**self.episode_info, "time_s": self._time()}

    def close(self) -> None:
        self._close_episode()
        shutil.rmtree(self._tmp, ignore_errors=True)

    # ---------- stepping ----------

    def _time(self) -> float:
        return float(self.ep.steps_done * self.app.simulation.step_length_s) if self.ep else 0.0

    def _ready(self) -> bool:
        st = self.ep.safety.status
        return st.mode is Mode.GREEN and st.time_in_mode_s + 1e-6 >= self.app.timing.min_green_s

    def _second_reward(self, snap: dict) -> float:
        w = self.rl.reward
        truth = snap["truth"]["approaches"]
        q = snap["kpi"]["total_queue_veh"] / (4 * 40.0)
        arrived = snap["kpi"]["arrived_total"]
        thr = (arrived - self._arrived) / 2.0
        self._arrived = arrived
        waits = [truth[a]["max_wait_s"] for a in APPROACHES]
        wmax = _clip01(max(waits) / 120.0)
        fair = _clip01(float(np.std(waits)) / 60.0)
        spill = 1.0 if snap["kpi"]["spillback"] else 0.0
        return (-w.queue * q + w.throughput * thr - w.max_wait * wmax - w.fairness * fair - w.spillback * spill)

    def _one_second(self, request: int) -> float:
        self.ext.request = request
        snap = self.ep.step()
        return self._second_reward(snap)

    def _advance_until_decision(self, first_request: int, min_seconds: int = 1) -> tuple[float, int]:
        """Run >= min_seconds; then keep following the filter's target until the next decision point."""
        total, n = 0.0, 0
        request = first_request
        while not self.ep.done and (n < min_seconds or not self._ready()):
            total += self._one_second(request)
            n += 1
            request = self.ep.safety.status.target_phase  # never fight the filter after the first second
        return total, n

    def step(self, action):
        if self.ep is None:
            raise RuntimeError("call reset() before step()")
        cur = self.ep.safety.status.green_phase
        request, hold = action_to_request(action, cur, self.rl.action.max_hold_s)
        switched = request != cur
        n_iv0 = len(self.ep.safety.interventions)
        if switched:
            r_sec, secs = self._advance_until_decision(request, min_seconds=1)
        else:
            r_sec, secs = 0.0, 0
            for _ in range(hold):
                if self.ep.done:
                    break
                # target == current green unless the filter forced a change (max green)
                r_sec += self._one_second(self.ep.safety.status.target_phase)
                secs += 1
            extra, n2 = self._advance_until_decision(self.ep.safety.status.target_phase, min_seconds=0)
            r_sec += extra
            secs += n2
        n_iv = len(self.ep.safety.interventions) - n_iv0
        w = self.rl.reward
        reward = r_sec * w.per_second_scale - w.switch * float(switched) - w.intervention * n_iv
        obs = self._obs()
        truncated = self.ep.done
        info: dict[str, Any] = {**self.episode_info, "time_s": self._time(), "seconds": secs,
                                "switched": switched, "interventions": n_iv,
                                "requested_phase": request, "hold_s": hold}
        if truncated:
            info["episode_metrics"] = self._close_episode()
        return obs, float(reward), False, truncated, info

    # ---------- observation ----------

    def _obs(self) -> np.ndarray:
        if self.ext is None or self.ext.state is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        st = self.ep.safety.status if self.ep is not None else None
        phase = st.green_phase if st else 0
        elapsed = _clip01(st.time_in_mode_s / self.app.timing.max_green_s) if st and st.mode is Mode.GREEN else 0.0
        tail = [1.0 if phase == 0 else 0.0, 1.0 if phase == 1 else 0.0, elapsed]
        feats: list[float] = []
        if self.obs_mode == "raw":
            for a in APPROACHES:
                s = self.ext.state.approaches[a]
                feats += [s.queue_veh / 40.0, s.vehicle_count / 50.0, s.arrival_veh_per_h / 1200.0,
                          s.max_wait_s / 120.0, s.occupancy_pct / 50.0, s.downstream_congestion]
            vec = feats + tail
        else:
            ctx = self.ext.context
            for a in APPROACHES:
                c = ctx.approaches[a]
                feats += [c.pressure, (c.momentum + 1.0) / 2.0, c.risk, c.fairness, (c.composite + 0.3) / 2.4,
                          c.downstream_congestion]
            vec = feats + tail + [1.0 if self.ext.strategy is s else 0.0 for s in STRATEGIES]
        return np.asarray([_clip01(float(x)) for x in vec], dtype=np.float32)
