"""Live signal-timing advisory from camera feeds. ADVISORY ONLY: it recommends; it never drives a signal.

Every second the advisor turns each camera's live perception into the same TrafficState the
controllers use, runs the CAMP engine + CAMP rule controller, and passes the request through the
same SafetyFilter (min/max green, 3 s yellow, 2 s all-red, starvation bound). Mode:

  default_loop  fixed 30 s green per direction, used when
                - all four directions have a working camera and have been empty for `clear_after_s`, or
                - a whole road (N-S or E-W) has no working camera, so its demand is unknown
  adaptive      CAMP decides when to switch, from live counts, queues, arrivals and measured waits

Directions without a working camera are reported as unknown and fed to CAMP as stale zeros; the
advisor never invents traffic for them. Remaining-green figures in adaptive mode are estimates
(queue discharge at ~1 vehicle per second per approach plus 2 s start-up), re-decided every second.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Callable

from camp_crl.config import APPROACHES, AppConfig
from camp_crl.control.controllers import CampRuleController, FixedTimeController
from camp_crl.control.safety import Mode, SafetyFilter
from camp_crl.sim.network import PHASE_APPROACHES, PHASE_NAMES, build_network, signal_plan
from camp_crl.state import ApproachState, TrafficState
from camp_crl.vision.types import ApproachPerception

log = logging.getLogger(__name__)

PHASE_LABEL = {0: "North–South", 1: "East–West"}
FREE_FLOW_MPS = 13.89
APPROACH_LENGTH_M = 190.0
STARTUP_S = 2.0      # start-up lost time before a queue moves
DISCHARGE_S = 1.0    # ~1 vehicle per second leaves a two-lane approach at saturation
ARRIVAL_TAU_S = 60.0  # arrival-rate averaging time


class Advisor:
    """Pure decision logic (no threads, no cameras): feed it perceptions with `tick`."""

    def __init__(self, cfg: AppConfig, default_green_s: float = 30.0, clear_after_s: float = 5.0) -> None:
        t = cfg.timing
        if not t.min_green_s <= default_green_s <= t.max_green_s:
            raise ValueError(f"default green {default_green_s}s outside [{t.min_green_s}, {t.max_green_s}]")
        self.cfg, self.default_green_s, self.clear_after_s = cfg, default_green_s, clear_after_s
        self.safety = SafetyFilter(signal_plan(build_network()), t, initial_phase=0)
        self.camp = CampRuleController(cfg)
        self.loop = FixedTimeController((default_green_s, default_green_s))
        self._t: float | None = None
        self._clear_since: float | None = None
        self._arr_prev: dict[str, int] = {}
        self._arr_rate: dict[str, float] = {a: 0.0 for a in APPROACHES}
        self.mode = "default_loop"

    def _state(self, t: float, per: dict[str, ApproachPerception | None], waits: dict[str, float],
               dt: float) -> tuple[TrafficState, set[str]]:
        known, out = set(), {}
        for a in APPROACHES:
            p = per.get(a)
            if p is None or not p.status.usable:
                out[a] = ApproachState(0, 0.0, 0.0, FREE_FLOW_MPS, 0.0, 0.0, 0.0, 0.0, 0.0)
                self._arr_prev.pop(a, None)
                continue
            known.add(a)
            new = max(0, p.arrivals_total - self._arr_prev.get(a, p.arrivals_total))
            self._arr_prev[a] = p.arrivals_total
            if dt > 0:  # ~60 s average: one car must not read as hundreds per hour
                k = min(1.0, dt / ARRIVAL_TAU_S)
                self._arr_rate[a] = (1 - k) * self._arr_rate[a] + k * new / dt * 3600.0
            moving_share = 1.0 - p.queue_veh / max(p.vehicle_count, 1)
            w = waits.get(a, 0.0)
            out[a] = ApproachState(
                vehicle_count=p.vehicle_count, queue_veh=float(p.queue_veh), arrival_veh_per_h=self._arr_rate[a],
                mean_speed_mps=FREE_FLOW_MPS * moving_share if p.vehicle_count else FREE_FLOW_MPS,
                occupancy_pct=min(100.0, p.vehicle_count * 7.5 / (2 * 40.0) * 100.0),  # view ~40 m, 2 lanes
                max_wait_s=w, mean_wait_s=w / 2.0,
                spillback_ratio=min(1.0, p.queue_veh * 7.5 / 2 / APPROACH_LENGTH_M),
                downstream_occupancy_pct=0.0, stopline_count=p.vehicle_count, arrivals_step=new)
        stale = tuple(a for a in APPROACHES if a not in known)
        return TrafficState(t, out, source="camera_live", stale=stale), known

    def _green_estimate(self, state: TrafficState, phase: int) -> float:
        q = max(state.approaches[a].queue_veh for a in PHASE_APPROACHES[phase])
        t = self.cfg.timing
        return float(min(t.max_green_s, max(t.min_green_s, math.ceil(STARTUP_S + q * DISCHARGE_S))))

    def tick(self, t: float, per: dict[str, ApproachPerception | None], waits: dict[str, float] | None = None) -> dict:
        dt = 1.0 if self._t is None else max(0.0, t - self._t)
        self._t = t
        state, known = self._state(t, per, waits or {}, dt)
        seen_phase = {p: any(a in known for a in PHASE_APPROACHES[p]) for p in PHASE_APPROACHES}
        total = sum(state.approaches[a].vehicle_count for a in known)
        if len(known) == 4 and total == 0:
            self._clear_since = t if self._clear_since is None else self._clear_since
        else:
            self._clear_since = None
        all_clear = self._clear_since is not None and t - self._clear_since >= self.clear_after_s

        status = self.safety.status
        camp_decision = self.camp.act(state, status)  # always run, so CAMP's trends stay current
        if not all(seen_phase.values()):
            blind = [PHASE_LABEL[p] for p, ok in seen_phase.items() if not ok]
            mode, why = "default_loop", f"no working camera on the {' and '.join(blind)} road, so its traffic is unknown"
        elif all_clear:
            mode, why = "default_loop", f"all four directions clear for {self.clear_after_s:.0f}+ s"
        elif status.mode is not Mode.GREEN:
            mode, why = "adaptive", f"changing to {PHASE_LABEL[status.target_phase]} green (yellow and all-red times are fixed)"
        else:
            mode, why = "adaptive", camp_decision.info.get("decision_reason", "CAMP rule")
        if mode != self.mode:
            log.info("advisory mode %s -> %s (%s)", self.mode, mode, why)
        self.mode = mode
        request = self.loop.act(state, status).requested_phase if mode == "default_loop" else camp_decision.requested_phase
        cmd = self.safety.step(request, dt if dt > 0 else 1.0)
        return self._snapshot(t, state, known, cmd, why, camp_decision.info)

    def _snapshot(self, t: float, state: TrafficState, known: set[str], cmd, why: str, info: dict) -> dict:
        st = self.safety.status
        tm = self.cfg.timing
        g, other = cmd.green_phase, 1 - cmd.green_phase
        if self.mode == "default_loop":
            green_cur, green_next = self.default_green_s, self.default_green_s
        else:
            green_cur, green_next = self._green_estimate(state, g), self._green_estimate(state, other)
        if cmd.mode is Mode.GREEN:
            el = st.time_in_mode_s
            most = max(0.0, tm.max_green_s - el)
            remaining: float | None = max(0.0, green_cur - el)
            if self.mode == "adaptive":
                remaining = min(max(remaining, tm.min_green_s - el, 0.0), most)
                if remaining == 0 and request_holds(info):
                    remaining = None  # CAMP keeps the green while demand lasts; only the maximum is known
            plan = [{"label": f"{PHASE_LABEL[g]} green", "color": "green",
                     "seconds": None if remaining is None else round(remaining), "max_seconds": round(most),
                     "note": "holding while traffic keeps coming (re-decided every second)" if remaining is None else None},
                    {"label": f"{PHASE_LABEL[g]} yellow", "color": "yellow", "seconds": tm.yellow_s},
                    {"label": "All red", "color": "red", "seconds": tm.all_red_s},
                    {"label": f"{PHASE_LABEL[other]} green", "color": "green", "seconds": round(green_next)}]
        else:
            left = (tm.yellow_s if cmd.mode is Mode.YELLOW else tm.all_red_s) - st.time_in_mode_s
            plan = ([{"label": f"{PHASE_LABEL[g]} yellow", "color": "yellow", "seconds": round(max(0.0, left))},
                     {"label": "All red", "color": "red", "seconds": tm.all_red_s}] if cmd.mode is Mode.YELLOW
                    else [{"label": "All red", "color": "red", "seconds": round(max(0.0, left))}])
            plan.append({"label": f"{PHASE_LABEL[cmd.target_phase]} green", "color": "green",
                         "seconds": round(self.default_green_s if self.mode == "default_loop"
                                          else self._green_estimate(state, cmd.target_phase))})
        ctx = self.camp.last_context
        approaches = {}
        for a in APPROACHES:
            on_green = a in PHASE_APPROACHES[g]
            light = ("green" if cmd.mode is Mode.GREEN else "yellow" if cmd.mode is Mode.YELLOW else "red") if on_green else "red"
            s = state.approaches[a]
            c = ctx.approaches[a] if ctx else None
            approaches[a] = {"known": a in known, "light": light, "vehicles": s.vehicle_count if a in known else None,
                             "queue": s.queue_veh if a in known else None,
                             "arrivals_veh_h": round(s.arrival_veh_per_h) if a in known else None,
                             "longest_wait_s": round(s.max_wait_s, 1) if a in known else None,
                             "trend": c.trend if (c and a in known) else None,
                             "surge": bool(c.anomaly) if (c and a in known) else None,
                             "urgency": round(c.composite, 3) if (c and a in known) else None,
                             "red_for_s": round(st.red_time_s.get(PHASE_APPROACHES_INV[a], 0.0)) if not on_green else 0}
        return {"time": t, "advisory_only": True, "mode": self.mode, "mode_reason": why,
                "strategy": info.get("strategy"), "strategy_reason": info.get("strategy_reason"),
                "pattern": ctx.pattern if ctx and known else None,
                "signal": {"mode": cmd.mode.value, "green_phase": g, "green_label": PHASE_LABEL[g],
                           "phase_name": PHASE_NAMES[g], "target_phase": cmd.target_phase,
                           "time_in_interval_s": round(st.time_in_mode_s)},
                "plan": plan, "approaches": approaches, "cameras_working": len(known),
                "interventions": [i.kind for i in cmd.interventions],
                "default_green_s": self.default_green_s,
                "timing": {"min_green_s": tm.min_green_s, "max_green_s": tm.max_green_s, "yellow_s": tm.yellow_s,
                           "all_red_s": tm.all_red_s, "max_red_s": tm.max_red_s}}


PHASE_APPROACHES_INV = {a: p for p, aa in PHASE_APPROACHES.items() for a in aa}


def request_holds(info: dict) -> bool:
    return str(info.get("decision_reason", "")).startswith("hold")


class LiveAdvisory:
    """Runs an Advisor once per second on the dashboard's live camera previews."""

    def __init__(self, cfg_provider: Callable[[], AppConfig], previews, camera_approaches: Callable[[], list[str]],
                 default_green_s: float = 30.0) -> None:
        self.cfg_provider, self.previews, self.camera_approaches = cfg_provider, previews, camera_approaches
        self.default_green_s = default_green_s
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.latest: dict | None = None
        self.error: str | None = None
        self.started_at: float | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        self.stop()
        advisor = Advisor(self.cfg_provider(), self.default_green_s)  # validate before starting any camera
        cams = self.camera_approaches()
        for a in cams:
            if not self.previews.running(a):
                self.previews.start(a)
        self._stop.clear()
        self.error, self.latest, self.started_at = None, None, time.monotonic()
        self._thread = threading.Thread(target=self._loop, args=(advisor, cams), name="live-advisory", daemon=True)
        self._thread.start()

    def _loop(self, advisor: Advisor, cams: list[str]) -> None:
        try:
            while not self._stop.is_set():
                t0 = time.monotonic()
                per, waits = {}, {}
                for a in cams:
                    p, w = self.previews.latest(a)
                    if p is None and not self._stop.is_set():  # preview idled out or crashed: bring it back
                        try:
                            self.previews.start(a)
                        except Exception as exc:
                            log.warning("advisory could not restart camera %s: %s", a, exc)
                    per[a], waits[a] = p, w
                snap = advisor.tick(t0 - self.started_at, per, waits)
                with self._lock:
                    self.latest = snap
                self._stop.wait(max(0.0, 1.0 - (time.monotonic() - t0)))
        except Exception as exc:
            log.exception("live advisory crashed")
            with self._lock:
                self.error = str(exc)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._thread = None

    def status(self) -> dict:
        with self._lock:
            return {"running": self.running, "error": self.error, "advice": self.latest}
