"""Live simulation manager: runs one Episode in a background thread, paced for viewing.

Only one live simulation can run at a time. All public methods are thread-safe.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from datetime import datetime
from typing import Callable

from camp_crl.config import APPROACHES, AppConfig
from camp_crl.runner import Episode

log = logging.getLogger(__name__)

HISTORY_POINTS = 900


class AlreadyRunningError(RuntimeError):
    pass


def history_point(snap: dict) -> dict:
    """Compact per-step record for charts."""
    truth = snap["truth"]["approaches"]
    point = {"t": snap["time_s"], "phase": snap["signal"]["green_phase"], "mode": snap["signal"]["mode"],
             "total_queue": snap["kpi"]["total_queue_veh"], "arrived": snap["kpi"]["arrived_total"],
             "mean_wait": round(snap["kpi"]["mean_wait_s"], 2)}
    for a in APPROACHES:
        point[f"q_{a}"] = truth[a]["queue_veh"]
        point[f"w_{a}"] = round(truth[a]["max_wait_s"], 1)
    if snap.get("camp"):
        for a in APPROACHES:
            point[f"p_{a}"] = round(snap["camp"]["approaches"][a]["pressure"], 3)
            point[f"c_{a}"] = round(snap["camp"]["approaches"][a]["composite"], 3)
    return point


class LiveSimulation:
    def __init__(self, cfg_provider: Callable[[], AppConfig], vision_provider: Callable | None = None) -> None:
        self._cfg_provider = cfg_provider
        self._vision_provider = vision_provider  # Phase 3: VisionConfig with dashboard camera sources merged
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.status = "idle"  # idle | starting | running | stopping | finished | stopped | error
        self.params: dict | None = None
        self.error: str | None = None
        self.latest: dict | None = None
        self.history: deque[dict] = deque(maxlen=HISTORY_POINTS)
        self.metrics: dict | None = None
        self.run_dir: str | None = None
        self.started_at: str | None = None
        self.last_update_wall: float | None = None
        self.step_latency_ms: float | None = None
        self.seq = 0  # increments with every new snapshot
        self.camera_demand = None  # Phase 3: set when a run uses camera-driven demand

    def start(self, controller: str, scenario: str, seed: int, speed: float, cameras: bool = False) -> dict:
        """Start a run. With cameras=True, demand comes from configs/vision.yaml cameras (Phase 3); raises
        ValueError if that config is unusable, or if real streams are configured and speed != 1."""
        with self._lock:
            if self.status in ("starting", "running", "stopping"):
                raise AlreadyRunningError(f"a simulation is already {self.status}")
            cfg = self._cfg_provider()
            demand = self._camera_demand(cfg, scenario, seed, speed) if cameras else None
            self.status, self.error, self.latest, self.metrics, self.run_dir = "starting", None, None, None, None
            self.history.clear()
            self.params = {"controller": controller, "scenario": scenario, "seed": seed, "speed": speed,
                           "camera_demand": cameras, "camera_clock": demand.clock_mode if demand else None}
            self.camera_demand = demand
            self.started_at = datetime.now().isoformat(timespec="seconds")
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, args=(cfg, controller, scenario, seed, speed, demand),
                                            name="live-sim", daemon=True)
            self._thread.start()
            return self.snapshot_status()

    def stop(self, timeout: float = 15.0) -> dict:
        with self._lock:
            thread = self._thread
            if self.status in ("starting", "running"):
                self.status = "stopping"
            self._stop.set()
        if thread is not None:
            thread.join(timeout)
        return self.snapshot_status()

    def _camera_demand(self, cfg: AppConfig, scenario: str, seed: int, speed: float):
        from pydantic import ValidationError

        from camp_crl.vision.config import load_vision_config
        from camp_crl.vision.session import all_synthetic, make_camera_demand

        try:
            vcfg = self._vision_provider() if self._vision_provider else load_vision_config()
        except (FileNotFoundError, ValidationError) as exc:
            raise ValueError(f"camera demand unavailable: {exc}") from exc
        if not vcfg.cameras:
            raise ValueError("no cameras configured: add at least one on the Cameras page")
        if not all_synthetic(vcfg) and speed != 1:
            raise ValueError("real camera streams run on the wall clock: set speed to 1 (real time)")
        if scenario not in cfg.scenarios:
            raise ValueError(f"unknown scenario '{scenario}'")
        try:
            return make_camera_demand(vcfg, seed, cfg.scenarios[scenario].turn_shares)
        except FileNotFoundError as exc:  # e.g. YOLOX model not downloaded
            raise ValueError(str(exc)) from exc

    def _run(self, cfg: AppConfig, controller: str, scenario: str, seed: int, speed: float, demand=None) -> None:
        ep: Episode | None = None
        try:
            ep = Episode(cfg, controller, scenario, seed, observe_camp=True, external_demand=demand)
            with self._lock:
                self.run_dir = str(ep.run_dir)
                if self.status == "starting":
                    self.status = "running"
            interval = 1.0 / speed if speed > 0 else 0.0
            next_t = time.perf_counter()
            while not ep.done and not self._stop.is_set():
                t0 = time.perf_counter()
                snap = ep.step()
                latency = (time.perf_counter() - t0) * 1000
                with self._lock:
                    self.latest = snap
                    self.history.append(history_point(snap))
                    self.last_update_wall = time.time()
                    self.step_latency_ms = latency
                    self.seq += 1
                if interval:
                    next_t += interval
                    delay = next_t - time.perf_counter()
                    if delay > 0:
                        self._stop.wait(delay)
                    else:
                        next_t = time.perf_counter()
            if self._stop.is_set() and not ep.done:
                ep.abort()
                with self._lock:
                    self.status = "stopped"
            else:
                result = ep.finish()
                with self._lock:
                    self.metrics = result.metrics
                    self.status = "finished"
        except Exception as exc:
            log.exception("live simulation failed")
            if ep is not None:
                try:
                    ep.abort()
                except Exception:
                    log.exception("abort after failure also failed")
            with self._lock:
                self.status, self.error = "error", str(exc)
        finally:
            if demand is not None:
                demand.close()

    def snapshot_status(self) -> dict:
        age = None if self.last_update_wall is None else round(time.time() - self.last_update_wall, 2)
        latest = self.latest
        return {
            "status": self.status,
            "params": self.params,
            "error": self.error,
            "started_at": self.started_at,
            "step": latest["step"] if latest else 0,
            "total_steps": latest["total_steps"] if latest else None,
            "sim_time_s": latest["time_s"] if latest else None,
            "last_update_age_s": age,
            "step_latency_ms": None if self.step_latency_ms is None else round(self.step_latency_ms, 2),
            "run_dir": self.run_dir,
            "metrics": self.metrics,
            "seq": self.seq,
        }
