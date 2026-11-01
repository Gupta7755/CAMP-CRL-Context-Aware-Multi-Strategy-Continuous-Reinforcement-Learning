"""Camera demand hooks for runner.Episode (`external_demand`): perception -> bridge -> SUMO vehicles.

CameraDemand      pipelines stepped on the *simulation* clock: between SUMO steps t-1 and t each
                  pipeline runs `fps` sub-steps. Deterministic (same config + seed -> same run); used by
                  the offline demo and by the dashboard when every camera is synthetic.
LiveCameraDemand  pipelines run in one thread each on the *wall* clock (time.monotonic), as real phone
                  streams require; each SUMO step reads the latest perception. The simulation must then
                  run in real time (speed 1), otherwise SUMO time and camera time drift apart.
Both append one row per approach and step to `rows`, and keep a JPEG overlay per approach for the API.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time

from camp_crl.config import APPROACHES
from camp_crl.vision.bridge import CameraDemandBridge
from camp_crl.vision.pipeline import CameraPipeline
from camp_crl.vision.types import ApproachPerception, Status

log = logging.getLogger(__name__)


def _jpeg(pipeline: CameraPipeline) -> bytes | None:
    import cv2

    img = pipeline.overlay()
    if img is None:
        return None
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok else None


class _Base:
    clock_mode = "sim"

    def __init__(self, pipelines: dict[str, CameraPipeline], bridge: CameraDemandBridge) -> None:
        self.pipelines, self.bridge = pipelines, bridge
        self.rows: list[dict] = []
        self.latest: dict[str, ApproachPerception] = {}
        self._lock = threading.Lock()
        self._overlays: dict[str, bytes] = {}

    def _record(self, t: float, perceptions: dict[str, ApproachPerception], inserted: dict[str, int]) -> dict:
        out = {}
        for a in APPROACHES:
            p = perceptions.get(a)
            row = {"time_s": t, "approach": a, "mode": self.bridge.mode[a], "inserted": inserted[a],
                   "fallback_rate_veh_h": round(self.bridge.rate[a], 1)}
            if p is not None:
                row.update({k: v for k, v in p.to_dict().items() if k not in ("approach", "ts")})
                src = self.pipelines[a].source
                inner = getattr(src, "inner", src)
                if hasattr(inner, "true_arrivals"):
                    row["true_arrivals"] = inner.true_arrivals
            self.rows.append(row)
            out[a] = row
        return out

    def status(self) -> dict:
        """Latest row per approach plus totals, for the API (thread-safe)."""
        with self._lock:
            last = {r["approach"]: dict(r) for r in self.rows[-len(APPROACHES):]} if self.rows else {}
        for a, r in last.items():
            r["inserted_total"] = self.bridge.inserted[a]
            r["source"] = self.pipelines[a].source.name if a in self.pipelines else None
            r["has_camera"] = a in self.pipelines
        return {"clock": self.clock_mode, "approaches": last}

    def overlay_jpeg(self, approach: str) -> bytes | None:
        with self._lock:
            return self._overlays.get(approach)


class CameraDemand(_Base):
    def __init__(self, pipelines: dict[str, CameraPipeline], bridge: CameraDemandBridge, fps: float,
                 snapshot_every_s: float = 0.0, live_overlays: bool = False) -> None:
        super().__init__(pipelines, bridge)
        self.fps = fps
        self.snapshot_every_s = snapshot_every_s
        self.live_overlays = live_overlays
        self.snapshots: list[tuple[str, float, object]] = []  # (approach, time, annotated BGR image)
        self._modes: dict[str, str] = {}
        self._t: float | None = None

    def advance(self, t: float) -> dict[str, ApproachPerception]:
        start = t if self._t is None else self._t
        n = max(1, round((t - start) * self.fps))
        for k in range(1, n + 1):
            now = start + (t - start) * k / n
            for a, p in self.pipelines.items():
                self.latest[a] = p.step(now)
        self._t = t
        return self.latest

    def step(self, conn, t: float) -> dict:
        perceptions = self.advance(t)
        inserted = self.bridge.step(conn, t, perceptions)
        self._snapshot(t)
        with self._lock:
            out = self._record(t, perceptions, inserted)
            if self.live_overlays:  # same thread as the pipelines, so drawing cannot race with tracking
                for a, p in self.pipelines.items():
                    jpg = _jpeg(p)
                    if jpg is not None:
                        self._overlays[a] = jpg
        return out

    def _snapshot(self, t: float) -> None:
        """Keep annotated frames every `snapshot_every_s` and whenever an approach changes mode."""
        if not self.snapshot_every_s:
            return
        periodic = int(t) % int(self.snapshot_every_s) == 0
        for a, p in self.pipelines.items():
            changed = self._modes.get(a) != self.bridge.mode[a]
            self._modes[a] = self.bridge.mode[a]
            if periodic or changed:
                img = p.overlay()
                if img is not None:
                    self.snapshots.append((a, t, img))

    def write_overlays(self, out_dir) -> int:
        import cv2

        out_dir.mkdir(parents=True, exist_ok=True)
        for a, t, img in self.snapshots:
            cv2.imwrite(str(out_dir / f"{a}_{int(t):05d}.jpg"), img)
        return len(self.snapshots)

    def close(self) -> None:
        for p in self.pipelines.values():
            p.close()


class LiveCameraDemand(_Base):
    clock_mode = "wall"

    def __init__(self, pipelines: dict[str, CameraPipeline], bridge: CameraDemandBridge, max_fps: float,
                 stale_after_s: float, clock=time.monotonic, overlay_every_s: float = 1.0) -> None:
        super().__init__(pipelines, bridge)
        self.max_fps, self.stale_after_s, self.clock = max_fps, stale_after_s, clock
        self.overlay_every_s = overlay_every_s
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.errors: dict[str, str] = {}

    def start(self) -> None:
        for a, p in self.pipelines.items():
            th = threading.Thread(target=self._loop, args=(a, p), name=f"camera-{a}", daemon=True)
            th.start()
            self._threads.append(th)

    def _loop(self, a: str, p: CameraPipeline) -> None:
        next_overlay = 0.0
        period = 1.0 / self.max_fps
        try:
            while not self._stop.is_set():
                now = self.clock()
                per = p.step(now)
                jpg = _jpeg(p) if now >= next_overlay else None
                with self._lock:
                    self.latest[a] = per
                    if jpg is not None:
                        self._overlays[a] = jpg
                if jpg is not None:
                    next_overlay = now + self.overlay_every_s
                self._stop.wait(max(0.0, period - (self.clock() - now)))
        except Exception as exc:  # a crashed camera thread must not take the simulation down
            log.exception("camera %s thread crashed", a)
            with self._lock:
                self.errors[a] = str(exc)
                self.latest.pop(a, None)  # the bridge then treats the approach as having no camera

    def current(self) -> dict[str, ApproachPerception]:
        """Latest perception per approach; a value not refreshed for stale_after_s is reported STALE
        (covers a camera thread that hangs inside a blocking read)."""
        now = self.clock()
        with self._lock:
            latest = dict(self.latest)
        out = {}
        for a, p in latest.items():
            if p.status.usable and now - p.ts > self.stale_after_s:
                p = dataclasses.replace(p, status=Status.STALE, confidence=0.0, frame_age_s=now - p.ts,
                                        reason=f"camera thread silent for {now - p.ts:.1f}s")
            out[a] = p
        return out

    def step(self, conn, t: float) -> dict:
        perceptions = self.current()
        inserted = self.bridge.step(conn, t, perceptions)
        with self._lock:
            return self._record(t, perceptions, inserted)

    def close(self) -> None:
        self._stop.set()
        for th in self._threads:
            th.join(timeout=5.0)
        for p in self.pipelines.values():
            p.close()


def all_synthetic(vcfg) -> bool:
    return all(c.source == "synthetic" for c in vcfg.cameras.values())


def make_camera_demand(vcfg, seed: int, turn_shares: tuple[float, float, float]) -> _Base:
    """Sim-clock CameraDemand when every camera is synthetic, otherwise a started LiveCameraDemand.

    Approaches without a configured camera get fallback demand from the bridge (default rate)."""
    from camp_crl.vision.config import build_pipelines

    pipelines = build_pipelines(vcfg, seed)
    bridge = CameraDemandBridge(vcfg.bridge.to_bridge(), turn_shares, seed)
    if all_synthetic(vcfg):
        return CameraDemand(pipelines, bridge, vcfg.fps, live_overlays=True)
    demand = LiveCameraDemand(pipelines, bridge, vcfg.fps, vcfg.health.stale_after_s)
    demand.start()
    return demand
