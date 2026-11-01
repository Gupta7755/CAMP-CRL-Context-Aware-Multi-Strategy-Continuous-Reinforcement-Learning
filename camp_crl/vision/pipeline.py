"""One camera's perception loop: source -> detector -> tracker -> approach mapper -> health.

`step(now)` is synchronous and clock-agnostic, so the same code runs the deterministic offline demo
(now = simulation time) and live phone streams (now = time.monotonic(), driven by StreamManager).
Reconnects use exponential backoff from `retry_s` up to `retry_max_s`.
A frame whose timestamp is not newer than the last processed one (a frozen stream re-delivering
the same image) is ignored, so health sees it as missing and the camera goes STALE.
A frame whose grey-level standard deviation is below `min_contrast` (lens covered, view blocked)
marks the camera DEGRADED.
Every new frame is kept as `latest_raw` (for the live video on the dashboard); detection runs on every
`process_every_n`-th frame and at most once per `detect_interval_s`, so a 25-30 fps phone stream can be
shown at full rate while detection runs at ~10 Hz. `live_image()` draws the latest detections onto the
latest raw frame and is safe to call from another thread.
"""

from __future__ import annotations

import logging

import numpy as np

from camp_crl.vision.detector import Detector
from camp_crl.vision.health import HealthConfig, HealthMonitor
from camp_crl.vision.mapping import ApproachMapper
from camp_crl.vision.sources import FrameSource, SourceError
from camp_crl.vision.tracker import ByteTracker
from camp_crl.vision.types import ApproachPerception, Frame, Status

log = logging.getLogger(__name__)


class CameraPipeline:
    def __init__(self, approach: str, source: FrameSource, detector: Detector, mapper: ApproachMapper,
                 tracker: ByteTracker | None = None, health: HealthConfig | None = None,
                 process_every_n: int = 1, retry_s: float = 1.0, retry_max_s: float = 30.0,
                 min_contrast: float = 6.0, detect_interval_s: float = 0.0) -> None:
        self.approach, self.source, self.detector, self.mapper = approach, source, detector, mapper
        self.tracker = tracker or ByteTracker()
        self.health = HealthMonitor(health or HealthConfig())
        self.every_n = max(1, process_every_n)
        self.retry_s, self.retry_max_s, self.min_contrast = retry_s, retry_max_s, min_contrast
        self._open = False
        self._next_retry = 0.0
        self._backoff = retry_s
        self._frames_seen = 0
        self._last_ts: float | None = None
        self.last_frame: Frame | None = None
        self.last_contrast: float | None = None
        self.counts = (0, 0, 0)  # vehicle_count, queue_veh, arrivals_total
        self.detect_ms: float | None = None
        self.detect_interval_s = detect_interval_s
        self._last_detect_ts: float | None = None
        self.latest_raw: Frame | None = None
        self.raw_frames = 0
        self.raw_fps = 0.0
        # (id, box, queued) of confirmed tracks after the last detection; replaced, never mutated
        self.track_snapshot: list[tuple[int, tuple[int, int, int, int], bool]] = []
        self.max_wait_s = 0.0
        self.events: list[tuple[float, str]] = []  # (time, message) log of connection changes

    def _note(self, now: float, msg: str) -> None:
        self.events.append((now, msg))
        log.info("camera %s: %s", self.approach, msg)

    def _try_open(self, now: float) -> None:
        try:
            self.source.open(now)
        except Exception as exc:  # any failure to open is a connection failure
            self.health.on_disconnect(str(exc))
            self._next_retry = now + self._backoff
            self._note(now, f"open failed ({exc}); retry in {self._backoff:.0f}s")
            self._backoff = min(self._backoff * 2, self.retry_max_s)
            return
        self._open, self._backoff = True, self.retry_s
        self._note(now, f"connected to {self.source.name}")

    def step(self, now: float) -> ApproachPerception:
        if not self._open and now >= self._next_retry:
            self._try_open(now)
        if self._open:
            try:
                frame = self.source.read(now)
            except SourceError as exc:
                self.source.close()
                self._open = False
                self.health.on_disconnect(str(exc))
                self._next_retry = now + self._backoff
                self._note(now, f"disconnected ({exc})")
                frame = None
            if frame is not None and (self._last_ts is None or frame.ts > self._last_ts):
                self._process(frame)
        else:
            tick = getattr(self.source, "tick", None) or getattr(self.source, "_tick", None)
            if tick is not None:  # synthetic scenes keep moving while the camera is down
                tick(now)
        return self.perception(now)

    def _process(self, frame: Frame) -> None:
        import time

        if self._last_ts is not None and frame.ts > self._last_ts and self.raw_frames:
            inst = 1.0 / (frame.ts - self._last_ts)
            self.raw_fps = inst if self.raw_frames == 1 else 0.2 * inst + 0.8 * self.raw_fps
        self._last_ts = frame.ts
        self.latest_raw = frame
        self.raw_frames += 1
        self._frames_seen += 1
        if (self._frames_seen - 1) % self.every_n:
            return
        if (self.detect_interval_s and self._last_detect_ts is not None
                and frame.ts - self._last_detect_ts < self.detect_interval_s - 1e-6):
            return
        self._last_detect_ts = frame.ts
        self.last_frame = frame
        grey = frame.image[::4, ::4].mean(axis=2)
        self.last_contrast = float(grey.std())
        t0 = time.perf_counter()
        dets = self.detector.detect(frame)
        self.detect_ms = (time.perf_counter() - t0) * 1000
        self.tracker.update(dets, frame.ts)
        active = self.tracker.active()
        h, w = frame.image.shape[:2]
        self.counts = self.mapper.update(active, frame.ts, w, h)
        self.track_snapshot = [(t.id, tuple(int(v) for v in t.box), t.still_since is not None) for t in active]
        # longest time any vehicle in the counting region has been stopped, measured from the video
        self.max_wait_s = max((frame.ts - t.still_since for t in active
                               if t.still_since is not None and self.mapper.in_roi(t, w, h)), default=0.0)
        self.health.on_frame(frame.ts, [t.score for t in active if self.mapper.in_roi(t, w, h)])

    def perception(self, now: float) -> ApproachPerception:
        status, conf, reason, age = self.health.evaluate(now)
        if status.usable and self.last_contrast is not None and self.last_contrast < self.min_contrast:
            status, conf = Status.DEGRADED, conf * 0.25
            reason = f"view obstructed (image contrast {self.last_contrast:.1f} < {self.min_contrast})"
        count, queue, arrivals = self.counts
        return ApproachPerception(self.approach, now, count, queue, arrivals, self.health.fps, conf, status,
                                  reason, age if np.isfinite(age) else -1.0)

    def close(self) -> None:
        if self._open:
            self.source.close()
            self._open = False

    def _annotate(self, img: np.ndarray, ts: float) -> np.ndarray:
        import cv2

        h, w = img.shape[:2]
        k = max(1.0, w / 640)  # keep lines and text readable on 720p/1080p streams
        poly = np.array([(x * w, y * h) for x, y in self.mapper.roi], np.int32)
        cv2.polylines(img, [poly], True, (212, 182, 6), max(2, int(2 * k)))  # cyan #06B6D4 in BGR
        for tid, (x1, y1, x2, y2), queued in self.track_snapshot:
            colour = (11, 158, 245) if queued else (235, 99, 37)  # amber = queued, blue = moving
            cv2.rectangle(img, (x1, y1), (x2, y2), colour, max(2, int(2 * k)))
            cv2.putText(img, str(tid), (x1, max(int(10 * k), y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * k, colour,
                        max(1, int(k)))
        p = self.perception(ts)
        label = f"{self.approach} {p.status.value} n={p.vehicle_count} q={p.queue_veh} fps={p.fps:.1f}"
        cv2.rectangle(img, (0, 0), (w, int(16 * k)), (39, 24, 16), -1)
        cv2.putText(img, label, (4, int(12 * k)), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * k, (240, 240, 240), max(1, int(k)))
        return img

    def overlay(self) -> np.ndarray | None:
        """Last processed frame annotated with ROI, tracks and status (for the dashboard)."""
        if self.last_frame is None:
            return None
        return self._annotate(self.last_frame.image.copy(), self.last_frame.ts)

    def live_image(self, annotate: bool = True) -> tuple[int, np.ndarray] | None:
        """(frame number, latest raw frame) with the latest detections drawn on it if `annotate`."""
        frame, n = self.latest_raw, self.raw_frames
        if frame is None:
            return None
        return n, self._annotate(frame.image.copy(), frame.ts) if annotate else frame.image
