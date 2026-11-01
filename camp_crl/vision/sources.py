"""Frame sources behind one interface, plus a fault injector for tests and the demo.

`read(now)` contract:
  returns a Frame   a new frame, stamped with the consumer's clock `now`
  returns None      no new frame yet (normal for low frame rates)
  raises SourceError  the source failed; the pipeline closes it and reconnects with backoff

OpenCVSource      phone stream (http MJPEG from e.g. Android "IP Webcam", rtsp), webcam index, or video file
SyntheticCamera   a deterministic one-approach road scene with its own signal and Poisson arrivals;
                  publishes ground-truth boxes and arrival counts so perception error can be measured
FaultInjector     wraps any source with scheduled disconnect / low_fps / occlusion / stale (frozen) faults
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from camp_crl.vision.types import Frame


class SourceError(RuntimeError):
    pass


class FrameSource(Protocol):
    name: str

    def open(self, now: float) -> None: ...

    def read(self, now: float) -> Frame | None: ...

    def close(self) -> None: ...


_CRED = re.compile(r"(//)([^/@:]+)(:[^/@]*)?@")


def redact(uri: str) -> str:
    """Hide credentials in stream URLs before they are logged or returned by the API."""
    return _CRED.sub(r"\1***@", uri)


class OpenCVSource:
    """cv2.VideoCapture over a file path, http(s)/rtsp URL or integer webcam index.

    Files are paced: one file frame per 1/file_fps of consumer time, and loop at the end, so a
    recorded clip behaves like a live feed and replays identically.
    """

    def __init__(self, uri: str, loop: bool = True, timeout_ms: int = 5000) -> None:
        self.uri, self.loop, self.timeout_ms = uri, loop, timeout_ms
        self.name = redact(uri)
        self.is_file = not uri.isdigit() and "://" not in uri
        self._cap = None
        self._seq = 0
        self._file_fps = 0.0
        self._next_due: float | None = None

    def open(self, now: float = 0.0) -> None:
        import cv2

        if self.is_file and not Path(self.uri).is_file():
            raise SourceError(f"video file not found: {self.uri}")
        if "://" in self.uri:  # network stream: bounded waits, so a frozen phone cannot hang the camera thread
            cap = cv2.VideoCapture(self.uri, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.timeout_ms,
                                                              cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.timeout_ms])
        else:
            cap = cv2.VideoCapture(int(self.uri) if self.uri.isdigit() else self.uri)
        if not cap.isOpened():
            cap.release()
            raise SourceError(f"cannot open {self.name}")
        self._cap = cap
        self._file_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 10.0
        self._next_due = None

    def read(self, now: float) -> Frame | None:
        import cv2

        if self._cap is None:
            raise SourceError("source not open")
        if self.is_file:
            if self._next_due is not None and now < self._next_due - 1e-6:  # tolerate float drift
                return None
            self._next_due = (now if self._next_due is None else self._next_due) + 1.0 / self._file_fps
        ok, img = self._cap.read()
        if not ok and self.is_file and self.loop:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, img = self._cap.read()
        if not ok or img is None:
            raise SourceError(f"read failed on {self.name}")
        self._seq += 1
        return Frame(img, now, self._seq)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


@dataclass
class _Car:
    lane: int
    y: float  # front (bottom) edge, px
    v: float = 0.0
    counted: bool = False


@dataclass
class SyntheticCamera:
    """Top-down-ish view of one approach: vehicles enter at the top and drive down to a stop line.

    The scene has its own fixed-cycle signal (the "real world" the camera watches), independent of
    the SUMO signal. Arrivals are Poisson with `rate_veh_per_h`, split over two lanes; vehicles follow
    a simple gap-keeping rule. Everything is seeded, so a (seed, schedule) pair replays exactly.
    """

    rate_veh_per_h: float
    seed: int = 0
    width: int = 320
    height: int = 240
    stopline_frac: float = 0.82
    cycle_s: float = 60.0
    green_s: float = 27.0
    offset_s: float = 0.0
    speed_px_s: float = 70.0
    car_len: float = 22.0
    gap: float = 6.0
    name: str = "synthetic"
    true_arrivals: int = 0
    _cars: list[_Car] = field(default_factory=list)
    _t: float | None = None
    _seq: int = 0
    _open: bool = False

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed + 15485863)
        self._bg = np.zeros((self.height, self.width, 3), np.uint8)
        self._bg[:] = (60, 110, 60)  # verge (BGR)
        x1, x2 = int(self.width * 0.3), int(self.width * 0.7)
        self._bg[:, x1:x2] = (85, 85, 85)
        self._bg[:, self.width // 2 - 1: self.width // 2 + 1] = (220, 220, 220)
        sy = int(self.height * self.stopline_frac)
        self._bg[sy: sy + 3, x1:x2] = (240, 240, 240)
        self._lane_x = (self.width * 0.4, self.width * 0.6)

    def open(self, now: float = 0.0) -> None:
        self._open = True
        self.tick(now)

    def tick(self, now: float) -> None:
        """Advance the scene without producing a frame (the road keeps moving while the camera is down)."""
        self._advance(now)

    def close(self) -> None:
        self._open = False

    def green(self, t: float) -> bool:
        return (t - self.offset_s) % self.cycle_s < self.green_s

    def _advance(self, t: float) -> None:
        if self._t is None:
            self._t = t
            return
        dt = t - self._t
        if dt <= 0:
            return
        self._t = t
        # arrivals over (t - dt, t]
        for _ in range(self.rng.poisson(self.rate_veh_per_h / 3600.0 * dt)):
            lane = int(self.rng.integers(2))
            back = min((c.y - self.car_len for c in self._cars if c.lane == lane), default=1e9)
            if back < self.gap:  # entry blocked: vehicle waits off-screen, i.e. is dropped here
                continue
            self._cars.append(_Car(lane, 0.0, self.speed_px_s))
            self.true_arrivals += 1
        stop_y = self.height * self.stopline_frac
        is_green = self.green(t)
        for lane in (0, 1):
            lane_cars = sorted((c for c in self._cars if c.lane == lane), key=lambda c: -c.y)
            leader_back = 1e9
            for c in lane_cars:
                limit = leader_back - self.gap
                if not is_green and c.y <= stop_y:
                    limit = min(limit, stop_y)
                c.y = min(c.y + self.speed_px_s * dt, max(c.y, limit))
                leader_back = c.y - self.car_len
        self._cars = [c for c in self._cars if c.y - self.car_len < self.height]

    def boxes(self) -> list[tuple[float, float, float, float]]:
        out = []
        for c in self._cars:
            cx, w = self._lane_x[c.lane], 16.0
            y1, y2 = max(0.0, c.y - self.car_len), min(float(self.height), c.y)
            if y2 - y1 > 4:
                out.append((cx - w / 2, y1, cx + w / 2, y2))
        return out

    def read(self, now: float) -> Frame | None:
        if not self._open:
            raise SourceError("synthetic camera closed")
        self._advance(now)
        img = self._bg.copy()
        boxes = self.boxes()
        for k, (x1, y1, x2, y2) in enumerate(boxes):
            img[int(y1): int(y2), int(x1): int(x2)] = ((40 + 50 * k) % 255, 60, 200)
        self._seq += 1
        return Frame(img, now, self._seq, {"gt_boxes": boxes, "true_arrivals": self.true_arrivals,
                                           "signal_green": self.green(now)})


@dataclass(frozen=True)
class Fault:
    kind: str  # disconnect | low_fps | occlusion | stale
    start_s: float
    end_s: float
    fps: float = 1.0          # low_fps: maximum delivered frame rate
    region: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)  # occlusion: normalised x1,y1,x2,y2

    def active(self, t: float) -> bool:
        return self.start_s <= t < self.end_s


class FaultInjector:
    """Wraps a source and applies scheduled faults. Times are on the consumer clock."""

    def __init__(self, inner: FrameSource, faults: list[Fault]) -> None:
        self.inner, self.faults = inner, faults
        self.name = inner.name
        self._last: Frame | None = None
        self._last_delivered = -1e9
        self._now = 0.0

    def _active(self, kind: str, t: float) -> Fault | None:
        return next((f for f in self.faults if f.kind == kind and f.active(t)), None)

    def open(self, now: float = 0.0) -> None:
        self._now = now
        self._tick(now)
        if self._active("disconnect", now):
            raise SourceError("injected fault: camera unreachable")
        self.inner.open(now)

    def _tick(self, now: float) -> None:
        tick = getattr(self.inner, "tick", None)
        if tick is not None:
            tick(now)

    def close(self) -> None:
        self.inner.close()

    def read(self, now: float) -> Frame | None:
        self._now = now
        if self._active("disconnect", now):
            self._tick(now)
            raise SourceError("injected fault: stream dropped")
        if self._active("stale", now) and self._last is not None:
            self.inner.read(now)  # world keeps moving; the consumer keeps getting the frozen frame
            return self._last
        frame = self.inner.read(now)
        if frame is None:
            return None
        low = self._active("low_fps", now)
        if low and now - self._last_delivered < 1.0 / low.fps - 1e-9:
            return None
        occ = self._active("occlusion", now)
        if occ:
            h, w = frame.image.shape[:2]
            x1, y1, x2, y2 = occ.region
            img = frame.image.copy()
            img[int(y1 * h): int(y2 * h), int(x1 * w): int(x2 * w)] = (128, 128, 128)
            gt = [b for b in frame.meta.get("gt_boxes", [])
                  if not (x1 * w <= (b[0] + b[2]) / 2 <= x2 * w and y1 * h <= (b[1] + b[3]) / 2 <= y2 * h)]
            frame = Frame(img, frame.ts, frame.seq, {**frame.meta, "gt_boxes": gt})
        self._last, self._last_delivered = frame, now
        return frame
