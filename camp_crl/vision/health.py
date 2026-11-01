"""Per-camera health: frame rate, staleness, connection state and a confidence score.

Status rules, evaluated at time `now` (same clock as frame timestamps):
  DISCONNECTED  the source reported failure and has not delivered a frame since
  STALE         last frame older than `stale_after_s` (covers frozen streams that still "connect",
                and frames whose timestamp lies in the past, e.g. a buffered replay)
  DEGRADED      fps below `min_fps`, or mean detection score of tracked vehicles below `min_det_score`
  OK            otherwise
Confidence in [0, 1] = freshness x rate x detection quality, where
  freshness = 1 - age / stale_after_s (clipped), rate = min(1, fps / target_fps),
  quality = min(1, mean track score / good_det_score) (1 when nothing is tracked: an empty road is
  not evidence of bad perception).
STALE and DISCONNECTED force confidence to 0.
FPS is an EWMA of 1 / inter-frame interval, so it reflects the frames actually processed.
"""

from __future__ import annotations

from dataclasses import dataclass

from camp_crl.vision.types import Status


@dataclass
class HealthConfig:
    stale_after_s: float = 2.0
    min_fps: float = 3.0
    target_fps: float = 8.0
    min_det_score: float = 0.35
    good_det_score: float = 0.6
    fps_alpha: float = 0.3


class HealthMonitor:
    def __init__(self, cfg: HealthConfig) -> None:
        self.cfg = cfg
        self.fps = 0.0
        self.last_frame_ts: float | None = None
        self.connected = False
        self.reconnects = 0
        self.frames = 0
        self.last_error = ""
        self._mean_score: float | None = None
        self._fps_prev: float | None = None  # previous frame time within the current connection
        self._fps_n = 0                      # frames since the current connection started

    def on_frame(self, ts: float, track_scores: list[float]) -> None:
        # FPS is measured only between frames of the same connection: the gap across an outage is not
        # a frame interval (seen on a real phone: a 6.5 s gap briefly made the camera DEGRADED after reconnect).
        if self._fps_prev is not None and ts > self._fps_prev:
            inst = 1.0 / (ts - self._fps_prev)
            self.fps = inst if self._fps_n <= 1 else self.cfg.fps_alpha * inst + (1 - self.cfg.fps_alpha) * self.fps
        self._fps_prev = ts if self._fps_prev is None else max(ts, self._fps_prev)
        self._fps_n += 1
        self.last_frame_ts = ts if self.last_frame_ts is None else max(ts, self.last_frame_ts)
        self.frames += 1
        self.connected = True
        self._mean_score = sum(track_scores) / len(track_scores) if track_scores else None

    def on_disconnect(self, error: str) -> None:
        if self.connected:
            self.reconnects += 1
        self.connected = False
        self.last_error = error
        self._fps_prev, self._fps_n = None, 0

    def evaluate(self, now: float) -> tuple[Status, float, str, float]:
        """Returns (status, confidence, reason, frame_age_s)."""
        c = self.cfg
        age = float("inf") if self.last_frame_ts is None else max(0.0, now - self.last_frame_ts)
        if not self.connected:
            return Status.DISCONNECTED, 0.0, self.last_error or "no frames received yet", age
        if age > c.stale_after_s:
            return Status.STALE, 0.0, f"last frame {age:.1f}s old (limit {c.stale_after_s}s)", age
        quality = 1.0 if self._mean_score is None else min(1.0, self._mean_score / c.good_det_score)
        conf = max(0.0, 1.0 - age / c.stale_after_s) * min(1.0, self.fps / c.target_fps) * quality
        if self._fps_n >= 2 and self.fps < c.min_fps:
            return Status.DEGRADED, conf, f"low frame rate {self.fps:.1f} fps (min {c.min_fps})", age
        if self._mean_score is not None and self._mean_score < c.min_det_score:
            return Status.DEGRADED, conf, f"low detection confidence {self._mean_score:.2f}", age
        return Status.OK, conf, "", age
