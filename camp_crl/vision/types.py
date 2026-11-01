"""Plain data types shared by the perception stages.

Coordinates are pixels in the frame that produced them; boxes are (x1, y1, x2, y2).
Timestamps `ts` are seconds on the clock of the pipeline that consumes them (simulation time in
the offline demo, `time.monotonic()` for live streams), so age = now - ts is always well defined.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


@dataclass
class Frame:
    image: np.ndarray  # H x W x 3, BGR uint8
    ts: float
    seq: int
    # Optional extras from the source, e.g. ground-truth boxes for the synthetic camera.
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]
    score: float
    cls: int = 2  # COCO "car"

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


class Status(str, Enum):
    """Health of one camera's perception, from best to worst."""

    OK = "ok"
    DEGRADED = "degraded"        # usable but low FPS / low detection confidence
    STALE = "stale"              # no fresh frame within `stale_after_s`
    DISCONNECTED = "disconnected"  # source closed or failing to (re)open

    @property
    def usable(self) -> bool:
        return self in (Status.OK, Status.DEGRADED)


@dataclass(frozen=True)
class ApproachPerception:
    """Approach-level features derived from one camera at time `ts` (see mapping.py for definitions)."""

    approach: str
    ts: float
    vehicle_count: int
    queue_veh: int
    arrivals_total: int
    fps: float
    confidence: float
    status: Status
    reason: str = ""
    frame_age_s: float = 0.0

    def to_dict(self) -> dict:
        return {"approach": self.approach, "ts": round(self.ts, 3), "vehicle_count": self.vehicle_count,
                "queue_veh": self.queue_veh, "arrivals_total": self.arrivals_total, "fps": round(self.fps, 2),
                "confidence": round(self.confidence, 3), "status": self.status.value, "reason": self.reason,
                "frame_age_s": round(self.frame_age_s, 2)}
