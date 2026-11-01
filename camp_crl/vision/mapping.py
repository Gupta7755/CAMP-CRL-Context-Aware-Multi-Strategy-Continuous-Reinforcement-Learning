"""Map tracks in one camera view to approach-level features.

Each camera watches one approach. Its config gives a region of interest (ROI) polygon in
normalised image coordinates (0..1), covering the approach lanes up to the stop line.

  vehicle_count   confirmed tracks whose box bottom-centre (ground contact point) is inside the ROI
  queue_veh       those tracks that have moved less than `stationary_speed` (frame heights / s)
                  continuously for at least `queue_dwell_s`
  arrivals_total  cumulative number of distinct confirmed track ids ever seen inside the ROI;
                  each id is counted once, so a vehicle that coasts through an occlusion and is
                  re-matched is not double counted (a broken track that gets a new id is)
Pixel speeds are divided by frame height so thresholds do not depend on stream resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from camp_crl.vision.tracker import Track


def point_in_polygon(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
    """Ray casting; points exactly on an edge may fall either way."""
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


@dataclass
class ApproachMapper:
    roi: list[tuple[float, float]]  # normalised (x, y)
    stationary_speed: float = 0.02  # frame heights per second
    queue_dwell_s: float = 2.0
    seen_ids: set[int] = field(default_factory=set)

    def ground_point(self, tr: Track, w: int, h: int) -> tuple[float, float]:
        return float((tr.box[0] + tr.box[2]) / 2 / w), float(tr.box[3] / h)

    def in_roi(self, tr: Track, w: int, h: int) -> bool:
        return point_in_polygon(*self.ground_point(tr, w, h), self.roi)

    def update(self, tracks: list[Track], ts: float, w: int, h: int) -> tuple[int, int, int]:
        """Returns (vehicle_count, queue_veh, arrivals_total) for the confirmed `tracks` at time ts."""
        count = queue = 0
        for tr in tracks:
            if not self.in_roi(tr, w, h):
                tr.still_since = None
                continue
            count += 1
            self.seen_ids.add(tr.id)
            if float(np.hypot(*tr.vel)) / h < self.stationary_speed:
                tr.still_since = ts if tr.still_since is None else tr.still_since
                queue += ts - tr.still_since >= self.queue_dwell_s
            else:
                tr.still_since = None
        return count, queue, len(self.seen_ids)
