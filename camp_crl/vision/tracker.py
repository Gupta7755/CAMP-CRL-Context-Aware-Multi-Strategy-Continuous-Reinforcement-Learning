"""Multi-object tracker: ByteTrack-style two-stage association, simplified.

Per frame:
  1. predict each track's box with a constant-velocity model (no Kalman filter);
  2. match high-score detections (>= high_thr) to tracks by IoU, greedily, best pair first;
  3. match the remaining tracks to low-score detections (low_thr..high_thr), which is what lets
     ByteTrack keep partially occluded vehicles whose score dropped;
  4. match tracks still unmatched to leftover high-score detections by centre distance, gated at
     `dist_gate` x the larger side of the track box. A new track has no velocity estimate yet, so at phone
     frame rates (5-10 fps) a moving car shifts most of a box length between frames and IoU alone
     never links its first frames (ByteTrack's Kalman filter at 30 fps does not face this);
  5. start tracks from unmatched high-score detections; a track is `confirmed` after `min_hits`
     matches and deleted after `max_age_s` without one.
Greedy matching replaces the Hungarian algorithm to avoid a SciPy dependency; with the few
vehicles in one approach view the difference is small, but it is a simplification.
Velocities are pixels/second, measured from consecutive matched box centres.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from camp_crl.vision.types import Detection


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def greedy_match(iou: np.ndarray, thr: float) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    if iou.size == 0:
        return pairs
    iou = iou.copy()
    while True:
        i, j = np.unravel_index(int(iou.argmax()), iou.shape)
        if iou[i, j] < thr:
            return pairs
        pairs.append((int(i), int(j)))
        iou[i, :] = -1
        iou[:, j] = -1


@dataclass
class Track:
    id: int
    box: np.ndarray  # x1, y1, x2, y2
    score: float
    last_ts: float
    first_ts: float
    hits: int = 1
    vel: np.ndarray = None  # type: ignore[assignment]  # centre velocity, px/s
    confirmed: bool = False
    still_since: float | None = None  # first ts of the current stationary spell (set by the mapper)

    def __post_init__(self) -> None:
        if self.vel is None:
            self.vel = np.zeros(2)

    @property
    def center(self) -> tuple[float, float]:
        return float((self.box[0] + self.box[2]) / 2), float((self.box[1] + self.box[3]) / 2)

    def predicted(self, ts: float) -> np.ndarray:
        dt = ts - self.last_ts
        return self.box + np.array([*self.vel, *self.vel]) * dt


class ByteTracker:
    def __init__(self, high_thr: float = 0.35, low_thr: float = 0.2, match_iou: float = 0.3,
                 min_hits: int = 3, max_age_s: float = 1.5, dist_gate: float = 1.5) -> None:
        self.high_thr, self.low_thr, self.match_iou, self.dist_gate = high_thr, low_thr, match_iou, dist_gate
        self.min_hits, self.max_age_s = min_hits, max_age_s
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, dets: list[Detection], ts: float) -> list[Track]:
        """Advance to time `ts`; returns the confirmed tracks matched at this frame."""
        high = [d for d in dets if d.score >= self.high_thr]
        low = [d for d in dets if self.low_thr <= d.score < self.high_thr]
        pred = np.array([t.predicted(ts) for t in self.tracks]).reshape(-1, 4)
        matched_now: set[int] = set()

        def apply(pairs: list[tuple[int, int]], track_idx: list[int], pool: list[Detection]) -> None:
            for ti, di in pairs:
                tr, d = self.tracks[track_idx[ti]], pool[di]
                box = np.array(d.box)
                dt = ts - tr.last_ts
                if dt > 0:
                    c_new = (box[:2] + box[2:]) / 2
                    c_old = (tr.box[:2] + tr.box[2:]) / 2
                    tr.vel = 0.5 * tr.vel + 0.5 * (c_new - c_old) / dt
                tr.box, tr.score, tr.last_ts = box, d.score, ts
                tr.hits += 1
                tr.confirmed = tr.confirmed or tr.hits >= self.min_hits
                matched_now.add(tr.id)

        all_idx = list(range(len(self.tracks)))
        hb = np.array([d.box for d in high]).reshape(-1, 4)
        p1 = greedy_match(iou_matrix(pred, hb), self.match_iou)
        apply(p1, all_idx, high)
        used_high = {j for _, j in p1}
        rest = [i for i in all_idx if self.tracks[i].id not in matched_now]
        lb = np.array([d.box for d in low]).reshape(-1, 4)
        p2 = greedy_match(iou_matrix(pred[rest], lb), self.match_iou) if rest else []
        apply(p2, rest, low)

        rest = [i for i in all_idx if self.tracks[i].id not in matched_now]
        free = [j for j in range(len(high)) if j not in used_high]
        if rest and free:
            pc = (pred[rest, :2] + pred[rest, 2:]) / 2
            dc = (hb[free, :2] + hb[free, 2:]) / 2
            dist = np.linalg.norm(pc[:, None] - dc[None], axis=2)
            size = np.maximum(pred[rest, 3] - pred[rest, 1], pred[rest, 2] - pred[rest, 0])
            gate = self.dist_gate * np.maximum(size, 1.0)[:, None]
            p3 = greedy_match(np.where(dist <= gate, 1.0 / (1.0 + dist), -1.0), 0.0)
            apply([(ti, free[k]) for ti, k in p3], rest, high)
            used_high |= {free[k] for _, k in p3}

        for j, d in enumerate(high):
            if j not in used_high:
                self.tracks.append(Track(self._next_id, np.array(d.box), d.score, ts, ts,
                                         confirmed=self.min_hits <= 1))
                matched_now.add(self._next_id)
                self._next_id += 1
        self.tracks = [t for t in self.tracks if ts - t.last_ts <= self.max_age_s]
        return [t for t in self.tracks if t.confirmed and t.id in matched_now]

    def active(self) -> list[Track]:
        """Confirmed tracks still alive (including ones coasting through a short occlusion)."""
        return [t for t in self.tracks if t.confirmed]
