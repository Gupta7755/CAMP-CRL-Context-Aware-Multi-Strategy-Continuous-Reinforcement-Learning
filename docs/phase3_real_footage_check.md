# Phase 3: detector check on real footage

Date: 2026-10-26. Purpose: replace thresholds that were tuned only on the synthetic camera with ones
checked against real video, and find out where detection stops working. This is a small check, not a
benchmark.

## Footage

- "Intersection Pie-IX-Sherbrooke" by Thomas1313, CC BY-SA 4.0,
  https://commons.wikimedia.org/wiki/File:Intersection_Pie-IX-Sherbrooke.webm
  (1920x1080, 30 fps, 60.6 s; fixed handheld camera at street level, vehicles mostly crossing
  left-right at 20-150 m). Kept locally in `runs/footage/`, not committed.
- Rejected: a Kolkata clip (dashcam, the camera moves) and a Ljubljana clip (edited multi-shot, mostly
  parked vans).

The view does **not** match the intended camera geometry (raised, looking along one approach towards
the stop line), so arrival counting and queue length could not be validated on it. Only detection
was checked.

## Method

- 6 frames at t = 5, 15, 25, 40, 45, 55 s. Every vehicle in the road band (y 380-780) was labelled by
  eye. **The labels were made by the AI coding assistant viewing the frames, not by a person**; check
  them against `runs/footage/` before relying on the numbers.
- Vehicles narrower than about 20 px (on the horizon) were not labelled.
- yolox_nano at its 416x416 input, CPU (onnxruntime 1.31), whole frame.
- Each box was judged as a hit, a duplicate or a false positive by viewing it on the frame.

## Results (yolox_nano, whole 1080p frame)

| Vehicle class (by eye) | Labelled | Detected at score >= 0.35 |
|---|---|---|
| Unoccluded, >= ~100 px wide (>= ~5 % of frame width) | 18 | 18 |
| Unoccluded, < ~100 px wide | 13 | 5 |
| Partly occluded by another vehicle or cones | 16 | 1 |
| **Total** | **47** | **24 (51 %)** |

- Precision at >= 0.35: 24 of 24 boxes were vehicles, with no duplicates.
- Boxes at 0.20-0.35: 7 boxes, of which 5 were missed vehicles and 2 were partial duplicates of
  vehicles already detected.
- 7 of the 24 correct boxes scored 0.35-0.50. With the old tracker setting (a track starts only at
  >= 0.50, chosen on the synthetic detector whose scores are ~0.8), those vehicles could never have
  been tracked or counted.
- yolox_tiny found more boxes (32 vs 24 on the same frames, not individually verified) but took
  10-25x longer per frame on this CPU while RL training was running.
- Running yolox_nano on two overlapping 1000 px tiles of the road band recovered the far bus,
  sedans and SUVs in the frames checked (41 vs 24 boxes), but produced duplicates at the tile seam.
  Not adopted: seam merging would need its own validation.
- Whole clip at 10 fps (606 frames), 960x540 input: 24 ms per frame. Distinct confirmed track IDs:
  50 with the old thresholds, 57 with the new ones; mean tracked vehicles per frame 4.5 -> 6.1. The
  true number of vehicles in the clip was not counted, so ID splits cannot be told apart from extra
  vehicles found.

## Changes made

- `detector.score_thr` 0.35 -> 0.20 (weak boxes reach the tracker),
  `tracker.high_thr` 0.50 -> 0.35 (only boxes >= 0.35 start tracks; 24/24 were real).
  Weak boxes can only extend existing tracks (ByteTrack's second stage), which drops duplicates.
- ROI guidance in `configs/vision.yaml`: end the ROI where vehicles are still >= ~5 % of the frame
  width (~100 px at 1920 px).

## Still needed

Footage from the intended geometry: a phone raised 3-5 m above one approach, looking along it to the
stop line, 5-10 minutes, with a hand count of vehicles crossing the stop line and the queue length
every 10 s. Then measure the arrival and queue error directly, and re-check `min_fps`,
`stationary_speed` and `queue_dwell_s`, which so far were set on the synthetic scene only.
