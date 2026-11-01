"""Check one real camera before using it: connect, run detection + tracking, report health, save overlays.

  python -m camp_crl.vision.probe --source http://192.168.1.21:8080/video --seconds 30
  python -m camp_crl.vision.probe --source 0                      # laptop webcam
  python -m camp_crl.vision.probe --source data/clip.mp4 --detector yolox

Runs on the wall clock (time.monotonic), so frame rate, staleness and reconnects behave as they would
live. Prints one status line per second and writes the last annotated frame to runs/vision_probe/.
Pull the phone off Wi-Fi during a run to see DISCONNECTED -> reconnect. Nothing here touches SUMO or
any signal; it only reads the stream.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from camp_crl.sim.network import ROOT
from camp_crl.vision.config import CameraConfig, DetectorConfig, HealthSettings, TrackerConfig
from camp_crl.vision.mapping import ApproachMapper
from camp_crl.vision.pipeline import CameraPipeline
from camp_crl.vision.sources import OpenCVSource
from camp_crl.vision.tracker import ByteTracker


def main() -> None:
    import cv2

    from camp_crl.vision.detector import YoloxDetector

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="http(s)/rtsp URL, webcam index, or video file in the project")
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--model", default="models/yolox_nano.onnx")
    p.add_argument("--max-fps", type=float, default=10.0, help="cap on frames processed per second")
    args = p.parse_args()
    cam = CameraConfig(source=args.source)  # same validation as configs/vision.yaml
    uri = cam.source if "://" in cam.source or cam.source.isdigit() else str(ROOT / cam.source)
    t, h = TrackerConfig(), HealthSettings()
    d = DetectorConfig()
    pipe = CameraPipeline("probe", OpenCVSource(uri), YoloxDetector(ROOT / args.model, d.score_thr, d.nms_thr, threads=d.threads),
                          ApproachMapper(cam.roi, cam.stationary_speed, cam.queue_dwell_s),
                          ByteTracker(t.high_thr, t.low_thr, t.match_iou, t.min_hits, t.max_age_s, t.dist_gate),
                          h.to_health(), min_contrast=h.min_contrast)
    out_dir = ROOT / "runs" / "vision_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    next_print = start
    try:
        while (now := time.monotonic()) - start < args.seconds:
            per = pipe.step(now)
            if now >= next_print:
                ms = "-" if pipe.detect_ms is None else f"{pipe.detect_ms:.0f}"
                print(f"t={now - start:5.1f}s {per.status.value:12} conf={per.confidence:.2f} fps={per.fps:4.1f} "
                      f"detect={ms}ms vehicles={per.vehicle_count} queue={per.queue_veh} "
                      f"arrivals={per.arrivals_total} {per.reason}")
                next_print += 1.0
            time.sleep(max(0.0, 1.0 / args.max_fps - (time.monotonic() - now)))
    finally:
        pipe.close()
    img = pipe.overlay()
    if img is not None:
        path = out_dir / f"probe_{time.strftime('%Y%m%d-%H%M%S')}.jpg"
        cv2.imwrite(str(path), img)
        print(f"overlay saved: {path}")
    for t_ev, msg in pipe.events:
        print(f"  event t={t_ev - start:5.1f}s {msg}")


if __name__ == "__main__":
    main()
