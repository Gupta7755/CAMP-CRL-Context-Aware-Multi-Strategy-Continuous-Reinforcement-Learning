"""Repeatable Phase-3 demo: cameras -> perception -> SUMO demand -> controller -> SUMO signal.

  python -m camp_crl.vision.demo --controller actuated --seed 1
  python -m camp_crl.vision.demo --duration 600 --no-faults
  python -m camp_crl.vision.demo --check-yolo      # run YOLOX on the fixture photos and report

With the default configs/vision.yaml the four cameras are synthetic (no phones needed) and each one
has a scripted fault, so the run exercises disconnect/reconnect, low FPS, occlusion and a frozen
stream, and shows the bridge falling back and recovering. Writes, next to the usual episode files:
  perception.csv      per step and approach: status, confidence, fps, counts, mode, inserted vehicles
  camera_events.csv   connection changes and live/fallback transitions
  vision_summary.json counting error vs synthetic ground truth, time in fallback, SUMO metrics
  overlays/           annotated camera frames (ROI, tracks, status) every minute and at each transition
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

from camp_crl.config import APPROACHES, Scenario, load_config
from camp_crl.control.controllers import CONTROLLER_NAMES
from camp_crl.runner import Episode
from camp_crl.sim.network import ROOT
from camp_crl.tuning import with_tuned_fixed_time
from camp_crl.vision.bridge import CameraDemandBridge
from camp_crl.vision.config import build_pipelines, load_vision_config
from camp_crl.vision.session import CameraDemand

DEMO_SCENARIO = "camera_demo"


def run_demo(controller: str, seed: int, duration: int | None, faults: bool,
             vision_path: Path = ROOT / "configs" / "vision.yaml", output_dir: Path | None = None,
             snapshot_every_s: float = 60.0, dashboard_cameras: bool = False) -> dict:
    app = with_tuned_fixed_time(load_config(ROOT / "configs" / "default.yaml"))
    # Repeatable by default: only configs/vision.yaml. Camera sources saved on the dashboard's Cameras page
    # (configs/cameras.local.json) are used only when asked for.
    from camp_crl.vision.config import CAMERA_OVERRIDES

    vcfg = load_vision_config(vision_path, overrides=CAMERA_OVERRIDES if dashboard_cameras else None)
    sim = app.simulation.model_copy(update={"duration_s": duration or app.simulation.duration_s,
                                            "warmup_s": min(app.simulation.warmup_s, (duration or 10**9) // 5)})
    # Demand comes from the cameras; the scenario only supplies turn shares and an empty flow set.
    scenario = Scenario(name=DEMO_SCENARIO, split="train", description="camera-driven demand (Phase 3 demo)",
                        demand_veh_per_h={a: 0.0 for a in APPROACHES})
    app = app.model_copy(update={"simulation": sim, "scenarios": {DEMO_SCENARIO: scenario}})
    pipelines = build_pipelines(vcfg, seed, with_faults=faults)
    bridge = CameraDemandBridge(vcfg.bridge.to_bridge(), scenario.turn_shares, seed)
    demand = CameraDemand(pipelines, bridge, vcfg.fps, snapshot_every_s=snapshot_every_s)
    t0 = time.perf_counter()
    ep = Episode(app, controller, DEMO_SCENARIO, seed, output_dir, observe_camp=True, external_demand=demand)
    try:
        while not ep.done:
            ep.step()
        result = ep.finish()
    except BaseException:
        ep.abort()
        raise
    finally:
        demand.close()
    run_dir = result.run_dir
    rows = demand.rows
    demand.write_overlays(run_dir / "overlays")
    with (run_dir / "perception.csv").open("w", newline="", encoding="utf-8") as fh:
        fields = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    events = [{"time_s": round(t, 2), "approach": a, "event": msg}
              for a, p in pipelines.items() for t, msg in p.events]
    events += [{"time_s": e["time_s"], "approach": e["approach"], "event": f"bridge -> {e['mode']} ({e['reason']})"}
               for e in bridge.transitions]
    events.sort(key=lambda e: (e["time_s"], e["approach"]))
    with (run_dir / "camera_events.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["time_s", "approach", "event"])
        w.writeheader()
        w.writerows(events)
    per_approach = {}
    last = {r["approach"]: r for r in rows}
    live_err = live_count_errors(rows)
    for a in APPROACHES:
        r = last.get(a, {})
        true, seen = r.get("true_arrivals"), r.get("arrivals_total")
        per_approach[a] = {
            "camera_source": pipelines[a].source.name if a in pipelines else None,
            "true_arrivals": true, "perceived_arrivals": seen,
            "count_error_pct": None if not true else round(100 * (seen - true) / true, 2),
            "live_count_error_pct": live_err.get(a),
            "inserted_into_sumo": bridge.inserted[a], "fallback_s": round(bridge.fallback_s[a], 1),
            "reconnects": pipelines[a].health.reconnects if a in pipelines else None,
            "statuses_s": {s: sum(1 for x in rows if x["approach"] == a and x.get("status") == s)
                           for s in ("ok", "degraded", "stale", "disconnected")},
        }
    summary = {"controller": controller, "seed": seed, "faults": faults, "duration_s": sim.duration_s,
               "detector": next(iter(pipelines.values())).detector.name,
               "note": "synthetic cameras and detector: perception error is measured against the synthetic "
                       "scene's ground truth, not real footage",
               "approaches": per_approach, "transitions": bridge.transitions, "sumo_metrics": result.metrics,
               "wall_time_s": round(time.perf_counter() - t0, 1), "run_dir": str(run_dir)}
    (run_dir / "vision_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def live_count_errors(rows: list[dict]) -> dict[str, float | None]:
    """Counting error restricted to steps where the bridge used the camera (mode live), in percent.

    Sums per-step increments of the perceived and the true arrival counters over live steps only, so
    arrivals missed while a camera was down (and replaced by fallback demand) do not count as errors.
    """
    out: dict[str, float | None] = {}
    for a in APPROACHES:
        seq = [r for r in rows if r["approach"] == a and "true_arrivals" in r]
        seen = true = 0
        for prev, cur in zip(seq, seq[1:]):
            if cur["mode"] == "live" and prev["mode"] == "live":
                seen += cur["arrivals_total"] - prev["arrivals_total"]
                true += cur["true_arrivals"] - prev["true_arrivals"]
        out[a] = round(100 * (seen - true) / true, 2) if true else None
    return out


def check_yolo(model: str) -> None:
    import cv2

    from camp_crl.vision.detector import YoloxDetector
    from camp_crl.vision.types import Frame

    det = YoloxDetector(ROOT / model)
    for img_path in sorted((ROOT / "tests" / "fixtures" / "vision").glob("*.jpg")):
        img = cv2.imread(str(img_path))
        det.detect(Frame(img, 0.0, 0))  # warm-up
        t = time.perf_counter()
        dets = det.detect(Frame(img, 0.0, 0))
        ms = (time.perf_counter() - t) * 1000
        print(f"{img_path.name:28s} {img.shape[1]}x{img.shape[0]}  {len(dets)} vehicles  {ms:6.1f} ms  "
              f"scores {[round(d.score, 2) for d in dets]}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--controller", default="actuated", choices=CONTROLLER_NAMES)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--duration", type=int, default=None, help="simulated seconds (default: config duration)")
    p.add_argument("--no-faults", action="store_true", help="ignore the scripted camera faults")
    p.add_argument("--vision-config", default=str(ROOT / "configs" / "vision.yaml"))
    p.add_argument("--dashboard-cameras", action="store_true",
                   help="also use camera sources saved on the dashboard's Cameras page (real streams: not repeatable)")
    p.add_argument("--check-yolo", action="store_true")
    p.add_argument("--model", default="models/yolox_nano.onnx")
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    if args.check_yolo:
        check_yolo(args.model)
        return
    s = run_demo(args.controller, args.seed, args.duration, not args.no_faults, Path(args.vision_config),
                 dashboard_cameras=args.dashboard_cameras)
    m = s["sumo_metrics"]
    print(f"controller={s['controller']} seed={s['seed']} faults={s['faults']} detector={s['detector']}")
    print(f"{'appr':5}{'true':>6}{'seen':>6}{'err%':>7}{'live err%':>10}{'inserted':>9}{'fallback s':>11}"
          f"{'reconn':>7}  status seconds")
    fmt = lambda v: "-" if v is None else f"{v:+.1f}"  # noqa: E731
    for a, r in s["approaches"].items():
        print(f"{a:5}{r['true_arrivals'] or 0:>6}{r['perceived_arrivals'] or 0:>6}{fmt(r['count_error_pct']):>7}"
              f"{fmt(r['live_count_error_pct']):>10}{r['inserted_into_sumo']:>9}"
              f"{r['fallback_s']:>11.0f}{r['reconnects']:>7}  {r['statuses_s']}")
    print("transitions:")
    for e in s["transitions"]:
        print(f"  t={e['time_s']:>6.0f} {e['approach']} -> {e['mode']:8} {e['reason']}")
    print(f"SUMO: delay {m['delay_mean_s']:.2f} s/veh, trips {m['trips_measured']}, teleports {m['teleports']}")
    print(f"wall time {s['wall_time_s']} s; saved {s['run_dir']}")


if __name__ == "__main__":
    main()
