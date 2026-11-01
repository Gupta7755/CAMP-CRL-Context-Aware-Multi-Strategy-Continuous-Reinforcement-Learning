"""Phase 3 perception: detector, tracker, mapping, health, faults, bridge, config and the SUMO demo."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest
from pydantic import ValidationError

from camp_crl.sim.network import ROOT
from camp_crl.vision.bridge import BridgeConfig, CameraDemandBridge
from camp_crl.vision.config import CameraConfig, FaultConfig, TrackerConfig, VisionConfig, load_vision_config
from camp_crl.vision.detector import SyntheticDetector, nms
from camp_crl.vision.health import HealthConfig, HealthMonitor
from camp_crl.vision.mapping import ApproachMapper, point_in_polygon
from camp_crl.vision.pipeline import CameraPipeline
from camp_crl.vision.sources import Fault, FaultInjector, OpenCVSource, SourceError, SyntheticCamera, redact
from camp_crl.vision.tracker import ByteTracker
from camp_crl.vision.types import ApproachPerception, Detection, Status

ROI = [(0.3, 0.0), (0.7, 0.0), (0.7, 0.84), (0.3, 0.84)]
YOLO = ROOT / "models" / "yolox_nano.onnx"
FIXTURES = ROOT / "tests" / "fixtures" / "vision"


def pipeline(faults=(), rate=600.0, seed=1, miss=0.0, **kw) -> tuple[CameraPipeline, SyntheticCamera]:
    cam = SyntheticCamera(rate, seed=seed)
    src = FaultInjector(cam, list(faults)) if faults else cam
    return CameraPipeline("N", src, SyntheticDetector(miss, 0.0, seed), ApproachMapper(ROI), **kw), cam


def run(p: CameraPipeline, t0: float, t1: float, fps: float = 10.0) -> list[ApproachPerception]:
    return [p.step(t0 + k / fps) for k in range(int(round((t1 - t0) * fps)))]


# --- detector -----------------------------------------------------------------------------------

@pytest.mark.skipif(not YOLO.is_file(), reason="models/yolox_nano.onnx not downloaded")
def test_yolox_finds_the_queued_cars_in_a_real_photo():
    import cv2

    from camp_crl.vision.detector import YoloxDetector
    from camp_crl.vision.types import Frame

    img = cv2.imread(str(FIXTURES / "king_st_queue.jpg"))
    dets = YoloxDetector(YOLO).detect(Frame(img, 0.0, 0))
    # The photo shows a taxi, a van and a sedan queued at the kerb (lower-left half of the image).
    # Measured with yolox_nano at 416 px: 3 detections, scores 0.74-0.84.
    assert len(dets) >= 2
    assert all(d.score >= 0.35 and d.cls in (2, 3, 5, 7) for d in dets)
    h, w = img.shape[:2]
    for d in dets:
        x1, y1, x2, y2 = d.box
        assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
        assert y2 > h * 0.5 and x1 < w * 0.65  # on the road at the queue, not in the sky/building


def test_missing_model_gives_clear_error(tmp_path):
    from camp_crl.vision.detector import YoloxDetector

    with pytest.raises(FileNotFoundError, match="YOLOX model not found"):
        YoloxDetector(tmp_path / "absent.onnx")


def test_nms_suppresses_overlaps_and_keeps_separate_boxes():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], float)
    assert nms(boxes, np.array([0.9, 0.8, 0.7]), 0.5) == [0, 2]


def test_synthetic_detector_requires_synthetic_frames():
    from camp_crl.vision.types import Frame

    with pytest.raises(ValueError, match="SyntheticCamera"):
        SyntheticDetector().detect(Frame(np.zeros((10, 10, 3), np.uint8), 0.0, 0))


# --- tracker ------------------------------------------------------------------------------------

def moving_box(t: float, speed: float = 70.0) -> tuple[float, float, float, float]:
    y = 10 + speed * t
    return (100.0, y, 116.0, y + 22.0)


@pytest.mark.parametrize("fps", [5, 10, 30])
def test_moving_vehicle_keeps_one_id_at_phone_frame_rates(fps):
    tr = ByteTracker()
    for k in range(3 * fps):
        tr.update([Detection(moving_box(k / fps), 0.9)], k / fps)
    assert [t.id for t in tr.active()] == [1]
    assert tr.active()[0].vel[1] == pytest.approx(70.0, rel=0.05)


def test_short_occlusion_coasts_long_occlusion_drops():
    tr = ByteTracker(max_age_s=1.5)
    for k in range(10):
        tr.update([Detection(moving_box(k / 10), 0.9)], k / 10)
    tr.update([], 1.5)  # 0.6 s without detections: kept
    assert len(tr.active()) == 1
    tr.update([Detection(moving_box(1.6), 0.9)], 1.6)
    assert [t.id for t in tr.active()] == [1]  # re-matched by prediction, same id
    tr.update([], 3.5)  # 1.9 s gap: deleted
    assert tr.active() == []


def test_low_score_detection_keeps_track_but_does_not_start_one():
    tr = ByteTracker()
    for k in range(5):
        tr.update([Detection(moving_box(k / 10), 0.9)], k / 10)
    tr.update([Detection(moving_box(0.5), 0.3)], 0.5)  # partially occluded: score drops
    assert tr.active()[0].last_ts == 0.5
    fresh = ByteTracker()
    fresh.update([Detection((0, 0, 10, 10), 0.3)], 0.0)
    assert fresh.tracks == []


def test_two_lanes_keep_separate_ids():
    tr = ByteTracker()
    for k in range(20):
        t = k / 10
        a, b = moving_box(t), moving_box(t, 50.0)
        tr.update([Detection(a, 0.9), Detection((a[0] + 40, b[1], a[2] + 40, b[3]), 0.9)], t)
    assert sorted(t.id for t in tr.active()) == [1, 2]


# --- mapping ------------------------------------------------------------------------------------

def test_point_in_polygon():
    assert point_in_polygon(0.5, 0.5, ROI)
    assert not point_in_polygon(0.1, 0.5, ROI)
    assert not point_in_polygon(0.5, 0.9, ROI)


def test_queue_needs_dwell_and_arrivals_count_each_id_once():
    m = ApproachMapper(ROI, stationary_speed=0.02, queue_dwell_s=2.0)
    tr = ByteTracker(min_hits=1)
    box = (150.0, 100.0, 166.0, 122.0)  # stationary, inside ROI of a 320x240 frame
    counts = [m.update(tr.update([Detection(box, 0.9)], k / 10) and tr.active(), k / 10, 320, 240)
              for k in range(30)]
    assert counts[5] == (1, 0, 1)    # 0.5 s still: counted, not yet queued
    assert counts[29] == (1, 1, 1)   # >= 2 s still: queued; still one arrival
    outside = ApproachMapper(ROI)
    t2 = ByteTracker(min_hits=1)
    assert outside.update(t2.update([Detection((10.0, 100.0, 26.0, 122.0), 0.9)], 0.0), 0.0, 320, 240) == (0, 0, 0)


@pytest.mark.parametrize("miss", [0.0, 0.15])
def test_counting_matches_ground_truth_at_10_fps(miss):
    p, cam = pipeline(rate=900, seed=2, miss=miss)
    out = run(p, 0, 300)
    # Vehicles that entered in the last ~0.5 s may not be confirmed yet.
    assert cam.true_arrivals - 2 <= out[-1].arrivals_total <= cam.true_arrivals
    assert out[-1].status is Status.OK


# --- health -------------------------------------------------------------------------------------

def test_health_states_and_confidence():
    h = HealthMonitor(HealthConfig(stale_after_s=2.0, min_fps=6.0, target_fps=10.0))
    assert h.evaluate(0.0)[0] is Status.DISCONNECTED
    for k in range(20):
        h.on_frame(k / 10, [0.8])
    st, conf, _, age = h.evaluate(1.9)
    assert st is Status.OK and h.fps == pytest.approx(10.0) and conf == pytest.approx(1.0) and age == 0.0
    assert h.evaluate(2.4)[1] < conf  # confidence decays with frame age
    st, conf, reason, _ = h.evaluate(4.0)
    assert st is Status.STALE and conf == 0.0 and "old" in reason
    h.on_disconnect("boom")
    assert h.evaluate(4.0)[:3] == (Status.DISCONNECTED, 0.0, "boom") and h.reconnects == 1


def test_low_fps_and_weak_detections_are_degraded():
    h = HealthMonitor(HealthConfig(min_fps=6.0))
    for k in range(10):
        h.on_frame(k / 2, [0.8])  # 2 fps
    assert h.evaluate(4.5)[0] is Status.DEGRADED
    h2 = HealthMonitor(HealthConfig(min_det_score=0.35))
    for k in range(10):
        h2.on_frame(k / 10, [0.2])
    assert h2.evaluate(0.9)[0] is Status.DEGRADED


# --- faults through the full pipeline ------------------------------------------------------------

def test_disconnect_reconnects_with_backoff_and_timestamps_stay_monotonic():
    p, _ = pipeline([Fault("disconnect", 10, 25)], retry_s=1.0, retry_max_s=4.0)
    out = run(p, 0, 40)
    ts = [o.ts for o in out]
    assert ts == sorted(ts)
    status_at = {round(o.ts, 1): o.status for o in out}
    assert status_at[9.9] is Status.OK
    assert status_at[10.5] is Status.DISCONNECTED and status_at[24.5] is Status.DISCONNECTED
    assert status_at[39.9] is Status.OK
    reconnect = next(t for t, msg in p.events if t > 25 and msg.startswith("connected"))
    assert reconnect <= 25 + 4.0 + 0.2  # backoff is capped at retry_max_s
    assert p.health.reconnects == 1


def test_frozen_stream_goes_stale_then_recovers():
    p, _ = pipeline([Fault("stale", 10, 20)])
    out = {round(o.ts, 1): o for o in run(p, 0, 30)}
    assert out[11.0].status is Status.OK          # within stale_after_s
    assert out[15.0].status is Status.STALE and out[15.0].confidence == 0.0
    assert out[15.0].frame_age_s == pytest.approx(5.1, abs=0.15)
    assert out[29.9].status is Status.OK


def test_full_occlusion_is_degraded_and_low_fps_is_degraded():
    p, _ = pipeline([Fault("occlusion", 10, 20)])
    out = {round(o.ts, 1): o for o in run(p, 0, 30)}
    assert out[15.0].status is Status.DEGRADED and "obstructed" in out[15.0].reason
    assert out[29.9].status is Status.OK
    p2, _ = pipeline([Fault("low_fps", 10, 20, fps=2.0)])
    out2 = {round(o.ts, 1): o for o in run(p2, 0, 30)}
    assert out2[15.0].status is Status.DEGRADED and "frame rate" in out2[15.0].reason
    assert out2[29.9].status is Status.OK


def test_partial_occlusion_hides_vehicles_without_breaking_the_stream():
    p, _ = pipeline([Fault("occlusion", 0, 60, region=(0.0, 0.0, 1.0, 0.5))], rate=900)
    q, cam = pipeline(rate=900)
    seen_occ = run(p, 0, 60)[-1].vehicle_count
    assert run(q, 0, 60)[-1].status is Status.OK and p.perception(59.9).status is Status.OK
    assert seen_occ <= q.counts[0]  # known limitation: partial occlusion lowers counts undetected


def test_synthetic_scene_is_deterministic():
    a = [o.to_dict() for o in run(pipeline(seed=5)[0], 0, 60)]
    b = [o.to_dict() for o in run(pipeline(seed=5)[0], 0, 60)]
    c = [o.to_dict() for o in run(pipeline(seed=6)[0], 0, 60)]
    assert a == b and a != c


# --- sources ------------------------------------------------------------------------------------

def test_opencv_source_missing_file_and_video_loop(tmp_path):
    import cv2

    with pytest.raises(SourceError, match="not found"):
        OpenCVSource(str(tmp_path / "none.mp4")).open()
    path = tmp_path / "clip.avi"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for k in range(5):
        vw.write(np.full((48, 64, 3), 40 * k, np.uint8))
    vw.release()
    src = OpenCVSource(str(path))
    src.open()
    frames = [src.read(k / 10) for k in range(12)]
    src.close()
    assert all(f is not None for f in frames)  # paced at the file's 10 fps and looping after 5 frames
    assert [f.ts for f in frames] == [k / 10 for k in range(12)]


def test_redact_hides_stream_credentials():
    assert redact("rtsp://admin:secret@192.168.1.5:554/live") == "rtsp://***@192.168.1.5:554/live"
    assert redact("http://192.168.1.21:8080/video") == "http://192.168.1.21:8080/video"


# --- bridge -------------------------------------------------------------------------------------

def perc(a, total, status=Status.OK, conf=0.9) -> ApproachPerception:
    return ApproachPerception(a, 0.0, 0, 0, total, 10.0, conf, status)


def test_bridge_live_inserts_camera_increments_and_learns_rate():
    b = CameraDemandBridge(BridgeConfig(ewma_tau_s=10.0), (0.2, 0.6, 0.2), seed=1)
    assert b.plan(0.0, {"N": perc("N", 0)})["N"] == 0
    total = 0
    for t in range(1, 121):
        total += 1 if t % 6 == 0 else 0  # 600 veh/h
        assert b.plan(float(t), {"N": perc("N", total)})["N"] == (1 if t % 6 == 0 else 0)
    assert b.mode["N"] == "live" and b.rate["N"] == pytest.approx(600, rel=0.35)


def test_bridge_falls_back_without_backlog_and_recovers():
    b = CameraDemandBridge(BridgeConfig(ewma_tau_s=10.0, min_trusted_s=30.0), (0.2, 0.6, 0.2), seed=1)
    total = 0
    for t in range(0, 60):
        total += 1 if t % 6 == 0 else 0
        b.plan(float(t), {"N": perc("N", total)})
    learned = b.rate["N"]
    fallback = [b.plan(float(t), {"N": perc("N", total + t, Status.STALE, 0.0)})["N"] for t in range(60, 660)]
    assert b.mode["N"] == "fallback" and b.fallback_s["N"] == pytest.approx(600)
    assert sum(fallback) == pytest.approx(learned / 6, rel=0.3)  # Poisson at the trusted rate over 600 s
    # Recovery: the counter moved by 659 during the outage; none of that is dumped into SUMO,
    # only the one vehicle counted since the previous step.
    assert b.plan(660.0, {"N": perc("N", total + 659 + 1)})["N"] == 1
    assert [e["mode"] for e in b.transitions if e["approach"] == "N"] == ["live", "fallback", "live"]


def test_bridge_uses_default_rate_before_trust_and_treats_degraded_as_fallback():
    b = CameraDemandBridge(BridgeConfig(default_veh_per_h={"N": 3600.0, "E": 0.0, "S": 0.0, "W": 0.0}),
                           (0.2, 0.6, 0.2), seed=3)
    b.plan(0.0, {})
    n = sum(b.plan(float(t), {"N": perc("N", 0, Status.DEGRADED, 0.9)})["N"] for t in range(1, 101))
    assert 70 <= n <= 130 and b.mode["N"] == "fallback"  # ~1 veh/s from the default rate
    b2 = CameraDemandBridge(BridgeConfig(min_confidence=0.5), (0.2, 0.6, 0.2), seed=3)
    assert not b2.is_live(perc("N", 0, Status.OK, conf=0.4))


# --- config -------------------------------------------------------------------------------------

def test_default_vision_config_is_valid():
    cfg = load_vision_config(overrides=None)
    assert set(cfg.cameras) == {"N", "E", "S", "W"}
    assert {f.kind for c in cfg.cameras.values() for f in c.faults} == {"disconnect", "low_fps", "occlusion", "stale"}


@pytest.mark.parametrize("source", ["ftp://x/y", "file:///etc/passwd", "../../outside.mp4"])
def test_config_rejects_unsafe_sources(source):
    with pytest.raises(ValidationError):
        CameraConfig(source=source)


@pytest.mark.parametrize("source", ["synthetic", "0", "http://192.168.1.21:8080/video", "rtsp://h/live",
                                    "data/clip.mp4"])
def test_config_accepts_supported_sources(source):
    assert CameraConfig(source=source).source == source


def test_config_rejects_invalid_values():
    with pytest.raises(ValidationError):
        VisionConfig(cameras={"X": CameraConfig()})
    with pytest.raises(ValidationError):
        CameraConfig(roi=[(0, 0), (1.5, 0), (1, 1)])
    with pytest.raises(ValidationError):
        FaultConfig(kind="stale", start_s=10, end_s=5)
    with pytest.raises(ValidationError):
        TrackerConfig(low_thr=0.6, high_thr=0.5)


# --- end-to-end with SUMO -----------------------------------------------------------------------

@pytest.mark.sumo
def test_camera_demo_drives_sumo_repeatably_and_falls_back(tmp_path):
    from camp_crl.vision.demo import run_demo

    vision = tmp_path / "vision.yaml"
    data = load_vision_config(overrides=None).model_dump(mode="json")
    data["cameras"]["N"]["faults"] = [{"kind": "disconnect", "start_s": 100, "end_s": 140}]
    for a in "ESW":
        data["cameras"][a]["faults"] = []
    import yaml

    vision.write_text(yaml.safe_dump(data), encoding="utf-8")
    runs = [run_demo("actuated", 3, 240, True, vision, tmp_path / f"r{k}") for k in range(2)]
    s = runs[0]
    rows = [list(csv.reader((tmp_path / f"r{k}").glob("*/perception.csv").__next__().open())) for k in range(2)]
    assert rows[0] == rows[1]  # same seed -> identical perception, insertions and fallbacks
    assert s["sumo_metrics"]["trips_measured"] > 0 and s["sumo_metrics"]["teleports"] == 0
    n = s["approaches"]["N"]
    # 40 s outage + backoff retries at 101, 103, 107, 115, 131, 161 s (1-2-4-8-16-30 s, capped at retry_max_s)
    assert n["reconnects"] == 1 and 40 <= n["fallback_s"] <= 40 + 30 + 2
    modes = [(e["approach"], e["mode"]) for e in s["transitions"]]
    assert ("N", "fallback") in modes and modes.count(("N", "live")) == 2
    for a in "NESW":
        assert s["approaches"][a]["inserted_into_sumo"] > 0
        assert abs(s["approaches"][a]["live_count_error_pct"]) <= 10
    run_dir = next((tmp_path / "r0").glob("*_camera_demo_s3"))
    routes = (run_dir / "routes.rou.xml").read_text(encoding="utf-8")
    assert "<flow" not in routes and 'id="r_N_T"' in routes  # demand comes only from the cameras
    assert json.loads((run_dir / "vision_summary.json").read_text())["detector"] == "synthetic"


# --- live (wall-clock) camera demand ------------------------------------------------------------

class FakeConn:
    """Records TraCI vehicle insertions."""

    def __init__(self):
        self.added = []
        self.vehicle = self

    def add(self, vid, route, **kw):
        self.added.append((vid, route))


def live_demand(pipes, **kw):
    from camp_crl.vision.session import LiveCameraDemand

    bridge = CameraDemandBridge(BridgeConfig(min_trusted_s=0.0), (0.2, 0.6, 0.2), seed=1)
    return LiveCameraDemand(pipes, bridge, max_fps=20.0, stale_after_s=2.0, **kw)


def test_live_demand_runs_cameras_in_threads_and_feeds_sumo():
    import time

    p, cam = pipeline(rate=3600, seed=4)  # busy road so arrivals happen within the test window
    demand = live_demand({"N": p})
    demand.start()
    try:
        conn, t_end = FakeConn(), time.monotonic() + 4.0
        sim_t = 0.0
        while time.monotonic() < t_end:
            demand.step(conn, sim_t)
            sim_t += 1
            time.sleep(0.25)
        st = demand.status()
        assert st["clock"] == "wall" and st["approaches"]["N"]["status"] == "ok"
        assert st["approaches"]["N"]["fps"] > 10
        assert demand.overlay_jpeg("N")[:2] == b"\xff\xd8"
        assert any(v[1].startswith("r_N_") for v in conn.added)  # camera arrivals became SUMO vehicles
        assert demand.overlay_jpeg("E") is None and "E" not in demand.current()
    finally:
        demand.close()
    assert all(not th.is_alive() for th in demand._threads)


def test_live_demand_marks_silent_camera_stale():
    clock = [100.0]
    demand = live_demand({}, clock=lambda: clock[0])
    demand.latest["N"] = ApproachPerception("N", 99.5, 3, 1, 10, 10.0, 0.9, Status.OK)
    assert demand.current()["N"].status is Status.OK
    clock[0] = 103.0  # thread stuck in a blocking read: no refresh for 3.5 s
    p = demand.current()["N"]
    assert p.status is Status.STALE and p.confidence == 0.0 and "silent" in p.reason


def test_crashing_camera_thread_falls_back_without_stopping_others():
    import time

    class Boom:
        approach, source = "E", type("S", (), {"name": "boom"})()

        def step(self, now):
            raise RuntimeError("driver crashed")

        def overlay(self):
            return None

        def close(self):
            pass

    good, _ = pipeline(seed=2)
    demand = live_demand({"N": good, "E": Boom()})
    demand.start()
    try:
        time.sleep(1.0)
        conn = FakeConn()
        demand.step(conn, 0.0)
        demand.step(conn, 1.0)
        assert "driver crashed" in demand.errors["E"] and "E" not in demand.current()
        assert demand.current()["N"].status is Status.OK and demand.bridge.mode["E"] == "fallback"
    finally:
        demand.close()


def test_fps_restarts_after_reconnect_instead_of_counting_the_outage():
    # Regression from a real phone stream: dropped at 10.9 s, back at 17.4 s, then briefly DEGRADED.
    h = HealthMonitor(HealthConfig(min_fps=6.0))
    for k in range(100):
        h.on_frame(k / 10, [])
    h.on_disconnect("read failed")
    for k in range(3):
        h.on_frame(17.4 + k / 10, [])
        st, *_ = h.evaluate(17.4 + k / 10)
        assert st is Status.OK
    assert h.fps == pytest.approx(10.0) and h.reconnects == 1


def test_raw_frames_at_full_rate_detection_rate_limited():
    p, _ = pipeline(seed=3)
    p.detect_interval_s = 0.1  # 10 Hz detection
    calls = []
    real = p.detector.detect
    p.detector.detect = lambda f: calls.append(f.ts) or real(f)
    for k in range(60):  # 2 s of a 30 fps stream
        p.step(k / 30)
    assert p.raw_frames == 60 and p.raw_fps == pytest.approx(30, rel=0.05)
    assert 19 <= len(calls) <= 21  # ~10 detections per second, not 30
    n, raw = p.live_image(annotate=False)
    n2, annotated = p.live_image(annotate=True)
    assert n == n2 == 60 and raw.shape == annotated.shape and (raw != annotated).any()
    assert raw is p.latest_raw.image  # raw view is the untouched camera frame
