"""Validated Phase-3 vision configuration and pipeline construction.

configs/vision.yaml          shared defaults (detector, tracker, health, bridge, demo cameras)
configs/cameras.local.json   camera sources set from the dashboard's Cameras page; merged over the YAML
                             per approach. Machine-specific and may contain stream credentials, so it is
                             git-ignored and never returned unredacted by the API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import json

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from camp_crl.config import APPROACHES
from camp_crl.sim.network import ROOT
from camp_crl.vision.bridge import BridgeConfig
from camp_crl.vision.health import HealthConfig


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DetectorConfig(_Strict):
    # auto: YOLOX for real streams/files/webcams, the synthetic ground-truth detector for synthetic cameras
    kind: Literal["auto", "yolox", "synthetic"] = "auto"
    model: str = "models/yolox_nano.onnx"  # relative to the project root
    score_thr: float = Field(0.2, gt=0, lt=1)  # low: weak boxes only extend tracks (tracker.low_thr)
    nms_thr: float = Field(0.45, gt=0, lt=1)
    threads: int = Field(2, ge=1, le=16)
    miss_prob: float = Field(0.05, ge=0, lt=1)            # synthetic only
    false_pos_per_frame: float = Field(0.02, ge=0, le=5)  # synthetic only


class TrackerConfig(_Strict):
    high_thr: float = Field(0.35, gt=0, lt=1)  # see configs/vision.yaml for the real-footage check
    low_thr: float = Field(0.2, gt=0, lt=1)
    match_iou: float = Field(0.3, gt=0, lt=1)
    min_hits: int = Field(3, ge=1)
    max_age_s: float = Field(1.5, gt=0)
    dist_gate: float = Field(1.5, gt=0)

    @model_validator(mode="after")
    def _order(self) -> "TrackerConfig":
        if self.low_thr >= self.high_thr:
            raise ValueError("tracker.low_thr must be below tracker.high_thr")
        return self


class HealthSettings(_Strict):
    stale_after_s: float = Field(2.0, gt=0)
    min_fps: float = Field(6.0, gt=0)
    target_fps: float = Field(10.0, gt=0)
    min_det_score: float = Field(0.35, ge=0, le=1)
    good_det_score: float = Field(0.6, gt=0, le=1)
    min_contrast: float = Field(6.0, ge=0)

    def to_health(self) -> HealthConfig:
        return HealthConfig(self.stale_after_s, self.min_fps, self.target_fps, self.min_det_score, self.good_det_score)


class BridgeSettings(_Strict):
    min_confidence: float = Field(0.5, ge=0, le=1)
    ewma_tau_s: float = Field(120.0, gt=0)
    min_trusted_s: float = Field(60.0, ge=0)
    default_veh_per_h: dict[str, float] = Field(default_factory=lambda: {a: 400.0 for a in APPROACHES})

    @field_validator("default_veh_per_h")
    @classmethod
    def _all_approaches(cls, v: dict[str, float]) -> dict[str, float]:
        if set(v) != set(APPROACHES) or any(not 0 <= x <= 3600 for x in v.values()):
            raise ValueError(f"default_veh_per_h needs {APPROACHES} with values in [0, 3600]")
        return v

    def to_bridge(self) -> BridgeConfig:
        return BridgeConfig(self.min_confidence, self.ewma_tau_s, self.min_trusted_s, dict(self.default_veh_per_h))


class SyntheticScene(_Strict):
    rate_veh_per_h: float = Field(400.0, ge=0, le=3600)
    cycle_s: float = Field(60.0, gt=0)
    green_s: float = Field(27.0, gt=0)
    offset_s: float = Field(0.0, ge=0)

    @model_validator(mode="after")
    def _green_fits(self) -> "SyntheticScene":
        if self.green_s >= self.cycle_s:
            raise ValueError("synthetic.green_s must be shorter than cycle_s")
        return self


class FaultConfig(_Strict):
    kind: Literal["disconnect", "low_fps", "occlusion", "stale"]
    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)
    fps: float = Field(1.0, gt=0)
    region: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)

    @model_validator(mode="after")
    def _valid(self) -> "FaultConfig":
        if self.end_s <= self.start_s:
            raise ValueError("fault end_s must be after start_s")
        x1, y1, x2, y2 = self.region
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("fault region must be normalised x1 < x2, y1 < y2 within [0, 1]")
        return self


class CameraConfig(_Strict):
    """`source`: "synthetic", a webcam index ("0"), an http(s)/rtsp URL, or a video file under the project."""

    source: str = "synthetic"
    roi: list[tuple[float, float]] = Field(
        default_factory=lambda: [(0.3, 0.0), (0.7, 0.0), (0.7, 0.84), (0.3, 0.84)])
    stationary_speed: float = Field(0.02, gt=0)
    queue_dwell_s: float = Field(2.0, ge=0)
    synthetic: SyntheticScene = SyntheticScene()
    faults: list[FaultConfig] = Field(default_factory=list)

    @field_validator("source")
    @classmethod
    def _source(cls, v: str) -> str:
        if v == "synthetic" or v.isdigit() or v.split("://")[0] in ("http", "https", "rtsp"):
            return v
        if "://" in v:
            raise ValueError(f"unsupported stream scheme in '{v.split('://')[0]}://' (use http, https or rtsp)")
        p = (ROOT / v).resolve()
        if ROOT.resolve() not in p.parents:
            raise ValueError("video file sources must be inside the project directory")
        return v

    @field_validator("roi")
    @classmethod
    def _roi(cls, v: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if len(v) < 3 or any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in v):
            raise ValueError("roi needs >= 3 points with normalised coordinates in [0, 1]")
        return v


class VisionConfig(_Strict):
    fps: float = Field(10.0, gt=0, le=30, description="Frames processed per second in the offline demo.")
    process_every_n: int = Field(1, ge=1)
    retry_s: float = Field(1.0, gt=0)
    retry_max_s: float = Field(30.0, gt=0)
    detector: DetectorConfig = DetectorConfig()
    tracker: TrackerConfig = TrackerConfig()
    health: HealthSettings = HealthSettings()
    bridge: BridgeSettings = BridgeSettings()
    cameras: dict[str, CameraConfig]

    @field_validator("cameras")
    @classmethod
    def _approaches(cls, v: dict[str, CameraConfig]) -> dict[str, CameraConfig]:
        bad = set(v) - set(APPROACHES)
        if bad:
            raise ValueError(f"cameras must be keyed by approaches {APPROACHES} (0 to 4 of them); got {sorted(v)}")
        return v


CAMERA_OVERRIDES = ROOT / "configs" / "cameras.local.json"
NO_CAMERA = "none"


def read_overrides(path: Path | None = CAMERA_OVERRIDES) -> dict[str, str]:
    """{approach: source} saved from the dashboard; source "none" means no camera on that approach."""
    if path is None or not Path(path).is_file():
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {a: str(c["source"]) for a, c in data.get("cameras", {}).items() if a in APPROACHES}


def write_overrides(sources: dict[str, str], path: Path = CAMERA_OVERRIDES) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"cameras": {a: {"source": s} for a, s in sorted(sources.items())}}, indent=2),
                   encoding="utf-8")
    tmp.replace(path)


def load_vision_config(path: str | Path = ROOT / "configs" / "vision.yaml",
                       overrides: Path | None = CAMERA_OVERRIDES) -> VisionConfig:
    """The YAML config with dashboard camera sources merged in. An override keeps the YAML camera's ROI
    and thresholds but replaces its source; scripted faults are kept only for synthetic sources."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"vision config not found: {p.resolve()}")
    cfg = VisionConfig.model_validate(yaml.safe_load(p.read_text(encoding="utf-8")) or {})
    cams = dict(cfg.cameras)
    for a, src in read_overrides(overrides).items():
        if src == NO_CAMERA:
            cams.pop(a, None)
            continue
        base = cams.get(a, CameraConfig())
        cams[a] = CameraConfig.model_validate({**base.model_dump(), "source": src,
                                               "faults": base.model_dump()["faults"] if src == "synthetic" else []})
    return cfg.model_copy(update={"cameras": cams})


def source_kind(source: str) -> str:
    if source == "synthetic":
        return "synthetic"
    return "stream" if "://" in source else "webcam" if source.isdigit() else "file"


def build_pipelines(cfg: VisionConfig, seed: int, with_faults: bool = True) -> dict:
    """One CameraPipeline per configured approach. Synthetic cameras and detectors are seeded per approach."""
    from camp_crl.vision.detector import SyntheticDetector, YoloxDetector
    from camp_crl.vision.mapping import ApproachMapper
    from camp_crl.vision.pipeline import CameraPipeline
    from camp_crl.vision.sources import Fault, FaultInjector, OpenCVSource, SyntheticCamera
    from camp_crl.vision.tracker import ByteTracker

    d = cfg.detector
    needs_yolo = d.kind == "yolox" or (d.kind == "auto" and any(c.source != "synthetic" for c in cfg.cameras.values()))
    if d.kind == "synthetic" and any(c.source != "synthetic" for c in cfg.cameras.values()):
        raise ValueError("detector.kind 'synthetic' only works with synthetic cameras; use 'auto' or 'yolox'")
    shared_yolo = YoloxDetector(ROOT / d.model, d.score_thr, d.nms_thr, threads=d.threads) if needs_yolo else None
    out = {}
    for k, (a, cam) in enumerate(sorted(cfg.cameras.items())):
        if cam.source == "synthetic":
            s = cam.synthetic
            src = SyntheticCamera(s.rate_veh_per_h, seed=seed * 10 + k, cycle_s=s.cycle_s, green_s=s.green_s,
                                  offset_s=s.offset_s, name=f"synthetic:{a}")
        else:
            src = OpenCVSource(cam.source if "://" in cam.source or cam.source.isdigit() else str(ROOT / cam.source))
        if with_faults and cam.faults:
            src = FaultInjector(src, [Fault(f.kind, f.start_s, f.end_s, f.fps, f.region) for f in cam.faults])
        use_yolo = d.kind == "yolox" or (d.kind == "auto" and cam.source != "synthetic")
        det = shared_yolo if use_yolo else SyntheticDetector(d.miss_prob, d.false_pos_per_frame, seed=seed * 10 + k)
        t = cfg.tracker
        out[a] = CameraPipeline(
            a, src, det, ApproachMapper(cam.roi, cam.stationary_speed, cam.queue_dwell_s),
            ByteTracker(t.high_thr, t.low_thr, t.match_iou, t.min_hits, t.max_age_s, t.dist_gate),
            cfg.health.to_health(), cfg.process_every_n, cfg.retry_s, cfg.retry_max_s, cfg.health.min_contrast)
    return out
