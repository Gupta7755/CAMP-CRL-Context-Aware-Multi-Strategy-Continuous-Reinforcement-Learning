"""FastAPI backend for the CAMP-CRL dashboard. Simulation only.

Run:  python -m uvicorn camp_crl.api.app:app --port 8000
Security: binds to localhost by default; inputs are validated against known controller/scenario
names; experiment ids are generated server-side; no endpoint accepts shell commands or file paths.
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import re
import subprocess
import threading
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from camp_crl import __version__
from camp_crl.api.live import AlreadyRunningError, LiveSimulation
from camp_crl.api.logbuffer import RingBufferHandler
from camp_crl.config import AppConfig, load_config
from camp_crl.control.controllers import CONTROLLER_NAMES
from camp_crl.experiments import ExperimentParams, ExperimentStore, run_experiment
from camp_crl.sim.network import ROOT, SumoNotFoundError, sumo_binary
from camp_crl.tuning import with_tuned_fixed_time

log = logging.getLogger("camp_crl.api")
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"
FRONTEND_DIST = ROOT / "frontend" / "dist"
STALE_AFTER_S = 3.0


@lru_cache(maxsize=1)
def sumo_info() -> dict:
    try:
        exe = sumo_binary("sumo")
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=20).stdout
        first = next((ln for ln in out.splitlines() if ln.strip()), "")
        return {"available": True, "version": first.strip()}
    except (SumoNotFoundError, OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "version": None, "error": str(exc)}


# ---------- request models ----------

class StartRequest(BaseModel):
    controller: Literal[CONTROLLER_NAMES]  # type: ignore[valid-type]
    scenario: str = Field(min_length=1, max_length=64)
    seed: int = Field(1, ge=0, le=1_000_000)
    speed: float = Field(10.0, ge=0, le=200, description="Simulated steps per wall-clock second; 0 = max.")
    camera_demand: bool = Field(False, description="Phase 3: vehicles come from the cameras in configs/vision.yaml.")


Approach = Literal["N", "E", "S", "W"]


class CameraSourceRequest(BaseModel):
    source: str = Field(min_length=1, max_length=500)


class CameraTestRequest(BaseModel):
    source: str | None = Field(None, max_length=500)


class ExperimentRequest(BaseModel):
    scenarios: list[str] = Field(min_length=1, max_length=20)
    controllers: list[Literal[CONTROLLER_NAMES]] = Field(min_length=1)  # type: ignore[valid-type]
    seeds: list[int] = Field(min_length=1, max_length=30)
    baseline: Literal[CONTROLLER_NAMES] | None = None  # type: ignore[valid-type]
    allow_test: bool = False
    workers: int = Field(8, ge=1, le=16)


class ConfigUpdate(BaseModel):
    config: dict


class AppState:
    def __init__(self, cfg_path: Path, output_root: Path | None, camera_overrides: Path | None = None) -> None:
        self.cfg_path = cfg_path
        self.base_cfg = with_tuned_fixed_time(load_config(cfg_path))
        self.cfg = self.base_cfg
        self.cfg_lock = threading.Lock()
        self.output_root = output_root or ROOT / self.cfg.output_dir
        from camp_crl.vision.config import CAMERA_OVERRIDES
        from camp_crl.vision.preview import CameraPreviews

        self.camera_overrides = camera_overrides or CAMERA_OVERRIDES
        self.live = LiveSimulation(self.get_cfg, self.get_vision)
        self.previews = CameraPreviews(self.get_vision)
        from camp_crl.vision.advisory import LiveAdvisory

        self.advisory = LiveAdvisory(self.get_cfg, self.previews, lambda: sorted(self.get_vision().cameras))
        self.store = ExperimentStore(self.output_root)
        self.exp_thread: threading.Thread | None = None
        self.exp_handle = None
        self.logs = RingBufferHandler()
        self.started = time.time()

    def get_cfg(self) -> AppConfig:
        with self.cfg_lock:
            return self.cfg

    def get_vision(self):
        import camp_crl.vision.config as vc  # module attribute looked up at call time (tests patch it)

        return vc.load_vision_config(ROOT / "configs" / "vision.yaml", overrides=self.camera_overrides)


def create_app(cfg_path: Path = DEFAULT_CONFIG, output_root: Path | None = None,
               camera_overrides: Path | None = None) -> FastAPI:
    state = AppState(cfg_path, output_root, camera_overrides)
    logging.getLogger().addHandler(state.logs)
    if logging.getLogger().level > logging.INFO or logging.getLogger().level == logging.NOTSET:
        logging.getLogger().setLevel(logging.INFO)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        log.info("backend started (config=%s)", cfg_path.name)
        yield
        state.live.stop()
        state.advisory.stop()
        state.previews.stop_all()
        if state.exp_handle is not None:
            state.exp_handle.cancel.set()
        logging.getLogger().removeHandler(state.logs)

    app = FastAPI(title="SignalSense AI backend (CAMP-CRL)", version=__version__, lifespan=lifespan)
    app.state.camp = state
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                       allow_methods=["*"], allow_headers=["*"])

    def require_scenario(name: str, allow_test: bool = True) -> None:
        cfg = state.get_cfg()
        if name not in cfg.scenarios:
            raise HTTPException(422, f"unknown scenario '{name}'")
        if not allow_test and cfg.scenarios[name].split == "test":
            raise HTTPException(422, f"scenario '{name}' is held out (test split); set allow_test for final evaluation")

    def require_controller(name: str) -> None:
        if name == "fixed_time_tuned" and state.get_cfg().fixed_time_tuned is None:
            raise HTTPException(422, "fixed_time_tuned has no plan; run `python -m camp_crl.cli tune-fixed`")

    def latest_or_404() -> dict:
        snap = state.live.latest
        if snap is None:
            raise HTTPException(404, "no live simulation data; start a simulation first")
        return snap

    # ---------- health / meta ----------

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__, "uptime_s": round(time.time() - state.started, 1),
                "sumo": sumo_info(), "live_status": state.live.status,
                "camera": camera_health()}

    def camera_health() -> dict:
        """Offline perception only: cameras drive SUMO through the demo CLI; no live stream runs in the backend."""
        path = ROOT / "configs" / "vision.yaml"
        if not path.is_file():
            return {"status": "not_configured", "reason": "configs/vision.yaml missing"}
        runs = _demo_runs()
        live = state.live.camera_demand is not None and state.live.status in ("starting", "running")
        return {"status": "live" if live else "configured",
                "reason": "camera-driven demand active in the live run" if live else
                "start a live run with camera_demand, or run `python -m camp_crl.vision.demo`",
                "demo_runs": len(runs), "latest_demo": runs[0].name if runs else None}

    @app.get("/api/scenarios")
    def scenarios() -> list[dict]:
        cfg = state.get_cfg()
        return [{"name": s.name, "split": s.split, "description": s.description,
                 "demand_veh_per_h": s.demand_veh_per_h, "has_profile": bool(s.profile),
                 "bottlenecks": [b.model_dump() for b in s.bottlenecks], "sensor": s.sensor.model_dump()}
                for s in cfg.scenarios.values()]

    @app.get("/api/controllers")
    def controllers() -> list[dict]:
        cfg = state.get_cfg()
        out = []
        for name in CONTROLLER_NAMES:
            item = {"name": name, "available": True, "reason": None}
            if name == "fixed_time_tuned" and cfg.fixed_time_tuned is None:
                item.update(available=False, reason="No tuned plan yet; run tune-fixed.")
            if name == "fixed_time_tuned" and cfg.fixed_time_tuned is not None:
                item["green_s"] = list(cfg.fixed_time_tuned.green_s)
            if name == "fixed_time":
                item["green_s"] = list(cfg.fixed_time.green_s)
            out.append(item)
        return out

    # ---------- live simulation ----------

    @app.get("/api/simulation/status")
    def sim_status() -> dict:
        s = state.live.snapshot_status()
        s["stale"] = (s["status"] == "running" and s["last_update_age_s"] is not None
                      and s["last_update_age_s"] > STALE_AFTER_S)
        return s

    @app.post("/api/simulation/start", status_code=202)
    def sim_start(req: StartRequest) -> dict:
        require_scenario(req.scenario)
        require_controller(req.controller)
        try:
            return state.live.start(req.controller, req.scenario, req.seed, req.speed, req.camera_demand)
        except AlreadyRunningError as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/simulation/stop")
    def sim_stop() -> dict:
        return state.live.stop()

    @app.get("/api/traffic/state")
    def traffic_state() -> dict:
        snap = latest_or_404()
        return {"time_s": snap["time_s"], "truth": snap["truth"], "observed": snap["observed"], "kpi": snap["kpi"],
                "signal": snap["signal"], "events": snap["events"]}

    @app.get("/api/camp/context")
    def camp_context() -> dict:
        snap = latest_or_404()
        cfg = state.get_cfg()
        return {"time_s": snap["time_s"], "context": snap["camp"],
                "strategy": snap["decision"].get("strategy"), "strategy_reason": snap["decision"].get("strategy_reason"),
                "decision_reason": snap["decision"].get("decision_reason"),
                "controller": snap["controller"],
                "used_for_control": snap["controller"] == "camp_rule",
                "weights": {"pressure": cfg.camp.pressure.model_dump(), "momentum": cfg.camp.momentum.model_dump(),
                            "lambda_momentum": cfg.camp.lambda_momentum, "lambda_risk": cfg.camp.lambda_risk,
                            "lambda_fairness": cfg.camp.lambda_fairness}}

    @app.get("/api/controller/action")
    def controller_action() -> dict:
        snap = latest_or_404()
        sig = snap["signal"]
        return {"time_s": snap["time_s"], "controller": snap["controller"],
                "requested_phase": snap["requested_phase"], "applied_phase": sig["green_phase"],
                "target_phase": sig["target_phase"], "mode": sig["mode"], "signal_state": sig["state"],
                "overridden": bool(snap["interventions"]), "interventions": snap["interventions"],
                "decision": snap["decision"]}

    @app.get("/api/simulation/history")
    def sim_history(limit: int = Query(600, ge=1, le=900)) -> list[dict]:
        return list(state.live.history)[-limit:]

    # ---------- experiments ----------

    def _run_exp(cfg: AppConfig, params: ExperimentParams, handle) -> None:
        try:
            run_experiment(cfg, params, state.store, handle)
        except Exception:
            log.exception("experiment %s failed", handle.id)

    def _annotate(m: dict) -> dict:
        own = state.exp_handle is not None and state.exp_handle.id == m["id"]
        m["cancellable"] = own and m["status"] in ("queued", "running")
        if m["status"] in ("queued", "running") and not own:
            m["note"] = "Started outside this backend (e.g. CLI); stop it where it was started."
        return m

    @app.get("/api/experiments")
    def experiments() -> list[dict]:
        return [_annotate(m) for m in state.store.list()]

    @app.post("/api/experiments/run", status_code=202)
    def experiments_run(req: ExperimentRequest) -> dict:
        if state.exp_thread is not None and state.exp_thread.is_alive():
            raise HTTPException(409, "an experiment is already running")
        for s in req.scenarios:
            require_scenario(s, allow_test=req.allow_test)
        for c in req.controllers:
            require_controller(c)
        if any(not 0 <= s <= 1_000_000 for s in req.seeds) or len(set(req.seeds)) != len(req.seeds):
            raise HTTPException(422, "seeds must be unique integers in [0, 1000000]")
        controllers = list(dict.fromkeys(req.controllers))
        baseline = req.baseline or controllers[0]
        if baseline not in controllers:
            raise HTTPException(422, "baseline must be one of the selected controllers")
        params = ExperimentParams(controllers, list(dict.fromkeys(req.scenarios)), req.seeds, baseline,
                                  kind="comparison", workers=min(req.workers, os.cpu_count() or 1))
        handle = state.store.create(params)
        state.exp_handle = handle
        state.exp_thread = threading.Thread(target=_run_exp, args=(state.get_cfg(), params, handle),
                                            name=f"exp-{handle.id}", daemon=True)
        state.exp_thread.start()
        log.info("experiment %s started: %d runs", handle.id,
                 len(controllers) * len(params.scenarios) * len(req.seeds))
        return state.store.manifest(handle.id)

    def _manifest_or_404(exp_id: str) -> dict:
        try:
            return state.store.manifest(exp_id)
        except KeyError as exc:
            raise HTTPException(404, f"experiment '{exp_id}' not found") from exc

    @app.get("/api/experiments/{exp_id}")
    def experiment(exp_id: str) -> dict:
        return _annotate(_manifest_or_404(exp_id))

    @app.post("/api/experiments/{exp_id}/cancel")
    def experiment_cancel(exp_id: str) -> dict:
        m = _annotate(_manifest_or_404(exp_id))
        if not m["cancellable"]:
            raise HTTPException(409, m.get("note") or f"experiment is {m['status']}; nothing to cancel")
        state.exp_handle.cancel.set()
        log.info("experiment %s cancel requested", exp_id)
        return _annotate(state.store.manifest(exp_id))

    @app.get("/api/experiments/{exp_id}/results")
    def experiment_results(exp_id: str, format: Literal["json", "csv"] = "json"):
        m = _manifest_or_404(exp_id)
        if format == "csv":
            path = state.store.csv_path(exp_id)
            if path is None:
                raise HTTPException(404, "results not available yet")
            return FileResponse(path, media_type="text/csv", filename=f"camp_crl_{exp_id}.csv")
        results = state.store.read(exp_id, "results.json")
        summary = state.store.read(exp_id, "summary.json")
        if results is None:
            raise HTTPException(404, f"results not available yet (status={m['status']})")
        return {"manifest": m, "results": results, "summary": summary}

    @app.get("/api/evaluation/latest")
    def evaluation_latest() -> dict:
        for m in state.store.list():
            if m["params"].get("kind") == "evaluation" and m["status"].startswith("completed"):
                return {"manifest": m, "summary": state.store.read(m["id"], "summary.json")}
        raise HTTPException(404, "no completed evaluation yet; run `python -m camp_crl.cli evaluate`")

    # ---------- RL training runs (Phase 2) ----------

    rl_root = state.output_root / "rl"

    def _rl_run_dir(run_id: str) -> Path:
        if not run_id.replace("-", "").replace("_", "").isalnum():
            raise HTTPException(404, f"run '{run_id}' not found")
        d = rl_root / run_id
        if not (d / "manifest.json").is_file():
            raise HTTPException(404, f"run '{run_id}' not found")
        return d

    @app.get("/api/rl/config")
    def rl_config() -> dict:
        from camp_crl.rl.config import load_rl_config  # lazy: the backend runs without RL extras installed
        path = ROOT / "configs" / "rl.yaml"
        try:
            return {"config": load_rl_config(path).model_dump(mode="json"), "source": path.name,
                    "note": "Read-only here: training is started from the command line (camp_crl.rl.train / batch)."}
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/rl/runs")
    def rl_runs() -> list[dict]:
        if not rl_root.is_dir():
            return []
        out = []
        for d in rl_root.iterdir():
            f = d / "manifest.json"
            if f.is_file():
                m = json.loads(f.read_text(encoding="utf-8"))
                out.append({k: m.get(k) for k in ("id", "algo", "obs", "seed", "timesteps", "n_envs", "status",
                                                  "created", "finished", "wall_time_s", "error")})
        return sorted(out, key=lambda m: m["created"] or "", reverse=True)

    @app.get("/api/rl/runs/{run_id}/curves")
    def rl_curves(run_id: str, window: int = Query(20, ge=1, le=200)) -> dict:
        d = _rl_run_dir(run_id)
        episodes = []
        for f in sorted((d / "monitor").glob("*.monitor.csv")):
            lines = f.read_text(encoding="utf-8").splitlines()[1:]  # skip '#{...}' metadata line
            for row in csv.DictReader(lines):
                try:
                    episodes.append({"r": float(row["r"]), "l": int(row["l"]), "t": float(row["t"]),
                                     "scenario": row.get("scenario", "")})
                except (KeyError, ValueError):
                    continue
        episodes.sort(key=lambda e: e["t"])
        steps, points, recent = 0, [], []
        for i, e in enumerate(episodes):
            steps += e["l"]
            recent = (recent + [e["r"]])[-window:]
            points.append({"episode": i + 1, "timesteps": steps, "reward": round(e["r"], 4),
                           "reward_avg": round(sum(recent) / len(recent), 4), "scenario": e["scenario"]})
        validation = []
        vf = d / "validation.csv"
        if vf.is_file():
            for row in csv.DictReader(vf.read_text(encoding="utf-8").splitlines()):
                validation.append({"timesteps": int(row["timesteps"]), "mean_reward": float(row["mean_reward"]),
                                   "mean_delay_s": float(row["mean_delay_s"]) if row["mean_delay_s"] not in ("", "nan") else None})
        return {"run": run_id, "episodes": points, "validation": validation, "window": window}

    # ---------- camera perception (Phase 3) ----------

    demo_id_re = re.compile(r"^\d{8}-\d{6}-\d{6}_[a-z_]+_camera_demo_s\d+$")
    overlay_re = re.compile(r"^[NESW]_\d{5}\.jpg$")

    def _demo_dir(run_id: str) -> Path:
        d = state.output_root / run_id
        if not demo_id_re.match(run_id) or not (d / "vision_summary.json").is_file():
            raise HTTPException(404, f"camera demo run '{run_id}' not found")
        return d

    def _demo_runs() -> list[Path]:
        if not state.output_root.is_dir():
            return []
        runs = [d for d in state.output_root.iterdir() if demo_id_re.match(d.name) and (d / "vision_summary.json").is_file()]
        return sorted(runs, key=lambda d: d.name, reverse=True)

    @app.get("/api/vision/status")
    def vision_status() -> dict:
        """Camera configuration (credentials redacted), detector availability and the latest demo run."""
        from camp_crl.vision.sources import redact

        try:
            vcfg = state.get_vision()
        except (FileNotFoundError, ValidationError) as exc:
            raise HTTPException(422, f"configs/vision.yaml: {exc}") from exc
        packages = {}
        for mod in ("cv2", "onnxruntime"):
            try:
                packages[mod] = __import__(mod).__version__
            except ImportError:
                packages[mod] = None
        model = ROOT / vcfg.detector.model
        runs = _demo_runs()
        latest = None
        if runs:
            s = json.loads((runs[0] / "vision_summary.json").read_text(encoding="utf-8"))
            latest = {"id": runs[0].name, **{k: s.get(k) for k in ("controller", "seed", "faults", "duration_s",
                                                                    "detector", "note", "approaches", "transitions",
                                                                    "wall_time_s")},
                      "delay_mean_s": s["sumo_metrics"].get("delay_mean_s")}
        return {
            "cameras": [{"approach": a, "source": redact(c.source), "kind": "synthetic" if c.source == "synthetic"
                         else "stream" if "://" in c.source else "webcam" if c.source.isdigit() else "file",
                         "faults": [f.model_dump() for f in c.faults]} for a, c in sorted(vcfg.cameras.items())],
            "detector": {"kind": vcfg.detector.kind, "model": vcfg.detector.model, "model_present": model.is_file(),
                         "license": "YOLOX (Megvii), Apache-2.0" if vcfg.detector.kind == "yolox"
                         else "synthetic ground-truth detector (not a vision model)"},
            "health": vcfg.health.model_dump(), "bridge": vcfg.bridge.model_dump(), "fps": vcfg.fps,
            "packages": packages, "latest_demo": latest, "demo_runs": len(runs),
            "how_to_run": "python -m camp_crl.vision.demo --controller actuated --seed 1",
            "note": "Camera data only drives the SUMO simulation; it never reaches real signal infrastructure.",
        }

    @app.get("/api/vision/live")
    def vision_live() -> dict:
        """Per-approach camera status of the current (or last) live run that used camera demand."""
        demand = state.live.camera_demand
        params = state.live.params or {}
        if demand is None or not params.get("camera_demand"):
            return {"active": False, "clock": None, "approaches": {}}
        return {"active": state.live.status in ("starting", "running"), "sim_status": state.live.status,
                **demand.status()}

    @app.get("/api/vision/live/{approach}/overlay.jpg")
    def vision_live_overlay(approach: Approach) -> Response:
        demand = state.live.camera_demand
        jpg = demand.overlay_jpeg(approach) if demand is not None else None
        if jpg is None:
            raise HTTPException(404, "no camera frame for this approach yet")
        return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/vision/demos/{run_id}/timeline")
    def vision_timeline(run_id: str, every_s: int = Query(10, ge=1, le=300)) -> dict:
        d = _demo_dir(run_id)
        points: list[dict] = []
        with (d / "perception.csv").open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                t = float(row["time_s"])
                if int(t) % every_s:
                    continue
                points.append({"t": t, "approach": row["approach"], "status": row.get("status", ""),
                               "mode": row["mode"], "confidence": float(row.get("confidence") or 0),
                               "fps": float(row.get("fps") or 0), "vehicle_count": int(row.get("vehicle_count") or 0),
                               "queue_veh": int(row.get("queue_veh") or 0)})
        events = list(csv.DictReader((d / "camera_events.csv").read_text(encoding="utf-8").splitlines()))
        overlays = sorted(p.name for p in (d / "overlays").glob("*.jpg")) if (d / "overlays").is_dir() else []
        return {"run": run_id, "every_s": every_s, "points": points, "events": events, "overlays": overlays}

    @app.get("/api/vision/demos/{run_id}/overlays/{name}")
    def vision_overlay(run_id: str, name: str) -> FileResponse:
        d = _demo_dir(run_id)
        f = d / "overlays" / name
        if not overlay_re.match(name) or not f.is_file():
            raise HTTPException(404, "overlay not found")
        return FileResponse(f, media_type="image/jpeg")

    # ---------- camera setup from the dashboard (Phase 3) ----------

    def _cameras_locked() -> None:
        if state.live.camera_demand is not None and state.live.status in ("starting", "running", "stopping"):
            raise HTTPException(409, "a live run is using the cameras; stop it before changing camera sources")

    @app.get("/api/cameras")
    def cameras() -> dict:
        from camp_crl.vision.config import NO_CAMERA, read_overrides, source_kind
        from camp_crl.vision.sources import redact

        try:
            vcfg = state.get_vision()
        except (FileNotFoundError, ValidationError, ValueError) as exc:
            raise HTTPException(422, f"camera configuration invalid: {exc}") from exc
        overrides = read_overrides(state.camera_overrides)
        previews = state.previews.status()
        out = []
        for a in ("N", "E", "S", "W"):
            cam = vcfg.cameras.get(a)
            out.append({"approach": a, "source": redact(cam.source) if cam else None,
                        "kind": source_kind(cam.source) if cam else NO_CAMERA,
                        "origin": "dashboard" if a in overrides else "vision.yaml",
                        "preview": previews.get(a)})
        model = ROOT / vcfg.detector.model
        return {"cameras": out, "detector": {"kind": vcfg.detector.kind, "model": vcfg.detector.model,
                                             "model_present": model.is_file()},
                "locked": state.live.camera_demand is not None and state.live.status in ("starting", "running"),
                "note": "Camera data only drives the SUMO simulation; it never reaches real signal infrastructure."}

    def _check_ui_source(source: str) -> str:
        from camp_crl.vision.config import NO_CAMERA, CameraConfig

        source = source.strip()
        if source in (NO_CAMERA, "synthetic") or (source.isdigit() and int(source) < 10):
            return source
        if source.split("://")[0].lower() not in ("http", "https", "rtsp"):
            raise HTTPException(422, "enter a stream URL starting with http://, https:// or rtsp:// "
                                     "(e.g. http://192.168.0.110:8080/video), a webcam number 0-9, 'synthetic' or 'none'")
        try:
            return CameraConfig(source=source).source
        except ValidationError as exc:
            raise HTTPException(422, str(exc.errors()[0]["msg"])) from exc

    @app.put("/api/cameras/{approach}")
    def set_camera(approach: Approach, req: CameraSourceRequest) -> dict:
        from camp_crl.vision.config import read_overrides, write_overrides

        _cameras_locked()
        source = _check_ui_source(req.source)
        sources = read_overrides(state.camera_overrides)
        sources[approach] = source
        write_overrides(sources, state.camera_overrides)
        state.previews.stop(approach)
        log.info("camera %s set from dashboard", approach)
        return cameras()

    @app.delete("/api/cameras/{approach}")
    def reset_camera(approach: Approach) -> dict:
        from camp_crl.vision.config import read_overrides, write_overrides

        _cameras_locked()
        sources = read_overrides(state.camera_overrides)
        sources.pop(approach, None)
        write_overrides(sources, state.camera_overrides)
        state.previews.stop(approach)
        return cameras()

    @app.post("/api/cameras/{approach}/test")
    def test_camera(approach: Approach, req: CameraTestRequest | None = None) -> dict:
        """Connect once and read a frame (up to ~5 s). Tests `source` if given, else the saved one."""
        from camp_crl.vision.preview import probe_source
        from camp_crl.vision.sources import redact

        if req is not None and req.source:
            source = _check_ui_source(req.source)
        else:
            cam = state.get_vision().cameras.get(approach)
            if cam is None:
                raise HTTPException(404, f"no camera configured for {approach}")
            source = cam.source
        if source == "none":
            raise HTTPException(422, "'none' means no camera; nothing to test")
        if source != "synthetic" and "://" not in source and not source.isdigit():
            source = str(ROOT / source)  # video file configured in vision.yaml
        return {"approach": approach, "source": redact(source), **probe_source(source)}

    @app.post("/api/cameras/{approach}/preview/start")
    def preview_start(approach: Approach) -> dict:
        try:
            return state.previews.start(approach)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/cameras/{approach}/preview/stop")
    def preview_stop(approach: Approach) -> dict:
        state.previews.stop(approach)
        return {"approach": approach, "running": False}

    # ---------- live signal-timing advisory (Phase 3, advisory only) ----------

    @app.post("/api/advisory/start")
    def advisory_start() -> dict:
        """Start recommending signal timings from the configured cameras (starts their live views)."""
        try:
            if not state.get_vision().cameras:
                raise ValueError("no cameras configured: add at least one on the Cameras page")
            state.advisory.start()
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(422, str(exc)) from exc
        return state.advisory.status()

    @app.post("/api/advisory/stop")
    def advisory_stop() -> dict:
        state.advisory.stop()
        return state.advisory.status()

    @app.get("/api/advisory")
    def advisory() -> dict:
        return {**state.advisory.status(),
                "note": "Advisory only: these timings are recommendations computed from the camera feeds. "
                        "Nothing is sent to any traffic signal."}

    @app.get("/api/cameras/{approach}/live.mjpg")
    def preview_stream(approach: Approach, annotate: bool = True,
                       frames: int | None = Query(None, ge=1, le=10_000)) -> StreamingResponse:
        """Live MJPEG video of a started preview at the camera's own frame rate. `annotate=false` gives the
        raw camera picture; otherwise the latest detections (ROI, boxes, counts) are drawn on each frame.
        `frames` ends the stream after that many frames (snapshots, tests)."""
        if not state.previews.running(approach):
            raise HTTPException(404, "live view not started for this approach")
        return StreamingResponse(state.previews.frames(approach, annotate, frames),
                                 media_type="multipart/x-mixed-replace; boundary=frame",
                                 headers={"Cache-Control": "no-store"})

    @app.get("/api/cameras/{approach}/preview.jpg")
    def preview_jpeg(approach: Approach, annotate: bool = True) -> Response:
        jpg = state.previews.jpeg(approach, annotate)
        if jpg is None:
            raise HTTPException(404, "no preview frame yet (start the preview, or the camera is still connecting)")
        return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    # ---------- config ----------

    @app.get("/api/config")
    def get_config() -> dict:
        return {"config": state.get_cfg().model_dump(mode="json"), "source": cfg_path.name,
                "modified": state.cfg is not state.base_cfg,
                "note": "Changes apply to new runs in this backend session and are not written to disk."}

    @app.post("/api/config/validate")
    def validate_config(body: ConfigUpdate) -> dict:
        try:
            AppConfig.model_validate(body.config)
            return {"valid": True, "errors": []}
        except ValidationError as exc:
            return {"valid": False, "errors": [{"loc": ".".join(map(str, e["loc"])), "msg": e["msg"]}
                                               for e in exc.errors()]}

    @app.put("/api/config")
    def put_config(body: ConfigUpdate) -> dict:
        if state.live.status in ("starting", "running", "stopping"):
            raise HTTPException(409, "stop the live simulation before changing configuration")
        try:
            new = AppConfig.model_validate(body.config)
        except ValidationError as exc:
            return JSONResponse(status_code=422, content={"valid": False, "errors": [
                {"loc": ".".join(map(str, e["loc"])), "msg": e["msg"]} for e in exc.errors()]})
        with state.cfg_lock:
            state.cfg = new
        log.info("configuration updated in memory")
        return get_config()

    @app.post("/api/config/reset")
    def reset_config() -> dict:
        with state.cfg_lock:
            state.cfg = state.base_cfg
        log.info("configuration reset to %s", cfg_path.name)
        return get_config()

    # ---------- logs ----------

    @app.get("/api/logs/recent")
    def logs_recent(limit: int = Query(200, ge=1, le=1000),
                    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO") -> list[dict]:
        return state.logs.recent(limit, level)

    # ---------- websocket ----------

    @app.websocket("/ws/telemetry")
    async def telemetry(ws: WebSocket) -> None:
        await ws.accept()
        last_seq = -1
        last_beat = 0.0
        try:
            while True:
                live = state.live
                if live.seq != last_seq and live.latest is not None:
                    last_seq = live.seq
                    snap = live.latest
                    await ws.send_json({"type": "snapshot", "status": live.snapshot_status(), "data": snap,
                                        "point": live.history[-1] if live.history else None})
                now = time.time()
                if now - last_beat >= 1.0:
                    last_beat = now
                    await ws.send_json({"type": "heartbeat", "status": live.snapshot_status(),
                                        "server_time": now})
                await asyncio.sleep(0.1)
        except (WebSocketDisconnect, RuntimeError):
            return

    # ---------- built frontend (optional) ----------

    if FRONTEND_DIST.is_dir():
        app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            if path.startswith(("api/", "ws/")) or path == "api":
                raise HTTPException(404, "not found")  # unknown API paths never fall through to the UI
            f = (FRONTEND_DIST / path).resolve()
            if path and f.is_file() and FRONTEND_DIST.resolve() in f.parents:
                return FileResponse(f)
            return FileResponse(FRONTEND_DIST / "index.html")

    return app


app = create_app()
