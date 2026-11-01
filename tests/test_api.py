"""Backend API tests (launch real SUMO; marker: sumo)."""

import csv
import io
import time

import pytest
import yaml
from fastapi.testclient import TestClient

from camp_crl.api.app import create_app

pytestmark = pytest.mark.sumo


@pytest.fixture()
def client(tmp_path, cfg):
    data = cfg.model_dump(mode="json")
    data["simulation"].update(duration_s=120, warmup_s=20)
    data["scenarios"] = {k: v for k, v in data["scenarios"].items() if k in ("balanced", "ew_dominant_heldout")}
    data["fixed_time_tuned"] = None
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data))
    app = create_app(path, output_root=tmp_path / "out", camera_overrides=tmp_path / "cameras.local.json")
    app.state.camp.cfg = app.state.camp.base_cfg = app.state.camp.base_cfg.model_copy(
        update={"fixed_time_tuned": None})
    with TestClient(app) as c:
        yield c


def wait_for(client, url, cond, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        body = client.get(url).json()
        if cond(body):
            return body
        time.sleep(0.2)
    raise AssertionError(f"timeout waiting on {url}: {body}")


def test_health_and_meta(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["sumo"]["available"] is True
    assert h["camera"]["status"] == "configured" and h["camera"]["demo_runs"] == 0
    names = {c["name"]: c for c in client.get("/api/controllers").json()}
    assert names["fixed_time_tuned"]["available"] is False and names["fixed_time_tuned"]["reason"]
    assert {s["name"] for s in client.get("/api/scenarios").json()} == {"balanced", "ew_dominant_heldout"}


def test_no_data_before_start(client):
    assert client.get("/api/simulation/status").json()["status"] == "idle"
    for url in ("/api/traffic/state", "/api/camp/context", "/api/controller/action"):
        assert client.get(url).status_code == 404


@pytest.mark.parametrize("body", [
    {"controller": "nope", "scenario": "balanced"},
    {"controller": "camp_rule", "scenario": "unknown"},
    {"controller": "camp_rule", "scenario": "balanced", "seed": -1},
    {"controller": "camp_rule", "scenario": "balanced", "speed": 1000},
    {"controller": "fixed_time_tuned", "scenario": "balanced"},
])
def test_start_validation(client, body):
    assert client.post("/api/simulation/start", json=body).status_code == 422


def test_live_run_to_completion_and_endpoints(client):
    r = client.post("/api/simulation/start", json={"controller": "camp_rule", "scenario": "balanced", "speed": 0})
    assert r.status_code == 202
    s = wait_for(client, "/api/simulation/status", lambda b: b["status"] == "finished")
    assert s["metrics"]["controller"] == "camp_rule" and s["step"] == 120
    st = client.get("/api/traffic/state").json()
    assert set(st["truth"]["approaches"]) == {"N", "E", "S", "W"}
    ctx = client.get("/api/camp/context").json()
    assert ctx["used_for_control"] is True and ctx["context"]["pattern"]
    act = client.get("/api/controller/action").json()
    assert act["requested_phase"] in (0, 1) and act["mode"] in ("GREEN", "YELLOW", "ALL_RED")
    assert len(client.get("/api/simulation/history").json()) == 120


def test_duplicate_start_and_stop(client):
    body = {"controller": "max_pressure", "scenario": "balanced", "speed": 20}
    assert client.post("/api/simulation/start", json=body).status_code == 202
    assert client.post("/api/simulation/start", json=body).status_code == 409
    wait_for(client, "/api/simulation/status", lambda b: b["step"] > 2)
    s = client.post("/api/simulation/stop").json()
    assert s["status"] == "stopped"
    # observer CAMP context is available for non-CAMP controllers, flagged as not used for control
    assert client.get("/api/camp/context").json()["used_for_control"] is False
    # can start again after stopping
    assert client.post("/api/simulation/start", json={**body, "speed": 0}).status_code == 202
    wait_for(client, "/api/simulation/status", lambda b: b["status"] == "finished")


def test_websocket_streams_snapshots_and_reconnects(client):
    client.post("/api/simulation/start", json={"controller": "actuated", "scenario": "balanced", "speed": 30})
    for _ in range(2):  # connect, disconnect, reconnect
        with client.websocket_connect("/ws/telemetry") as ws:
            types = set()
            for _ in range(10):
                msg = ws.receive_json()
                types.add(msg["type"])
                if msg["type"] == "snapshot":
                    assert msg["data"]["controller"] == "actuated"
                    assert "q_N" in msg["point"]
                    break
            assert "snapshot" in types
    client.post("/api/simulation/stop")


def test_experiment_run_and_exports_match(client):
    bad = client.post("/api/experiments/run", json={"scenarios": ["ew_dominant_heldout"],
                                                    "controllers": ["fixed_time"], "seeds": [1]})
    assert bad.status_code == 422  # held-out without allow_test
    r = client.post("/api/experiments/run", json={"scenarios": ["balanced"], "controllers": ["fixed_time", "camp_rule"],
                                                  "seeds": [1, 2], "workers": 4})
    assert r.status_code == 202
    exp_id = r.json()["id"]
    m = wait_for(client, f"/api/experiments/{exp_id}", lambda b: b["status"] not in ("queued", "running"), 120)
    assert m["status"] == "completed" and m["completed_runs"] == 4
    res = client.get(f"/api/experiments/{exp_id}/results").json()
    assert len(res["results"]) == 4
    assert res["summary"]["baseline"] == "fixed_time"
    rows = list(csv.DictReader(io.StringIO(client.get(f"/api/experiments/{exp_id}/results?format=csv").text)))
    assert len(rows) == 4
    for row, js in zip(rows, res["results"]):
        assert row["controller"] == js["controller"] and int(row["seed"]) == js["seed"]
        assert float(row["delay_mean_s"]) == pytest.approx(js["delay_mean_s"])
    assert any(e["id"] == exp_id for e in client.get("/api/experiments").json())
    assert client.get("/api/experiments/does-not-exist").status_code == 404
    assert client.get("/api/experiments/..%2F..%2Fetc").status_code == 404


def test_config_validate_update_reset(client):
    cfg = client.get("/api/config").json()["config"]
    cfg["timing"]["min_green_s"] = 99
    v = client.post("/api/config/validate", json={"config": cfg}).json()
    assert v["valid"] is False and v["errors"]
    assert client.put("/api/config", json={"config": cfg}).status_code == 422
    cfg["timing"]["min_green_s"] = 12
    assert client.put("/api/config", json={"config": cfg}).json()["modified"] is True
    assert client.post("/api/config/reset").json()["modified"] is False


def test_logs(client):
    client.post("/api/config/reset")
    logs = client.get("/api/logs/recent?limit=50").json()
    assert any("configuration reset" in r["message"] for r in logs)


def test_redaction():
    from camp_crl.api.logbuffer import redact
    assert redact("connect token=abc123 ok") == "connect token=*** ok"


def test_cancel_only_own_experiments(client):
    r = client.post("/api/experiments/run", json={"scenarios": ["balanced"], "controllers": ["fixed_time"],
                                                  "seeds": [1, 2, 3, 4], "workers": 1})
    exp_id = r.json()["id"]
    m = client.get(f"/api/experiments/{exp_id}").json()
    if m["status"] in ("queued", "running"):
        assert m["cancellable"] is True
        assert client.post(f"/api/experiments/{exp_id}/cancel").status_code == 200
    final = wait_for(client, f"/api/experiments/{exp_id}", lambda b: b["status"] not in ("queued", "running"), 120)
    assert final["status"] in ("cancelled", "completed")
    assert final["cancellable"] is False
    assert client.post(f"/api/experiments/{exp_id}/cancel").status_code == 409


def test_rl_runs_and_curves(client, tmp_path):
    root = client.app.state.camp.output_root / "rl" / "20261025-000000_td3_camp_s1"
    (root / "monitor").mkdir(parents=True)
    (root / "manifest.json").write_text('{"id": "20261025-000000_td3_camp_s1", "algo": "td3", "obs": "camp", '
                                        '"seed": 1, "status": "completed", "created": "2026-10-25T00:00:00"}')
    (root / "monitor" / "env0.monitor.csv").write_text('#{"t_start": 0}\nr,l,t,scenario\n-2.0,100,5.0,balanced\n-1.0,120,9.0,surge_n\n')
    (root / "monitor" / "env1.monitor.csv").write_text('#{"t_start": 0}\nr,l,t,scenario\n-3.0,90,7.0,low_demand\n')
    (root / "validation.csv").write_text('timesteps,mean_reward,mean_delay_s,per_scenario\n300,-2.5,15.4,"[]"\n')
    runs = client.get("/api/rl/runs").json()
    assert runs[0]["id"] == root.name and runs[0]["status"] == "completed"
    c = client.get(f"/api/rl/runs/{root.name}/curves?window=2").json()
    assert [p["timesteps"] for p in c["episodes"]] == [100, 190, 310]  # ordered by wall time across envs
    assert c["episodes"][-1]["reward_avg"] == pytest.approx(-2.0)
    assert c["validation"] == [{"timesteps": 300, "mean_reward": -2.5, "mean_delay_s": 15.4}]
    assert client.get("/api/rl/runs/..%2Fetc/curves").status_code == 404
    assert client.get("/api/rl/runs/nope/curves").status_code == 404


def test_rl_config_endpoint(client):
    body = client.get("/api/rl/config").json()
    assert body["config"]["train_scenarios"] and "Read-only" in body["note"]


def test_vision_endpoints(client):
    from camp_crl.vision.demo import run_demo

    status = client.get("/api/vision/status").json()
    assert [c["approach"] for c in status["cameras"]] == ["E", "N", "S", "W"]
    assert status["latest_demo"] is None and status["packages"]["cv2"]
    assert "never reaches real signal" in status["note"]
    s = run_demo("actuated", 2, 120, True, output_dir=client.app.state.camp.output_root, snapshot_every_s=60)
    run_id = s["run_dir"].replace("\\", "/").split("/")[-1]
    status = client.get("/api/vision/status").json()
    assert status["latest_demo"]["id"] == run_id and set(status["latest_demo"]["approaches"]) == set("NESW")
    assert client.get("/health").json()["camera"]["latest_demo"] == run_id
    tl = client.get(f"/api/vision/demos/{run_id}/timeline?every_s=30").json()
    assert {p["t"] for p in tl["points"]} == {0.0, 30.0, 60.0, 90.0} and len(tl["points"]) == 16
    assert tl["overlays"] and all(n.endswith(".jpg") for n in tl["overlays"])
    img = client.get(f"/api/vision/demos/{run_id}/overlays/{tl['overlays'][0]}")
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg" and img.content[:2] == b"\xff\xd8"
    for bad in (f"/api/vision/demos/{run_id}/overlays/..%2Fconfig.json", f"/api/vision/demos/{run_id}/overlays/x.jpg",
                "/api/vision/demos/..%2F..%2Fetc/timeline", "/api/vision/demos/nope/timeline"):
        assert client.get(bad).status_code == 404, bad


def test_live_run_with_camera_demand(client):
    r = client.post("/api/simulation/start", json={"controller": "actuated", "scenario": "balanced", "seed": 1,
                                                   "speed": 0, "camera_demand": True})
    assert r.status_code == 202 and r.json()["params"]["camera_clock"] == "sim"
    wait_for(client, "/api/simulation/status", lambda b: b["step"] >= 30 or b["status"] in ("finished", "error"))
    live = client.get("/api/vision/live").json()
    assert set(live["approaches"]) == set("NESW") and live["clock"] == "sim"
    assert all(a["mode"] in ("live", "fallback") and a["has_camera"] for a in live["approaches"].values())
    img = client.get("/api/vision/live/N/overlay.jpg")
    assert img.status_code == 200 and img.content[:2] == b"\xff\xd8"
    snap = client.get("/api/traffic/state")
    assert snap.status_code == 200
    body = wait_for(client, "/api/simulation/status", lambda b: b["status"] in ("finished", "error"), timeout=120)
    assert body["status"] == "finished", body
    run_dir = client.app.state.camp.live.run_dir
    assert "<flow" not in open(f"{run_dir}/routes.rou.xml", encoding="utf-8").read()
    assert client.get("/api/vision/live").json()["active"] is False  # finished, status still readable
    assert client.get("/api/vision/live/X/overlay.jpg").status_code == 422


def test_live_camera_demand_rejects_real_streams_off_real_time(client, monkeypatch):
    import camp_crl.vision.config as vc

    real = vc.load_vision_config(overrides=None)
    phone = real.model_copy(update={"cameras": {"N": vc.CameraConfig(source="http://127.0.0.1:9/video")}})
    monkeypatch.setattr(vc, "load_vision_config", lambda *a, **k: phone)
    r = client.post("/api/simulation/start", json={"controller": "actuated", "scenario": "balanced", "seed": 1,
                                                   "speed": 10, "camera_demand": True})
    assert r.status_code == 422 and "speed to 1" in r.json()["detail"]
    assert client.get("/api/simulation/status").json()["status"] == "idle"
    assert client.get("/api/vision/live").json() == {"active": False, "clock": None, "approaches": {}}


def test_camera_setup_from_dashboard(client, tmp_path):
    cams = {c["approach"]: c for c in client.get("/api/cameras").json()["cameras"]}
    assert cams["N"]["kind"] == "synthetic" and cams["N"]["origin"] == "vision.yaml"
    # bad inputs never reach the overrides file
    for bad in ("ftp://x/y", "C:/Windows/win.ini", "../secrets.mp4", "file:///etc/passwd", "42"):
        assert client.put("/api/cameras/N", json={"source": bad}).status_code == 422, bad
    assert client.put("/api/cameras/X", json={"source": "synthetic"}).status_code == 422
    assert not (tmp_path / "cameras.local.json").exists()
    # a phone URL with credentials is stored but only ever returned redacted
    r = client.put("/api/cameras/E", json={"source": "rtsp://admin:pw@192.168.0.110:554/live"})
    assert r.status_code == 200
    e = next(c for c in r.json()["cameras"] if c["approach"] == "E")
    assert e == {**e, "source": "rtsp://***@192.168.0.110:554/live", "kind": "stream", "origin": "dashboard"}
    assert "pw" not in client.get("/api/cameras").text and "pw" not in client.get("/api/vision/status").text
    assert "admin:pw@" in (tmp_path / "cameras.local.json").read_text()
    # 'none' removes the camera; the yaml camera's faults are dropped for a real source
    client.put("/api/cameras/W", json={"source": "none"})
    vcfg = client.app.state.camp.get_vision()
    assert "W" not in vcfg.cameras and vcfg.cameras["E"].faults == [] and vcfg.cameras["N"].faults
    # reset returns to vision.yaml
    client.delete("/api/cameras/E")
    e = next(c for c in client.get("/api/cameras").json()["cameras"] if c["approach"] == "E")
    assert e["origin"] == "vision.yaml" and e["kind"] == "synthetic"


def test_camera_test_and_live_preview(client):
    ok = client.post("/api/cameras/N/test", json={}).json()
    assert ok["ok"] is True and ok["width"] == 320
    dead = client.post("/api/cameras/N/test", json={"source": "http://127.0.0.1:9/video"}).json()
    assert dead["ok"] is False and dead["error"] and dead["ms"] < 15000
    assert client.get("/api/cameras/N/preview.jpg").status_code == 404
    assert client.post("/api/cameras/N/preview/start").json()["running"] is True
    end = time.time() + 10
    while time.time() < end and client.get("/api/cameras/N/preview.jpg").status_code != 200:
        time.sleep(0.2)
    img = client.get("/api/cameras/N/preview.jpg")
    assert img.status_code == 200 and img.content[:2] == b"\xff\xd8"
    status = next(c for c in client.get("/api/cameras").json()["cameras"] if c["approach"] == "N")["preview"]
    assert status["running"] and status["perception"]["status"] in ("ok", "degraded")
    client.post("/api/cameras/N/preview/stop")
    assert client.get("/api/cameras/N/preview.jpg").status_code == 404
    client.put("/api/cameras/S", json={"source": "none"})
    assert client.post("/api/cameras/S/preview/start").status_code == 422


def test_cameras_locked_during_camera_run(client):
    client.post("/api/simulation/start", json={"controller": "actuated", "scenario": "balanced", "seed": 1,
                                               "speed": 2, "camera_demand": True})
    wait_for(client, "/api/simulation/status", lambda b: b["status"] == "running")
    assert client.put("/api/cameras/N", json={"source": "none"}).status_code == 409
    assert client.get("/api/cameras").json()["locked"] is True
    client.post("/api/simulation/stop")
    assert client.put("/api/cameras/N", json={"source": "none"}).status_code == 200


def test_no_cameras_configured_rejects_camera_run(client):
    for a in "NESW":
        client.put(f"/api/cameras/{a}", json={"source": "none"})
    r = client.post("/api/simulation/start", json={"controller": "actuated", "scenario": "balanced", "seed": 1,
                                                   "speed": 0, "camera_demand": True})
    assert r.status_code == 422 and "no cameras configured" in r.json()["detail"]


def test_live_mjpeg_stream_raw_and_annotated(client):
    assert client.get("/api/cameras/N/live.mjpg").status_code == 404  # not started
    client.post("/api/cameras/N/preview/start")
    end = time.time() + 10
    while time.time() < end and client.get("/api/cameras/N/preview.jpg").status_code != 200:
        time.sleep(0.1)
    for annotate in ("true", "false"):
        r = client.get(f"/api/cameras/N/live.mjpg?annotate={annotate}&frames=5")
        assert r.status_code == 200 and r.headers["content-type"].startswith("multipart/x-mixed-replace")
        parts = [p for p in r.content.split(b"--frame\r\n") if p]
        assert len(parts) == 5
        for part in parts:
            head, body = part.split(b"\r\n\r\n", 1)
            assert b"Content-Type: image/jpeg" in head and body[:2] == b"\xff\xd8"
            assert int(head.split(b"Content-Length: ")[1]) == len(body) - 2  # minus trailing CRLF
    status = next(c for c in client.get("/api/cameras").json()["cameras"] if c["approach"] == "N")["preview"]
    assert status["perception"]["stream_fps"] > 15 and status["frames"] >= 5  # read at full rate, not the 10 Hz detection rate
    client.post("/api/cameras/N/preview/stop")
    assert client.get("/api/cameras/N/live.mjpg").status_code == 404


def test_live_advisory_endpoints(client):
    body = client.get("/api/advisory").json()
    assert body["running"] is False and body["advice"] is None and "Nothing is sent" in body["note"]
    assert client.post("/api/advisory/start").status_code == 200
    body = wait_for(client, "/api/advisory", lambda b: b["advice"] is not None, timeout=20)
    adv = body["advice"]
    assert adv["advisory_only"] is True and adv["cameras_working"] >= 0
    assert set(adv["approaches"]) == set("NESW") and adv["plan"][0]["color"] in ("green", "yellow", "red")
    assert all(client.app.state.camp.previews.running(a) for a in "NESW")  # advisory keeps the cameras on
    client.post("/api/advisory/stop")
    assert client.get("/api/advisory").json()["running"] is False
    for a in "NESW":
        client.put(f"/api/cameras/{a}", json={"source": "none"})
    r = client.post("/api/advisory/start")
    assert r.status_code == 422 and "no cameras" in r.json()["detail"]
