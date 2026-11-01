# SignalSense AI: context-aware adaptive traffic signals (CAMP-CRL method)

**SignalSense AI** is the application; **CAMP-CRL** (Context-Aware Multi-Strategy Continuous Reinforcement Learning) is the research method inside it. Phases: simulation foundation, RL, camera perception and live advisory.

Context-Aware Multi-Strategy Continuous Reinforcement Learning for adaptive traffic signal control.
**Research prototype, simulation only.** Never connect this controller to real traffic signals.

| Component | Status |
|---|---|
| Four-arm SUMO intersection (2 lanes/approach, 190 m approaches, 50 km/h) | done |
| 11 scenarios: train (4) / validation (3) / held-out test (4), incl. surges, time-varying dominance, downstream bottlenecks, noisy/missing sensors | done |
| TraCI state collector + sensor degradation model (noise, dropout → stale) | done |
| Fixed-Time (untuned 30/30 and tuned on train), Actuated, Max-Pressure, CAMP rule controllers | done |
| CAMP: pressure, momentum, heuristic risk, fairness, composite urgency | done |
| CAMP explainability: trend, pattern, Poisson-CUSUM surge detection, data-confidence score | done (reported only, not used for control) |
| Rule-based strategy selection with hysteresis | done |
| Independent safety filter, every override logged | done |
| Metrics incl. spillback / blocking, multi-seed evaluation with bootstrap CIs | done |
| FastAPI backend (REST + WebSocket telemetry) | done |
| React dashboard: 8 pages | done |
| Protected left-turn phases, pedestrians | not in scope yet |
| Phase 2: Gymnasium env, reward, DDPG / TD3 training (SB3), CAMP prior, fusion gate | done |
| Phase 2: trained policies + independent evaluation | in progress |
| Phase 3: camera adapters (phone stream, webcam, video file, synthetic), YOLOX detection, ByteTrack-style tracking | done |
| Phase 3: approach features, health/fallback, camera -> SUMO demand, repeatable demo, overlays, System Health panel | done |
| Phase 3: camera-driven demand in live dashboard runs (tick "Camera-driven demand" on Overview) | done |
| Phase 3: tested on a real phone stream; counting validated on real footage | partial (see report) |

## Quick start (Windows, PowerShell)

```powershell
cd C:\Users\rahul\Desktop\pp2
.\start.ps1            # installs deps if needed, builds the dashboard if needed, opens http://127.0.0.1:8000
.\start.ps1 -Test      # run all Python + frontend tests first
.\start.ps1 -Gui       # instead: watch one episode in the SUMO GUI window
```

Requirements: Python 3.14 (3.11+ should work), Node.js 20+ (only to build the dashboard).
`eclipse-sumo` installs SUMO inside the venv, so no system SUMO or `SUMO_HOME` is needed.

### Developing the dashboard

```powershell
.\.venv\Scripts\python -m uvicorn camp_crl.api.app:app --port 8000   # terminal 1
cd frontend; npm run dev                                              # terminal 2 -> http://localhost:5173
```

## Phase 2 (reinforcement learning)

```powershell
python -m pip install -r requirements-rl.txt
python -m camp_crl.rl.train --algo td3 --obs camp --seed 1 --timesteps 60000 --n-envs 2
python -m camp_crl.rl.batch --timesteps 60000 --seeds 1 2 3 --parallel 5 --n-envs 2     # 3 configs x 3 seeds
python -m camp_crl.rl.evaluate --runs runs/rl/<id> ... --baseline-experiment <evaluation id>
```

- Environment (`camp_crl/rl/env.py`): decisions only at green decision points (min green elapsed); action
  `[-1,1]^2` = switch preference + hold length (1–15 s); the Phase-1 safety filter still owns the signal.
  Observations: raw (27) or CAMP (33) features from the possibly-degraded sensor view. Reward from simulator
  ground truth (documented in `configs/rl.yaml`). Time limit -> `truncated`; never `terminated`.
- Training uses only train scenarios; best-model selection only validation scenarios; held-out never.
- Fusion (`camp_crl/rl/fusion.py`): `a_mix = (1-α) a_CAMP + α a_RL`, `α = sigmoid(w·u + b)` with all `w ≤ 0`
  (worse data, surges, novelty or critic disagreement never increase trust in RL). Applied at deployment;
  ablation modes `camp` (α=0), `rl` (α=1), `fixed` (α=0.5), `gated`.

## Command line

```powershell
.\.venv\Scripts\Activate.ps1
python -m camp_crl.cli run --controller camp_rule --scenario surge_n --seed 1 [--gui]
python -m camp_crl.cli tune-fixed --grid 10 12 15 20 25                 # train scenarios only
python -m camp_crl.cli compare --scenario ns_dominant --seeds 1 2 3
python -m camp_crl.cli evaluate --splits train validation               # 10 seeds, parallel
python -m camp_crl.cli evaluate --splits train validation test --allow-test   # final report only
python -m pytest -q                       # Python tests (SUMO integration tests marked `sumo`)
cd frontend; npx vitest run               # frontend tests
```

Held-out (`split: test`) scenarios are refused by `tune-fixed` always, and by `compare`/`evaluate`/the API
unless explicitly allowed. Never tune on held-out results.

## Outputs

Every run writes `runs/<timestamp>_<controller>_<scenario>_s<seed>/`: `config.json` (full config, seed, software
versions), `routes.rou.xml`, `tripinfo.xml` (raw SUMO output), `steps.csv`, `interventions.csv`, `metrics.json`
(`ABORTED` marker instead of metrics if stopped early). Experiments live in `runs/experiments/<id>/` with
`manifest.json`, `results.json`, `results.csv` and `summary.json`.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | backend, SUMO version, camera status |
| GET | `/api/simulation/status` | live run status, staleness |
| POST | `/api/simulation/start` | `{controller, scenario, seed, speed}`; 409 if already running |
| POST | `/api/simulation/stop` | stop and clean up SUMO |
| GET | `/api/simulation/history` | recent per-step points for charts |
| GET | `/api/traffic/state` | ground truth + what the controller observed |
| GET | `/api/camp/context` | CAMP terms, pattern, anomaly, strategy, weights |
| GET | `/api/controller/action` | requested vs applied phase, interventions |
| GET | `/api/scenarios`, `/api/controllers` | metadata (unavailable controllers carry a reason) |
| GET / POST | `/api/experiments`, `/api/experiments/run` | list / start (one at a time) |
| GET | `/api/experiments/{id}`, `/{id}/results?format=json|csv` | manifest / results export |
| POST | `/api/experiments/{id}/cancel` | only experiments started by this backend |
| GET | `/api/evaluation/latest` | latest completed `evaluate` summary |
| GET / PUT / POST | `/api/config`, `/api/config/validate`, `/api/config/reset` | in-memory config (not written to disk) |
| GET | `/api/logs/recent` | redacted recent log records |
| WS | `/ws/telemetry` | per-step snapshots + 1 s heartbeats |

Binds to 127.0.0.1. Inputs are validated against known names; experiment ids are server-generated; no endpoint
accepts file paths or commands.

## Phase 3 (camera perception, drives SUMO only)

**Easiest:** open the dashboard's **Cameras** page, paste each phone's stream URL (e.g. Android IP Webcam:
`http://192.168.0.110:8080/video`) into the direction it watches, press **Test**, then **Start live view** to
see the feed with detected vehicles. Settings are saved in `configs/cameras.local.json` (git-ignored; may
hold stream passwords, which the API never returns). Then tick "Camera-driven demand" on Overview.

```powershell
python -m pip install -r requirements-vision.txt
curl -L -o models/yolox_nano.onnx https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx
python -m camp_crl.vision.demo --controller actuated --seed 1      # 4 synthetic cameras, scripted faults
python -m camp_crl.vision.demo --check-yolo                         # YOLOX on the fixture photos
python -m camp_crl.vision.probe --source http://192.168.1.21:8080/video --seconds 30   # check a phone
```

- Pipeline per camera (`camp_crl/vision/`): source -> detector -> tracker -> approach mapper -> health.
  Sources: http(s)/rtsp phone streams, webcam index, video file in the project, or a synthetic road scene
  that publishes ground truth. Detector: YOLOX nano/tiny ONNX on CPU (Apache-2.0, chosen over AGPL
  Ultralytics), or a synthetic ground-truth detector with seeded misses for the repeatable demo.
- Health: STALE (no new frame for 2 s, incl. frozen streams), DISCONNECTED (reconnect with 1-2-4…30 s
  backoff), DEGRADED (below 6 fps, weak detections, or an obstructed view detected by low image contrast).
- Bridge (`bridge.py`): SUMO is the digital twin. A camera that is OK with confidence ≥ 0.5 inserts one SUMO
  vehicle per counted arrival on its approach; otherwise that approach falls back to Poisson arrivals at the
  last trusted rate (or a default). Counts made during a fault are discarded, not replayed.
- Demo output (`runs/<id>_camera_demo_s<seed>/`): `perception.csv`, `camera_events.csv`, `overlays/`,
  `vision_summary.json` (counting error vs synthetic ground truth, fallback time, SUMO metrics).
- Limits: the synthetic detector is not a vision model; partial occlusion lowers counts without being
  detected; turning movements are not observed (scenario turn shares are used); tracking was tuned on the
  synthetic scene — below ~6 fps it over-counted by up to 14 %. Validate on recorded footage before trusting counts.

## Layout

```
configs/default.yaml            scenarios + all parameters (untuned starting defaults)
configs/tuned/fixed_time.json   tuned Fixed-Time plan with full provenance
sumo/net/                       network source (built by netconvert into sumo/build/)
camp_crl/config.py              Pydantic schema + cross-checks
camp_crl/sim/                   SUMO discovery, network, signal plan, demand, TraCI session
camp_crl/state.py               traffic-state schema, collector, sensor degradation
camp_crl/camp/                  CAMP features + strategy selection
camp_crl/control/               safety filter, controllers
camp_crl/runner.py              step-wise Episode + run_episode
camp_crl/experiments.py         parallel experiments, bootstrap statistics, registry
camp_crl/tuning.py              Fixed-Time grid search (train only)
camp_crl/api/                   FastAPI app, live-simulation manager, log buffer
camp_crl/rl/                    Phase 2: Gymnasium env, training, fusion, evaluation
camp_crl/vision/                Phase 3: sources, detector, tracker, mapping, health, bridge, demo, probe
configs/vision.yaml             cameras, detector, health thresholds, bridge, scripted demo faults
models/                         YOLOX ONNX weights (downloaded, not committed; see models/README.md)
frontend/                       React + TypeScript + Vite + Tailwind + Recharts dashboard
tests/                          Python unit, SUMO integration and API tests
```

## Method notes and assumptions

Equations, glossary, limitations and references are on the dashboard's **Methodology** page. Key points:

- Parts of the specification PDF are cut off at the right margin; where text was missing: R in the pressure
  formula = spillback ratio; strategies = the six visible names.
- SUMO 1.28 `getLastStepOccupancy` returns a fraction in [0, 1] (verified); converted to percent.
- Delay statistics cover completed trips only; always read them with `unfinished`.
- The CUSUM surge detector was calibrated on synthetic Poisson arrivals (~0.7 false alarms per approach-hour,
  median detection ~75 s for a +550 veh/h surge) and spot-checked on SUMO (0 false alarms on balanced demand,
  3/3 surges detected in 42–73 s). It is not an accuracy-validated detector.
- `confidence` is the share of fresh observations — a data-quality score, not a calibrated probability.
- Overflow signal: `downstream_congestion = 1 − exit mean speed / 13.89 m/s`. Exit occupancy (peaked at ~24 %)
  and halted-vehicle counts stayed low during a real spillback because a slowed exit blocks the junction before
  it fills, so neither could trigger OVERFLOW_PREVENTION. Overflow is checked before fairness (a green into a
  blocked exit discharges no one). Both changes were made on the `bottleneck_s` training scenario.
- The bottleneck scenarios originally used 2 m/s, which never produced congestion (exit capacity stayed far above
  demand); they were changed to 0.3 m/s before the final evaluation because they failed to create their stated
  condition. No controller parameter was changed in response to held-out results.
- No claim of novelty or general superiority is made.
