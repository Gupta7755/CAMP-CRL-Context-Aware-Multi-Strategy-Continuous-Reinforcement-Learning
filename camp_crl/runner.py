"""Run SUMO episodes with a controller behind the safety filter and save raw outputs.

`Episode` advances one control step at a time (used by the live backend); `run_episode`
runs one to completion (used by the CLI, tuning and evaluation).

Each run writes to <output_dir>/<timestamp>_<controller>_<scenario>_s<seed>/:
  config.json        full validated config + scenario + seed + versions
  routes.rou.xml     generated demand
  tripinfo.xml       raw SUMO trip output
  steps.csv          per-step signal state, queues, requested/applied phase, strategy
  interventions.csv  every safety-filter override
  metrics.json       episode summary (see camp_crl.metrics); absent for aborted runs
"""

from __future__ import annotations

import csv
import json
import logging
import platform
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

import numpy as np
import pydantic

from camp_crl import __version__
from camp_crl.camp.features import CampContext, CampEngine
from camp_crl.config import APPROACHES, AppConfig
from camp_crl.control.controllers import CampRuleController, make_controller
from camp_crl.control.safety import SafetyFilter
from camp_crl.metrics import parse_tripinfo, summarize
from camp_crl.sim.demand import write_routes
from camp_crl.sim.network import PHASE_NAMES, ROOT, TLS_ID, build_network, signal_plan
from camp_crl.sim.traci_adapter import SumoSession
from camp_crl.state import SensorDegrader, StateCollector

log = logging.getLogger(__name__)

SPILLBACK_RATIO = 0.9  # queue reaches >= 90 % of the approach length
BLOCKING_RATIO = 0.5  # vehicles fill >= 50 % of a through-exit's storage (see state.downstream_fill_ratio)


class ExternalDemand(Protocol):
    """Inserts vehicles through TraCI each step instead of route-file flows (Phase-3 camera demand)."""

    def step(self, conn, t: float) -> dict: ...


@dataclass
class EpisodeResult:
    run_dir: Path
    metrics: dict


def _versions(sumo_version: str) -> dict:
    return {"camp_crl": __version__, "python": platform.python_version(), "platform": platform.platform(),
            "sumo": sumo_version, "numpy": np.__version__, "pydantic": pydantic.__version__}


class Episode:
    """One SUMO episode, advanced with `step()`. Always call `finish()` or `abort()`."""

    def __init__(self, cfg: AppConfig, controller_name: str, scenario_name: str, seed: int,
                 output_dir: Path | None = None, observe_camp: bool = False,
                 external_demand: ExternalDemand | None = None) -> None:
        if scenario_name not in cfg.scenarios:
            raise ValueError(f"unknown scenario '{scenario_name}'; available: {sorted(cfg.scenarios)}")
        self.cfg = cfg
        self.controller_name = controller_name
        self.scenario_name = scenario_name
        self.scenario = cfg.scenarios[scenario_name]
        self.seed = seed
        self.controller = make_controller(controller_name, cfg)
        # Observer engine so CAMP context is available for explainability with any controller.
        self._observer = CampEngine(cfg.camp) if observe_camp and not isinstance(
            self.controller, CampRuleController) else None
        base = Path(output_dir) if output_dir is not None else ROOT / cfg.output_dir
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.run_dir = base / f"{stamp}_{controller_name}_{scenario_name}_s{seed}"
        self.run_dir.mkdir(parents=True, exist_ok=False)

        sim = cfg.simulation
        net = build_network()
        self.safety = SafetyFilter(signal_plan(net), cfg.timing, initial_phase=0)
        self.external_demand = external_demand
        routes = write_routes(self.scenario, sim.duration_s, self.run_dir / "routes.rou.xml",
                              include_flows=external_demand is None)
        self.tripinfo = self.run_dir / "tripinfo.xml"
        self.session = SumoSession(net, routes, seed, sim.step_length_s, sim.time_to_teleport_s,
                                   self.tripinfo, sim.gui)
        self.conn = self.session.start()
        self.sumo_version = self.conn.getVersion()[1]
        self.collector = StateCollector(self.conn, cfg.camp.momentum.ewma_alpha, cfg.actuated.detection_zone_m)
        self.degrader = SensorDegrader(self.scenario.sensor, seed)
        self.n_steps = int(round(sim.duration_s / sim.step_length_s))
        self.steps_done = 0
        self.step_rows: list[dict] = []
        self.queues: list[float] = []
        self.teleports = 0
        self.arrived = 0
        self.spillback_steps = 0
        self.blocking_steps = 0
        self._bottleneck_on: set[int] = set()
        self._t0 = time.perf_counter()
        self.closed = False

    @property
    def done(self) -> bool:
        return self.steps_done >= self.n_steps

    def _apply_events(self, t: float) -> list[str]:
        notes = []
        for i, ev in enumerate(self.scenario.bottlenecks):
            active = ev.start_s <= t < ev.end_s
            if active and i not in self._bottleneck_on:
                self.conn.edge.setMaxSpeed(ev.edge, ev.speed_mps)
                self._bottleneck_on.add(i)
                notes.append(f"bottleneck on {ev.edge}: {ev.speed_mps} m/s")
            elif not active and i in self._bottleneck_on:
                for k in range(self.conn.edge.getLaneNumber(ev.edge)):
                    self.conn.lane.setMaxSpeed(f"{ev.edge}_{k}", 13.89)
                self._bottleneck_on.discard(i)
                notes.append(f"bottleneck on {ev.edge} cleared")
        return notes

    def step(self) -> dict:
        """Advance one control interval and return a JSON-serialisable snapshot."""
        if self.done or self.closed:
            raise RuntimeError("episode already finished")
        sim = self.cfg.simulation
        truth = self.collector.collect()
        events = self._apply_events(truth.time_s)
        observed = self.degrader.apply(truth)
        decision = self.controller.act(observed, self.safety.status)
        cmd = self.safety.step(decision.requested_phase, sim.step_length_s)
        if cmd.changed:
            self.conn.trafficlight.setRedYellowGreenState(TLS_ID, cmd.state)
        ctx: CampContext | None = None
        if isinstance(self.controller, CampRuleController):
            ctx = self.controller.last_context
        elif self._observer is not None:
            ctx = self._observer.update(observed)

        q = sum(truth.approaches[a].queue_veh for a in APPROACHES)
        measuring = truth.time_s >= sim.warmup_s
        spill = any(truth.approaches[a].spillback_ratio >= SPILLBACK_RATIO for a in APPROACHES)
        block = any(truth.approaches[a].downstream_fill_ratio >= BLOCKING_RATIO for a in APPROACHES)
        if measuring:
            self.queues.append(q)
            self.spillback_steps += spill
            self.blocking_steps += block
        self.step_rows.append({
            "time_s": truth.time_s, "mode": cmd.mode.value, "green_phase": cmd.green_phase,
            "requested_phase": decision.requested_phase, "signal_state": cmd.state,
            **{f"queue_{a}": truth.approaches[a].queue_veh for a in APPROACHES},
            **{f"maxwait_{a}": round(truth.approaches[a].max_wait_s, 1) for a in APPROACHES},
            "strategy": decision.info.get("strategy", ""),
            "stale": ";".join(observed.stale),
            "intervention": ";".join(i.kind for i in cmd.interventions),
        })
        perception = self.external_demand.step(self.conn, truth.time_s) if self.external_demand else None
        self.conn.simulationStep()
        self.teleports += self.conn.simulation.getStartingTeleportNumber()
        self.arrived += self.conn.simulation.getArrivedNumber()
        self.steps_done += 1
        return {
            "time_s": truth.time_s,
            "step": self.steps_done,
            "total_steps": self.n_steps,
            "controller": self.controller_name,
            "scenario": self.scenario_name,
            "seed": self.seed,
            "signal": {"mode": cmd.mode.value, "green_phase": cmd.green_phase,
                       "green_phase_name": PHASE_NAMES[cmd.green_phase], "target_phase": cmd.target_phase,
                       "state": cmd.state, "time_in_mode_s": self.safety.status.time_in_mode_s},
            "requested_phase": decision.requested_phase,
            "decision": {k: v for k, v in decision.info.items() if isinstance(v, (str, int, float, dict))},
            "interventions": [{"kind": i.kind, "detail": i.detail} for i in cmd.interventions],
            "events": events,
            "truth": truth.to_dict(),
            "observed": observed.to_dict(),
            "camp": ctx.to_dict() if ctx is not None else None,
            "perception": perception,
            "kpi": {"total_queue_veh": q, "arrived_total": self.arrived, "teleports": self.teleports,
                    "phase_switches": self.safety.switches, "interventions_total": len(self.safety.interventions),
                    "vehicles_in_network": self.conn.vehicle.getIDCount(),
                    "mean_wait_s": float(np.mean([truth.approaches[a].mean_wait_s for a in APPROACHES])),
                    "spillback": spill, "downstream_blocking": block},
        }

    def _write_common(self) -> None:
        if self.step_rows:
            with (self.run_dir / "steps.csv").open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(self.step_rows[0]))
                w.writeheader()
                w.writerows(self.step_rows)
        with (self.run_dir / "interventions.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["time_s", "kind", "requested", "applied", "detail"])
            w.writerows([i.time_s, i.kind, i.requested, i.applied, i.detail] for i in self.safety.interventions)
        (self.run_dir / "config.json").write_text(json.dumps({
            "controller": self.controller_name, "scenario": self.scenario_name, "seed": self.seed,
            "config": self.cfg.model_dump(mode="json"), "versions": _versions(self.sumo_version)}, indent=2),
            encoding="utf-8")

    def abort(self) -> None:
        """Stop early: close SUMO, keep partial step logs, write no metrics."""
        if self.closed:
            return
        self.session.close()
        self.closed = True
        self._write_common()
        (self.run_dir / "ABORTED").write_text(f"stopped at step {self.steps_done}/{self.n_steps}\n", encoding="utf-8")
        log.info("episode aborted at step %d dir=%s", self.steps_done, self.run_dir)

    def finish(self) -> EpisodeResult:
        if not self.done:
            raise RuntimeError("episode not complete; use abort() to stop early")
        self.session.close()
        self.closed = True
        sim = self.cfg.simulation
        counts = Counter(i.kind for i in self.safety.interventions)
        metrics = summarize(parse_tripinfo(self.tripinfo), self.queues, sim.warmup_s, self.safety.switches,
                            dict(counts), self.teleports)
        metrics.update({
            "spillback_steps": self.spillback_steps, "downstream_blocking_steps": self.blocking_steps,
            "controller": self.controller_name, "scenario": self.scenario_name, "split": self.scenario.split,
            "seed": self.seed, "wall_time_s": round(time.perf_counter() - self._t0, 2)})
        if isinstance(self.controller, CampRuleController):
            metrics["strategy_time_share"] = {
                k: v / len(self.step_rows) for k, v in Counter(r["strategy"] for r in self.step_rows).items()}
        self._write_common()
        (self.run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        log.info("episode done controller=%s scenario=%s seed=%d dir=%s",
                 self.controller_name, self.scenario_name, self.seed, self.run_dir)
        return EpisodeResult(self.run_dir, metrics)


def run_episode(cfg: AppConfig, controller_name: str, scenario_name: str, seed: int,
                output_dir: Path | None = None) -> EpisodeResult:
    ep = Episode(cfg, controller_name, scenario_name, seed, output_dir)
    try:
        while not ep.done:
            ep.step()
        return ep.finish()
    except BaseException:
        ep.abort()
        raise
