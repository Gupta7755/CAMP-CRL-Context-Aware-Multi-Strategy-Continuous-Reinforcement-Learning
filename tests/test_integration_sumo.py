"""Integration tests that launch real SUMO processes (marker: sumo)."""

import csv
import json
import subprocess
import sys

import pytest

from camp_crl.control.controllers import CONTROLLER_NAMES
from camp_crl.runner import run_episode
from camp_crl.sim.demand import write_routes
from camp_crl.sim.network import build_network
from camp_crl.sim.traci_adapter import SumoSession

pytestmark = pytest.mark.sumo


def sumo_process_count() -> int:
    if sys.platform == "win32":
        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq sumo.exe", "/NH"], capture_output=True, text=True).stdout
        return out.lower().count("sumo.exe")
    out = subprocess.run(["pgrep", "-x", "sumo"], capture_output=True, text=True).stdout
    return len(out.split())


def test_repeated_start_stop_leaves_no_orphans(cfg, tmp_path):
    net = build_network()
    routes = write_routes(cfg.scenarios["balanced"], 60, tmp_path / "r.rou.xml")
    before = sumo_process_count()
    for seed in range(3):
        with SumoSession(net, routes, seed) as conn:
            for _ in range(5):
                conn.simulationStep()
    # Also when the body raises.
    with pytest.raises(RuntimeError):
        with SumoSession(net, routes, 0) as conn:
            raise RuntimeError("boom")
    assert sumo_process_count() == before


def test_missing_inputs_raise_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        SumoSession(tmp_path / "missing.net.xml", tmp_path / "missing.rou.xml", 0)


@pytest.mark.parametrize("controller", CONTROLLER_NAMES)
def test_full_episode_outputs_and_ranges(short_cfg, tmp_path, controller):
    res = run_episode(short_cfg, controller, "balanced", seed=7, output_dir=tmp_path)
    m = res.metrics
    for f in ("config.json", "metrics.json", "steps.csv", "interventions.csv", "tripinfo.xml", "routes.rou.xml"):
        assert (res.run_dir / f).is_file(), f
    # Expected ranges for a 360 s balanced 4x400 veh/h run (sanity bounds, not targets).
    assert m["throughput_completed"] > 50
    assert 0 <= m["delay_mean_s"] < 120
    assert m["phase_switches"] >= 2
    assert m["teleports"] == 0
    # Every baseline already respects min/max green itself, so the filter should not need to intervene.
    assert m["safety_interventions_total"] == 0
    assert json.loads((res.run_dir / "metrics.json").read_text()) == m
    with (res.run_dir / "steps.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == short_cfg.simulation.duration_s
    # Safety invariant on the real trace: never two different greens without clearance in between.
    prev = None
    for r in rows:
        if prev and prev["mode"] == "GREEN" and r["mode"] == "GREEN":
            assert prev["green_phase"] == r["green_phase"]
        prev = r


def test_same_seed_is_reproducible_and_seeds_differ(short_cfg, tmp_path):
    a = run_episode(short_cfg, "camp_rule", "balanced", seed=3, output_dir=tmp_path).metrics
    b = run_episode(short_cfg, "camp_rule", "balanced", seed=3, output_dir=tmp_path).metrics
    c = run_episode(short_cfg, "camp_rule", "balanced", seed=4, output_dir=tmp_path).metrics
    drop = {"wall_time_s"}
    assert {k: v for k, v in a.items() if k not in drop} == {k: v for k, v in b.items() if k not in drop}
    assert a["delay_mean_s"] != c["delay_mean_s"] or a["throughput_completed"] != c["throughput_completed"]


def test_baselines_run_identical_demand(short_cfg, tmp_path):
    runs = [run_episode(short_cfg, c, "balanced", seed=11, output_dir=tmp_path) for c in CONTROLLER_NAMES]
    routes = {(r.run_dir / "routes.rou.xml").read_text() for r in runs}
    assert len(routes) == 1
    assert len({json.loads((r.run_dir / "config.json").read_text())["seed"] for r in runs}) == 1
