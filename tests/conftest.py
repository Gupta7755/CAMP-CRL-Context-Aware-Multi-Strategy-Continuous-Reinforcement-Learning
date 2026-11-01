from __future__ import annotations

from pathlib import Path

import pytest

from camp_crl.config import AppConfig, load_config
from camp_crl.sim.network import ROOT, SignalPlan, build_network, signal_plan
from camp_crl.state import ApproachState, TrafficState


@pytest.fixture(scope="session")
def cfg() -> AppConfig:
    return load_config(ROOT / "configs" / "default.yaml")


@pytest.fixture(scope="session")
def short_cfg(cfg: AppConfig) -> AppConfig:
    """Short episodes for integration tests (360 s with 60 s warm-up)."""
    data = cfg.model_dump()
    data["simulation"].update(duration_s=360, warmup_s=60)
    data["fixed_time_tuned"] = {"green_s": (25, 25)}  # stand-in plan; real one comes from tune-fixed
    return AppConfig.model_validate(data)


@pytest.fixture(scope="session")
def plan(tmp_path_factory: pytest.TempPathFactory) -> SignalPlan:
    return signal_plan(build_network(Path(tmp_path_factory.mktemp("net")), force=True))


def approach(**kw) -> ApproachState:
    base = dict(vehicle_count=0, queue_veh=0.0, arrival_veh_per_h=0.0, mean_speed_mps=13.89,
                occupancy_pct=0.0, max_wait_s=0.0, mean_wait_s=0.0, spillback_ratio=0.0,
                downstream_occupancy_pct=0.0)
    base.update(kw)
    return ApproachState(**base)


def traffic(t: float, **per_approach: dict) -> TrafficState:
    return TrafficState(t, {a: approach(**per_approach.get(a, {})) for a in "NESW"})
