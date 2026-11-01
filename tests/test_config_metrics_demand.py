import math
import xml.etree.ElementTree as ET

import pytest
from pydantic import ValidationError

from camp_crl.config import AppConfig, PressureWeights, Scenario, TimingConfig, load_config
from camp_crl.control.controllers import FixedTimeController
from camp_crl.control.safety import Mode, SignalStatus
from camp_crl.metrics import Trip, improvement_pct, mean_std, summarize
from camp_crl.sim.demand import write_routes
from tests.conftest import traffic


class TestConfig:
    def test_default_loads(self, cfg):
        assert cfg.timing.min_green_s <= cfg.timing.max_green_s
        assert {s.split for s in cfg.scenarios.values()} == {"train", "validation", "test"}

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nope.yaml")

    @pytest.mark.parametrize("kw", [dict(min_green_s=70, max_green_s=60), dict(yellow_s=2),
                                    dict(max_red_s=30), dict(min_green_s=-1)])
    def test_bad_timing(self, kw):
        with pytest.raises(ValidationError):
            TimingConfig(**kw)

    def test_pressure_weights_must_sum_to_one(self):
        with pytest.raises(ValidationError):
            PressureWeights(queue=0.9)

    def test_unknown_keys_rejected(self, cfg):
        data = cfg.model_dump()
        data["timing"]["min_gren_s"] = 5
        with pytest.raises(ValidationError):
            AppConfig.model_validate(data)

    def test_fixed_time_must_respect_limits(self, cfg):
        data = cfg.model_dump()
        data["fixed_time"]["green_s"] = (5, 30)
        with pytest.raises(ValidationError):
            AppConfig.model_validate(data)

    def test_scenario_validation(self):
        with pytest.raises(ValidationError):
            Scenario(name="x", demand_veh_per_h={"N": 1, "E": 1, "S": 1})
        with pytest.raises(ValidationError):
            Scenario(name="x", demand_veh_per_h={a: 100 for a in "NESW"}, turn_shares=(0.5, 0.5, 0.5))


class TestDemand:
    def test_routes_are_deterministic_and_rates_match(self, cfg, tmp_path):
        sc = cfg.scenarios["ns_dominant"]
        a = write_routes(sc, 600, tmp_path / "a.rou.xml").read_text()
        b = write_routes(sc, 600, tmp_path / "b.rou.xml").read_text()
        assert a == b
        flows = ET.fromstring(a).findall("flow")
        assert len(flows) == 12
        n_rate = sum(float(f.get("probability")) for f in flows if f.get("from") == "N2C") * 3600
        assert n_rate == pytest.approx(sc.demand_veh_per_h["N"], rel=1e-4)


class TestMetrics:
    def _trip(self, vid, app, dep, arr, loss):
        return Trip(vid, app, dep, arr, 40.0, loss, loss / 2, 1)

    def test_summary_excludes_warmup_and_counts_unfinished(self):
        trips = [self._trip("a", "N", 10, 50, 100.0),  # warm-up: excluded
                 self._trip("b", "N", 100, 150, 10.0),
                 self._trip("c", "E", 120, 170, 30.0),
                 self._trip("d", "S", 130, -1, 99.0)]  # unfinished
        m = summarize(trips, [1.0, 3.0], warmup_s=60, phase_switches=4, interventions={}, teleports=0)
        assert m["trips_measured"] == 3 and m["throughput_completed"] == 2 and m["unfinished"] == 1
        assert m["delay_mean_s"] == pytest.approx(20.0)
        assert m["queue_mean_veh"] == 2.0 and m["queue_max_veh"] == 3.0
        assert m["delay_by_approach_s"]["W"] is None
        assert m["approach_delay_std_s"] == pytest.approx(10.0)

    def test_empty_episode_gives_none_not_zero(self):
        m = summarize([], [], 0, 0, {}, 0)
        assert m["delay_mean_s"] is None and m["queue_mean_veh"] is None

    def test_improvement_pct(self):
        assert improvement_pct(20, 15) == pytest.approx(25.0)
        assert improvement_pct(100, 110, lower_is_better=False) == pytest.approx(10.0)
        assert improvement_pct(1e-9, 5) is None
        assert improvement_pct(math.nan, 5) is None

    def test_mean_std(self):
        assert mean_std([1.0, None, 3.0]) == (2.0, pytest.approx(math.sqrt(2)), 2)
        assert mean_std([None]) == (None, None, 0)


def test_fixed_time_controller_cycles():
    ctl = FixedTimeController((30.0, 20.0))
    st = traffic(0.0)
    assert ctl.act(st, SignalStatus(Mode.GREEN, 0, 0, 29.0)).requested_phase == 0
    assert ctl.act(st, SignalStatus(Mode.GREEN, 0, 0, 30.0)).requested_phase == 1
    assert ctl.act(st, SignalStatus(Mode.GREEN, 1, 1, 20.0)).requested_phase == 0
    assert ctl.act(st, SignalStatus(Mode.YELLOW, 0, 1, 1.0)).requested_phase == 1
