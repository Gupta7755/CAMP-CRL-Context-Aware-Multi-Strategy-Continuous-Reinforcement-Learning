import json

import pytest
from pydantic import ValidationError

from camp_crl.config import ActuatedConfig, AppConfig
from camp_crl.control.controllers import (CONTROLLER_NAMES, ActuatedController, MaxPressureController,
                                          make_controller, phase_pressures)
from camp_crl.control.safety import Mode, SignalStatus
from camp_crl.tuning import tune_fixed_time, with_tuned_fixed_time
from tests.conftest import traffic


def green(phase, elapsed):
    return SignalStatus(Mode.GREEN, phase, phase, elapsed)


class TestActuated:
    def make(self):
        return ActuatedController(ActuatedConfig(gap_s=3, max_green_s=45), min_green_s=10)

    def test_extends_while_detected_then_gaps_out(self):
        ctl = self.make()
        other_waiting = {"E": {"vehicle_count": 3}}
        for t in range(0, 20):  # detections on N throughout
            d = ctl.act(traffic(float(t), N={"stopline_count": 1}, **other_waiting), green(0, t))
            assert d.requested_phase == 0
        # Zone empties at t=20; gap reaches 3 s at t=22 (last detection at t=19).
        reqs = [ctl.act(traffic(float(t), **other_waiting), green(0, t)).requested_phase for t in range(20, 24)]
        assert reqs == [0, 0, 1, 1]

    def test_no_gap_out_before_min_green(self):
        ctl = self.make()
        reqs = [ctl.act(traffic(float(t), E={"vehicle_count": 2}), green(0, t)).requested_phase for t in range(10)]
        assert reqs == [0] * 10

    def test_rests_in_green_without_conflicting_demand(self):
        ctl = self.make()
        assert all(ctl.act(traffic(float(t)), green(0, t)).requested_phase == 0 for t in range(100))

    def test_max_out_despite_continuous_detection(self):
        ctl = self.make()
        reqs = [ctl.act(traffic(float(t), N={"stopline_count": 2}, W={"vehicle_count": 1}), green(0, t))
                .requested_phase for t in range(50)]
        assert reqs.index(1) == 45

    def test_follows_transition_target(self):
        d = self.make().act(traffic(0.0), SignalStatus(Mode.YELLOW, 0, 1, 1.0))
        assert d.requested_phase == 1


class TestMaxPressure:
    def test_pressure_definition(self):
        st = traffic(0.0, N={"vehicle_count": 5, "downstream_count": 2}, S={"vehicle_count": 1},
                     E={"vehicle_count": 4})
        assert phase_pressures(st) == {0: 4.0, 1: 4.0}

    def test_selects_max_and_keeps_current_on_tie(self):
        ctl = MaxPressureController(min_green_s=10)
        tie = traffic(0.0, N={"vehicle_count": 3}, E={"vehicle_count": 3})
        assert ctl.act(tie, green(1, 20)).requested_phase == 1
        assert ctl.act(tie, green(0, 20)).requested_phase == 0
        heavy_ew = traffic(0.0, N={"vehicle_count": 1}, E={"vehicle_count": 6})
        assert ctl.act(heavy_ew, green(0, 20)).requested_phase == 1

    def test_downstream_blocking_lowers_pressure(self):
        ctl = MaxPressureController(min_green_s=10)
        st = traffic(0.0, N={"vehicle_count": 6, "downstream_count": 6}, E={"vehicle_count": 2})
        assert ctl.act(st, green(0, 20)).requested_phase == 1

    def test_holds_during_min_green(self):
        ctl = MaxPressureController(min_green_s=10)
        st = traffic(0.0, E={"vehicle_count": 20})
        assert ctl.act(st, green(0, 5)).requested_phase == 0


class TestTunedFixedTime:
    def test_missing_plan_gives_clear_error(self, cfg):
        with pytest.raises(ValueError, match="tune-fixed"):
            make_controller("fixed_time_tuned", cfg)

    def test_loads_plan_and_validates_limits(self, cfg, tmp_path):
        f = tmp_path / "ft.json"
        f.write_text(json.dumps({"green_s": [35, 25]}))
        tuned = with_tuned_fixed_time(cfg, f)
        assert tuned.fixed_time_tuned.green_s == (35, 25)
        assert make_controller("fixed_time_tuned", tuned).name == "fixed_time_tuned"
        f.write_text(json.dumps({"green_s": [5, 25]}))
        with pytest.raises(ValidationError):
            with_tuned_fixed_time(cfg, f)
        assert with_tuned_fixed_time(cfg, tmp_path / "absent.json") is cfg

    @pytest.mark.parametrize("scenario", ["ns_dominant", "ew_dominant_heldout"])
    def test_tuning_refuses_non_training_scenarios(self, cfg, scenario):
        with pytest.raises(ValueError, match="tune only on train"):
            tune_fixed_time(cfg, [scenario], [1], [30])

    def test_tuning_grid_must_respect_limits(self, cfg):
        with pytest.raises(ValueError, match="outside"):
            tune_fixed_time(cfg, ["balanced"], [1], [5, 30])


def test_actuated_max_green_must_fit_timing(cfg):
    data = cfg.model_dump()
    data["actuated"]["max_green_s"] = 90
    with pytest.raises(ValidationError):
        AppConfig.model_validate(data)


def test_every_named_controller_is_constructible(cfg):
    tuned = AppConfig.model_validate({**cfg.model_dump(), "fixed_time_tuned": {"green_s": (30, 30)}})
    assert {make_controller(n, tuned).name for n in CONTROLLER_NAMES} == set(CONTROLLER_NAMES)


def test_tuning_objective_is_not_dominated_by_high_delay_scenarios(cfg, monkeypatch):
    import camp_crl.tuning as tuning

    # plan (10, 10): 20 s better on the ~130 s bottleneck, but twice as bad on the 10 s scenario
    delays = {((10.0, 10.0), "bottleneck_s"): 110.0, ((10.0, 10.0), "balanced"): 20.0,
              ((20.0, 20.0), "bottleneck_s"): 130.0, ((20.0, 20.0), "balanced"): 10.0,
              ((10.0, 20.0), "bottleneck_s"): 140.0, ((10.0, 20.0), "balanced"): 15.0,
              ((20.0, 10.0), "bottleneck_s"): 140.0, ((20.0, 10.0), "balanced"): 15.0}

    class FakePool:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def submit(self, fn, data, ctl, sc, seed, out):
            plan = tuple(float(g) for g in data["fixed_time"]["green_s"])
            m = {"delay_mean_s": delays[(plan, sc)], "teleports": 0, "unfinished": 0, "trips_measured": 100}
            return type("F", (), {"result": lambda self: m})()

    monkeypatch.setattr(tuning, "ProcessPoolExecutor", FakePool)
    res = tuning.tune_fixed_time(cfg, ["balanced", "bottleneck_s"], [1], [10, 20])
    raw_best = min(res["candidates"], key=lambda c: c["delay_mean_s"])["green_s"]
    assert raw_best == [10.0, 10.0]           # the old objective would pick this plan
    assert res["green_s"] == [20.0, 20.0]      # relative regret picks the plan that is good everywhere
    best = next(c for c in res["candidates"] if c["green_s"] == [20.0, 20.0])
    assert best["relative_regret"] == pytest.approx((10 / 10 + 130 / 110) / 2)
