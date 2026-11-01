import math
import xml.etree.ElementTree as ET

import pytest
from pydantic import ValidationError

from camp_crl.camp.features import CampEngine
from camp_crl.config import BottleneckEvent, Scenario, SensorModel
from camp_crl.experiments import aggregate, bootstrap_ci
from camp_crl.sim.demand import write_routes
from camp_crl.state import SensorDegrader
from tests.conftest import traffic

FLAT = {a: 400.0 for a in "NESW"}


class TestScenarios:
    def test_segments_cover_duration(self):
        sc = Scenario(name="x", demand_veh_per_h=FLAT, profile=[
            {"start_s": 900, "demand_veh_per_h": {**FLAT, "N": 900}},
            {"start_s": 1200, "demand_veh_per_h": FLAT}])
        segs = sc.segments(1800)
        assert [(b, e) for b, e, _ in segs] == [(0, 900), (900, 1200), (1200, 1800)]
        assert segs[1][2]["N"] == 900
        assert [(b, e) for b, e, _ in sc.segments(1000)] == [(0, 900), (900, 1000)]

    def test_profile_must_increase(self):
        with pytest.raises(ValidationError):
            Scenario(name="x", demand_veh_per_h=FLAT, profile=[
                {"start_s": 900, "demand_veh_per_h": FLAT}, {"start_s": 600, "demand_veh_per_h": FLAT}])

    def test_routes_per_segment_sorted(self, tmp_path):
        sc = Scenario(name="x", demand_veh_per_h=FLAT, profile=[{"start_s": 600, "demand_veh_per_h": {**FLAT, "W": 0}}])
        flows = ET.fromstring(write_routes(sc, 1200, tmp_path / "r.xml").read_text()).findall("flow")
        begins = [float(f.get("begin")) for f in flows]
        assert begins == sorted(begins)
        assert len(flows) == 12 + 9  # W has no flows in the second segment
        assert all(f.get("end") == "600" for f in flows if f.get("begin") == "0")

    @pytest.mark.parametrize("kw", [dict(edge="N2C"), dict(end_s=100), dict(speed_mps=0)])
    def test_bottleneck_validation(self, kw):
        base = dict(edge="C2S", speed_mps=2.0, start_s=600, end_s=1200)
        with pytest.raises(ValidationError):
            BottleneckEvent(**{**base, **kw})

    def test_default_config_has_all_splits_and_kinds(self, cfg):
        splits = {s.split for s in cfg.scenarios.values()}
        assert splits == {"train", "validation", "test"}
        assert any(s.profile for s in cfg.scenarios.values() if s.split == "test")
        assert any(s.bottlenecks for s in cfg.scenarios.values() if s.split == "test")
        assert any(s.sensor.dropout_prob > 0 for s in cfg.scenarios.values() if s.split == "test")


class TestSensorDegrader:
    def test_inactive_is_identity(self):
        st = traffic(1.0, N={"queue_veh": 5.0})
        assert SensorDegrader(SensorModel(), 1).apply(st) is st

    def test_deterministic_per_seed(self):
        m = SensorModel(noise_std_frac=0.3, dropout_prob=0.2)
        seq = [traffic(float(t), N={"queue_veh": 10.0, "vehicle_count": 12}) for t in range(50)]
        a = [SensorDegrader(m, 5).apply(s) for s in seq]
        b = [SensorDegrader(m, 5).apply(s) for s in seq]
        c = [SensorDegrader(m, 6).apply(s) for s in seq]
        assert [x.to_dict() for x in a] == [x.to_dict() for x in b]
        assert [x.to_dict() for x in a] != [x.to_dict() for x in c]

    def test_dropout_repeats_last_value_and_flags_stale(self):
        d = SensorDegrader(SensorModel(dropout_prob=0.999999), 0)
        first = d.apply(traffic(0.0, N={"queue_veh": 3.0}))  # nothing cached yet -> fresh
        assert first.stale == ()
        second = d.apply(traffic(1.0, N={"queue_veh": 30.0}))
        assert set(second.stale) == set("NESW")
        assert second.approaches["N"].queue_veh == first.approaches["N"].queue_veh
        assert second.source == "degraded_sensor"

    def test_noise_keeps_values_valid(self):
        d = SensorDegrader(SensorModel(noise_std_frac=1.0), 0)
        for t in range(200):
            st = d.apply(traffic(float(t), N={"queue_veh": 2.0, "vehicle_count": 3}))
            n = st.approaches["N"]
            assert n.queue_veh >= 0 and isinstance(n.vehicle_count, int) and n.vehicle_count >= 0


class TestExplainabilitySignals:
    @staticmethod
    def _arrivals(rng, rate_n, rate_other, t0, t1):
        for t in range(t0, t1):
            yield t, {a: {"arrivals_step": int(rng.random() < (rate_n if a == "N" else rate_other) / 3600)}
                      for a in "NESW"}

    def test_cusum_low_false_alarms_on_poisson_noise(self, cfg):
        import numpy as np
        onsets = 0
        for seed in range(5):
            eng, rng, prev = CampEngine(cfg.camp), np.random.default_rng(seed), set()
            for t, per in self._arrivals(rng, 400, 400, 0, 3600):
                ctx = eng.update(traffic(float(t), **per))
                now = {a for a, c in ctx.approaches.items() if c.anomaly}
                onsets += len(now - prev)
                prev = now
        # 20 approach-hours of stationary Poisson arrivals; calibration target ~0.7 per approach-hour
        assert onsets / 20 < 2.0

    def test_cusum_detects_surge_on_the_surging_approach(self, cfg):
        import numpy as np
        detected, delays = 0, []
        for seed in range(10):
            eng, rng = CampEngine(cfg.camp), np.random.default_rng(100 + seed)
            for t, per in self._arrivals(rng, 350, 350, 0, 900):
                eng.update(traffic(float(t), **per))
            for t, per in self._arrivals(rng, 950, 350, 900, 1200):
                ctx = eng.update(traffic(float(t), **per))
                if ctx.approaches["N"].anomaly:
                    detected += 1
                    delays.append(t - 900)
                    break
        assert detected >= 9
        assert sorted(delays)[len(delays) // 2] < 150

    def test_cusum_ignores_stale_and_repeated_steps(self, cfg):
        from camp_crl.state import TrafficState
        eng = CampEngine(cfg.camp)
        for t in range(200):
            eng.update(traffic(float(t)))
        burst = traffic(200.0, N={"arrivals_step": 5})
        stale = TrafficState(200.0, burst.approaches, "degraded_sensor", ("N",))
        for t in range(201, 230):
            ctx = eng.update(TrafficState(float(t), stale.approaches, "degraded_sensor", ("N",)))
        assert ctx.approaches["N"].anomaly_score == 0.0
        eng.update(traffic(230.0, N={"arrivals_step": 1}))
        s1 = eng.update(traffic(230.0, N={"arrivals_step": 1})).approaches["N"].anomaly_score  # same timestamp
        s0 = eng.update(traffic(231.0)).approaches["N"].anomaly_score
        assert s1 >= s0  # no extra accumulation from the repeated step

    def test_no_cusum_during_warmup(self, cfg):
        eng = CampEngine(cfg.camp)
        ctx = None
        for t in range(0, 100):  # < cusum_warmup_s
            ctx = eng.update(traffic(float(t), N={"arrivals_step": 3 if t > 50 else 0}))
        assert ctx.approaches["N"].anomaly_score == 0.0

    def test_trend_labels(self, cfg):
        eng = CampEngine(cfg.camp)
        for t in range(30):
            ctx = eng.update(traffic(float(t), N={"queue_veh": float(t)}, S={"queue_veh": float(30 - t)}))
        assert ctx.approaches["N"].trend == "RISING"
        assert ctx.approaches["S"].trend == "FALLING"
        assert ctx.approaches["E"].trend == "STABLE"

    @pytest.mark.parametrize("per,expected", [
        ({}, "LOW_DEMAND"),
        ({a: {"queue_veh": 40.0, "max_wait_s": 120.0, "occupancy_pct": 50.0, "spillback_ratio": 1.0}
          for a in "NESW"}, "CONGESTED"),
        ({"N": {"queue_veh": 40.0, "max_wait_s": 60.0}}, "NS_DOMINANT"),
        ({"W": {"queue_veh": 40.0, "max_wait_s": 60.0}}, "EW_DOMINANT"),
        ({a: {"queue_veh": 16.0} for a in "NESW"}, "BALANCED"),
    ])
    def test_pattern(self, cfg, per, expected):
        assert CampEngine(cfg.camp).update(traffic(0.0, **per)).pattern == expected

    def test_confidence_reflects_stale_share(self, cfg):
        from camp_crl.state import TrafficState
        st = traffic(0.0)
        ctx = CampEngine(cfg.camp).update(TrafficState(0.0, st.approaches, "degraded_sensor", ("N",)))
        assert ctx.confidence == 0.75 and ctx.approaches["N"].stale is True


class TestStatistics:
    def test_bootstrap_ci(self):
        assert bootstrap_ci([1.0]) is None
        lo, hi = bootstrap_ci([10.0, 12.0, 11.0, 13.0, 9.0])
        assert lo < 11.0 < hi and 9.0 <= lo and hi <= 13.0
        assert bootstrap_ci([5.0, 5.0, 5.0]) == (5.0, 5.0)
        assert bootstrap_ci([1.0, 2.0, 3.0], seed=1) == bootstrap_ci([1.0, 2.0, 3.0], seed=1)

    def test_aggregate_paired(self):
        rows = []
        for seed in range(1, 6):
            rows.append({"controller": "base", "scenario": "s", "seed": seed, "delay_mean_s": 20.0 + seed})
            rows.append({"controller": "cand", "scenario": "s", "seed": seed, "delay_mean_s": 15.0 + seed})
        out = aggregate(rows, "base", {"s": "train"})
        p = out["scenarios"]["s"]["paired_vs_baseline"]["cand"]["delay_mean_s"]
        assert p["mean_diff"] == pytest.approx(-5.0) and p["ci_excludes_zero"] is True
        assert p["improvement_pct_of_means"] == pytest.approx(100 * 5 / 23)
        assert out["by_split"]["train"]["cand"]["delay_mean_s"] == pytest.approx(18.0)
        missing = out["scenarios"]["s"]["controllers"]["cand"]["throughput_completed"]
        assert missing["mean"] is None and missing["n"] == 0
        assert not math.isnan(out["scenarios"]["s"]["controllers"]["base"]["delay_mean_s"]["std"])
