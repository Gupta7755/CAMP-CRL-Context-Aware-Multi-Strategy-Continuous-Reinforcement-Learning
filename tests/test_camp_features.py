import math

import pytest

from camp_crl.camp.features import (CampEngine, composite, fairness, momentum, normalize, pressure, risk,
                                    signed_scale)
from tests.conftest import traffic


class TestNormalize:
    def test_boundaries(self):
        assert normalize(0, 0, 10) == 0.0
        assert normalize(10, 0, 10) == 1.0
        assert normalize(5, 0, 10) == 0.5

    def test_out_of_range_and_negative_are_clipped(self):
        assert normalize(-3, 0, 10) == 0.0
        assert normalize(25, 0, 10) == 1.0
        assert normalize(-5, -10, 0) == 0.5

    @pytest.mark.parametrize("x", [math.nan, math.inf, -math.inf])
    def test_non_finite_returns_safe_value(self, x):
        assert normalize(x, 0, 10) == 0.0
        assert normalize(x, 0, 10, safe_value=0.5) == 0.5

    def test_zero_range_returns_safe_value(self):
        assert normalize(5, 3, 3) == 0.0
        assert normalize(5, 3, 3 + 1e-12, safe_value=0.25) == 0.25

    def test_signed_scale(self):
        assert signed_scale(0.25, 0.5) == 0.5
        assert signed_scale(-9, 0.5) == -1.0
        assert signed_scale(math.nan, 1) == 0.0
        assert signed_scale(1, 0) == 0.0


class TestFormulas:
    def test_pressure_is_weighted_sum_in_unit_interval(self, cfg):
        ones = dict(queue=1, arrival=1, wait=1, density=1, spillback=1)
        assert pressure(ones, cfg.camp) == pytest.approx(1.0)
        assert pressure({k: 0 for k in ones}, cfg.camp) == 0.0
        only_q = {**{k: 0 for k in ones}, "queue": 1}
        assert pressure(only_q, cfg.camp) == pytest.approx(cfg.camp.pressure.queue)

    def test_momentum_signs(self, cfg):
        assert momentum(1, 1, -1, cfg.camp) == pytest.approx(1.0)  # growing queue, falling speed
        assert momentum(-1, -1, 1, cfg.camp) == pytest.approx(-1.0)
        assert momentum(0, 0, 1, cfg.camp) < 0  # speeding up reduces momentum

    def test_risk_is_clipped_heuristic(self, cfg):
        c = cfg.camp
        assert risk(0.5, 0.0, c) == 0.5
        assert risk(0.5, 1.0, c) == pytest.approx(0.5 + c.risk_horizon_s * c.risk_k_m)
        assert risk(0.0, -1.0, c) == 0.0
        assert risk(1.0, 1.0, c) == c.risk_max

    def test_fairness(self, cfg):
        assert fairness(0, cfg.camp) == 0.0
        assert fairness(cfg.camp.fairness_ref_wait_s / 2, cfg.camp) == pytest.approx(0.5)
        assert fairness(10_000, cfg.camp) == 1.0
        assert fairness(math.nan, cfg.camp) == 0.0

    def test_composite(self, cfg):
        c = cfg.camp
        assert composite(0.4, 0.2, 0.5, 1.0, c) == pytest.approx(
            0.4 + c.lambda_momentum * 0.2 + c.lambda_risk * 0.5 + c.lambda_fairness * 1.0)


class TestEngine:
    def test_queue_growth_gives_positive_momentum(self, cfg):
        eng = CampEngine(cfg.camp)
        for t in range(10):
            ctx = eng.update(traffic(float(t), N={"queue_veh": 2.0 * t}))
        assert ctx.approaches["N"].momentum > 0
        assert ctx.approaches["E"].momentum == 0
        assert ctx.approaches["N"].pressure > ctx.approaches["E"].pressure

    def test_stale_timestamp_does_not_change_derivatives(self, cfg):
        eng = CampEngine(cfg.camp)
        eng.update(traffic(0.0))
        m1 = eng.update(traffic(1.0, N={"queue_veh": 5.0})).approaches["N"].momentum
        m2 = eng.update(traffic(1.0, N={"queue_veh": 50.0})).approaches["N"].momentum  # same time
        assert m2 == m1

    def test_nan_observation_stays_finite(self, cfg):
        eng = CampEngine(cfg.camp)
        eng.update(traffic(0.0))
        ctx = eng.update(traffic(1.0, S={"queue_veh": math.nan, "max_wait_s": math.nan,
                                         "mean_wait_s": math.nan, "mean_speed_mps": math.nan}))
        for c in ctx.approaches.values():
            assert all(math.isfinite(v) for v in (c.pressure, c.momentum, c.risk, c.fairness, c.composite))
        assert math.isfinite(ctx.wait_dispersion_s)

    def test_wait_dispersion_and_max(self, cfg):
        ctx = CampEngine(cfg.camp).update(traffic(0.0, N={"mean_wait_s": 40.0, "max_wait_s": 70.0}))
        assert ctx.max_wait_s == 70.0
        assert ctx.wait_dispersion_s == pytest.approx(math.sqrt((30**2 + 3 * 10**2) / 4))
