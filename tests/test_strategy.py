from camp_crl.camp.features import CampEngine
from camp_crl.camp.strategy import Strategy, StrategySelector, classify
from tests.conftest import traffic

HEAVY_NS = {"queue_veh": 40.0, "max_wait_s": 30.0, "occupancy_pct": 40.0, "arrival_veh_per_h": 900.0}


def ctx_of(cfg, t=0.0, **per):
    return CampEngine(cfg.camp).update(traffic(t, **per))


def test_rules(cfg):
    sc = cfg.strategy
    assert classify(ctx_of(cfg), sc)[0] is Strategy.NORMAL_ADAPTIVE
    assert classify(ctx_of(cfg, N={"max_wait_s": 1000.0}), sc)[0] is Strategy.FAIRNESS_RECOVERY
    assert classify(ctx_of(cfg, E={"downstream_congestion": 0.8}), sc)[0] is Strategy.OVERFLOW_PREVENTION
    assert classify(ctx_of(cfg, N=HEAVY_NS, S=HEAVY_NS), sc)[0] is Strategy.DIRECTIONAL_PRIORITY
    allheavy = {a: {**HEAVY_NS, "spillback_ratio": 1.0} for a in "NESW"}
    assert classify(ctx_of(cfg, **allheavy), sc)[0] is Strategy.CONGESTION_RECOVERY
    moderate = {a: {"queue_veh": 20.0, "arrival_veh_per_h": 500.0} for a in "NESW"}  # P ~= 0.24
    assert classify(ctx_of(cfg, **moderate), sc)[0] is Strategy.BALANCED_FLOW


def test_overflow_has_priority_over_fairness(cfg):
    s, _ = classify(ctx_of(cfg, N={"max_wait_s": 1000.0}, E={"downstream_congestion": 0.9}), cfg.strategy)
    assert s is Strategy.OVERFLOW_PREVENTION


def test_hysteresis_requires_confirmation_and_hold(cfg):
    sel = StrategySelector(cfg.strategy)
    eng = CampEngine(cfg.camp)
    sel.update(eng.update(traffic(0.0)))  # starts NORMAL at t=0
    trigger = {"N": {"max_wait_s": 1000.0}}
    # Before min_hold_s (15 s) a persistent trigger must not switch.
    for t in range(1, 10):
        assert sel.update(eng.update(traffic(float(t), **trigger))).strategy is Strategy.NORMAL_ADAPTIVE
    changed_at = None
    for t in range(10, 30):
        d = sel.update(eng.update(traffic(float(t), **trigger)))
        if d.changed:
            changed_at = t
            break
    assert changed_at == cfg.strategy.min_hold_s
    assert sel.current is Strategy.FAIRNESS_RECOVERY


def test_single_spike_does_not_switch(cfg):
    sel = StrategySelector(cfg.strategy)
    eng = CampEngine(cfg.camp)
    for t in range(0, 40):
        per = {"N": {"max_wait_s": 1000.0}} if t % 3 == 0 else {}
        assert sel.update(eng.update(traffic(float(t), **per))).strategy is Strategy.NORMAL_ADAPTIVE
