"""Safety-filter invariants, checked against adversarial request sequences."""

import random

import pytest

from camp_crl.config import TimingConfig
from camp_crl.control.safety import Mode, SafetyFilter


def run(filt: SafetyFilter, requests, dt=1.0):
    return [filt.step(r, dt) for r in requests]


def segments(cmds):
    """Collapse a command trace into [(mode, green_phase, length_in_steps)]."""
    out = []
    for c in cmds:
        key = (c.mode, c.green_phase)
        if out and out[-1][0] == key:
            out[-1][1] += 1
        else:
            out.append([key, 1])
    return [(k[0], k[1], n) for k, n in out]


def test_signal_plan_has_no_conflicting_greens(plan):
    g0, g1 = plan.green[0], plan.green[1]
    assert len(g0) == len(g1) == len(plan.all_red) == 16
    for a, b in zip(g0, g1):
        assert not (a in "Gg" and b in "Gg"), "a link is green in both phases"
    assert set(plan.all_red) == {"r"}
    # every link is served by exactly one phase
    assert all((a in "Gg") != (b in "Gg") for a, b in zip(g0, g1))
    # left turns are permissive
    assert g0.count("g") == 2 and g1.count("g") == 2


@pytest.mark.parametrize("seed", range(5))
def test_invariants_under_random_requests(plan, seed):
    t = TimingConfig()
    filt = SafetyFilter(plan, t)
    rng = random.Random(seed)
    reqs = [rng.choice([0, 1, 1, 0, 7, None, -1, "x", True]) for _ in range(2000)]
    cmds = run(filt, reqs)
    segs = segments(cmds)
    for i, (mode, phase, n) in enumerate(segs):
        if mode is Mode.GREEN:
            last = i == len(segs) - 1
            assert n <= t.max_green_s
            if i > 0 and not last:
                assert n >= t.min_green_s
        elif mode is Mode.YELLOW:
            assert n == t.yellow_s or i == len(segs) - 1
            assert segs[i - 1][0] is Mode.GREEN and segs[i - 1][1] == phase
        elif mode is Mode.ALL_RED:
            assert n == t.all_red_s or i == len(segs) - 1
            assert segs[i - 1][0] is Mode.YELLOW
            if i + 1 < len(segs):
                assert segs[i + 1][0] is Mode.GREEN and segs[i + 1][1] != phase
    for c in cmds:  # emitted state always matches the mode
        expected = {Mode.GREEN: plan.green, Mode.YELLOW: plan.yellow}.get(c.mode)
        assert c.state == (expected[c.green_phase] if expected else plan.all_red)


def test_min_green_hold_is_logged(plan):
    filt = SafetyFilter(plan, TimingConfig())
    cmds = run(filt, [1] * 3)
    assert all(c.mode is Mode.GREEN and c.green_phase == 0 for c in cmds)
    assert [i.kind for c in cmds for i in c.interventions] == ["min_green_hold"] * 3


def test_switch_after_min_green_goes_through_clearance(plan):
    t = TimingConfig(min_green_s=10, yellow_s=3, all_red_s=2)
    filt = SafetyFilter(plan, t)
    cmds = run(filt, [0] * 10 + [1] * 10)
    modes = [c.mode for c in cmds]
    assert modes[:10] == [Mode.GREEN] * 10
    assert modes[10:13] == [Mode.YELLOW] * 3
    assert modes[13:15] == [Mode.ALL_RED] * 2
    assert modes[15] is Mode.GREEN and cmds[15].green_phase == 1
    assert filt.switches == 1
    assert not any(c.interventions for c in cmds)


def test_max_green_forces_switch(plan):
    t = TimingConfig(max_green_s=60)
    filt = SafetyFilter(plan, t)
    cmds = run(filt, [0] * 70)
    assert sum(c.mode is Mode.GREEN and c.green_phase == 0 for c in cmds) == 60
    # Only the forced switch before phase 1 turns green; later holds are for phase 1's min green.
    kinds = [i.kind for c in cmds[:65] for i in c.interventions]
    assert kinds == ["max_green_force"]


def test_invalid_phase_holds_current(plan):
    filt = SafetyFilter(plan, TimingConfig())
    cmds = run(filt, [5, None, "1", 2.0])
    assert all(c.green_phase == 0 and c.mode is Mode.GREEN for c in cmds)
    assert [i.kind for c in cmds for i in c.interventions] == ["invalid_phase"] * 4


def test_starvation_bound(plan):
    # max_red below what max_green would allow, so starvation triggers first.
    t = TimingConfig(min_green_s=5, max_green_s=40, yellow_s=3, all_red_s=2, max_red_s=45)
    filt = SafetyFilter(plan, t)
    run(filt, [0] * 30)
    filt.red_time[1] = 46.0  # simulate a long red
    cmd = filt.step(0, 1.0)
    assert cmd.mode is Mode.YELLOW
    assert [i.kind for i in cmd.interventions] == ["starvation_force"]


def test_requests_ignored_during_transition(plan):
    filt = SafetyFilter(plan, TimingConfig())
    run(filt, [0] * 10)
    cmds = run(filt, [1, 0, 0, 0, 0])  # change, then try to cancel it mid-yellow
    assert [c.mode for c in cmds] == [Mode.YELLOW] * 3 + [Mode.ALL_RED] * 2
    assert filt.step(0, 1.0).green_phase == 1


def test_initial_phase_validated(plan):
    with pytest.raises(ValueError):
        SafetyFilter(plan, TimingConfig(), initial_phase=3)
