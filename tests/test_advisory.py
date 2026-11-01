"""Live signal-timing advisory: decision rules, default 30 s loop, blind roads, safety timing."""

from __future__ import annotations

import pytest

from camp_crl.vision.advisory import Advisor
from camp_crl.vision.types import ApproachPerception, Status


def per(n=0, q=0, arrivals=0, status=Status.OK) -> ApproachPerception:
    return ApproachPerception("N", 0.0, n, q, arrivals, 10.0, 0.9, status)


def run(adv, t0, t1, feed, waits=None):
    out = []
    for t in range(t0, t1):
        out.append(adv.tick(float(t), feed(t), waits(t) if waits else None))
    return out


CLEAR = lambda t: {a: per() for a in "NESW"}  # noqa: E731


def test_all_clear_runs_default_30s_loop(cfg):
    snaps = run(Advisor(cfg), 0, 140, CLEAR)
    assert all(s["mode"] == "default_loop" for s in snaps[5:])
    assert "all four directions clear" in snaps[-1]["mode_reason"]
    # green intervals of exactly 30 s, separated by 3 s yellow + 2 s all-red
    seq, prev, run_len = [], None, 0
    for s in snaps:
        key = (s["signal"]["mode"], s["signal"]["green_phase"])
        if key != prev and prev is not None:
            seq.append((prev, run_len))
            run_len = 0
        prev, run_len = key, run_len + 1
    greens = [n for (m, _), n in seq if m == "GREEN"][1:]  # the first green started before the loop took over
    assert greens and all(n == 30 for n in greens)
    assert all(n == 3 for (m, _), n in seq if m == "YELLOW") and all(n == 2 for (m, _), n in seq if m == "ALL_RED")
    assert snaps[-1]["advisory_only"] is True


def test_plan_counts_down_in_default_loop(cfg):
    s = run(Advisor(cfg), 0, 18, CLEAR)[-1]
    assert s["signal"]["mode"] == "GREEN" and s["signal"]["time_in_interval_s"] == 18  # 18 one-second ticks
    labels = [(p["color"], p["seconds"]) for p in s["plan"]]
    assert labels == [("green", 12), ("yellow", 3.0), ("red", 2.0), ("green", 30)]


def test_queue_on_one_road_turns_adaptive_and_gets_green(cfg):
    adv = Advisor(cfg)
    run(adv, 0, 12, CLEAR)  # N-S green from the start, everything empty
    arr = {"n": 0}

    def ew_queue(t):
        arr["n"] += 1
        return {"N": per(), "S": per(), "E": per(8, 7, arr["n"]), "W": per(5, 4, arr["n"])}

    snaps = run(adv, 12, 40, ew_queue, lambda t: {"E": float(t - 12)})
    assert snaps[-1]["mode"] == "adaptive"
    assert any(s["signal"]["green_phase"] == 1 and s["signal"]["mode"] == "GREEN" for s in snaps)
    ew = snaps[-1]["approaches"]["E"]
    assert ew["known"] and ew["vehicles"] == 8 and ew["queue"] == 7 and ew["light"] == "green"


def test_holding_green_reports_maximum_not_a_fake_countdown(cfg):
    adv = Advisor(cfg)
    run(adv, 0, 6, CLEAR)
    snaps = run(adv, 6, 40, lambda t: {"N": per(9, 8, t), "S": per(6, 5, t), "E": per(), "W": per()})
    s = snaps[-1]
    assert s["mode"] == "adaptive" and s["signal"]["green_phase"] == 0
    green = s["plan"][0]
    assert green["seconds"] is None and green["max_seconds"] == 60 - s["signal"]["time_in_interval_s"]


def test_max_green_still_enforced_under_continuous_demand(cfg):
    adv = Advisor(cfg)
    snaps = run(adv, 0, 90, lambda t: {"N": per(9, 8, t), "S": per(6, 5, t), "E": per(), "W": per()})
    assert any("max_green_force" in s["interventions"] for s in snaps)
    greens = [s["signal"]["time_in_interval_s"] for s in snaps if s["signal"]["mode"] == "GREEN"]
    assert max(greens) <= 60


@pytest.mark.parametrize("cams,blind", [({"N": per(3, 2, 1)}, "East–West"),
                                        ({"E": per(3, 2, 1), "W": per()}, "North–South")])
def test_road_without_camera_stays_on_default_loop_and_is_unknown(cfg, cams, blind):
    s = run(Advisor(cfg), 0, 5, lambda t: cams)[-1]
    assert s["mode"] == "default_loop" and blind in s["mode_reason"]
    unseen = [a for a in "NESW" if a not in cams]
    for a in unseen:
        assert s["approaches"][a]["known"] is False and s["approaches"][a]["vehicles"] is None


def test_disconnected_camera_counts_as_unknown(cfg):
    feed = lambda t: {"N": per(4, 3, t), "S": per(), "E": per(status=Status.DISCONNECTED), "W": per(status=Status.STALE)}  # noqa: E731
    s = run(Advisor(cfg), 0, 5, feed)[-1]
    assert s["mode"] == "default_loop" and s["cameras_working"] == 2
    assert s["approaches"]["E"]["known"] is False and s["approaches"]["W"]["vehicles"] is None


def test_default_green_must_respect_timing_limits(cfg):
    with pytest.raises(ValueError, match="outside"):
        Advisor(cfg, default_green_s=90)
