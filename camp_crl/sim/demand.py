"""Route/demand generation from a validated Scenario."""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import quoteattr

from camp_crl.config import APPROACHES, Scenario

# Turn destination for each approach, ordered (left, straight, right).
# Vehicles on N2C travel south; their left is east.
TURN_EXITS: dict[str, tuple[str, str, str]] = {
    "N": ("C2E", "C2S", "C2W"),
    "S": ("C2W", "C2N", "C2E"),
    "E": ("C2S", "C2W", "C2N"),
    "W": ("C2N", "C2E", "C2S"),
}


def write_routes(scenario: Scenario, duration_s: int, path: Path, include_flows: bool = True) -> Path:
    """Write a .rou.xml with one Bernoulli flow per movement and demand segment.

    Each flow inserts with per-second probability rate/3600, so the arrival process is
    stochastic and controlled by SUMO's --seed (same seed -> same demand realisation).
    Named routes r_<approach>_<L|T|R> are always defined so vehicles can also be added through
    TraCI; with include_flows=False only those are written (camera-driven demand, Phase 3).
    """
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f"<!-- scenario {scenario.name} (split={scenario.split}) -->",
        "<routes>",
        '    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="13.89"/>',
    ]
    for a in APPROACHES:
        for turn, exit_edge in zip(("L", "T", "R"), TURN_EXITS[a]):
            lines.append(f'    <route id="r_{a}_{turn}" edges="{a}2C {exit_edge}"/>')
    flows = []
    for k, (begin, end, demand) in enumerate(scenario.segments(duration_s)):
        for a in APPROACHES:
            for turn, share, exit_edge in zip(("L", "T", "R"), scenario.turn_shares, TURN_EXITS[a]):
                prob = demand[a] * share / 3600.0
                if prob <= 0 or not include_flows:
                    continue
                flows.append((begin, f"{a}_{turn}_{k}", end, prob, a, exit_edge))
    for begin, fid, end, prob, a, exit_edge in sorted(flows):  # SUMO requires flows sorted by begin
        lines.append(
            f"    <flow id={quoteattr(fid)} type=\"car\" begin=\"{begin:g}\" end=\"{end:g}\" "
            f"probability=\"{prob:.6f}\" from=\"{a}2C\" to=\"{exit_edge}\" "
            f'departLane="best" departSpeed="max"/>'
        )
    lines.append("</routes>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
