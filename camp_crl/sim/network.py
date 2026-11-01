"""SUMO binary discovery, network build and signal-state construction."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import sumolib

from camp_crl.config import APPROACHES

ROOT = Path(__file__).resolve().parents[2]
NET_SRC_DIR = ROOT / "sumo" / "net"
BUILD_DIR = ROOT / "sumo" / "build"
TLS_ID = "C"

# Phase index -> approaches that receive green. Phase-1 uses a two-phase plan with
# permissive (yielding) left turns; protected left phases are future work.
PHASE_APPROACHES: dict[int, tuple[str, ...]] = {0: ("N", "S"), 1: ("E", "W")}
PHASE_NAMES: dict[int, str] = {0: "NS_GREEN", 1: "EW_GREEN"}

# Through-movement exit edge for each approach (used for downstream/overflow checks).
THROUGH_EXIT: dict[str, str] = {"N": "C2S", "S": "C2N", "E": "C2W", "W": "C2E"}


class SumoNotFoundError(RuntimeError):
    """Raised when no SUMO installation can be located."""


def sumo_home() -> Path:
    """Locate SUMO_HOME: the env var first, then the pip-installed eclipse-sumo package."""
    env = os.environ.get("SUMO_HOME")
    if env and Path(env, "bin").is_dir():
        return Path(env)
    try:
        import sumo  # eclipse-sumo wheel

        return Path(sumo.SUMO_HOME)
    except ImportError as exc:
        raise SumoNotFoundError(
            "SUMO not found. Install with `pip install -r requirements.txt` or set SUMO_HOME."
        ) from exc


def sumo_binary(name: str = "sumo") -> str:
    """Absolute path of a SUMO tool (sumo, sumo-gui, netconvert)."""
    exe = sumo_home() / "bin" / (name + (".exe" if os.name == "nt" else ""))
    if exe.is_file():
        return str(exe)
    found = shutil.which(name)
    if found:
        return found
    raise SumoNotFoundError(f"SUMO tool '{name}' not found under {sumo_home()} or on PATH")


def build_network(out_dir: Path = BUILD_DIR, force: bool = False) -> Path:
    """Run netconvert on the plain-XML sources; returns the .net.xml path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    net = out_dir / "intersection.net.xml"
    if net.is_file() and not force:
        return net
    cmd = [
        sumo_binary("netconvert"),
        "--node-files", str(NET_SRC_DIR / "intersection.nod.xml"),
        "--edge-files", str(NET_SRC_DIR / "intersection.edg.xml"),
        "--output-file", str(net),
        "--no-turnarounds", "true",
        "--tls.default-type", "static",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0 or not net.is_file():
        raise RuntimeError(f"netconvert failed ({result.returncode}): {result.stderr.strip()}")
    return net


@dataclass(frozen=True)
class SignalPlan:
    """Signal state strings for each logical phase, derived from the network's link order.

    Link characters follow SUMO: 'G' priority green, 'g' permissive green (yield),
    'y' yellow, 'r' red. Left turns get 'g' because they must yield to opposing through traffic.
    """

    green: dict[int, str]
    yellow: dict[int, str]
    all_red: str
    link_approach: tuple[str, ...]

    @property
    def phases(self) -> tuple[int, ...]:
        return tuple(sorted(self.green))


def signal_plan(net_file: Path) -> SignalPlan:
    """Derive per-phase state strings from the controlled links of TLS 'C'."""
    net = sumolib.net.readNet(str(net_file))
    links = sorted(net.getTLS(TLS_ID).getConnections(), key=lambda c: c[2])
    n = len(links)
    if [c[2] for c in links] != list(range(n)):
        raise RuntimeError("unexpected TLS link indexing")
    approaches: list[str] = []
    is_left: list[bool] = []
    for in_lane, out_lane, _ in links:
        edge_id = in_lane.getEdge().getID()
        approach = edge_id.split("2")[0]
        if approach not in APPROACHES:
            raise RuntimeError(f"unexpected incoming edge {edge_id}")
        conn = [c for c in in_lane.getOutgoing() if c.getToLane() == out_lane]
        approaches.append(approach)
        is_left.append(bool(conn) and conn[0].getDirection() in ("l", "L", "t"))
    green: dict[int, str] = {}
    yellow: dict[int, str] = {}
    for phase, served in PHASE_APPROACHES.items():
        green[phase] = "".join(
            ("g" if left else "G") if a in served else "r" for a, left in zip(approaches, is_left)
        )
        yellow[phase] = "".join("y" if a in served else "r" for a in approaches)
    return SignalPlan(green=green, yellow=yellow, all_red="r" * n, link_approach=tuple(approaches))
