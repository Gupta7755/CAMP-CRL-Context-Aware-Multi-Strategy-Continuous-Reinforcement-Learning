"""SUMO process lifecycle via TraCI.

Each session uses a unique TraCI label so several sessions can coexist, and `close()` is
idempotent and always waits for the SUMO process to exit (no orphan processes).
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from types import TracebackType

import traci

from camp_crl.sim.network import sumo_binary

log = logging.getLogger(__name__)


class SumoSession:
    """Context manager owning one SUMO process and its TraCI connection."""

    def __init__(
        self,
        net_file: Path,
        route_file: Path,
        seed: int,
        step_length_s: float = 1.0,
        time_to_teleport_s: int = 300,
        tripinfo_file: Path | None = None,
        gui: bool = False,
        gui_delay_ms: int = 50,
    ) -> None:
        for f in (net_file, route_file):
            if not Path(f).is_file():
                raise FileNotFoundError(f"SUMO input missing: {f}")
        self.label = f"camp-{uuid.uuid4().hex[:8]}"
        self._cmd = [
            sumo_binary("sumo-gui" if gui else "sumo"),
            "--net-file", str(net_file),
            "--route-files", str(route_file),
            "--seed", str(seed),
            "--step-length", str(step_length_s),
            "--time-to-teleport", str(time_to_teleport_s),
            # Keep the full waiting history so accumulated waiting time = total time stopped.
            "--waiting-time-memory", "100000",
            "--no-step-log", "true",
            "--no-warnings", "true",
            "--duration-log.disable", "true",
        ]
        if tripinfo_file is not None:
            self._cmd += ["--tripinfo-output", str(tripinfo_file), "--tripinfo-output.write-unfinished", "true"]
        if gui:
            # Start immediately, close the window at the end, and slow down enough to watch (ms per step).
            self._cmd += ["--start", "true", "--quit-on-end", "true", "--delay", str(gui_delay_ms)]
        self.conn: traci.connection.Connection | None = None

    def start(self) -> traci.connection.Connection:
        if self.conn is not None:
            raise RuntimeError("session already started")
        log.info("starting SUMO label=%s", self.label)
        traci.start(self._cmd, label=self.label)
        self.conn = traci.getConnection(self.label)
        return self.conn

    def close(self) -> None:
        if self.conn is None:
            return
        try:
            self.conn.close(wait=True)
        except Exception as exc:  # SUMO may already have died; still mark closed
            log.warning("error while closing SUMO label=%s: %s", self.label, exc)
        finally:
            self.conn = None
            log.info("closed SUMO label=%s", self.label)

    def __enter__(self) -> traci.connection.Connection:
        return self.start()

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()
