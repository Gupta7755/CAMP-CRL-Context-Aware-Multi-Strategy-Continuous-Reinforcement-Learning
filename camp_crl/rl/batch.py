"""Launch the Phase-2 training matrix as parallel processes.

  python -m camp_crl.rl.batch --timesteps 60000 --seeds 1 2 3 --parallel 4 --n-envs 3

Configurations (the experimental-design list from the build spec):
  ddpg/raw  - raw-state DDPG (reference baseline)
  td3/raw   - raw-state TD3
  td3/camp  - CAMP-state TD3 (also the policy inside full CAMP-CRL fusion)
Each run is an independent `camp_crl.rl.train` process; stdout/stderr go to runs/rl/logs/.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from camp_crl.sim.network import ROOT

CONFIGS = (("ddpg", "raw"), ("td3", "raw"), ("td3", "camp"))
OPTIONAL = (("sac", "camp"),)  # optional comparison from the build spec


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--timesteps", type=int, default=60_000)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--parallel", type=int, default=4, help="training processes at once")
    p.add_argument("--n-envs", type=int, default=3, help="SUMO environments per training process")
    p.add_argument("--validate-every", type=int, default=10_000)
    p.add_argument("--configs", nargs="+", default=None, help="subset like td3/camp sac/camp (default: the 3 core configs)")
    args = p.parse_args()
    logs = ROOT / "runs" / "rl" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    configs = [tuple(c.split("/")) for c in args.configs] if args.configs else list(CONFIGS)
    for c in configs:
        if c not in CONFIGS + OPTIONAL:
            p.error(f"unknown config {'/'.join(c)}")
    queue = [(a, o, s) for s in args.seeds for a, o in configs]
    running: list[tuple[subprocess.Popen, str]] = []
    failures = 0
    while queue or running:
        while queue and len(running) < args.parallel:
            algo, obs, seed = queue.pop(0)
            tag = f"{algo}_{obs}_s{seed}"
            cmd = [sys.executable, "-m", "camp_crl.rl.train", "--algo", algo, "--obs", obs, "--seed", str(seed),
                   "--timesteps", str(args.timesteps), "--n-envs", str(args.n_envs),
                   "--validate-every", str(args.validate_every), "--checkpoint-every", str(args.validate_every)]
            fh = (logs / f"{tag}.log").open("w", encoding="utf-8")
            running.append((subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT), tag))
            print(f"started {tag}", flush=True)
        time.sleep(5)
        for proc, tag in list(running):
            if proc.poll() is not None:
                running.remove((proc, tag))
                failures += proc.returncode != 0
                print(f"finished {tag}: exit {proc.returncode}", flush=True)
    print(f"all done, {failures} failed", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
