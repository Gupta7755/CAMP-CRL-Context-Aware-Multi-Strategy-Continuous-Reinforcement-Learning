"""Train DDPG / TD3 / SAC (Stable-Baselines3) on the CAMP-CRL environment.

  python -m camp_crl.rl.train --algo td3 --obs camp --seed 1 --timesteps 100000 --n-envs 4

Each run writes runs/rl/<id>/:
  manifest.json      arguments, seeds, RL + app config, software versions, status
  monitor/           SB3 Monitor CSVs (per-episode training reward / length)
  checkpoints/       periodic model snapshots
  validation.csv     periodic deterministic evaluation on validation scenarios (fixed SUMO seeds)
  best_model.zip     best mean validation reward (selection uses validation scenarios only)
  obs_stats.json     running mean/variance of training observations (novelty signal for fusion)
  final_model.zip    model after the last timestep
Held-out test scenarios are never used here.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import platform
import time
from datetime import datetime
from pathlib import Path

import gymnasium
import numpy as np
import stable_baselines3
import torch
from stable_baselines3 import DDPG, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from camp_crl.config import AppConfig, load_config
from camp_crl.rl.config import RLConfig, load_rl_config
from camp_crl.rl.env import TrafficSignalEnv
from camp_crl.rl.fusion import ObsStats
from camp_crl.sim.network import ROOT
from camp_crl.tuning import with_tuned_fixed_time

log = logging.getLogger(__name__)
ALGOS = {"td3": TD3, "ddpg": DDPG, "sac": SAC}
VALIDATION_SEEDS = (1001, 1002)


def _make_env(app_data: dict, rl_data: dict, obs: str, duration_s: int | None, seed: int, monitor_dir: str, rank: int):
    def _init():
        torch.set_num_threads(1)
        env = TrafficSignalEnv(AppConfig.model_validate(app_data), RLConfig.model_validate(rl_data), obs_mode=obs,
                               duration_s=duration_s, run_label=f"rl_{obs}")
        env = Monitor(env, str(Path(monitor_dir) / f"env{rank}"), info_keywords=("scenario",))
        env.reset(seed=seed + rank)
        env.action_space.seed(seed + rank)
        return env
    return _init


def evaluate_policy_on(model, app: AppConfig, rl: RLConfig, obs: str, scenarios: list[str], sumo_seeds,
                       duration_s: int | None) -> list[dict]:
    """Deterministic rollouts; returns one record per (scenario, sumo seed)."""
    out = []
    env = TrafficSignalEnv(app, rl, scenarios=scenarios, obs_mode=obs, duration_s=duration_s, run_label=f"rl_{obs}")
    try:
        for sc in scenarios:
            for s in sumo_seeds:
                o, _ = env.reset(seed=s, options={"scenario": sc, "sumo_seed": s})
                total, done, info = 0.0, False, {}
                while not done:
                    action, _ = model.predict(o, deterministic=True)
                    o, r, term, trunc, info = env.step(action)
                    total += r
                    done = term or trunc
                m = info.get("episode_metrics") or {}
                out.append({"scenario": sc, "sumo_seed": s, "reward": total, "delay_mean_s": m.get("delay_mean_s"),
                            "throughput": m.get("throughput_completed"), "interventions": m.get("safety_interventions_total")})
    finally:
        env.close()
    return out


class ObsStatsCallback(BaseCallback):
    def __init__(self, path: Path):
        super().__init__()
        self.stats, self.path = ObsStats(), path

    def _on_step(self) -> bool:
        self.stats.update(self.locals["new_obs"])
        return True

    def _on_training_end(self) -> None:
        self.stats.save(self.path)


class ValidationCallback(BaseCallback):
    def __init__(self, every: int, run_dir: Path, app: AppConfig, rl: RLConfig, obs: str, duration_s: int | None):
        super().__init__()
        self.every, self.run_dir, self.app, self.rl, self.obs, self.duration_s = every, run_dir, app, rl, obs, duration_s
        self.best = -np.inf
        self._next = every
        self.last_validated = -1
        self.csv = run_dir / "validation.csv"
        with self.csv.open("w", newline="") as fh:
            csv.writer(fh).writerow(["timesteps", "mean_reward", "mean_delay_s", "per_scenario"])

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            self._next += self.every
            self.validate()
        return True

    def validate(self) -> float:
        self.last_validated = self.num_timesteps
        recs = evaluate_policy_on(self.model, self.app, self.rl, self.obs, self.rl.validation_scenarios,
                                  VALIDATION_SEEDS, self.duration_s)
        mean_r = float(np.mean([r["reward"] for r in recs]))
        delays = [r["delay_mean_s"] for r in recs if r["delay_mean_s"] is not None]
        mean_d = float(np.mean(delays)) if delays else float("nan")
        with self.csv.open("a", newline="") as fh:
            csv.writer(fh).writerow([self.num_timesteps, round(mean_r, 4), round(mean_d, 3), json.dumps(recs)])
        if mean_r > self.best:
            self.best = mean_r
            self.model.save(self.run_dir / "best_model")
        log.info("validation at %d steps: reward=%.3f delay=%.2f (best %.3f)", self.num_timesteps, mean_r, mean_d, self.best)
        return mean_r


def train(algo: str, obs: str, seed: int, timesteps: int, n_envs: int = 4, duration_s: int | None = None,
          checkpoint_every: int = 20_000, validate_every: int = 20_000, out_root: Path | None = None,
          app: AppConfig | None = None, rl: RLConfig | None = None) -> Path:
    if algo not in ALGOS:
        raise ValueError(f"algo must be one of {sorted(ALGOS)}")
    app = app or with_tuned_fixed_time(load_config(ROOT / "configs" / "default.yaml"))
    rl = rl or load_rl_config(ROOT / "configs" / "rl.yaml")
    rl.check_against(app)
    torch.set_num_threads(1)
    set_random_seed(seed)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = (out_root or ROOT / "runs" / "rl") / f"{stamp}_{algo}_{obs}_s{seed}"
    (run_dir / "monitor").mkdir(parents=True)
    manifest = {
        "id": run_dir.name, "algo": algo, "obs": obs, "seed": seed, "timesteps": timesteps, "n_envs": n_envs,
        "episode_duration_s": duration_s or app.simulation.duration_s, "status": "running",
        "created": datetime.now().isoformat(timespec="seconds"), "validation_sumo_seeds": list(VALIDATION_SEEDS),
        "rl_config": rl.model_dump(mode="json"), "app_config": app.model_dump(mode="json"),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "stable_baselines3": stable_baselines3.__version__, "gymnasium": gymnasium.__version__,
                     "numpy": np.__version__},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    fns = [_make_env(app.model_dump(mode="json"), rl.model_dump(mode="json"), obs, duration_s, seed * 1000,
                     str(run_dir / "monitor"), i) for i in range(n_envs)]
    venv = SubprocVecEnv(fns) if n_envs > 1 else DummyVecEnv(fns)
    a = rl.algo
    kwargs = dict(learning_rate=a.learning_rate, buffer_size=a.buffer_size, batch_size=a.batch_size, gamma=a.gamma,
                  tau=a.tau, learning_starts=a.learning_starts, seed=seed, verbose=0, device="cpu",
                  policy_kwargs={"net_arch": a.net_arch})
    if algo != "sac":  # SAC explores through its stochastic policy (automatic entropy tuning)
        kwargs["action_noise"] = NormalActionNoise(np.zeros(2), a.action_noise_std * np.ones(2))
    if algo == "td3":
        kwargs["policy_delay"] = a.policy_delay
    model = ALGOS[algo]("MlpPolicy", venv, **kwargs)
    callbacks = [CheckpointCallback(max(checkpoint_every // n_envs, 1), str(run_dir / "checkpoints"), name_prefix=algo),
                 ValidationCallback(validate_every, run_dir, app, rl, obs, duration_s),
                 ObsStatsCallback(run_dir / "obs_stats.json")]
    t0 = time.perf_counter()
    try:
        model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=False)
        model.save(run_dir / "final_model")
        if callbacks[1].last_validated != model.num_timesteps:
            callbacks[1].validate()  # final validation point (may update best_model)
        manifest.update(status="completed")
    except BaseException as exc:
        manifest.update(status="failed", error=repr(exc))
        raise
    finally:
        venv.close()
        manifest.update(wall_time_s=round(time.perf_counter() - t0, 1), finished=datetime.now().isoformat(timespec="seconds"))
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return run_dir


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("camp_crl.sim.traci_adapter", "camp_crl.runner"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", choices=sorted(ALGOS), required=True)
    p.add_argument("--obs", choices=["raw", "camp"], required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--timesteps", type=int, default=100_000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--episode-s", type=int, default=None, help="shorter episodes for smoke tests")
    p.add_argument("--checkpoint-every", type=int, default=20_000)
    p.add_argument("--validate-every", type=int, default=20_000)
    args = p.parse_args()
    run = train(args.algo, args.obs, args.seed, args.timesteps, args.n_envs, args.episode_s,
                args.checkpoint_every, args.validate_every)
    print(f"run saved: {run}")


if __name__ == "__main__":
    main()
