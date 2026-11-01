"""Reward sensitivity analysis: does the reward rank policies the way the traffic metrics do?

  python -m camp_crl.rl.reward_sensitivity --seeds 1 2 --workers 6

Five fixed policies (CAMP prior, always switch, hold 5 s, hold 15 s, uniform random) are rolled out on
the train scenarios. Each episode records the summed reward components separately, so the total reward
can be recomputed for any weighting without re-simulating. For the configured weights and for each
weight scaled x0 / x0.5 / x2 (one at a time) it reports, per (scenario, seed) group:
  spearman  rank correlation between total reward and -mean delay across the five policies
  top_agree whether the policy with the highest reward also has the lowest mean delay
Uses train scenarios only. Output: runs/rl/reward_sensitivity.json
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from camp_crl.sim.network import ROOT

POLICIES = ("camp_prior", "always_switch", "hold_5s", "hold_15s", "random")
TERMS = ("queue", "throughput", "max_wait", "fairness", "spillback")  # per-second terms (sign applied below)
SIGN = {"queue": -1.0, "throughput": 1.0, "max_wait": -1.0, "fairness": -1.0, "spillback": -1.0}


def rollout(policy: str, scenario: str, seed: int) -> dict:
    import torch

    from camp_crl.config import APPROACHES, load_config
    from camp_crl.control.controllers import CampRuleController
    from camp_crl.rl.config import load_rl_config
    from camp_crl.rl.env import TrafficSignalEnv, _clip01
    from camp_crl.rl.fusion import FusedPolicy
    from camp_crl.tuning import with_tuned_fixed_time

    torch.set_num_threads(1)
    app = with_tuned_fixed_time(load_config(ROOT / "configs" / "default.yaml"))
    rl = load_rl_config(ROOT / "configs" / "rl.yaml")
    sums = {t: 0.0 for t in TERMS} | {"switches": 0.0, "interventions": 0.0}

    class ComponentEnv(TrafficSignalEnv):
        def _second_reward(self, snap: dict) -> float:
            truth = snap["truth"]["approaches"]
            arrived = snap["kpi"]["arrived_total"]
            waits = [truth[a]["max_wait_s"] for a in APPROACHES]
            sums["queue"] += snap["kpi"]["total_queue_veh"] / (4 * 40.0)
            sums["throughput"] += (arrived - self._arrived) / 2.0
            sums["max_wait"] += _clip01(max(waits) / 120.0)
            sums["fairness"] += _clip01(float(np.std(waits)) / 60.0)
            sums["spillback"] += 1.0 if snap["kpi"]["spillback"] else 0.0
            return super()._second_reward(snap)

    env = ComponentEnv(app, rl, scenarios=[scenario], obs_mode="camp")
    rng = np.random.default_rng(seed)
    prior, rule = FusedPolicy("camp"), CampRuleController(app)
    fixed = {"always_switch": [1.0, 0.0], "hold_5s": [-1.0, -1.0 + 2 * 4 / 14], "hold_15s": [-1.0, 1.0]}
    try:
        obs, _ = env.reset(seed=seed, options={"scenario": scenario, "sumo_seed": seed})
        done, info, total = False, {}, 0.0
        while not done:
            if policy == "camp_prior":
                a = prior.act(env, obs, rule)
            elif policy == "random":
                a = rng.uniform(-1, 1, 2)
            else:
                a = np.array(fixed[policy])
            obs, r, term, trunc, info = env.step(np.asarray(a, np.float32))
            total += r
            sums["switches"] += float(info["switched"])
            sums["interventions"] += info["interventions"]
            done = term or trunc
    finally:
        env.close()
    m = info["episode_metrics"]
    return {"policy": policy, "scenario": scenario, "seed": seed, "reward_logged": total, "components": sums,
            "delay_mean_s": m["delay_mean_s"], "max_wait_s": m["max_wait_s"], "throughput": m["throughput_completed"]}


def total_reward(comp: dict, w: dict) -> float:
    per_sec = sum(SIGN[t] * w[t] * comp[t] for t in TERMS)
    return per_sec * w["per_second_scale"] - w["switch"] * comp["switches"] - w["intervention"] * comp["interventions"]


def spearman(x: list[float], y: list[float]) -> float:
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1]) if np.std(rx) and np.std(ry) else float("nan")


def analyse(rows: list[dict], base: dict) -> list[dict]:
    variants = [("configured", dict(base))]
    for k in TERMS + ("switch", "intervention"):
        for f in (0.0, 0.5, 2.0):
            variants.append((f"{k} x{f:g}", {**base, k: base[k] * f}))
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["scenario"], r["seed"]), []).append(r)
    out = []
    for name, w in variants:
        rhos, agree = [], []
        for g in groups.values():
            rewards = [total_reward(r["components"], w) for r in g]
            neg_delay = [-r["delay_mean_s"] for r in g]
            rhos.append(spearman(rewards, neg_delay))
            agree.append(int(np.argmax(rewards)) == int(np.argmax(neg_delay)))
        out.append({"weights": name, "spearman_mean": float(np.nanmean(rhos)), "spearman_min": float(np.nanmin(rhos)),
                    "top_agree_rate": float(np.mean(agree)), "groups": len(groups)})
    return out


def main() -> None:
    from camp_crl.config import load_config
    from camp_crl.rl.config import load_rl_config

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    app = load_config(ROOT / "configs" / "default.yaml")
    rl = load_rl_config(ROOT / "configs" / "rl.yaml")
    rl.check_against(app)
    jobs = [(pol, sc, s) for sc in rl.train_scenarios for s in args.seeds for pol in POLICIES]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(rollout, *zip(*jobs)))
    base = rl.reward.model_dump()
    # consistency check: recomputed total must equal the reward the environment actually returned
    max_err = max(abs(total_reward(r["components"], base) - r["reward_logged"]) for r in rows)
    result = {"rows": rows, "analysis": analyse(rows, base), "recompute_max_abs_error": max_err,
              "scenarios": rl.train_scenarios, "seeds": args.seeds, "policies": list(POLICIES)}
    out = ROOT / "runs" / "rl" / "reward_sensitivity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"recompute check: max |error| = {max_err:.2e}")
    print(f"{'weights':<22}{'spearman mean':>15}{'min':>8}{'top agree':>11}")
    for a in result["analysis"]:
        print(f"{a['weights']:<22}{a['spearman_mean']:>15.2f}{a['spearman_min']:>8.2f}{a['top_agree_rate']:>11.0%}")
    print("\nMean delay by policy (s):")
    for pol in POLICIES:
        d = [r["delay_mean_s"] for r in rows if r["policy"] == pol]
        print(f"  {pol:<15}{np.mean(d):8.2f}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
