import math

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from camp_crl.rl.config import RLConfig, load_rl_config
from camp_crl.rl.env import CAMP_DIM, RAW_DIM, TrafficSignalEnv, action_to_request
from camp_crl.sim.network import ROOT


@pytest.fixture(scope="module")
def rl():
    return load_rl_config(ROOT / "configs" / "rl.yaml")


class TestActionMapping:
    def test_switch_and_hold_bounds(self):
        assert action_to_request(np.array([0.5, 0.0]), 0, 15) == (1, 0)
        assert action_to_request(np.array([1.0, -1.0]), 1, 15) == (0, 0)
        assert action_to_request(np.array([-1.0, -1.0]), 0, 15) == (0, 1)
        assert action_to_request(np.array([-1.0, 1.0]), 0, 15) == (0, 15)
        assert action_to_request(np.array([0.0, 0.0]), 1, 15) == (1, 8)  # a0 == 0 holds
        assert action_to_request(np.array([-5.0, 9.0]), 0, 15) == (0, 15)  # clipped

    @pytest.mark.parametrize("bad", [np.array([np.nan, 0.0]), np.array([0.1]), np.array([0.1, 0.2, 0.3])])
    def test_invalid_actions_raise(self, bad):
        with pytest.raises(ValueError):
            action_to_request(bad, 0, 15)


class TestRLConfig:
    def test_scenario_splits_enforced(self, cfg, rl):
        rl.check_against(cfg)
        bad = RLConfig.model_validate({**rl.model_dump(), "train_scenarios": ["balanced", "ew_dominant_heldout"]})
        with pytest.raises(ValueError, match="train-split"):
            bad.check_against(cfg)
        bad = RLConfig.model_validate({**rl.model_dump(), "validation_scenarios": ["balanced"]})
        with pytest.raises(ValueError, match="validation-split"):
            bad.check_against(cfg)

    def test_env_refuses_test_scenarios_for_training(self, cfg, rl):
        bad = RLConfig.model_validate({**rl.model_dump(), "train_scenarios": ["saturated_heldout"]})
        with pytest.raises(ValueError):
            TrafficSignalEnv(cfg, bad)


class TestRewardSigns:
    def _snap(self, queue=0.0, arrived=0, waits=(0, 0, 0, 0), spill=False):
        return {"truth": {"approaches": {a: {"max_wait_s": w} for a, w in zip("NESW", waits)}},
                "kpi": {"total_queue_veh": queue, "arrived_total": arrived, "spillback": spill}}

    def test_components(self, cfg, rl):
        env = TrafficSignalEnv(cfg, rl, duration_s=60)
        try:
            base = env._second_reward(self._snap())
            assert base == 0.0
            env._arrived = 0
            assert env._second_reward(self._snap(queue=40)) < 0
            env._arrived = 0
            assert env._second_reward(self._snap(arrived=2)) > 0
            env._arrived = 0
            assert env._second_reward(self._snap(waits=(120, 0, 0, 0))) < env._second_reward(self._snap(waits=(60, 60, 60, 60)))
            env._arrived = 0
            assert env._second_reward(self._snap(spill=True)) == pytest.approx(-rl.reward.spillback)
        finally:
            env.close()


@pytest.mark.sumo
class TestEnvironment:
    @pytest.mark.parametrize("mode,dim", [("raw", RAW_DIM), ("camp", CAMP_DIM)])
    def test_gymnasium_api(self, cfg, rl, mode, dim):
        env = TrafficSignalEnv(cfg, rl, obs_mode=mode, duration_s=120)
        try:
            assert env.observation_space.shape == (dim,)
            check_env(env, skip_render_check=True)
        finally:
            env.close()

    def test_episode_truncates_with_metrics_and_safety_invariants(self, cfg, rl):
        env = TrafficSignalEnv(cfg, rl, obs_mode="camp", duration_s=240)
        try:
            obs, info = env.reset(seed=3)
            assert env.observation_space.contains(obs) and info["scenario"] in rl.train_scenarios
            rng = np.random.default_rng(0)
            total, steps, kinds = 0.0, 0, set()
            while True:
                obs, r, term, trunc, info = env.step(rng.uniform(-1, 1, size=2).astype(np.float32))
                assert env.observation_space.contains(obs) and math.isfinite(r)
                assert term is False
                total += r
                steps += 1
                if trunc:
                    break
            m = info["episode_metrics"]
            assert m["controller"] == "rl" and m["throughput_completed"] > 0
            # The env only requests at decision points, so the filter never has to hold a min green
            # or reject an invalid phase; only max-green forcing may legitimately occur.
            assert set(m["safety_interventions"]) <= {"max_green_force"}
            assert 10 < steps < 240
        finally:
            env.close()

    def test_seeding_is_reproducible(self, cfg, rl):
        def rollout(seed):
            env = TrafficSignalEnv(cfg, rl, obs_mode="raw", duration_s=150)
            try:
                obs, info = env.reset(seed=seed)
                out = [obs.copy()]
                for k in range(8):
                    obs, r, _, trunc, _ = env.step(np.array([1.0 if k % 3 == 0 else -1.0, 0.0], dtype=np.float32))
                    out.append(obs.copy())
                    out.append(np.array([r]))
                    if trunc:
                        break
                return info, np.concatenate(out)
            finally:
                env.close()
        (i1, a), (i2, b), (i3, c) = rollout(5), rollout(5), rollout(6)
        assert i1 == i2 and np.array_equal(a, b)
        assert i1 != i3 or not np.array_equal(a, c)

    def test_no_orphan_sumo(self, cfg, rl):
        from tests.test_integration_sumo import sumo_process_count
        before = sumo_process_count()
        for _ in range(2):
            env = TrafficSignalEnv(cfg, rl, duration_s=60)
            env.reset(seed=1)
            env.step(np.array([1.0, 0.0], dtype=np.float32))
            env.close()
        assert sumo_process_count() == before


@pytest.mark.sumo
def test_training_smoke_saves_artifacts_and_reloads(cfg, rl, tmp_path):
    """Pipeline check only (no learning happens before learning_starts)."""
    import json
    from stable_baselines3 import TD3
    from camp_crl.rl.train import train
    run = train("td3", "raw", seed=1, timesteps=60, n_envs=1, duration_s=120, checkpoint_every=30,
                validate_every=1000, out_root=tmp_path, app=cfg, rl=rl.model_copy(update={"validation_scenarios": ["ns_dominant"]}))
    m = json.loads((run / "manifest.json").read_text())
    assert m["status"] == "completed" and m["seed"] == 1 and "torch" in m["versions"]
    assert (run / "final_model.zip").is_file() and (run / "best_model.zip").is_file()
    assert any((run / "checkpoints").glob("td3_*_steps.zip"))
    assert (run / "obs_stats.json").is_file()
    rows = (run / "validation.csv").read_text().strip().splitlines()
    assert len(rows) == 2  # header + one final validation (no duplicate)
    model = TD3.load(run / "final_model.zip", device="cpu")
    a, _ = model.predict(np.zeros(27, dtype=np.float32), deterministic=True)
    assert a.shape == (2,) and np.all(np.abs(a) <= 1)


@pytest.mark.sumo
def test_sac_trains_and_saves(cfg, rl, tmp_path):
    from camp_crl.rl.train import train
    run = train("sac", "camp", seed=2, timesteps=40, n_envs=1, duration_s=120, checkpoint_every=1000,
                validate_every=1000, out_root=tmp_path, app=cfg, rl=rl.model_copy(update={"validation_scenarios": ["ns_dominant"]}))
    assert (run / "best_model.zip").is_file() and (run / "obs_stats.json").is_file()


@pytest.mark.sumo
def test_training_is_reproducible_with_same_seed(cfg, rl, tmp_path):
    """Acceptance: same seed + config -> identical learned weights (gradient updates included)."""
    import torch
    from stable_baselines3 import TD3
    from camp_crl.rl.train import train
    quick = rl.model_copy(update={"validation_scenarios": ["ns_dominant"],
                                  "algo": rl.algo.model_copy(update={"learning_starts": 20, "batch_size": 16})})

    def weights(seed, sub):
        run = train("td3", "raw", seed=seed, timesteps=60, n_envs=1, duration_s=120, checkpoint_every=1000,
                    validate_every=1000, out_root=tmp_path / sub, app=cfg, rl=quick)
        return TD3.load(run / "final_model.zip", device="cpu").policy.state_dict()

    a, b, c = weights(7, "a"), weights(7, "b"), weights(8, "c")
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert any(not torch.equal(a[k], c[k]) for k in a)
