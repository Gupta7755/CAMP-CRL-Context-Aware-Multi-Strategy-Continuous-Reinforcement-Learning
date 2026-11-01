"""Fusion gate, CAMP prior and ablation tests (Phase 2)."""

import numpy as np
import pytest

from camp_crl.rl.fusion import FusedPolicy, GateParams, ObsStats, critic_uncertainty, fuse, gate_alpha, sigmoid


class TestGate:
    def test_conservative_initialisation(self):
        assert gate_alpha(np.zeros(4), GateParams()) == pytest.approx(sigmoid(-1.0))
        assert gate_alpha(np.zeros(4), GateParams()) < 0.3

    @pytest.mark.parametrize("i", range(4))
    def test_worse_conditions_never_increase_rl_trust(self, i):
        p = GateParams()
        for level in (0.25, 0.5, 1.0):
            u_lo, u_hi = np.full(4, 0.2), np.full(4, 0.2)
            u_hi[i] = level
            u_lo[i] = level - 0.25
            assert gate_alpha(u_hi, p) < gate_alpha(u_lo, p)

    def test_missing_signal_is_worst_case(self):
        p = GateParams()
        assert gate_alpha(np.array([np.nan, 0, 0, 0]), p) == pytest.approx(gate_alpha(np.array([1, 0, 0, 0]), p))

    def test_positive_weight_rejected(self):
        with pytest.raises(ValueError):
            GateParams(w_anomaly=0.5)
        with pytest.raises(ValueError):
            GateParams(fixed_alpha=1.5)

    def test_out_of_range_inputs_clipped(self):
        p = GateParams()
        assert gate_alpha(np.array([5, 5, 5, 5]), p) == pytest.approx(gate_alpha(np.ones(4), p))
        assert 0.0 < gate_alpha(np.ones(4), p) < gate_alpha(np.zeros(4), p) < 1.0


class TestFuse:
    def test_endpoints_and_units(self):
        a_c, a_r = np.array([-0.4, -1.0]), np.array([0.8, 0.6])
        assert np.allclose(fuse(a_c, a_r, 0.0), a_c)
        assert np.allclose(fuse(a_c, a_r, 1.0), a_r)
        assert np.allclose(fuse(a_c, a_r, 0.5), [0.2, -0.2])
        assert np.allclose(fuse(a_c, a_r, 7.0), a_r)  # alpha clipped
        assert np.all(np.abs(fuse(np.array([1, 1]), np.array([1, 1]), 0.3)) <= 1)


class TestObsStats:
    def test_novelty_and_roundtrip(self, tmp_path):
        rng = np.random.default_rng(0)
        st = ObsStats()
        st.update(rng.normal(0.5, 0.05, size=(500, 3)))
        assert st.novelty(np.array([0.5, 0.5, 0.5])) < 0.1
        assert st.novelty(np.array([1.0, 0.0, 1.0])) == 1.0
        st.save(tmp_path / "s.json")
        st2 = ObsStats.load(tmp_path / "s.json")
        assert st2.n == 500 and np.allclose(st2.mean, st.mean)
        assert ObsStats().novelty(np.zeros(3)) == 1.0  # no statistics -> unfamiliar


class _ConstModel:
    def __init__(self, a):
        self.a = np.asarray(a, np.float32)

    def predict(self, obs, deterministic=True):
        return self.a, None


def test_policy_requires_model_except_camp_mode():
    with pytest.raises(ValueError):
        FusedPolicy("gated")
    FusedPolicy("camp")


def test_critic_uncertainty_bounds():
    import gymnasium as gym
    from stable_baselines3 import DDPG, TD3

    class E(gym.Env):
        observation_space = gym.spaces.Box(0, 1, (3,), np.float32)
        action_space = gym.spaces.Box(-1, 1, (2,), np.float32)
    td3 = TD3("MlpPolicy", E(), seed=0, device="cpu")
    ddpg = DDPG("MlpPolicy", E(), seed=0, device="cpu")
    u = critic_uncertainty(td3, np.full(3, 0.5), np.zeros(2))
    assert 0.0 <= u <= 1.0
    assert critic_uncertainty(ddpg, np.full(3, 0.5), np.zeros(2)) == 0.0


@pytest.fixture(scope="module")
def rl():
    from camp_crl.rl.config import load_rl_config
    from camp_crl.sim.network import ROOT
    return load_rl_config(ROOT / "configs" / "rl.yaml")


@pytest.mark.sumo
class TestAblationInEnvironment:
    def _run(self, cfg, rl, policy, scenario="ns_dominant", seed=4, duration=600):
        from camp_crl.control.controllers import CampRuleController
        from camp_crl.rl.env import TrafficSignalEnv
        env = TrafficSignalEnv(cfg, rl, scenarios=[scenario], obs_mode="camp", duration_s=duration)
        try:
            obs, _ = env.reset(seed=seed, options={"scenario": scenario, "sumo_seed": seed})
            rule, done, info = CampRuleController(cfg), False, {}
            while not done:
                obs, _, t, tr, info = env.step(policy.act(env, obs, rule))
                done = t or tr
            return info["episode_metrics"]
        finally:
            env.close()

    def test_camp_mode_executes_prior_and_tracks_rule_controller(self, cfg, rl, tmp_path):
        from camp_crl.runner import run_episode
        pol = FusedPolicy("camp", _ConstModel([1.0, 1.0]))  # model must be ignored
        m = self._run(cfg, rl, pol)
        assert all(r["alpha"] == 0.0 and r["a_mix"] == r["a_camp"] for r in pol.log)
        short = cfg.model_copy(update={"simulation": cfg.simulation.model_copy(update={"duration_s": 600, "warmup_s": 120})})
        ref = run_episode(short, "camp_rule", "ns_dominant", 4, tmp_path).metrics
        # Decisions lag the rule controller by one 1-s step, so results are close, not identical.
        assert abs(m["delay_mean_s"] - ref["delay_mean_s"]) / ref["delay_mean_s"] < 0.25

    def test_rl_mode_executes_model_action(self, cfg, rl):
        pol = FusedPolicy("rl", _ConstModel([1.0, -1.0]))  # always switch
        m = self._run(cfg, rl, pol, duration=300)
        assert all(r["alpha"] == 1.0 and r["a_mix"] == [1.0, -1.0] for r in pol.log)
        assert m["phase_switches"] >= len(pol.log) - 1

    def test_gated_mode_logs_bounded_alpha_and_signals(self, cfg, rl):
        pol = FusedPolicy("gated", _ConstModel([-1.0, 1.0]), ObsStats())
        self._run(cfg, rl, pol, scenario="noisy_balanced", duration=300)
        assert pol.log and all(0.0 < r["alpha"] < 1.0 and len(r["u"]) == 4 for r in pol.log)
        assert all(r["u"][2] == 1.0 for r in pol.log)  # empty stats -> maximal novelty -> low trust


@pytest.mark.sumo
def test_rl_evaluation_merges_with_baselines(cfg, rl, tmp_path):
    """End-to-end: tiny baseline experiment + tiny trained TD3-CAMP -> merged paired evaluation."""
    import json
    from camp_crl.config import AppConfig
    from camp_crl.experiments import ExperimentParams, ExperimentStore, run_experiment
    from camp_crl.rl.evaluate import evaluate
    from camp_crl.rl.train import train

    data = cfg.model_dump(mode="json")
    data["simulation"].update(duration_s=150, warmup_s=30)
    data["fixed_time_tuned"] = None
    small = AppConfig.model_validate(data)
    store = ExperimentStore(tmp_path)
    params = ExperimentParams(["fixed_time", "camp_rule"], ["ns_dominant"], [1, 2], "fixed_time", kind="evaluation", workers=2)
    h = store.create(params)
    run_experiment(small, params, store, h)
    run = train("td3", "camp", seed=1, timesteps=40, n_envs=1, duration_s=150, checkpoint_every=1000,
                validate_every=1000, out_root=tmp_path / "rl", app=small,
                rl=rl.model_copy(update={"validation_scenarios": ["ns_dominant"]}))
    out = evaluate([run], h.id, workers=2, out_root=tmp_path)
    summary = json.loads((out / "summary.json").read_text())
    ctrls = set(summary["controllers"])
    assert {"fixed_time", "camp_rule", "td3_camp", "camp_crl_camp", "camp_crl_fixed", "camp_crl_gated"} <= ctrls
    paired = summary["scenarios"]["ns_dominant"]["paired_vs_baseline"]["camp_crl_gated"]["delay_mean_s"]
    assert paired["n_pairs"] == 2
    rows = json.loads((out / "rl_rollouts.json").read_text())
    gated = [r for r in rows if r["fusion_mode"] == "gated"]
    assert gated and all(0 < r["alpha_mean"] < 1 for r in gated)
    assert all(r["alpha_mean"] == 0.0 for r in rows if r["fusion_mode"] == "camp")
    assert store.manifest(out.name)["params"]["kind"] == "evaluation"
    assert 0 < summary["fusion_alpha"]["camp_crl_gated"]["ns_dominant"] < 1
    assert summary["fusion_alpha"]["camp_crl_camp"]["ns_dominant"] == 0.0
