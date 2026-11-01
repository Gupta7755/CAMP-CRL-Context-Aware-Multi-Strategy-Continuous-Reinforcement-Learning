"""Validated Phase-2 RL configuration (separate from the Phase-1 AppConfig)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from camp_crl.config import AppConfig


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ActionConfig(_Strict):
    max_hold_s: float = Field(15.0, ge=1.0, le=60.0)


class RewardWeights(_Strict):
    """Per-second terms (all dimensionless, roughly in [0, 1]):
    q    total halting vehicles / (4 x 40)
    thr  vehicles that completed their trip this second / 2 veh/s
    w    longest current wait on any approach / 120 s, clipped to 1
    f    std of per-approach longest waits / 60 s, clipped to 1
    spill 1 if any approach queue reaches 90 % of its length
    Per decision: the per-second sum times `per_second_scale`, minus `switch` if the agent switched
    and `intervention` per safety-filter override. Delay is not a separate term: summed over time,
    the queue term already integrates waiting (avoids double counting)."""

    queue: float = Field(2.0, ge=0)  # see configs/rl.yaml for the sensitivity-analysis rationale
    throughput: float = Field(0.5, ge=0)
    max_wait: float = Field(0.5, ge=0)
    fairness: float = Field(0.25, ge=0)
    spillback: float = Field(1.0, ge=0)
    switch: float = Field(0.1, ge=0)
    intervention: float = Field(0.5, ge=0)
    per_second_scale: float = Field(0.1, gt=0)


class AlgoConfig(_Strict):
    learning_rate: float = Field(1e-3, gt=0)
    buffer_size: int = Field(200_000, ge=1000)
    batch_size: int = Field(256, ge=8)
    gamma: float = Field(0.98, gt=0, lt=1)
    tau: float = Field(0.005, gt=0, le=1)
    learning_starts: int = Field(2000, ge=0)
    action_noise_std: float = Field(0.2, ge=0)
    net_arch: list[int] = Field(default_factory=lambda: [256, 256])
    policy_delay: int = Field(2, ge=1)


class RLConfig(_Strict):
    action: ActionConfig = ActionConfig()
    reward: RewardWeights = RewardWeights()
    train_scenarios: list[str]
    validation_scenarios: list[str]
    algo: AlgoConfig = AlgoConfig()

    def check_against(self, app: AppConfig) -> None:
        """Training may only use train-split scenarios; selection only validation ones."""
        for name in self.train_scenarios:
            if name not in app.scenarios or app.scenarios[name].split != "train":
                raise ValueError(f"train_scenarios: '{name}' is not a train-split scenario")
        for name in self.validation_scenarios:
            if name not in app.scenarios or app.scenarios[name].split != "validation":
                raise ValueError(f"validation_scenarios: '{name}' is not a validation-split scenario")

    @model_validator(mode="after")
    def _nonempty(self) -> "RLConfig":
        if not self.train_scenarios:
            raise ValueError("train_scenarios must not be empty")
        return self


def load_rl_config(path: str | Path) -> RLConfig:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"RL config not found: {p.resolve()}")
    return RLConfig.model_validate(yaml.safe_load(p.read_text(encoding="utf-8")) or {})
