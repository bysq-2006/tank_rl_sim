from __future__ import annotations

from dataclasses import dataclass

import torch

from rl.curriculum import StageConfig
from rl.envs import TankRLEnv
from rl.training.rollout import stack_observations


@dataclass(frozen=True)
class EvaluationResult:
    """一轮固定种子评估的汇总结果。"""

    results: list[str]
    mean_reward: float

    @property
    def win_rate(self) -> float:
        # 计算本轮评估的获胜比例。
        return sum(result == "win" for result in self.results) / max(len(self.results), 1)

    @property
    def timeout_rate(self) -> float:
        # 计算本轮评估的超时比例。
        return sum(result == "timeout" for result in self.results) / max(len(self.results), 1)


@torch.no_grad()
def evaluate_policy(
    model,
    stage: StageConfig,
    games: int,
    seed_start: int,
    device: torch.device,
    action_repeat: int = 2,
) -> EvaluationResult:
    # 在固定且未训练的地图种子上使用确定性动作评估策略。
    was_training = model.training
    model.eval()
    results: list[str] = []
    rewards: list[float] = []
    for game_index in range(games):
        env = TankRLEnv(action_repeat=action_repeat)
        observation = env.reset(stage, seed_start + game_index)
        episode_reward = 0.0
        done = False
        while not done:
            inputs = stack_observations([observation], device)
            action, _, _, _ = model.get_action_and_value(*inputs, deterministic=True)
            observation, reward, done, info = env.step(action[0].cpu().numpy())
            episode_reward += reward
        results.append(info["result"])
        rewards.append(episode_reward)
    if was_training:
        model.train()
    return EvaluationResult(results, sum(rewards) / max(len(rewards), 1))
