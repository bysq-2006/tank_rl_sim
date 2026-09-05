from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from rl.curriculum import CurriculumManager
from rl.envs import TankRLEnv
from rl.observation import Observation


OBSERVATION_KEYS = ("map", "self", "tanks", "tank_mask", "bullets", "bullet_mask")


def stack_observations(observations: list[Observation], device: torch.device) -> tuple[torch.Tensor, ...]:
    # 将多个环境的NumPy观察按模型参数顺序堆叠成张量。
    return tuple(
        torch.as_tensor(np.stack([observation[key] for observation in observations]), device=device)
        for key in OBSERVATION_KEYS
    )


@dataclass
class RolloutBatch:
    """一次PPO更新所需的时序批次。"""

    observations: tuple[torch.Tensor, ...]
    actions: torch.Tensor
    log_probabilities: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    next_value: torch.Tensor
    next_done: torch.Tensor
    next_observations: list[Observation]
    episode_results: list[dict]


@torch.no_grad()
def collect_rollout(
    model,
    envs: list[TankRLEnv],
    observations: list[Observation],
    curriculum: CurriculumManager,
    steps: int,
    device: torch.device,
    next_episode_seed,
    opponent_pool=None,
) -> RolloutBatch:
    # 使用当前策略同步采集多个环境的固定长度轨迹。
    observation_storage = [[] for _ in OBSERVATION_KEYS]
    actions, log_probabilities, rewards, dones, values = [], [], [], [], []
    episode_results: list[dict] = []
    current_done = torch.zeros(len(envs), dtype=torch.float32, device=device)

    for _ in range(steps):
        model_inputs = stack_observations(observations, device)
        action, log_probability, _, value = model.get_action_and_value(*model_inputs)
        opponent_actions = (
            opponent_pool.batch_actions(envs)
            if opponent_pool is not None
            else [None] * len(envs)
        )
        for storage, tensor in zip(observation_storage, model_inputs):
            storage.append(tensor)
        actions.append(action)
        log_probabilities.append(log_probability)
        values.append(value)
        dones.append(current_done)

        next_observations: list[Observation] = []
        step_rewards = []
        step_dones = []
        for index, env in enumerate(envs):
            observation, reward, done, info = env.step(
                action[index].cpu().numpy(), opponent_action=opponent_actions[index]
            )
            step_rewards.append(reward)
            step_dones.append(float(done))
            if done:
                episode_results.append(info)
                stage = curriculum.sample_stage()
                observation = env.reset(stage, next_episode_seed())
            next_observations.append(observation)
        observations = next_observations
        rewards.append(torch.tensor(step_rewards, dtype=torch.float32, device=device))
        current_done = torch.tensor(step_dones, dtype=torch.float32, device=device)

    next_inputs = stack_observations(observations, device)
    _, next_value = model.action_logits(*next_inputs)
    return RolloutBatch(
        observations=tuple(torch.stack(storage) for storage in observation_storage),
        actions=torch.stack(actions),
        log_probabilities=torch.stack(log_probabilities),
        rewards=torch.stack(rewards),
        dones=torch.stack(dones),
        values=torch.stack(values),
        next_value=next_value,
        next_done=current_done,
        next_observations=observations,
        episode_results=episode_results,
    )
