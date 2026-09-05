from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .rollout import RolloutBatch


@dataclass(frozen=True)
class PPOConfig:
    """PPO更新使用的超参数。"""

    gamma: float = 0.995
    gae_lambda: float = 0.99
    clip_coefficient: float = 0.20
    value_coefficient: float = 0.50
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.50
    target_kl: float = 0.02
    update_epochs: int = 4
    minibatch_size: int = 512


def _advantages(rollout: RolloutBatch, config: PPOConfig) -> tuple[torch.Tensor, torch.Tensor]:
    # 使用广义优势估计从奖励序列计算优势和价值目标。
    advantage = torch.zeros_like(rollout.rewards)
    last_gae = torch.zeros_like(rollout.next_value)
    for step in reversed(range(rollout.rewards.shape[0])):
        if step == rollout.rewards.shape[0] - 1:
            next_nonterminal = 1.0 - rollout.next_done
            next_value = rollout.next_value
        else:
            next_nonterminal = 1.0 - rollout.dones[step + 1]
            next_value = rollout.values[step + 1]
        delta = rollout.rewards[step] + config.gamma * next_value * next_nonterminal - rollout.values[step]
        last_gae = delta + config.gamma * config.gae_lambda * next_nonterminal * last_gae
        advantage[step] = last_gae
    return advantage, advantage + rollout.values


def _flatten(tensor: torch.Tensor) -> torch.Tensor:
    # 将时间和并行环境两个批次维度合并为一个维度。
    return tensor.reshape((-1,) + tensor.shape[2:])


def ppo_update(model, optimizer, rollout: RolloutBatch, config: PPOConfig) -> dict[str, float]:
    # 对同一批轨迹执行多轮裁剪策略和价值函数更新。
    advantages, returns = _advantages(rollout, config)
    flat_observations = tuple(_flatten(item) for item in rollout.observations)
    flat_actions = _flatten(rollout.actions)
    flat_old_log_probabilities = _flatten(rollout.log_probabilities)
    flat_advantages = _flatten(advantages)
    flat_returns = _flatten(returns)
    batch_size = flat_actions.shape[0]
    indices = np.arange(batch_size)
    metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
    updates = 0

    normalized_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)
    stop_early = False
    for _ in range(config.update_epochs):
        np.random.shuffle(indices)
        for start in range(0, batch_size, config.minibatch_size):
            # 将随机小批量索引放到模型所在设备以同时兼容CPU和CUDA。
            batch_indices = torch.as_tensor(
                indices[start:start + config.minibatch_size],
                device=flat_actions.device,
            )
            inputs = tuple(item[batch_indices] for item in flat_observations)
            _, new_log_probability, entropy, new_value = model.get_action_and_value(
                *inputs, action=flat_actions[batch_indices]
            )
            log_ratio = new_log_probability - flat_old_log_probabilities[batch_indices]
            ratio = log_ratio.exp()
            batch_advantage = normalized_advantages[batch_indices]
            policy_loss = torch.max(
                -batch_advantage * ratio,
                -batch_advantage * ratio.clamp(1.0 - config.clip_coefficient, 1.0 + config.clip_coefficient),
            ).mean()
            value_loss = 0.5 * (new_value - flat_returns[batch_indices]).pow(2).mean()
            loss = policy_loss + config.value_coefficient * value_loss - config.entropy_coefficient * entropy.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            approximate_kl = ((ratio - 1.0) - log_ratio).mean().detach()
            metrics["policy_loss"] += float(policy_loss.detach())
            metrics["value_loss"] += float(value_loss.detach())
            metrics["entropy"] += float(entropy.mean().detach())
            metrics["approx_kl"] += float(approximate_kl)
            updates += 1
            if float(approximate_kl) > config.target_kl:
                stop_early = True
                break
        if stop_early:
            break
    return {key: value / max(updates, 1) for key, value in metrics.items()}
