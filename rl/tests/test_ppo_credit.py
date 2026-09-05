import torch

from rl.training.ppo import PPOConfig, _advantages
from rl.training.rollout import RolloutBatch


def test_delayed_hit_reward_reaches_actions_five_seconds_earlier():
    # 验证十二赫兹决策下五秒后的命中奖励仍能明显传回开炮动作。
    steps = 61
    rollout = RolloutBatch(
        observations=(),
        actions=torch.zeros((steps, 1, 3), dtype=torch.long),
        log_probabilities=torch.zeros((steps, 1)),
        rewards=torch.zeros((steps, 1)),
        dones=torch.zeros((steps, 1)),
        values=torch.zeros((steps, 1)),
        next_value=torch.zeros(1),
        next_done=torch.zeros(1),
        next_observations=[],
        episode_results=[],
    )
    rollout.rewards[-1, 0] = 1.0
    advantages, _ = _advantages(rollout, PPOConfig())
    expected = (PPOConfig().gamma * PPOConfig().gae_lambda) ** 60
    assert torch.isclose(advantages[0, 0], torch.tensor(expected), atol=1e-6)
    assert advantages[0, 0] > 0.40
