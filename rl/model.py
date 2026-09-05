from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical

from .observation import (
    BULLET_FEATURES,
    MAP_CHANNELS,
    MAX_MAP_CELLS,
    SELF_FEATURES,
    TANK_FEATURES,
)


ACTION_COUNT = 18
MAP_EMBEDDING = 1024
STATE_EMBEDDING = 512


def encode_actions(actions: torch.Tensor) -> torch.Tensor:
    """Convert (throttle, steer, fire) to one of 18 joint actions."""
    # 将三个离散控制量合并成单个联合动作编号。
    return actions[..., 0] * 6 + actions[..., 1] * 2 + actions[..., 2]


def decode_actions(actions: torch.Tensor) -> torch.Tensor:
    """Convert a joint action index back to (throttle, steer, fire)."""
    # 将联合动作编号还原成油门、转向和开火三个控制量。
    return torch.stack((actions // 6, actions % 6 // 2, actions % 2), dim=-1)


def _masked_sum_max(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Order-independent aggregation for a padded entity set."""
    # 忽略补齐位置并汇总所有真实实体的总体与最显著特征。
    valid = mask.bool().unsqueeze(-1)
    summed = (features * valid).sum(dim=1)
    maximum = features.masked_fill(~valid, -torch.inf).max(dim=1).values
    any_valid = valid.any(dim=1)
    maximum = torch.where(any_valid, maximum, torch.zeros_like(maximum))
    return torch.cat((summed, maximum), dim=-1)


def _block(in_features: int, out_features: int) -> nn.Sequential:
    # 创建包含线性层、归一化和激活函数的基础网络块。
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.LayerNorm(out_features),
        nn.SiLU(),
    )


class TankActorCritic(nn.Module):
    """Actor-critic whose inputs are all recoverable from rendered game frames."""

    def __init__(self) -> None:
        # 初始化地图、自身、敌人、子弹及策略价值分支的全部网络层。
        super().__init__()

        # 96 -> 48 -> 24 -> 12. The last feature location corresponds roughly
        # to one maze cell, so the global wall layout is not pooled away.
        self.map_cnn = nn.Sequential(
            nn.Conv2d(MAP_CHANNELS, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(64, 32, 3, stride=2, padding=1), nn.SiLU(),
        )
        flattened_map_features = 32 * MAX_MAP_CELLS * MAX_MAP_CELLS
        self.map_mlp = nn.Sequential(
            _block(flattened_map_features, 2048),
            _block(2048, MAP_EMBEDDING),
        )

        self.self_mlp = nn.Sequential(_block(SELF_FEATURES, 32), _block(32, 32))
        self.tank_mlp = nn.Sequential(_block(TANK_FEATURES, 32), _block(32, 32))
        self.bullet_mlp = nn.Sequential(_block(BULLET_FEATURES, 64), _block(64, 64))

        # map 1024 + self 32 + tanks (sum+max) 64 + bullets (sum+max) 128
        self.fusion = nn.Sequential(_block(1248, 768), _block(768, STATE_EMBEDDING))
        self.policy_head = nn.Sequential(nn.Linear(STATE_EMBEDDING, 256), nn.SiLU(), nn.Linear(256, ACTION_COUNT))
        self.value_head = nn.Sequential(nn.Linear(STATE_EMBEDDING, 256), nn.SiLU(), nn.Linear(256, 1))
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        # 使用正交初始化提高强化学习训练初期的数值稳定性。
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.policy_head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)

    def encode(
        self,
        map_tensor: torch.Tensor,
        self_vector: torch.Tensor,
        tanks: torch.Tensor,
        tank_mask: torch.Tensor,
        bullets: torch.Tensor,
        bullet_mask: torch.Tensor,
    ) -> torch.Tensor:
        # 分别编码四类输入并融合成固定的512维状态特征。
        map_feature = self.map_mlp(self.map_cnn(map_tensor.float()).flatten(1))
        self_feature = self.self_mlp(self_vector.float())
        tank_feature = _masked_sum_max(self.tank_mlp(tanks.float()), tank_mask)
        bullet_feature = _masked_sum_max(self.bullet_mlp(bullets.float()), bullet_mask)
        return self.fusion(
            torch.cat((map_feature, self_feature, tank_feature, bullet_feature), dim=-1)
        )

    def action_logits(
        self,
        map_tensor: torch.Tensor,
        self_vector: torch.Tensor,
        tanks: torch.Tensor,
        tank_mask: torch.Tensor,
        bullets: torch.Tensor,
        bullet_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 根据融合状态同时计算18类动作分数和状态价值。
        feature = self.encode(
            map_tensor, self_vector, tanks, tank_mask, bullets, bullet_mask
        )
        return self.policy_head(feature), self.value_head(feature).squeeze(-1)

    def get_action_and_value(
        self,
        map_tensor: torch.Tensor,
        self_vector: torch.Tensor,
        tanks: torch.Tensor,
        tank_mask: torch.Tensor,
        bullets: torch.Tensor,
        bullet_mask: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 从策略分布选择动作并返回PPO训练所需的统计量。
        logits, value = self.action_logits(
            map_tensor, self_vector, tanks, tank_mask, bullets, bullet_mask
        )
        distribution = Categorical(logits=logits)
        if action is not None:
            joint_action = encode_actions(action.long())
        elif deterministic:
            joint_action = distribution.probs.argmax(dim=-1)
        else:
            joint_action = distribution.sample()
        return (
            decode_actions(joint_action),
            distribution.log_prob(joint_action),
            distribution.entropy(),
            value,
        )


def load_actor_critic_state(model: TankActorCritic, state: dict) -> None:
    """Load only checkpoints created for this observable-input architecture."""
    # 严格加载参数并拒绝与当前模型结构不兼容的旧检查点。
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            "checkpoint is incompatible with the observable-input 1024/512 model"
        ) from error
