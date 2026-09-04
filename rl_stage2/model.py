from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical

from .observation import BULLET_FEATURES, MAP_CHANNELS, SELF_FEATURES, TANK_FEATURES


def _masked_mean(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """对可变长集合做掩码平均；集合为空时返回全零。"""
    weights = mask.unsqueeze(-1)
    total = weights.sum(dim=1).clamp(min=1e-6)
    return (features * weights).sum(dim=1) / total


class TankActorCritic(nn.Module):
    """CNN 读地图，坦克和子弹各自用集合编码器，再融合输出动作和价值。"""

    def __init__(self) -> None:
        super().__init__()
        self.map_cnn = nn.Sequential(
            nn.Conv2d(MAP_CHANNELS, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(),
        )
        self.self_mlp = nn.Sequential(
            nn.Linear(SELF_FEATURES, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.tank_mlp = nn.Sequential(
            nn.Linear(TANK_FEATURES, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.bullet_mlp = nn.Sequential(
            nn.Linear(BULLET_FEATURES, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(nn.Linear(256 + 64 + 64 + 64, 256), nn.ReLU())
        self.throttle_head = nn.Linear(256, 3)
        self.steer_head = nn.Linear(256, 3)
        self.fire_head = nn.Linear(256, 2)
        self.value_head = nn.Linear(256, 1)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """使用适合 PPO 的正交初始化，并让初始动作概率接近均匀。"""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(module.bias)
        for head in (self.throttle_head, self.steer_head, self.fire_head):
            nn.init.orthogonal_(head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def encode(
        self,
        map_tensor: torch.Tensor,
        self_vector: torch.Tensor,
        tanks: torch.Tensor,
        tank_mask: torch.Tensor,
        bullets: torch.Tensor,
        bullet_mask: torch.Tensor,
    ) -> torch.Tensor:
        """把地图、自身、其他坦克集合和子弹集合融合成 256 维状态特征。"""
        map_feature = self.map_cnn(map_tensor.float())
        self_feature = self.self_mlp(self_vector.float())
        tank_feature = _masked_mean(self.tank_mlp(tanks.float()), tank_mask.float())
        bullet_feature = _masked_mean(self.bullet_mlp(bullets.float()), bullet_mask.float())
        return self.fusion(torch.cat((map_feature, self_feature, tank_feature, bullet_feature), dim=1))

    def action_logits(
        self,
        map_tensor: torch.Tensor,
        self_vector: torch.Tensor,
        tanks: torch.Tensor,
        tank_mask: torch.Tensor,
        bullets: torch.Tensor,
        bullet_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回三个动作分支未经采样的 logits 和状态价值。"""
        feature = self.encode(map_tensor, self_vector, tanks, tank_mask, bullets, bullet_mask)
        return (
            self.throttle_head(feature),
            self.steer_head(feature),
            self.fire_head(feature),
            self.value_head(feature).squeeze(1),
        )

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
        """采样或评估三分支动作，并返回动作、联合对数概率、熵和状态价值。"""
        throttle_logits, steer_logits, fire_logits, value = self.action_logits(
            map_tensor, self_vector, tanks, tank_mask, bullets, bullet_mask
        )
        distributions = (
            Categorical(logits=throttle_logits),
            Categorical(logits=steer_logits),
            Categorical(logits=fire_logits),
        )
        if action is None:
            if deterministic:
                action = torch.stack([distribution.probs.argmax(dim=1) for distribution in distributions], dim=1)
            else:
                action = torch.stack([distribution.sample() for distribution in distributions], dim=1)
        log_probability = sum(distribution.log_prob(action[:, index]) for index, distribution in enumerate(distributions))
        entropy = sum(distribution.entropy() for distribution in distributions)
        return action, log_probability, entropy, value
