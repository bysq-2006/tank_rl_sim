from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn
from torch.distributions import Categorical

from .observation import BULLET_FEATURES, MAP_CHANNELS, SELF_FEATURES, TANK_FEATURES


ACTION_COUNT = 18


def encode_actions(actions: torch.Tensor) -> torch.Tensor:
    """(油门, 转向, 开火) 转为单个0..17联合动作。"""
    return actions[..., 0] * 6 + actions[..., 1] * 2 + actions[..., 2]


def decode_actions(actions: torch.Tensor) -> torch.Tensor:
    """0..17联合动作转回游戏使用的三个控制值。"""
    return torch.stack((actions // 6, actions % 6 // 2, actions % 2), dim=-1)


def _sample_map(feature_map: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """在实体的连续世界位置双线性读取墙体特征。"""
    single = positions.ndim == 2
    if single:
        positions = positions.unsqueeze(1)
    sampled = functional.grid_sample(
        feature_map,
        positions.unsqueeze(2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).squeeze(-1).transpose(1, 2)
    return sampled.squeeze(1) if single else sampled


def _masked_sum_max(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """顺序无关的集合汇总；sum保留数量，max保留最显著实体。"""
    valid = mask.bool().unsqueeze(-1)
    summed = (features * valid).sum(dim=1)
    maximum = features.masked_fill(~valid, -torch.inf).max(dim=1).values
    maximum = torch.where(mask.bool().any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum))
    return torch.cat((summed, maximum), dim=1)


class TankActorCritic(nn.Module):
    """精确墙拓扑与连续实体对齐的小型前馈 Actor-Critic。"""

    def __init__(self) -> None:
        super().__init__()
        self.map_cnn = nn.Sequential(
            nn.Conv2d(MAP_CHANNELS, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=4, dilation=4), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
        )
        self.map_mlp = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        self.self_mlp = nn.Sequential(
            nn.Linear(SELF_FEATURES + 32, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()
        )
        self.tank_mlp = nn.Sequential(
            nn.Linear(TANK_FEATURES + 32, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU()
        )
        self.bullet_mlp = nn.Sequential(
            nn.Linear(BULLET_FEATURES + 32, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(320, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU()
        )
        self.policy_head = nn.Linear(128, ACTION_COUNT)
        self.value_head = nn.Linear(128, 1)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def encode(
        self,
        map_tensor: torch.Tensor,
        self_vector: torch.Tensor,
        self_pos: torch.Tensor,
        tanks: torch.Tensor,
        tank_pos: torch.Tensor,
        tank_mask: torch.Tensor,
        bullets: torch.Tensor,
        bullet_pos: torch.Tensor,
        bullet_mask: torch.Tensor,
    ) -> torch.Tensor:
        wall_features = self.map_cnn(map_tensor.float())
        valid = map_tensor[:, 4:5].float().flatten(2)
        flat_map = wall_features.flatten(2)
        map_mean = (flat_map * valid).sum(2) / valid.sum(2).clamp(min=1.0)
        map_max = flat_map.masked_fill(valid <= 0.0, -torch.inf).max(2).values
        map_feature = self.map_mlp(torch.cat((map_mean, map_max), dim=1))

        self_feature = self.self_mlp(
            torch.cat((self_vector.float(), _sample_map(wall_features, self_pos)), dim=1)
        )
        tank_entities = self.tank_mlp(
            torch.cat((tanks.float(), _sample_map(wall_features, tank_pos)), dim=2)
        )
        bullet_entities = self.bullet_mlp(
            torch.cat((bullets.float(), _sample_map(wall_features, bullet_pos)), dim=2)
        )
        return self.fusion(
            torch.cat(
                (
                    map_feature,
                    self_feature,
                    _masked_sum_max(tank_entities, tank_mask),
                    _masked_sum_max(bullet_entities, bullet_mask),
                ),
                dim=1,
            )
        )

    def action_logits(self, *observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature = self.encode(*observation)
        return self.policy_head(feature), self.value_head(feature).squeeze(1)

    def get_action_and_value(
        self,
        *observation: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.action_logits(*observation)
        distribution = Categorical(logits=logits)
        if action is not None:
            joint_action = encode_actions(action.long())
        elif deterministic:
            joint_action = distribution.probs.argmax(1)
        else:
            joint_action = distribution.sample()
        return decode_actions(joint_action), distribution.log_prob(joint_action), distribution.entropy(), value


def load_actor_critic_state(model: TankActorCritic, state: dict) -> None:
    """新结构有意不兼容旧观察和旧三动作头。"""
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise RuntimeError("checkpoint is incompatible with exact-map joint-action model") from error
