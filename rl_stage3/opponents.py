from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from core import TankGame
from core.entities import Tank

from .model import TankActorCritic
from .observation import Observation

SCRIPT_NAMES = ("idle", "move", "random", "aim", "chase", "dodge")
OPPONENT_CHOICES = ("self", "mix", "model") + SCRIPT_NAMES


def _wrap_angle(error: float) -> float:
    """把朝向误差折到 (-pi, pi]。"""
    return (error + math.pi) % (2 * math.pi) - math.pi


def _steer_toward(error: float, deadzone: float = 0.12) -> int:
    """steer 0 减小朝向，steer 2 增大朝向。"""
    if error > deadzone:
        return 2
    if error < -deadzone:
        return 0
    return 1


def _enemy_of(game: TankGame, tank_id: int) -> Tank | None:
    """返回另一辆仍存活的坦克；没有则返回 None。"""
    for tank in game.tanks:
        if tank.tank_id != tank_id and tank.alive:
            return tank
    return None


def _aim_error(tank: Tank, enemy: Tank) -> float:
    """炮口指向敌人中心所需的朝向误差。"""
    return _wrap_angle(math.atan2(enemy.y - tank.y, enemy.x - tank.x) - tank.heading)


def script_action(game: TankGame, tank_id: int, kind: str, rng: np.random.Generator) -> tuple[int, int, int]:
    """用规则脚本给一辆坦克生成 (throttle, steer, fire)。"""
    tank = next(item for item in game.tanks if item.tank_id == tank_id)
    if not tank.alive:
        return (1, 1, 0)
    if kind == "idle":
        return (1, 1, 0)
    if kind == "move":
        throttle = 2 if rng.random() < 0.85 else 0
        steer = int(rng.choice((0, 1, 1, 2)))
        return (throttle, steer, 0)
    if kind == "random":
        return (int(rng.integers(0, 3)), int(rng.integers(0, 3)), int(rng.integers(0, 2)))
    enemy = _enemy_of(game, tank_id)
    if enemy is None:
        return (1, 1, 0)
    if kind == "aim":
        error = _aim_error(tank, enemy)
        fire = int(abs(error) < 0.12)
        return (1, _steer_toward(error), fire)
    if kind == "chase":
        error = _aim_error(tank, enemy)
        throttle = 2 if abs(error) < 0.7 else 1
        fire = int(abs(error) < 0.15)
        return (throttle, _steer_toward(error), fire)
    if kind == "dodge":
        return _dodge_action(game, tank, enemy, rng)
    raise ValueError(f"unknown script opponent: {kind}")


def _dodge_action(game: TankGame, tank: Tank, enemy: Tank, rng: np.random.Generator) -> tuple[int, int, int]:
    """侧移躲开飞来的敌弹；没有威胁时沿与视线垂直的方向移动。"""
    vx, vy = 0.0, 0.0
    closest = None
    for bullet in game.bullets:
        if bullet.owner_tank_id == tank.tank_id:
            continue
        dx, dy = tank.x - bullet.x, tank.y - bullet.y
        distance = math.hypot(dx, dy)
        speed = math.hypot(bullet.vx, bullet.vy)
        if distance < 1e-6 or speed < 1e-6:
            continue
        approaching = (bullet.vx * dx + bullet.vy * dy) / (speed * distance)
        if approaching <= 0.1:
            continue
        if closest is None or distance < closest:
            closest = distance
            px, py = -bullet.vy, bullet.vx
            if rng.random() < 0.5:
                px, py = -px, -py
            vx, vy = px, py
    if closest is None:
        dx, dy = tank.x - enemy.x, tank.y - enemy.y
        vx, vy = -dy, dx
        if rng.random() < 0.5:
            vx, vy = -vx, -vy
    error = _wrap_angle(math.atan2(vy, vx) - tank.heading)
    throttle = 2 if abs(error) < 0.8 else 1
    aim_error = _aim_error(tank, enemy)
    fire = int(abs(aim_error) < 0.12)
    return (throttle, _steer_toward(error, deadzone=0.2), fire)


class OpponentController:
    """规则脚本或冻结模型对手；self 对战不使用这个类。"""

    def __init__(
        self,
        kind: str,
        model: TankActorCritic | None = None,
        device: torch.device | None = None,
        seed: int = 0,
    ) -> None:
        if kind not in SCRIPT_NAMES + ("mix", "model"):
            raise ValueError(f"opponent must be one of {OPPONENT_CHOICES}")
        if kind == "model" and model is None:
            raise ValueError("model opponent needs a loaded policy")
        self.kind = kind
        self.model = model
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.env_kinds: list[str] = []

    def reset_env(self, env_index: int) -> str:
        """每局给该环境抽一个脚本名；model 对手保持不变。"""
        name = str(self.rng.choice(SCRIPT_NAMES)) if self.kind == "mix" else self.kind
        while len(self.env_kinds) <= env_index:
            self.env_kinds.append(name)
        self.env_kinds[env_index] = name
        return name

    def action(self, env_index: int, game: TankGame, tank_id: int, observation: Observation) -> np.ndarray:
        """返回对手的离散动作。"""
        kind = self.env_kinds[env_index] if env_index < len(self.env_kinds) else self.kind
        if kind == "model":
            return self._model_action(observation)
        return np.asarray(script_action(game, tank_id, kind, self.rng), dtype=np.int64)

    def _model_action(self, observation: Observation) -> np.ndarray:
        """用冻结策略的 argmax 动作。"""
        assert self.model is not None and self.device is not None
        from .train import _model_batch, stack_observations

        with torch.no_grad():
            actions, _, _, _ = self.model.get_action_and_value(
                *_model_batch(stack_observations([observation]), self.device),
                deterministic=True,
            )
        return actions.cpu().numpy()[0]


def load_opponent_model(path: Path, device: torch.device) -> TankActorCritic:
    """加载一份只用于当对手、不再更新的模型。"""
    if not path.is_file():
        raise SystemExit(f"未找到对手模型：{path.resolve()}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = TankActorCritic().to(device)
    try:
        model.load_state_dict(checkpoint["model_state"])
    except RuntimeError as error:
        raise SystemExit("对手模型结构与当前集合编码器不兼容。") from error
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
