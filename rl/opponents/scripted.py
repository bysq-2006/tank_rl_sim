from __future__ import annotations

import math
from typing import Protocol

import numpy as np

from core import TankGame


Control = tuple[int, int, int]


class Opponent(Protocol):
    """所有脚本对手共同遵循的动作接口。"""

    def act(self, game: TankGame, tank_id: int) -> Control:
        # 根据当前游戏状态返回油门、转向和开火动作。
        ...


def _find_tank(game: TankGame, tank_id: int):
    # 按编号查找脚本对手当前控制的坦克。
    for tank in game.tanks:
        if tank.tank_id == tank_id:
            return tank
    raise ValueError(f"unknown tank_id {tank_id}")


def _wrapped_angle(angle: float) -> float:
    # 将任意角度转换到负π至正π之间。
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class IdleOpponent:
    def act(self, game: TankGame, tank_id: int) -> Control:
        # 静止对手始终停车且不开火。
        return 1, 1, 0


class RandomMoverOpponent:
    def __init__(self, rng: np.random.Generator) -> None:
        # 初始化随机移动对手的动作保持状态。
        self.rng = rng
        self.remaining = 0
        self.action: Control = (1, 1, 0)

    def act(self, game: TankGame, tank_id: int) -> Control:
        # 每隔若干物理帧重新随机选择移动方向但从不开火。
        if self.remaining <= 0:
            self.action = (int(self.rng.integers(0, 3)), int(self.rng.integers(0, 3)), 0)
            self.remaining = int(self.rng.integers(8, 31))
        self.remaining -= 1
        return self.action


class WeakShooterOpponent(RandomMoverOpponent):
    def __init__(self, rng: np.random.Generator, fire_probability: float) -> None:
        # 初始化随机移动和低频开火参数。
        super().__init__(rng)
        self.fire_probability = fire_probability

    def act(self, game: TankGame, tank_id: int) -> Control:
        # 在随机移动动作上按较低概率追加开火操作。
        throttle, steer, _ = super().act(game, tank_id)
        return throttle, steer, int(self.rng.random() < self.fire_probability)


class ChaserOpponent:
    def __init__(self, rng: np.random.Generator, fire_probability: float) -> None:
        # 初始化追踪型对手的随机数和开火概率。
        self.rng = rng
        self.fire_probability = fire_probability

    def act(self, game: TankGame, tank_id: int) -> Control:
        # 朝最近敌人转向并在基本瞄准后尝试开火。
        own = _find_tank(game, tank_id)
        enemies = [tank for tank in game.tanks if tank.tank_id != tank_id and tank.alive]
        if not own.alive or not enemies:
            return 1, 1, 0
        enemy = min(enemies, key=lambda tank: math.hypot(tank.x - own.x, tank.y - own.y))
        dx, dy = enemy.x - own.x, enemy.y - own.y
        error = _wrapped_angle(math.atan2(dy, dx) - own.heading)
        steer = 2 if error > math.radians(4) else 0 if error < -math.radians(4) else 1
        throttle = 2 if math.hypot(dx, dy) > 2.5 else 1
        aimed = abs(error) < math.radians(10)
        fire = int(aimed and self.rng.random() < self.fire_probability)
        return throttle, steer, fire


def make_opponent(name: str, rng: np.random.Generator, fire_probability: float = 0.0) -> Opponent:
    # 根据课程配置名称创建相应的脚本对手实例。
    if name == "idle":
        return IdleOpponent()
    if name == "random_mover":
        return RandomMoverOpponent(rng)
    if name == "weak_shooter":
        return WeakShooterOpponent(rng, fire_probability)
    if name == "chaser":
        return ChaserOpponent(rng, fire_probability)
    raise ValueError(f"unknown opponent {name}")
