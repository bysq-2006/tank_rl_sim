from __future__ import annotations

import math
from typing import TypedDict

import numpy as np

from core import TankGame
from core.entities import Bullet, Tank


MAP_SIZE = 12
MAP_CHANNELS = 5  # 上、右、下、左墙，以及有效格掩码。
MAX_OTHER_TANKS = 2  # 最终最多三辆坦克。
MAX_BULLETS = 15  # 每辆坦克最多五颗。
SELF_FEATURES = 12
TANK_FEATURES = 10
BULLET_FEATURES = 9


class Observation(TypedDict):
    """精确墙拓扑、连续实体状态，以及实体在地图特征图上的查询位置。"""

    map: np.ndarray
    self: np.ndarray
    self_pos: np.ndarray
    tanks: np.ndarray
    tank_pos: np.ndarray
    tank_mask: np.ndarray
    bullets: np.ndarray
    bullet_pos: np.ndarray
    bullet_mask: np.ndarray


def _ego(own: Tank, dx: float, dy: float) -> tuple[float, float]:
    """把世界向量转成自身坐标系：前方、右方。"""
    cos_h, sin_h = math.cos(own.heading), math.sin(own.heading)
    return dx * cos_h + dy * sin_h, -dx * sin_h + dy * cos_h


def _relative_heading(own: Tank, heading: float) -> tuple[float, float]:
    offset = heading - own.heading
    return math.cos(offset), math.sin(offset)


def _fraction(value: float) -> float:
    return value - math.floor(value)


def _map_position(x: float, y: float) -> np.ndarray:
    """转换成 grid_sample(align_corners=False) 使用的连续坐标。"""
    return np.asarray((2.0 * x / MAP_SIZE - 1.0, 2.0 * y / MAP_SIZE - 1.0), dtype=np.float32)


def _wall_map(game: TankGame) -> np.ndarray:
    """把任意不超过12×12的迷宫无损填充到固定画布。"""
    maze = game.maze
    if maze.rows > MAP_SIZE or maze.cols > MAP_SIZE:
        raise ValueError(f"maze {maze.rows}x{maze.cols} exceeds observation limit {MAP_SIZE}x{MAP_SIZE}")
    result = np.zeros((MAP_CHANNELS, MAP_SIZE, MAP_SIZE), dtype=np.float32)
    for row in range(maze.rows):
        for col in range(maze.cols):
            result[:, row, col] = (
                maze.horizontal[row, col],
                maze.vertical[row, col + 1],
                maze.horizontal[row + 1, col],
                maze.vertical[row, col],
                1.0,
            )
    return result


def _nearest(items: list, own: Tank, limit: int, position) -> list:
    if len(items) <= limit:
        return items
    return sorted(items, key=lambda item: math.hypot(position(item)[0] - own.x, position(item)[1] - own.y))[:limit]


def _tank_features(own: Tank, tank: Tank, game: TankGame) -> np.ndarray:
    forward, right = _ego(own, tank.x - own.x, tank.y - own.y)
    heading_cos, heading_sin = _relative_heading(own, tank.heading)
    return np.asarray(
        (
            forward / MAP_SIZE, right / MAP_SIZE, heading_cos, heading_sin,
            tank.speed / game.max_speed, tank.angular_velocity / game.max_turn_rate,
            min(tank.cooldown / game.fire_cooldown, 1.0), float(tank.alive),
            _fraction(tank.x), _fraction(tank.y),
        ),
        dtype=np.float32,
    )


def _bullet_features(own: Tank, bullet: Bullet, game: TankGame) -> np.ndarray:
    forward, right = _ego(own, bullet.x - own.x, bullet.y - own.y)
    velocity_forward, velocity_right = _ego(own, bullet.vx, bullet.vy)
    return np.asarray(
        (
            forward / MAP_SIZE, right / MAP_SIZE,
            velocity_forward / game.bullet_speed, velocity_right / game.bullet_speed,
            min(bullet.age / game.bullet_lifetime, 1.0),
            bullet.bounces / max(game.max_bounces, 1),
            1.0 if bullet.owner_tank_id == own.tank_id else -1.0,
            _fraction(bullet.x), _fraction(bullet.y),
        ),
        dtype=np.float32,
    )


def build_observation(game: TankGame, tank_id: int) -> Observation:
    tank_by_id = {tank.tank_id: tank for tank in game.tanks}
    if tank_id not in tank_by_id:
        raise ValueError(f"unknown tank_id {tank_id}")
    own = tank_by_id[tank_id]
    others = _nearest(
        [tank for tank in game.tanks if tank.tank_id != tank_id], own,
        MAX_OTHER_TANKS, lambda tank: (tank.x, tank.y),
    )
    active_own_bullets = sum(bullet.owner_tank_id == tank_id for bullet in game.bullets)
    self_vector = np.asarray(
        (
            own.speed / game.max_speed, own.angular_velocity / game.max_turn_rate,
            min(own.cooldown / game.fire_cooldown, 1.0), float(own.alive),
            game.maze.rows / MAP_SIZE, game.maze.cols / MAP_SIZE,
            min(game.elapsed / game.time_limit, 1.0), math.cos(own.heading), math.sin(own.heading),
            _fraction(own.x), _fraction(own.y),
            active_own_bullets / max(game.max_bullets_per_tank, 1),
        ),
        dtype=np.float32,
    )

    tanks = np.zeros((MAX_OTHER_TANKS, TANK_FEATURES), dtype=np.float32)
    tank_pos = np.zeros((MAX_OTHER_TANKS, 2), dtype=np.float32)
    tank_mask = np.zeros(MAX_OTHER_TANKS, dtype=np.float32)
    for index, tank in enumerate(others):
        tanks[index] = _tank_features(own, tank, game)
        tank_pos[index] = _map_position(tank.x, tank.y)
        tank_mask[index] = 1.0

    visible_bullets = _nearest(list(game.bullets), own, MAX_BULLETS, lambda bullet: (bullet.x, bullet.y))
    bullets = np.zeros((MAX_BULLETS, BULLET_FEATURES), dtype=np.float32)
    bullet_pos = np.zeros((MAX_BULLETS, 2), dtype=np.float32)
    bullet_mask = np.zeros(MAX_BULLETS, dtype=np.float32)
    for index, bullet in enumerate(visible_bullets):
        bullets[index] = _bullet_features(own, bullet, game)
        bullet_pos[index] = _map_position(bullet.x, bullet.y)
        bullet_mask[index] = 1.0

    return {
        "map": _wall_map(game), "self": self_vector, "self_pos": _map_position(own.x, own.y),
        "tanks": tanks, "tank_pos": tank_pos, "tank_mask": tank_mask,
        "bullets": bullets, "bullet_pos": bullet_pos, "bullet_mask": bullet_mask,
    }
