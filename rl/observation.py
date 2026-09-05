from __future__ import annotations

import math
from typing import TypedDict

import numpy as np

from core import TankGame
from core.entities import Bullet, Tank


# The image keeps a fixed physical scale: one maze cell is always eight pixels.
# A 96-pixel canvas leaves room for maps up to 12 x 12 without stretching them.
PIXELS_PER_CELL = 8
MAX_MAP_CELLS = 12
MAP_SIZE = PIXELS_PER_CELL * MAX_MAP_CELLS
MAP_CHANNELS = 2  # wall pixels and valid-map mask

MAX_OTHER_TANKS = 3
MAX_BULLETS = 15

# Only information that can be recovered from rendered frames is exposed.
SELF_FEATURES = 4  # absolute x/y, cos(heading), sin(heading)
TANK_FEATURES = 4  # ego-relative x/y, cos(relative heading), sin(relative heading)
BULLET_FEATURES = 4  # ego-relative x/y and velocity x/y


class Observation(TypedDict):
    map: np.ndarray
    self: np.ndarray
    tanks: np.ndarray
    tank_mask: np.ndarray
    bullets: np.ndarray
    bullet_mask: np.ndarray


def _ego(own: Tank, dx: float, dy: float) -> tuple[float, float]:
    """Rotate a world-space vector into the controlled tank's frame."""
    # 将世界坐标系中的方向转换到自身坦克坐标系。
    cos_h, sin_h = math.cos(own.heading), math.sin(own.heading)
    return dx * cos_h + dy * sin_h, -dx * sin_h + dy * cos_h


def _nearest(items: list, own: Tank, limit: int, position) -> list:
    """Keep shapes bounded if a future game mode exceeds the configured limit."""
    # 当实体超出上限时，只保留距离自身最近的若干个实体。
    if len(items) <= limit:
        return items
    return sorted(
        items,
        key=lambda item: math.hypot(position(item)[0] - own.x, position(item)[1] - own.y),
    )[:limit]


def _wall_map(game: TankGame) -> np.ndarray:
    """Rasterize walls without resizing and mark the part belonging to the map."""
    # 按固定像素比例绘制墙体，并生成有效地图区域掩码。
    maze = game.maze
    if maze.rows > MAX_MAP_CELLS or maze.cols > MAX_MAP_CELLS:
        raise ValueError(
            f"maze {maze.rows}x{maze.cols} exceeds observation limit "
            f"{MAX_MAP_CELLS}x{MAX_MAP_CELLS}"
        )

    result = np.zeros((MAP_CHANNELS, MAP_SIZE, MAP_SIZE), dtype=np.float32)
    height = maze.rows * PIXELS_PER_CELL
    width = maze.cols * PIXELS_PER_CELL
    result[1, :height, :width] = 1.0

    wall = result[0]
    for row, col in np.argwhere(maze.horizontal):
        y = min(int(row) * PIXELS_PER_CELL, height - 1)
        x0 = int(col) * PIXELS_PER_CELL
        x1 = min((int(col) + 1) * PIXELS_PER_CELL + 1, width)
        wall[y, x0:x1] = 1.0
    for row, col in np.argwhere(maze.vertical):
        x = min(int(col) * PIXELS_PER_CELL, width - 1)
        y0 = int(row) * PIXELS_PER_CELL
        y1 = min((int(row) + 1) * PIXELS_PER_CELL + 1, height)
        wall[y0:y1, x] = 1.0
    return result


def _self_features(own: Tank) -> np.ndarray:
    # 提取自身坦克可从画面识别的坐标和朝向信息。
    return np.asarray(
        (
            2.0 * own.x / MAX_MAP_CELLS - 1.0,
            2.0 * own.y / MAX_MAP_CELLS - 1.0,
            math.cos(own.heading),
            math.sin(own.heading),
        ),
        dtype=np.float32,
    )


def _tank_features(own: Tank, tank: Tank) -> np.ndarray:
    # 提取敌方坦克相对自身的坐标和朝向信息。
    forward, right = _ego(own, tank.x - own.x, tank.y - own.y)
    relative_heading = tank.heading - own.heading
    return np.asarray(
        (
            forward / MAX_MAP_CELLS,
            right / MAX_MAP_CELLS,
            math.cos(relative_heading),
            math.sin(relative_heading),
        ),
        dtype=np.float32,
    )


def _bullet_features(own: Tank, bullet: Bullet, game: TankGame) -> np.ndarray:
    # 提取子弹相对自身的位置和运动速度信息。
    forward, right = _ego(own, bullet.x - own.x, bullet.y - own.y)
    velocity_forward, velocity_right = _ego(own, bullet.vx, bullet.vy)
    return np.asarray(
        (
            forward / MAX_MAP_CELLS,
            right / MAX_MAP_CELLS,
            velocity_forward / game.bullet_speed,
            velocity_right / game.bullet_speed,
        ),
        dtype=np.float32,
    )


def build_observation(game: TankGame, tank_id: int) -> Observation:
    """Build a fixed-shape observation containing observable information only."""
    # 将当前游戏画面状态整理成模型使用的固定形状观察。
    tank_by_id = {tank.tank_id: tank for tank in game.tanks}
    if tank_id not in tank_by_id:
        raise ValueError(f"unknown tank_id {tank_id}")
    own = tank_by_id[tank_id]

    others = _nearest(
        [tank for tank in game.tanks if tank.tank_id != tank_id and tank.alive],
        own,
        MAX_OTHER_TANKS,
        lambda tank: (tank.x, tank.y),
    )
    tanks = np.zeros((MAX_OTHER_TANKS, TANK_FEATURES), dtype=np.float32)
    tank_mask = np.zeros(MAX_OTHER_TANKS, dtype=np.float32)
    for index, tank in enumerate(others):
        tanks[index] = _tank_features(own, tank)
        tank_mask[index] = 1.0

    visible_bullets = _nearest(
        list(game.bullets), own, MAX_BULLETS, lambda bullet: (bullet.x, bullet.y)
    )
    bullets = np.zeros((MAX_BULLETS, BULLET_FEATURES), dtype=np.float32)
    bullet_mask = np.zeros(MAX_BULLETS, dtype=np.float32)
    for index, bullet in enumerate(visible_bullets):
        bullets[index] = _bullet_features(own, bullet, game)
        bullet_mask[index] = 1.0

    return {
        "map": _wall_map(game),
        "self": _self_features(own),
        "tanks": tanks,
        "tank_mask": tank_mask,
        "bullets": bullets,
        "bullet_mask": bullet_mask,
    }
