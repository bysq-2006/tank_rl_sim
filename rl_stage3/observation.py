from __future__ import annotations

import math
from typing import TypedDict

import numpy as np

from core import TankGame
from core.entities import Bullet, Tank

from .planning import astar_path, tank_cell


MAP_SIZE = 48  # 以自身为中心的局部墙图边长。
MAP_CHANNELS = 1  # 只编码墙壁。
LOCAL_RADIUS = 3.5  # 迷宫里多看一点周围的墙。
MAX_OTHER_TANKS = 7
MAX_BULLETS = 40
SELF_FEATURES = 12  # 自身运动、时间、相对路点；不含绝对坐标和朝向。
TANK_FEATURES = 8
BULLET_FEATURES = 7


class Observation(TypedDict):
    """一个智能体使用的观察：局部墙图加可变长的坦克、子弹集合。"""

    map: np.ndarray
    self: np.ndarray
    tanks: np.ndarray
    tank_mask: np.ndarray
    bullets: np.ndarray
    bullet_mask: np.ndarray


def _ego(own: Tank, dx: float, dy: float) -> tuple[float, float]:
    """世界位移转到自身坐标系：前方、右方。"""
    cos_h, sin_h = math.cos(own.heading), math.sin(own.heading)
    forward = dx * cos_h + dy * sin_h
    right = -dx * sin_h + dy * cos_h
    return forward, right


def _relative_heading(own: Tank, heading: float) -> tuple[float, float]:
    """把世界朝向变成相对自身炮口的 cos/sin。"""
    offset = heading - own.heading
    return math.cos(offset), math.sin(offset)


def _local_wall_map(own: Tank, wall_rects: list[tuple[float, float, float, float]]) -> np.ndarray:
    """以坦克为中心、按炮口朝向旋转后的局部墙通道。"""
    channel = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
    extent = 2.0 * LOCAL_RADIUS
    pixel = (np.arange(MAP_SIZE, dtype=np.float32) + 0.5) / MAP_SIZE
    right = (pixel - 0.5) * extent
    forward = (0.5 - pixel) * extent
    cos_h, sin_h = math.cos(own.heading), math.sin(own.heading)
    world_x = own.x + forward[:, None] * cos_h - right[None, :] * sin_h
    world_y = own.y + forward[:, None] * sin_h + right[None, :] * cos_h
    for x0, y0, x1, y1 in wall_rects:
        inside = (world_x >= x0) & (world_x <= x1) & (world_y >= y0) & (world_y <= y1)
        channel[inside] = 1.0
    return channel


def _nearest(items: list, own: Tank, limit: int, position) -> list:
    """数量超出上限时保留离自己最近的实体。"""
    if len(items) <= limit:
        return items
    return sorted(items, key=lambda item: math.hypot(position(item)[0] - own.x, position(item)[1] - own.y))[:limit]


def _tank_features(own: Tank, tank: Tank, game: TankGame) -> np.ndarray:
    """其他坦克：相对位置、相对朝向和运动状态。"""
    forward, right = _ego(own, tank.x - own.x, tank.y - own.y)
    heading_cos, heading_sin = _relative_heading(own, tank.heading)
    scale = LOCAL_RADIUS
    return np.asarray(
        (
            forward / scale,
            right / scale,
            heading_cos,
            heading_sin,
            tank.speed / game.max_speed,
            tank.angular_velocity / game.max_turn_rate,
            min(tank.cooldown / game.fire_cooldown, 1.0),
            float(tank.alive),
        ),
        dtype=np.float32,
    )


def _bullet_features(own: Tank, bullet: Bullet, game: TankGame) -> np.ndarray:
    """子弹：相对位置和相对速度。"""
    forward, right = _ego(own, bullet.x - own.x, bullet.y - own.y)
    vx_forward, vx_right = _ego(own, bullet.vx, bullet.vy)
    scale = LOCAL_RADIUS
    speed = max(game.bullet_speed, 1e-6)
    return np.asarray(
        (
            forward / scale,
            right / scale,
            vx_forward / speed,
            vx_right / speed,
            min(bullet.age / game.bullet_lifetime, 1.0),
            bullet.bounces / max(game.max_bounces, 1),
            1.0 if bullet.owner_tank_id == own.tank_id else -1.0,
        ),
        dtype=np.float32,
    )


def build_observation(game: TankGame, tank_id: int) -> Observation:
    """局部墙图 + 相对自身的坦克/子弹集合。"""
    tank_by_id = {tank.tank_id: tank for tank in game.tanks}
    if tank_id not in tank_by_id:
        raise ValueError(f"unknown tank_id {tank_id}")

    own = tank_by_id[tank_id]
    others = [tank for tank in game.tanks if tank.tank_id != tank_id]
    others = _nearest(others, own, MAX_OTHER_TANKS, lambda tank: (tank.x, tank.y))
    map_tensor = _local_wall_map(own, game.wall_rects)[None, ...]

    focus = next((tank for tank in others if tank.alive), None)
    if focus is None:
        waypoint_x, waypoint_y = own.x, own.y
        path_length = 0
    else:
        path = astar_path(game.maze, tank_cell(own, game.maze), tank_cell(focus, game.maze))
        if len(path) >= 2:
            next_row, next_col = path[1]
            waypoint_x, waypoint_y = next_col + 0.5, next_row + 0.5
        else:
            waypoint_x, waypoint_y = focus.x, focus.y
        path_length = len(path) - 1
    waypoint_forward, waypoint_right = _ego(own, waypoint_x - own.x, waypoint_y - own.y)
    scale = LOCAL_RADIUS
    self_vector = np.asarray(
        (
            own.speed / game.max_speed,
            own.angular_velocity / game.max_turn_rate,
            min(own.cooldown / game.fire_cooldown, 1.0),
            float(own.alive),
            game.maze.rows / 12.0,
            game.maze.cols / 12.0,
            game.maze.rows / game.maze.cols,
            min(game.elapsed / game.time_limit, 1.0),
            waypoint_forward / scale,
            waypoint_right / scale,
            math.hypot(waypoint_forward, waypoint_right) / scale,
            path_length / max(game.maze.rows * game.maze.cols, 1),
        ),
        dtype=np.float32,
    )

    tanks = np.zeros((MAX_OTHER_TANKS, TANK_FEATURES), dtype=np.float32)
    tank_mask = np.zeros(MAX_OTHER_TANKS, dtype=np.float32)
    for index, tank in enumerate(others):
        tanks[index] = _tank_features(own, tank, game)
        tank_mask[index] = 1.0

    visible_bullets = _nearest(list(game.bullets), own, MAX_BULLETS, lambda bullet: (bullet.x, bullet.y))
    bullets = np.zeros((MAX_BULLETS, BULLET_FEATURES), dtype=np.float32)
    bullet_mask = np.zeros(MAX_BULLETS, dtype=np.float32)
    for index, bullet in enumerate(visible_bullets):
        bullets[index] = _bullet_features(own, bullet, game)
        bullet_mask[index] = 1.0

    return {
        "map": map_tensor,
        "self": self_vector,
        "tanks": tanks,
        "tank_mask": tank_mask,
        "bullets": bullets,
        "bullet_mask": bullet_mask,
    }
