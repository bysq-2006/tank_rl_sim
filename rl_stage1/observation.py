from __future__ import annotations

import math
from typing import TypedDict

import numpy as np

from core import TankGame
from core.entities import Bullet, Tank

from .planning import astar_path, tank_cell


MAP_SIZE = 128  # 模型接收的固定地图边长。
MAP_CHANNELS = 5  # 墙、自己、其他坦克、自己的子弹、其他子弹。
MAX_OTHER_TANKS = 7  # 除自己外最多编码的坦克数，便于以后加车。
MAX_BULLETS = 40  # 子弹集合上限；超过时保留离自己最近的。
SELF_FEATURES = 18  # 自身状态、地图尺寸、时间和导航提示。
TANK_FEATURES = 8  # 每辆其他坦克的相对数值特征。
BULLET_FEATURES = 7  # 每颗子弹的相对数值特征。


class Observation(TypedDict):
    """一个智能体使用的观察：地图加可变长的坦克、子弹集合。"""

    map: np.ndarray
    self: np.ndarray
    tanks: np.ndarray
    tank_mask: np.ndarray
    bullets: np.ndarray
    bullet_mask: np.ndarray


def _world_to_pixel(value: float, extent: float) -> float:
    """把一个世界坐标按整张地图拉伸到模型像素坐标。"""
    return value / extent * MAP_SIZE


def _fill_world_rect(channel: np.ndarray, rect: tuple[float, float, float, float], width: float, height: float) -> None:
    """把轴对齐的世界矩形填入一个观察通道。"""
    x0, y0, x1, y1 = rect
    px0 = max(0, min(MAP_SIZE - 1, int(math.floor(_world_to_pixel(x0, width)))))
    py0 = max(0, min(MAP_SIZE - 1, int(math.floor(_world_to_pixel(y0, height)))))
    px1 = max(px0 + 1, min(MAP_SIZE, int(math.ceil(_world_to_pixel(x1, width)))))
    py1 = max(py0 + 1, min(MAP_SIZE, int(math.ceil(_world_to_pixel(y1, height)))))
    channel[py0:py1, px0:px1] = 1.0


def _fill_oriented_rect(
    channel: np.ndarray,
    x: float,
    y: float,
    heading: float,
    half_length: float,
    half_width: float,
    map_width: float,
    map_height: float,
) -> None:
    """把旋转矩形实体按像素中心采样并填入观察通道。"""
    radius = math.hypot(half_length, half_width)
    px0 = max(0, int(math.floor(_world_to_pixel(x - radius, map_width))))
    px1 = min(MAP_SIZE, int(math.ceil(_world_to_pixel(x + radius, map_width))))
    py0 = max(0, int(math.floor(_world_to_pixel(y - radius, map_height))))
    py1 = min(MAP_SIZE, int(math.ceil(_world_to_pixel(y + radius, map_height))))
    if px0 >= px1 or py0 >= py1:
        return
    pixel_x = (np.arange(px0, px1, dtype=np.float32) + 0.5) / MAP_SIZE * map_width
    pixel_y = (np.arange(py0, py1, dtype=np.float32) + 0.5) / MAP_SIZE * map_height
    dx = pixel_x[None, :] - x
    dy = pixel_y[:, None] - y
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    local_forward = dx * cos_h + dy * sin_h
    local_right = -dx * sin_h + dy * cos_h
    mask = (np.abs(local_forward) <= half_length) & (np.abs(local_right) <= half_width)
    channel[py0:py1, px0:px1][mask] = 1.0


def _fill_circle(channel: np.ndarray, x: float, y: float, radius: float, width: float, height: float) -> None:
    """把圆形实体填入观察通道，并保证很小的子弹至少占一个像素。"""
    px = int(np.clip(_world_to_pixel(x, width), 0, MAP_SIZE - 1))
    py = int(np.clip(_world_to_pixel(y, height), 0, MAP_SIZE - 1))
    radius_x = max(1, int(math.ceil(radius / width * MAP_SIZE)))
    radius_y = max(1, int(math.ceil(radius / height * MAP_SIZE)))
    x0, x1 = max(0, px - radius_x), min(MAP_SIZE, px + radius_x + 1)
    y0, y1 = max(0, py - radius_y), min(MAP_SIZE, py + radius_y + 1)
    xs = (np.arange(x0, x1, dtype=np.float32) + 0.5) / MAP_SIZE * width
    ys = (np.arange(y0, y1, dtype=np.float32) + 0.5) / MAP_SIZE * height
    mask = (xs[None, :] - x) ** 2 + (ys[:, None] - y) ** 2 <= radius**2
    if not mask.any():
        channel[py, px] = 1.0
    else:
        channel[y0:y1, x0:x1][mask] = 1.0


def _nearest(items: list, own: Tank, limit: int, position) -> list:
    """数量超出上限时保留离自己最近的实体，集合编码不依赖排列顺序。"""
    if len(items) <= limit:
        return items
    return sorted(items, key=lambda item: math.hypot(position(item)[0] - own.x, position(item)[1] - own.y))[:limit]


def _tank_features(own: Tank, tank: Tank, game: TankGame, width: float, height: float) -> np.ndarray:
    """把另一辆坦克编码成相对自身的固定长度向量。"""
    return np.asarray(
        (
            (tank.x - own.x) / width,
            (tank.y - own.y) / height,
            math.cos(tank.heading),
            math.sin(tank.heading),
            tank.speed / game.max_speed,
            tank.angular_velocity / game.max_turn_rate,
            min(tank.cooldown / game.fire_cooldown, 1.0),
            float(tank.alive),
        ),
        dtype=np.float32,
    )


def _bullet_features(own: Tank, bullet: Bullet, game: TankGame, width: float, height: float) -> np.ndarray:
    """把一颗子弹编码成相对自身的固定长度向量。"""
    return np.asarray(
        (
            (bullet.x - own.x) / width,
            (bullet.y - own.y) / height,
            bullet.vx / game.bullet_speed,
            bullet.vy / game.bullet_speed,
            min(bullet.age / game.bullet_lifetime, 1.0),
            bullet.bounces / max(game.max_bounces, 1),
            1.0 if bullet.owner_tank_id == own.tank_id else -1.0,
        ),
        dtype=np.float32,
    )


def build_observation(game: TankGame, tank_id: int) -> Observation:
    """从精确游戏状态生成指定坦克看到的地图和可变长实体集合。"""
    tank_by_id = {tank.tank_id: tank for tank in game.tanks}
    if tank_id not in tank_by_id:
        raise ValueError(f"unknown tank_id {tank_id}")

    own = tank_by_id[tank_id]
    others = [tank for tank in game.tanks if tank.tank_id != tank_id]
    others = _nearest(others, own, MAX_OTHER_TANKS, lambda tank: (tank.x, tank.y))
    width, height = game.maze.width, game.maze.height
    longest_extent = max(width, height)
    map_tensor = np.zeros((MAP_CHANNELS, MAP_SIZE, MAP_SIZE), dtype=np.float32)

    for wall in game.wall_rects:
        _fill_world_rect(map_tensor[0], wall, width, height)
    if own.alive:
        _fill_oriented_rect(map_tensor[1], own.x, own.y, own.heading, game.tank_half_length, game.tank_half_width, width, height)
    for tank in others:
        if tank.alive:
            _fill_oriented_rect(map_tensor[2], tank.x, tank.y, tank.heading, game.tank_half_length, game.tank_half_width, width, height)
    for bullet in game.bullets:
        channel = 3 if bullet.owner_tank_id == tank_id else 4
        _fill_circle(map_tensor[channel], bullet.x, bullet.y, game.bullet_radius, width, height)

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
    waypoint_dx, waypoint_dy = waypoint_x - own.x, waypoint_y - own.y
    cos_h, sin_h = math.cos(own.heading), math.sin(own.heading)
    waypoint_forward = waypoint_dx * cos_h + waypoint_dy * sin_h
    waypoint_right = -waypoint_dx * sin_h + waypoint_dy * cos_h
    waypoint_angle = math.atan2(waypoint_dy, waypoint_dx) - own.heading
    self_vector = np.asarray(
        (
            own.x / width,
            own.y / height,
            math.cos(own.heading),
            math.sin(own.heading),
            own.speed / game.max_speed,
            own.angular_velocity / game.max_turn_rate,
            min(own.cooldown / game.fire_cooldown, 1.0),
            float(own.alive),
            game.maze.rows / 12.0,
            game.maze.cols / 12.0,
            game.maze.rows / game.maze.cols,
            min(game.elapsed / game.time_limit, 1.0),
            waypoint_forward / longest_extent,
            waypoint_right / longest_extent,
            math.hypot(waypoint_dx, waypoint_dy) / longest_extent,
            math.sin(waypoint_angle),
            math.cos(waypoint_angle),
            path_length / max(game.maze.rows * game.maze.cols, 1),
        ),
        dtype=np.float32,
    )

    tanks = np.zeros((MAX_OTHER_TANKS, TANK_FEATURES), dtype=np.float32)
    tank_mask = np.zeros(MAX_OTHER_TANKS, dtype=np.float32)
    for index, tank in enumerate(others):
        tanks[index] = _tank_features(own, tank, game, width, height)
        tank_mask[index] = 1.0

    visible_bullets = _nearest(list(game.bullets), own, MAX_BULLETS, lambda bullet: (bullet.x, bullet.y))
    bullets = np.zeros((MAX_BULLETS, BULLET_FEATURES), dtype=np.float32)
    bullet_mask = np.zeros(MAX_BULLETS, dtype=np.float32)
    for index, bullet in enumerate(visible_bullets):
        bullets[index] = _bullet_features(own, bullet, game, width, height)
        bullet_mask[index] = 1.0

    return {
        "map": map_tensor,
        "self": self_vector,
        "tanks": tanks,
        "tank_mask": tank_mask,
        "bullets": bullets,
        "bullet_mask": bullet_mask,
    }
