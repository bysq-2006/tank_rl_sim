from __future__ import annotations

import math
from dataclasses import dataclass

from core import TankGame
from core.entities import Bullet, Tank


@dataclass(frozen=True)
class PathSegment:
    """弹道在两次反弹之间的一段直线及其时间范围。"""

    x0: float
    y0: float
    x1: float
    y1: float
    time0: float
    time1: float


@dataclass(frozen=True)
class TrajectoryResult:
    """静态弹道对自己和敌人的最近距离及预测首个命中对象。"""

    enemy_distance: float
    self_distance: float
    predicted_hit: int | None
    segments: tuple[PathSegment, ...]


def trace_bullet_trajectory(game: TankGame, bullet: Bullet, owner_tank_id: int, max_seconds: float = 10.0) -> TrajectoryResult:
    """用膨胀墙体的射线反射快速预演子弹，不修改真实游戏状态。"""
    remaining_time = max(0.0, min(max_seconds, game.bullet_lifetime - bullet.age))
    speed = max(math.hypot(bullet.vx, bullet.vy), 1e-12)
    direction_x, direction_y = bullet.vx / speed, bullet.vy / speed
    x, y, elapsed = bullet.x, bullet.y, 0.0
    segments: list[PathSegment] = []
    remaining_distance = speed * remaining_time
    bounces_left = max(0, game.max_bounces - bullet.bounces)
    epsilon = 1e-6

    while remaining_distance > epsilon:
        nearest = remaining_distance
        hit_x = False
        hit_y = False
        for wall in game.wall_rects:
            expanded = (
                wall[0] - game.bullet_radius,
                wall[1] - game.bullet_radius,
                wall[2] + game.bullet_radius,
                wall[3] + game.bullet_radius,
            )
            hit = _ray_rect_entry(x, y, direction_x, direction_y, expanded, epsilon)
            if hit is None:
                continue
            distance, normal_x, normal_y = hit
            if distance < nearest - epsilon:
                nearest, hit_x, hit_y = distance, normal_x, normal_y
            elif abs(distance - nearest) <= epsilon:
                hit_x = hit_x or normal_x
                hit_y = hit_y or normal_y

        end_x, end_y = x + direction_x * nearest, y + direction_y * nearest
        duration = nearest / speed
        segments.append(PathSegment(x, y, end_x, end_y, elapsed, elapsed + duration))
        elapsed += duration
        remaining_distance -= nearest
        if not hit_x and not hit_y:
            break
        if bounces_left <= 0:
            break
        if hit_x:
            direction_x *= -1.0
        if hit_y:
            direction_y *= -1.0
        bounces_left -= 1
        x, y = end_x + direction_x * epsilon, end_y + direction_y * epsilon
        remaining_distance = max(0.0, remaining_distance - epsilon)

    own = next(tank for tank in game.tanks if tank.tank_id == owner_tank_id)
    enemy = next(tank for tank in game.tanks if tank.tank_id != owner_tank_id)
    self_distance = math.inf
    enemy_distance = math.inf
    predicted_hit: int | None = None
    ignore_time = max(0.08 - bullet.age, 0.0)
    for segment in segments:
        usable = _trim_segment_start(segment, ignore_time)
        if usable is None:
            continue
        own_distance = _segment_tank_distance(usable, own, game)
        target_distance = _segment_tank_distance(usable, enemy, game)
        self_distance = min(self_distance, own_distance)
        enemy_distance = min(enemy_distance, target_distance)
        own_entry = _segment_tank_entry(usable, own, game)
        enemy_entry = _segment_tank_entry(usable, enemy, game)
        if own_entry is not None or enemy_entry is not None:
            if enemy_entry is None or (own_entry is not None and own_entry <= enemy_entry):
                predicted_hit = own.tank_id
                self_distance = 0.0
            else:
                predicted_hit = enemy.tank_id
                enemy_distance = 0.0
            break
    return TrajectoryResult(enemy_distance, self_distance, predicted_hit, tuple(segments))


def proximity_score(distance: float) -> float:
    """把最近距离平滑转换为 0 到 1；命中为 1，越远越接近 0。"""
    if math.isinf(distance):
        return 0.0
    return math.exp(-max(distance, 0.0) / 0.75)


def _ray_rect_entry(
    x: float,
    y: float,
    dx: float,
    dy: float,
    rect: tuple[float, float, float, float],
    epsilon: float,
) -> tuple[float, bool, bool] | None:
    """返回射线进入矩形的距离，以及应反射 x、y 速度中的哪一项。"""
    entries: list[tuple[float, str]] = []
    exits: list[float] = []
    for origin, direction, lower, upper, axis in ((x, dx, rect[0], rect[2], "x"), (y, dy, rect[1], rect[3], "y")):
        if abs(direction) < 1e-12:
            if origin < lower or origin > upper:
                return None
            entries.append((-math.inf, axis))
            exits.append(math.inf)
            continue
        first, second = (lower - origin) / direction, (upper - origin) / direction
        entries.append((min(first, second), axis))
        exits.append(max(first, second))
    entry = max(value for value, _ in entries)
    exit_ = min(exits)
    if entry > exit_ or exit_ <= epsilon or entry <= epsilon:
        return None
    hit_x = any(axis == "x" and abs(value - entry) <= epsilon for value, axis in entries)
    hit_y = any(axis == "y" and abs(value - entry) <= epsilon for value, axis in entries)
    return entry, hit_x, hit_y


def _trim_segment_start(segment: PathSegment, minimum_time: float) -> PathSegment | None:
    """移除游戏规则规定的出膛后短暂无碰撞时间对应的线段前部。"""
    if segment.time1 <= minimum_time:
        return None
    if segment.time0 >= minimum_time:
        return segment
    fraction = (minimum_time - segment.time0) / max(segment.time1 - segment.time0, 1e-12)
    return PathSegment(
        segment.x0 + (segment.x1 - segment.x0) * fraction,
        segment.y0 + (segment.y1 - segment.y0) * fraction,
        segment.x1,
        segment.y1,
        minimum_time,
        segment.time1,
    )


def _to_tank_local(x: float, y: float, tank: Tank) -> tuple[float, float]:
    """把世界坐标转换到以坦克中心和朝向为基准的局部坐标。"""
    dx, dy = x - tank.x, y - tank.y
    cos_h, sin_h = math.cos(tank.heading), math.sin(tank.heading)
    return dx * cos_h + dy * sin_h, -dx * sin_h + dy * cos_h


def _segment_tank_entry(segment: PathSegment, tank: Tank, game: TankGame) -> float | None:
    """返回弹道进入扩张后坦克矩形的线段比例，没有相交则返回 None。"""
    x0, y0 = _to_tank_local(segment.x0, segment.y0, tank)
    x1, y1 = _to_tank_local(segment.x1, segment.y1, tank)
    half_length = game.tank_half_length + game.bullet_radius
    half_width = game.tank_half_width + game.bullet_radius
    return _segment_aabb_entry(x0, y0, x1, y1, (-half_length, -half_width, half_length, half_width))


def _segment_tank_distance(segment: PathSegment, tank: Tank, game: TankGame) -> float:
    """计算弹道线段到扩张后坦克矩形的最短距离。"""
    x0, y0 = _to_tank_local(segment.x0, segment.y0, tank)
    x1, y1 = _to_tank_local(segment.x1, segment.y1, tank)
    half_length = game.tank_half_length + game.bullet_radius
    half_width = game.tank_half_width + game.bullet_radius
    rect = (-half_length, -half_width, half_length, half_width)
    if _segment_aabb_entry(x0, y0, x1, y1, rect) is not None:
        return 0.0
    distances = [_point_rect_distance(x0, y0, rect), _point_rect_distance(x1, y1, rect)]
    corners = ((rect[0], rect[1]), (rect[0], rect[3]), (rect[2], rect[3]), (rect[2], rect[1]))
    for index in range(4):
        ax, ay = corners[index]
        bx, by = corners[(index + 1) % 4]
        distances.append(_segment_segment_distance(x0, y0, x1, y1, ax, ay, bx, by))
    return min(distances)


def _segment_aabb_entry(x0: float, y0: float, x1: float, y1: float, rect: tuple[float, float, float, float]) -> float | None:
    """返回有限线段进入轴对齐矩形时的 0 到 1 比例。"""
    t_min, t_max = 0.0, 1.0
    for origin, delta, lower, upper in ((x0, x1 - x0, rect[0], rect[2]), (y0, y1 - y0, rect[1], rect[3])):
        if abs(delta) < 1e-12:
            if origin < lower or origin > upper:
                return None
            continue
        first, second = (lower - origin) / delta, (upper - origin) / delta
        entry, exit_ = min(first, second), max(first, second)
        t_min, t_max = max(t_min, entry), min(t_max, exit_)
        if t_min > t_max:
            return None
    return t_min


def _point_rect_distance(x: float, y: float, rect: tuple[float, float, float, float]) -> float:
    """计算点到轴对齐矩形的欧氏距离。"""
    dx = max(rect[0] - x, 0.0, x - rect[2])
    dy = max(rect[1] - y, 0.0, y - rect[3])
    return math.hypot(dx, dy)


def _point_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    """计算点到有限线段的欧氏距离。"""
    dx, dy = bx - ax, by - ay
    denominator = dx * dx + dy * dy
    if denominator <= 1e-20:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denominator))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _orientation(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
    """返回三点的二维叉积，用于判断两条线段是否相交。"""
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _segments_intersect(ax: float, ay: float, bx: float, by: float, cx: float, cy: float, dx: float, dy: float) -> bool:
    """判断两条有限线段是否相交或接触。"""
    first = _orientation(ax, ay, bx, by, cx, cy)
    second = _orientation(ax, ay, bx, by, dx, dy)
    third = _orientation(cx, cy, dx, dy, ax, ay)
    fourth = _orientation(cx, cy, dx, dy, bx, by)
    epsilon = 1e-12
    if first * second < -epsilon and third * fourth < -epsilon:
        return True

    def on_segment(px: float, py: float, qx: float, qy: float, rx: float, ry: float) -> bool:
        return min(px, rx) - epsilon <= qx <= max(px, rx) + epsilon and min(py, ry) - epsilon <= qy <= max(py, ry) + epsilon

    return (
        (abs(first) <= epsilon and on_segment(ax, ay, cx, cy, bx, by))
        or (abs(second) <= epsilon and on_segment(ax, ay, dx, dy, bx, by))
        or (abs(third) <= epsilon and on_segment(cx, cy, ax, ay, dx, dy))
        or (abs(fourth) <= epsilon and on_segment(cx, cy, bx, by, dx, dy))
    )


def _segment_segment_distance(ax: float, ay: float, bx: float, by: float, cx: float, cy: float, dx: float, dy: float) -> float:
    """计算两条有限线段之间的最短距离。"""
    if _segments_intersect(ax, ay, bx, by, cx, cy, dx, dy):
        return 0.0
    return min(
        _point_segment_distance(ax, ay, cx, cy, dx, dy),
        _point_segment_distance(bx, by, cx, cy, dx, dy),
        _point_segment_distance(cx, cy, ax, ay, bx, by),
        _point_segment_distance(dx, dy, ax, ay, bx, by),
    )
