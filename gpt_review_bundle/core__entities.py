from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Tank:
    """一辆坦克在游戏世界中的完整状态。"""

    x: float  # 坦克中心的世界 x 坐标，单位为格子。
    y: float  # 坦克中心的世界 y 坐标，单位为格子。
    heading: float  # 炮口朝向，单位为弧度。
    tank_id: int  # 坦克的唯一编号，用于记录每颗子弹的具体发射者。
    speed: float = 0.0  # 沿当前朝向的线速度，负数表示倒车。
    angular_velocity: float = 0.0  # 当前旋转角速度，单位为弧度/秒。
    cooldown: float = 0.0  # 距离允许下次开火还剩多少秒。
    alive: bool = True  # 是否仍然存活；任意子弹命中一次就变为 False。


@dataclass
class Bullet:
    """一颗飞行中子弹的完整状态。"""

    x: float  # 子弹中心的世界 x 坐标。
    y: float  # 子弹中心的世界 y 坐标。
    vx: float  # 子弹在 x 方向的速度。
    vy: float  # 子弹在 y 方向的速度。
    owner_tank_id: int  # 发射这颗子弹的具体坦克编号。
    age: float = 0.0  # 子弹已经存在的秒数。
    bounces: int = 0  # 子弹已经反弹的次数。
