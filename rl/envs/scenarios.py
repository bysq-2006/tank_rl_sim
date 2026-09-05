from __future__ import annotations

import math

import numpy as np

from core import TankGame
from core.maze import Maze


def make_open_maze(rows: int, cols: int) -> Maze:
    # 创建只有外边框且内部完全开放的矩形地图。
    horizontal = np.zeros((rows + 1, cols), dtype=np.bool_)
    vertical = np.zeros((rows, cols + 1), dtype=np.bool_)
    horizontal[0, :] = True
    horizontal[-1, :] = True
    vertical[:, 0] = True
    vertical[:, -1] = True
    return Maze(rows, cols, horizontal, vertical)


def make_simple_maze(rows: int, cols: int, rng: np.random.Generator) -> Maze:
    # 在开放地图中央加入一堵留有随机缺口的简单隔墙。
    maze = make_open_maze(rows, cols)
    column = max(1, min(cols - 1, cols // 2))
    gap = int(rng.integers(0, rows))
    for row in range(rows):
        if row != gap:
            maze.vertical[row, column] = True
    return maze


def apply_layout(game: TankGame, layout: str, rng: np.random.Generator) -> None:
    # 按课程要求替换地图布局并同步碰撞墙体数据。
    if layout == "maze":
        return
    if layout == "open":
        game.maze = make_open_maze(game.maze.rows, game.maze.cols)
    elif layout == "simple":
        game.maze = make_simple_maze(game.maze.rows, game.maze.cols, rng)
    else:
        raise ValueError(f"unknown layout {layout}")
    game.wall_rects = game.maze.wall_rects(game.wall_thickness)


def _clear_motion(game: TankGame) -> None:
    # 清除重设出生点后遗留的速度、冷却和子弹状态。
    for tank in game.tanks:
        tank.speed = 0.0
        tank.angular_velocity = 0.0
        tank.cooldown = 0.0
        tank.alive = True
    game.bullets.clear()


def apply_spawn(game: TankGame, spawn: str, rng: np.random.Generator) -> None:
    # 根据课程阶段设置双方出生距离和初始朝向。
    if len(game.tanks) < 2:
        raise ValueError("curriculum requires at least two tanks")
    own, enemy = game.tanks[:2]
    if spawn in {"close_facing", "random_heading"}:
        center_y = game.maze.rows / 2.0
        own.x, own.y = 1.5, center_y
        enemy.x, enemy.y = min(3.5, game.maze.cols - 1.5), center_y
        if spawn == "close_facing":
            own.heading, enemy.heading = 0.0, math.pi
        else:
            own.heading = float(rng.uniform(-math.pi, math.pi))
            enemy.heading = float(rng.uniform(-math.pi, math.pi))
    elif spawn in {"far_random", "random"}:
        own.heading = float(rng.uniform(-math.pi, math.pi))
        enemy.heading = float(rng.uniform(-math.pi, math.pi))
    else:
        raise ValueError(f"unknown spawn {spawn}")
    _clear_motion(game)
