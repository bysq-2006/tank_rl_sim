from __future__ import annotations

import math
from heapq import heappop, heappush

import numpy as np

from core.entities import Tank
from core.maze import Maze


Cell = tuple[int, int]


def tank_cell(tank: Tank, maze: Maze) -> Cell:
    """把坦克连续坐标限制并转换为所在的迷宫格。"""
    row = int(np.clip(math.floor(tank.y), 0, maze.rows - 1))
    col = int(np.clip(math.floor(tank.x), 0, maze.cols - 1))
    return row, col


def astar_path(maze: Maze, start: Cell, goal: Cell) -> list[Cell]:
    """使用 Manhattan 启发函数返回包含起点和终点的 A* 最短路径。"""
    if start == goal:
        return [start]
    open_heap: list[tuple[int, int, Cell]] = []
    heappush(open_heap, (_heuristic(start, goal), 0, start))
    costs = {start: 0}
    parent: dict[Cell, Cell | None] = {start: None}
    while open_heap:
        _, cost, current = heappop(open_heap)
        if cost != costs.get(current):
            continue
        if current == goal:
            path = [goal]
            while parent[path[-1]] is not None:
                path.append(parent[path[-1]])
            return list(reversed(path))
        for neighbor in _neighbors(maze, current):
            new_cost = cost + 1
            if new_cost >= costs.get(neighbor, math.inf):
                continue
            costs[neighbor] = new_cost
            parent[neighbor] = current
            heappush(open_heap, (new_cost + _heuristic(neighbor, goal), new_cost, neighbor))
    return [start]  # 正常生成的迷宫必定连通，此分支用于防御异常地图。


def astar_distance(maze: Maze, start: Cell, goal: Cell) -> int:
    """返回 A* 最短路径包含的移动格数。"""
    return len(astar_path(maze, start, goal)) - 1


def _heuristic(first: Cell, second: Cell) -> int:
    """返回网格上不高估真实距离的 Manhattan 启发值。"""
    return abs(first[0] - second[0]) + abs(first[1] - second[1])


def _neighbors(maze: Maze, cell: Cell) -> list[Cell]:
    """按照墙壁数据返回一个格子能够直接到达的相邻格。"""
    row, col = cell
    neighbors: list[Cell] = []
    if row > 0 and not maze.horizontal[row, col]:
        neighbors.append((row - 1, col))
    if row + 1 < maze.rows and not maze.horizontal[row + 1, col]:
        neighbors.append((row + 1, col))
    if col > 0 and not maze.vertical[row, col]:
        neighbors.append((row, col - 1))
    if col + 1 < maze.cols and not maze.vertical[row, col + 1]:
        neighbors.append((row, col + 1))
    return neighbors
