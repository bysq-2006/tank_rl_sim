from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Maze:
    """用两张布尔表保存网格迷宫的全部墙壁。"""

    rows: int  # 地图的格子行数。
    cols: int  # 地图的格子列数。
    horizontal: np.ndarray  # 水平墙，形状为 (rows + 1, cols)。
    vertical: np.ndarray  # 垂直墙，形状为 (rows, cols + 1)。

    @property
    def width(self) -> float:
        """返回地图在世界坐标中的宽度。"""
        return float(self.cols)

    @property
    def height(self) -> float:
        """返回地图在世界坐标中的高度。"""
        return float(self.rows)

    def wall_rects(self, thickness: float) -> list[tuple[float, float, float, float]]:
        """把所有墙线转换为相同厚度的碰撞矩形。"""
        half = thickness / 2.0
        extension = thickness / 4.0
        rects: list[tuple[float, float, float, float]] = []
        for r, c in np.argwhere(self.horizontal):
            # 整段总共增加半个墙宽，因此在两端各延伸四分之一个墙宽。
            rects.append((float(c) - extension, float(r) - half, float(c + 1) + extension, float(r) + half))
        for r, c in np.argwhere(self.vertical):
            rects.append((float(c) - half, float(r) - extension, float(c) + half, float(r + 1) + extension))
        return rects


def generate_maze(
    rng: np.random.Generator,
    rows: int | None = None,
    cols: int | None = None,
    min_size: int = 6,
    max_size: int = 9,
    loop_probability: float = 0.08,
) -> Maze:
    """生成连通迷宫，并随机拆除少量内部墙形成环路。"""
    # 未指定尺寸时，每次分别随机决定行数和列数。
    rows = int(rows if rows is not None else rng.integers(min_size, max_size + 1))
    cols = int(cols if cols is not None else rng.integers(min_size, max_size + 1))
    if rows < 2 or cols < 2:
        raise ValueError("maze rows and cols must both be at least 2")

    horizontal = np.ones((rows + 1, cols), dtype=np.bool_)  # 初始状态下所有水平墙都存在。
    vertical = np.ones((rows, cols + 1), dtype=np.bool_)  # 初始状态下所有垂直墙都存在。
    visited = np.zeros((rows, cols), dtype=np.bool_)  # 记录迷宫生成时访问过的格子。
    start = (int(rng.integers(rows)), int(rng.integers(cols)))  # 深度优先搜索的随机起点。
    stack = [start]  # 保存深度优先搜索路径，走不通时用于回退。
    visited[start] = True

    # 随机深度优先搜索：每进入一个新格子，就拆掉两格之间的墙。
    while stack:
        r, c = stack[-1]
        neighbors: list[tuple[int, int]] = []
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                neighbors.append((nr, nc))
        if not neighbors:
            stack.pop()
            continue
        nr, nc = neighbors[int(rng.integers(len(neighbors)))]
        if nr != r:
            horizontal[max(r, nr), c] = False
        else:
            vertical[r, max(c, nc)] = False
        visited[nr, nc] = True
        stack.append((nr, nc))

    # 再随机拆掉少量内部墙，使迷宫中出现可绕行的环路。
    for r in range(1, rows):
        for c in range(cols):
            if horizontal[r, c] and rng.random() < loop_probability:
                horizontal[r, c] = False
    for r in range(rows):
        for c in range(1, cols):
            if vertical[r, c] and rng.random() < loop_probability:
                vertical[r, c] = False
    return Maze(rows, cols, horizontal, vertical)
