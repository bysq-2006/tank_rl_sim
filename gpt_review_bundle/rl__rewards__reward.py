from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from core import TankGame
from core.geometry import segment_intersects_rect


@dataclass(frozen=True)
class RewardConfig:
    """一个课程阶段使用的奖励系数。"""

    win_reward: float = 1.0
    loss_penalty: float = -1.0
    draw_penalty: float = -1.0
    timeout_penalty: float = -1.0
    kill_bonus: float = 0.25
    death_penalty: float = -0.25
    shot_reward: float = 0.0
    max_shot_reward: float = 0.0
    unsafe_shot_penalty: float = -0.03
    bad_shot_penalty: float = -0.03
    aimed_shot_reward: float = 0.03
    max_aimed_shot_reward: float = 0.30
    opponent_self_kill_reward: float = 0.0
    shaping_scale: float = 0.30
    gamma: float = 0.995


def _tank_by_id(game: TankGame, tank_id: int):
    # 按编号查找坦克并在编号无效时明确报错。
    for tank in game.tanks:
        if tank.tank_id == tank_id:
            return tank
    raise ValueError(f"unknown tank_id {tank_id}")


def _tank_cell(game: TankGame, tank) -> tuple[int, int]:
    # 将坦克的连续坐标转换成迷宫格子坐标。
    row = min(max(int(tank.y), 0), game.maze.rows - 1)
    col = min(max(int(tank.x), 0), game.maze.cols - 1)
    return row, col


def _path_distance(game: TankGame, start: tuple[int, int], goal: tuple[int, int]) -> int:
    # 使用广度优先搜索计算两格之间的最短迷宫距离。
    if start == goal:
        return 0
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        (row, col), distance = queue.popleft()
        candidates = (
            (row - 1, col, row > 0 and not game.maze.horizontal[row, col]),
            (row + 1, col, row + 1 < game.maze.rows and not game.maze.horizontal[row + 1, col]),
            (row, col - 1, col > 0 and not game.maze.vertical[row, col]),
            (row, col + 1, col + 1 < game.maze.cols and not game.maze.vertical[row, col + 1]),
        )
        for next_row, next_col, open_path in candidates:
            next_cell = (next_row, next_col)
            if open_path and next_cell not in visited:
                if next_cell == goal:
                    return distance + 1
                visited.add(next_cell)
                queue.append((next_cell, distance + 1))
    return game.maze.rows * game.maze.cols


def _wrapped_angle(angle: float) -> float:
    # 将任意角度折算到负π至正π范围。
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _has_line_of_sight(game: TankGame, own, enemy) -> bool:
    # 检查自身与敌人之间的线段是否被任意墙体阻挡。
    return not any(
        segment_intersects_rect(own.x, own.y, enemy.x, enemy.y, wall)
        for wall in game.wall_rects
    )


def calculate_potential(game: TankGame, tank_id: int) -> float:
    # 用寻路距离、瞄准误差和直视状态组成范围稳定的进度势函数。
    own = _tank_by_id(game, tank_id)
    enemies = [tank for tank in game.tanks if tank.tank_id != tank_id and tank.alive]
    if not own.alive or not enemies:
        return 0.0
    enemy = min(enemies, key=lambda tank: math.hypot(tank.x - own.x, tank.y - own.y))
    max_distance = max(game.maze.rows + game.maze.cols - 2, 1)
    distance = _path_distance(game, _tank_cell(game, own), _tank_cell(game, enemy))
    distance_score = -min(distance / max_distance, 1.0)
    target_angle = math.atan2(enemy.y - own.y, enemy.x - own.x)
    aim_score = -abs(_wrapped_angle(target_angle - own.heading)) / math.pi
    los_bonus = 0.10 if _has_line_of_sight(game, own, enemy) else 0.0
    return 0.55 * distance_score + 0.35 * aim_score + los_bonus


class RewardTracker:
    """按决策步累计事件、势函数与终局奖励。"""

    def __init__(self, tank_id: int, config: RewardConfig) -> None:
        # 保存受控坦克和奖励配置并初始化势函数状态。
        self.tank_id = tank_id
        self.config = config
        self.previous_potential = 0.0
        self.previous_shots_fired = 0
        self.previous_unsafe_shots = 0
        self.previous_good_shots = 0
        self.previous_bad_shots = 0
        self.shot_reward_paid = 0.0
        self.opponent_self_killed = False
        self.terminal_paid = False

    def reset(self, game: TankGame) -> None:
        # 在新对局开始时重置终局标志并记录初始势函数。
        self.previous_potential = calculate_potential(game, self.tank_id)
        self.previous_shots_fired = game.shots_fired_by_tank.get(self.tank_id, 0)
        self.previous_unsafe_shots = game.unsafe_shots_by_tank.get(self.tank_id, 0)
        self.previous_good_shots = game.good_shots_by_tank.get(self.tank_id, 0)
        self.previous_bad_shots = game.bad_shots_by_tank.get(self.tank_id, 0)
        self.shot_reward_paid = 0.0
        self.opponent_self_killed = False
        self.terminal_paid = False

    def calculate(self, game: TankGame, events: list[dict]) -> float:
        # 合并即时击杀、死亡、势函数变化和一次性终局奖励。
        reward = self._shot_reward(game)
        reward += self._event_reward(events)
        if not (game.is_over and self.opponent_self_killed):
            reward += self._shaping_reward(game)
        if game.is_over and not self.terminal_paid:
            reward += self._terminal_reward(game)
            self.terminal_paid = True
        return reward

    def _shot_reward(self, game: TankGame) -> float:
        # 只奖励预测能接近敌人的子弹，其他轨迹一律惩罚，避免无目标乱射。
        current_shots = game.shots_fired_by_tank.get(self.tank_id, 0)
        current_unsafe = game.unsafe_shots_by_tank.get(self.tank_id, 0)
        new_shots = max(current_shots - self.previous_shots_fired, 0)
        new_unsafe = max(current_unsafe - self.previous_unsafe_shots, 0)
        current_good = game.good_shots_by_tank.get(self.tank_id, 0)
        current_bad = game.bad_shots_by_tank.get(self.tank_id, 0)
        new_good = max(current_good - self.previous_good_shots, 0)
        new_bad = max(current_bad - self.previous_bad_shots, 0)
        self.previous_shots_fired = current_shots
        self.previous_unsafe_shots = current_unsafe
        self.previous_good_shots = current_good
        self.previous_bad_shots = current_bad
        remaining = max(self.config.max_aimed_shot_reward - self.shot_reward_paid, 0.0)
        safe_shots = max(new_shots - new_unsafe, 0)
        safe_reward = min(new_good * self.config.aimed_shot_reward, remaining)
        self.shot_reward_paid += safe_reward
        return safe_reward + new_unsafe * self.config.unsafe_shot_penalty + new_bad * self.config.bad_shot_penalty

    def _event_reward(self, events: list[dict]) -> float:
        # 只奖励受控坦克的主动击杀并惩罚自身被摧毁。
        reward = 0.0
        for event in events:
            shooter, victim = event["shooter"], event["victim"]
            if shooter == victim and victim != self.tank_id:
                self.opponent_self_killed = True
            if shooter == self.tank_id and victim != self.tank_id:
                reward += self.config.kill_bonus
            if victim == self.tank_id:
                reward += self.config.death_penalty
        return reward

    def _shaping_reward(self, game: TankGame) -> float:
        # 根据相邻状态的势函数差提供不能往复刷取的过程信号。
        current = 0.0 if game.is_over else calculate_potential(game, self.tank_id)
        reward = self.config.shaping_scale * (
            self.config.gamma * current - self.previous_potential
        )
        self.previous_potential = current
        return reward

    def _terminal_reward(self, game: TankGame) -> float:
        # 根据最终胜者以及是否超时返回主要任务奖励。
        if game.winner == self.tank_id:
            if self.opponent_self_killed:
                return self.config.opponent_self_kill_reward
            return self.config.win_reward
        if game.winner is not None:
            return self.config.loss_penalty
        if game.elapsed >= game.time_limit:
            return self.config.timeout_penalty
        return self.config.draw_penalty
