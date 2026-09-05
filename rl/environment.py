from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from core import TankGame
from core.geometry import segment_intersects_rect
from core.maze import Maze

from .observation import Observation, build_observation
from .planning import Cell, astar_path

LAYOUTS = ("maze", "open")
SPAWNS = ("default", "close_facing", "far_random")


def make_open_maze(rows: int, cols: int) -> Maze:
    """只保留外墙的空场。"""
    horizontal = np.zeros((rows + 1, cols), dtype=np.bool_)
    vertical = np.zeros((rows, cols + 1), dtype=np.bool_)
    horizontal[0] = True
    horizontal[-1] = True
    vertical[:, 0] = True
    vertical[:, -1] = True
    return Maze(rows, cols, horizontal, vertical)


@dataclass
class RewardConfig:
    """终局胜负，以及不会改变最优策略的势函数塑形。"""

    win: float = 1.0  # 最终存活并获胜。
    loss: float = -1.0  # 被敌人击毁。
    self_kill: float = -1.0  # 被自己的子弹击毁。
    timeout: float = -1.0  # 超时双方惩罚。
    potential_scale: float = 0.0  # 训练时建议 0.2；评估环境保持 0。
    potential_gamma: float = 0.995  # 必须与 PPO 的决策步 gamma 一致。


class TankSelfPlayEnv:
    """把纯游戏核心包装成两个共享策略智能体使用的自博弈环境。"""

    num_agents = 2  # 当前核心固定生成两辆相互独立的坦克。

    def __init__(
        self,
        action_repeat: int = 2,
        rows: int | None = None,
        cols: int | None = None,
        time_limit: float = 90.0,
        layout: str = "maze",
        spawn: str = "default",
        reward_config: RewardConfig | None = None,
    ) -> None:
        """创建环境；每个模型动作默认连续保持两个 24 Hz 物理帧。"""
        if action_repeat < 1:
            raise ValueError("action_repeat must be at least 1")
        if layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}")
        if spawn not in SPAWNS:
            raise ValueError(f"spawn must be one of {SPAWNS}")
        self.action_repeat = action_repeat
        self.layout = layout
        self.spawn = spawn
        self.game = TankGame(rows=rows, cols=cols, time_limit=time_limit)
        self.reward_config = reward_config if reward_config is not None else RewardConfig()
        self.agent_ids = tuple(tank.tank_id for tank in self.game.tanks)
        self._potential = np.zeros(self.num_agents, dtype=np.float32)

    def reset(self, seed: int | None = None) -> list[Observation]:
        """开始新对局并返回两辆坦克各自视角的观察。"""
        self.game.reset(seed=seed)
        if self.layout == "open":
            self.game.maze = make_open_maze(self.game.maze.rows, self.game.maze.cols)
            self.game.wall_rects = self.game.maze.wall_rects(self.game.wall_thickness)
        self._place_tanks()
        self.agent_ids = tuple(tank.tank_id for tank in self.game.tanks)
        self._potential[:] = self._state_potential()
        return self._observations()

    def _place_tanks(self) -> None:
        """按关卡出生规则放置两辆坦克，不改变核心的胜负和物理。"""
        first, second = self.game.tanks
        rng = self.game.rng
        rows, cols = self.game.maze.rows, self.game.maze.cols
        if self.spawn == "close_facing":
            mid_x, mid_y = cols * 0.5, rows * 0.5
            first.x, first.y = mid_x - 1.1, mid_y
            second.x, second.y = mid_x + 1.1, mid_y
            first.heading = float(rng.uniform(-0.2, 0.2))
            second.heading = math.pi + float(rng.uniform(-0.2, 0.2))
        elif self.spawn == "far_random":
            first.x, first.y = 1.5, 1.5
            second.x, second.y = cols - 1.5, rows - 1.5
            if rng.random() < 0.5:
                first.x, first.y, second.x, second.y = second.x, rows - 1.5, 1.5, 1.5
            first.heading = float(rng.uniform(-math.pi, math.pi))
            second.heading = float(rng.uniform(-math.pi, math.pi))
        for tank in self.game.tanks:
            tank.speed = 0.0
            tank.angular_velocity = 0.0
            tank.cooldown = 0.0
            tank.alive = True

    def step(self, actions: np.ndarray | list[tuple[int, int, int]]) -> tuple[list[Observation], np.ndarray, bool, dict[str, object]]:
        """执行联合动作，返回观察、两名智能体的奖励、结束标志和调试信息。"""
        action_array = np.asarray(actions, dtype=np.int64)
        if action_array.shape != (self.num_agents, 3):
            raise ValueError(f"actions must have shape ({self.num_agents}, 3)")
        rewards = np.zeros(self.num_agents, dtype=np.float32)
        config = self.reward_config
        previous_potential = self._potential.copy()
        all_events: list[dict[str, int]] = []
        frames_executed = 0

        for _ in range(self.action_repeat):
            if self.game.is_over:
                break
            events = self.game.update([tuple(map(int, action)) for action in action_array])
            all_events.extend(events)
            frames_executed += 1

        if self.game.is_over:
            death_shooter = {event["victim"]: event["shooter"] for event in all_events}
            if self.game.winner is None:
                if not all_events:
                    rewards += config.timeout
                else:
                    for index, tank_id in enumerate(self.agent_ids):
                        rewards[index] += config.self_kill if death_shooter.get(tank_id) == tank_id else config.loss
            else:
                for index, tank_id in enumerate(self.agent_ids):
                    if tank_id == self.game.winner:
                        rewards[index] += config.win
                    elif death_shooter.get(tank_id) == tank_id:
                        rewards[index] += config.self_kill
                    else:
                        rewards[index] += config.loss
        next_potential = np.zeros(self.num_agents, dtype=np.float32) if self.game.is_over else self._state_potential()
        if config.potential_scale != 0.0:
            rewards += config.potential_scale * (
                config.potential_gamma * next_potential - previous_potential
            )
        self._potential[:] = next_potential
        info: dict[str, object] = {
            "events": all_events,
            "winner": self.game.winner,
            "frames_executed": frames_executed,
        }
        return self._observations(), rewards, self.game.is_over, info

    def _alive_enemy(self, tank):
        """返回这辆坦克当前要打的存活敌人。"""
        for other in self.game.tanks:
            if other.tank_id != tank.tank_id and other.alive:
                return other
        return None

    def _cell_of(self, x: float, y: float) -> Cell:
        """把世界坐标限制并转换为迷宫格。"""
        maze = self.game.maze
        row = int(np.clip(math.floor(y), 0, maze.rows - 1))
        col = int(np.clip(math.floor(x), 0, maze.cols - 1))
        return row, col

    def _path_distance_xy(self, x0: float, y0: float, x1: float, y1: float) -> float:
        """沿 A* 折线从一点到另一点的连续世界距离。"""
        path = astar_path(self.game.maze, self._cell_of(x0, y0), self._cell_of(x1, y1))
        if len(path) <= 1:
            return math.hypot(x1 - x0, y1 - y0)
        points = [(col + 0.5, row + 0.5) for row, col in path[1:]]
        points.append((x1, y1))
        total = 0.0
        prev_x, prev_y = x0, y0
        for x, y in points:
            total += math.hypot(x - prev_x, y - prev_y)
            prev_x, prev_y = x, y
        return total

    def _state_potential(self) -> np.ndarray:
        """用路径距离和可射击朝向描述局势；只通过 gamma*Phi(s')-Phi(s) 进入奖励。"""
        result = np.zeros(self.num_agents, dtype=np.float32)
        distance_scale = max(float(self.game.maze.rows + self.game.maze.cols), 1.0)
        for index, tank in enumerate(self.game.tanks):
            enemy = self._alive_enemy(tank)
            if not tank.alive or enemy is None:
                continue
            distance = self._path_distance_xy(tank.x, tank.y, enemy.x, enemy.y)
            navigation = -min(distance / distance_scale, 1.0)
            has_los = not any(
                segment_intersects_rect(tank.x, tank.y, enemy.x, enemy.y, wall)
                for wall in self.game.wall_rects
            )
            target_heading = math.atan2(enemy.y - tank.y, enemy.x - tank.x)
            heading_error = (target_heading - tank.heading + math.pi) % (2.0 * math.pi) - math.pi
            aim = max(0.0, math.cos(heading_error)) if has_los else 0.0
            result[index] = 0.7 * navigation + 0.3 * aim
        return result

    def _observations(self) -> list[Observation]:
        """按稳定的智能体编号顺序构造当前观察。"""
        return [build_observation(self.game, tank_id) for tank_id in self.agent_ids]
