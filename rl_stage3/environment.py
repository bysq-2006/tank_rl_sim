from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from core import TankGame
from core.maze import Maze

from .observation import Observation, build_observation

LAYOUTS = ("maze", "open")
SPAWNS = ("default", "close_facing", "far_random")


def make_open_maze(rows: int, cols: int) -> Maze:
    """只保留外墙的空场，用于分关训练的前两关。"""
    horizontal = np.zeros((rows + 1, cols), dtype=np.bool_)
    vertical = np.zeros((rows, cols + 1), dtype=np.bool_)
    horizontal[0] = True
    horizontal[-1] = True
    vertical[:, 0] = True
    vertical[:, -1] = True
    return Maze(rows, cols, horizontal, vertical)


@dataclass
class RewardConfig:
    """控制奖励各部分的权重。"""

    win: float = 2.0  # 最终存活并获胜的奖励，与最后一颗子弹由谁发射无关。
    loss: float = -1.0  # 被敌人击毁或双方同时死亡时的惩罚。
    self_kill: float = -0.1  # 被自己的子弹击毁时的小额惩罚，小于被敌人击毁。
    timeout: float = -1.0  # 对局超时的双方惩罚，避免不开火拖延成为稳定策略。
    fire_bonus: float = 0.003  # 每颗真正生成的子弹获得的小额探索奖励。


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

    def reset(self, seed: int | None = None) -> list[Observation]:
        """开始新对局并返回两辆坦克各自视角的观察。"""
        self.game.reset(seed=seed)
        if self.layout == "open":
            self.game.maze = make_open_maze(self.game.maze.rows, self.game.maze.cols)
            self.game.wall_rects = self.game.maze.wall_rects(self.game.wall_thickness)
        self._place_tanks()
        self.agent_ids = tuple(tank.tank_id for tank in self.game.tanks)
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
        all_events: list[dict[str, int]] = []
        frames_executed = 0

        for _ in range(self.action_repeat):
            if self.game.is_over:
                break
            existing_bullets = {id(bullet) for bullet in self.game.bullets}
            events = self.game.update([tuple(map(int, action)) for action in action_array])
            all_events.extend(events)
            frames_executed += 1
            for bullet in self.game.bullets:
                if id(bullet) not in existing_bullets:
                    owner_index = self.agent_ids.index(bullet.owner_tank_id)
                    rewards[owner_index] += config.fire_bonus

        if self.game.is_over:
            death_shooter = {event["victim"]: event["shooter"] for event in all_events}
            if self.game.winner is None:
                # 超时不给安全收益；同时死亡时自杀用较小惩罚，其余按失败处理。
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
        info: dict[str, object] = {
            "events": all_events,
            "winner": self.game.winner,
            "frames_executed": frames_executed,
        }
        return self._observations(), rewards, self.game.is_over, info

    def _observations(self) -> list[Observation]:
        """按稳定的智能体编号顺序构造当前观察。"""
        return [build_observation(self.game, tank_id) for tank_id in self.agent_ids]
