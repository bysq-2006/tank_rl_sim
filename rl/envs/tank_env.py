from __future__ import annotations

from typing import Any

import numpy as np

from core import TankGame
from rl.curriculum import StageConfig
from rl.observation import Observation, build_observation
from rl.opponents import FrozenModelOpponent, make_opponent
from rl.rewards import RewardTracker

from .scenarios import apply_layout, apply_spawn


class TankRLEnv:
    """将纯物理核心包装为单智能体课程学习环境。"""

    def __init__(self, action_repeat: int = 2, controlled_tank_id: int = 0, opponent_pool=None) -> None:
        # 初始化动作重复次数和受控坦克编号。
        if action_repeat < 1:
            raise ValueError("action_repeat must be positive")
        self.action_repeat = action_repeat
        self.controlled_tank_id = controlled_tank_id
        self.opponent_pool = opponent_pool
        self.opponent_tank_id = 1
        self.game: TankGame | None = None
        self.stage: StageConfig | None = None
        self.opponent = None
        self.reward_tracker: RewardTracker | None = None
        self.rng = np.random.default_rng()

    def reset(self, stage: StageConfig, seed: int | None = None) -> Observation:
        # 根据阶段随机生成尺寸、场景、出生点、对手和奖励跟踪器。
        self.stage = stage
        self.rng = np.random.default_rng(seed)
        rows = int(self.rng.integers(stage.rows[0], stage.rows[1] + 1))
        cols = int(self.rng.integers(stage.cols[0], stage.cols[1] + 1))
        self.game = TankGame(rows=rows, cols=cols)
        self.game.reset(seed)
        apply_layout(self.game, stage.layout, self.rng)
        apply_spawn(self.game, stage.spawn, self.rng)
        if self.opponent_pool is None:
            self.opponent = make_opponent(stage.opponent, self.rng, stage.opponent_fire_probability)
        else:
            self.opponent = self.opponent_pool.make_opponent(stage, self.rng)
        self.reward_tracker = RewardTracker(self.controlled_tank_id, stage.reward)
        self.reward_tracker.reset(self.game)
        return build_observation(self.game, self.controlled_tank_id)

    def step(
        self,
        action: tuple[int, int, int] | list[int] | np.ndarray,
        opponent_action: tuple[int, int, int] | list[int] | np.ndarray | None = None,
    ) -> tuple[Observation, float, bool, dict[str, Any]]:
        # 执行一次模型动作及若干物理帧并返回标准强化学习结果。
        if self.game is None or self.stage is None or self.opponent is None or self.reward_tracker is None:
            raise RuntimeError("reset must be called before step")
        events: list[dict] = []
        frames = 0
        for _ in range(self.action_repeat):
            current_opponent_action = (
                self.opponent.act(self.game, self.opponent_tank_id)
                if opponent_action is None
                else tuple(map(int, opponent_action))
            )
            controls = [tuple(map(int, action)), current_opponent_action]
            events.extend(self.game.update(controls))
            frames += 1
            if self.game.is_over:
                break
        reward = self.reward_tracker.calculate(self.game, events)
        done = self.game.is_over
        info = {
            "stage": self.stage.index,
            "stage_name": self.stage.name,
            "opponent": getattr(self.opponent, "name", type(self.opponent).__name__),
            "historical_opponent": isinstance(self.opponent, FrozenModelOpponent),
            "shots_fired": self.game.shots_fired_by_tank.get(self.controlled_tank_id, 0),
            "opponent_self_kill": self.reward_tracker.opponent_self_killed,
            "result": self._result() if done else None,
            "events": events,
            "frames_executed": frames,
        }
        return build_observation(self.game, self.controlled_tank_id), reward, done, info

    def _result(self) -> str:
        # 将游戏终局状态转换成课程评估使用的结果名称。
        assert self.game is not None
        if self.game.winner == self.controlled_tank_id and self.reward_tracker is not None:
            if self.reward_tracker.opponent_self_killed:
                return "opponent_self_kill"
        if self.game.winner == self.controlled_tank_id:
            return "win"
        if self.game.winner is not None:
            return "loss"
        if self.game.elapsed >= self.game.time_limit:
            return "timeout"
        return "draw"
