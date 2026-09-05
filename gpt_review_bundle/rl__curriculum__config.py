from __future__ import annotations

from dataclasses import dataclass

from rl.rewards import RewardConfig


@dataclass(frozen=True)
class StageConfig:
    """一个课程阶段的地图、对手、奖励和晋级参数。"""

    index: int
    name: str
    rows: tuple[int, int]
    cols: tuple[int, int]
    layout: str
    spawn: str
    opponent: str
    opponent_fire_probability: float
    reward: RewardConfig
    promotion_win_rate: float
    promotion_timeout_rate: float
    evaluation_games: int = 200
    historical_opponent_probability: float = 0.0


def _reward(kill: float, shaping: float, death: float | None = None) -> RewardConfig:
    # 按阶段生成击杀、死亡和势函数奖励配置，死亡惩罚可单独加强。
    death_penalty = -kill if death is None else -death
    return RewardConfig(kill_bonus=kill, death_penalty=death_penalty, shaping_scale=shaping)


STAGES: tuple[StageConfig, ...] = (
    StageConfig(0, "basic_target", (6, 6), (6, 6), "open", "far_random", "idle", 0.0, _reward(0.30, 0.40), 0.90, 0.08, historical_opponent_probability=0.0),
    StageConfig(1, "random_aim", (6, 6), (6, 6), "open", "random_heading", "idle", 0.0, _reward(0.30, 0.40), 0.90, 0.08, historical_opponent_probability=0.15),
    StageConfig(2, "move_and_aim", (6, 6), (6, 6), "open", "far_random", "idle", 0.0, _reward(0.25, 0.30), 0.85, 0.10, historical_opponent_probability=0.25),
    StageConfig(3, "simple_walls", (6, 6), (6, 6), "simple", "far_random", "dodger", 0.25, _reward(0.20, 0.25, 0.35), 0.80, 0.12, historical_opponent_probability=0.0),
    StageConfig(4, "moving_enemy", (6, 7), (6, 7), "maze", "random", "random_mover", 0.0, _reward(0.15, 0.20, 0.30), 0.75, 0.15, historical_opponent_probability=0.45),
    StageConfig(5, "weak_shooter", (6, 7), (6, 7), "maze", "random", "weak_shooter", 0.10, _reward(0.10, 0.10, 0.25), 0.65, 0.15, historical_opponent_probability=0.55),
    StageConfig(6, "full_scripted", (6, 9), (6, 9), "maze", "random", "chaser", 0.80, _reward(0.05, 0.05, 0.20), 0.60, 0.15, historical_opponent_probability=0.65),
)

STAGE_TITLES: tuple[str, ...] = (
    "远距离随机朝向基础靶场",
    "随机朝向瞄准",
    "远距离移动并瞄准",
    "简单墙体寻路",
    "追踪移动敌人",
    "对抗低频射击敌人",
    "完整迷宫脚本对战",
)
