from __future__ import annotations

from collections import deque

import numpy as np

from .config import STAGES, StageConfig


class CurriculumManager:
    """抽取新旧课程任务并根据固定评估结果自动晋级。"""

    def __init__(self, start_stage: int = 0, seed: int = 0, fixed_stage: bool = False) -> None:
        # 初始化当前阶段、随机数生成器和每阶段的评估记录。
        if not 0 <= start_stage < len(STAGES):
            raise ValueError(f"invalid start stage {start_stage}")
        self.current_stage = start_stage
        self.fixed_stage = fixed_stage
        self.rng = np.random.default_rng(seed)
        self.results = {stage.index: deque(maxlen=stage.evaluation_games) for stage in STAGES}

    @property
    def current(self) -> StageConfig:
        # 返回当前主要训练阶段的完整配置。
        return STAGES[self.current_stage]

    def sample_stage(self) -> StageConfig:
        # 按当前六成、旧阶段三成和下一阶段一成抽取训练任务。
        if self.fixed_stage:
            return self.current
        value = float(self.rng.random())
        if value < 0.60:
            return self.current
        if value < 0.90 and self.current_stage > 0:
            previous = [stage for stage in STAGES[:self.current_stage] if stage.opponent != "idle"]
            if previous:
                return previous[int(self.rng.integers(0, len(previous)))]
            return self.current
        return STAGES[min(self.current_stage + 1, len(STAGES) - 1)]

    def replace_evaluation(self, stage_index: int, results: list[str]) -> None:
        # 用最新一轮固定种子评估结果替换该阶段的旧记录。
        history = self.results[stage_index]
        history.clear()
        history.extend(results)

    def statistics(self, stage_index: int | None = None) -> dict[str, float]:
        # 统计指定阶段近期评估的局数、胜率和超时率。
        index = self.current_stage if stage_index is None else stage_index
        results = self.results[index]
        games = len(results)
        return {
            "games": float(games),
            "win_rate": sum(item == "win" for item in results) / max(games, 1),
            "timeout_rate": sum(item == "timeout" for item in results) / max(games, 1),
        }

    def try_promote(self) -> bool:
        # 当前阶段达到胜率和超时率要求时推进到下一阶段。
        if self.fixed_stage:
            return False
        stage = self.current
        stats = self.statistics()
        passed = (
            stats["games"] >= stage.evaluation_games
            and stats["win_rate"] >= stage.promotion_win_rate
            and stats["timeout_rate"] <= stage.promotion_timeout_rate
        )
        if passed and self.current_stage < len(STAGES) - 1:
            self.current_stage += 1
            return True
        return False

    def set_stage(self, stage_index: int) -> None:
        # 手动切换当前阶段并检查阶段编号是否合法。
        if not 0 <= stage_index < len(STAGES):
            raise ValueError(f"invalid stage {stage_index}")
        self.current_stage = stage_index

    def state_dict(self) -> dict:
        # 导出可随检查点保存的课程进度和随机状态。
        return {
            "current_stage": self.current_stage,
            "fixed_stage": self.fixed_stage,
            "rng_state": self.rng.bit_generator.state,
            "results": {index: list(items) for index, items in self.results.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        # 从检查点恢复课程阶段、随机状态和近期评估记录。
        self.current_stage = int(state["current_stage"])
        self.fixed_stage = bool(state.get("fixed_stage", False))
        self.rng.bit_generator.state = state["rng_state"]
        for index, items in state.get("results", {}).items():
            self.results[int(index)].clear()
            self.results[int(index)].extend(items)
