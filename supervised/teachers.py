from __future__ import annotations

import numpy as np

from core import TankGame
from rl.opponents import script_action


class HunterTeacher:
    """寻路开火人机：A* 绕墙、看见就打、有弹就躲。"""

    name = "hunter"

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def action(self, game: TankGame, tank_id: int) -> tuple[int, int, int]:
        """给一辆坦克生成导师动作。"""
        return script_action(game, tank_id, "hunter", self.rng)
