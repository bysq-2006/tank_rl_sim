"""第三关：随机迷宫，观察和奖励与第一关同一套。"""

from .environment import TankSelfPlayEnv
from .observation import MAP_CHANNELS, MAP_SIZE, build_observation

__all__ = ["MAP_CHANNELS", "MAP_SIZE", "TankSelfPlayEnv", "build_observation"]
