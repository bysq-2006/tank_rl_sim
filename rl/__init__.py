"""强化学习包：迷宫对战，观察与人机脚本共用。"""

from .environment import TankSelfPlayEnv
from .observation import MAP_CHANNELS, MAP_SIZE, build_observation

__all__ = ["MAP_CHANNELS", "MAP_SIZE", "TankSelfPlayEnv", "build_observation"]
