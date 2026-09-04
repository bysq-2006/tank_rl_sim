"""第二关：空场远距离随机朝向。"""

from .environment import TankSelfPlayEnv
from .observation import MAP_CHANNELS, MAP_SIZE, build_observation

__all__ = ["MAP_CHANNELS", "MAP_SIZE", "TankSelfPlayEnv", "build_observation"]
