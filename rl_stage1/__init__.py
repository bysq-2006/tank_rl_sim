"""第一关：空场近距离对位。"""

from .environment import TankSelfPlayEnv
from .observation import MAP_CHANNELS, MAP_SIZE, build_observation

__all__ = ["MAP_CHANNELS", "MAP_SIZE", "TankSelfPlayEnv", "build_observation"]
