"""Tank simulator reinforcement-learning components."""

from .model import TankActorCritic
from .observation import build_observation

__all__ = ["TankActorCritic", "build_observation"]
