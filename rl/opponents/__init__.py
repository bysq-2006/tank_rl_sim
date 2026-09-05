"""课程使用的脚本对手。"""

from .historical import FrozenModelOpponent, HistoricalOpponentPool, SnapshotInfo
from .scripted import ChaserOpponent, IdleOpponent, RandomMoverOpponent, WeakShooterOpponent, make_opponent

__all__ = [
    "ChaserOpponent", "FrozenModelOpponent", "HistoricalOpponentPool",
    "IdleOpponent", "RandomMoverOpponent", "SnapshotInfo",
    "WeakShooterOpponent", "make_opponent",
]
