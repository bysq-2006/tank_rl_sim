"""课程使用的脚本对手。"""

from .historical import FrozenModelOpponent, HistoricalOpponentPool, SnapshotInfo
from .scripted import ChaserOpponent, DodgerOpponent, IdleOpponent, RandomMoverOpponent, WeakShooterOpponent, make_opponent

__all__ = [
    "ChaserOpponent", "DodgerOpponent", "FrozenModelOpponent", "HistoricalOpponentPool",
    "IdleOpponent", "RandomMoverOpponent", "SnapshotInfo",
    "WeakShooterOpponent", "make_opponent",
]
