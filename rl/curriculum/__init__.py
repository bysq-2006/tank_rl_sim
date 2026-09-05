"""课程学习配置与进度管理。"""

from .config import STAGES, STAGE_TITLES, StageConfig
from .manager import CurriculumManager

__all__ = ["STAGES", "STAGE_TITLES", "StageConfig", "CurriculumManager"]
