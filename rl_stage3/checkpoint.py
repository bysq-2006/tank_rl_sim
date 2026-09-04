from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def atomic_torch_save(payload: Any, path: Path) -> None:
    """完整写入并验证临时文件后再替换目标，避免中断损坏现有断点。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload, temporary_path)
        # 重新读取一次，确保 zip 记录和 pickle 均完整后才发布为正式断点。
        torch.load(temporary_path, map_location="cpu", weights_only=False)
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
