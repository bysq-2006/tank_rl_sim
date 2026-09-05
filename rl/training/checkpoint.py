from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(path: str | Path, model, optimizer, curriculum, update: int, total_steps: int) -> None:
    # 原子保存模型、优化器、课程进度和训练步数。
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "curriculum": curriculum.state_dict(),
        "update": update,
        "total_steps": total_steps,
    }, temporary)
    temporary.replace(destination)


def load_checkpoint(path: str | Path, model, optimizer, curriculum, device: torch.device) -> tuple[int, int]:
    # 恢复完整训练状态并返回下一次更新编号和累计步数。
    state = torch.load(Path(path), map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    curriculum.load_state_dict(state["curriculum"])
    return int(state["update"]) + 1, int(state["total_steps"])


def save_preview_checkpoint(
    path: str | Path,
    model,
    curriculum,
    update: int,
    total_steps: int,
) -> None:
    # 原子保存可直接观战且不含优化器状态的轻量预览模型。
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save({
        "model": model.state_dict(),
        "curriculum": curriculum.state_dict(),
        "update": update,
        "total_steps": total_steps,
        "preview": True,
    }, temporary)
    temporary.replace(destination)


def prune_preview_checkpoints(directory: str | Path, maximum: int) -> list[Path]:
    # 按更新时间只保留最近若干预览模型并返回被清理的旧文件。
    if maximum <= 0:
        return []
    candidates = sorted(Path(directory).glob("preview_update_*_step_*.pt"))
    removed = candidates[:-maximum]
    for path in removed:
        path.unlink()
    return removed
