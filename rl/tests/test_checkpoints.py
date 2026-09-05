import torch

from rl.curriculum import CurriculumManager
from rl.training.checkpoint import prune_preview_checkpoints, save_preview_checkpoint
from rl.training.dashboard import TrainingDashboard


def test_preview_checkpoints_are_watchable_and_rolling(tmp_path):
    # 验证预览文件包含模型与关卡信息并只保留指定数量的新版本。
    model = torch.nn.Linear(2, 1)
    curriculum = CurriculumManager(start_stage=2, seed=1)
    preview_directory = tmp_path / "previews"
    for update in (10, 20, 30):
        path = preview_directory / f"preview_update_{update:06d}_step_{update * 100:012d}.pt"
        save_preview_checkpoint(path, model, curriculum, update - 1, update * 100)

    removed = prune_preview_checkpoints(preview_directory, maximum=2)
    remaining = sorted(preview_directory.glob("*.pt"))
    assert len(removed) == 1
    assert [path.name for path in remaining] == [
        "preview_update_000020_step_000000002000.pt",
        "preview_update_000030_step_000000003000.pt",
    ]
    state = torch.load(remaining[-1], map_location="cpu", weights_only=False)
    assert state["preview"] is True
    assert state["curriculum"]["current_stage"] == 2
    assert state["total_steps"] == 3000


def test_dashboard_metrics_persist_and_resume_without_gui(tmp_path):
    # 验证关闭图形窗口时指标仍会持久化并能按续训轮次恢复。
    dashboard = TrainingDashboard(tmp_path, enabled=False, history_before_update=0)
    dashboard.record({"update": 1, "total_steps": 100, "stage": 0})
    dashboard.record({"update": 2, "total_steps": 200, "stage": 1})
    restored = TrainingDashboard(tmp_path, enabled=False, history_before_update=2)
    assert [record["update"] for record in restored.records] == [1]
    assert len((tmp_path / "training_metrics.jsonl").read_text(encoding="utf-8").splitlines()) == 2
