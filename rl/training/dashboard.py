from __future__ import annotations

import json
import math
from pathlib import Path


class TrainingDashboard:
    """持久记录训练指标并在可用时显示实时折线图窗口。"""

    def __init__(
        self,
        output_directory: str | Path,
        enabled: bool = True,
        maximum_points: int = 1000,
        history_before_update: int | None = None,
    ) -> None:
        # 初始化指标文件、历史记录和非阻塞式Matplotlib窗口。
        self.output_directory = Path(output_directory)
        self.metrics_path = self.output_directory / "training_metrics.jsonl"
        self.image_path = self.output_directory / "training_dashboard.png"
        self.maximum_points = max(int(maximum_points), 20)
        self.records = self._load_records(history_before_update)
        self.enabled = False
        self.closed_by_user = False
        self.plt = None
        self.figure = None
        if enabled:
            self._open_window()

    def _load_records(self, history_before_update: int | None) -> list[dict]:
        # 从JSONL日志恢复旧指标并按更新轮次去重排序。
        if not self.metrics_path.exists():
            return []
        by_update: dict[int, dict] = {}
        with self.metrics_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    record = json.loads(line)
                    update = int(record["update"])
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
                if history_before_update is None or update < history_before_update:
                    by_update[update] = record
        return [by_update[index] for index in sorted(by_update)]

    def _open_window(self) -> None:
        # 延迟导入绘图库并创建不会阻塞训练循环的独立窗口。
        try:
            import matplotlib.pyplot as plt

            plt.rcParams["font.sans-serif"] = [
                "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans",
            ]
            plt.rcParams["axes.unicode_minus"] = False
            plt.ion()
            self.plt = plt
            self.figure = plt.figure(figsize=(13.5, 9.0), num="坦克强化学习实时训练指标")
            self.enabled = True
            self._draw()
            plt.show(block=False)
            plt.pause(0.001)
        except Exception:
            self.enabled = False
            self.plt = None
            self.figure = None

    def record(self, record: dict) -> None:
        # 追加一轮训练指标并刷新仍然打开的图表窗口。
        self.output_directory.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        update = int(record["update"])
        self.records = [item for item in self.records if int(item["update"]) != update]
        self.records.append(record)
        self.records.sort(key=lambda item: int(item["update"]))
        if len(self.records) > self.maximum_points:
            self.records = self.records[-self.maximum_points:]
        self._refresh_window()

    def _refresh_window(self) -> None:
        # 检测用户关闭事件并安全刷新窗口而不打断PPO训练。
        if not self.enabled or self.plt is None or self.figure is None:
            return
        if not self.plt.fignum_exists(self.figure.number):
            self.enabled = False
            self.closed_by_user = True
            return
        try:
            self._draw()
            self.figure.canvas.draw_idle()
            self.figure.canvas.flush_events()
            self.plt.pause(0.001)
        except Exception:
            self.enabled = False

    def _series(self, key: str) -> list[float]:
        # 将缺失指标转换成折线图可以跳过的NaN数值。
        values: list[float] = []
        for record in self.records:
            value = record.get(key)
            values.append(float(value) if value is not None else math.nan)
        return values

    def _sparse_series(self, key: str) -> tuple[list[int], list[float]]:
        # 过滤未评估轮次以把间隔出现的评估点连接成连续折线。
        points = [record for record in self.records if record.get(key) is not None]
        return (
            [int(record["update"]) for record in points],
            [float(record[key]) for record in points],
        )

    def _draw(self) -> None:
        # 使用六个子图绘制奖励、结果、损失、探索度、开炮和课程进度。
        if self.figure is None:
            return
        self.figure.clear()
        axes = self.figure.subplots(3, 2)
        updates = [int(record["update"]) for record in self.records]
        eval_reward_updates, eval_rewards = self._sparse_series("eval_mean_reward")
        eval_win_updates, eval_win_rates = self._sparse_series("eval_win_rate")
        eval_timeout_updates, eval_timeout_rates = self._sparse_series("eval_timeout_rate")

        axes[0, 0].plot(updates, self._series("mean_step_reward"), label="训练平均单步奖励")
        axes[0, 0].plot(eval_reward_updates, eval_rewards, "o-", label="评估平均累计奖励")
        axes[0, 0].set_title("奖励")
        axes[0, 0].legend(loc="best")

        axes[0, 1].plot(updates, self._series("rollout_win_rate"), label="训练完成局胜率")
        axes[0, 1].plot(eval_win_updates, eval_win_rates, "o-", label="评估胜率")
        axes[0, 1].plot(eval_timeout_updates, eval_timeout_rates, "o-", label="评估超时率")
        axes[0, 1].set_ylim(-0.03, 1.03)
        axes[0, 1].set_title("胜率与超时率")
        axes[0, 1].legend(loc="best")

        axes[1, 0].plot(updates, self._series("policy_loss"), label="策略损失")
        axes[1, 0].plot(updates, self._series("value_loss"), label="价值损失")
        axes[1, 0].set_title("PPO损失")
        axes[1, 0].legend(loc="best")

        axes[1, 1].plot(updates, self._series("entropy"), color="tab:purple", label="策略熵")
        kl_axis = axes[1, 1].twinx()
        kl_axis.plot(updates, self._series("approx_kl"), color="tab:red", label="近似KL")
        axes[1, 1].set_title("探索程度与更新幅度")
        axes[1, 1].legend(loc="upper left")
        kl_axis.legend(loc="upper right")

        axes[2, 0].plot(updates, self._series("shots_per_game"), label="每个完成对局平均开炮数")
        axes[2, 0].plot(
            updates, self._series("unsafe_shots_per_game"),
            color="tab:red", label="每个完成对局平均危险开炮数",
        )
        history_axis = axes[2, 0].twinx()
        history_axis.plot(
            updates, self._series("historical_opponent_ratio"),
            color="tab:orange", label="历史模型对局占比",
        )
        history_axis.set_ylim(-0.03, 1.03)
        axes[2, 0].set_title("开炮与历史对手")
        axes[2, 0].legend(loc="upper left")
        history_axis.legend(loc="upper right")

        axes[2, 1].step(updates, self._series("stage"), where="post", label="当前关卡")
        learning_axis = axes[2, 1].twinx()
        learning_axis.plot(
            updates, self._series("learning_rate"),
            color="tab:green", label="学习率",
        )
        axes[2, 1].set_ylim(-0.2, 6.2)
        axes[2, 1].set_title("课程进度与学习率")
        axes[2, 1].legend(loc="upper left")
        learning_axis.legend(loc="upper right")

        for axis in axes.flat:
            axis.set_xlabel("PPO更新轮次")
            axis.grid(alpha=0.25)
        latest = self.records[-1] if self.records else {}
        self.figure.suptitle(
            f"坦克强化学习实时指标｜累计决策步数：{int(latest.get('total_steps', 0)):,}",
            fontsize=14,
        )
        self.figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))

    def save_image(self) -> None:
        # 将当前图表保存为PNG以便训练结束或窗口关闭后查看。
        if self.figure is None or not self.records:
            return
        try:
            self._draw()
            self.figure.savefig(self.image_path, dpi=120)
        except Exception:
            return
