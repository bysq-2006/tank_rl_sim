from __future__ import annotations

from pathlib import Path


class TrainingPlot:
    """训练过程中弹出窗口，实时画 return100 / win100 / entropy。"""

    def __init__(self) -> None:
        import matplotlib.pyplot as plt

        self.plt = plt
        plt.ion()
        self.fig, self.axes = plt.subplots(3, 1, sharex=True, figsize=(8, 7))
        manager = getattr(self.fig.canvas, "manager", None)
        if manager is not None:
            manager.set_window_title("stage3 training")
        self.steps: list[int] = []
        self.returns: list[float] = []
        self.wins: list[float] = []
        self.entropies: list[float] = []
        titles = ("return100", "win100", "entropy")
        self.lines = []
        for axis, title in zip(self.axes, titles):
            (line,) = axis.plot([], [], color="#1f77b4", linewidth=1.4)
            axis.set_ylabel(title)
            axis.grid(True, alpha=0.3)
            self.lines.append(line)
        self.axes[-1].set_xlabel("step")
        self.fig.tight_layout()
        self.fig.show()

    def load_csv(self, path: Path) -> None:
        """续训时把已有日志画进去，避免窗口从空开始。"""
        import csv

        if not path.is_file():
            return
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    self.steps.append(int(float(row["global_step"])))
                    self.returns.append(float(row["mean_return_100"]))
                    self.wins.append(float(row["learner_win_100"]))
                    self.entropies.append(float(row["entropy"]))
                except (KeyError, ValueError):
                    continue
        if self.steps:
            self._redraw()

    def update(self, step: int, mean_return: float, win100: float, entropy: float) -> None:
        """追加一次更新的三个指标并刷新窗口。"""
        self.steps.append(int(step))
        self.returns.append(float(mean_return))
        self.wins.append(float(win100))
        self.entropies.append(float(entropy))
        self._redraw()

    def _redraw(self) -> None:
        series = (self.returns, self.wins, self.entropies)
        for line, axis, values in zip(self.lines, self.axes, series):
            line.set_data(self.steps, values)
            axis.relim()
            axis.autoscale_view()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self.plt.pause(0.001)

    def close(self) -> None:
        try:
            self.plt.close(self.fig)
        except Exception:
            pass


def try_create_plot() -> TrainingPlot | None:
    """打不开窗口时返回 None，训练继续。"""
    try:
        return TrainingPlot()
    except Exception as error:
        print(f"live plot disabled: {error}")
        return None
