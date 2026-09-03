from __future__ import annotations

import math
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk


@dataclass
class PointSample:
    """某一视频帧中的一个位置样本。"""

    frame: int
    x: float
    y: float


@dataclass
class DirectionSample:
    """某一视频帧中由起点和终点定义的朝向样本。"""

    frame: int
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def angle(self) -> float:
        """返回图像坐标系中的方向角，向右为 0，顺时针为正。"""
        return math.atan2(self.y1 - self.y0, self.x1 - self.x0)


def calculate_linear_speed(first: PointSample, second: PointSample, fps: float, unit_pixels: float) -> tuple[float, float]:
    """返回线位移（单位距离）和线速度（单位距离/秒）。"""
    elapsed = abs(second.frame - first.frame) / fps
    if elapsed <= 0.0:
        raise ValueError("两个样本必须位于不同时间")
    distance = math.hypot(second.x - first.x, second.y - first.y) / unit_pixels
    return distance, distance / elapsed


def calculate_angular_speed(first: DirectionSample, second: DirectionSample, fps: float) -> tuple[float, float]:
    """返回最短有符号转角和角速度，单位分别是度和度/秒。"""
    elapsed = abs(second.frame - first.frame) / fps
    if elapsed <= 0.0:
        raise ValueError("两个样本必须位于不同时间")
    delta = math.atan2(math.sin(second.angle - first.angle), math.cos(second.angle - first.angle))
    degrees = math.degrees(delta)
    return degrees, degrees / elapsed


def calculate_distance(start: tuple[float, float], end: tuple[float, float], unit_pixels: float) -> tuple[float, float]:
    """返回两点之间的像素距离和单位格距离。"""
    if unit_pixels <= 0.0:
        raise ValueError("单位像素长度必须大于 0")
    pixels = math.hypot(end[0] - start[0], end[1] - start[1])
    return pixels, pixels / unit_pixels


class VideoSpeedAnalyzer:
    """通过视频中的人工标记测量线速度和角速度。"""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("视频运动速度测量")
        self.root.geometry("1100x780")
        self.root.minsize(760, 560)

        self.capture: cv2.VideoCapture | None = None  # 当前打开的视频。
        self.video_path: Path | None = None  # 当前视频文件路径。
        self.fps = 24.0  # 视频帧率，打开文件后从视频读取。
        self.frame_count = 0  # 视频总帧数。
        self.current_frame = 0  # 当前显示的帧编号。
        self.source_width = 0  # 原视频宽度。
        self.source_height = 0  # 原视频高度。
        self.current_bgr = None  # 当前帧的 OpenCV BGR 图像。
        self.photo: ImageTk.PhotoImage | None = None  # 防止 Tk 图片被垃圾回收。
        self.display_scale = 1.0  # 原视频坐标到画布坐标的缩放比例。
        self.display_left = 0.0  # 画面在画布中的左侧留白。
        self.display_top = 0.0  # 画面在画布中的顶部留白。
        self.mode: str | None = None  # calibrate、linear、angular、distance 或 None。
        self.drag_start: tuple[float, float] | None = None  # 当前拖动在原视频中的起点。
        self.drag_current: tuple[float, float] | None = None  # 当前拖动在原视频中的终点。
        self.unit_square: tuple[float, float, float, float] | None = None  # 一个单位距离的正方形。
        self.unit_pixels: float | None = None  # 一个单位距离对应的原视频像素数。
        self.linear_samples: list[PointSample] = []  # 线速度的两个位置样本。
        self.angular_samples: list[DirectionSample] = []  # 角速度的两条方向样本。
        self.distance_sample: DirectionSample | None = None  # 最近一次静态距离测量线段。
        self.playing = False  # 是否正在自动播放视频。
        self.play_job: str | None = None  # Tk 自动播放定时任务编号。

        self.status_text = tk.StringVar(value="打开视频后，先点击“1 标定单位”。")
        self.time_text = tk.StringVar(value="00:00.000 / 00:00.000   帧 0 / 0")
        self._build_ui()

    def _build_ui(self) -> None:
        """创建工具栏、视频画布、时间轴和状态栏。"""
        toolbar = ttk.Frame(self.root, padding=6)
        toolbar.pack(fill=tk.X)
        ttk.Button(toolbar, text="打开视频", command=self.open_video).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="播放/暂停", command=self.toggle_play).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="上一帧", command=lambda: self.seek(self.current_frame - 1)).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="下一帧", command=lambda: self.seek(self.current_frame + 1)).pack(side=tk.LEFT, padx=3)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Button(toolbar, text="1 标定单位", command=lambda: self.set_mode("calibrate")).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="2 测线速度", command=lambda: self.set_mode("linear")).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="3 测角速度", command=lambda: self.set_mode("angular")).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="4 测距离", command=lambda: self.set_mode("distance")).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="清除测量", command=self.clear_measurements).pack(side=tk.LEFT, padx=8)

        self.canvas = tk.Canvas(self.root, background="#17191d", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=(2, 4))
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Configure>", lambda _event: self.render_frame())

        timeline_frame = ttk.Frame(self.root, padding=(8, 2))
        timeline_frame.pack(fill=tk.X)
        self.timeline = tk.Canvas(timeline_frame, height=46, background="#30343b", highlightthickness=0, cursor="hand2")
        self.timeline.pack(fill=tk.X)
        self.timeline.bind("<Button-1>", self.on_timeline_click)
        ttk.Label(timeline_frame, textvariable=self.time_text, anchor=tk.CENTER).pack(fill=tk.X, pady=(2, 0))
        ttk.Label(self.root, textvariable=self.status_text, anchor=tk.W, padding=(10, 5)).pack(fill=tk.X)

        self.root.bind("<Left>", lambda _event: self.seek(self.current_frame - 1))
        self.root.bind("<Right>", lambda _event: self.seek(self.current_frame + 1))
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def open_video(self) -> None:
        """选择视频文件，读取基础信息并显示第一帧。"""
        filename = filedialog.askopenfilename(
            title="选择视频",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv *.webm"), ("所有文件", "*.*")],
        )
        if not filename:
            return
        capture = cv2.VideoCapture(filename)
        if not capture.isOpened():
            messagebox.showerror("打开失败", "无法读取这个视频文件。")
            return
        if self.capture is not None:
            self.capture.release()
        self.capture = capture
        self.video_path = Path(filename)
        self.fps = float(capture.get(cv2.CAP_PROP_FPS)) or 24.0
        self.frame_count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.clear_measurements()
        self.seek(0)
        self.status_text.set(f"已打开 {self.video_path.name}，帧率 {self.fps:.3f} FPS。请先标定单位距离。")

    def seek(self, frame: int) -> None:
        """跳到指定帧并刷新视频画面和时间轴。"""
        if self.capture is None:
            return
        self.current_frame = max(0, min(int(frame), self.frame_count - 1))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ok, image = self.capture.read()
        if not ok:
            self.status_text.set(f"无法读取第 {self.current_frame} 帧。")
            return
        self.current_bgr = image
        self.render_frame()
        self.draw_timeline()
        current_seconds = self.current_frame / self.fps
        total_seconds = max(0, self.frame_count - 1) / self.fps
        self.time_text.set(
            f"{self._format_time(current_seconds)} / {self._format_time(total_seconds)}   "
            f"帧 {self.current_frame} / {self.frame_count - 1}"
        )

    def render_frame(self) -> None:
        """按画布大小缩放当前帧，并重画所有测量标记。"""
        if self.current_bgr is None:
            return
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        self.display_scale = min(canvas_width / self.source_width, canvas_height / self.source_height)
        shown_width = max(1, int(round(self.source_width * self.display_scale)))
        shown_height = max(1, int(round(self.source_height * self.display_scale)))
        self.display_left = (canvas_width - shown_width) / 2.0
        self.display_top = (canvas_height - shown_height) / 2.0
        rgb = cv2.cvtColor(self.current_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb).resize((shown_width, shown_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(self.display_left, self.display_top, image=self.photo, anchor=tk.NW)
        self._draw_overlays()

    def set_mode(self, mode: str) -> None:
        """切换标定、线速度或角速度操作模式。"""
        if self.capture is None:
            messagebox.showinfo("尚未打开视频", "请先打开一段视频。")
            return
        self.mode = mode
        if mode == "calibrate":
            self.status_text.set("标定：在画面中按住左键拖出正方形，其边长代表一个单位距离。")
        elif mode == "linear":
            self.linear_samples.clear()
            self.status_text.set("线速度：选好第一帧后点物体位置，再选第二帧并再次点击。")
        elif mode == "angular":
            self.angular_samples.clear()
            self.status_text.set("角速度：选好第一帧并拖出朝向，再到第二帧拖出新的朝向。")
        else:
            self.distance_sample = None
            self.status_text.set("距离：在当前画面中，从物体的一端拖到另一端。")
        self.render_frame()
        self.draw_timeline()

    def on_canvas_press(self, event: tk.Event) -> None:
        """记录框选或方向拖动起点；线速度模式直接记录位置点。"""
        point = self._canvas_to_video(event.x, event.y)
        if point is None or self.mode is None:
            return
        if self.mode == "linear":
            self._add_linear_sample(*point)
            return
        self.drag_start = point
        self.drag_current = point

    def on_canvas_drag(self, event: tk.Event) -> None:
        """拖动时实时预览单位方框或方向箭头。"""
        if self.drag_start is None:
            return
        point = self._canvas_to_video(event.x, event.y, clamp=True)
        if point is not None:
            self.drag_current = point
            self.render_frame()

    def on_canvas_release(self, event: tk.Event) -> None:
        """结束框选或方向拖动，并保存当前帧的样本。"""
        if self.drag_start is None or self.mode not in ("calibrate", "angular", "distance"):
            return
        point = self._canvas_to_video(event.x, event.y, clamp=True)
        if point is None:
            self.drag_start = None
            return
        if self.mode == "calibrate":
            self._finish_calibration(self.drag_start, point)
        elif self.mode == "angular":
            self._add_angular_sample(self.drag_start, point)
        else:
            self._finish_distance(self.drag_start, point)
        self.drag_start = None
        self.drag_current = None
        self.render_frame()
        self.draw_timeline()

    def _finish_calibration(self, start: tuple[float, float], end: tuple[float, float]) -> None:
        """把拖动范围修正为正方形，并将其边长保存为一个单位。"""
        dx, dy = end[0] - start[0], end[1] - start[1]
        side = max(abs(dx), abs(dy))
        if side < 2.0:
            self.status_text.set("标定方框太小，请重新拖动。")
            return
        x1 = start[0] + math.copysign(side, dx if dx else 1.0)
        y1 = start[1] + math.copysign(side, dy if dy else 1.0)
        self.unit_square = (start[0], start[1], x1, y1)
        self.unit_pixels = side
        self.status_text.set(f"标定完成：1 单位 = {side:.2f} 像素。现在可以测量线速度或角速度。")

    def _add_linear_sample(self, x: float, y: float) -> None:
        """加入一个位置样本；已有两个样本时自动开始一组新测量。"""
        if self.unit_pixels is None:
            self.status_text.set("请先使用“1 标定单位”框选一个单位距离。")
            return
        if len(self.linear_samples) >= 2:
            self.linear_samples.clear()
        self.linear_samples.append(PointSample(self.current_frame, x, y))
        if len(self.linear_samples) == 1:
            self.status_text.set("已记录线速度起点。请移动时间轴，再点击物体的新位置。")
        else:
            try:
                distance, speed = calculate_linear_speed(self.linear_samples[0], self.linear_samples[1], self.fps, self.unit_pixels)
                elapsed = abs(self.linear_samples[1].frame - self.linear_samples[0].frame) / self.fps
                self.status_text.set(f"线位移 {distance:.4f} 单位，时间 {elapsed:.4f} 秒，线速度 {speed:.4f} 单位/秒。")
            except ValueError as error:
                self.linear_samples.pop()
                self.status_text.set(str(error))
        self.render_frame()
        self.draw_timeline()

    def _add_angular_sample(self, start: tuple[float, float], end: tuple[float, float]) -> None:
        """加入一条方向向量；收集两条后计算最短角速度。"""
        if math.hypot(end[0] - start[0], end[1] - start[1]) < 2.0:
            self.status_text.set("方向箭头太短，请重新拖动。")
            return
        if len(self.angular_samples) >= 2:
            self.angular_samples.clear()
        self.angular_samples.append(DirectionSample(self.current_frame, *start, *end))
        if len(self.angular_samples) == 1:
            self.status_text.set("已记录第一条朝向。请移动时间轴，再拖出第二条朝向。")
        else:
            try:
                angle, speed = calculate_angular_speed(self.angular_samples[0], self.angular_samples[1], self.fps)
                elapsed = abs(self.angular_samples[1].frame - self.angular_samples[0].frame) / self.fps
                direction = "顺时针" if speed > 0 else "逆时针" if speed < 0 else "未旋转"
                self.status_text.set(f"转角 {angle:.3f}°，时间 {elapsed:.4f} 秒，角速度 {speed:.3f}°/秒（{direction}）。")
            except ValueError as error:
                self.angular_samples.pop()
                self.status_text.set(str(error))

    def _finish_distance(self, start: tuple[float, float], end: tuple[float, float]) -> None:
        """保存当前距离线段，并按标定比例换算成单位格。"""
        if self.unit_pixels is None:
            self.status_text.set("请先使用“1 标定单位”框选一个单位格。")
            return
        pixels, units = calculate_distance(start, end, self.unit_pixels)
        if pixels < 2.0:
            self.status_text.set("测量线段太短，请重新拖动。")
            return
        self.distance_sample = DirectionSample(self.current_frame, *start, *end)
        self.status_text.set(f"距离 {pixels:.2f} 像素 = {units:.4f} 单位格。")

    def _draw_overlays(self) -> None:
        """在视频画面上绘制标定框、位置点、方向箭头和拖动预览。"""
        if self.unit_square is not None:
            x0, y0, x1, y1 = self.unit_square
            self.canvas.create_rectangle(*self._video_box_to_canvas(x0, y0, x1, y1), outline="#26e07f", width=2)
            tx, ty = self._video_to_canvas(x0, y0)
            self.canvas.create_text(tx + 4, ty + 4, text="1 单位", fill="#26e07f", anchor=tk.NW)
        for index, sample in enumerate(self.linear_samples, start=1):
            x, y = self._video_to_canvas(sample.x, sample.y)
            self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, outline="#ffd447", width=3)
            self.canvas.create_text(x + 8, y - 8, text=f"P{index}", fill="#ffd447", anchor=tk.SW)
        for index, sample in enumerate(self.angular_samples, start=1):
            self._draw_arrow(sample.x0, sample.y0, sample.x1, sample.y1, f"A{index}", "#4bc8ff")
        if self.distance_sample is not None:
            sample = self.distance_sample
            self._draw_measurement_line(sample.x0, sample.y0, sample.x1, sample.y1)
        if self.drag_start is not None and self.drag_current is not None:
            if self.mode == "calibrate":
                start, end = self.drag_start, self.drag_current
                side = max(abs(end[0] - start[0]), abs(end[1] - start[1]))
                x1 = start[0] + math.copysign(side, end[0] - start[0] or 1.0)
                y1 = start[1] + math.copysign(side, end[1] - start[1] or 1.0)
                self.canvas.create_rectangle(*self._video_box_to_canvas(start[0], start[1], x1, y1), outline="#ffffff", dash=(5, 3), width=2)
            elif self.mode == "angular":
                self._draw_arrow(*self.drag_start, *self.drag_current, "", "#ffffff")
            elif self.mode == "distance":
                self._draw_measurement_line(*self.drag_start, *self.drag_current, preview=True)

    def _draw_arrow(self, x0: float, y0: float, x1: float, y1: float, label: str, color: str) -> None:
        """在画布上绘制一条带箭头的方向向量。"""
        sx, sy = self._video_to_canvas(x0, y0)
        ex, ey = self._video_to_canvas(x1, y1)
        self.canvas.create_line(sx, sy, ex, ey, fill=color, width=3, arrow=tk.LAST, arrowshape=(12, 15, 5))
        if label:
            self.canvas.create_text(sx + 7, sy - 7, text=label, fill=color, anchor=tk.SW)

    def _draw_measurement_line(self, x0: float, y0: float, x1: float, y1: float, preview: bool = False) -> None:
        """绘制带有两个端点和距离文字的静态测量线。"""
        sx, sy = self._video_to_canvas(x0, y0)
        ex, ey = self._video_to_canvas(x1, y1)
        color = "#ffffff" if preview else "#ff63d8"
        self.canvas.create_line(sx, sy, ex, ey, fill=color, width=3, dash=(5, 3) if preview else ())
        for x, y in ((sx, sy), (ex, ey)):
            self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill=color, outline=color)
        if self.unit_pixels is not None:
            _, units = calculate_distance((x0, y0), (x1, y1), self.unit_pixels)
            self.canvas.create_text((sx + ex) / 2, (sy + ey) / 2 - 8, text=f"{units:.4f} 格", fill=color, anchor=tk.S)

    def draw_timeline(self) -> None:
        """绘制时间轴、当前播放头以及两种测量的帧标记。"""
        self.timeline.delete("all")
        width = max(1, self.timeline.winfo_width())
        left, right, y = 12, width - 12, 25
        self.timeline.create_line(left, y, right, y, fill="#aeb4be", width=3)
        if self.frame_count <= 1:
            return

        def frame_x(frame: int) -> float:
            return left + frame / (self.frame_count - 1) * (right - left)

        for index, sample in enumerate(self.linear_samples, start=1):
            x = frame_x(sample.frame)
            self.timeline.create_line(x, 8, x, 39, fill="#ffd447", width=3)
            self.timeline.create_text(x, 6, text=f"P{index}", fill="#ffd447", anchor=tk.S)
        for index, sample in enumerate(self.angular_samples, start=1):
            x = frame_x(sample.frame)
            self.timeline.create_line(x, 8, x, 39, fill="#4bc8ff", width=3)
            self.timeline.create_text(x, 6, text=f"A{index}", fill="#4bc8ff", anchor=tk.S)
        playhead = frame_x(self.current_frame)
        self.timeline.create_polygon(playhead - 6, 0, playhead + 6, 0, playhead, 9, fill="#ff5a5f")
        self.timeline.create_line(playhead, 9, playhead, 43, fill="#ff5a5f", width=2)

    def on_timeline_click(self, event: tk.Event) -> None:
        """把时间轴点击位置转换成帧编号并跳转。"""
        if self.frame_count <= 1:
            return
        width = max(1, self.timeline.winfo_width())
        ratio = (event.x - 12) / max(1, width - 24)
        self.seek(round(max(0.0, min(1.0, ratio)) * (self.frame_count - 1)))

    def toggle_play(self) -> None:
        """开始或暂停按视频原始帧率播放。"""
        if self.capture is None:
            return
        self.playing = not self.playing
        if self.playing:
            self._play_next()
        elif self.play_job is not None:
            self.root.after_cancel(self.play_job)
            self.play_job = None

    def _play_next(self) -> None:
        """播放下一帧，并按 FPS 安排下一次调用。"""
        if not self.playing:
            return
        if self.current_frame >= self.frame_count - 1:
            self.playing = False
            return
        self.seek(self.current_frame + 1)
        self.play_job = self.root.after(max(1, round(1000 / self.fps)), self._play_next)

    def clear_measurements(self) -> None:
        """删除单位标定和全部速度样本，但保留已打开的视频。"""
        self.mode = None
        self.drag_start = None
        self.drag_current = None
        self.unit_square = None
        self.unit_pixels = None
        self.linear_samples.clear()
        self.angular_samples.clear()
        self.distance_sample = None
        self.status_text.set("测量已清除。请先点击“1 标定单位”。")
        self.render_frame()
        self.draw_timeline()

    def _canvas_to_video(self, x: float, y: float, clamp: bool = False) -> tuple[float, float] | None:
        """把画布坐标转换为原视频坐标。"""
        if self.current_bgr is None:
            return None
        vx = (x - self.display_left) / self.display_scale
        vy = (y - self.display_top) / self.display_scale
        if clamp:
            return max(0.0, min(self.source_width - 1.0, vx)), max(0.0, min(self.source_height - 1.0, vy))
        if not (0.0 <= vx < self.source_width and 0.0 <= vy < self.source_height):
            return None
        return vx, vy

    def _video_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        """把原视频坐标转换为画布坐标。"""
        return self.display_left + x * self.display_scale, self.display_top + y * self.display_scale

    def _video_box_to_canvas(self, x0: float, y0: float, x1: float, y1: float) -> tuple[float, float, float, float]:
        """把原视频中的矩形转换成画布矩形。"""
        left, top = self._video_to_canvas(x0, y0)
        right, bottom = self._video_to_canvas(x1, y1)
        return left, top, right, bottom

    @staticmethod
    def _format_time(seconds: float) -> str:
        """把秒数格式化为 分:秒.毫秒。"""
        minutes = int(seconds // 60)
        remainder = seconds - minutes * 60
        return f"{minutes:02d}:{remainder:06.3f}"

    def close(self) -> None:
        """释放视频资源并关闭窗口。"""
        self.playing = False
        if self.play_job is not None:
            self.root.after_cancel(self.play_job)
        if self.capture is not None:
            self.capture.release()
        self.root.destroy()


def main() -> None:
    """启动视频速度测量桌面程序。"""
    root = tk.Tk()
    VideoSpeedAnalyzer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
