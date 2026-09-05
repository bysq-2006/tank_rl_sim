from __future__ import annotations

import math

import pygame

from core import TankGame


class PygameRenderer:
    """只读取 TankGame 状态并绘图，不改变游戏世界。"""

    def __init__(self, max_pixels: int = 820, hud_rows: int = 1) -> None:
        """创建固定大小的 Pygame 窗口和帧率计时器。"""
        pygame.init()
        self.max_pixels = max_pixels  # 地图可使用的最大显示像素。
        self.scale = 1.0  # 一个世界格子当前对应的像素数。
        self.margin = 24  # 地图和窗口边缘之间的留白像素。
        self.hud_row_h = 22
        self.hud_padding = 8
        self.hud_rows = max(2, hud_rows)
        self.hud_height = self.hud_padding * 2 + self.hud_row_h * self.hud_rows
        self.offset_x = float(self.margin)  # 地图左上角在窗口中的 x 偏移。
        self.offset_y = float(self.hud_height + self.margin)  # 地图左上角在窗口中的 y 偏移。
        self.barrel_width = 0.091  # 实测炮管显示宽度；炮管不参与碰撞。
        self.barrel_length = 0.26795  # 炮管从车身中心到炮口的固定显示长度。
        size = (max_pixels + self.margin * 2, self.hud_height + max_pixels + self.margin * 2)
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption("Tank Game  |  N/空格 跳过本局  Esc 退出")
        self.clock = pygame.time.Clock()
        self.font = self._load_font(18)
        self.font_small = self._load_font(16)

    def _load_font(self, size: int) -> pygame.font.Font:
        """优先用系统中文字体，计分板才能显示中文。"""
        for name in ("microsoftyahei", "msyh", "simhei", "simsun", "nirmala"):
            path = pygame.font.match_font(name)
            if path:
                return pygame.font.Font(path, size)
        return pygame.font.SysFont("microsoftyahei,simhei,arial", size)

    def poll_control(self) -> str:
        """处理窗口事件。返回 skip 跳过本局，quit 结束观战，空字符串继续。"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_n, pygame.K_SPACE, pygame.K_RETURN):
                    return "skip"
                if event.key == pygame.K_ESCAPE:
                    return "quit"
        return ""

    def xy(self, x: float, y: float) -> tuple[int, int]:
        """把核心中的世界坐标转换为窗口像素坐标。"""
        return int(self.offset_x + x * self.scale), int(self.offset_y + y * self.scale)

    def draw(self, game: TankGame, scoreboard: list[str] | None = None) -> None:
        """读取当前游戏状态，并绘制墙壁、坦克和子弹。"""
        self.scale = min(self.max_pixels / game.maze.width, self.max_pixels / game.maze.height)
        self.offset_x = self.margin + (self.max_pixels - game.maze.width * self.scale) / 2
        self.offset_y = self.hud_height + self.margin + (self.max_pixels - game.maze.height * self.scale) / 2
        self.screen.fill((225, 228, 232))
        self._draw_scoreboard(scoreboard)
        wall_width = max(3, int(game.wall_thickness * self.scale))
        # 整段总共增加半个墙宽，所以两端分别增加四分之一个墙宽。
        wall_extension = game.wall_thickness / 4.0
        for r, c in zip(*game.maze.horizontal.nonzero()):
            pygame.draw.line(
                self.screen,
                (35, 38, 42),
                self.xy(c - wall_extension, r),
                self.xy(c + 1 + wall_extension, r),
                wall_width,
            )
        for r, c in zip(*game.maze.vertical.nonzero()):
            pygame.draw.line(
                self.screen,
                (35, 38, 42),
                self.xy(c, r - wall_extension),
                self.xy(c, r + 1 + wall_extension),
                wall_width,
            )
        colors = [(45, 115, 220), (220, 65, 55)]
        for tank in game.tanks:
            if not tank.alive:
                continue
            center = self.xy(tank.x, tank.y)
            corners = [self.xy(x, y) for x, y in game._tank_corners(tank.x, tank.y, tank.heading)]
            pygame.draw.polygon(self.screen, colors[tank.tank_id % len(colors)], corners)
            # 用旋转矩形绘制固定尺寸炮管，避免粗线端点在斜角时出现视觉歪斜。
            forward_x, forward_y = math.cos(tank.heading), math.sin(tank.heading)
            right_x, right_y = -forward_y, forward_x
            half_barrel_width = self.barrel_width / 2.0
            tip_x = tank.x + forward_x * self.barrel_length
            tip_y = tank.y + forward_y * self.barrel_length
            barrel_corners = [
                self.xy(tank.x + right_x * half_barrel_width, tank.y + right_y * half_barrel_width),
                self.xy(tip_x + right_x * half_barrel_width, tip_y + right_y * half_barrel_width),
                self.xy(tip_x - right_x * half_barrel_width, tip_y - right_y * half_barrel_width),
                self.xy(tank.x - right_x * half_barrel_width, tank.y - right_y * half_barrel_width),
            ]
            pygame.draw.polygon(self.screen, (20, 20, 20), barrel_corners)
        for bullet in game.bullets:
            pygame.draw.circle(self.screen, colors[bullet.owner_tank_id % len(colors)], self.xy(bullet.x, bullet.y), max(3, int(game.bullet_radius * self.scale)))
        pygame.display.flip()

    def _draw_scoreboard(self, lines: list[str] | None) -> None:
        """窗口顶部画计分板。"""
        hud = pygame.Rect(0, 0, self.screen.get_width(), self.hud_height)
        pygame.draw.rect(self.screen, (36, 40, 48), hud)
        pygame.draw.line(self.screen, (70, 76, 88), (0, self.hud_height - 1), (self.screen.get_width(), self.hud_height - 1))
        if not lines:
            text = self.font.render("计分板  |  N/空格 跳过本局  Esc 退出", True, (230, 232, 236))
            self.screen.blit(text, (12, self.hud_padding))
            return
        y = self.hud_padding
        for line in lines[: self.hud_rows]:
            self.screen.blit(self.font.render(line, True, (255, 214, 90)), (12, y))
            y += self.hud_row_h

    def tick(self, fps: int) -> None:
        """限制显示循环的现实运行帧率；它不会改变核心的 dt。"""
        self.clock.tick(fps)

    def close(self) -> None:
        """关闭 Pygame 窗口并释放显示资源。"""
        pygame.quit()
