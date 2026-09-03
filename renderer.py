from __future__ import annotations

import math

import pygame

from core import TankGame


class PygameRenderer:
    """只读取 TankGame 状态并绘图，不改变游戏世界。"""

    def __init__(self, max_pixels: int = 820) -> None:
        """创建固定大小的 Pygame 窗口和帧率计时器。"""
        pygame.init()
        self.max_pixels = max_pixels  # 地图可使用的最大显示像素。
        self.scale = 1.0  # 一个世界格子当前对应的像素数。
        self.margin = 24  # 地图和窗口边缘之间的留白像素。
        self.offset_x = float(self.margin)  # 地图左上角在窗口中的 x 偏移。
        self.offset_y = float(self.margin)  # 地图左上角在窗口中的 y 偏移。
        self.barrel_width = 0.091  # 实测炮管显示宽度；炮管不参与碰撞。
        self.barrel_length = 0.26795  # 炮管从车身中心到炮口的固定显示长度。
        size = (max_pixels + self.margin * 2, max_pixels + self.margin * 2)
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption("Tank Game")
        self.clock = pygame.time.Clock()

    def xy(self, x: float, y: float) -> tuple[int, int]:
        """把核心中的世界坐标转换为窗口像素坐标。"""
        return int(self.offset_x + x * self.scale), int(self.offset_y + y * self.scale)

    def draw(self, game: TankGame) -> None:
        """读取当前游戏状态，并绘制墙壁、坦克和子弹。"""
        self.scale = min(self.max_pixels / game.maze.width, self.max_pixels / game.maze.height)
        self.offset_x = self.margin + (self.max_pixels - game.maze.width * self.scale) / 2
        self.offset_y = self.margin + (self.max_pixels - game.maze.height * self.scale) / 2
        self.screen.fill((225, 228, 232))
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

    def tick(self, fps: int) -> None:
        """限制显示循环的现实运行帧率；它不会改变核心的 dt。"""
        self.clock.tick(fps)

    def close(self) -> None:
        """关闭 Pygame 窗口并释放显示资源。"""
        pygame.quit()
