from __future__ import annotations

import math
import random

import pygame

from core import TankGame

BLUE = (45, 115, 220)
MAGENTA = (196, 40, 150)
TANK_COLORS = [BLUE, MAGENTA]


def _rgba(*channels: float) -> tuple[int, ...]:
    """把颜色通道钳到 pygame 允许的 0..255。"""
    return tuple(max(0, min(255, int(value))) for value in channels)


def rotated_rect(x: float, y: float, heading: float, half_w: float, half_h: float) -> list[tuple[float, float]]:
    fx, fy = math.cos(heading), math.sin(heading)
    rx, ry = -fy, fx
    return [
        (x + fx * a * half_w + rx * b * half_h, y + fy * a * half_w + ry * b * half_h)
        for a, b in ((-1, -1), (-1, 1), (1, 1), (1, -1))
    ]


class PygameRenderer:
    """只读取 TankGame 状态并绘图，不改变游戏世界。可被 demo 和其他脚本复用。"""

    def __init__(self, max_pixels: int = 820, caption: str = "Tank Game") -> None:
        """创建固定大小的 Pygame 窗口和帧率计时器。窗口尺寸不额外加底部条。"""
        pygame.init()
        self.max_pixels = max_pixels
        self.scale = 1.0
        self.margin = 24
        self.offset_x = float(self.margin)
        self.offset_y = float(self.margin)
        self.barrel_width = 0.091
        self.barrel_length = 0.26795
        size = (max_pixels + self.margin * 2, max_pixels + self.margin * 2)
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption(caption)
        self.clock = pygame.time.Clock()
        self.font = self._load_font(18)
        self.font_score = self._load_font(42)
        self._alive_prev: dict[int, bool] = {}
        self._explosions: list[dict] = []
        self._explosion_duration = 1.15

    def _load_font(self, size: int) -> pygame.font.Font:
        for name in ("microsoftyahei", "msyh", "simhei", "simsun", "nirmala"):
            path = pygame.font.match_font(name)
            if path:
                return pygame.font.Font(path, size)
        return pygame.font.SysFont("microsoftyahei,simhei,arial", size)

    def poll_control(self) -> str:
        """处理窗口事件。返回 skip 跳过本局，quit 结束，空字符串继续。"""
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
        return int(self.offset_x + x * self.scale), int(self.offset_y + y * self.scale)

    def reset_fx(self) -> None:
        self._alive_prev.clear()
        self._explosions.clear()

    def _spawn_explosion(self, tank, color: tuple[int, int, int], now: float) -> None:
        rng = random.Random(tank.tank_id * 997 + int(tank.x * 1000) + int(tank.y * 1000))
        debris = []
        for _ in range(10):
            angle = rng.uniform(0.0, 2 * math.pi)
            speed = rng.uniform(0.55, 1.85)
            debris.append(
                {
                    "vx": math.cos(angle) * speed,
                    "vy": math.sin(angle) * speed,
                    "spin": rng.uniform(-8.0, 8.0),
                    "heading": rng.uniform(0.0, 2 * math.pi),
                    "w": rng.uniform(0.06, 0.14),
                    "h": rng.uniform(0.04, 0.09),
                }
            )
        sparks = [{"vx": math.cos(a) * s, "vy": math.sin(a) * s} for a, s in ((rng.uniform(0.0, 2 * math.pi), rng.uniform(1.2, 3.2)) for _ in range(18))]
        self._explosions.append(
            {
                "x": tank.x,
                "y": tank.y,
                "color": color,
                "t0": now,
                "debris": debris,
                "sparks": sparks,
            }
        )

    def _sync_explosions(self, game: TankGame) -> None:
        for tank in game.tanks:
            was_alive = self._alive_prev.get(tank.tank_id, True)
            if was_alive and not tank.alive:
                self._spawn_explosion(tank, TANK_COLORS[tank.tank_id % len(TANK_COLORS)], game.elapsed)
            self._alive_prev[tank.tank_id] = tank.alive
        self._explosions = [fx for fx in self._explosions if game.elapsed - fx["t0"] < self._explosion_duration]

    def _draw_explosions(self, game: TankGame) -> None:
        for fx in self._explosions:
            t = max(0.0, game.elapsed - fx["t0"])
            u = t / self._explosion_duration
            cx, cy = fx["x"], fx["y"]
            flash = max(0.0, 1.0 - u * 3.2)
            if flash > 0.0:
                radius = max(4, int((0.12 + t * 1.8) * self.scale))
                pygame.draw.circle(self.screen, _rgba(255, 220 * flash + 40, 80 * flash), self.xy(cx, cy), radius)
            ring_u = min(1.0, u * 1.6)
            if ring_u < 1.0:
                ring_r = max(3, int((0.18 + ring_u * 0.85) * self.scale))
                alpha = int(200 * (1.0 - ring_u))
                ring = pygame.Surface((ring_r * 2 + 4, ring_r * 2 + 4), pygame.SRCALPHA)
                pygame.draw.circle(ring, _rgba(255, 170, 60, alpha), (ring_r + 2, ring_r + 2), ring_r, max(2, int(0.04 * self.scale)))
                px, py = self.xy(cx, cy)
                self.screen.blit(ring, (px - ring_r - 2, py - ring_r - 2))
            smoke_u = min(1.0, u)
            smoke_r = max(6, int((0.22 + smoke_u * 0.7) * self.scale))
            smoke = pygame.Surface((smoke_r * 2 + 4, smoke_r * 2 + 4), pygame.SRCALPHA)
            pygame.draw.circle(smoke, _rgba(40, 40, 44, 140 * (1.0 - smoke_u)), (smoke_r + 2, smoke_r + 2), smoke_r)
            px, py = self.xy(cx, cy)
            self.screen.blit(smoke, (px - smoke_r - 2, py - smoke_r - 2))
            body = fx["color"]
            for piece in fx["debris"]:
                pxw = cx + piece["vx"] * t
                pyw = cy + piece["vy"] * t
                heading = piece["heading"] + piece["spin"] * t
                fade = max(0.0, 1.0 - u)
                color = _rgba(*(c * (0.35 + 0.65 * fade) + 40 * (1.0 - fade) for c in body))
                corners = rotated_rect(pxw, pyw, heading, piece["w"], piece["h"])
                pygame.draw.polygon(self.screen, color, [self.xy(x, y) for x, y in corners])
            for spark in fx["sparks"]:
                if u > 0.45:
                    continue
                sx0, sy0 = cx + spark["vx"] * t * 0.35, cy + spark["vy"] * t * 0.35
                sx1, sy1 = cx + spark["vx"] * t, cy + spark["vy"] * t
                pygame.draw.line(self.screen, (255, 210, 90), self.xy(sx0, sy0), self.xy(sx1, sy1), max(1, int(0.02 * self.scale)))

    def _draw_kills(self, kills: list[int] | tuple[int, ...]) -> None:
        """在现有窗口底部左右叠击杀数，不增加窗口高度或宽度。"""
        left = kills[0] if len(kills) > 0 else 0
        right = kills[1] if len(kills) > 1 else 0
        left_surf = self.font_score.render(str(int(left)), True, BLUE)
        right_surf = self.font_score.render(str(int(right)), True, MAGENTA)
        pad = 18
        y = self.screen.get_height() - pad
        self.screen.blit(left_surf, (pad, y - left_surf.get_height()))
        self.screen.blit(right_surf, (self.screen.get_width() - pad - right_surf.get_width(), y - right_surf.get_height()))

    def draw(self, game: TankGame, kills: list[int] | tuple[int, ...] | None = None) -> None:
        """读取当前游戏状态，绘制墙壁、坦克、子弹、爆炸和底部击杀数。"""
        self.scale = min(self.max_pixels / game.maze.width, self.max_pixels / game.maze.height)
        self.offset_x = self.margin + (self.max_pixels - game.maze.width * self.scale) / 2
        self.offset_y = self.margin + (self.max_pixels - game.maze.height * self.scale) / 2
        self.screen.fill((225, 228, 232))
        wall_width = max(3, int(game.wall_thickness * self.scale))
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
        self._sync_explosions(game)
        for tank in game.tanks:
            if not tank.alive:
                continue
            color = TANK_COLORS[tank.tank_id % len(TANK_COLORS)]
            corners = [self.xy(x, y) for x, y in game._tank_corners(tank.x, tank.y, tank.heading)]
            pygame.draw.polygon(self.screen, color, corners)
            forward_x, forward_y = math.cos(tank.heading), math.sin(tank.heading)
            right_x, right_y = -forward_y, forward_x
            half_barrel_width = game.barrel_width / 2.0
            tip_x = tank.x + forward_x * game.barrel_length
            tip_y = tank.y + forward_y * game.barrel_length
            barrel_corners = [
                self.xy(tank.x + right_x * half_barrel_width, tank.y + right_y * half_barrel_width),
                self.xy(tip_x + right_x * half_barrel_width, tip_y + right_y * half_barrel_width),
                self.xy(tip_x - right_x * half_barrel_width, tip_y - right_y * half_barrel_width),
                self.xy(tank.x - right_x * half_barrel_width, tank.y - right_y * half_barrel_width),
            ]
            pygame.draw.polygon(self.screen, (20, 20, 20), barrel_corners)
        self._draw_explosions(game)
        for bullet in game.bullets:
            pygame.draw.circle(
                self.screen,
                TANK_COLORS[bullet.owner_tank_id % len(TANK_COLORS)],
                self.xy(bullet.x, bullet.y),
                max(3, int(game.bullet_radius * self.scale)),
            )
        if kills is not None:
            self._draw_kills(kills)
        pygame.display.flip()

    def tick(self, fps: int) -> None:
        self.clock.tick(fps)

    def close(self) -> None:
        pygame.quit()
