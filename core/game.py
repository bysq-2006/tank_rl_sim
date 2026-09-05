from __future__ import annotations

import math
from typing import Any

import numpy as np

from .entities import Bullet, Tank
from .maze import Maze, generate_maze


Control = tuple[int, int, int]  # (油门, 转向, 开火) 三个离散控制值。


class TankGame:
    """纯游戏核心：只维护世界状态和物理规则，不负责渲染或模型输入。"""

    def __init__(self, physics_hz: int = 24, time_limit: float = 90.0, rows: int | None = None, cols: int | None = None) -> None:
        """设置游戏参数，并立即创建第一局随机游戏。"""
        self.physics_hz = physics_hz  # 每秒固定执行的物理帧数，默认 24。
        self.dt = 1.0 / physics_hz  # 每次 update 推进的固定秒数，默认 1/24 秒。
        self.time_limit = time_limit  # 单局最长时间，超时后判为无胜者。
        self.fixed_rows = rows  # 固定地图行数；None 表示每局随机。
        self.fixed_cols = cols  # 固定地图列数；None 表示每局随机。
        self.tank_half_length = 0.22795  # 实测车身前后长度 0.4559 格的一半。
        self.tank_half_width = 0.16945  # 实测车身左右宽度 0.3389 格的一半。
        self.barrel_length = 0.26795  # 炮管从车身中心到炮口，参与撞墙，避免插入墙内。
        self.barrel_width = 0.091  # 炮管显示宽度，同样用于撞墙。
        self.wall_thickness = 0.0735  # 实测地图内部墙和外边框的统一宽度。
        self.max_speed = 1.8622  # 根据视频测量得到的坦克最大前进/后退速度，单位为格子/秒。
        self.acceleration = 7.0  # 坦克每秒最多增加或减少的速度。
        self.drag = 6.0  # 松开油门时的减速强度。
        self.max_turn_rate = math.radians(150)  # 坦克最大旋转速度，单位为弧度/秒。
        self.bullet_speed = 2.2738  # 根据视频测量得到的子弹飞行速度，单位为格子/秒。
        self.bullet_radius = 0.045  # 实测子弹直径 0.09 格的一半。
        self.bullet_lifetime = 10.0  # 子弹最多存在的秒数。
        self.max_bounces = 12  # 子弹被移除前允许的最大反弹次数。
        self.max_bullets_per_tank = 5  # 每辆坦克允许同时存在的子弹上限。
        self.fire_cooldown = 0.234  # 实测最大射速间隔：2.967 秒 - 2.733 秒。
        self.rng = np.random.default_rng()  # 地图、出生点和朝向共用的随机数生成器。
        self.maze: Maze  # 当前局使用的迷宫。
        self.tanks: list[Tank] = []  # 当前局中的全部坦克。
        self.bullets: list[Bullet] = []  # 当前仍在飞行的全部子弹。
        self.wall_rects: list[tuple[float, float, float, float]] = []  # 用于碰撞检测的墙矩形。
        self.elapsed = 0.0  # 当前局已经推进的游戏时间。
        self.death_grace = 3.4  # 首辆坦克死亡后，再等待这么多秒才结束对局。
        self.first_death_at: float | None = None  # 首次出现死亡时的 elapsed；None 表示尚未有坦克死亡。
        self.is_over = False  # 当前局是否已经结束。
        self.winner: int | None = None  # 获胜坦克的 tank_id；None 表示未结束或无胜者。
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        """清空旧状态并开始一局新游戏；指定 seed 时结果可复现。"""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.maze = generate_maze(self.rng, self.fixed_rows, self.fixed_cols)
        self.wall_rects = self.maze.wall_rects(self.wall_thickness)
        cells = [(r, c) for r in range(self.maze.rows) for c in range(self.maze.cols)]  # 所有可选出生格。
        first = cells[int(self.rng.integers(len(cells)))]  # 玩家使用随机出生格。
        distances = [abs(r - first[0]) + abs(c - first[1]) for r, c in cells]  # 各格到玩家的网格距离。
        far = cells[int(np.argmax(distances))]  # 敌人出生在网格距离最远的格子。
        self.tanks = [
            Tank(first[1] + 0.5, first[0] + 0.5, float(self.rng.uniform(-math.pi, math.pi)), tank_id=0),
            Tank(far[1] + 0.5, far[0] + 0.5, float(self.rng.uniform(-math.pi, math.pi)), tank_id=1),
        ]
        self.bullets = []
        self.elapsed = 0.0
        self.first_death_at = None
        self.is_over = False
        self.winner = None

    def update(self, controls: list[Any] | tuple[Any, ...]) -> list[dict[str, Any]]:
        """根据外部提供的全部坦克控制，让游戏世界前进一个固定物理帧。"""
        if self.is_over:
            return []
        if len(controls) != len(self.tanks):
            raise ValueError(f"expected {len(self.tanks)} tank controls, got {len(controls)}")
        parsed_controls = [self._parse_control(control) for control in controls]  # 本帧全部坦克的外部控制。
        events: list[dict[str, Any]] = []  # 本帧命中记录，包含发射方、受击方和子弹飞行时间。
        self._physics_tick(parsed_controls, events)
        self.elapsed += self.dt
        self._update_game_result()
        return events

    @staticmethod
    def _parse_control(control: Any) -> Control:
        """检查字典或元组形式的控制输入，并统一转换为三元组。"""
        values = (control["throttle"], control["steer"], control["fire"]) if isinstance(control, dict) else tuple(control)
        if len(values) != 3 or not all(isinstance(v, (int, np.integer)) for v in values):
            raise ValueError("control must contain three integers")
        throttle, steer, fire = map(int, values)
        if throttle not in (0, 1, 2) or steer not in (0, 1, 2) or fire not in (0, 1):
            raise ValueError("expected throttle=0..2, steer=0..2, fire=0..1")
        return throttle, steer, fire

    def _update_game_result(self) -> None:
        """根据坦克存活状态和时间限制更新结束状态与胜者。"""
        if any(not tank.alive for tank in self.tanks):
            if self.first_death_at is None:
                self.first_death_at = self.elapsed
            # 宽限期内世界继续运转：剩余子弹仍可能打死另一方。
            if self.elapsed - self.first_death_at < self.death_grace:
                return
            self.is_over = True
            survivors = [tank.tank_id for tank in self.tanks if tank.alive]
            self.winner = survivors[0] if len(survivors) == 1 else None
        elif self.elapsed >= self.time_limit:
            self.is_over = True
            self.winner = None

    def _physics_tick(self, controls: list[Control], events: list[dict[str, Any]]) -> None:
        """执行一帧完整物理：先更新坦克，再更新子弹。"""
        for tank, (throttle, steer, fire) in zip(self.tanks, controls):
            if not tank.alive:
                continue
            target_speed = (throttle - 1) * self.max_speed  # 当前油门希望达到的速度。
            change = np.clip(target_speed - tank.speed, -self.acceleration * self.dt, self.acceleration * self.dt)  # 本帧允许的速度变化。
            tank.speed += float(change)
            if throttle == 1:
                tank.speed *= max(0.0, 1.0 - self.drag * self.dt)
            tank.angular_velocity = (steer - 1) * self.max_turn_rate
            next_heading = (tank.heading + tank.angular_velocity * self.dt) % (2 * math.pi)
            # 旋转始终生效；矩形角压进墙壁时，再把坦克沿最短方向推出墙面。
            old_x, old_y, old_heading = tank.x, tank.y, tank.heading
            tank.heading = next_heading
            if not self._push_tank_out_of_walls(tank):
                # 极少数被多面墙夹死的情况回退，防止坦克被推出地图。
                tank.x, tank.y, tank.heading = old_x, old_y, old_heading
                tank.angular_velocity = 0.0
            self._move_tank(tank, math.cos(tank.heading) * tank.speed * self.dt, math.sin(tank.heading) * tank.speed * self.dt)
            # 保留不足一帧的时间余量，使 0.234 秒间隔在 24 Hz 下长期平均仍然准确。
            tank.cooldown -= self.dt
            if fire and tank.cooldown <= 0.0:
                self._fire(tank)
            elif tank.cooldown < 0.0:
                tank.cooldown = 0.0

        for bullet in list(self.bullets):
            self._move_bullet(bullet)
            bullet.age += self.dt
            hit = self._bullet_hit(bullet)
            if hit is not None:
                hit.alive = False
                events.append(
                    {
                        "shooter": bullet.owner_tank_id,
                        "victim": hit.tank_id,
                        "bullet_age": float(bullet.age),
                    }
                )
                self.bullets.remove(bullet)
            elif bullet.age >= self.bullet_lifetime or bullet.bounces > self.max_bounces:
                self.bullets.remove(bullet)

    def _move_tank(self, tank: Tank, dx: float, dy: float) -> None:
        """分别尝试沿 x 和 y 移动坦克，并阻止它穿过墙壁。"""
        next_x = tank.x + dx
        if not self._tank_hits_wall(next_x, tank.y, tank.heading):
            tank.x = next_x
        else:
            tank.speed *= 0.25
        next_y = tank.y + dy
        if not self._tank_hits_wall(tank.x, next_y, tank.heading):
            tank.y = next_y
        else:
            tank.speed *= 0.25

    def _tank_corners(self, x: float, y: float, heading: float, front: float | None = None) -> list[tuple[float, float]]:
        """返回旋转矩形的四个顶点。撞墙时 front 用炮管长度，绘制车身时用车身半长。"""
        forward = (math.cos(heading), math.sin(heading))
        right = (-forward[1], forward[0])
        back, nose = self.tank_half_length, self.tank_half_length if front is None else front
        half_width = self.tank_half_width
        return [
            (x + forward[0] * a + right[0] * b * half_width, y + forward[1] * a + right[1] * b * half_width)
            for a, b in ((-back, -1), (-back, 1), (nose, 1), (nose, -1))
        ]

    def _tank_hits_wall(self, x: float, y: float, heading: float) -> bool:
        """使用分离轴方法判断带炮管的坦克是否与墙壁相交。"""
        return any(self._tank_wall_separation(x, y, heading, wall) is not None for wall in self.wall_rects)

    def _tank_wall_separation(
        self,
        x: float,
        y: float,
        heading: float,
        wall: tuple[float, float, float, float],
    ) -> tuple[float, float] | None:
        """返回将带炮管的旋转矩形推出一面墙所需的最小位移。"""
        tank_corners = self._tank_corners(x, y, heading, front=self.barrel_length)
        axes = [(1.0, 0.0), (0.0, 1.0), (math.cos(heading), math.sin(heading)), (-math.sin(heading), math.cos(heading))]
        x0, y0, x1, y1 = wall
        wall_corners = [(x0, y0), (x0, y1), (x1, y1), (x1, y0)]
        smallest_overlap = math.inf
        smallest_axis = (0.0, 0.0)
        for ax, ay in axes:
            tank_projection = [px * ax + py * ay for px, py in tank_corners]
            wall_projection = [px * ax + py * ay for px, py in wall_corners]
            overlap = min(max(tank_projection), max(wall_projection)) - max(min(tank_projection), min(wall_projection))
            if overlap <= 0.0:
                return None
            if overlap < smallest_overlap:
                smallest_overlap = overlap
                smallest_axis = (ax, ay)

        ax, ay = smallest_axis
        wall_center_x, wall_center_y = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        if (x - wall_center_x) * ax + (y - wall_center_y) * ay < 0.0:
            ax, ay = -ax, -ay
        distance = smallest_overlap + 1e-6
        return ax * distance, ay * distance

    def _push_tank_out_of_walls(self, tank: Tank) -> bool:
        """反复应用最小分离位移，让旋转中的坦克从相邻墙壁间滑出。"""
        for _ in range(12):
            correction = None
            for wall in self.wall_rects:
                correction = self._tank_wall_separation(tank.x, tank.y, tank.heading, wall)
                if correction is not None:
                    break
            if correction is None:
                return True
            tank.x += correction[0]
            tank.y += correction[1]
        return not self._tank_hits_wall(tank.x, tank.y, tank.heading)

    def _circle_hits_wall(self, x: float, y: float, radius: float) -> bool:
        """判断给定圆形是否与任意墙壁矩形相交。"""
        for x0, y0, x1, y1 in self.wall_rects:
            qx, qy = min(max(x, x0), x1), min(max(y, y0), y1)
            if (x - qx) ** 2 + (y - qy) ** 2 < radius**2:
                return True
        return False

    def _fire(self, tank: Tank) -> None:
        """未达到该坦克的子弹上限时，在炮口前生成一颗子弹。"""
        active_count = sum(bullet.owner_tank_id == tank.tank_id for bullet in self.bullets)
        if active_count >= self.max_bullets_per_tank:
            return
        offset = self.tank_half_length + self.bullet_radius + 0.04
        ux, uy = math.cos(tank.heading), math.sin(tank.heading)
        x, y = tank.x + ux * offset, tank.y + uy * offset
        vx, vy, bounces = ux * self.bullet_speed, uy * self.bullet_speed, 0
        if self._circle_hits_wall(x, y, self.bullet_radius):
            x = tank.x + ux * (self.tank_half_length - self.bullet_radius)
            y = tank.y + uy * (self.tank_half_length - self.bullet_radius)
            vx, vy, bounces = -vx, -vy, 1
        self.bullets.append(Bullet(x, y, vx, vy, owner_tank_id=tank.tank_id, bounces=bounces))
        # 使用累加而不是覆盖，以保留固定帧更新产生的少量超时误差。
        tank.cooldown += self.fire_cooldown

    def _move_bullet(self, bullet: Bullet) -> None:
        """用多个小步移动子弹，并在碰到水平或垂直墙时反射。"""
        distance = math.hypot(bullet.vx, bullet.vy) * self.dt  # 子弹本帧应移动的总距离。
        steps = max(1, int(math.ceil(distance / (self.wall_thickness * 0.45))))  # 为防穿墙而拆分的小步数量。
        step_dt = self.dt / steps  # 每个碰撞检测小步代表的时间。
        for _ in range(steps):
            nx = bullet.x + bullet.vx * step_dt
            if self._circle_hits_wall(nx, bullet.y, self.bullet_radius):
                bullet.vx *= -1
                bullet.bounces += 1
            else:
                bullet.x = nx
            ny = bullet.y + bullet.vy * step_dt
            if self._circle_hits_wall(bullet.x, ny, self.bullet_radius):
                bullet.vy *= -1
                bullet.bounces += 1
            else:
                bullet.y = ny

    def _bullet_hit(self, bullet: Bullet) -> Tank | None:
        """返回被子弹命中的坦克；反弹弹可以命中其发射者。"""
        for tank in self.tanks:
            if not tank.alive:
                continue
            cos_h, sin_h = math.cos(tank.heading), math.sin(tank.heading)
            dx, dy = bullet.x - tank.x, bullet.y - tank.y
            local_x = dx * cos_h + dy * sin_h
            local_y = -dx * sin_h + dy * cos_h
            closest_x = min(max(local_x, -self.tank_half_length), self.tank_half_length)
            closest_y = min(max(local_y, -self.tank_half_width), self.tank_half_width)
            if (local_x - closest_x) ** 2 + (local_y - closest_y) ** 2 <= self.bullet_radius**2:
                return tank
        return None
