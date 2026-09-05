from __future__ import annotations

import math
import weakref
from pathlib import Path

import numpy as np
import torch

from core import TankGame
from core.entities import Bullet, Tank
from core.geometry import segment_intersects_rect

from .model import TankActorCritic, load_actor_critic_state
from .observation import Observation
from .planning import astar_path, tank_cell
from .trajectory import PathSegment, trace_bullet_trajectory

SCRIPT_NAMES = ("idle", "move", "random", "aim", "chase", "dodge", "hunter")
OPPONENT_CHOICES = ("self", "mix", "model") + SCRIPT_NAMES
_RICOCHET_CACHE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _wrap_angle(error: float) -> float:
    """把朝向误差折到 (-pi, pi]。"""
    return (error + math.pi) % (2 * math.pi) - math.pi


def _steer_toward(error: float, deadzone: float = 0.12) -> int:
    """steer 0 减小朝向，steer 2 增大朝向。"""
    if error > deadzone:
        return 2
    if error < -deadzone:
        return 0
    return 1


def _enemy_of(game: TankGame, tank_id: int) -> Tank | None:
    """返回另一辆仍存活的坦克；没有则返回 None。"""
    for tank in game.tanks:
        if tank.tank_id != tank_id and tank.alive:
            return tank
    return None


def _aim_error(tank: Tank, enemy: Tank) -> float:
    """炮口指向敌人中心所需的朝向误差。"""
    return _wrap_angle(math.atan2(enemy.y - tank.y, enemy.x - tank.x) - tank.heading)


def script_action(game: TankGame, tank_id: int, kind: str, rng: np.random.Generator) -> tuple[int, int, int]:
    """用规则脚本给一辆坦克生成 (throttle, steer, fire)。"""
    tank = next(item for item in game.tanks if item.tank_id == tank_id)
    if not tank.alive:
        return (1, 1, 0)
    if kind == "idle":
        return (1, 1, 0)
    if kind == "move":
        throttle = 2 if rng.random() < 0.85 else 0
        steer = int(rng.choice((0, 1, 1, 2)))
        return (throttle, steer, 0)
    if kind == "random":
        return (int(rng.integers(0, 3)), int(rng.integers(0, 3)), int(rng.integers(0, 2)))
    enemy = _enemy_of(game, tank_id)
    if enemy is None:
        return (1, 1, 0)
    if kind == "aim":
        error = _aim_error(tank, enemy)
        fire = int(abs(error) < 0.12)
        return (1, _steer_toward(error), fire)
    if kind == "chase":
        error = _aim_error(tank, enemy)
        throttle = 2 if abs(error) < 0.7 else 1
        fire = int(abs(error) < 0.15)
        return (throttle, _steer_toward(error), fire)
    if kind == "dodge":
        return _dodge_action(game, tank, enemy, rng)
    if kind == "hunter":
        return _hunter_action(game, tank, enemy, rng)
    raise ValueError(f"unknown script opponent: {kind}")


def _has_los(game: TankGame, tank: Tank, enemy: Tank) -> bool:
    """两车中心连线是否被墙挡住。"""
    for wall in game.wall_rects:
        if segment_intersects_rect(tank.x, tank.y, enemy.x, enemy.y, wall):
            return False
    return True


def _drive_toward(tank: Tank, x: float, y: float, deadzone: float = 0.18) -> tuple[int, int]:
    """朝目标点转向，对准后前进。"""
    error = _wrap_angle(math.atan2(y - tank.y, x - tank.x) - tank.heading)
    steer = _steer_toward(error, deadzone=deadzone)
    if abs(error) < 0.35:
        throttle = 2
    elif abs(error) > 1.2:
        throttle = 1
    else:
        throttle = 2
    return throttle, steer


def _hypothetical_shot(game: TankGame, tank: Tank, heading: float) -> Bullet:
    """按真实出膛位置和速度构造一颗不写入游戏状态的子弹。"""
    direction_x, direction_y = math.cos(heading), math.sin(heading)
    offset = game.tank_half_length + game.bullet_radius + 0.04
    return Bullet(
        tank.x + direction_x * offset,
        tank.y + direction_y * offset,
        direction_x * game.bullet_speed,
        direction_y * game.bullet_speed,
        owner_tank_id=tank.tank_id,
    )


def _shot_hits_enemy(game: TankGame, tank: Tank, enemy: Tank, heading: float) -> bool:
    """真实墙体反射弹道上的首个目标是否为敌人。"""
    result = trace_bullet_trajectory(
        game,
        _hypothetical_shot(game, tank, heading),
        tank.tank_id,
        max_seconds=6.0,
    )
    return result.predicted_hit == enemy.tank_id


def _ricochet_heading(game: TankGame, tank: Tank, enemy: Tank) -> float | None:
    """用镜像目标产生一反候选，再以完整弹道模拟验证（验证时允许继续多次反弹）。"""
    game_cache = _RICOCHET_CACHE.setdefault(game, {})
    cached = game_cache.get(tank.tank_id)
    if cached is not None:
        cached_maze_id, cached_time, cached_tank_xy, cached_enemy_xy, cached_heading = cached
        if (
            cached_maze_id == id(game.maze)
            and game.elapsed >= cached_time
            and game.elapsed - cached_time < 0.18
            and math.hypot(tank.x - cached_tank_xy[0], tank.y - cached_tank_xy[1]) < 0.3
            and math.hypot(enemy.x - cached_enemy_xy[0], enemy.y - cached_enemy_xy[1]) < 0.3
        ):
            return cached_heading

    candidates: list[tuple[float, float]] = []
    radius = game.bullet_radius
    for x0, y0, x1, y1 in game.wall_rects:
        for surface in (x0 - radius, x1 + radius):
            mirror_x = 2.0 * surface - enemy.x
            denominator = mirror_x - tank.x
            if abs(denominator) > 1e-8:
                fraction = (surface - tank.x) / denominator
                cross_y = tank.y + fraction * (enemy.y - tank.y)
                if 0.0 < fraction < 1.0 and y0 - radius <= cross_y <= y1 + radius:
                    candidates.append((
                        math.atan2(enemy.y - tank.y, mirror_x - tank.x),
                        math.hypot(mirror_x - tank.x, enemy.y - tank.y) / game.bullet_speed + 0.15,
                    ))
        for surface in (y0 - radius, y1 + radius):
            mirror_y = 2.0 * surface - enemy.y
            denominator = mirror_y - tank.y
            if abs(denominator) > 1e-8:
                fraction = (surface - tank.y) / denominator
                cross_x = tank.x + fraction * (enemy.x - tank.x)
                if 0.0 < fraction < 1.0 and x0 - radius <= cross_x <= x1 + radius:
                    candidates.append((
                        math.atan2(mirror_y - tank.y, enemy.x - tank.x),
                        math.hypot(enemy.x - tank.x, mirror_y - tank.y) / game.bullet_speed + 0.15,
                    ))

    # 优先选择当前炮口转角小的解；角度量化只用于去掉同一墙面产生的重复候选。
    unique: dict[int, tuple[float, float]] = {}
    for heading, duration in candidates:
        unique.setdefault(round(_wrap_angle(heading) * 10000), (heading, duration))
    result_heading = None
    ordered = sorted(unique.values(), key=lambda value: abs(_wrap_angle(value[0] - tank.heading)))
    for heading, duration in ordered[:8]:
        result = trace_bullet_trajectory(
            game,
            _hypothetical_shot(game, tank, heading),
            tank.tank_id,
            max_seconds=min(duration, 6.0),
        )
        if result.predicted_hit == enemy.tank_id:
            result_heading = heading
            break
    game_cache[tank.tank_id] = (
        id(game.maze),
        game.elapsed,
        (tank.x, tank.y),
        (enemy.x, enemy.y),
        result_heading,
    )
    return result_heading


def _segment_position(segments: tuple[PathSegment, ...], time_s: float) -> tuple[float, float] | None:
    """读取预演弹道在未来某时刻的位置。"""
    for segment in segments:
        if segment.time0 <= time_s <= segment.time1 + 1e-9:
            fraction = (time_s - segment.time0) / max(segment.time1 - segment.time0, 1e-9)
            fraction = min(max(fraction, 0.0), 1.0)
            return (
                segment.x0 + (segment.x1 - segment.x0) * fraction,
                segment.y0 + (segment.y1 - segment.y0) * fraction,
            )
    return None


def _simulate_escape(
    game: TankGame,
    tank: Tank,
    throttle: int,
    steer: int,
    trajectories: list[tuple[Bullet, tuple[PathSegment, ...]]],
    horizon: float,
) -> tuple[float, float, int]:
    """用与核心相同的加速/转向近似预测一个恒定动作的安全余量。"""
    x, y, heading, speed = tank.x, tank.y, tank.heading, tank.speed
    minimum_distance = math.inf
    wall_contacts = 0
    steps = max(1, int(math.ceil(horizon / game.dt)))
    target_speed = (throttle - 1) * game.max_speed
    angular_velocity = (steer - 1) * game.max_turn_rate
    for step in range(1, steps + 1):
        change = float(np.clip(target_speed - speed, -game.acceleration * game.dt, game.acceleration * game.dt))
        speed += change
        if throttle == 1:
            speed *= max(0.0, 1.0 - game.drag * game.dt)
        heading = (heading + angular_velocity * game.dt) % (2.0 * math.pi)
        nx = x + math.cos(heading) * speed * game.dt
        if not game._tank_hits_wall(nx, y, heading):
            x = nx
        else:
            speed *= 0.25
            wall_contacts += 1
        ny = y + math.sin(heading) * speed * game.dt
        if not game._tank_hits_wall(x, ny, heading):
            y = ny
        else:
            speed *= 0.25
            wall_contacts += 1
        time_s = step * game.dt
        for bullet, segments in trajectories:
            if bullet.age + time_s < 0.08:
                continue
            position = _segment_position(segments, time_s)
            if position is not None:
                minimum_distance = min(minimum_distance, math.hypot(x - position[0], y - position[1]))
    displacement = math.hypot(x - tank.x, y - tank.y)
    return minimum_distance, displacement, wall_contacts


def _best_dodge_action(game: TankGame, tank: Tank) -> tuple[int, int] | None:
    """综合所有当前及反弹后的子弹，选择未来1.25秒内最安全的油门/转向。"""
    horizon = 1.25
    trajectories: list[tuple[Bullet, tuple[PathSegment, ...]]] = []
    for bullet in game.bullets:
        if math.hypot(bullet.x - tank.x, bullet.y - tank.y) > game.bullet_speed * horizon + 1.5:
            continue
        result = trace_bullet_trajectory(game, bullet, bullet.owner_tank_id, max_seconds=horizon)
        trajectories.append((bullet, result.segments))
    if not trajectories:
        return None

    collision_radius = math.hypot(game.tank_half_length, game.tank_half_width) + game.bullet_radius
    neutral = _simulate_escape(game, tank, 1, 1, trajectories, horizon)
    if neutral[0] > collision_radius + 0.32:
        return None

    best_action = (1, 1)
    best_score = -math.inf
    for throttle in (0, 1, 2):
        for steer in (0, 1, 2):
            distance, displacement, contacts = _simulate_escape(
                game, tank, throttle, steer, trajectories, horizon
            )
            # 首要目标是拉大与所有未来弹道的最小距离；随后才偏好确实移动且不撞墙。
            score = distance + 0.04 * displacement - 0.08 * contacts
            if score > best_score:
                best_score = score
                best_action = (throttle, steer)
    return best_action


def _fire_for_action(
    game: TankGame,
    tank: Tank,
    enemy: Tank,
    steer: int,
    has_los: bool,
    shot_heading: float | None,
) -> int:
    """按核心“先转一物理帧、再开火”的顺序判断本动作是否应开火。"""
    if shot_heading is None or tank.cooldown > game.dt + 1e-6:
        return 0
    firing_heading = (tank.heading + (steer - 1) * game.max_turn_rate * game.dt) % (2.0 * math.pi)
    if has_los:
        # 直射保留少量容差，避免离散转向每决策步跨过瞄准点后永远不开火。
        direct_heading = math.atan2(enemy.y - tank.y, enemy.x - tank.x)
        return int(abs(_wrap_angle(direct_heading - firing_heading)) < 0.14)
    # 反弹射击不猜：预演从下一物理帧朝向发射的子弹，确认首个命中敌人才开火。
    return int(_shot_hits_enemy(game, tank, enemy, firing_heading))


def _hunter_action(game: TankGame, tank: Tank, enemy: Tank, rng: np.random.Generator) -> tuple[int, int, int]:
    """A* 寻路、验证直射/反弹弹道，并通过短时动作模拟躲避多颗子弹。"""
    del rng
    direct_heading = math.atan2(enemy.y - tank.y, enemy.x - tank.x)
    has_los = _has_los(game, tank, enemy)
    shot_heading = direct_heading if has_los else _ricochet_heading(game, tank, enemy)
    aim_error = _wrap_angle((shot_heading if shot_heading is not None else direct_heading) - tank.heading)
    dodge = _best_dodge_action(game, tank)
    if dodge is not None:
        fire = _fire_for_action(game, tank, enemy, dodge[1], has_los, shot_heading)
        return (dodge[0], dodge[1], fire)
    if shot_heading is not None:
        distance = math.hypot(enemy.x - tank.x, enemy.y - tank.y)
        steer = _steer_toward(aim_error, deadzone=0.025)
        fire = _fire_for_action(game, tank, enemy, steer, has_los, shot_heading)
        if not has_los:
            # 若离散转向跨不过精确反弹角，缓慢改变射击几何，不能原地形成“不转也不开火”的死锁。
            throttle = 1 if fire or abs(aim_error) > 0.16 else 2
            return (throttle, steer, fire)
        if distance < 1.6:
            throttle = 0
        elif distance > 3.2 and abs(aim_error) < 0.6:
            throttle = 2
        else:
            throttle = 1
        return (throttle, steer, fire)
    path = astar_path(game.maze, tank_cell(tank, game.maze), tank_cell(enemy, game.maze))
    if len(path) <= 1:
        steer = _steer_toward(aim_error, deadzone=0.025)
        fire = _fire_for_action(game, tank, enemy, steer, has_los, shot_heading)
        return (2, steer, fire)
    next_row, next_col = path[1]
    throttle, steer = _drive_toward(tank, next_col + 0.5, next_row + 0.5)
    return (throttle, steer, 0)


def _dodge_action(game: TankGame, tank: Tank, enemy: Tank, rng: np.random.Generator) -> tuple[int, int, int]:
    """侧移躲开飞来的敌弹；没有威胁时沿与视线垂直的方向移动。"""
    vx, vy = 0.0, 0.0
    closest = None
    for bullet in game.bullets:
        if bullet.owner_tank_id == tank.tank_id:
            continue
        dx, dy = tank.x - bullet.x, tank.y - bullet.y
        distance = math.hypot(dx, dy)
        speed = math.hypot(bullet.vx, bullet.vy)
        if distance < 1e-6 or speed < 1e-6:
            continue
        approaching = (bullet.vx * dx + bullet.vy * dy) / (speed * distance)
        if approaching <= 0.1:
            continue
        if closest is None or distance < closest:
            closest = distance
            px, py = -bullet.vy, bullet.vx
            if rng.random() < 0.5:
                px, py = -px, -py
            vx, vy = px, py
    if closest is None:
        dx, dy = tank.x - enemy.x, tank.y - enemy.y
        vx, vy = -dy, dx
        if rng.random() < 0.5:
            vx, vy = -vx, -vy
    error = _wrap_angle(math.atan2(vy, vx) - tank.heading)
    throttle = 2 if abs(error) < 0.8 else 1
    aim_error = _aim_error(tank, enemy)
    fire = int(abs(aim_error) < 0.12)
    return (throttle, _steer_toward(error, deadzone=0.2), fire)


class OpponentSlot:
    """池里的一个对手：规则脚本，或一份冻结模型。"""

    def __init__(self, kind: str, model: TankActorCritic | None = None, label: str = "") -> None:
        self.kind = kind
        self.model = model
        self.label = label or kind


class OpponentController:
    """对手池：每局从池里随机抽一个脚本或冻结模型。self 对战不使用这个类。"""

    def __init__(
        self,
        kind: str | None = None,
        model: TankActorCritic | None = None,
        device: torch.device | None = None,
        seed: int = 0,
        pool: list[OpponentSlot] | None = None,
        weights: list[float] | None = None,
        deterministic_models: bool = True,
    ) -> None:
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.deterministic_models = deterministic_models
        self.env_slots: list[int] = []
        if pool is not None:
            if not pool:
                raise ValueError("opponent pool is empty")
            self.pool = pool
        else:
            if kind is None or kind not in SCRIPT_NAMES + ("mix", "model"):
                raise ValueError(f"opponent must be one of {OPPONENT_CHOICES}")
            if kind == "model" and model is None:
                raise ValueError("model opponent needs a loaded policy")
            if kind == "mix":
                self.pool = [OpponentSlot(name, label=name) for name in SCRIPT_NAMES]
            elif kind == "model":
                self.pool = [OpponentSlot("model", model, label="model")]
            else:
                self.pool = [OpponentSlot(kind, label=kind)]
        self.kind = "pool" if len(self.pool) > 1 else self.pool[0].kind
        self.model = self.pool[0].model if len(self.pool) == 1 else None
        if weights is None:
            self.weights = np.full(len(self.pool), 1.0 / len(self.pool), dtype=np.float64)
        else:
            weight_array = np.asarray(weights, dtype=np.float64)
            if weight_array.shape != (len(self.pool),):
                raise ValueError(f"opponent weights need {len(self.pool)} values, got {len(weight_array)}")
            if np.any(weight_array < 0.0) or float(weight_array.sum()) <= 0.0:
                raise ValueError("opponent weights must be non-negative and contain a positive value")
            self.weights = weight_array / weight_array.sum()

    def describe(self) -> str:
        """打印对手池内容。"""
        return ", ".join(f"{slot.label}({weight:.0%})" for slot, weight in zip(self.pool, self.weights))

    def reset_env(self, env_index: int) -> str:
        """每局给该环境从池里抽一个对手。"""
        index = int(self.rng.choice(len(self.pool), p=self.weights))
        while len(self.env_slots) <= env_index:
            self.env_slots.append(index)
        self.env_slots[env_index] = index
        return self.pool[index].label

    def current_label(self, env_index: int) -> str:
        """返回某个环境本局正在使用的对手标签。"""
        slot_index = self.env_slots[env_index] if env_index < len(self.env_slots) else 0
        return self.pool[slot_index].label

    def action(self, env_index: int, game: TankGame, tank_id: int, observation: Observation) -> np.ndarray:
        """返回当前局里抽中的那个对手的离散动作。"""
        slot_index = self.env_slots[env_index] if env_index < len(self.env_slots) else 0
        slot = self.pool[slot_index]
        if slot.kind == "model":
            return self._model_action(slot.model, observation)
        return np.asarray(script_action(game, tank_id, slot.kind, self.rng), dtype=np.int64)

    def _model_action(self, model: TankActorCritic | None, observation: Observation) -> np.ndarray:
        """用冻结策略的 argmax 动作。"""
        assert model is not None and self.device is not None
        from .train import _model_batch, stack_observations

        with torch.no_grad():
            actions, _, _, _ = model.get_action_and_value(
                *_model_batch(stack_observations([observation]), self.device),
                deterministic=self.deterministic_models,
            )
        return actions.cpu().numpy()[0]


def expand_opponent_tokens(tokens: list[str] | None) -> list[str]:
    """把空格和逗号分隔的对手描述拆成条目。"""
    if not tokens:
        return ["self"]
    items: list[str] = []
    for token in tokens:
        items.extend(part.strip() for part in str(token).replace(";", ",").split(",") if part.strip())
    return items or ["self"]


def build_opponent_controller(
    tokens: list[str] | None,
    model_paths: list[Path] | None,
    device: torch.device,
    seed: int,
    weights: list[float] | None = None,
    deterministic_models: bool = True,
) -> OpponentController | None:
    """解析对手列表。只有 self 时返回 None（镜像自博弈）；否则组成随机对手池。"""
    names = expand_opponent_tokens(tokens)
    extra_models = list(model_paths or [])
    if "self" in names:
        if len(names) > 1:
            raise SystemExit("self 不能和其它对手放进同一个池，镜像自博弈请单独写 --opponent self")
        return None
    pool: list[OpponentSlot] = []
    for name in names:
        if name == "mix":
            pool.extend(OpponentSlot(script, label=script) for script in SCRIPT_NAMES)
            continue
        if name in SCRIPT_NAMES:
            pool.append(OpponentSlot(name, label=name))
            continue
        if name == "model":
            if not extra_models:
                raise SystemExit("--opponent model 需要同时提供 --opponent-model")
            path = extra_models.pop(0)
            pool.append(OpponentSlot("model", load_opponent_model(path, device), label=str(path)))
            continue
        path = Path(name)
        pool.append(OpponentSlot("model", load_opponent_model(path, device), label=str(path)))
    for path in extra_models:
        pool.append(OpponentSlot("model", load_opponent_model(path, device), label=str(path)))
    if not pool:
        raise SystemExit("对手池为空")
    try:
        return OpponentController(
            device=device,
            seed=seed,
            pool=pool,
            weights=weights,
            deterministic_models=deterministic_models,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error


def load_opponent_model(path: Path, device: torch.device) -> TankActorCritic:
    """加载一份只用于当对手、不再更新的模型。"""
    if not path.is_file():
        raise SystemExit(f"未找到对手模型：{path.resolve()}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = TankActorCritic().to(device)
    try:
        load_actor_critic_state(model, checkpoint["model_state"])
    except RuntimeError as error:
        raise SystemExit("对手模型使用旧墙观察/模型结构，不能放入当前对手池。") from error
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
