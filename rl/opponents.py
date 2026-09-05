from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from core import TankGame
from core.entities import Tank
from core.geometry import segment_intersects_rect

from .model import TankActorCritic, load_actor_critic_state
from .observation import Observation
from .planning import astar_path, tank_cell

SCRIPT_NAMES = ("idle", "move", "random", "aim", "chase", "dodge", "hunter")
OPPONENT_CHOICES = ("self", "mix", "model") + SCRIPT_NAMES


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


def _incoming_bullet(game: TankGame, tank: Tank) -> tuple[float, float] | None:
    """最近一发朝自己飞来的敌弹速度；没有则 None。"""
    closest = None
    best = None
    for bullet in game.bullets:
        if bullet.owner_tank_id == tank.tank_id:
            continue
        dx, dy = tank.x - bullet.x, tank.y - bullet.y
        distance = math.hypot(dx, dy)
        speed = math.hypot(bullet.vx, bullet.vy)
        if distance < 1e-6 or speed < 1e-6:
            continue
        approaching = (bullet.vx * dx + bullet.vy * dy) / (speed * distance)
        if approaching <= 0.15:
            continue
        if closest is None or distance < closest:
            closest = distance
            best = (bullet.vx, bullet.vy, dx, dy, distance)
    if best is None:
        return None
    vx, vy, dx, dy, _distance = best
    px, py = -vy, vx
    # 侧移方向取远离子弹轨迹的那一侧。
    if px * dx + py * dy < 0:
        px, py = -px, -py
    return px, py


def _hunter_action(game: TankGame, tank: Tank, enemy: Tank, rng: np.random.Generator) -> tuple[int, int, int]:
    """A* 绕墙靠近，看见敌人就瞄准开火；有敌弹飞来则侧移或后退。"""
    del rng
    aim_error = _aim_error(tank, enemy)
    fire = int(_has_los(game, tank, enemy) and abs(aim_error) < 0.14)
    dodge = _incoming_bullet(game, tank)
    if dodge is not None:
        px, py = dodge
        error = _wrap_angle(math.atan2(py, px) - tank.heading)
        if abs(error) > 2.2:
            return (0, _steer_toward(_wrap_angle(error - math.pi), deadzone=0.25), fire)
        throttle = 2 if abs(error) < 0.7 else 1
        return (throttle, _steer_toward(error, deadzone=0.2), fire)
    if _has_los(game, tank, enemy):
        distance = math.hypot(enemy.x - tank.x, enemy.y - tank.y)
        steer = _steer_toward(aim_error)
        if distance < 1.6:
            throttle = 0
        elif distance > 3.2 and abs(aim_error) < 0.6:
            throttle = 2
        else:
            throttle = 1
        return (throttle, steer, fire)
    path = astar_path(game.maze, tank_cell(tank, game.maze), tank_cell(enemy, game.maze))
    if len(path) <= 1:
        return (1, _steer_toward(aim_error), fire)
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
        raise SystemExit("对手模型结构与当前集合编码器不兼容。") from error
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
