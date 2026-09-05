from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from core import TankGame
from rl.model import TankActorCritic
from rl.observation import Observation, build_observation

from .scripted import Control, Opponent, make_opponent


OBSERVATION_KEYS = ("map", "self", "tanks", "tank_mask", "bullets", "bullet_mask")
SNAPSHOT_PATTERN = re.compile(r"snapshot_stage_(\d+)_step_(\d+)\.pt$")
STAGE_CHECKPOINT_PATTERN = re.compile(r"stage_(\d+)\.pt$")


@dataclass(frozen=True)
class SnapshotInfo:
    """历史对手快照的位置和训练来源。"""

    path: Path
    completed_stage: int
    total_steps: int


class FrozenModelOpponent:
    """引用历史模型池中某个冻结策略的对手控制器。"""

    def __init__(self, pool: "HistoricalOpponentPool", snapshot: SnapshotInfo) -> None:
        # 保存所属模型池和本局固定使用的历史快照。
        self.pool = pool
        self.snapshot = snapshot
        self.name = f"历史模型-阶段{snapshot.completed_stage}-步数{snapshot.total_steps}"

    def act(self, game: TankGame, tank_id: int) -> Control:
        # 在非批量调用场景中单独计算一次历史模型动作。
        return self.pool.single_action(self.snapshot, game, tank_id)


class HistoricalOpponentPool:
    """发现、保存、抽取并批量运行冻结的历史模型。"""

    def __init__(
        self,
        output_directory: str | Path,
        device: torch.device,
        seed: int = 0,
        deterministic: bool = False,
        enabled: bool = True,
    ) -> None:
        # 初始化模型池目录、推理设备、随机状态和模型缓存。
        self.output_directory = Path(output_directory)
        self.pool_directory = self.output_directory / "opponent_pool"
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.deterministic = deterministic
        self.enabled = enabled
        self.snapshots: list[SnapshotInfo] = []
        self._model_cache: dict[Path, TankActorCritic] = {}
        self.refresh()

    def refresh(self) -> None:
        # 扫描晋级检查点和模型池快照并去除重复路径。
        discovered: dict[Path, SnapshotInfo] = {}
        if self.output_directory.exists():
            for path in self.output_directory.glob("stage_*.pt"):
                match = STAGE_CHECKPOINT_PATTERN.match(path.name)
                if match:
                    promoted_stage = int(match.group(1))
                    discovered[path.resolve()] = SnapshotInfo(
                        path.resolve(), max(promoted_stage - 1, 0), -1
                    )
        if self.pool_directory.exists():
            for path in self.pool_directory.glob("snapshot_stage_*_step_*.pt"):
                match = SNAPSHOT_PATTERN.match(path.name)
                if match:
                    discovered[path.resolve()] = SnapshotInfo(
                        path.resolve(), int(match.group(1)), int(match.group(2))
                    )
        self.snapshots = sorted(
            discovered.values(),
            key=lambda item: (item.completed_stage, item.total_steps, str(item.path)),
        )

    def eligible(self, stage_index: int) -> list[SnapshotInfo]:
        # 返回当前关卡允许抽取的所有已完成阶段和同阶段旧快照。
        return [item for item in self.snapshots if item.completed_stage <= stage_index]

    def choose(self, stage_index: int) -> SnapshotInfo | None:
        # 从可用历史模型中偏向较新阶段随机抽取一个快照。
        candidates = self.eligible(stage_index)
        if not self.enabled or not candidates:
            return None
        weights = np.asarray([2.0 ** item.completed_stage for item in candidates], dtype=np.float64)
        weights /= weights.sum()
        return candidates[int(self.rng.choice(len(candidates), p=weights))]

    def make_opponent(self, stage, rng: np.random.Generator) -> Opponent:
        # 按关卡概率选择历史模型，否则回退到该关卡的脚本对手。
        snapshot = None
        if self.enabled and rng.random() < stage.historical_opponent_probability:
            snapshot = self.choose(stage.index)
        if snapshot is not None:
            return FrozenModelOpponent(self, snapshot)
        return make_opponent(stage.opponent, rng, stage.opponent_fire_probability)

    def register_stage_checkpoint(self, path: str | Path, completed_stage: int) -> None:
        # 将晋级时保存的完整检查点登记为后续训练对手。
        resolved = Path(path).resolve()
        self.snapshots = [item for item in self.snapshots if item.path != resolved]
        self.snapshots.append(SnapshotInfo(resolved, completed_stage, -1))
        self.snapshots.sort(key=lambda item: (item.completed_stage, item.total_steps, str(item.path)))

    def save_snapshot(self, model: TankActorCritic, stage_index: int, total_steps: int) -> Path:
        # 原子保存只含模型权重的轻量历史快照并立即登记。
        self.pool_directory.mkdir(parents=True, exist_ok=True)
        destination = self.pool_directory / f"snapshot_stage_{stage_index}_step_{total_steps}.pt"
        if not destination.exists():
            temporary = destination.with_suffix(".pt.tmp")
            torch.save(
                {"model": model.state_dict(), "completed_stage": stage_index, "total_steps": total_steps},
                temporary,
            )
            temporary.replace(destination)
        self.refresh()
        return destination

    def _load_model(self, snapshot: SnapshotInfo) -> TankActorCritic:
        # 首次使用时加载冻结模型并在后续对局中复用同一实例。
        path = snapshot.path.resolve()
        if path not in self._model_cache:
            state = torch.load(path, map_location=self.device, weights_only=False)
            weights = state["model"] if isinstance(state, dict) and "model" in state else state
            model = TankActorCritic().to(self.device)
            model.load_state_dict(weights, strict=True)
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            self._model_cache[path] = model
        return self._model_cache[path]

    def _stack_observations(self, observations: list[Observation]) -> tuple[torch.Tensor, ...]:
        # 将多个历史对手的观察合并成一次批量模型输入。
        return tuple(
            torch.as_tensor(
                np.stack([observation[key] for observation in observations]),
                device=self.device,
            )
            for key in OBSERVATION_KEYS
        )

    @torch.no_grad()
    def single_action(self, snapshot: SnapshotInfo, game: TankGame, tank_id: int) -> Control:
        # 为单个环境计算历史模型动作并转换为游戏控制元组。
        model = self._load_model(snapshot)
        inputs = self._stack_observations([build_observation(game, tank_id)])
        action, _, _, _ = model.get_action_and_value(*inputs, deterministic=self.deterministic)
        return tuple(int(value) for value in action[0].cpu().tolist())

    @torch.no_grad()
    def batch_actions(self, envs: list) -> list[Control | None]:
        # 按快照分组批量推理所有使用历史模型的并行环境。
        overrides: list[Control | None] = [None] * len(envs)
        groups: dict[Path, list[tuple[int, FrozenModelOpponent]]] = {}
        for index, env in enumerate(envs):
            if isinstance(env.opponent, FrozenModelOpponent):
                groups.setdefault(env.opponent.snapshot.path.resolve(), []).append((index, env.opponent))
        for members in groups.values():
            snapshot = members[0][1].snapshot
            model = self._load_model(snapshot)
            observations = [
                build_observation(envs[index].game, envs[index].opponent_tank_id)
                for index, _ in members
            ]
            inputs = self._stack_observations(observations)
            actions, _, _, _ = model.get_action_and_value(*inputs, deterministic=self.deterministic)
            for (env_index, _), action in zip(members, actions.cpu().tolist()):
                overrides[env_index] = tuple(int(value) for value in action)
        return overrides

    @property
    def cached_model_count(self) -> int:
        # 返回当前已经加载到推理设备的冻结模型数量。
        return len(self._model_cache)
