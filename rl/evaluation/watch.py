from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from core import TankGame
from renderer import PygameRenderer
from rl.curriculum import STAGES, STAGE_TITLES
from rl.envs.scenarios import apply_layout, apply_spawn
from rl.model import TankActorCritic
from rl.observation import build_observation
from rl.opponents import make_opponent
from rl.training.rollout import stack_observations


SCRIPTED_OPPONENTS = ("idle", "random_mover", "weak_shooter", "chaser")


def parse_args() -> argparse.Namespace:
    # 解析主模型、对手、关卡、局数和播放速度等观战参数。
    parser = argparse.ArgumentParser(description="使用现有渲染器观看模型对战")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/tank_rl_curriculum_v2/latest.pt"))
    parser.add_argument("--opponent", default="chaser", help="脚本策略名称或另一个.pt检查点")
    parser.add_argument("--stage", type=int, choices=range(len(STAGES)))
    parser.add_argument("--games", type=int, default=0, help="0表示持续播放")
    parser.add_argument("--seed", type=int, default=2_000_000)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--sample-actions", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _configure_console() -> None:
    # 将终端输出切换为UTF-8以正确显示中文观战信息。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def _load_model(path: Path, device: torch.device) -> tuple[TankActorCritic, dict]:
    # 从训练检查点或纯参数文件加载一个冻结的对战模型。
    if not path.exists():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location=device, weights_only=False)
    weights = state["model"] if isinstance(state, dict) and "model" in state else state
    model = TankActorCritic().to(device)
    model.load_state_dict(weights, strict=True)
    model.eval()
    return model, state if isinstance(state, dict) else {}


class ModelController:
    """把模型包装成与脚本对手一致的动作接口。"""

    def __init__(self, model: TankActorCritic, device: torch.device, deterministic: bool) -> None:
        # 保存冻结模型、运算设备和动作选择方式。
        self.model = model
        self.device = device
        self.deterministic = deterministic

    @torch.no_grad()
    def act(self, game: TankGame, tank_id: int) -> tuple[int, int, int]:
        # 为指定坦克构造观察并选择一次联合动作。
        observation = build_observation(game, tank_id)
        inputs = stack_observations([observation], self.device)
        action, _, _, _ = self.model.get_action_and_value(
            *inputs, deterministic=self.deterministic
        )
        return tuple(int(value) for value in action[0].cpu().tolist())


def _opponent_controller(name_or_path: str, device: torch.device, rng: np.random.Generator, deterministic: bool):
    # 根据参数创建脚本控制器或另一个检查点模型控制器。
    if name_or_path in SCRIPTED_OPPONENTS:
        probability = 0.10 if name_or_path == "weak_shooter" else 0.80
        return make_opponent(name_or_path, rng, probability), name_or_path
    path = Path(name_or_path)
    model, _ = _load_model(path, device)
    return ModelController(model, device, deterministic), path.name


def _new_game(stage, seed: int, rng: np.random.Generator) -> TankGame:
    # 按指定课程阶段创建一局可复现的观战游戏。
    rows = int(rng.integers(stage.rows[0], stage.rows[1] + 1))
    cols = int(rng.integers(stage.cols[0], stage.cols[1] + 1))
    game = TankGame(rows=rows, cols=cols)
    game.reset(seed)
    apply_layout(game, stage.layout, rng)
    apply_spawn(game, stage.spawn, rng)
    return game


def watch(args: argparse.Namespace) -> None:
    # 运行带窗口渲染、跳局和计分功能的连续模型对战。
    if args.speed <= 0.0 or args.action_repeat < 1:
        raise ValueError("speed and action-repeat must be positive")
    device = torch.device(args.device)
    player_model, checkpoint = _load_model(args.checkpoint, device)
    checkpoint_stage = checkpoint.get("curriculum", {}).get("current_stage", 0)
    stage_index = checkpoint_stage if args.stage is None else args.stage
    stage = STAGES[int(stage_index)]
    rng = np.random.default_rng(args.seed)
    player = ModelController(player_model, device, not args.sample_actions)
    opponent, opponent_name = _opponent_controller(
        args.opponent, device, rng, not args.sample_actions
    )
    renderer = PygameRenderer(
        caption=f"蓝方 {args.checkpoint.name}  VS  紫方 {opponent_name}  |  N跳局 Esc退出"
    )
    score = [0, 0]
    completed = 0
    quit_requested = False

    try:
        while not quit_requested and (args.games == 0 or completed < args.games):
            game_seed = args.seed + completed
            game = _new_game(stage, game_seed, rng)
            renderer.reset_fx()
            frame_index = 0
            controls = [(1, 1, 0), (1, 1, 0)]
            skip = False
            print(
                f"第{completed + 1}局｜随机种子：{game_seed}｜"
                f"关卡：{stage.index}-{STAGE_TITLES[stage.index]}｜地图：{game.maze.rows}×{game.maze.cols}"
            )
            while not game.is_over and not skip and not quit_requested:
                command = renderer.poll_control()
                skip = command == "skip"
                quit_requested = command == "quit"
                if skip or quit_requested:
                    break
                if frame_index % args.action_repeat == 0:
                    controls[0] = player.act(game, 0)
                    controls[1] = opponent.act(game, 1)
                game.update(controls)
                renderer.draw(game, score)
                renderer.tick(max(1, round(game.physics_hz * args.speed)))
                frame_index += 1
            if game.is_over:
                completed += 1
                if game.winner in (0, 1):
                    score[game.winner] += 1
                result = "蓝方胜" if game.winner == 0 else "紫方胜" if game.winner == 1 else "平局"
                print(f"本局结果：{result}｜累计比分：蓝方 {score[0]} : {score[1]} 紫方")
            elif skip:
                completed += 1
                print("本局结果：手动跳过")
    finally:
        renderer.close()


def main() -> None:
    # 从命令行启动模型观战窗口。
    _configure_console()
    watch(parse_args())


if __name__ == "__main__":
    main()
