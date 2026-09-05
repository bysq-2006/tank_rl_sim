from __future__ import annotations

import argparse
from collections import defaultdict
import math
from pathlib import Path

import numpy as np
import torch

from .environment import TankSelfPlayEnv
from .model import TankActorCritic, load_actor_critic_state
from .opponents import build_opponent_controller
from .train import _model_batch, stack_observations

SCRIPT_LABELS = {
    "idle": "站桩",
    "move": "会走不开枪",
    "random": "乱打",
    "aim": "瞄准脚本",
    "chase": "追击脚本",
    "dodge": "躲避脚本",
    "hunter": "寻路开火",
}


def opponent_display_name(label: str) -> str:
    """把对手池条目变成计分板上的短中文名。"""
    if label in SCRIPT_LABELS:
        return SCRIPT_LABELS[label]
    if label == "镜像自己":
        return "镜像自己"
    path = Path(label)
    folder = path.parent.name or path.stem
    return folder


def total_hud(results: dict[str, int]) -> list[str]:
    """窗口只显示总比分：学员胜-对手胜。"""
    learner_wins = results.get("learner", 0)
    opponent_wins = results.get("opponent", 0)
    return [
        f"计分板  学员 {learner_wins}-{opponent_wins} 对手   平{results.get('draw', 0)} 跳{results.get('skip', 0)}"
    ]


def parse_args() -> argparse.Namespace:
    """读取模型路径和试玩选项。"""
    parser = argparse.ArgumentParser(description="Evaluate a trained tank policy against itself.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/rl/latest.pt"))
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--cols", type=int, default=None)
    parser.add_argument("--layout", choices=("maze", "open"), default=None, help="省略时使用 checkpoint 里保存的关卡布局")
    parser.add_argument("--spawn", choices=("default", "close_facing", "far_random"), default=None)
    parser.add_argument("--time-limit", type=float, default=None, help="省略时使用 checkpoint 里保存的对局时长")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--stochastic", action="store_true", help="按概率采样动作，而不是总选概率最高动作")
    parser.add_argument(
        "--paired-sides",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="相邻两局使用同一地图并交换双方位置，消除出生位偏差",
    )
    parser.add_argument("--no-render", action="store_true", help="关闭 Pygame 窗口并只输出统计")
    parser.add_argument(
        "--opponent",
        nargs="+",
        default=None,
        help="对手池；省略时使用 checkpoint 训练时保存的对手列表",
    )
    parser.add_argument(
        "--opponent-model",
        type=Path,
        nargs="*",
        default=None,
        help="配合 --opponent model 的冻结权重，可写多个",
    )
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> dict[str, int]:
    """加载检查点，让同一个模型分别控制双方完成若干局对战。"""
    if not args.checkpoint.is_file():
        raise SystemExit(
            f"未找到模型文件：{args.checkpoint.resolve()}\n"
            "请先运行训练命令生成模型，例如：\n"
            "python -m rl.train"
        )
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = TankActorCritic().to(device)
    try:
        load_actor_critic_state(model, checkpoint["model_state"])
    except RuntimeError as error:
        raise SystemExit("模型结构与当前集合编码器不兼容，需要用新观察重新训练。") from error
    model.eval()
    paired_sides = bool(getattr(args, "paired_sides", True))
    config = checkpoint.get("config", {})
    opponent_tokens = args.opponent
    opponent_models = args.opponent_model
    if opponent_tokens is None:
        saved = config.get("opponent", ["self"])
        if isinstance(saved, (str, Path)):
            opponent_tokens = [str(saved)]
        else:
            opponent_tokens = [str(item) for item in saved]
        if opponent_models is None:
            saved_models = config.get("opponent_model")
            if saved_models:
                if not isinstance(saved_models, (list, tuple)):
                    saved_models = [saved_models]
                opponent_models = [Path(item) for item in saved_models]
    saved_weights = config.get("opponent_weights") if args.opponent is None else None
    opponent = build_opponent_controller(
        opponent_tokens,
        opponent_models,
        device,
        args.seed + 3,
        weights=saved_weights,
        deterministic_models=not args.stochastic,
    )
    print(f"观战对手池：{opponent_tokens if opponent is not None else ['self']}")
    action_repeat = int(config.get("action_repeat", 2))
    env = TankSelfPlayEnv(
        action_repeat=action_repeat,
        rows=args.rows if args.rows is not None else config.get("rows", 6),
        cols=args.cols if args.cols is not None else config.get("cols", 6),
        time_limit=args.time_limit if args.time_limit is not None else float(config.get("time_limit", 90.0)),
        layout=args.layout or config.get("layout", "maze"),
        spawn=args.spawn or config.get("spawn", "default"),
    )
    learner_name = args.checkpoint.parent.name or args.checkpoint.stem
    if opponent is None:
        matchup_order = ["镜像自己"]
    else:
        matchup_order = list(dict.fromkeys(slot.label for slot in opponent.pool))
    board: dict[str, dict[str, int]] = defaultdict(lambda: {"win": 0, "lose": 0, "draw": 0, "skip": 0})
    for label in matchup_order:
        board[label]
    renderer = None
    if not args.no_render:
        from renderer import PygameRenderer

        renderer = PygameRenderer(hud_rows=1)

    results = {"learner": 0, "opponent": 0, "tank_0": 0, "tank_1": 0, "draw": 0, "skip": 0}
    abort = False
    try:
        for game_index in range(args.games):
            if abort:
                break
            pair_index = game_index // 2 if paired_sides else game_index
            observations = env.reset(seed=args.seed + pair_index)
            learner_slot = game_index % 2 if paired_sides else 0
            current_label = "镜像自己"
            if opponent is not None:
                if not paired_sides or game_index % 2 == 0:
                    opponent.reset_env(0)
                current_label = opponent.current_label(0)
            hud = total_hud(results)
            done = False
            skipped = False
            info: dict[str, object] = {"winner": None}
            while not done:
                if renderer is not None:
                    control = renderer.poll_control()
                    if control == "quit":
                        abort = True
                        break
                    if control == "skip":
                        skipped = True
                        break
                if opponent is None:
                    with torch.no_grad():
                        actions, _, _, _ = model.get_action_and_value(
                            *_model_batch(stack_observations(observations), device),
                            deterministic=not args.stochastic,
                        )
                    joint = actions.cpu().numpy()
                else:
                    with torch.no_grad():
                        learner_action, _, _, _ = model.get_action_and_value(
                            *_model_batch(stack_observations([observations[learner_slot]]), device),
                            deterministic=not args.stochastic,
                        )
                    other = 1 - learner_slot
                    joint = np.zeros((TankSelfPlayEnv.num_agents, 3), dtype=np.int64)
                    joint[learner_slot] = learner_action.cpu().numpy()[0]
                    joint[other] = opponent.action(0, env.game, env.agent_ids[other], observations[other])
                observations, _, done, info = env.step(joint)
                if renderer is not None:
                    renderer.draw(env.game, hud)
                    renderer.tick(max(1, env.game.physics_hz // action_repeat))
            if abort:
                print("观战已退出")
                break
            stats = board[current_label]
            if skipped:
                stats["skip"] += 1
                results["skip"] += 1
                print(f"game={game_index + 1}/{args.games} skipped vs={opponent_display_name(current_label)} elapsed={env.game.elapsed:.2f}s")
            else:
                winner = info["winner"]
                key = "draw" if winner is None else f"tank_{winner}"
                results[key] += 1
                if winner is None:
                    stats["draw"] += 1
                    outcome = "平"
                elif opponent is None:
                    if winner == env.agent_ids[0]:
                        stats["win"] += 1
                        results["learner"] += 1
                        outcome = "蓝胜"
                    else:
                        stats["lose"] += 1
                        results["opponent"] += 1
                        outcome = "红胜"
                elif winner == env.agent_ids[learner_slot]:
                    stats["win"] += 1
                    results["learner"] += 1
                    outcome = "学员胜"
                else:
                    stats["lose"] += 1
                    results["opponent"] += 1
                    outcome = "对手胜"
                print(
                    f"game={game_index + 1}/{args.games} vs={opponent_display_name(current_label)} "
                    f"{outcome} elapsed={env.game.elapsed:.2f}s  "
                    f"{learner_name} {stats['win']}-{stats['lose']} {opponent_display_name(current_label)}"
                )
    finally:
        if renderer is not None:
            renderer.close()
    print("计分板")
    for label in matchup_order:
        stats = board[label]
        decisive = stats["win"] + stats["lose"]
        if decisive:
            proportion = stats["win"] / decisive
            z = 1.96
            denominator = 1.0 + z * z / decisive
            center = (proportion + z * z / (2.0 * decisive)) / denominator
            margin = z * math.sqrt(proportion * (1.0 - proportion) / decisive + z * z / (4.0 * decisive * decisive)) / denominator
            interval = f"  胜率{proportion:.1%} 95%CI[{max(0.0, center-margin):.1%},{min(1.0, center+margin):.1%}]"
        else:
            interval = ""
        print(
            f"  vs {opponent_display_name(label)}: "
            f"{learner_name} {stats['win']}-{stats['lose']} 对手  平{stats['draw']} 跳{stats['skip']}{interval}"
        )
    return results


def main() -> None:
    """命令行评估入口。"""
    args = parse_args()
    results = evaluate(args)
    print(f"results: {results}")


if __name__ == "__main__":
    main()
