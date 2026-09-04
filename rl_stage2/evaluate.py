from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .environment import TankSelfPlayEnv
from .model import TankActorCritic
from .train import _model_batch, stack_observations


def parse_args() -> argparse.Namespace:
    """读取模型路径和试玩选项。"""
    parser = argparse.ArgumentParser(description="Evaluate a trained tank policy against itself.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/combat_stage2_open_far/latest.pt"))
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--cols", type=int, default=None)
    parser.add_argument("--layout", choices=("maze", "open"), default=None, help="省略时使用 checkpoint 里保存的关卡布局")
    parser.add_argument("--spawn", choices=("default", "close_facing", "far_random"), default=None)
    parser.add_argument("--time-limit", type=float, default=None, help="省略时使用 checkpoint 里保存的对局时长")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--stochastic", action="store_true", help="按概率采样动作，而不是总选概率最高动作")
    parser.add_argument("--no-render", action="store_true", help="关闭 Pygame 窗口并只输出统计")
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> dict[str, int]:
    """加载检查点，让同一个模型分别控制双方完成若干局对战。"""
    if not args.checkpoint.is_file():
        raise SystemExit(
            f"未找到模型文件：{args.checkpoint.resolve()}\n"
            "请先运行训练命令生成模型，例如：\n"
            "python -m rl_stage2.train"
        )
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = TankActorCritic().to(device)
    try:
        model.load_state_dict(checkpoint["model_state"])
    except RuntimeError as error:
        raise SystemExit("模型结构与当前集合编码器不兼容，需要用新观察重新训练。") from error
    model.eval()
    config = checkpoint.get("config", {})
    action_repeat = int(config.get("action_repeat", 2))
    env = TankSelfPlayEnv(
        action_repeat=action_repeat,
        rows=args.rows if args.rows is not None else config.get("rows", 6),
        cols=args.cols if args.cols is not None else config.get("cols", 6),
        time_limit=args.time_limit if args.time_limit is not None else float(config.get("time_limit", 90.0)),
        layout=args.layout or config.get("layout", "open"),
        spawn=args.spawn or config.get("spawn", "far_random"),
    )
    renderer = None
    if not args.no_render:
        from renderer import PygameRenderer

        renderer = PygameRenderer()

    results = {"tank_0": 0, "tank_1": 0, "draw": 0}
    try:
        for game_index in range(args.games):
            observations = env.reset(seed=args.seed + game_index)
            done = False
            info: dict[str, object] = {"winner": None}
            while not done:
                with torch.no_grad():
                    actions, _, _, _ = model.get_action_and_value(
                        *_model_batch(stack_observations(observations), device),
                        deterministic=not args.stochastic,
                    )
                observations, _, done, info = env.step(actions.cpu().numpy())
                if renderer is not None:
                    renderer.draw(env.game)
                    renderer.tick(max(1, env.game.physics_hz // action_repeat))
            winner = info["winner"]
            key = "draw" if winner is None else f"tank_{winner}"
            results[key] += 1
            print(f"game={game_index + 1}/{args.games} winner={winner} elapsed={env.game.elapsed:.2f}s")
    finally:
        if renderer is not None:
            renderer.close()
    return results


def main() -> None:
    """命令行评估入口。"""
    args = parse_args()
    results = evaluate(args)
    print(f"results: {results}")


if __name__ == "__main__":
    main()
