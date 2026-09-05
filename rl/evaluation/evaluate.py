from __future__ import annotations

import argparse
import json
import sys

import torch

from rl.curriculum import STAGES, STAGE_TITLES
from rl.evaluation import evaluate_policy
from rl.model import TankActorCritic


def parse_args() -> argparse.Namespace:
    # 解析独立评估模型所需的检查点、阶段和种子参数。
    parser = argparse.ArgumentParser(description="评估课程强化学习模型")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stage", type=int, default=0, choices=range(len(STAGES)))
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1_000_000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _configure_console() -> None:
    # 将标准输出设置为UTF-8以正确显示中文评估信息。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    # 加载指定检查点并输出固定种子评估指标。
    _configure_console()
    args = parse_args()
    device = torch.device(args.device)
    model = TankActorCritic().to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    result = evaluate_policy(model, STAGES[args.stage], args.games, args.seed, device)
    print(json.dumps({
        "评估关卡": args.stage,
        "关卡名称": STAGE_TITLES[args.stage],
        "评估局数": args.games,
        "胜率": result.win_rate,
        "超时率": result.timeout_rate,
        "平均累计奖励": result.mean_reward,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
