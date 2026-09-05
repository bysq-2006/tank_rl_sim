from __future__ import annotations

import argparse
from pathlib import Path

from rl.evaluate import evaluate as rl_evaluate


def parse_args() -> argparse.Namespace:
    """评估监督模型；默认对人机。"""
    parser = argparse.ArgumentParser(description="Evaluate a cloned hunter policy.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/hunter_bc/latest.pt"))
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--cols", type=int, default=None)
    parser.add_argument("--layout", choices=("maze", "open"), default=None)
    parser.add_argument("--spawn", choices=("default", "close_facing", "far_random"), default=None)
    parser.add_argument("--time-limit", type=float, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--opponent", nargs="+", default=["hunter"])
    parser.add_argument("--opponent-model", type=Path, nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    """复用 RL 观战入口，默认对手是寻路开火人机。"""
    args = parse_args()
    results = rl_evaluate(args)
    print(f"results: {results}")


if __name__ == "__main__":
    main()
