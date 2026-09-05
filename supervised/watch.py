from __future__ import annotations

import argparse

import numpy as np
import pygame
import torch

from renderer import PygameRenderer
from rl.environment import TankSelfPlayEnv
from rl.model import TankActorCritic
from rl.train import _model_batch, stack_observations

from .teachers import HunterTeacher


def parse_args() -> argparse.Namespace:
    """观看人机对打，或监督模型对人机。"""
    parser = argparse.ArgumentParser(description="Watch hunter vs hunter, or a cloned model vs hunter.")
    parser.add_argument("--checkpoint", type=str, default=None, help="省略则双方都是人机")
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--layout", choices=("maze", "open"), default="maze")
    parser.add_argument("--spawn", choices=("default", "close_facing", "far_random"), default="default")
    parser.add_argument("--time-limit", type=float, default=90.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    return parser.parse_args()


def main() -> None:
    """打开窗口看寻路开火人机，或加载克隆模型打人机。"""
    args = parse_args()
    env = TankSelfPlayEnv(
        action_repeat=2,
        rows=args.rows,
        cols=args.cols,
        time_limit=args.time_limit,
        layout=args.layout,
        spawn=args.spawn,
    )
    teachers = [HunterTeacher(seed=args.seed + 1), HunterTeacher(seed=args.seed + 2)]
    model = None
    device = torch.device("cpu")
    if args.checkpoint:
        from pathlib import Path

        path = Path(args.checkpoint)
        if not path.is_file():
            raise SystemExit(f"未找到模型：{path.resolve()}")
        device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
        if device_name == "auto":
            device_name = "cpu"
        device = torch.device(device_name)
        payload = torch.load(path, map_location=device, weights_only=False)
        model = TankActorCritic().to(device)
        from rl.model import load_actor_critic_state

        load_actor_critic_state(model, payload["model_state"])
        model.eval()
    renderer = PygameRenderer(hud_rows=1)
    seed = args.seed
    observations = env.reset(seed=seed)
    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        seed += 1
                        observations = env.reset(seed=seed)
                    elif event.key in (pygame.K_n, pygame.K_SPACE, pygame.K_RETURN):
                        seed += 1
                        observations = env.reset(seed=seed)
            if not running:
                break
            joint = np.zeros((2, 3), dtype=np.int64)
            if model is None:
                for index, tank_id in enumerate(env.agent_ids):
                    joint[index] = teachers[index].action(env.game, tank_id)
            else:
                with torch.no_grad():
                    learner, _, _, _ = model.get_action_and_value(
                        *_model_batch(stack_observations([observations[0]]), device),
                        deterministic=True,
                    )
                joint[0] = learner.cpu().numpy()[0]
                joint[1] = teachers[1].action(env.game, env.agent_ids[1])
            observations, _, done, info = env.step(joint)
            winner = info.get("winner")
            hud = [f"寻路开火人机  winner={winner}  R换图  N跳过  Esc退出"]
            renderer.draw(env.game, hud)
            renderer.tick(max(1, env.game.physics_hz // env.action_repeat))
            if done:
                seed += 1
                observations = env.reset(seed=seed)
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
