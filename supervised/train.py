from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from rl.checkpoint import atomic_torch_save
from rl.environment import LAYOUTS, SPAWNS, TankSelfPlayEnv
from rl.model import TankActorCritic, load_actor_critic_state
from rl.observation import BULLET_FEATURES, MAP_CHANNELS, MAP_SIZE, MAX_BULLETS, MAX_OTHER_TANKS, SELF_FEATURES, TANK_FEATURES
from rl.train import _model_batch, stack_observations

from .teachers import HunterTeacher


def parse_args() -> argparse.Namespace:
    """行为克隆：两边都由寻路开火人机开车，模型只学动作。"""
    parser = argparse.ArgumentParser(description="Supervised clone of the hunter script.")
    parser.add_argument("--total-steps", type=int, default=200_000, help="采集的决策样本总数（两边坦克都算）")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--layout", choices=LAYOUTS, default="maze")
    parser.add_argument("--spawn", choices=SPAWNS, default="default")
    parser.add_argument("--time-limit", type=float, default=90.0)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--fire-weight", type=float, default=4.0, help="开火正样本的分类权重")
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/hunter_bc_lstm"))
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--initialize-from", type=Path, default=None)
    return parser.parse_args()


def save_checkpoint(path: Path, model: TankActorCritic, optimizer: torch.optim.Optimizer, args: argparse.Namespace, global_step: int) -> None:
    """保存监督模型断点。"""
    atomic_torch_save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "global_step": global_step,
            "config": vars(args),
        },
        path,
    )


def _accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    """分类准确率。"""
    return float((logits.argmax(dim=1) == target).float().mean().item())


def train(args: argparse.Namespace) -> Path:
    """人机对打采集标签，用交叉熵模仿三个动作头。"""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    model = TankActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
    global_step = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        load_actor_critic_state(model, checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        global_step = int(checkpoint.get("global_step", 0))
        print(f"resumed checkpoint={args.resume} global_step={global_step}")
    elif args.initialize_from is not None:
        checkpoint = torch.load(args.initialize_from, map_location=device, weights_only=False)
        load_actor_critic_state(model, checkpoint["model_state"])
        print(f"initialized model weights from={args.initialize_from}")

    environments = [
        TankSelfPlayEnv(
            action_repeat=args.action_repeat,
            rows=args.rows,
            cols=args.cols,
            time_limit=args.time_limit,
            layout=args.layout,
            spawn=args.spawn,
        )
        for _ in range(args.num_envs)
    ]
    teachers = [HunterTeacher(seed=args.seed + 17 + index) for index in range(args.num_envs)]
    current_observations = [env.reset(seed=args.seed + index) for index, env in enumerate(environments)]
    agents_per_batch = args.num_envs * TankSelfPlayEnv.num_agents
    batch_size = args.rollout_steps * agents_per_batch
    remaining_steps = max(0, args.total_steps - global_step)
    updates = int(np.ceil(remaining_steps / batch_size))
    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output / "teacher_log.csv"
    write_header = not log_path.is_file() or args.resume is None
    log_file = log_path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(
        log_file,
        fieldnames=(
            "update",
            "global_step",
            "loss",
            "throttle_acc",
            "steer_acc",
            "fire_acc",
            "fire_precision",
            "fire_recall",
            "fire_rate",
            "steps_per_second",
        ),
    )
    if write_header:
        writer.writeheader()
    fire_weight = torch.tensor([1.0, args.fire_weight], device=device)
    run_start_step = global_step
    started = time.perf_counter()
    print(f"supervised hunter BC device={device} updates={updates} output={args.output}")

    try:
        for update in range(1, updates + 1):
            maps = np.empty((args.rollout_steps, agents_per_batch, MAP_CHANNELS, MAP_SIZE, MAP_SIZE), dtype=np.uint8)
            selves = np.empty((args.rollout_steps, agents_per_batch, SELF_FEATURES), dtype=np.float32)
            tanks = np.empty((args.rollout_steps, agents_per_batch, MAX_OTHER_TANKS, TANK_FEATURES), dtype=np.float32)
            tank_mask = np.empty((args.rollout_steps, agents_per_batch, MAX_OTHER_TANKS), dtype=np.float32)
            bullets = np.empty((args.rollout_steps, agents_per_batch, MAX_BULLETS, BULLET_FEATURES), dtype=np.float32)
            bullet_mask = np.empty((args.rollout_steps, agents_per_batch, MAX_BULLETS), dtype=np.float32)
            labels = np.empty((args.rollout_steps, agents_per_batch, 3), dtype=np.int64)

            for rollout_step in range(args.rollout_steps):
                observations = [observation for pair in current_observations for observation in pair]
                batch = stack_observations(observations)
                maps[rollout_step] = batch["map"]
                selves[rollout_step] = batch["self"]
                tanks[rollout_step] = batch["tanks"]
                tank_mask[rollout_step] = batch["tank_mask"]
                bullets[rollout_step] = batch["bullets"]
                bullet_mask[rollout_step] = batch["bullet_mask"]
                next_observations = []
                for env_index, env in enumerate(environments):
                    joint = np.zeros((TankSelfPlayEnv.num_agents, 3), dtype=np.int64)
                    for agent_index, tank_id in enumerate(env.agent_ids):
                        joint[agent_index] = teachers[env_index].action(env.game, tank_id)
                    labels[rollout_step, env_index * 2 : env_index * 2 + 2] = joint
                    observations, _, done, _ = env.step(joint)
                    if done:
                        observations = env.reset()
                    next_observations.append(observations)
                current_observations = next_observations
                global_step += agents_per_batch

            packed = {
                "map": maps.reshape(batch_size, MAP_CHANNELS, MAP_SIZE, MAP_SIZE),
                "self": selves.reshape(batch_size, SELF_FEATURES),
                "tanks": tanks.reshape(batch_size, MAX_OTHER_TANKS, TANK_FEATURES),
                "tank_mask": tank_mask.reshape(batch_size, MAX_OTHER_TANKS),
                "bullets": bullets.reshape(batch_size, MAX_BULLETS, BULLET_FEATURES),
                "bullet_mask": bullet_mask.reshape(batch_size, MAX_BULLETS),
            }
            losses = []
            throttle_acc = []
            steer_acc = []
            fire_acc = []
            fire_tp = 0.0
            fire_fp = 0.0
            fire_fn = 0.0
            indices = np.arange(batch_size)
            for _ in range(args.epochs):
                np.random.shuffle(indices)
                for start in range(0, batch_size, args.minibatch_size):
                    mini = indices[start : start + args.minibatch_size]
                    throttle_logits, steer_logits, fire_logits, _value = model.action_logits(
                        *_model_batch(packed, device, mini)
                    )
                    target = torch.from_numpy(labels.reshape(batch_size, 3)[mini]).to(device)
                    loss = (
                        nn.functional.cross_entropy(throttle_logits, target[:, 0])
                        + nn.functional.cross_entropy(steer_logits, target[:, 1])
                        + nn.functional.cross_entropy(fire_logits, target[:, 2], weight=fire_weight)
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    losses.append(float(loss.item()))
                    throttle_acc.append(_accuracy(throttle_logits, target[:, 0]))
                    steer_acc.append(_accuracy(steer_logits, target[:, 1]))
                    fire_pred = fire_logits.argmax(dim=1)
                    fire_true = target[:, 2]
                    fire_acc.append(float((fire_pred == fire_true).float().mean().item()))
                    fire_tp += float(((fire_pred == 1) & (fire_true == 1)).sum().item())
                    fire_fp += float(((fire_pred == 1) & (fire_true == 0)).sum().item())
                    fire_fn += float(((fire_pred == 0) & (fire_true == 1)).sum().item())

            fire_rate = float((labels.reshape(-1, 3)[:, 2] == 1).mean())
            precision = fire_tp / max(fire_tp + fire_fp, 1e-6)
            recall = fire_tp / max(fire_tp + fire_fn, 1e-6)
            elapsed = max(time.perf_counter() - started, 1e-6)
            row = {
                "update": update,
                "global_step": global_step,
                "loss": float(np.mean(losses)),
                "throttle_acc": float(np.mean(throttle_acc)),
                "steer_acc": float(np.mean(steer_acc)),
                "fire_acc": float(np.mean(fire_acc)),
                "fire_precision": precision,
                "fire_recall": recall,
                "fire_rate": fire_rate,
                "steps_per_second": (global_step - run_start_step) / elapsed,
            }
            writer.writerow(row)
            log_file.flush()
            print(
                f"update={update}/{updates} step={global_step} loss={row['loss']:.4f} "
                f"thr={row['throttle_acc']:.3f} str={row['steer_acc']:.3f} "
                f"fire_acc={row['fire_acc']:.3f} fire_p={precision:.3f} fire_r={recall:.3f} fire_rate={fire_rate:.3f}"
            )
            save_checkpoint(args.output / "latest.pt", model, optimizer, args, global_step)
            if update % args.save_every == 0:
                save_checkpoint(args.output / f"step_{global_step}.pt", model, optimizer, args, global_step)
    finally:
        log_file.close()
    return args.output / "latest.pt"


def main() -> None:
    """命令行入口。"""
    train(parse_args())


if __name__ == "__main__":
    main()
