from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .checkpoint import atomic_torch_save
from .environment import LAYOUTS, SPAWNS, RewardConfig, TankSelfPlayEnv
from .model import TankActorCritic, load_actor_critic_state
from .observation import BULLET_FEATURES, MAP_CHANNELS, MAP_SIZE, MAX_BULLETS, MAX_OTHER_TANKS, Observation, SELF_FEATURES, TANK_FEATURES
from .live_plot import try_create_plot
from .opponents import build_opponent_controller, script_action

def parse_args() -> argparse.Namespace:
    """迷宫强化学习。"""
    parser = argparse.ArgumentParser(description="Train RL tank combat in a random maze.")
    parser.add_argument("--total-steps", type=int, default=200_000, help="智能体决策样本总数")
    parser.add_argument("--num-envs", type=int, default=16, help="并行维护的游戏局数")
    parser.add_argument("--rollout-steps", type=int, default=128, help="每次 PPO 更新前收集的决策步数")
    parser.add_argument("--action-repeat", type=int, default=2, help="一个动作保持的 24 Hz 物理帧数")
    parser.add_argument("--rows", type=int, default=6, help="固定地图行数")
    parser.add_argument("--cols", type=int, default=6, help="固定地图列数")
    parser.add_argument("--layout", choices=LAYOUTS, default="maze", help="maze 为随机迷宫，open 为只有外墙的空场")
    parser.add_argument("--spawn", choices=SPAWNS, default="default", help="坦克出生方式")
    parser.add_argument("--time-limit", type=float, default=90.0, help="每局最多持续的游戏秒数")
    parser.add_argument("--epochs", type=int, default=4, help="每批轨迹重复学习的轮数")
    parser.add_argument("--minibatch-size", type=int, default=512, help="PPO 小批量大小")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.99)
    parser.add_argument(
        "--potential-scale",
        type=float,
        default=0.2,
        help="势函数塑形强度；用 gamma*Phi(s')-Phi(s)，不会靠反复靠近刷累计奖励",
    )
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument(
        "--teacher-coef",
        type=float,
        default=0.0,
        help="在采到的状态上模仿 hunter 的辅助损失系数；建议从 0.03 开始并随学习率一起衰减",
    )
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.02, help="一次 PPO 更新超过该近似 KL 时提前停止 epoch；0 表示关闭")
    parser.add_argument("--anneal-lr", action=argparse.BooleanOptionalAction, default=True, help="训练过程中线性降低学习率")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/rl"), help="输出目录；断点续训时默认沿用模型所在目录")
    parser.add_argument("--save-every", type=int, default=10, help="每多少次更新额外保存一个模型")
    parser.add_argument(
        "--opponent",
        nargs="+",
        default=None,
        help="对手池：self，或脚本名 idle/move/aim/...，或冻结模型 .pt 路径，可写多个，每局随机抽一个",
    )
    parser.add_argument(
        "--opponent-model",
        type=Path,
        nargs="*",
        default=None,
        help="配合 --opponent model 的冻结权重，可写多个，都会进对手池",
    )
    parser.add_argument(
        "--opponent-weights",
        type=float,
        nargs="*",
        default=None,
        help="与展开后的对手池一一对应的采样权重；省略时均匀采样",
    )
    loading = parser.add_mutually_exclusive_group()
    loading.add_argument("--resume", type=Path, default=None, help="恢复模型、优化器和累计步数，继续同一训练阶段")
    loading.add_argument("--initialize-from", type=Path, default=None, help="只继承模型权重，并从0开始新的训练阶段")
    parser.add_argument("--no-plot", action="store_true", help="不弹出实时折线图窗口")
    return parser.parse_args()


def stack_observations(observations: list[Observation]) -> dict[str, np.ndarray]:
    """把若干智能体观察合并为一个批次，地图用 uint8 节省轨迹内存。"""
    return {
        "map": np.stack([observation["map"] for observation in observations]).astype(np.uint8),
        "self": np.stack([observation["self"] for observation in observations]).astype(np.float32),
        "tanks": np.stack([observation["tanks"] for observation in observations]).astype(np.float32),
        "tank_mask": np.stack([observation["tank_mask"] for observation in observations]).astype(np.float32),
        "bullets": np.stack([observation["bullets"] for observation in observations]).astype(np.float32),
        "bullet_mask": np.stack([observation["bullet_mask"] for observation in observations]).astype(np.float32),
    }


def _model_batch(batch: dict[str, np.ndarray], device: torch.device, indices: np.ndarray | None = None) -> tuple[torch.Tensor, ...]:
    """把观察批次转到模型设备，可选地切出一个小批量。"""
    tensors = []
    for key in ("map", "self", "tanks", "tank_mask", "bullets", "bullet_mask"):
        array = batch[key] if indices is None else batch[key][indices]
        tensors.append(torch.from_numpy(array).to(device))
    return tuple(tensors)


def save_checkpoint(
    path: Path,
    model: TankActorCritic,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    global_step: int,
) -> None:
    """保存可继续训练或用于试玩的完整检查点。"""
    atomic_torch_save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "global_step": global_step,
            "config": vars(args),
        },
        path,
    )


def train(args: argparse.Namespace) -> Path:
    """收集多局双智能体轨迹并循环执行 PPO 更新。"""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    if args.output is None:
        args.output = args.resume.parent if args.resume is not None else Path("checkpoints")
    model = TankActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
    global_step = 0
    saved_config: dict = {}
    if args.resume is not None:
        if not args.resume.is_file():
            raise SystemExit(f"未找到断点文件：{args.resume.resolve()}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        try:
            load_actor_critic_state(model, checkpoint["model_state"])
        except RuntimeError as error:
            raise SystemExit("断点模型结构与当前集合编码器不兼容，需要重新训练。") from error
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        global_step = int(checkpoint.get("global_step", 0))
        saved_config = checkpoint.get("config", {})
        for name in ("num_envs", "rollout_steps", "action_repeat", "rows", "cols", "layout", "spawn", "time_limit", "epochs", "minibatch_size", "gamma", "gae_lambda", "potential_scale", "clip_coef", "entropy_coef", "teacher_coef", "value_coef", "max_grad_norm", "target_kl", "anneal_lr"):
            if name in saved_config:
                setattr(args, name, saved_config[name])
        print(f"resumed checkpoint={args.resume} global_step={global_step}")
    elif args.initialize_from is not None:
        if not args.initialize_from.is_file():
            raise SystemExit(f"未找到初始化模型：{args.initialize_from.resolve()}")
        checkpoint = torch.load(args.initialize_from, map_location=device, weights_only=False)
        try:
            load_actor_critic_state(model, checkpoint["model_state"])
        except RuntimeError as error:
            raise SystemExit("初始化模型结构与当前集合编码器不兼容，需要重新训练。") from error
        print(f"initialized model weights from={args.initialize_from}")

    if args.resume is not None:
        if args.opponent is None:
            args.opponent = saved_config.get("opponent")
        if args.opponent_model is None:
            args.opponent_model = saved_config.get("opponent_model")
        if args.opponent_weights is None:
            args.opponent_weights = saved_config.get("opponent_weights")

    opponent = build_opponent_controller(
        args.opponent,
        args.opponent_model,
        device,
        args.seed + 97,
        weights=args.opponent_weights,
        deterministic_models=False,
    )
    if opponent is None:
        print("opponent=self (shared policy)")
    else:
        print(f"opponent pool=[{opponent.describe()}]")

    environments = [
        TankSelfPlayEnv(
            action_repeat=args.action_repeat,
            rows=args.rows,
            cols=args.cols,
            time_limit=args.time_limit,
            layout=getattr(args, "layout", "maze"),
            spawn=getattr(args, "spawn", "default"),
            reward_config=RewardConfig(
                potential_scale=args.potential_scale,
                potential_gamma=args.gamma,
            ),
        )
        for _ in range(args.num_envs)
    ]
    current_observations = [env.reset(seed=args.seed + index) for index, env in enumerate(environments)]
    learner_index = np.zeros(args.num_envs, dtype=np.int64)
    if opponent is not None:
        for env_index in range(args.num_envs):
            learner_index[env_index] = np.random.randint(0, TankSelfPlayEnv.num_agents)
            opponent.reset_env(env_index)

    learners_per_env = TankSelfPlayEnv.num_agents if opponent is None else 1
    agents_per_batch = args.num_envs * learners_per_env
    batch_size = args.rollout_steps * agents_per_batch
    minibatch_size = min(args.minibatch_size, batch_size)
    remaining_steps = max(0, args.total_steps - global_step)
    updates = int(np.ceil(remaining_steps / batch_size))
    episode_returns = np.zeros((args.num_envs, TankSelfPlayEnv.num_agents), dtype=np.float32)
    teacher_rng = np.random.default_rng(args.seed + 313)
    teacher_fire_weight = torch.tensor([1.0, 4.0], device=device)
    recent_returns: deque[float] = deque(maxlen=100)
    recent_results: deque[int] = deque(maxlen=100)  # 1 为分出胜负，0 为平局。
    recent_learner_wins: deque[int] = deque(maxlen=100)
    recent_matchups: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=100))
    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output / "training_log.csv"
    start_time = time.perf_counter()
    run_steps = 0

    if updates == 0:
        save_checkpoint(args.output / "latest.pt", model, optimizer, args, global_step)
        print(f"target total steps already reached: {global_step} >= {args.total_steps}")
        return args.output / "latest.pt"

    plot = None if args.no_plot else try_create_plot()
    if plot is not None and args.resume is not None:
        plot.load_csv(log_path)

    append_log = args.resume is not None and log_path.exists()
    with log_path.open("a" if append_log else "w", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        if not append_log:
            writer.writerow((
                "update", "global_step", "mean_return_100", "decisive_rate_100", "learner_win_100",
                "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "explained_variance",
                "grad_norm", "teacher_loss", "reverse_rate", "forward_rate", "turn_rate", "fire_action_rate",
                "learning_rate", "opponent_results_100", "steps_per_second",
            ))

        for update in range(1, updates + 1):
            if args.anneal_lr:
                fraction = 1.0 - (update - 1.0) / max(updates, 1)
                optimizer.param_groups[0]["lr"] = args.learning_rate * fraction
            rollout = {
                "map": np.empty((args.rollout_steps, agents_per_batch, MAP_CHANNELS, MAP_SIZE, MAP_SIZE), dtype=np.uint8),
                "self": np.empty((args.rollout_steps, agents_per_batch, SELF_FEATURES), dtype=np.float32),
                "tanks": np.empty((args.rollout_steps, agents_per_batch, MAX_OTHER_TANKS, TANK_FEATURES), dtype=np.float32),
                "tank_mask": np.empty((args.rollout_steps, agents_per_batch, MAX_OTHER_TANKS), dtype=np.float32),
                "bullets": np.empty((args.rollout_steps, agents_per_batch, MAX_BULLETS, BULLET_FEATURES), dtype=np.float32),
                "bullet_mask": np.empty((args.rollout_steps, agents_per_batch, MAX_BULLETS), dtype=np.float32),
            }
            actions = np.empty((args.rollout_steps, agents_per_batch, 3), dtype=np.int64)
            teacher_actions = (
                np.empty((args.rollout_steps, agents_per_batch, 3), dtype=np.int64)
                if args.teacher_coef > 0.0 else None
            )
            log_probs = np.empty((args.rollout_steps, agents_per_batch), dtype=np.float32)
            values = np.empty((args.rollout_steps, agents_per_batch), dtype=np.float32)
            rewards = np.empty((args.rollout_steps, agents_per_batch), dtype=np.float32)
            dones = np.empty((args.rollout_steps, agents_per_batch), dtype=np.float32)

            for rollout_step in range(args.rollout_steps):
                if opponent is None:
                    learner_observations = [observation for pair in current_observations for observation in pair]
                else:
                    learner_observations = [
                        current_observations[env_index][int(learner_index[env_index])]
                        for env_index in range(args.num_envs)
                    ]
                if teacher_actions is not None:
                    if opponent is None:
                        teacher_actions[rollout_step] = np.asarray([
                            script_action(env.game, tank_id, "hunter", teacher_rng)
                            for env in environments for tank_id in env.agent_ids
                        ], dtype=np.int64)
                    else:
                        teacher_actions[rollout_step] = np.asarray([
                            script_action(
                                env.game,
                                env.agent_ids[int(learner_index[env_index])],
                                "hunter",
                                teacher_rng,
                            )
                            for env_index, env in enumerate(environments)
                        ], dtype=np.int64)
                step_batch = stack_observations(learner_observations)
                for key, array in step_batch.items():
                    rollout[key][rollout_step] = array
                with torch.no_grad():
                    action_tensor, log_prob_tensor, _, value_tensor = model.get_action_and_value(*_model_batch(step_batch, device))
                learner_actions = action_tensor.cpu().numpy()
                actions[rollout_step] = learner_actions
                log_probs[rollout_step] = log_prob_tensor.cpu().numpy()
                values[rollout_step] = value_tensor.cpu().numpy()

                next_observations: list[list[Observation]] = []
                for env_index, env in enumerate(environments):
                    if opponent is None:
                        begin = env_index * TankSelfPlayEnv.num_agents
                        end = begin + TankSelfPlayEnv.num_agents
                        joint_actions = learner_actions[begin:end]
                        slot = None
                    else:
                        slot = int(learner_index[env_index])
                        other = 1 - slot
                        joint_actions = np.zeros((TankSelfPlayEnv.num_agents, 3), dtype=np.int64)
                        joint_actions[slot] = learner_actions[env_index]
                        joint_actions[other] = opponent.action(
                            env_index,
                            env.game,
                            env.agent_ids[other],
                            current_observations[env_index][other],
                        )
                    observations, env_rewards, done, info = env.step(joint_actions)
                    if opponent is None:
                        begin = env_index * TankSelfPlayEnv.num_agents
                        end = begin + TankSelfPlayEnv.num_agents
                        rewards[rollout_step, begin:end] = env_rewards
                        dones[rollout_step, begin:end] = float(done)
                    else:
                        rewards[rollout_step, env_index] = env_rewards[slot]
                        dones[rollout_step, env_index] = float(done)
                    episode_returns[env_index] += env_rewards
                    if done:
                        if opponent is None:
                            recent_returns.extend(float(value) for value in episode_returns[env_index])
                        else:
                            recent_returns.append(float(episode_returns[env_index, slot]))
                            learner_won = int(info["winner"] == env.agent_ids[slot])
                            recent_learner_wins.append(learner_won)
                            label = opponent.current_label(env_index)
                            outcome = 1 if learner_won else (0 if info["winner"] is None else -1)
                            recent_matchups[label].append(outcome)
                        recent_results.append(int(info["winner"] is not None))
                        episode_returns[env_index].fill(0.0)
                        observations = env.reset()
                        if opponent is not None:
                            learner_index[env_index] = np.random.randint(0, TankSelfPlayEnv.num_agents)
                            opponent.reset_env(env_index)
                    next_observations.append(observations)
                current_observations = next_observations
                global_step += agents_per_batch
                run_steps += agents_per_batch

            if opponent is None:
                flat_next = [observation for pair in current_observations for observation in pair]
            else:
                flat_next = [
                    current_observations[env_index][int(learner_index[env_index])]
                    for env_index in range(args.num_envs)
                ]
            with torch.no_grad():
                _, _, _, next_values_tensor = model.get_action_and_value(
                    *_model_batch(stack_observations(flat_next), device),
                    deterministic=True,
                )
            next_values = next_values_tensor.cpu().numpy()
            advantages = np.zeros_like(rewards)
            last_gae = np.zeros(agents_per_batch, dtype=np.float32)
            for step in reversed(range(args.rollout_steps)):
                following_values = next_values if step == args.rollout_steps - 1 else values[step + 1]
                nonterminal = 1.0 - dones[step]
                delta = rewards[step] + args.gamma * following_values * nonterminal - values[step]
                last_gae = delta + args.gamma * args.gae_lambda * nonterminal * last_gae
                advantages[step] = last_gae
            returns = advantages + values

            flat_batch = {key: array.reshape((batch_size, *array.shape[2:])) for key, array in rollout.items()}
            flat_actions = actions.reshape((-1, 3))
            flat_teacher_actions = None if teacher_actions is None else teacher_actions.reshape((-1, 3))
            flat_log_probs = log_probs.reshape(-1)
            flat_advantages = advantages.reshape(-1)
            flat_returns = returns.reshape(-1)
            indices = np.arange(batch_size)
            metrics = []
            for _epoch in range(args.epochs):
                np.random.shuffle(indices)
                epoch_kls = []
                for start in range(0, batch_size, minibatch_size):
                    batch_indices = indices[start : start + minibatch_size]
                    _, new_log_prob, entropy, new_value = model.get_action_and_value(
                        *_model_batch(flat_batch, device, batch_indices),
                        action=torch.from_numpy(flat_actions[batch_indices]).to(device),
                    )
                    old_log_prob = torch.from_numpy(flat_log_probs[batch_indices]).to(device)
                    batch_advantage = torch.from_numpy(flat_advantages[batch_indices]).to(device)
                    batch_advantage = (batch_advantage - batch_advantage.mean()) / (batch_advantage.std() + 1e-8)
                    log_ratio = new_log_prob - old_log_prob
                    ratio = log_ratio.exp()
                    unclipped = ratio * batch_advantage
                    clipped = torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef) * batch_advantage
                    policy_loss = -torch.min(unclipped, clipped).mean()
                    target_return = torch.from_numpy(flat_returns[batch_indices]).to(device)
                    value_loss = 0.5 * (new_value - target_return).pow(2).mean()
                    entropy_mean = entropy.mean()
                    teacher_loss = torch.zeros((), device=device)
                    if flat_teacher_actions is not None:
                        throttle_logits, steer_logits, fire_logits, _ = model.action_logits(
                            *_model_batch(flat_batch, device, batch_indices)
                        )
                        teacher_target = torch.from_numpy(flat_teacher_actions[batch_indices]).to(device)
                        teacher_loss = (
                            nn.functional.cross_entropy(throttle_logits, teacher_target[:, 0])
                            + nn.functional.cross_entropy(steer_logits, teacher_target[:, 1])
                            + nn.functional.cross_entropy(
                                fire_logits, teacher_target[:, 2], weight=teacher_fire_weight
                            )
                        ) / 3.0
                    teacher_weight = args.teacher_coef * (
                        optimizer.param_groups[0]["lr"] / max(args.learning_rate, 1e-12)
                    )
                    loss = (
                        policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy_mean
                        + teacher_weight * teacher_loss
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    with torch.no_grad():
                        approx_kl = ((ratio - 1.0) - log_ratio).mean()
                        clip_fraction = ((ratio - 1.0).abs() > args.clip_coef).float().mean()
                    epoch_kls.append(float(approx_kl.item()))
                    metrics.append((
                        policy_loss.item(), value_loss.item(), entropy_mean.item(), approx_kl.item(),
                        clip_fraction.item(), float(grad_norm.item()), float(teacher_loss.item()),
                    ))
                if args.target_kl > 0 and epoch_kls and float(np.mean(epoch_kls)) > args.target_kl:
                    break

            mean_metrics = np.mean(metrics, axis=0)
            return_variance = float(np.var(flat_returns))
            explained_variance = (
                float("nan") if return_variance < 1e-8
                else 1.0 - float(np.var(flat_returns - values.reshape(-1))) / return_variance
            )
            elapsed = max(time.perf_counter() - start_time, 1e-6)
            mean_return = float(np.mean(recent_returns)) if recent_returns else float("nan")
            decisive_rate = float(np.mean(recent_results)) if recent_results else float("nan")
            learner_win = float(np.mean(recent_learner_wins)) if recent_learner_wins else float("nan")
            fire_action_rate = float((flat_actions[:, 2] == 1).mean())
            reverse_rate = float((flat_actions[:, 0] == 0).mean())
            forward_rate = float((flat_actions[:, 0] == 2).mean())
            turn_rate = float((flat_actions[:, 1] != 1).mean())
            matchup_summary = {
                label: {
                    "win": int(sum(value == 1 for value in outcomes)),
                    "loss": int(sum(value == -1 for value in outcomes)),
                    "draw": int(sum(value == 0 for value in outcomes)),
                }
                for label, outcomes in recent_matchups.items()
            }
            matchup_text = " ".join(
                f"{Path(label).parent.name or label}:"
                f"{stats['win']}-{stats['loss']}-{stats['draw']}"
                for label, stats in matchup_summary.items()
            )
            steps_per_second = int(run_steps / elapsed)
            print(
                f"update={update}/{updates} step={global_step} return100={mean_return:.3f} "
                f"decisive100={decisive_rate:.2f} win100={learner_win:.2f} policy={mean_metrics[0]:.4f} "
                f"value={mean_metrics[1]:.4f} entropy={mean_metrics[2]:.4f} "
                f"kl={mean_metrics[3]:.5f} clip={mean_metrics[4]:.3f} ev={explained_variance:.3f} "
                f"teacher={mean_metrics[6]:.3f} "
                f"fire_rate={fire_action_rate:.3f} pool=[{matchup_text}] sps={steps_per_second}"
            )
            writer.writerow((
                update, global_step, mean_return, decisive_rate, learner_win,
                mean_metrics[0], mean_metrics[1], mean_metrics[2], mean_metrics[3], mean_metrics[4],
                explained_variance, mean_metrics[5], mean_metrics[6], reverse_rate, forward_rate, turn_rate, fire_action_rate,
                optimizer.param_groups[0]["lr"], json.dumps(matchup_summary, ensure_ascii=False), steps_per_second,
            ))
            log_file.flush()
            if plot is not None:
                plot.update(global_step, mean_return, learner_win, float(mean_metrics[2]))
            save_checkpoint(args.output / "latest.pt", model, optimizer, args, global_step)
            if args.save_every > 0 and update % args.save_every == 0:
                save_checkpoint(args.output / f"step_{global_step}.pt", model, optimizer, args, global_step)

    if plot is not None:
        plot.close()
    return args.output / "latest.pt"


def main() -> None:
    """命令行训练入口。"""
    args = parse_args()
    checkpoint = train(args)
    print(f"training finished: {checkpoint.resolve()}")


if __name__ == "__main__":
    main()
