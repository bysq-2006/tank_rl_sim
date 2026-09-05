from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from rl.curriculum import STAGES, STAGE_TITLES, CurriculumManager
from rl.envs import TankRLEnv
from rl.evaluation import evaluate_policy
from rl.model import TankActorCritic
from rl.opponents import HistoricalOpponentPool

from .checkpoint import (
    load_checkpoint,
    prune_preview_checkpoints,
    save_checkpoint,
    save_preview_checkpoint,
)
from .dashboard import TrainingDashboard
from .ppo import PPOConfig, ppo_update
from .rollout import collect_rollout


def parse_args() -> argparse.Namespace:
    # 解析课程PPO训练所需的命令行参数。
    parser = argparse.ArgumentParser(description="从零开始进行坦克课程强化学习")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/tank_rl_curriculum_v2"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--total-steps", type=int, default=3_000_000)
    parser.add_argument("--additional-steps", type=int)
    parser.add_argument("--stage", type=int, choices=range(len(STAGES)))
    stage_mode = parser.add_mutually_exclusive_group()
    stage_mode.add_argument("--fixed-stage", dest="fixed_stage", action="store_true")
    stage_mode.add_argument("--auto-curriculum", dest="fixed_stage", action="store_false")
    parser.set_defaults(fixed_stage=None)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--evaluation-games", type=int, default=200)
    parser.add_argument("--evaluation-interval", type=int, default=50)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--preview-interval", type=int, default=10)
    parser.add_argument("--max-preview-checkpoints", type=int, default=20)
    parser.add_argument("--opponent-snapshot-interval", type=int, default=200)
    parser.add_argument("--no-historical-opponents", action="store_true")
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--dashboard-points", type=int, default=1000)
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _configure_console() -> None:
    # 在Windows等环境中强制使用UTF-8输出中文训练信息。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def _result_counts(results: list[dict]) -> dict[str, int]:
    # 汇总本次轨迹内结束对局的胜负、平局和超时数量。
    counts = {
        "本轮胜利局数": 0,
        "本轮失败局数": 0,
        "本轮平局数": 0,
        "本轮超时局数": 0,
        "本轮历史模型对局数": 0,
        "本轮己方实际开炮数": 0,
        "本轮敌方自毁局数": 0,
    }
    labels = {
        "win": "本轮胜利局数",
        "loss": "本轮失败局数",
        "draw": "本轮平局数",
        "timeout": "本轮超时局数",
    }
    for item in results:
        if item["result"] in labels:
            counts[labels[item["result"]]] += 1
        counts["本轮历史模型对局数"] += int(item.get("historical_opponent", False))
        counts["本轮己方实际开炮数"] += int(item.get("shots_fired", 0))
        counts["本轮敌方自毁局数"] += int(item.get("opponent_self_kill", False))
    return counts


def train(args: argparse.Namespace) -> None:
    # 初始化模型和环境并持续执行课程采集、PPO更新、评估与保存。
    if args.resume is None and (args.output / "latest.pt").exists():
        raise FileExistsError(
            f"输出目录已经存在训练检查点：{args.output / 'latest.pt'}；"
            "请使用 --resume 续训，或为全新训练指定另一个 --output。"
        )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    model = TankActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
    curriculum = CurriculumManager(start_stage=args.stage or 0, seed=args.seed)
    ppo_config = PPOConfig(minibatch_size=args.minibatch_size)
    start_update, total_steps = 0, 0
    if args.resume is not None:
        start_update, total_steps = load_checkpoint(args.resume, model, optimizer, curriculum, device)
    if args.stage is not None:
        curriculum.set_stage(args.stage)
    if args.fixed_stage is not None:
        curriculum.fixed_stage = args.fixed_stage

    args.output.mkdir(parents=True, exist_ok=True)
    opponent_pool = HistoricalOpponentPool(
        args.output,
        device,
        seed=args.seed + 2000,
        enabled=not args.no_historical_opponents,
    )
    if (
        opponent_pool.enabled
        and args.resume is not None
        and curriculum.current_stage >= 1
        and not opponent_pool.snapshots
    ):
        bootstrap_snapshot = opponent_pool.save_snapshot(model, curriculum.current_stage, total_steps)
        print(json.dumps({
            "已冻结续训起点模型": str(bootstrap_snapshot),
            "用途": "作为首个历史模型对手",
        }, ensure_ascii=False))
    seed_rng = np.random.default_rng(args.seed + 1000)

    def next_episode_seed() -> int:
        # 为每个新对局生成彼此独立且可复现的随机种子。
        return int(seed_rng.integers(0, 2**31 - 1))

    envs = [
        TankRLEnv(action_repeat=args.action_repeat, opponent_pool=opponent_pool)
        for _ in range(args.num_envs)
    ]
    observations = [env.reset(curriculum.sample_stage(), next_episode_seed()) for env in envs]
    steps_per_update = args.num_envs * args.rollout_steps
    target_steps = total_steps + args.additional_steps if args.additional_steps is not None else args.total_steps
    if target_steps <= total_steps:
        raise ValueError(
            f"target steps {target_steps} must exceed checkpoint steps {total_steps}; "
            "use --additional-steps to continue training"
        )
    remaining_updates = int(np.ceil((target_steps - total_steps) / steps_per_update))
    total_updates = start_update + remaining_updates
    dashboard = TrainingDashboard(
        args.output,
        enabled=not args.no_dashboard,
        maximum_points=args.dashboard_points,
        history_before_update=start_update,
    )
    print(json.dumps({
        "历史模型对手": "已启用" if opponent_pool.enabled else "已禁用",
        "已发现历史模型数": len(opponent_pool.snapshots),
        "历史模型目录": str(opponent_pool.pool_directory),
        "训练图表窗口": "已弹出" if dashboard.enabled else "未启用或绘图库不可用",
        "训练指标文件": str(dashboard.metrics_path),
    }, ensure_ascii=False))

    for update in range(start_update, total_updates):
        progress = min(total_steps / max(target_steps, 1), 1.0)
        learning_rate = args.learning_rate * (1.0 - progress)
        optimizer.param_groups[0]["lr"] = learning_rate
        entropy = 0.01 + progress * (0.001 - 0.01)
        current_ppo = replace(ppo_config, entropy_coefficient=entropy)
        rollout = collect_rollout(
            model, envs, observations, curriculum, args.rollout_steps,
            device, next_episode_seed, opponent_pool,
        )
        observations = rollout.next_observations
        metrics = ppo_update(model, optimizer, rollout, current_ppo)
        total_steps += steps_per_update
        result_counts = _result_counts(rollout.episode_results)
        completed_games = sum(
            result_counts[key]
            for key in (
                "本轮胜利局数", "本轮失败局数", "本轮平局数", "本轮超时局数",
                "本轮敌方自毁局数",
            )
        )
        log = {
            "更新轮次": update,
            "累计决策步数": total_steps,
            "当前关卡": curriculum.current_stage,
            "关卡名称": STAGE_TITLES[curriculum.current_stage],
            "学习率": learning_rate,
            **result_counts,
            "策略损失": metrics["policy_loss"],
            "价值损失": metrics["value_loss"],
            "策略熵": metrics["entropy"],
            "近似KL散度": metrics["approx_kl"],
        }
        print(json.dumps(log, ensure_ascii=False))

        dashboard_record = {
            "update": update + 1,
            "total_steps": total_steps,
            "stage": curriculum.current_stage,
            "learning_rate": learning_rate,
            "mean_step_reward": float(rollout.rewards.mean()),
            "rollout_win_rate": (
                result_counts["本轮胜利局数"] / completed_games
                if completed_games else None
            ),
            "shots_per_game": (
                result_counts["本轮己方实际开炮数"] / completed_games
                if completed_games else None
            ),
            "historical_opponent_ratio": (
                result_counts["本轮历史模型对局数"] / completed_games
                if completed_games else None
            ),
            "policy_loss": metrics["policy_loss"],
            "value_loss": metrics["value_loss"],
            "entropy": metrics["entropy"],
            "approx_kl": metrics["approx_kl"],
            "eval_win_rate": None,
            "eval_timeout_rate": None,
            "eval_mean_reward": None,
        }

        if (update + 1) % args.evaluation_interval == 0:
            evaluation = evaluate_policy(
                model, curriculum.current, args.evaluation_games,
                seed_start=1_000_000 + curriculum.current_stage * 10_000,
                device=device, action_repeat=args.action_repeat,
            )
            curriculum.replace_evaluation(curriculum.current_stage, evaluation.results)
            dashboard_record["eval_win_rate"] = evaluation.win_rate
            dashboard_record["eval_timeout_rate"] = evaluation.timeout_rate
            dashboard_record["eval_mean_reward"] = evaluation.mean_reward
            evaluated_stage = curriculum.current_stage
            promoted = curriculum.try_promote()
            print(json.dumps({
                "评估关卡": curriculum.current_stage - int(promoted),
                "评估胜率": evaluation.win_rate,
                "评估超时率": evaluation.timeout_rate,
                "平均累计奖励": evaluation.mean_reward,
                "是否晋级": "是" if promoted else "否",
            }, ensure_ascii=False))
            if promoted:
                stage_checkpoint = args.output / f"stage_{curriculum.current_stage}.pt"
                save_checkpoint(
                    stage_checkpoint,
                    model, optimizer, curriculum, update, total_steps,
                )
                opponent_pool.register_stage_checkpoint(stage_checkpoint, evaluated_stage)
                print(json.dumps({
                    "历史模型已登记": str(stage_checkpoint),
                    "代表已完成关卡": evaluated_stage,
                    "当前历史模型总数": len(opponent_pool.snapshots),
                }, ensure_ascii=False))

        dashboard_record["stage"] = curriculum.current_stage
        dashboard.record(dashboard_record)

        if (
            opponent_pool.enabled
            and curriculum.current_stage >= 1
            and args.opponent_snapshot_interval > 0
            and (update + 1) % args.opponent_snapshot_interval == 0
        ):
            snapshot_path = opponent_pool.save_snapshot(model, curriculum.current_stage, total_steps)
            print(json.dumps({
                "已保存历史对手快照": str(snapshot_path),
                "当前历史模型总数": len(opponent_pool.snapshots),
            }, ensure_ascii=False))

        if (update + 1) % args.checkpoint_interval == 0:
            save_checkpoint(args.output / "latest.pt", model, optimizer, curriculum, update, total_steps)

        if args.preview_interval > 0 and (update + 1) % args.preview_interval == 0:
            preview_directory = args.output / "previews"
            preview_path = preview_directory / (
                f"preview_update_{update + 1:06d}_step_{total_steps:012d}.pt"
            )
            save_preview_checkpoint(
                preview_path, model, curriculum, update, total_steps,
            )
            removed_previews = prune_preview_checkpoints(
                preview_directory, args.max_preview_checkpoints,
            )
            print(json.dumps({
                "已保存预览模型": str(preview_path),
                "当前关卡": curriculum.current_stage,
                "累计决策步数": total_steps,
                "清理旧预览数量": len(removed_previews),
            }, ensure_ascii=False))
            dashboard.save_image()

    save_checkpoint(args.output / "latest.pt", model, optimizer, curriculum, total_updates - 1, total_steps)
    dashboard.save_image()


def main() -> None:
    # 从命令行参数启动完整课程强化学习流程。
    _configure_console()
    train(parse_args())


if __name__ == "__main__":
    main()
