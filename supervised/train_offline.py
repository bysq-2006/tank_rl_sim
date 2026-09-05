from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from rl.checkpoint import atomic_torch_save
from rl.model import TankActorCritic, decode_actions, encode_actions, load_actor_critic_state
from rl.train import _model_batch

from .dataset import iter_minibatches, load_manifest, load_shard, split_shards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train behavior cloning on a fixed seed-disjoint dataset.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/hunter_bc_exact_v4"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--fire-weight", type=float, default=6.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--save-every", type=int, default=1)
    return parser.parse_args()


def _empty_metrics() -> dict[str, float]:
    return {
        "samples": 0.0,
        "loss": 0.0,
        "throttle_loss": 0.0,
        "steer_loss": 0.0,
        "fire_loss": 0.0,
        "throttle_correct": 0.0,
        "steer_correct": 0.0,
        "fire_correct": 0.0,
        "fire_tp": 0.0,
        "fire_fp": 0.0,
        "fire_fn": 0.0,
    }


def _finish_metrics(total: dict[str, float]) -> dict[str, float]:
    count = max(total["samples"], 1.0)
    result = {
        key: total[key] / count
        for key in ("loss", "throttle_loss", "steer_loss", "fire_loss")
    }
    result.update(
        {
            "throttle_acc": total["throttle_correct"] / count,
            "steer_acc": total["steer_correct"] / count,
            "fire_acc": total["fire_correct"] / count,
            "fire_precision": total["fire_tp"] / max(total["fire_tp"] + total["fire_fp"], 1.0),
            "fire_recall": total["fire_tp"] / max(total["fire_tp"] + total["fire_fn"], 1.0),
            "samples": total["samples"],
        }
    )
    return result


def _run_split(
    model: TankActorCritic,
    shard_paths: list[Path],
    device: torch.device,
    minibatch_size: int,
    fire_weight: torch.Tensor,
    rng: np.random.Generator,
    optimizer: torch.optim.Optimizer | None,
    max_grad_norm: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total = _empty_metrics()
    order = np.arange(len(shard_paths))
    if training:
        rng.shuffle(order)
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for shard_index in order:
            shard = load_shard(shard_paths[int(shard_index)])
            for observations, labels_array in iter_minibatches(
                shard, minibatch_size, rng, shuffle=training
            ):
                labels = torch.from_numpy(labels_array).to(device)
                logits, _value = model.action_logits(*_model_batch(observations, device))
                joint_labels = encode_actions(labels)
                per_sample = nn.functional.cross_entropy(logits, joint_labels, reduction="none")
                weights = fire_weight[labels[:, 2]]
                loss = (per_sample * weights).sum() / weights.sum()
                predictions = decode_actions(logits.argmax(dim=1))
                # 仅用于可读日志；优化目标是上面的联合18类交叉熵。
                throttle_loss = nn.functional.cross_entropy(
                    logits.reshape(-1, 3, 3, 2).logsumexp((2, 3)), labels[:, 0]
                )
                steer_loss = nn.functional.cross_entropy(
                    logits.reshape(-1, 3, 3, 2).logsumexp((1, 3)), labels[:, 1]
                )
                fire_loss = nn.functional.cross_entropy(
                    logits.reshape(-1, 3, 3, 2).logsumexp((1, 2)), labels[:, 2], weight=fire_weight
                )
                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                count = float(labels.shape[0])
                total["samples"] += count
                total["loss"] += float(loss.item()) * count
                total["throttle_loss"] += float(throttle_loss.item()) * count
                total["steer_loss"] += float(steer_loss.item()) * count
                total["fire_loss"] += float(fire_loss.item()) * count
                total["throttle_correct"] += float((predictions[:, 0] == labels[:, 0]).sum().item())
                total["steer_correct"] += float((predictions[:, 1] == labels[:, 1]).sum().item())
                total["fire_correct"] += float((predictions[:, 2] == labels[:, 2]).sum().item())
                total["fire_tp"] += float(((predictions[:, 2] == 1) & (labels[:, 2] == 1)).sum().item())
                total["fire_fp"] += float(((predictions[:, 2] == 1) & (labels[:, 2] == 0)).sum().item())
                total["fire_fn"] += float(((predictions[:, 2] == 0) & (labels[:, 2] == 1)).sum().item())
    return _finish_metrics(total)


def _save(
    path: Path,
    model: TankActorCritic,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
) -> None:
    atomic_torch_save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "global_step": global_step,
            "dataset_epoch": epoch,
            "config": vars(args),
        },
        path,
    )


def train(args: argparse.Namespace) -> Path:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    manifest = load_manifest(args.dataset)
    train_shards = split_shards(args.dataset, manifest, "train")
    validation_shards = split_shards(args.dataset, manifest, "validation")
    model = TankActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
    start_epoch = 0
    global_step = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        load_actor_critic_state(model, checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint.get("dataset_epoch", 0))
        global_step = int(checkpoint.get("global_step", 0))
    elif (args.output / "latest.pt").exists():
        raise SystemExit("输出目录已有模型；请使用 --resume 或换一个新输出目录。")
    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output / "offline_log.csv"
    append = args.resume is not None and log_path.exists()
    log_file = log_path.open("a" if append else "w", encoding="utf-8", newline="")
    fields = [
        "epoch", "global_step",
        "train_loss", "train_throttle_loss", "train_steer_loss", "train_fire_loss",
        "train_throttle_acc", "train_steer_acc", "train_fire_acc", "train_fire_precision", "train_fire_recall",
        "validation_loss", "validation_throttle_loss", "validation_steer_loss", "validation_fire_loss",
        "validation_throttle_acc", "validation_steer_acc", "validation_fire_acc",
        "validation_fire_precision", "validation_fire_recall",
    ]
    writer = csv.DictWriter(log_file, fieldnames=fields)
    if not append:
        writer.writeheader()
    fire_weight = torch.tensor([1.0, args.fire_weight], device=device)
    rng = np.random.default_rng(args.seed)
    print(
        f"offline BC device={device} train={manifest['splits']['train']['samples']} "
        f"validation={manifest['splits']['validation']['samples']} epochs={args.epochs}"
    )
    try:
        for epoch in range(start_epoch + 1, args.epochs + 1):
            train_metrics = _run_split(
                model, train_shards, device, args.minibatch_size, fire_weight,
                rng, optimizer, args.max_grad_norm,
            )
            global_step += int(train_metrics["samples"])
            validation_metrics = _run_split(
                model, validation_shards, device, args.minibatch_size, fire_weight,
                rng, None, args.max_grad_norm,
            )
            row = {"epoch": epoch, "global_step": global_step}
            for prefix, metrics in (("train", train_metrics), ("validation", validation_metrics)):
                for key in (
                    "loss", "throttle_loss", "steer_loss", "fire_loss", "throttle_acc",
                    "steer_acc", "fire_acc", "fire_precision", "fire_recall",
                ):
                    row[f"{prefix}_{key}"] = metrics[key]
            writer.writerow(row)
            log_file.flush()
            print(
                f"epoch={epoch}/{args.epochs} train_loss={train_metrics['loss']:.4f} "
                f"val_loss={validation_metrics['loss']:.4f} "
                f"val_acc=thr:{validation_metrics['throttle_acc']:.3f}/"
                f"str:{validation_metrics['steer_acc']:.3f}/fire:{validation_metrics['fire_acc']:.3f} "
                f"val_fire=p:{validation_metrics['fire_precision']:.3f}/r:{validation_metrics['fire_recall']:.3f}"
            )
            _save(args.output / "latest.pt", model, optimizer, args, epoch, global_step)
            if epoch % args.save_every == 0:
                _save(args.output / f"epoch_{epoch:03d}.pt", model, optimizer, args, epoch, global_step)
    finally:
        log_file.close()
    return args.output / "latest.pt"


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
