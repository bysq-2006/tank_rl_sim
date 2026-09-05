from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from rl.environment import LAYOUTS, SPAWNS, TankSelfPlayEnv
from rl.train import stack_observations

from .dataset import FORMAT_VERSION, OBSERVATION_KEYS, observation_spec
from .teachers import HunterTeacher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect a fixed, seed-disjoint hunter dataset.")
    parser.add_argument("--output", type=Path, default=Path("datasets/hunter_exact_v4"))
    parser.add_argument("--train-seeds", type=int, default=800)
    parser.add_argument("--validation-seeds", type=int, default=200)
    parser.add_argument("--seed-start", type=int, default=10_000)
    parser.add_argument("--episodes-per-shard", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4, help="并行采集进程数；Windows 下每局彼此独立")
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--rows", type=int, default=None, help="省略则每张地图随机6..12")
    parser.add_argument("--cols", type=int, default=None, help="省略则每张地图随机6..12")
    parser.add_argument("--layout", choices=LAYOUTS, default="maze")
    parser.add_argument("--spawn", choices=SPAWNS, default="default")
    parser.add_argument("--time-limit", type=float, default=30.0)
    return parser.parse_args()


def _episode(env: TankSelfPlayEnv, map_seed: int) -> dict[str, np.ndarray]:
    """收集一张固定地图上的完整对局；两辆 hunter 都作为监督样本。"""
    teachers = (
        HunterTeacher(seed=map_seed * 2 + 1),
        HunterTeacher(seed=map_seed * 2 + 2),
    )
    observations = env.reset(seed=map_seed)
    storage: dict[str, list[np.ndarray]] = {key: [] for key in OBSERVATION_KEYS}
    actions: list[np.ndarray] = []
    done = False
    while not done:
        batch = stack_observations(observations)
        for key in OBSERVATION_KEYS:
            storage[key].append(batch[key])
        joint = np.asarray(
            [teachers[index].action(env.game, tank_id) for index, tank_id in enumerate(env.agent_ids)],
            dtype=np.int64,
        )
        actions.append(joint)
        observations, _rewards, done, _info = env.step(joint)
    result = {key: np.concatenate(values, axis=0) for key, values in storage.items()}
    result["actions"] = np.concatenate(actions, axis=0)
    result["map_seeds"] = np.full(result["actions"].shape[0], map_seed, dtype=np.int64)
    return result


def _episode_worker(task: tuple[dict[str, object], int]) -> tuple[int, dict[str, np.ndarray]]:
    """进程池入口：每个地图种子创建独立环境，结果与调度顺序无关。"""
    config, map_seed = task
    env = TankSelfPlayEnv(**config)
    return map_seed, _episode(env, map_seed)


def _write_shard(path: Path, episodes: list[tuple[int, dict[str, np.ndarray]]]) -> int:
    """原子写入若干完整对局；任何一局都不会跨越分片。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    offsets = [0]
    for _seed, episode in episodes:
        offsets.append(offsets[-1] + int(episode["actions"].shape[0]))
    payload = {
        key: np.concatenate([episode[key] for _seed, episode in episodes], axis=0)
        for key in (*OBSERVATION_KEYS, "actions", "map_seeds")
    }
    payload["episode_offsets"] = np.asarray(offsets, dtype=np.int64)
    payload["episode_seeds"] = np.asarray([seed for seed, _episode_data in episodes], dtype=np.int64)
    temporary = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)
    return int(payload["actions"].shape[0])


def collect(args: argparse.Namespace) -> Path:
    if args.train_seeds < 1 or args.validation_seeds < 1:
        raise SystemExit("train-seeds 和 validation-seeds 都必须至少为1")
    if args.episodes_per_shard < 1:
        raise SystemExit("episodes-per-shard 必须至少为1")
    workers = int(getattr(args, "workers", 1))
    if workers < 1:
        raise SystemExit("workers 必须至少为1")
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        raise SystemExit(f"输出目录已有数据清单，为避免混入旧数据请换新目录：{manifest_path.resolve()}")
    train = list(range(args.seed_start, args.seed_start + args.train_seeds))
    validation = list(
        range(args.seed_start + args.train_seeds, args.seed_start + args.train_seeds + args.validation_seeds)
    )
    env_config = {
        "action_repeat": args.action_repeat,
        "rows": args.rows,
        "cols": args.cols,
        "time_limit": args.time_limit,
        "layout": args.layout,
        "spawn": args.spawn,
    }
    split_records: dict[str, dict[str, object]] = {}
    for split_name, seeds in (("train", train), ("validation", validation)):
        shard_names: list[str] = []
        total_samples = 0
        pending: list[tuple[int, dict[str, np.ndarray]]] = []
        tasks = [(env_config, map_seed) for map_seed in seeds]
        if workers == 1:
            results = map(_episode_worker, tasks)
            executor = None
        else:
            executor = ProcessPoolExecutor(max_workers=workers)
            results = executor.map(_episode_worker, tasks, chunksize=1)
        for index, (map_seed, episode) in enumerate(results, start=1):
            pending.append((map_seed, episode))
            if len(pending) == args.episodes_per_shard or index == len(seeds):
                relative = Path(split_name) / f"shard_{len(shard_names):05d}.npz"
                count = _write_shard(args.output / relative, pending)
                shard_names.append(relative.as_posix())
                total_samples += count
                pending = []
                print(
                    f"split={split_name} maps={index}/{len(seeds)} "
                    f"samples={total_samples} shards={len(shard_names)}"
                )
        if executor is not None:
            executor.shutdown()
        split_records[split_name] = {"seeds": seeds, "shards": shard_names, "samples": total_samples}
    manifest = {
        "format_version": FORMAT_VERSION,
        "observation_spec": observation_spec(),
        "game": {
            "action_repeat": args.action_repeat,
            "rows": args.rows,
            "cols": args.cols,
            "layout": args.layout,
            "spawn": args.spawn,
            "time_limit": args.time_limit,
        },
        "trajectory_replay": "episode_seeds + episode_offsets + joint actions",
        "splits": split_records,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_name(".manifest.json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, manifest_path)
    print(f"dataset ready: {manifest_path.resolve()}")
    return manifest_path


def main() -> None:
    collect(parse_args())


if __name__ == "__main__":
    main()
