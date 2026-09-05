from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np

from rl.observation import (
    BULLET_FEATURES,
    MAP_CHANNELS,
    MAP_SIZE,
    MAX_BULLETS,
    MAX_OTHER_TANKS,
    SELF_FEATURES,
    TANK_FEATURES,
)


FORMAT_VERSION = 2
OBSERVATION_KEYS = (
    "map", "self", "self_pos", "tanks", "tank_pos", "tank_mask",
    "bullets", "bullet_pos", "bullet_mask",
)


def observation_spec() -> dict[str, object]:
    """记录数据集对应的观察结构，避免静默读取不兼容的旧数据。"""
    return {
        "map": [MAP_CHANNELS, MAP_SIZE, MAP_SIZE],
        "self": [SELF_FEATURES],
        "self_pos": [2],
        "tanks": [MAX_OTHER_TANKS, TANK_FEATURES],
        "tank_pos": [MAX_OTHER_TANKS, 2],
        "tank_mask": [MAX_OTHER_TANKS],
        "bullets": [MAX_BULLETS, BULLET_FEATURES],
        "bullet_pos": [MAX_BULLETS, 2],
        "bullet_mask": [MAX_BULLETS],
        "action": [3],
    }


def load_manifest(dataset_dir: Path) -> dict:
    """读取并验证离线数据清单与当前代码的观察结构。"""
    path = dataset_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"未找到数据清单：{path.resolve()}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"不支持的数据格式版本：{manifest.get('format_version')}")
    if manifest.get("observation_spec") != observation_spec():
        raise ValueError("数据集观察结构与当前模型不兼容，请重新采集。")
    train_seeds = set(manifest["splits"]["train"]["seeds"])
    validation_seeds = set(manifest["splits"]["validation"]["seeds"])
    overlap = train_seeds & validation_seeds
    if overlap:
        raise ValueError(f"训练/验证地图种子重叠：{sorted(overlap)[:8]}")
    return manifest


def split_shards(dataset_dir: Path, manifest: dict, split: str) -> list[Path]:
    """返回清单明确列出的分片；不会通过目录扫描误读临时文件。"""
    if split not in ("train", "validation"):
        raise ValueError(f"unknown dataset split: {split}")
    paths = [dataset_dir / item for item in manifest["splits"][split]["shards"]]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"数据分片缺失：{missing[0].resolve()}")
    return paths


def load_shard(path: Path) -> dict[str, np.ndarray]:
    """把一个压缩分片完整解压到内存；单个分片大小由采集参数控制。"""
    with np.load(path, allow_pickle=False) as payload:
        required = (*OBSERVATION_KEYS, "actions", "map_seeds", "episode_offsets", "episode_seeds")
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"分片 {path} 缺少字段：{missing}")
        return {key: payload[key] for key in required}


def iter_minibatches(
    shard: dict[str, np.ndarray],
    minibatch_size: int,
    rng: np.random.Generator,
    shuffle: bool,
) -> Iterator[tuple[dict[str, np.ndarray], np.ndarray]]:
    """从一个分片产生观察小批量和动作标签。"""
    count = int(shard["actions"].shape[0])
    indices = np.arange(count)
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, count, minibatch_size):
        selected = indices[start : start + minibatch_size]
        observations = {key: shard[key][selected] for key in OBSERVATION_KEYS}
        yield observations, shard["actions"][selected]
