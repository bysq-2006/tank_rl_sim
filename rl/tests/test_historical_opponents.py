from dataclasses import replace

import numpy as np
import torch

from rl.curriculum import STAGES
from rl.envs import TankRLEnv
from rl.model import TankActorCritic
from rl.opponents import FrozenModelOpponent, HistoricalOpponentPool


def test_historical_pool_selects_and_batches_one_frozen_model(tmp_path):
    # 验证旧策略能被强制抽中并为多个环境执行共享的批量推理。
    model = TankActorCritic()
    pool = HistoricalOpponentPool(tmp_path, torch.device("cpu"), seed=7, deterministic=True)
    snapshot = pool.save_snapshot(model, stage_index=0, total_steps=10)
    assert snapshot.exists()

    forced_history_stage = replace(STAGES[1], historical_opponent_probability=1.0)
    envs = [TankRLEnv(action_repeat=1, opponent_pool=pool) for _ in range(2)]
    for index, env in enumerate(envs):
        env.reset(forced_history_stage, seed=100 + index)
        assert isinstance(env.opponent, FrozenModelOpponent)

    actions = pool.batch_actions(envs)
    assert all(action is not None and len(action) == 3 for action in actions)
    assert all(all(value in (0, 1, 2) for value in action[:2]) for action in actions)
    assert all(action[2] in (0, 1) for action in actions)
    assert pool.cached_model_count == 1

    for env, action in zip(envs, actions):
        _, _, _, info = env.step(np.asarray((1, 1, 0)), opponent_action=action)
        assert info["historical_opponent"]
