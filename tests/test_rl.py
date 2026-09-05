import math
from argparse import Namespace

import numpy as np
import torch

from core import TankGame
from core.entities import Bullet
from rl.environment import RewardConfig, TankSelfPlayEnv
from rl.model import TankActorCritic, decode_actions, encode_actions, load_actor_critic_state
from rl.observation import BULLET_FEATURES, MAP_CHANNELS, MAP_SIZE, MAX_BULLETS, MAX_OTHER_TANKS, SELF_FEATURES, TANK_FEATURES, build_observation
from rl.train import _model_batch, stack_observations
from rl.planning import astar_distance, tank_cell
from rl.trajectory import proximity_score, trace_bullet_trajectory
from supervised.teachers import HunterTeacher
from supervised.collect import collect as collect_supervised_dataset
from supervised.dataset import load_manifest, load_shard, split_shards
from supervised.train_offline import train as train_offline_bc


def make_open_arena(game: TankGame, rows: int = 6, cols: int = 6) -> None:
    from rl.environment import make_open_maze

    game.maze = make_open_maze(rows, cols)
    game.wall_rects = game.maze.wall_rects(game.wall_thickness)


def test_observation_has_fixed_shapes_for_different_maps():
    for rows, cols in ((6, 12), (12, 6), (9, 9)):
        game = TankGame(rows=rows, cols=cols)
        observation = build_observation(game, tank_id=0)
        assert observation["map"].shape == (MAP_CHANNELS, MAP_SIZE, MAP_SIZE)
        assert observation["self"].shape == (SELF_FEATURES,)
        assert observation["self_pos"].shape == (2,)
        assert observation["tanks"].shape == (MAX_OTHER_TANKS, TANK_FEATURES)
        assert observation["tank_pos"].shape == (MAX_OTHER_TANKS, 2)
        assert observation["bullets"].shape == (MAX_BULLETS, BULLET_FEATURES)
        assert observation["bullet_pos"].shape == (MAX_BULLETS, 2)
        assert observation["tank_mask"].shape == (MAX_OTHER_TANKS,)
        assert observation["bullet_mask"].shape == (MAX_BULLETS,)


def test_observation_map_is_high_resolution_wall_pixels():
    game = TankGame(rows=6, cols=6)
    observation = build_observation(game, tank_id=0)
    assert MAP_CHANNELS == 2
    assert MAP_SIZE == 96
    assert observation["map"].shape == (2, MAP_SIZE, MAP_SIZE)
    assert observation["map"].max() <= 1.0
    wall, valid = observation["map"]
    assert wall.sum() > 0
    assert valid[:48, :48].min() == 1
    assert valid[48:, :].sum() == 0
    assert valid[:, 48:].sum() == 0


def test_bullet_set_marks_enemy_shot_and_ignores_empty_slots():
    game = TankGame(rows=6, cols=6)
    game._fire(game.tanks[1])
    observation = build_observation(game, tank_id=0)
    assert observation["bullet_mask"].sum() == 1
    assert observation["bullets"][0, 6] == -1.0
    assert observation["bullets"][1:].sum() == 0


def test_bullet_position_keeps_subcell_precision():
    game = TankGame(rows=6, cols=6)
    enemy = game.tanks[1]
    game.bullets = [Bullet(2.125, 3.875, 1.0, 0.0, enemy.tank_id)]
    observation = build_observation(game, tank_id=0)
    assert np.isclose(observation["bullets"][0, 7], 0.125)
    assert np.isclose(observation["bullets"][0, 8], 0.875)
    assert np.allclose(observation["bullet_pos"][0], (2 * 2.125 / 12 - 1, 2 * 3.875 / 12 - 1))


def test_observation_encodes_extra_tanks_without_changing_shapes():
    game = TankGame(rows=6, cols=6)
    extra = game.tanks[1]
    game.tanks.append(type(extra)(3.5, 3.5, 0.0, tank_id=2))
    observation = build_observation(game, tank_id=0)
    assert observation["tank_mask"].sum() == 2
    assert observation["tanks"].shape == (MAX_OTHER_TANKS, TANK_FEATURES)


def test_environment_advances_repeated_physics_frames():
    env = TankSelfPlayEnv(action_repeat=2)
    env.reset(seed=3)
    _, rewards, done, info = env.step([(1, 1, 0), (1, 1, 0)])
    assert env.game.elapsed == 2 * env.game.dt
    assert rewards.shape == (2,)
    assert np.allclose(rewards, 0.0)
    assert done is False
    assert info["frames_executed"] == 2


def test_disabled_path_progress_does_not_shape_movement():
    env = TankSelfPlayEnv(action_repeat=1, layout="open", spawn="close_facing")
    env.reset(seed=3)
    idle = 0.0
    for _ in range(12):
        _, rewards, _, _ = env.step([(1, 1, 0), (2, 1, 0)])
        idle += float(rewards[0])
    env.reset(seed=3)
    approaching = 0.0
    for _ in range(12):
        _, rewards, _, _ = env.step([(2, 1, 0), (1, 1, 0)])
        approaching += float(rewards[0])
    assert approaching == idle == 0.0


def test_potential_shaping_rewards_useful_state_progress():
    env = TankSelfPlayEnv(
        action_repeat=1,
        layout="open",
        spawn="close_facing",
        reward_config=RewardConfig(potential_scale=1.0, potential_gamma=1.0),
    )
    env.reset(seed=3)
    _, rewards, done, _ = env.step([(2, 1, 0), (1, 1, 0)])
    assert done is False
    assert rewards[0] > 0.0


def test_survivor_wins_even_when_enemy_destroys_itself():
    env = TankSelfPlayEnv(action_repeat=1)
    env.reset(seed=4)
    env.game.tanks[0].alive = False
    _, rewards, done, info = env.step([(1, 1, 0), (1, 1, 0)])
    assert done is True
    assert info["winner"] == 1
    assert np.isclose(rewards[0], RewardConfig().loss)
    assert np.isclose(rewards[1], RewardConfig().win)


def test_own_bullet_kill_uses_small_self_kill_penalty():
    env = TankSelfPlayEnv(action_repeat=1)
    env.reset(seed=8)
    victim = env.game.tanks[0]
    env.game.bullets.append(Bullet(victim.x, victim.y, 0.0, 0.0, owner_tank_id=victim.tank_id, age=1.0))
    _, rewards, done, info = env.step([(1, 1, 0), (1, 1, 0)])
    assert done is True
    assert info["winner"] == 1
    assert np.isclose(rewards[0], RewardConfig().self_kill)
    assert np.isclose(rewards[1], RewardConfig().win)


def test_opponent_self_kill_can_give_no_passive_win_reward():
    env = TankSelfPlayEnv(
        action_repeat=1,
        reward_config=RewardConfig(opponent_self_kill_win=0.0),
    )
    env.reset(seed=8)
    victim = env.game.tanks[0]
    env.game.bullets.append(
        Bullet(victim.x, victim.y, 0.0, 0.0, owner_tank_id=victim.tank_id, age=1.0)
    )
    _, rewards, done, info = env.step([(1, 1, 0), (1, 1, 0)])
    assert done is True
    assert info["winner"] == 1
    assert np.isclose(rewards[0], RewardConfig().self_kill)
    assert np.isclose(rewards[1], 0.0)


def test_timeout_is_worse_than_waiting_without_terminal_result():
    env = TankSelfPlayEnv(action_repeat=1, time_limit=1.0 / 24.0)
    env.reset(seed=5)
    _, rewards, done, info = env.step([(1, 1, 0), (1, 1, 0)])
    assert done is True
    assert info["winner"] is None
    assert np.allclose(rewards, RewardConfig().timeout)


def test_open_close_spawn_places_tanks_near_and_facing():
    env = TankSelfPlayEnv(layout="open", spawn="close_facing", rows=6, cols=6)
    env.reset(seed=11)
    first, second = env.game.tanks
    assert math.hypot(second.x - first.x, second.y - first.y) < 3.0
    assert env.game.maze.horizontal[3].sum() == 0


def test_open_far_spawn_places_tanks_apart():
    env = TankSelfPlayEnv(layout="open", spawn="far_random", rows=6, cols=6)
    env.reset(seed=12)
    first, second = env.game.tanks
    assert math.hypot(second.x - first.x, second.y - first.y) > 3.0


def test_aim_script_stops_and_fires_when_facing_enemy():
    from rl.opponents import script_action

    env = TankSelfPlayEnv(layout="open", spawn="close_facing", rows=6, cols=6)
    env.reset(seed=11)
    first, second = env.game.tanks
    first.x, first.y, first.heading = 2.0, 3.0, 0.0
    second.x, second.y = 4.0, 3.0
    throttle, steer, fire = script_action(env.game, first.tank_id, "aim", np.random.default_rng(0))
    assert throttle == 1
    assert steer == 1
    assert fire == 1


def test_hunter_fires_when_facing_with_line_of_sight():
    from rl.opponents import script_action

    env = TankSelfPlayEnv(layout="open", spawn="close_facing", rows=6, cols=6)
    env.reset(seed=11)
    first, second = env.game.tanks
    first.x, first.y, first.heading = 2.0, 3.0, 0.0
    second.x, second.y = 4.0, 3.0
    _throttle, steer, fire = script_action(env.game, first.tank_id, "hunter", np.random.default_rng(0))
    assert steer == 1
    assert fire == 1


def test_hunter_has_no_aiming_deadzone_wider_than_its_fire_window():
    from rl.opponents import script_action

    env = TankSelfPlayEnv(layout="open", spawn="close_facing", rows=6, cols=6)
    env.reset(seed=11)
    first, second = env.game.tanks
    first.x, first.y, first.heading = 1.0, 3.0, 0.08
    second.x, second.y = 5.0, 3.0
    _throttle, steer, fire = script_action(
        env.game, first.tank_id, "hunter", np.random.default_rng(0)
    )
    assert steer == 0
    assert fire == 1


def test_opponent_pool_samples_listed_scripts():
    from rl.opponents import build_opponent_controller

    opponent = build_opponent_controller(["idle", "move"], None, torch.device("cpu"), seed=0)
    assert opponent is not None
    names = {opponent.reset_env(0) for _ in range(40)}
    assert names <= {"idle", "move"}
    assert len(names) == 2


def test_opponent_pool_respects_explicit_weights():
    from rl.opponents import build_opponent_controller

    opponent = build_opponent_controller(
        ["idle", "move"], None, torch.device("cpu"), seed=0, weights=[1.0, 0.0]
    )
    assert opponent is not None
    assert {opponent.reset_env(0) for _ in range(20)} == {"idle"}
    assert opponent.current_label(0) == "idle"


def test_idle_script_never_moves_or_fires():
    from rl.opponents import script_action

    env = TankSelfPlayEnv(layout="open", spawn="close_facing", rows=6, cols=6)
    env.reset(seed=11)
    assert script_action(env.game, env.game.tanks[0].tank_id, "idle", np.random.default_rng(1)) == (1, 1, 0)


def test_move_script_never_fires():
    from rl.opponents import script_action

    env = TankSelfPlayEnv(layout="open", spawn="close_facing", rows=6, cols=6)
    env.reset(seed=11)
    rng = np.random.default_rng(2)
    for _ in range(20):
        _throttle, _steer, fire = script_action(env.game, env.game.tanks[0].tank_id, "move", rng)
        assert fire == 0


def test_shortest_path_distance_is_symmetric():
    env = TankSelfPlayEnv(rows=6, cols=6)
    env.reset(seed=6)
    first = tank_cell(env.game.tanks[0], env.game.maze)
    second = tank_cell(env.game.tanks[1], env.game.maze)
    assert astar_distance(env.game.maze, first, second) == astar_distance(env.game.maze, second, first)


def test_hunter_teacher_fires_when_enemy_is_directly_ahead():
    game = TankGame(rows=6, cols=6)
    make_open_arena(game)
    own, enemy = game.tanks
    own.x, own.y, own.heading = 1.0, 3.0, 0.0
    enemy.x, enemy.y = 5.0, 3.0
    _throttle, steer, fire = HunterTeacher().action(game, own.tank_id)
    assert steer == 1
    assert fire == 1


def test_hunter_teacher_does_not_fire_through_blocking_wall():
    game = TankGame(rows=6, cols=6)
    make_open_arena(game)
    game.maze.vertical[3, 3] = True
    game.wall_rects = game.maze.wall_rects(game.wall_thickness)
    own, enemy = game.tanks
    own.x, own.y, own.heading = 1.5, 3.5, 0.0
    enemy.x, enemy.y = 4.5, 3.5
    _, _, fire = HunterTeacher().action(game, own.tank_id)
    assert fire == 0


def test_hunter_teacher_can_aim_and_fire_a_verified_ricochet():
    from rl.opponents import _has_los, _ricochet_heading

    game = TankGame(rows=6, cols=6)
    make_open_arena(game)
    # 近处横墙挡住直射；右边界仍提供一条可验证的一次反弹弹道。
    game.maze.horizontal[2, 0] = True
    game.wall_rects = game.maze.wall_rects(game.wall_thickness)
    own, enemy = game.tanks
    own.x, own.y = 1.0, 3.0
    enemy.x, enemy.y = 1.0, 1.0
    assert not _has_los(game, own, enemy)
    heading = _ricochet_heading(game, own, enemy)
    assert heading is not None
    own.heading = heading
    throttle, steer, fire = HunterTeacher().action(game, own.tank_id)
    assert (throttle, steer, fire) == (1, 1, 1)


def test_hunter_dodges_a_future_direct_bullet_path():
    game = TankGame(rows=6, cols=6)
    make_open_arena(game)
    own, enemy = game.tanks
    own.x, own.y, own.heading = 3.0, 3.0, 0.0
    enemy.x, enemy.y = 5.0, 5.0
    game.bullets.append(Bullet(1.0, 3.0, game.bullet_speed, 0.0, enemy.tank_id))
    throttle, steer, _fire = HunterTeacher().action(game, own.tank_id)
    assert (throttle, steer) != (1, 1)


def test_trajectory_predicts_direct_enemy_hit():
    game = TankGame(rows=6, cols=6)
    make_open_arena(game)
    own, enemy = game.tanks
    own.x, own.y, own.heading = 1.0, 3.0, 0.0
    enemy.x, enemy.y = 5.0, 3.0
    bullet = Bullet(1.35, 3.0, game.bullet_speed, 0.0, owner_tank_id=own.tank_id)
    result = trace_bullet_trajectory(game, bullet, own.tank_id)
    assert result.predicted_hit == enemy.tank_id
    assert result.enemy_distance == 0.0


def test_trajectory_predicts_ricochet_self_hit():
    game = TankGame(rows=6, cols=6)
    make_open_arena(game)
    own, enemy = game.tanks
    own.x, own.y, own.heading = 1.0, 3.0, math.pi
    enemy.x, enemy.y = 5.0, 1.0
    bullet = Bullet(0.65, 3.0, -game.bullet_speed, 0.0, owner_tank_id=own.tank_id)
    result = trace_bullet_trajectory(game, bullet, own.tank_id)
    assert result.predicted_hit == own.tank_id
    assert result.self_distance == 0.0


def test_proximity_score_is_larger_for_closer_trajectory():
    assert proximity_score(0.1) > proximity_score(1.0)


def test_environment_rejects_wrong_action_shape():
    env = TankSelfPlayEnv()
    with np.testing.assert_raises(ValueError):
        env.step([(1, 1, 0)])


def test_model_outputs_three_valid_actions_and_values():
    model = TankActorCritic()
    game = TankGame(rows=6, cols=12)
    batch = stack_observations([build_observation(game, 0), build_observation(game, 1)])
    actions, log_probability, entropy, value = model.get_action_and_value(*_model_batch(batch, torch.device("cpu")))
    assert actions.shape == (2, 3)
    assert log_probability.shape == (2,)
    assert entropy.shape == (2,)
    assert value.shape == (2,)
    assert torch.all((0 <= actions[:, 0]) & (actions[:, 0] <= 2))
    assert torch.all((0 <= actions[:, 1]) & (actions[:, 1] <= 2))
    assert torch.all((0 <= actions[:, 2]) & (actions[:, 2] <= 1))


def test_model_stays_near_the_intended_small_size():
    parameters = sum(parameter.numel() for parameter in TankActorCritic().parameters())
    assert 100_000 <= parameters <= 250_000


def test_joint_action_encoding_round_trip():
    actions = torch.tensor([[0, 0, 0], [1, 2, 1], [2, 2, 1]])
    assert torch.equal(decode_actions(encode_actions(actions)), actions)


def test_old_model_checkpoint_is_rejected():
    source = TankActorCritic()
    old_state = dict(source.state_dict())
    old_state.pop("policy_head.weight")
    restored = TankActorCritic()
    with np.testing.assert_raises(RuntimeError):
        load_actor_critic_state(restored, old_state)


def test_offline_dataset_keeps_complete_map_seeds_in_disjoint_splits(tmp_path):
    dataset_dir = tmp_path / "dataset"
    collect_supervised_dataset(
        Namespace(
            output=dataset_dir,
            train_seeds=2,
            validation_seeds=1,
            seed_start=700,
            episodes_per_shard=1,
            action_repeat=1,
            rows=6,
            cols=6,
            layout="maze",
            spawn="default",
            time_limit=1.0 / 24.0,
        )
    )
    manifest = load_manifest(dataset_dir)
    assert manifest["splits"]["train"]["seeds"] == [700, 701]
    assert manifest["splits"]["validation"]["seeds"] == [702]
    train_seen = set()
    for path in split_shards(dataset_dir, manifest, "train"):
        shard = load_shard(path)
        assert set(shard["map_seeds"]) == set(shard["episode_seeds"])
        train_seen.update(map(int, shard["map_seeds"]))
    validation_shard = load_shard(split_shards(dataset_dir, manifest, "validation")[0])
    validation_seen = set(map(int, validation_shard["map_seeds"]))
    assert train_seen == {700, 701}
    assert validation_seen == {702}
    assert train_seen.isdisjoint(validation_seen)


def test_offline_dataset_trains_and_validates_without_collecting_new_states(tmp_path):
    dataset_dir = tmp_path / "dataset"
    collect_supervised_dataset(
        Namespace(
            output=dataset_dir,
            train_seeds=2,
            validation_seeds=1,
            seed_start=800,
            episodes_per_shard=2,
            action_repeat=1,
            rows=6,
            cols=6,
            layout="maze",
            spawn="default",
            time_limit=1.0 / 24.0,
        )
    )
    output = tmp_path / "checkpoint"
    checkpoint = train_offline_bc(
        Namespace(
            dataset=dataset_dir,
            output=output,
            epochs=1,
            minibatch_size=4,
            learning_rate=3e-4,
            fire_weight=6.0,
            max_grad_norm=0.5,
            seed=1,
            device="cpu",
            resume=None,
            save_every=1,
        )
    )
    assert checkpoint.is_file()
    assert "validation_loss" in (output / "offline_log.csv").read_text(encoding="utf-8")
