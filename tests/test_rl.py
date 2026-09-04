import math

import numpy as np
import pytest
import torch

from core import TankGame
from core.entities import Bullet
from rl_stage3.environment import RewardConfig, TankSelfPlayEnv
from rl_stage3.model import TankActorCritic
from rl_stage3.observation import BULLET_FEATURES, MAP_CHANNELS, MAP_SIZE, MAX_BULLETS, MAX_OTHER_TANKS, SELF_FEATURES, TANK_FEATURES, build_observation
from rl_stage3.planning import astar_distance, tank_cell
from rl_stage3.trajectory import proximity_score, trace_bullet_trajectory

try:
    from supervised.teachers import AStarDrivingTeacher, WeakCombatTeacher
except ImportError:
    AStarDrivingTeacher = None
    WeakCombatTeacher = None


def make_open_arena(game: TankGame, rows: int = 6, cols: int = 6) -> None:
    from rl_stage3.environment import make_open_maze

    game.maze = make_open_maze(rows, cols)
    game.wall_rects = game.maze.wall_rects(game.wall_thickness)


def test_observation_has_fixed_shapes_for_different_maps():
    for rows, cols in ((6, 12), (12, 6), (9, 9)):
        game = TankGame(rows=rows, cols=cols)
        observation = build_observation(game, tank_id=0)
        assert observation["map"].shape == (MAP_CHANNELS, MAP_SIZE, MAP_SIZE)
        assert observation["self"].shape == (SELF_FEATURES,)
        assert observation["tanks"].shape == (MAX_OTHER_TANKS, TANK_FEATURES)
        assert observation["bullets"].shape == (MAX_BULLETS, BULLET_FEATURES)
        assert observation["tank_mask"].shape == (MAX_OTHER_TANKS,)
        assert observation["bullet_mask"].shape == (MAX_BULLETS,)


def test_observation_map_is_single_local_wall_channel():
    game = TankGame(rows=6, cols=6)
    observation = build_observation(game, tank_id=0)
    assert MAP_CHANNELS == 1
    assert observation["map"].shape == (1, MAP_SIZE, MAP_SIZE)
    assert observation["map"].max() <= 1.0


def test_bullet_set_marks_enemy_shot_and_ignores_empty_slots():
    game = TankGame(rows=6, cols=6)
    game._fire(game.tanks[1])
    observation = build_observation(game, tank_id=0)
    assert observation["bullet_mask"].sum() == 1
    assert observation["bullets"][0, 6] == -1.0
    assert observation["bullets"][1:].sum() == 0


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


def test_survivor_wins_even_when_enemy_destroys_itself():
    env = TankSelfPlayEnv(action_repeat=1)
    env.reset(seed=4)
    env.game.tanks[0].alive = False
    _, rewards, done, info = env.step([(1, 1, 0), (1, 1, 0)])
    assert done is True
    assert info["winner"] == 1
    assert np.isclose(rewards[0], RewardConfig().loss)
    assert rewards[1] >= RewardConfig().win - 1e-5


def test_own_bullet_kill_uses_small_self_kill_penalty():
    env = TankSelfPlayEnv(action_repeat=1)
    env.reset(seed=8)
    victim = env.game.tanks[0]
    env.game.bullets.append(Bullet(victim.x, victim.y, 0.0, 0.0, owner_tank_id=victim.tank_id, age=1.0))
    _, rewards, done, info = env.step([(1, 1, 0), (1, 1, 0)])
    assert done is True
    assert info["winner"] == 1
    assert np.isclose(rewards[0], RewardConfig().self_kill)
    assert rewards[1] >= RewardConfig().win - 1e-5


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
    from rl_stage1.opponents import script_action

    env = TankSelfPlayEnv(layout="open", spawn="close_facing", rows=6, cols=6)
    env.reset(seed=11)
    first, second = env.game.tanks
    first.x, first.y, first.heading = 2.0, 3.0, 0.0
    second.x, second.y = 4.0, 3.0
    throttle, steer, fire = script_action(env.game, first.tank_id, "aim", np.random.default_rng(0))
    assert throttle == 1
    assert steer == 1
    assert fire == 1


def test_idle_script_never_moves_or_fires():
    from rl_stage1.opponents import script_action

    env = TankSelfPlayEnv(layout="open", spawn="close_facing", rows=6, cols=6)
    env.reset(seed=11)
    assert script_action(env.game, env.game.tanks[0].tank_id, "idle", np.random.default_rng(1)) == (1, 1, 0)


def test_move_script_never_fires():
    from rl_stage1.opponents import script_action

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


@pytest.mark.skipif(AStarDrivingTeacher is None, reason="supervised teachers are not in this workspace")
def test_astar_teacher_moves_forward_when_aligned():
    game = TankGame(rows=6, cols=6)
    make_open_arena(game)
    own, enemy = game.tanks
    own.x, own.y, own.heading = 1.5, 3.5, 0.0
    enemy.x, enemy.y = 4.5, 3.5
    assert AStarDrivingTeacher().action(game, own.tank_id) == (2, 1, 0)


@pytest.mark.skipif(AStarDrivingTeacher is None, reason="supervised teachers are not in this workspace")
def test_astar_teacher_turns_in_place_instead_of_reversing():
    game = TankGame(rows=6, cols=6)
    make_open_arena(game)
    own, enemy = game.tanks
    own.x, own.y, own.heading = 1.5, 3.5, math.pi
    enemy.x, enemy.y = 4.5, 3.5
    throttle, steer, fire = AStarDrivingTeacher().action(game, own.tank_id)
    assert throttle == 1
    assert steer in (0, 2)
    assert fire == 0


@pytest.mark.skipif(WeakCombatTeacher is None, reason="supervised teachers are not in this workspace")
def test_weak_combat_teacher_fires_when_enemy_is_directly_ahead():
    game = TankGame(rows=6, cols=6)
    make_open_arena(game)
    own, enemy = game.tanks
    own.x, own.y, own.heading = 1.0, 3.0, 0.0
    enemy.x, enemy.y = 5.0, 3.0
    assert WeakCombatTeacher().action(game, own.tank_id) == (1, 1, 1)


@pytest.mark.skipif(WeakCombatTeacher is None, reason="supervised teachers are not in this workspace")
def test_weak_combat_teacher_does_not_fire_through_blocking_wall():
    game = TankGame(rows=6, cols=6)
    make_open_arena(game)
    game.maze.vertical[3, 3] = True
    game.wall_rects = game.maze.wall_rects(game.wall_thickness)
    own, enemy = game.tanks
    own.x, own.y, own.heading = 1.5, 3.5, 0.0
    enemy.x, enemy.y = 4.5, 3.5
    _, _, fire = WeakCombatTeacher().action(game, own.tank_id)
    assert fire == 0


@pytest.mark.skipif(WeakCombatTeacher is None, reason="supervised teachers are not in this workspace")
def test_weak_combat_teacher_stops_and_turns_when_line_of_sight_is_clear():
    game = TankGame(rows=6, cols=6)
    make_open_arena(game)
    own, enemy = game.tanks
    own.x, own.y, own.heading = 1.0, 3.0, math.pi / 2
    enemy.x, enemy.y = 5.0, 3.0
    throttle, steer, fire = WeakCombatTeacher().action(game, own.tank_id)
    assert throttle == 1
    assert steer in (0, 2)
    assert fire == 0


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
    maps = torch.zeros((2, MAP_CHANNELS, MAP_SIZE, MAP_SIZE))
    selves = torch.zeros((2, SELF_FEATURES))
    tanks = torch.zeros((2, MAX_OTHER_TANKS, TANK_FEATURES))
    tank_mask = torch.zeros((2, MAX_OTHER_TANKS))
    bullets = torch.zeros((2, MAX_BULLETS, BULLET_FEATURES))
    bullet_mask = torch.zeros((2, MAX_BULLETS))
    actions, log_probability, entropy, value = model.get_action_and_value(
        maps, selves, tanks, tank_mask, bullets, bullet_mask
    )
    assert actions.shape == (2, 3)
    assert log_probability.shape == (2,)
    assert entropy.shape == (2,)
    assert value.shape == (2,)
    assert torch.all((0 <= actions[:, 0]) & (actions[:, 0] <= 2))
    assert torch.all((0 <= actions[:, 1]) & (actions[:, 1] <= 2))
    assert torch.all((0 <= actions[:, 2]) & (actions[:, 2] <= 1))
