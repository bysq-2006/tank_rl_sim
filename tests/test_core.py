import math

import numpy as np
import pytest

from core import TankGame
from core.entities import Bullet


def maze_signature(game: TankGame) -> tuple[bytes, bytes]:
    """把迷宫墙壁转为便于测试比较的字节数据。"""
    return game.maze.horizontal.tobytes(), game.maze.vertical.tobytes()


def test_seed_reproduces_the_same_game():
    first = TankGame()
    first.reset(seed=42)
    second = TankGame()
    second.reset(seed=42)
    assert (first.maze.rows, first.maze.cols) == (second.maze.rows, second.maze.cols)
    assert maze_signature(first) == maze_signature(second)
    assert first.tanks == second.tanks


def test_reset_without_seed_advances_random_generator():
    game = TankGame()
    game.reset(seed=12)
    first = (game.maze.rows, game.maze.cols, maze_signature(game))
    game.reset()
    second = (game.maze.rows, game.maze.cols, maze_signature(game))
    assert first != second


def test_update_advances_exactly_one_physics_frame():
    game = TankGame(rows=9, cols=12)
    game.reset(seed=1)
    game.update([(2, 2, 1), (1, 1, 0)])
    assert game.elapsed == pytest.approx(1 / 24)
    assert len(game.bullets) == 1


def test_generated_maze_outer_boundary_is_closed():
    game = TankGame()
    game.reset(seed=9)
    assert game.maze.horizontal[0].all()
    assert game.maze.horizontal[-1].all()
    assert game.maze.vertical[:, 0].all()
    assert game.maze.vertical[:, -1].all()


def test_invalid_control_is_rejected():
    game = TankGame()
    with pytest.raises(ValueError):
        game.update([(3, 1, 0), (1, 1, 0)])


def test_control_count_must_match_tank_count():
    game = TankGame()
    with pytest.raises(ValueError):
        game.update([(1, 1, 0)])


def test_ricochet_can_hit_the_shooter():
    game = TankGame(rows=6, cols=6)
    game.reset(seed=3)
    tank = game.tanks[0]
    returned_bullet = Bullet(tank.x, tank.y, 0.0, 0.0, owner_tank_id=tank.tank_id, age=1.0, bounces=1)
    assert game._bullet_hit(returned_bullet) is tank


def test_tank_uses_rotating_rectangle_corners():
    game = TankGame(rows=6, cols=6)
    corners = game._tank_corners(2.0, 2.0, math.pi / 4)
    distances = [math.hypot(x - 2.0, y - 2.0) for x, y in corners]
    corner_distance = math.hypot(game.tank_half_length, game.tank_half_width)
    assert distances == pytest.approx([corner_distance] * 4)
    assert game.tank_half_length > game.tank_half_width


def test_measured_object_dimensions_are_used():
    game = TankGame(rows=6, cols=6)
    assert game.tank_half_length * 2 == pytest.approx(0.4559)
    assert game.tank_half_width * 2 == pytest.approx(0.3389)
    assert game.barrel_length == pytest.approx(0.26795)
    assert game.barrel_width == pytest.approx(0.091)
    assert game.bullet_radius * 2 == pytest.approx(0.09)


def test_fire_interval_matches_video_measurement():
    game = TankGame(rows=6, cols=6)
    assert game.fire_cooldown == pytest.approx(2.967 - 2.733)
    assert 1.0 / game.fire_cooldown == pytest.approx(4.2735042735)


def test_fractional_fire_interval_is_carried_between_frames():
    game = TankGame(rows=6, cols=6)
    tank = game.tanks[0]
    shot_frames: list[int] = []
    for frame in range(80):
        tank.cooldown -= game.dt
        if tank.cooldown <= 0.0:
            game._fire(tank)
            shot_frames.append(frame)
            game.bullets.clear()
    frame_intervals = [second - first for first, second in zip(shot_frames, shot_frames[1:])]
    assert set(frame_intervals) == {5, 6}
    average_seconds = sum(frame_intervals) / len(frame_intervals) * game.dt
    assert average_seconds == pytest.approx(game.fire_cooldown, abs=game.dt / len(frame_intervals))


def test_requested_lifetime_and_uniform_wall_width():
    game = TankGame(rows=6, cols=6)
    assert game.bullet_lifetime == 10.0
    assert game.wall_thickness == pytest.approx(0.0735)


def test_barrel_cannot_enter_a_wall():
    game = TankGame(rows=6, cols=6)
    game.reset(seed=1)
    tank = game.tanks[0]
    tank.x = game.wall_thickness / 2 + 0.01
    tank.y = 1.5
    tank.heading = math.pi
    assert game._tank_hits_wall(tank.x, tank.y, tank.heading)
    assert game._push_tank_out_of_walls(tank)
    assert not game._tank_hits_wall(tank.x, tank.y, tank.heading)
    assert tank.x >= game.barrel_length + game.wall_thickness / 2


def test_firing_into_a_nearby_wall_kills_the_shooter():
    game = TankGame(rows=6, cols=6)
    game.reset(seed=1)
    tank = game.tanks[0]
    tank.x = game.barrel_length + game.wall_thickness / 2 + 0.01
    tank.y = 1.5
    tank.heading = math.pi
    tank.cooldown = 0.0
    assert not game._tank_hits_wall(tank.x, tank.y, tank.heading)
    events = []
    for _ in range(8):
        events.extend(game.update([(1, 1, 1), (1, 1, 0)]))
        if not tank.alive:
            break
    assert not tank.alive
    assert any(event["shooter"] == tank.tank_id and event["victim"] == tank.tank_id for event in events)


def test_fresh_shot_does_not_destroy_the_shooter():
    game = TankGame(rows=6, cols=6)
    game.reset(seed=2)
    tank = game.tanks[0]
    tank.cooldown = 0.0
    game._fire(tank)
    game.update([(1, 1, 0), (1, 1, 0)])
    assert tank.alive
    assert any(bullet.owner_tank_id == tank.tank_id for bullet in game.bullets)


def test_each_tank_can_have_at_most_five_active_bullets():
    game = TankGame(rows=6, cols=6)
    game.reset(seed=4)
    tank = game.tanks[0]
    for _ in range(8):
        game._fire(tank)
        tank.cooldown = 0.0
    own_bullets = [bullet for bullet in game.bullets if bullet.owner_tank_id == tank.tank_id]
    assert len(own_bullets) == 5
    assert game.max_bullets_per_tank == 5


def test_bullet_speed_matches_video_measurement():
    game = TankGame(rows=6, cols=6)
    assert game.bullet_speed == pytest.approx(2.2738)
    assert game.bullet_speed * game.dt == pytest.approx(2.2738 / 24)


def test_tank_speed_matches_video_measurement():
    game = TankGame(rows=6, cols=6)
    assert game.max_speed == pytest.approx(1.8622)
    assert game.max_speed * game.dt == pytest.approx(1.8622 / 24)


def test_wall_rectangles_extend_into_junctions():
    game = TankGame(rows=6, cols=6)
    game.reset(seed=5)
    half = game.wall_thickness / 2
    extension = game.wall_thickness / 4
    # 最上方第一段水平墙的总增长量应为半个墙宽。
    top_wall = game.maze.wall_rects(game.wall_thickness)[0]
    assert top_wall == pytest.approx((-extension, -half, 1.0 + extension, half))


def test_tank_rotation_pushes_it_away_from_wall():
    game = TankGame(rows=6, cols=6)
    game.reset(seed=8)
    tank = game.tanks[0]
    tank.x = game.wall_thickness / 2 + game.tank_half_width
    tank.y = 1.5
    tank.heading = math.pi / 2
    old_x, old_heading = tank.x, tank.heading

    game.update([(1, 0, 0), (1, 1, 0)])

    assert tank.heading != pytest.approx(old_heading)
    assert tank.x > old_x
    assert not game._tank_hits_wall(tank.x, tank.y, tank.heading)


def test_one_bullet_hit_immediately_destroys_tank():
    game = TankGame(rows=6, cols=6)
    game.reset(seed=6)
    target = game.tanks[1]
    game.bullets.append(
        Bullet(target.x, target.y, 0.0, 0.0, owner_tank_id=game.tanks[0].tank_id, age=1.0)
    )

    events = game.update([(1, 1, 0), (1, 1, 0)])

    assert not target.alive
    assert not game.is_over
    assert game.winner is None
    assert events[0]["shooter"] == game.tanks[0].tank_id
    assert events[0]["victim"] == target.tank_id
    assert events[0]["bullet_age"] >= 1.0

    while game.elapsed - game.first_death_at < game.death_grace:
        game.update([(1, 1, 0), (1, 1, 0)])

    assert game.is_over
    assert game.winner == game.tanks[0].tank_id


def test_remaining_tank_can_die_during_death_grace():
    game = TankGame(rows=6, cols=6)
    game.reset(seed=6)
    first, second = game.tanks
    game.bullets.append(Bullet(second.x, second.y, 0.0, 0.0, owner_tank_id=first.tank_id, age=1.0))
    game.update([(1, 1, 0), (1, 1, 0)])
    assert not second.alive
    assert first.alive
    assert not game.is_over

    game.bullets.append(Bullet(first.x, first.y, 0.0, 0.0, owner_tank_id=second.tank_id, age=1.0))
    events = game.update([(1, 1, 0), (1, 1, 0)])
    assert not first.alive
    assert events[0]["victim"] == first.tank_id
    assert not game.is_over

    while game.elapsed - game.first_death_at < game.death_grace:
        game.update([(1, 1, 0), (1, 1, 0)])

    assert game.is_over
    assert game.winner is None


def test_many_random_updates_remain_finite():
    game = TankGame(time_limit=20)
    game.reset(seed=10)
    rng = np.random.default_rng(10)
    for _ in range(200):
        control = (int(rng.integers(3)), int(rng.integers(3)), int(rng.integers(2)))
        game.update([control, (1, 1, 0)])
        values = [value for tank in game.tanks for value in (tank.x, tank.y, tank.heading, tank.speed, tank.angular_velocity)]
        values.extend(value for bullet in game.bullets for value in (bullet.x, bullet.y, bullet.vx, bullet.vy))
        assert all(math.isfinite(value) for value in values)
        if game.is_over:
            break
