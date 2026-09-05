import numpy as np

from rl.curriculum import STAGES, CurriculumManager
from rl.envs import TankRLEnv
from rl.observation import build_observation


def test_every_stage_can_reset_and_step():
    # 验证全部课程阶段都能生成固定形状观察并推进物理。
    for stage in STAGES:
        env = TankRLEnv(action_repeat=2)
        observation = env.reset(stage, seed=100 + stage.index)
        next_observation, _, _, info = env.step((1, 1, 0))
        assert observation["map"].shape == (2, 96, 96)
        assert next_observation["tanks"].shape == (3, 4)
        assert info["stage"] == stage.index


def test_stage_zero_shot_produces_kill_and_win_rewards():
    # 验证基础靶场主动击杀会先奖励事件并在宽限期后奖励胜利。
    env = TankRLEnv(action_repeat=2)
    env.reset(STAGES[0], seed=123)
    own, enemy = env.game.tanks[:2]
    own.x, own.y, own.heading = 1.5, 3.0, 0.0
    enemy.x, enemy.y, enemy.heading = 4.5, 3.0, np.pi
    env.reward_tracker.reset(env.game)
    _, total_reward, done, info = env.step((1, 1, 1))
    saw_kill = False
    for _ in range(1000):
        if done:
            break
        _, reward, done, info = env.step((1, 1, 0))
        total_reward += reward
        saw_kill = saw_kill or any(event["victim"] == 1 for event in info["events"])
    assert done
    assert saw_kill
    assert info["result"] == "win"
    assert total_reward > 1.0


def test_stage_zero_spawns_tanks_far_apart_with_random_headings():
    # 验证第零关不再把双方固定成近距离面对面出生。
    headings = []
    for seed in range(8):
        env = TankRLEnv()
        env.reset(STAGES[0], seed=seed)
        own, enemy = env.game.tanks[:2]
        assert np.hypot(enemy.x - own.x, enemy.y - own.y) >= 5.0
        headings.append((own.heading, enemy.heading))
    assert len(set(headings)) > 1


def test_successful_fire_gets_small_capped_reward():
    # 验证实际开炮获得小额奖励且同一局累计奖励不会超过上限。
    firing_env = TankRLEnv(action_repeat=1)
    idle_env = TankRLEnv(action_repeat=1)
    firing_env.reset(STAGES[0], seed=321)
    idle_env.reset(STAGES[0], seed=321)
    _, firing_reward, _, firing_info = firing_env.step((1, 1, 1))
    _, idle_reward, _, _ = idle_env.step((1, 1, 0))
    assert firing_info["shots_fired"] == 1
    assert np.isclose(firing_reward - idle_reward, STAGES[0].reward.shot_reward)

    capped_env = TankRLEnv(action_repeat=1)
    capped_env.reset(STAGES[0], seed=654)
    own = capped_env.game.tanks[0]
    for _ in range(capped_env.game.max_bullets_per_tank):
        capped_env.game._fire(own)
    capped_env.reward_tracker.calculate(capped_env.game, [])
    assert np.isclose(
        capped_env.reward_tracker.shot_reward_paid,
        STAGES[0].reward.max_shot_reward,
    )


def test_opponent_self_kill_is_not_counted_as_player_win():
    # 验证诱导敌方自毁不会获得胜利奖励、不会额外扣分并会被单独标记。
    env = TankRLEnv(action_repeat=1)
    env.reset(STAGES[0], seed=987)
    env.game.tanks[1].alive = False
    env.game.is_over = True
    env.game.winner = env.controlled_tank_id
    reward = env.reward_tracker.calculate(
        env.game, [{"shooter": 1, "victim": 1, "bullet_age": 0.5}]
    )
    assert env._result() == "opponent_self_kill"
    assert np.isclose(reward, STAGES[0].reward.opponent_self_kill_reward)


def test_bullet_hidden_truth_does_not_change_observation():
    # 验证发射者、年龄和反弹次数不会泄漏进模型观察。
    env = TankRLEnv()
    env.reset(STAGES[0], seed=9)
    env.game._fire(env.game.tanks[1])
    bullet = env.game.bullets[0]
    visible_before = np.array([bullet.x, bullet.y, bullet.vx, bullet.vy])
    observation_before = build_observation(env.game, 0)
    bullet.owner_tank_id = 0
    bullet.age = 9.0
    bullet.bounces = 99
    observation_after = build_observation(env.game, 0)
    assert np.array_equal(visible_before, np.array([bullet.x, bullet.y, bullet.vx, bullet.vy]))
    assert np.array_equal(observation_before["bullets"], observation_after["bullets"])


def test_curriculum_promotes_after_passing_evaluation():
    # 验证达到固定评估门槛后课程只晋升一个阶段。
    curriculum = CurriculumManager(start_stage=0, seed=1)
    curriculum.replace_evaluation(0, ["win"] * STAGES[0].evaluation_games)
    assert curriculum.try_promote()
    assert curriculum.current_stage == 1


def test_fixed_stage_disables_mixing_and_promotion():
    # 验证锁定关卡后所有任务都来自指定阶段且不会自动晋级。
    curriculum = CurriculumManager(start_stage=3, seed=1, fixed_stage=True)
    assert all(curriculum.sample_stage().index == 3 for _ in range(50))
    curriculum.replace_evaluation(3, ["win"] * STAGES[3].evaluation_games)
    assert not curriculum.try_promote()
    assert curriculum.current_stage == 3
