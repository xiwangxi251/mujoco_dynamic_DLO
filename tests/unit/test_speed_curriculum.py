from dataclasses import asdict
import json
import numpy as np
import pytest

from panda_cable_grasp.rl.speed_curriculum import (SpeedState, SpeedEnvironment,
    verify_workers, PRETRAINING_STEPS, schedule_value)


def test_gate_minimum_exact_windows_and_restore():
    s = SpeedState()
    for _ in range(300):
        assert s.add(True, 250000, 199999) is None
    assert not s.window and not s.pending
    for i in range(100):
        decision = s.add(i < 80, 260000, 200000)
    assert decision['passed'] and s.consecutive == 1 and not s.pending
    saved = json.loads(json.dumps(s.record()))
    s = SpeedState(saved)
    for i in range(100):
        decision = s.add(i < 79, 270000, 210000)
    assert not decision['passed'] and s.consecutive == 0
    for _ in range(200):
        s.add(True, 280000, 210000)
    assert s.pending
    s.advance(294912)
    assert s.scale == 1.25 and s.generation == 1 and s.level_start_added == 294912
    assert not s.window and not s.pending and not s.consecutive
    assert s.add(True, 600000, 494911) is None


def test_all_levels_final_gate_and_added_schedule():
    s = SpeedState()
    for level in (1, 1.25, 1.5, 1.75, 2):
        assert s.scale == level and s.mode == 'speed'
        for _ in range(200):
            s.add(True, s.level_start_added+210000, s.level_start_added+200000)
        s.advance(s.level_start_added+230000)
    assert s.mode == 'scenes' and s.scene_phase == 1 and s.scale == 2
    assert schedule_value(7e-5, 3e-5, 0) == 7e-5
    assert schedule_value(7e-5, 3e-5, 20000000) == pytest.approx(3e-5)
    assert PRETRAINING_STEPS+20000000 == 36997280


def test_every_live_stage_survives_reset_without_physics_change():
    env = SpeedEnvironment(robot='nero', controller_profile='enhanced6_bounded_dls_v1',
        table_execution_guard_enabled=True, geometric_safety_enabled=False,
        table_finger_collision_filter_enabled=False,
        scenario_names=('dev_static_x075_v1',))
    try:
        base_hashes = env.execution_configuration()['compiled_hashes']
        original_joints = np.asarray(env.base_env.config.arm_joint_velocity_limits)
        for i in range(5):
            s = SpeedState(dict(version='warm_speed_curriculum_v1', level_index=i, generation=i))
            env.configure_speed_stage(s.record())
            env.reset(seed=20260804)
            a = env.execution_configuration()
            verify_workers([a], s.record(), base_hashes)
            np.testing.assert_allclose(a['effective_arm_joint_velocity_limits'], original_joints*s.scale)
            _, _, _, _, info = env.step(np.zeros(5))
            assert info['speed_generation'] == i
            assert all(np.isfinite(info['actual_'+k+'_speed']) for k in ('tcp_linear', 'tcp_angular', 'joint'))
    finally:
        env.close()


def test_old_buffer_is_optimized_before_stage_reset(tmp_path):
    import torch
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    from panda_cable_grasp.rl.speed_curriculum import SpeedPPO, SpeedCallback
    env = VecMonitor(DummyVecEnv([lambda: SpeedEnvironment(robot='nero',
        controller_profile='enhanced6_bounded_dls_v1', table_execution_guard_enabled=True,
        geometric_safety_enabled=False, scenario_names=('dev_static_x075_v1',))]))
    try:
        state = SpeedState()
        env.env_method('configure_speed_stage', state.record())
        model = SpeedPPO('MlpPolicy', env, n_steps=4, batch_size=4, n_epochs=1,
                         policy_kwargs=dict(net_arch=[8]), device='cpu')
        model.num_timesteps = PRETRAINING_STEPS+300000
        cfg = dict(rl_config=env.env_method('execution_configuration')[0]['rl_config'],
                   invocation_target_added=20000000)
        cb = SpeedCallback(tmp_path, state.record(), cfg)
        model._setup_learn(4, reset_num_timesteps=False)
        cb.init_callback(model)
        cb.on_training_start({}, {})
        cb.state.pending = True
        cb.last_checkpoint = 300000
        calls = []
        original_train = __import__('stable_baselines3').PPO.train
        # Call after_update with evidence that train was completed, rather than
        # reimplementing optimization in this boundary test.
        original_method = model._speed_callback.after_update
        def observed():
            calls.append(model._n_updates)
            original_method()
        model._speed_callback.after_update = observed
        from stable_baselines3.common.callbacks import CallbackList
        assert model.collect_rollouts(env, cb, model.rollout_buffer, 4)
        model.train()
        assert calls == [1] and cb.state.scale == 1.25
        assert np.all(model._last_episode_starts)
        assert model._last_obs.shape == (1, 99)
        after = tmp_path/'checkpoints'/f'c_{cb.added}_after_transition/model.zip'
        reloaded = SpeedPPO.load(after, device='cpu')
        assert reloaded.c_speed_state['level_index'] == 1
        assert reloaded._n_updates == 1 and reloaded.c_added_steps == cb.added
        cb.on_training_end()
    finally:
        env.close()
