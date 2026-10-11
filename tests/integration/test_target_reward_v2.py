import numpy as np
import pytest

from panda_cable_grasp.env.environment import GraspState
from panda_cable_grasp.rl.environment import RLConfig, make_rl_env
from panda_cable_grasp.rl.lift_credit import HighWaterLiftCredit


def test_five_requested_sequences():
    credit = HighWaterLiftCredit([.02, .02], 0.)
    assert credit.update([.06, .02], .1, 0, 1.) == pytest.approx((.04, .1))
    assert credit.update([.10, .02], .2, 0, 1.) == pytest.approx((.04, .1))
    credit.update([.02, .02], 0., None, 0.)
    assert credit.update([.10, .02], .2, 0, 1.) == pytest.approx((0., 0.))
    assert credit.update([.12, .02], .25, 0, 1.) == pytest.approx((.02, .05))
    # Unheld external lift consumes the observed ceiling but emits no reward.
    assert credit.update([.16, .16], .3, None, 0.) == (0., 0.)
    assert credit.update([.16, .16], .3, 1, 1.) == (0., 0.)
    # Switch to a node which rises only to the already credited height.
    assert credit.update([.16, .12], .25, 0, 1.)[0] == 0.
    assert credit.update([.16, .16], .3, 1, 1.)[0] == 0.


def test_smaller_node_error_cannot_reclaim_discounted_progress():
    credit = HighWaterLiftCredit([.02], 0.)
    assert credit.update([.10], .2, 0, .25) == pytest.approx((.02, .05))
    assert credit.update([.10], .2, 0, 1.) == (0., 0.)
    assert credit.update([.12], .3, 0, 1.) == pytest.approx((.02, .1))
    assert credit.update([.08], .1, 0, .25) == pytest.approx((-.04, -.2))


@pytest.fixture
def env():
    instance = make_rl_env(robot='nero', action_mode='task_space_vertical_down',
                          scenario_names=['id_static'], rl_config=RLConfig(
                              target_node_task=True, target_reward_profile='tolerance_progress_v2'))
    instance.reset(seed=81, options={'goal_node_index': 20})
    yield instance
    instance.close()


@pytest.mark.parametrize('cause,key,penalty', [
    ('active_open', 'reward_active_open_after_secured', -8.),
    ('physical_slip', 'reward_physical_slip', -3.),
    ('open_during_contact_loss', 'reward_physical_slip', -3.),
])
@pytest.mark.parametrize('node,qualified', [(20, True), (23, False)])
def test_delayed_release_uses_secured_session_target_qualification(env, cause, key, penalty, node, qualified):
    held(env, node)
    status = dict(pinch_confirmed=True, secured_grasp=True, grasp_lift_delta=.05,
                  grasp_body_height_above_table=.09, strict_success_hold=0., height_band_hold=0.)
    env._secured_session_active = True
    env._reward(np.zeros(5), False, {'lifted_fraction': .1}, status)
    # Contact qualification disappears before the delayed physical break.
    env.base_env.grasp_state = None
    status.update(pinch_confirmed=False, secured_grasp=False)
    env._reward(np.zeros(5), False, {'lifted_fraction': 0.}, status)
    assert not env.goal.previous_secured
    env.base_env.last_grasp_break = dict(time=1., reason='physical_grasp_geometry_lost', causal_class=cause)
    env.base_env.grasp_break_history.append(env.base_env.last_grasp_break)
    _, parts = env._reward(np.zeros(5), False, {'lifted_fraction': 0.}, status)
    assert parts[key] == (penalty if qualified else 0.)
    assert not env._secured_session_target_match
    _, parts = env._reward(np.zeros(5), False, {'lifted_fraction': 0.}, status)
    assert parts[key] == 0.


def held(env, index):
    env.base_env.grasp_state = GraspState(body_id=env.base_env.cable_ids[index],
        candidate_time=0., bilateral_confirmed=True, last_bilateral_time=0., lost_contact_time=0.)


@pytest.mark.parametrize('error,reward,scale,terminate', [
    (0, 2., 1., False), (1, 1.75, 1., False), (2, 1.5, 1., False),
    (3, 1.25, .5, False), (4, 1., .25, False), (5, 0., 0., True),
])
def test_first_position_and_lift_tolerances_are_independent(env, error, reward, scale, terminate):
    held(env, 20+error)
    env.goal.qualify(False)
    parts = {}
    status = {'pinch_confirmed': True, 'secured_grasp': True}
    env.goal.finish_reward(parts, status)
    assert parts['reward_first_grasp_proximity'] == reward
    assert env.goal.lift_reward_scale() == scale
    assert env.goal.matches() is (error <= 2)
    assert env.goal.first_grasp_outside_reward_range is terminate
    assert parts['reward_wrong_target_grasp'] == (-2. if error > 4 else 0.)
    env.goal.finish_reward(parts, status)
    assert parts['reward_first_grasp_proximity'] == parts['reward_wrong_target_grasp'] == 0.


def test_actual_reward_path_discounts_lift_without_awarding_success_milestones(env):
    held(env, 23)
    status = dict(pinch_confirmed=True, secured_grasp=True, grasp_lift_delta=.05,
                  grasp_body_height_above_table=.09, strict_success_hold=.8, height_band_hold=.8)
    top = env.data.geom_xpos[env._table_geom_id, 2]+env.model.geom_size[env._table_geom_id, 2]
    env.data.xpos[env.base_env.cable_ids[23], 2] = top+.09
    _, parts = env._reward(np.zeros(5), False, {'lifted_fraction': .1}, status)
    assert parts['reward_lift_progress'] == pytest.approx(40*.5*(.09-.045))
    assert parts['reward_cable_lift_progress'] == pytest.approx(3*.5*.1)
    assert parts['reward_new_secured_grasp'] == parts['reward_strict_hold'] == parts['reward_success'] == 0.
    env.base_env.grasp_state = None
    env.data.xpos[env.base_env.cable_ids[23], 2] = top+.02
    status.update(pinch_confirmed=False, secured_grasp=False)
    env._reward(np.zeros(5), False, {'lifted_fraction': 0.}, status)
    held(env, 24)
    status.update(pinch_confirmed=True, secured_grasp=True)
    env.data.xpos[env.base_env.cable_ids[24], 2] = top+.09
    _, parts = env._reward(np.zeros(5), False, {'lifted_fraction': .1}, status)
    assert parts['reward_lift_progress'] == parts['reward_cable_lift_progress'] == 0.
    env.data.xpos[env.base_env.cable_ids[24], 2] = top+.11
    _, parts = env._reward(np.zeros(5), False, {'lifted_fraction': .15}, status)
    assert parts['reward_lift_progress'] == pytest.approx(40*.25*.02)
    assert parts['reward_cable_lift_progress'] == pytest.approx(3*.25*.05)


def test_wrong_first_event_allows_recovery_without_success(env, monkeypatch):
    held(env, 25)
    monkeypatch.setattr(env.base_env, 'step', lambda action: (None, 0., False, False,
                                                           env.base_env.info()))
    _, _, terminated, _, info = env.step(np.zeros(5))
    assert not terminated and not info['success']
    assert env.goal.first_grasp_outside_reward_range
    assert info.get('termination_reason') != 'wrong_target_first_grasp'


def test_wrong_events_cap_and_regrasp_never_reissues_first_reward(env):
    parts = {}
    for _ in range(6):
        held(env, 25)
        env.goal.finish_reward(parts, dict(pinch_confirmed=True, secured_grasp=False))
        env.base_env.grasp_state = None
        env.goal.finish_reward(parts, dict(pinch_confirmed=False, secured_grasp=False))
    assert env.goal.wrong_penalty == 8.
    held(env, 20)
    env.goal.finish_reward(parts, dict(pinch_confirmed=True, secured_grasp=True))
    assert parts['reward_first_grasp_proximity'] == 0.


def test_revised_reward_completes_real_ppo_updates_with_finite_metrics(env, tmp_path):
    import json
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from panda_cable_grasp.rl.health import TrainingHealthCallback
    torch.set_num_threads(1)
    vec = DummyVecEnv([lambda: env])
    model = PPO('MlpPolicy', vec, n_steps=8, batch_size=4, n_epochs=1, device='cpu')
    health = tmp_path/'health.json'
    model.learn(16, callback=TrainingHealthCallback(health))
    record = json.loads(health.read_text())
    assert record['normal_training'] and record['parameters_finite']
    assert record['updates'] > record['starting_updates']


def test_reference_config_preserves_other_rewards_and_legacy_resume_is_explicit(tmp_path, monkeypatch):
    import json
    from dataclasses import asdict
    from panda_cable_grasp.rl.train import parse_args, rl_config_from_args
    cfg = asdict(RLConfig(target_node_task=True, target_node_tolerance=4,
                         graspable_end_fraction=0., safety_reward_enabled=True))
    reference = tmp_path/'training_config.json'
    reference.write_text(json.dumps(dict(rl_config=cfg, rl_interface_version='full6_workspace_target_node_v1')))
    checkpoint = tmp_path/'checkpoint.zip'; checkpoint.touch()
    argv = ['train', '--resume', str(checkpoint), '--resume-legacy-full6', '--execution-profile', 'nero',
            '--target-node-task', '--target-reward-profile', 'tolerance_progress_v2',
            '--rl-config-file', str(reference), '--robot', 'nero',
            '--action-mode', 'task_space_vertical_down',
            '--controller-profile', 'enhanced6_bounded_dls_v1']
    monkeypatch.setattr('sys.argv', argv)
    args = parse_args()
    new = asdict(rl_config_from_args(args))
    for key, value in cfg.items():
        if key in new and key not in ('target_node_tolerance', 'target_reward_profile'):
            assert new[key] == value
    assert new['target_node_tolerance'] == 2
    monkeypatch.setattr('sys.argv', [item for item in argv if item != '--resume-legacy-full6'])
    with pytest.raises(SystemExit):
        parse_args()
