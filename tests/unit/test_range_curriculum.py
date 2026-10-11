import numpy as np
import pytest
from panda_cable_grasp.rl.range_curriculum import ALL_SCENES, Evidence, cells, sample_episode, axes
from panda_cable_grasp.scenarios.registry import get_scenario

GOOD=dict(success=1,loaded=1,secured=1,attempt=1)
BAD=dict(success=0,loaded=0,secured=0,attempt=1)

def test_weak_cell_cannot_be_hidden_by_strong_cells():
    e=Evidence();name=ALL_SCENES[-1]
    for c in cells(name):
        for _ in range(50):e.add(name,c,BAD if c=='111' else GOOD)
    for n in ALL_SCENES[:-1]:
        for _ in range(20):e.add(n,cells(n)[0],GOOD)
    result=e.decide(5)
    assert result is not None and not result['passed']
    assert e.decide(5) is None  # decision evidence must be fresh

def test_mastery_moves_probability_up_without_eliminating_any_cell():
    e=Evidence();name=ALL_SCENES[-2];before=e.probabilities(name)
    for _ in range(50):e.add(name,'00',GOOD)
    after=e.probabilities(name)
    assert after[0]<before[0] and after[1]>before[1] and after[2]>before[2]
    assert all(p>=.2/4 for p in after)
    assert Evidence(e.state()).probabilities(name)==pytest.approx(after)

def test_episode_parameters_are_independent_reproducible_and_bounded():
    n=ALL_SCENES[-1];s=get_scenario(n);spec=dict(probabilities=[1/8]*8)
    rng=np.random.default_rng(123)
    draws=[sample_episode(s,rng,spec,5) for _ in range(1000)]
    assert sample_episode(s,np.random.default_rng(123),spec,5)==draws[0]
    params=np.array([[d['parameters'][a] for a in axes(n)] for d in draws])
    assert params[:,0].min()>=.25*7/9 and params[:,0].max()<.30
    assert params[:,1].min()>=1 and params[:,1].max()<1.5
    assert params[:,2].min()>=7/9 and params[:,2].max()<1
    assert np.max(np.abs(np.corrcoef(params.T)-np.eye(3)))<.12
    assert len({d['cell'] for d in draws})==8
    assert all(d['overrides']['rigid_frequency_scale']==1 for d in draws)

def test_static_and_replay_have_only_meaningful_axes():
    assert axes(ALL_SCENES[0])==()
    assert axes(ALL_SCENES[4])==('replay_time_scale',)

def test_partial_window_does_not_pass_or_change_full_mastery():
    e=Evidence();name=ALL_SCENES[0]
    for _ in range(99):e.add(name,'',GOOD)
    assert e.decide(0) is None
    e.add(name,'',GOOD)
    assert e.decide(0)['passed']

def test_split_clock_preserves_legacy_and_decouples_combined_rigid_speed():
    from types import SimpleNamespace
    from panda_cable_grasp.env.environment import CableGraspEnv, EnvConfig
    obj=SimpleNamespace(config=EnvConfig())
    obj.config.motion_frequency_scale=.8
    assert CableGraspEnv._rigid_motion_speed(obj)==pytest.approx(.2)
    obj.config.rigid_frequency_scale=1.
    obj.config.rigid_motion_nominal_speed=.3
    obj.config.motion_frequency_scale=.9
    assert CableGraspEnv._rigid_motion_speed(obj)==pytest.approx(.3)
    obj.config.motion_frequency_scale=.7
    assert CableGraspEnv._rigid_motion_speed(obj)==pytest.approx(.3)
