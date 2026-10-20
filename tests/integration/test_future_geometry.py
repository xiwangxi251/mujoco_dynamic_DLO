import json
import os
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest

from panda_cable_grasp.rl import train
from panda_cable_grasp.rl.future_geometry import FutureGeometryObservation, natural_branch, robot_cable_contact
from panda_cable_grasp.rl.ordered_curriculum import GROUPS
from panda_cable_grasp.scenarios.registry import SCENARIO_FAMILIES


def args_for(tmp_path, monkeypatch, family='cable_x075_v1'):
    root = Path(__file__).parents[2]
    bank = os.environ.get('PANDA_CABLE_TEST_REPLAY_BANK_DIR')
    if not bank or not Path(bank).is_dir():
        pytest.skip('Set PANDA_CABLE_TEST_REPLAY_BANK_DIR to run motion-bank isolation checks')
    monkeypatch.setattr(sys, 'argv', ['train', '--output', str(tmp_path),
        '--robot','nero','--action-mode','task_space_vertical_down','--observation-mode','state',
        '--workspace-scene-family',family,'--ordinary-family-curriculum',
        '--rollout-ability-curriculum','--rollout-gate-profile','fast',
        '--eval-freq','0','--eval-workers','0','--ordinary-reward-profile','session_height_v1',
        '--rl-config-file',str(root/'archive/session_height_ordered_20m_20261011/rollout_only_20261011/reference_config.json'),
        '--replay-bank-dir',bank,'--replay-seed-fallback','resample',
        '--controller-profile','enhanced6_bounded_dls_v1','--table-execution-guard',
        '--disable-table-finger-collision-filter','--contact-read-cache','--enable-safety-penalties'])
    return train.parse_args()


def physics(env):
    typ = mujoco.mjtState.mjSTATE_INTEGRATION
    result = np.empty(mujoco.mj_stateSize(env.model,typ))
    mujoco.mj_getState(env.model,env.data,result,typ)
    return result


@pytest.mark.parametrize('scene', (*SCENARIO_FAMILIES['cable_x075_v1'],
    *SCENARIO_FAMILIES['cable_x075_sector_speed_v3'],
    SCENARIO_FAMILIES['cable_x075_force_v1'][1]))
def test_all_scenes_live_isolation_and_original_steps(scene, tmp_path, monkeypatch):
    family = ('cable_x075_sector_speed_v3' if scene in SCENARIO_FAMILIES['cable_x075_sector_speed_v3']
              else 'cable_x075_force_v1' if scene in SCENARIO_FAMILIES['cable_x075_force_v1']
              else 'cable_x075_v1')
    args = args_for(tmp_path, monkeypatch, family)
    reference = train.make_worker(0,args)()
    core = train.make_worker(0,args)()
    difficulty = 0. if '_static' in scene else 1.
    reference.set_rollout_curriculum(0, (scene,), (1.,), {scene: difficulty})
    core.set_rollout_curriculum(0, (scene,), (1.,), {scene: difficulty})
    try:
        expected,_ = reference.reset(seed=20260804)
        snapshots = {}
        original_reset = core.reset
        def capture_reset(**kwargs):
            result = original_reset(**kwargs)
            snapshots['physics'] = physics(core.base_env)
            snapshots['rng'] = {k:json.dumps(getattr(core.base_env,k).bit_generator.state)
                                 for k in ('rng', '_shape_rng', '_replay_rng')}
            snapshots['model'] = {k:getattr(core.model,k).copy() for k in
                                 ('eq_data','geom_contype','geom_conaffinity','pair_margin','body_pos')}
            compiler = core.base_env._replay_force_compiler
            if compiler is not None:
                snapshots['compiler'] = physics(compiler)
                snapshots['compiler_gravity'] = compiler._gravity_data.qpos.copy()
            return result
        monkeypatch.setattr(core, 'reset', capture_reset)
        wrapper = FutureGeometryObservation(core)
        got, info = wrapper.reset(seed=20260804)
        assert got.shape == (1419,) and np.isfinite(got).all()
        assert np.array_equal(got[:99], expected)
        assert np.array_equal(physics(core.base_env), snapshots['physics'])
        for k,v in snapshots['rng'].items():
            assert json.dumps(getattr(core.base_env,k).bit_generator.state) == v
        for k,v in snapshots['model'].items():
            assert np.array_equal(getattr(core.model,k),v)
        if 'compiler' in snapshots:
            assert np.array_equal(physics(core.base_env._replay_force_compiler), snapshots['compiler'])
            assert np.array_equal(core.base_env._replay_force_compiler._gravity_data.qpos,
                                  snapshots['compiler_gravity'])
        assert info['future_valid_frames'] == 30
        shadow = natural_branch(core.base_env)
        assert not robot_cable_contact(shadow)
        assert np.array_equal(shadow.data.eq_active, core.base_env.data.eq_active)
        errors = []
        segments, alpha = wrapper.material_mapping
        # Exercise the new surface driver after its .82s startup boundary too.
        steps = 60 if scene == SCENARIO_FAMILIES['cable_x075_sector_speed_v3'][1] else 15
        for step in range(1, steps+1):
            a = np.zeros(5, dtype=np.float32)
            expected, r1, d1, t1, _ = reference.step(a)
            got, r2, d2, t2, _ = wrapper.step(a)
            assert np.array_equal(got[:99], expected)
            assert (r1,d1,t1) == (r2,d2,t2)
            assert np.array_equal(physics(reference.base_env), physics(core.base_env))
            if step % 5 == 0 and wrapper.enabled:
                reader = mujoco.MjData(reference.model)
                mujoco.mj_copyData(reader, reference.model, reference.data)
                mujoco.mj_forward(reference.model, reader)
                xyz = reader.xpos[reference.base_env.cable_ids].copy()[reference._target_object_slice()]
                actual = xyz[segments]*(1-alpha[:,None])+xyz[segments+1]*alpha[:,None]
                errors.extend(np.linalg.norm(actual-wrapper.frames[step//5-1],axis=1))
        assert errors and max(errors) < .005
        # Physics-substep latch survives a subsequent absent contact and retry.
        import panda_cable_grasp.rl.future_geometry as module
        monkeypatch.setattr(module,'robot_cable_contact',lambda env:True)
        core.base_env._update_physical_grasp_state(False)
        assert not wrapper.enabled
        monkeypatch.setattr(module,'robot_cable_contact',lambda env:False)
        core.base_env._update_physical_grasp_state(False)
        assert not wrapper.enabled and not wrapper._observation(expected)[99:].any()
        wrapper.reset(seed=20260804)
        assert wrapper.enabled and wrapper._observation(expected)[99:].reshape(30,44)[:,43].sum()==30
    finally:
        reference.close()
        if 'wrapper' in locals():wrapper.close()
        else:core.close()


def test_future_training_and_native_evaluation_entrypoints(tmp_path, monkeypatch):
    from stable_baselines3 import PPO
    from panda_cable_grasp.rl import evaluate
    out = tmp_path/'train'
    argv = ['train', '--output', str(out), '--future-geometry',
        '--training-scenarios', 'dev_static_x075_v1', '--curriculum-stage-steps', '0',
        '--workers', '2', '--n-steps', '2', '--batch-size', '2', '--n-epochs', '1',
        '--timesteps', '6', '--eval-freq', '0', '--eval-workers', '0', '--torch-threads', '1']
    monkeypatch.setattr(sys, 'argv', argv)
    monkeypatch.setattr(train, 'plot_training_curves', lambda *a, **k: None)
    train.main()
    config = json.loads((out/'training_config.json').read_text())
    assert config['future_geometry_enabled'] and config['observation_dimension'] == 1419
    assert len(config['observation_names']) == 1419
    assert config['sampler'] == 'async_fixed_quota'
    model = PPO.load(out/'final_model.zip', device='cpu')
    assert model.num_timesteps == 6 and model.observation_space.shape == (1419,)
    assert model.future_geometry_protocol['version'] == 'initial_future3s_geometry_v1'
    monkeypatch.setattr(sys, 'argv', ['train', '--resume', str(out/'final_model.zip')])
    assert train.parse_args().future_geometry
    monkeypatch.setattr(sys, 'argv', ['train', '--resume', str(out/'final_model.zip'), '--no-future-geometry'])
    with pytest.raises(SystemExit):train.parse_args()
    monkeypatch.setattr(sys, 'argv', ['eval', '--model', str(out/'final_model.zip'),
        '--headless', '--episodes', '1', '--episode-seconds', '.06', '--no-video',
        '--scenario', 'dev_static_x075_v1', '--video-dir', str(tmp_path/'eval')])
    evaluate.main()
    manifests=list((tmp_path/'eval').glob('*/run_manifest.json'))
    assert len(manifests)==1
    manifest=json.loads(manifests[0].read_text())
    assert manifest['status']=='completed'
    assert manifest['environment']['observation_interface_version']=='state99_future3s_1419_v1'
    assert len(manifest['environment']['observation_names'])==1419
