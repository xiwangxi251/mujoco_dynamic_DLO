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


def args_for(tmp_path, monkeypatch):
    root = Path(__file__).parents[2]
    bank = os.environ['PANDA_CABLE_TEST_REPLAY_BANK_DIR']
    monkeypatch.setattr(sys, 'argv', ['train', '--output', str(tmp_path),
        '--robot','nero','--action-mode','task_space_vertical_down','--observation-mode','state',
        '--workspace-scene-family','cable_x075_v1','--ordinary-family-curriculum',
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


@pytest.mark.parametrize('scene', [name for group in GROUPS for name in group])
def test_all_scenes_live_isolation_and_original_steps(scene, tmp_path, monkeypatch):
    args = args_for(tmp_path, monkeypatch)
    reference = train.make_worker(0,args)()
    core = train.make_worker(0,args)()
    reference.set_rollout_curriculum(0, (scene,), (1.,), {scene: 1.})
    core.set_rollout_curriculum(0, (scene,), (1.,), {scene: 1.})
    try:
        expected,_ = reference.reset(seed=20260804)
        snapshots = {}
        original_reset = core.reset
        def capture_reset(**kwargs):
            result = original_reset(**kwargs)
            snapshots['physics'] = physics(core.base_env)
            snapshots['rng'] = json.dumps(core.base_env.rng.bit_generator.state)
            return result
        monkeypatch.setattr(core, 'reset', capture_reset)
        wrapper = FutureGeometryObservation(core)
        got, info = wrapper.reset(seed=20260804)
        assert got.shape == (1419,) and np.isfinite(got).all()
        assert np.array_equal(got[:99], expected)
        assert np.array_equal(physics(core.base_env), snapshots['physics'])
        assert json.dumps(core.base_env.rng.bit_generator.state) == snapshots['rng']
        assert info['future_valid_frames'] == 30
        shadow = natural_branch(core.base_env)
        assert not robot_cable_contact(shadow)
        assert np.array_equal(shadow.data.eq_active, core.base_env.data.eq_active)
        errors = []
        segments, alpha = wrapper.material_mapping
        for step in range(1, 16):
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
    finally:
        reference.close()
        core.close()
