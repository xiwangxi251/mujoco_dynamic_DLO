"""Make an initialized cable's current curvature its elastic reference.

Call once after reset, before stepping. A fresh environment is required for
each snapshot; reset alone does not restore the modified model reference.
"""
import time
import mujoco
import numpy as np


def set_current_cable_reference(env):
    if getattr(env, '_snapshot_reference_baked', False):
        raise RuntimeError('Reference already baked; create a fresh environment')
    started = time.perf_counter()
    model, data = env.model, env.data
    if float(data.time) != 0.:
        raise ValueError('Bake only immediately after reset')
    xyz, rotation = data.xpos[env.cable_ids].copy(), data.xmat[env.cable_ids].copy()
    qpos, ctrl = data.qpos.copy(), data.ctrl.copy()
    before = float(np.max(np.abs(data.qfrc_passive[env.cable_free_dadr:])))
    for joint, address in zip(env.cable_ball_joint_ids, env.cable_ball_qadr):
        body = model.jnt_bodyid[joint]
        quat = np.empty(4)
        mujoco.mju_mulQuat(quat, model.body_quat[body], qpos[address:address+4])
        model.body_quat[body] = quat
        qpos[address:address+4] = [1., 0., 0., 0.]
    # The elasticity plugin captures reference curvature when MjData is created.
    env.data = mujoco.MjData(model)
    env.data.qpos[:] = qpos
    env.data.qvel[:] = 0.
    env.data.ctrl[:] = ctrl
    env._invalidate_contact_cache()
    mujoco.mj_forward(model, env.data)
    position_error = float(np.max(np.abs(env.data.xpos[env.cable_ids] - xyz)))
    rotation_error = float(np.max(np.abs(env.data.xmat[env.cable_ids] - rotation)))
    passive = float(np.max(np.abs(env.data.qfrc_passive[env.cable_free_dadr:])))
    if position_error > 1e-9 or rotation_error > 1e-9 or passive > 1e-8:
        raise AssertionError((position_error, rotation_error, passive))
    env._snapshot_reference_baked = True
    return dict(seconds=time.perf_counter()-started, position_error_m=position_error,
                rotation_error=rotation_error, passive_before=before, passive_after=passive)
