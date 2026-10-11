"""Check passive measurement and the geometry Jacobian against real rollout."""
import hashlib
import json
import os
from pathlib import Path
import sys
RUN = Path(os.environ['FULL6_PPO_RUN'])
sys.path.insert(0,str(RUN))
import train_targetnode_full6_workspace_eval13 as trainer
from table_execution_guard import TableGeometry
import mujoco
import numpy as np


def state(base):
    result = np.empty(mujoco.mj_stateSize(base.model,mujoco.mjtState.mjSTATE_FULLPHYSICS))
    mujoco.mj_getState(base.model,base.data,result,mujoco.mjtState.mjSTATE_FULLPHYSICS)
    return result


if __name__ == '__main__':
    directory = Path(sys.argv[1])
    trainer.torch.set_num_threads(1)
    meta = json.loads((directory/'metadata.json').read_text())
    source = np.load(directory/'trajectory.npz')
    env = trainer.CourseEnv(1000+meta['episode'],evaluation=True)
    env.set_stage(0)
    obs,initial = env.reset(seed=meta['seed'])
    b = env.env.base_env
    geometry = TableGeometry(b)
    start_state = state(b)
    # Independent probe data only. The actual rollout state is never edited.
    probe = mujoco.MjData(b.model)
    mujoco.mj_setState(b.model,probe,start_state,mujoco.mjtState.mjSTATE_FULLPHYSICS)
    mujoco.mj_forward(b.model,probe)
    error = 0.
    for g,vertices in geometry.meshes:
        world = vertices @ b.data.geom_xmat[g].reshape(3,3).T+b.data.geom_xpos[g]
        vi = int(np.argmin(world[:,2]))
        jp,jr = np.zeros((3,b.model.nv)),np.zeros((3,b.model.nv))
        mujoco.mj_jac(b.model,b.data,jp,jr,world[vi],int(b.model.geom_bodyid[g]))
        for j,qadr in enumerate(b.arm_qpos_adr):
            probe.qpos[:] = b.data.qpos
            probe.qpos[qadr] += 1e-6
            mujoco.mj_forward(b.model,probe)
            perturbed = probe.geom_xpos[g]+probe.geom_xmat[g].reshape(3,3)@vertices[vi]
            error = max(error,float(np.max(abs((perturbed-world[vi])/1e-6-jp[:,b.arm_dof_adr[j]]))))
    assert np.array_equal(state(b),start_state)
    assert error < 2e-6, error
    # Full baseline evaluation without the Observer or any guard hook.
    policy = trainer.PPO.load(meta['checkpoint'],device='cpu')
    difference = float(np.max(abs(state(b)-source['states'][0])))
    for i,expected in enumerate(source['actions']):
        action,_ = policy.predict(obs,deterministic=True)
        assert np.array_equal(action,expected)
        obs,reward,done,truncated,info = env.step(action)
        difference = max(difference,float(np.max(abs(state(b)-source['states'][i+1]))))
        assert np.array_equal(obs,source['observations'][i+1])
        assert reward == source['rewards'][i]
        assert bool(done or truncated) == (i == len(source['actions'])-1)
    assert difference == 0, difference
    report = dict(passive_observer_full_episode_exact_match=True,frames=len(source['states']),
        baseline_max_state_difference=difference,geometry_jacobian_max_error=error,
        geometry_probe_did_not_modify_actual_state=True,
        measured_geometry=[dict(geom=g,body=mujoco.mj_id2name(b.model,mujoco.mjtObj.mjOBJ_BODY,b.model.geom_bodyid[g]),
            hull_vertices=len(v)) for g,v in geometry.meshes],
        checkpoint_sha256=meta['checkpoint_sha256'],initial_state_sha256=meta['initial_state_sha256'],
        source_sha256=trainer.sha(Path(__file__)))
    env.close()
    (directory.parents[1]/'measurement_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
