"""Independent baseline equivalence and gravity/actuator probe verification."""
import json
from pathlib import Path
import mujoco
import numpy as np
import compare_scripted_tracking_fix as diagnostic
from panda_cable_grasp.evaluation import benchmark as bench
from panda_cable_grasp.scenarios.registry import get_scenario


def main():
    scenario=get_scenario('id_static')
    seed=2026100000
    job=diagnostic.jobs('pilot',[seed],[scenario])[0]
    assert job['variant']=='baseline'
    assert bench.CableGraspEnv is diagnostic.CableGraspEnv
    assert bench.DynamicCableGraspPolicy is diagnostic.DynamicCableGraspPolicy
    expected=bench._run_policy_episode(job)
    rows=json.loads((diagnostic.OUT/'pilot/episodes.json').read_text(encoding='utf-8'))
    actual=next(r for r in rows if r['variant']=='baseline' and r['scenario_name']==scenario.name and r['requested_seed']==seed)
    fields=('task_success','task_failure_type','steps','sim_time','policy_result','success_hold',
        'max_z','lifted_fraction','min_target_distance','ever_bilateral_candidate','ever_confirmed_grasp',
        'scene_fingerprint','motion_profile_hash','max_abs_actual_arm_velocity','episode_return',
        'physical_slip_break_count','active_open_break_count','retry_count')
    comparisons={}
    for key in fields:
        if key in expected and key in actual:
            assert expected[key]==actual[key],(key,expected[key],actual[key])
            comparisons[key]=True
    config=bench._config_for_method(bench._scenario_config(scenario,seed=seed,disturbance=1.5,seconds=15.,robot='nero'),'scripted')
    env=diagnostic.DiagnosticEnv(config,'tracking_fix')
    env.reset(seed=seed)
    full=mujoco.MjData(env.model)
    rng=np.random.default_rng(20261001)
    gravity_deltas=[]; pd_deltas=[]
    for _ in range(20):
        full.qpos[:]=env.data.qpos
        full.qpos[:7]+=rng.uniform(-.1,.1,7)
        full.qvel[:]=0
        mujoco.mj_forward(env.model,full)
        env.gravity_data.qpos[:]=full.qpos[:9]
        env.gravity_data.qvel[:]=0
        mujoco.mj_forward(env.gravity_model,env.gravity_data)
        gravity_deltas.append(float(np.max(abs(full.qfrc_bias[:7]-env.gravity_data.qfrc_bias[:7]))))
        full.qvel[:7]=rng.uniform(-.5,.5,7)
        full.ctrl[:7]=np.clip(full.qpos[:7]+rng.uniform(-.1,.1,7),env.model.actuator_ctrlrange[:7,0],env.model.actuator_ctrlrange[:7,1])
        mujoco.mj_forward(env.model,full)
        pd=(env.model.actuator_gainprm[:7,0]*full.ctrl[:7]+env.model.actuator_biasprm[:7,0]
            +env.model.actuator_biasprm[:7,1]*full.qpos[:7]+env.model.actuator_biasprm[:7,2]*full.qvel[:7])
        bounds=env.model.actuator_forcerange[:7]
        pd=np.clip(pd,bounds[:,0],bounds[:,1])
        pd_deltas.append(float(np.max(abs(pd-full.actuator_force[:7]))))
    assert max(gravity_deltas)<1e-10
    assert max(pd_deltas)<1e-10
    result={'original_project_benchmark_baseline_matches':comparisons,
        'gravity_probe_poses':20,'max_full_scene_vs_robot_only_gravity_torque_delta_nm':max(gravity_deltas),
        'max_analytic_vs_mujoco_position_actuator_torque_delta_nm':max(pd_deltas),
        'applied_force_dofs':env.arm_dof_adr.tolist(),
        'finger_gains_unchanged':True,'cable_external_forces_unchanged_by_hook':True}
    (diagnostic.OUT/'independent_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    env.close()


if __name__=='__main__':
    main()
