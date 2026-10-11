"""Frozen server-2 range PPO driver, preflight and independent supervisor."""
import argparse
from collections import deque
import concurrent.futures
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback
import zipfile

BASE=Path('/root/rivermind-data')
SOURCE=BASE/'ppo_cpu_project/range-ppo-20261012'
ROOT=BASE/'runs/ppo_range_session_height_scratch_20m_20261012'
PY=BASE/'conda-envs/ppo-cpu/bin/python'
BANK=BASE/'ppo_cpu_project/range_inputs'
MAP=SOURCE/'src/panda_cable_grasp/rl/workspaces/workspace3d_20261010.json'
BUDGET=20000000

def utc(t=None):
    return dt.datetime.fromtimestamp(time.time() if t is None else t,dt.timezone.utc).isoformat()

def read(p):
    return json.loads(p.read_text()) if p.exists() else {}

def write(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(d,indent=2,allow_nan=False));tmp.replace(p)

def identity(pid):
    try:
        root=Path('/proc')/str(pid)
        return dict(pid=pid,birth=root.joinpath('stat').read_text().rsplit(')',1)[1].split()[19],
            command=root.joinpath('cmdline').read_bytes().decode().strip('\0').split('\0'),
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except (FileNotFoundError,ProcessLookupError):return None

def setup():
    os.environ.update(PYTHONPATH=str(SOURCE/'src'),PYTHONPYCACHEPREFIX=str(ROOT/'pycache'),
        PANDA_CABLE_OUTPUT_ROOT=str(BASE/'runs'),MUJOCO_MENAGERIE_PATH=str(BASE/'ppo_cpu_project/mujoco_menagerie'),
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MUJOCO_GL='osmesa',PYOPENGL_PLATFORM='osmesa')
    sys.path.insert(0,str(SOURCE/'src'))
    resource.setrlimit(resource.RLIMIT_NOFILE,(4096,resource.getrlimit(resource.RLIMIT_NOFILE)[1]))

def arguments(output,workers=40,steps=BUDGET,nsteps=1024,epochs=5,batch=512):
    return ['--output',str(output),'--timesteps',str(steps),'--workers',str(workers),'--torch-threads','1',
        '--robot','nero','--action-mode','task_space_vertical_down','--observation-mode','state',
        '--controller-profile','enhanced6_bounded_dls_v1','--table-execution-guard',
        '--disable-table-finger-collision-filter','--contact-read-cache','--enable-safety-penalties',
        '--episode-seconds','15','--disturbance','1.5','--seed','20261010',
        '--training-distribution','l1','--workspace-scene-family','cable_x075_v1','--ordinary-family-curriculum',
        '--range-motion-curriculum','--penalty-ramp-reference-workers','48','--ordinary-reward-profile','session_height_v1',
        '--rl-config-file',str(ROOT/'reference_config.json'),'--action-workspace',str(MAP),
        '--replay-bank-dir',str(BANK),'--replay-seed-fallback','resample',
        '--schedule-total-timesteps','20000000','--n-steps',str(nsteps),'--batch-size',str(batch),
        '--n-epochs',str(epochs),'--device','cpu','--learning-rate-initial','.0003','--learning-rate-final','.00003',
        '--entropy-coef-initial','.01','--entropy-coef-final','.001','--checkpoint-steps','200000',
        '--eval-freq','0','--eval-workers','0']

def parsed(output,workers=40,steps=BUDGET,nsteps=1024,epochs=5,batch=512):
    setup()
    from panda_cable_grasp.rl import train
    sys.argv=['train',*arguments(output,workers,steps,nsteps,epochs,batch)]
    return train,train.parse_args()

def real_scene_preflight(name):
    import numpy as np
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from panda_cable_grasp.rl.range_curriculum import Evidence,cells
    train,args=parsed(ROOT/'preflight',workers=2,steps=6,nsteps=3,epochs=1,batch=2)
    factories=[train.make_worker(i,args) for i in range(2)]
    v=SubprocVecEnv(factories,start_method='spawn')
    try:
        v.env_method('set_range_curriculum',5,(name,),(1.,),{name:dict(probabilities=[1/len(cells(name))]*len(cells(name)))})
        obs=v.reset();samples=[]
        for _ in range(3):
            obs,rewards,dones,infos=v.step(np.zeros((2,5),dtype=np.float32))
            assert np.isfinite(obs).all() and np.isfinite(rewards).all()
            samples.append([i['range_episode'] for i in infos])
        assert samples[0]==samples[-1]  # first .06s: parameters fixed, speed window not started
        return dict(scene=name,workers=2,real_steps=3,observation_shape=list(obs.shape),samples=samples[0])
    finally:v.close()

def calibration_one(job):
    import numpy as np
    setup()
    from panda_cable_grasp.rl.range_curriculum import axes,cells,BOUNDS
    name,cell,seed=job
    train,args=parsed(ROOT/'calibration')
    env=train.make_worker(0,args)()
    try:
        p=[float(c==cell) for c in cells(name)]
        env.set_range_curriculum(5,(name,),(1.,),{name:dict(probabilities=p)})
        env.reset(seed=seed)
        # Parameter corners, no policy, no recording. The same physical factory.
        params={a:BOUNDS[a][int(b)] for a,b in zip(axes(name),cell)}
        cfg=env.base_env.config
        if 'rigid_speed' in params:cfg.rigid_motion_nominal_speed=params['rigid_speed'];cfg.rigid_frequency_scale=1.
        if 'shape_strength' in params:cfg.disturbance_strength=params['shape_strength'];cfg.motion_frequency_scale=params['shape_frequency']
        if 'replay_time_scale' in params:cfg.replay_time_scale=params['replay_time_scale']
        values=[]
        while env.data.time<7.4:
            env.base_env.step(env.base_env.ready_ctrl.copy())
            if env.data.time>=.82:
                velocity=np.array([env.base_env.body_linear_velocity(i) for i in env.base_env.cable_ids])
                values.append(float(np.average(np.linalg.norm(velocity,axis=1),weights=env.base_env.cable_mass)))
        return dict(scene=name,cell=cell,seed=seed,parameters=params,mean_speed=float(np.mean(values)),
            peak_speed=max(values),samples=len(values),finite=bool(np.isfinite(values).all()),
            measurement='no-policy mass-weighted mean node speed .82..7.4 seconds')
    finally:env.close()

def preflight():
    setup()
    from panda_cable_grasp.rl.range_curriculum import ALL_SCENES,cells, Evidence, VERSION, checkpoint_state
    from stable_baselines3 import PPO
    import torch
    scenes=[real_scene_preflight(n) for n in ALL_SCENES]
    train,args=parsed(ROOT/'smoke_train',workers=3,steps=14,nsteps=2,epochs=1,batch=2)
    train.main()
    model=PPO.load(ROOT/'smoke_train/final_model.zip',device='cpu')
    assert model.num_timesteps==14 and model._n_updates==3 and model.range_curriculum_state['phase']==0
    assert all(torch.isfinite(p).all() for p in model.policy.parameters())
    e=Evidence()
    for _ in range(50):e.add(ALL_SCENES[2],'0',dict(success=1,loaded=1,secured=1,attempt=1))
    model.range_curriculum_state=dict(version=VERSION,phase=2,evidence=e.state(),episode_counter=50)
    model.save(ROOT/'smoke_train/resume_input.zip')
    setup()
    from panda_cable_grasp.rl import train
    sys.argv=['train',*arguments(ROOT/'smoke_resume',workers=3,steps=12,nsteps=2,epochs=1,batch=2),
        '--resume',str(ROOT/'smoke_train/resume_input.zip')]
    args=train.parse_args()
    assert args.rollout_initial_phase==2
    probe=train.make_worker(0,args)()
    try:
        _,info=probe.reset(seed=20261010)
        assert info['range_episode']['phase']==2
    finally:probe.close()
    train.main()
    resumed=PPO.load(ROOT/'smoke_resume/final_model.zip',device='cpu')
    assert resumed.num_timesteps==26 and resumed.range_curriculum_state['phase']==2
    assert resumed.penalty_ramp_schedule['height_full_samples']==480000
    assert len(resumed.range_curriculum_state['evidence']['queues'][ALL_SCENES[2]]['0'])==50
    # Exercise actual terminal-info->callback->checkpoint path with complete episodes.
    train,args=parsed(ROOT/'episode_smoke',workers=2,steps=1504,nsteps=752,epochs=1,batch=64)
    train.main()
    ep_model=PPO.load(ROOT/'episode_smoke/final_model.zip',device='cpu')
    assert ep_model.range_curriculum_state['episode_counter']>=2
    jobs=[(n,c,20260804) for n in ALL_SCENES[2:] for c in cells(n)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as pool:
        calibration=list(pool.map(calibration_one,jobs))
    assert all(x['finite'] for x in calibration)
    write(ROOT/'preflight.json',dict(passed=True,scenes=scenes,exact_short_budget=14,normal_updates=3,
        resume_exact_steps=26,resume_phase=2,penalty_ramp_samples=(480000,240000),
        complete_callback_episodes=ep_model.range_curriculum_state['episode_counter'],calibration=calibration,
        scientific_result=False,completed_at=utc()))
    print('preflight_passed',flush=True)

def summary():
    gate=read(ROOT/'range_gate_status.json');result=dict(gate=gate)
    p=ROOT/'range_episodes.csv'
    if p.exists():
        with p.open() as f:rows=deque(csv.DictReader(f),maxlen=300)
        for size in (100,300):
            q=list(rows)[-size:]
            result['training_last'+str(size)]=dict(episodes=len(q),phase_counts={ph:sum(r['phase']==ph for r in q) for ph in {r['phase'] for r in q}},
                counts={k:sum(int(r[k]) for r in q) for k in ('success','loaded','secured','attempt')})
    p=ROOT/'range_gate_rounds.jsonl'
    if p.exists():
        result['recent_rounds']=[json.loads(x) for x in deque(p.read_text().splitlines(),maxlen=3)]
    return result

def verify_configuration():
    cfg=read(ROOT/'training_config.json')
    if not cfg:return False
    rl=cfg['rl_config']
    assert cfg['actual_eval_workers']==0 and not cfg['independent_evaluation_enabled']
    assert cfg['ordered_curriculum']['version']=='range_ability_rollout_v1'
    assert cfg['seed']==20261010 and cfg['controller_profile']=='enhanced6_bounded_dls_v1'
    assert rl['ordinary_reward_profile']=='session_height_v1' and rl['height_band_reward_enabled'] and rl['safety_reward_enabled']
    assert not rl['target_node_task'] and rl['action_workspace_sha256']=='43840a44aa6a58e2bfd36ba8c282c29646a0457004f5cfb858ef95789429a756'
    return True

def supervise(failure=False):
    setup()
    output=ROOT/'intentional_failure' if failure else ROOT
    output.mkdir(parents=True,exist_ok=True)
    manifest=read(ROOT/'launch_manifest.json') if not failure else {}
    status=read(output/'status.json')
    now=time.time()
    status.update(status='starting',run=str(output),source_updated_at=utc(now),started_at=utc(now),
        last_progress_at=utc(now),progress_deadline_at=utc(now+3600),supervisor=identity(os.getpid()),completed=0,
        planned=BUDGET,source=str(SOURCE),source_commit=manifest.get('source_commit'),source_interval_seconds=120,
        heartbeat_timeout_seconds=300,progress_timeout_seconds=3600)
    write(output/'status.json',status)
    child=None
    try:
        command=[str(PY),'-c','import sys;sys.exit(7)'] if failure else [str(PY),str(ROOT/'runner.py'),'--train']
        with (output/'stdout.log').open('ab',buffering=0) as out,(output/'stderr.log').open('ab',buffering=0) as err:
            child=subprocess.Popen(command,cwd=SOURCE,env=os.environ.copy(),stdout=out,stderr=err,start_new_session=True)
            status['trainer']=identity(child.pid);write(output/'status.json',status)
            while True:
                now=time.time();health=read(output/'training_health.json')
                status.update(source_updated_at=utc(now),heartbeat_at=utc(now),training_alive=child.poll() is None,
                    identity_verified=identity(child.pid)==status['trainer'] if child.poll() is None else False)
                if health:
                    status['health']=health
                    if health['timesteps']>status['completed']:
                        status.update(completed=health['timesteps'],last_progress_at=utc(now),progress_deadline_at=utc(now+3600))
                    if health.get('worker_ready'):
                        status.update(status='running',configuration_verified=verify_configuration(),
                            first_update_confirmed=health.get('normal_training',False),summary=summary())
                code=child.poll()
                if code is not None:
                    status.update(exit_code=code,training_alive=False,child_exited_at=utc(now))
                    if code:
                        status.update(status='failed',error=(output/'stderr.log').read_text(errors='replace')[-2500:] or f'trainer exited {code}')
                    else:
                        import torch
                        model=output/'final_model.zip'
                        with zipfile.ZipFile(model) as z:
                            meta=json.loads(z.read('data'))
                            tensors=[]
                            for name in ('policy.pth','policy.optimizer.pth'):
                                def collect(value):
                                    if torch.is_tensor(value):tensors.append(value)
                                    elif isinstance(value,dict):
                                        for v in value.values():collect(v)
                                    elif isinstance(value,(list,tuple)):
                                        for v in value:collect(v)
                                collect(torch.load(z.open(name),map_location='cpu',weights_only=False))
                        finite=bool(tensors) and all(bool(torch.isfinite(p).all()) for p in tensors)
                        with (output/'stdout.log').open('rb') as f:
                            f.seek(max(0,(output/'stdout.log').stat().st_size-10000));saved=b'model_saved=' in f.read()
                        assert meta['num_timesteps']==BUDGET and finite and saved and verify_configuration()
                        status.update(status='complete',completed=BUDGET,final_model_valid=True,final_model_timesteps=BUDGET,
                            final_model=str(model),final_model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                            model_saved=saved,parameters_finite=finite,summary=summary())
                    write(output/'status.json',status);return
                if now>dt.datetime.fromisoformat(status['progress_deadline_at']).timestamp():
                    status.update(status='needs_attention',error='No real sample progress for 3600 seconds')
                write(output/'status.json',status)
                time.sleep(1 if failure else 120)
    except BaseException:
        status.update(status='monitor_failed',error=traceback.format_exc(),source_updated_at=utc(),
            training_alive=child is not None and child.poll() is None)
        write(output/'status.json',status);raise

if __name__=='__main__':
    if '--preflight' in sys.argv:preflight()
    elif '--supervise-failure' in sys.argv:supervise(True)
    elif '--supervise' in sys.argv:supervise()
    elif '--train' in sys.argv:
        launch=read(ROOT/'launch_manifest.json');train,args=parsed(ROOT,workers=launch['workers']);train.main()
