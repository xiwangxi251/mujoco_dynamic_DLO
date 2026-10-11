"""One frozen scratch training job with independent supervision."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import zipfile

BASE = Path('/data1/hxai/mujoco/mujoco_dynamic_DLO/outputs/rl/train/ppo_session_height_ordered_scratch_20m_20261011')
ROOT = BASE/'rollout_only_20261011'
SOURCE = Path('/data1/hxai/mujoco/worktrees/session-lift-curriculum-20261011')
REPO = Path('/data1/hxai/mujoco/mujoco_dynamic_DLO')
PY = '/data1/hxai/miniconda3/envs/dynamicvla/bin/python'
BUDGET = 20000000
MAP = SOURCE/'src/panda_cable_grasp/rl/workspaces/workspace3d_20261010.json'
MAP_HASH = '43840a44aa6a58e2bfd36ba8c282c29646a0457004f5cfb858ef95789429a756'
FIELDS = ('success', 'lift_attempt', 'loaded_lift', 'ever_grasped', 'ever_aligned_pinch')

def read(p):
    return json.loads(p.read_text()) if p.exists() else {}

def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix('.tmp')
    t.write_text(json.dumps(value, indent=2, allow_nan=False))
    t.replace(p)

def identity(pid):
    try:
        p = Path('/proc')/str(pid)
        st = (p/'stat').read_text().rsplit(')', 1)[1].split()
        return dict(birth=st[19], state=st[0], command=(p/'cmdline').read_bytes().decode().strip('\0').split('\0'))
    except (FileNotFoundError, ProcessLookupError):
        return None

def setup():
    os.environ.update(PYTHONPATH=str(SOURCE/'src'), PANDA_CABLE_OUTPUT_ROOT=str(REPO/'outputs'),
        PYTHONPYCACHEPREFIX=str(ROOT/'pycache'), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', MUJOCO_GL='egl')
    sys.path.insert(0, str(SOURCE/'src'))
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, resource.getrlimit(resource.RLIMIT_NOFILE)[1]))

def arguments():
    return ['--output', str(ROOT), '--timesteps', str(read(ROOT/'resume_plan.json')['remaining_timesteps']), '--resume', read(ROOT/'resume_plan.json')['checkpoint']['path'], '--workers', '48', '--torch-threads', '1',
        '--robot', 'nero', '--action-mode', 'task_space_vertical_down', '--observation-mode', 'state',
        '--controller-profile', 'enhanced6_bounded_dls_v1', '--table-execution-guard',
        '--disable-table-finger-collision-filter', '--contact-read-cache', '--enable-safety-penalties',
        '--episode-seconds', '15', '--disturbance', '1.5', '--seed', '20261010',
        '--training-distribution', 'l1', '--eval-distribution', 'l1',
        '--workspace-scene-family', 'cable_x075_v1', '--ordinary-family-curriculum', '--ordered-ability-curriculum',
        '--ordinary-reward-profile', 'session_height_v1', '--rl-config-file', str(ROOT/'reference_config.json'),
        '--action-workspace', str(MAP), '--replay-bank-dir', str(REPO/'outputs/motion_banks'),
        '--replay-seed-fallback', 'resample', '--eval-replay-bank-dir', str(REPO/'outputs/rl/train/ppo_nero_targetnode_state6_warmstart3m_20260928/eval_banks'),
        '--eval-seed-pool-size', '50', '--curriculum-stage-steps', '500000', '--curriculum-required-evals', '2',
        '--curriculum-secured-rate', '.6', '--curriculum-success-rate', '.6',
        '--n-steps', '1024', '--batch-size', '512', '--n-epochs', '5', '--device', 'cpu',
        '--learning-rate-initial', '.0003', '--learning-rate-final', '.00003',
        '--entropy-coef-initial', '.01', '--entropy-coef-final', '.001',
        '--checkpoint-steps', '200000', '--eval-freq', '0', '--eval-workers', '1',
        '--eval-episodes', '20', '--eval-confirmation-episodes', '20', '--eval-seed', '20260804']

def parsed():
    setup()
    from panda_cable_grasp.rl import train
    sys.argv = ['train', *arguments()]
    return train, train.parse_args()

def preflight():
    import numpy as np
    from dataclasses import asdict
    train, args = parsed()
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
    from panda_cable_grasp.rl.exact_budget import install_exact_budget
    from panda_cable_grasp.rl.ordered_curriculum import GROUPS, PHASES, MIXES, PROTOCOL
    from panda_cable_grasp.rl.session_lift import SessionLiftCredit
    assert subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'], text=True).strip() == read(ROOT/'plan.json')['commit']
    assert not subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain'], text=True).strip()
    assert hashlib.sha256(MAP.read_bytes()).hexdigest() == MAP_HASH
    cfg = asdict(train.rl_config_from_args(args))
    assert cfg == read(ROOT/'reference_config.json')['rl_config']
    assert args.resume is None and args.warm_start is None and not args.target_node_task
    assert cfg['height_band_reward_enabled'] and cfg['safety_reward_enabled'] and cfg['reward_lift_progress'] == 40
    assert len(PHASES) == 14 and args.curriculum_training_mixes == MIXES
    # Same spawned factories, every target scene, real physics steps and full reward path.
    records = []
    v = SubprocVecEnv([train.make_worker(j, args) for j in range(2)], start_method='spawn')
    try:
        for scene in (n for group in GROUPS for n in group):
            v.env_method('set_training_scenarios', (scene,))
            v.env_method('set_motion_difficulty', 1.)
            obs = v.reset()
            assert obs.shape == (2, 99)
            for _ in range(3):
                obs, rewards, dones, infos = v.step(np.zeros((2, 5), dtype=np.float32))
                assert np.isfinite(obs).all() and np.isfinite(rewards).all()
                assert all(info['scenario_name'] == scene for info in infos)
                assert all(info['height_band_reward_enabled'] for info in infos)
                assert all('session_lift_balance_m' in info for info in infos)
            records.append(dict(scenario=scene, workers=2, real_steps=3))
    finally:
        v.close()
    # Exercise the non-divisible final vector step through real spawned workers and VecMonitor.
    v = VecMonitor(SubprocVecEnv([train.make_worker(j,args) for j in range(3)], start_method='spawn'))
    try:
        install_exact_budget()
        model = PPO('MlpPolicy', v, n_steps=2, batch_size=2, n_epochs=1, device='cpu', verbose=0)
        model.learn(total_timesteps=14)
        assert model.num_timesteps == 14 and model._n_updates == 3
        assert int(v.episode_lengths.sum()) == 14 and list(v.episode_lengths) == [5,5,4]
        assert model.rollout_buffer.n_envs == 2 and model.rollout_buffer.buffer_size == 1
    finally:
        v.close()
    c = SessionLiftCredit([.02], .18)
    got = [c.update([.02],0), c.update([.05],0), c.update([.04],0), c.update([.01],None), c.update([.01],0), c.update([.02],0)]
    assert np.allclose(np.asarray(got)*40, [0,1.2,-.4,-.8,0,.4])
    # Test the same independent supervisor loop with an actual exit-7 child.
    subprocess.run([PY, str(Path(__file__).resolve()), '--failure-test'], cwd=SOURCE, check=True, timeout=30)
    failure = read(ROOT/'preflight_failure/status.json')
    assert failure['status'] == 'failed' and failure['exit_code'] == 7
    out = dict(status='passed', source=str(SOURCE), commit=read(ROOT/'plan.json')['commit'],
        scene_smokes=records, exact_budget_test=dict(real_samples=14, workers=3, ppo_rounds=3, last_active_workers=2),
        supervisor_exit_test=failure, expected_rl_config=cfg, ordered_curriculum=PROTOCOL, created_at=time.time())
    write(ROOT/'preflight.json', out)
    print(json.dumps(out), flush=True)

def train_job():
    import torch
    train, args = parsed()
    Original = train.CheckpointCallback
    class Health(Original):
        def health(self):
            cfg=read(ROOT/'training_config.json')
            if cfg and 'actual_eval_workers' not in cfg:
                cfg.update(actual_eval_workers=0,rollout_gate_module_sha256=hashlib.sha256((ROOT/'rollout_gate.py').read_bytes()).hexdigest(),
                    statistics_source='stochastic training episodes',independent_evaluation_enabled=False)
                write(ROOT/'training_config.json',cfg)
            tensors = list(self.model.policy.parameters())
            for state in self.model.policy.optimizer.state.values():
                tensors.extend(v for v in state.values() if torch.is_tensor(v))
            opt = {k:float(v) for k,v in self.model.logger.name_to_value.items() if k.startswith('train/') and isinstance(v,(int,float))}
            finite = all(bool(torch.isfinite(t).all()) for t in tensors) and all(math.isfinite(v) for v in opt.values())
            write(ROOT/'worker_health.json', dict(updated_at=time.time(), timesteps=self.model.num_timesteps,
                ppo_rounds=self.model._n_updates//self.model.n_epochs, optimizer_epochs=self.model._n_updates,
                worker_ready=True, parameters_finite=finite, normal_training=finite, optimization=opt,
                rollout_sampling_seconds=getattr(self,'sampling_seconds',None),
                optimizer_and_bookkeeping_seconds=getattr(self,'optimization_seconds',None),
                rollout_fps=getattr(self,'sampling_fps',None),
                phase=getattr(self.model,'ordinary_evaluation_phase',0),
                difficulty=getattr(self.model,'ordinary_evaluation_difficulty',0.)))
        def _on_training_start(self):
            super()._on_training_start(); self.health()
        def _on_rollout_start(self):
            now=time.monotonic()
            if hasattr(self,'sampling_ended'):self.optimization_seconds=now-self.sampling_ended
            super()._on_rollout_start(); self.health()
            self.sampling_started=time.monotonic();self.sampling_start_steps=self.model.num_timesteps
        def _on_rollout_end(self):
            self.sampling_ended=time.monotonic()
            self.sampling_seconds=self.sampling_ended-self.sampling_started
            self.sampling_fps=(self.model.num_timesteps-self.sampling_start_steps)/self.sampling_seconds
            super()._on_rollout_end()
        def _on_training_end(self):
            super()._on_training_end(); self.health()
    train.CheckpointCallback = Health
    install_resume_hooks(train)
    train.main()

def window(rows, n):
    rows = rows[-n:]
    return dict(episodes=len(rows), counts={k:sum(int(float(r.get(k,0))) for r in rows) for k in FIELDS},
        from_steps=int(rows[0]['timesteps']) if rows else None, to_steps=int(rows[-1]['timesteps']) if rows else None,
        phases=sorted(set(r.get('phase_at_completion','') for r in rows)))

def summary():
    p = ROOT/'rollout_episodes.csv'
    rows = list(csv.DictReader(p.open())) if p.exists() else []
    p = ROOT/'rollout_gate_rounds.jsonl'
    rounds = [json.loads(line) for line in p.read_text().splitlines() if line] if p.exists() else []
    return dict(training_last100=window(rows,100), training_last300=window(rows,300),
        recent_rounds=rounds[-3:], completed_rounds=len(rounds),
        statistics_source='stochastic training episodes',gate=read(ROOT/'rollout_gate_status.json'))

def status(m, code):
    now = time.time(); old = read(ROOT/'status.json'); h = read(ROOT/'worker_health.json'); cfg = read(ROOT/'training_config.json')
    ident = identity(m['pid'])
    alive = bool(ident and ident['birth'] == m['birth'] and ident['command'] == m['command'] and ident['state'] != 'Z')
    expected = read(ROOT/'preflight.json')
    verified = bool(cfg and cfg.get('rl_config') == expected.get('expected_rl_config') and cfg.get('workers') == 48
        and cfg.get('starting_timesteps') == read(ROOT/'resume_plan.json')['checkpoint']['timesteps']
        and cfg.get('resume_checkpoint') == read(ROOT/'resume_plan.json')['checkpoint']['path']
        and cfg.get('cumulative_target_timesteps') == BUDGET and cfg.get('seed') == 20261010
        and cfg.get('ordered_curriculum') == expected.get('ordered_curriculum')
        and cfg.get('table_execution_guard_enabled') and not cfg.get('table_finger_collision_filter_enabled')
        and cfg.get('controller_profile') == 'enhanced6_bounded_dls_v1' and cfg.get('singularity_avoidance_enabled')
        and cfg.get('resume_schedule_mode') == 'continue_from_checkpoint' and cfg.get('learning_rate_final') == .00003
        and cfg.get('entropy_coef_initial') == read(ROOT/'resume_plan.json')['checkpoint']['entropy'] and cfg.get('entropy_coef_final') == .001
        and read(ROOT/'resume_load_receipt.json').get('policy_optimizer_exact') is True
        and cfg.get('actual_eval_workers') == 0 and cfg.get('eval_freq') == 0
        and cfg.get('rollout_gate_module_sha256') == hashlib.sha256((ROOT/'rollout_gate.py').read_bytes()).hexdigest()
        and cfg.get('episode_seconds') == 15 and cfg.get('n_steps') == 1024 and cfg.get('n_epochs') == 5)
    token = [h.get('timesteps'), h.get('ppo_rounds')]
    evidence = [m['started_at']]
    for name in ['rollout_episodes.csv', 'rollout_gate_rounds.jsonl']:
        p = ROOT/name
        token.append(p.stat().st_size if p.exists() else None)
        if p.exists(): evidence.append(p.stat().st_mtime)
    logpath = ROOT/'train.log'
    log = logpath.read_text(errors='replace')[-18000:] if logpath.exists() else ''
    lines = [line for line in log.splitlines() if line.startswith('strict_eval_progress') or line.startswith('strict_eval_start')]
    token.append(lines[-1] if lines else None)
    if lines: evidence.append(logpath.stat().st_mtime)
    if h: evidence.append(h['updated_at'])
    progress = old.get('last_progress_at', m['started_at'])
    if token != old.get('progress_token'): progress = max(progress, max(evidence))
    out = dict(status='running' if h else 'starting', run=str(ROOT), source=str(SOURCE), commit=m['commit'],
        pid=m['pid'], birth=m['birth'], supervisor_pid=m['supervisor_pid'], supervisor_birth=m['supervisor_birth'],
        training_alive=alive, identity_verified=alive, configuration_verified=verified, health=h, budget=BUDGET, actual_eval_workers=0,
        rollout_gate_module_sha256=hashlib.sha256((ROOT/'rollout_gate.py').read_bytes()).hexdigest(),
        source_updated_at=now, heartbeat_timeout_seconds=300, last_progress_at=progress, progress_deadline_at=progress+3600,
        progress_token=token, summary=summary(), first_update_confirmed=bool(verified and h.get('parameters_finite') and h.get('optimizer_epochs',0)>read(ROOT/'resume_plan.json')['checkpoint']['optimizer_epochs']))
    if not alive:
        final = ROOT/'final_model.zip'; steps = None; valid = False
        try:
            with zipfile.ZipFile(final) as z:
                steps = json.loads(z.read('data'))['num_timesteps']
                valid = z.testzip() is None and 'policy.pth' in z.namelist() and 'policy.optimizer.pth' in z.namelist()
        except (OSError, ValueError, KeyError, zipfile.BadZipFile): pass
        ok = code == 0 and valid and steps == BUDGET and h.get('parameters_finite') and verified and 'model_saved='+str(final) in log
        out.update(status='complete' if ok else 'failed', exit_code=code, final_model=str(final), final_model_timesteps=steps, final_model_valid=valid)
        if not ok: out.update(reason='Training exited without valid exact-budget completion', alert_key='child_exit', log_tail=log[-2500:])
    elif cfg and not verified: out.update(status='needs_attention', reason='Actual configuration mismatch', alert_key='configuration_mismatch')
    elif h and not h.get('parameters_finite'): out.update(status='needs_attention', reason='Nonfinite parameters', alert_key='nonfinite')
    elif not h and now > m['started_at']+900: out.update(status='needs_attention', reason='No worker-ready health within 15 minutes', alert_key='startup_deadline')
    elif now > progress+3600: out.update(status='needs_attention', reason='No real progress for 60 minutes', alert_key='progress_deadline')
    write(ROOT/'status.json', out)
    return out

def supervision_loop(child, m, failure=False):
    while True:
        code = child.poll()
        if failure:
            if code is not None:
                write(ROOT/'preflight_failure/status.json', dict(status='failed', exit_code=code, pid=child.pid,
                    supervisor_pid=os.getpid(), source_updated_at=time.time(), reason='Intentional child exit'))
                return
        else:
            try: out = status(m, code)
            except Exception as error:
                write(ROOT/'status.json', dict(status='monitor_failed', run=str(ROOT), reason=str(error),
                    alert_key='monitor_failed', source_updated_at=time.time()))
                # Keep the independent supervisor alive so a transient read can recover.
                out = {}
            if out.get('status') in ('complete','failed'): return
        time.sleep(.1 if failure else 120)

def supervise():
    setup()
    assert read(ROOT/'resume_preflight.json').get('status') == 'passed'
    assert not (ROOT/'launch_manifest.json').exists(), 'Duplicate training launch refused'
    command = [PY, '-u', str(Path(__file__).resolve()), '--train']
    with (ROOT/'train.log').open('wb') as log:
        child = subprocess.Popen(command, cwd=SOURCE, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    m = dict(run=str(ROOT), source=str(SOURCE), commit=read(ROOT/'plan.json')['commit'], pid=child.pid,
        birth=identity(child.pid)['birth'], command=command, supervisor_pid=os.getpid(),
        supervisor_birth=identity(os.getpid())['birth'], started_at=time.time(), arguments=arguments())
    write(ROOT/'launch_manifest.json', m)
    supervision_loop(child, m)

def launch():
    assert read(ROOT/'resume_preflight.json').get('status') == 'passed'
    assert read(ROOT/'stop_previous_receipt.json').get('status') == 'stopped_for_rollout_gates'
    if (ROOT/'supervisor_launch.json').exists():
        print(json.dumps(read(ROOT/'supervisor_launch.json'))); return
    write(ROOT/'status.json', dict(status='starting', run=str(ROOT), source_updated_at=time.time(),
        last_progress_at=time.time(), progress_deadline_at=time.time()+900, heartbeat_timeout_seconds=300))
    with (ROOT/'supervisor.log').open('ab') as log:
        child = subprocess.Popen([PY, str(Path(__file__).resolve()), '--supervise'], cwd=SOURCE,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = dict(pid=child.pid, birth=identity(child.pid)['birth'], started_at=time.time(), run=str(ROOT))
    write(ROOT/'supervisor_launch.json', receipt)
    print(json.dumps(receipt))

def confirm():
    m = read(ROOT/'launch_manifest.json'); h = read(ROOT/'worker_health.json')
    child, supervisor = identity(m['pid']), identity(m['supervisor_pid'])
    assert child and child['birth'] == m['birth'] and child['command'] == m['command'] and child['state'] != 'Z'
    assert supervisor and supervisor['birth'] == m['supervisor_birth'] and supervisor['state'] != 'Z'
    assert h.get('worker_ready') and h.get('parameters_finite') and time.time()-h['updated_at'] < 300
    out = status(m, None)
    assert out['configuration_verified'], out
    receipt = dict(status='worker_ready', launch=m, health=h, configuration_verified=True, identity_verified=True,
        source_checked_at=time.time(), runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write(ROOT/'startup_receipt.json', receipt)
    print(json.dumps(receipt))


def assert_same(a, b):
    import torch
    if torch.is_tensor(a):
        assert torch.equal(a.cpu(),b.cpu())
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: assert_same(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b): assert_same(x,y)
    else: assert a==b

def verify_model(model):
    import torch
    plan=read(ROOT/'resume_plan.json'); checkpoint=plan['checkpoint']
    p=Path(checkpoint['path'])
    assert hashlib.sha256(p.read_bytes()).hexdigest()==checkpoint['sha256']
    with zipfile.ZipFile(p) as z:
        with z.open('policy.pth') as f: assert_same(torch.load(f,map_location='cpu',weights_only=False),model.policy.state_dict())
        with z.open('policy.optimizer.pth') as f: assert_same(torch.load(f,map_location='cpu',weights_only=False),model.policy.optimizer.state_dict())
    assert model.num_timesteps==checkpoint['timesteps'] and model._n_updates==checkpoint['optimizer_epochs']
    assert model.penalty_ramp_schedule['height_full_samples']==480000
    assert model.penalty_ramp_schedule['safety_full_samples']==240000
    return dict(policy_optimizer_exact=True,timesteps=model.num_timesteps,optimizer_epochs=model._n_updates,
        checkpoint_sha256=checkpoint['sha256'],verified_at=time.time(),
        optimizer_learning_rate=model.policy.optimizer.param_groups[0]['lr'],
        penalty_ramp_schedule=model.penalty_ramp_schedule)

def install_resume_hooks(train):
    from rollout_gate import install
    install(train,ROOT,read(ROOT/'resume_plan.json')['restored_phase'])
    # SB3 progress_remaining is cumulative when reset_num_timesteps=False.
    train.linear_schedule=lambda initial,final,**kw: lambda progress: .00003+.00027*max(0.,min(1.,progress))
    def entropy_apply(self):
        self.model.ent_coef=.01-.009*min(1.,self.num_timesteps/BUDGET)
        self.logger.record('train/entropy_coefficient',self.model.ent_coef)
    train.EntropyCoefficientScheduleCallback._apply=entropy_apply
    original_load=train.PPO.load
    def checked_load(*args,**kwargs):
        model=original_load(*args,**kwargs)
        write(ROOT/'resume_load_receipt.json',verify_model(model))
        return model
    train.PPO.load=staticmethod(checked_load)

def resume_preflight():
    setup()
    import torch
    torch.set_num_threads(1)
    import numpy as np
    from dataclasses import asdict
    train,args=parsed()
    assert args.eval_freq==0
    install_resume_hooks(train)
    from rollout_gate import configure_env,PROTOCOL
    from panda_cable_grasp.rl.ordered_curriculum import OrderedMotionCurriculum,PROTOCOL as FULL_PROTOCOL
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv,VecMonitor
    assert asdict(train.rl_config_from_args(args))==read(ROOT/'reference_config.json')['rl_config']
    subprocess.run([PY,str(ROOT/'test_gate.py')],check=True,timeout=20)
    v=VecMonitor(SubprocVecEnv([train.make_worker(j,args) for j in range(2)],start_method='spawn'))
    try:
        model=PPO.load(args.resume,env=v,device='cpu',n_steps=1024,batch_size=512,n_epochs=5)
        equality=verify_model(model)
        from panda_cable_grasp.rl.penalty_ramp import apply_progress
        apply_progress(v,model,model.penalty_ramp_schedule,training_workers=48)
        obs=v.reset()
        for _ in range(3):
            actions,_=model.predict(obs,deterministic=True)
            obs,rewards,dones,infos=v.step(actions)
            assert obs.shape==(2,99) and np.isfinite(obs).all() and np.isfinite(rewards).all()
            assert all(i['height_band_reward_enabled'] and i['rollout_episode_phase']==0 for i in infos)
        verify_model(model)
        # Phase changes affect future resets, never relabel a currently running episode.
        from rollout_gate import difficulties
        names=('dev_rigid_l1_nominal_x075_v1',)
        v.env_method('set_rollout_curriculum',5,names,(1.,),difficulties(5))
        _,_,_,infos=v.step(np.zeros((2,5),dtype=np.float32))
        assert all(i['rollout_episode_phase']==0 for i in infos)
        v.reset()
        _,_,_,infos=v.step(np.zeros((2,5),dtype=np.float32))
        assert all(i['rollout_episode_phase']==5 and i['rollout_episode_difficulty']==1. for i in infos)
        # Real callback lifecycle and same factory without constructing eval workers.
        from panda_cable_grasp.rl.exact_budget import install_exact_budget
        install_exact_budget()
        small=PPO('MlpPolicy',v,n_steps=2,batch_size=2,n_epochs=1,device='cpu',verbose=0)
        callback=OrderedMotionCurriculum(stages=args.curriculum_stages,training_mixes=args.curriculum_training_mixes)
        small.learn(6,callback=callback)
        assert small.num_timesteps==6 and small.rollout_gate_state['version']==PROTOCOL['version']
    finally:v.close()
    # Smoke progress must not enter formal training or promotion windows.
    smoke=ROOT/'preflight_gate_records';smoke.mkdir(exist_ok=True)
    for name in ['rollout_episodes.csv','rollout_gate_status.json','rollout_gate_rounds.jsonl']:
        p=ROOT/name
        if p.exists():p.replace(smoke/name)
    failure_classifier()
    old=read(ROOT/'preflight.json');old['ordered_curriculum']=FULL_PROTOCOL
    write(ROOT/'preflight.json',old)
    write(ROOT/'resume_preflight.json',dict(status='passed',workers=2,real_steps=3,callback_smoke_samples=6,
        mixed_phase_and_mastered_replay_verified=True,model=equality,protocol=FULL_PROTOCOL,created_at=time.time()))
    print(json.dumps(read(ROOT/'resume_preflight.json')),flush=True)

def failure_classifier():
    global ROOT
    saved=ROOT
    ROOT=saved/'resume_failure_test'
    ROOT.mkdir(exist_ok=True)
    (ROOT/'rollout_gate.py').write_bytes((saved/'rollout_gate.py').read_bytes())
    command=[PY,'-c','raise SystemExit(7)']
    child=subprocess.Popen(command)
    m=dict(pid=child.pid,birth=identity(child.pid)['birth'],command=command,
        supervisor_pid=os.getpid(),supervisor_birth=identity(os.getpid())['birth'],
        started_at=time.time(),commit=read(saved/'resume_plan.json')['commit'])
    try:
        child.wait(timeout=15)
        supervision_loop(child,m)
        result=read(ROOT/'status.json')
        assert result['status']=='failed' and result['exit_code']==7 and not result['training_alive']
    finally: ROOT=saved


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for flag in ['preflight','train','supervise','launch','confirm','failure-test']: parser.add_argument('--'+flag, action='store_true')
    args = parser.parse_args()
    if args.preflight: resume_preflight()
    elif args.train: train_job()
    elif args.supervise: supervise()
    elif args.launch: launch()
    elif args.confirm: confirm()
    elif args.failure_test:
        child = subprocess.Popen([PY, '-c', 'raise SystemExit(7)'], cwd=SOURCE)
        supervision_loop(child, {}, failure=True)
