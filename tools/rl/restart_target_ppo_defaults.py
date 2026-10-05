"""Versioned continuation into the validated target PPO defaults on Linux."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from stable_baselines3.common.save_util import load_from_zip_file
from panda_cable_grasp.rl.target_defaults import TARGET_PPO_DEFAULTS
from panda_cable_grasp.rl.workspace import WORKSPACE_SCENE_FAMILIES
from panda_cable_grasp.rl.train import parse_args as training_args, rl_config_from_args
from restart_ppo_reward_v2 import write, identity, process_tree


def stop_selected(pid, source):
    argv=[v.decode() for v in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if v]
    if 'panda_cable_grasp.rl.train' not in argv or str(source) not in argv:
        raise RuntimeError('PID is not the selected formal PPO run')
    os.kill(pid,signal.SIGSTOP)
    tree=process_tree(pid)
    for child,birth in tree.items():
        if identity(child) and identity(child)[1]==birth:
            try:
                os.kill(child,signal.SIGTERM)
                os.kill(child,signal.SIGCONT)
            except ProcessLookupError:
                pass
    deadline=time.monotonic()+15
    while time.monotonic()<deadline:
        if not any(identity(p) and identity(p)[1]==birth for p,birth in tree.items()):
            break
        time.sleep(.2)
    killed=[]
    for child,birth in tree.items():
        if identity(child) and identity(child)[1]==birth:
            try:
                os.kill(child,signal.SIGKILL)
                killed.append(child)
            except ProcessLookupError:
                pass
    return dict(parent_pid=pid,argv=argv,descendants=sorted(tree),forced_kill=killed)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--expected-commit',required=True)
    parser.add_argument('--worker-affinity', choices=('none','numa'))
    parser.add_argument('--workers', type=int,
                        help='explicitly override the number of training environments')
    parser.add_argument('--source-stop-record',type=Path,
                        help='validated stop record when the user stopped the source before restart')
    opts=parser.parse_args()
    if opts.workers is not None and opts.workers < 1:
        parser.error('--workers must be positive')
    root=Path(__file__).resolve().parents[2]
    git=lambda *argv:subprocess.check_output(['git',*argv],cwd=root,text=True).strip()
    if git('branch','--show-current')!='dev' or git('rev-parse','HEAD')!=opts.expected_commit or git('status','--porcelain'):
        raise RuntimeError('Expected clean dev commit required')
    source=opts.source_run.resolve()
    output=opts.output.resolve()
    old=json.loads((source/'launch_manifest.json').read_text())
    cfg=json.loads((source/'training_config.json').read_text())
    candidates=list((source/'checkpoints').glob('*.zip'))
    if (source/'final_model.zip').is_file():
        candidates.append(source/'final_model.zip')
    # A short run may not have reached its first 200k checkpoint yet.
    fallback=Path(cfg['resume_checkpoint'])
    if fallback.is_file():
        candidates.append(fallback)
    valid=[]
    for checkpoint in candidates:
        try:
            data,params,_=load_from_zip_file(checkpoint,device='cpu',custom_objects={
                'learning_rate':0.,'lr_schedule':lambda _:0.,'clip_range':.2})
            valid.append((int(data['num_timesteps']),checkpoint,data,params))
        except (OSError,ValueError,EOFError):
            continue
    if not valid:
        raise RuntimeError('No completed readable checkpoint')
    start,checkpoint,data,params=max(valid,key=lambda item:item[0])
    remaining=int(cfg['cumulative_target_timesteps'])-start
    if remaining<=0:
        raise RuntimeError('No remaining budget')
    stopped=None
    if opts.source_stop_record:
        stop_path=opts.source_stop_record.resolve()
        if stop_path.parent!=source:
            raise RuntimeError('Stop record must belong to the source run')
        stopped=json.loads(stop_path.read_text())
        if (stopped['parent_pid']!=old['pid'] or str(source) not in stopped['argv']
                or 'panda_cable_grasp.rl.train' not in stopped['argv'] or identity(old['pid'])):
            raise RuntimeError('Stop record does not match an absent source training process')
    elif not identity(old['pid']):
        raise RuntimeError('Selected process is absent; provide its validated stop record')
    if not Path(cfg['target_workspace']['path']).is_file():
        raise FileNotFoundError('Workspace missing')
    for name in ('replay_bank_dir','eval_replay_bank_dir'):
        if not Path(cfg[name]).is_dir():
            raise FileNotFoundError(name)
    output.mkdir(parents=True,exist_ok=False)
    reference=output/'reference_training_config.json'
    write(reference,cfg)
    previous_state=json.loads((source/'evaluation/workspace_curriculum.json').read_text())
    same_family=cfg['workspace_scene_family']==TARGET_PPO_DEFAULTS['workspace_scene_family']
    state=(dict(previous_state) if same_family else
        dict(stage=previous_state['stage'],history=[],
             scene_names=WORKSPACE_SCENE_FAMILIES[TARGET_PPO_DEFAULTS['workspace_scene_family']]))
    (output/'evaluation').mkdir()
    write(output/'evaluation/workspace_curriculum.json',state)
    write(output/'source_curriculum_state.json',previous_state)
    command=list(old['command'])
    command[0]=sys.executable
    if opts.workers is not None:
        command[command.index('--workers')+1]=str(opts.workers)
    if opts.worker_affinity is not None:
        if '--worker-affinity' in command:
            command[command.index('--worker-affinity')+1]=opts.worker_affinity
        else:
            command.extend(['--worker-affinity',opts.worker_affinity])
    for option,value in [('--resume',str(checkpoint)),('--rl-config-file',str(reference)),
                         ('--timesteps',str(remaining)),('--output',str(output))]:
        command[command.index(option)+1]=value
    # Let the formal parser apply the central default instead of copying the
    # obsolete scene/guard settings from a historical launcher.
    for option in ('--robot','--action-mode','--controller-profile','--target-reward-profile',
                   '--workspace-scene-family'):
        if option in command:
            index=command.index(option)
            del command[index:index+2]
    for option in ('--table-execution-guard','--no-table-execution-guard',
                   '--disable-table-finger-collision-filter','--enable-table-finger-collision-filter'):
        if option in command:
            command.remove(option)
    previous_argv=sys.argv
    try:
        sys.argv=['train',*command[4:]]
        actual=training_args()
        if any(getattr(actual,key)!=value for key,value in TARGET_PPO_DEFAULTS.items()):
            raise RuntimeError('Resolved target PPO defaults do not match the intended configuration')
        if rl_config_from_args(actual).target_reward_profile!='tolerance_progress_v2':
            raise RuntimeError('Reward v2 missing')
    finally:
        sys.argv=previous_argv
    subprocess.run(['git','archive','--format=tar','-o',str(output/'code_snapshot.tar'),'HEAD'],cwd=root,check=True)
    manifest=dict(branch='dev',commit=opts.expected_commit,source_run=str(source),
        source_checkpoint=str(checkpoint),source_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        checkpoint_fallback_used=checkpoint==fallback and not (source/'checkpoints'/checkpoint.name).is_file(),
        starting_steps=start,starting_updates=int(data['_n_updates']),
        source_learning_rate=float(params['policy.optimizer']['param_groups'][0]['lr']),
        source_entropy_coef=float(data['ent_coef']),source_stage=state['stage'],command=command,
        resolved_target_ppo_defaults=TARGET_PPO_DEFAULTS,
        changes=dict(workspace_scene_family=dict(old=cfg['workspace_scene_family'],new=actual.workspace_scene_family),
                     train_workers=dict(old=cfg['workers'],new=actual.workers),
                     rollout_transitions=dict(old=cfg['workers']*cfg['n_steps'],new=actual.workers*actual.n_steps),
                     contact_read_cache_enabled=dict(old=cfg.get('contact_read_cache_enabled',False),new=actual.contact_read_cache),
                     worker_affinity=dict(old=cfg.get('worker_affinity','none'),new=actual.worker_affinity),
                     table_finger_collision_filter_enabled=dict(old=cfg.get('table_finger_collision_filter_enabled',True),
                                                               new=not actual.disable_table_finger_collision_filter),
                     table_execution_guard_enabled=dict(old=cfg['table_execution_guard_enabled'],new=actual.table_execution_guard)),
        preserved='reward v2 + release fix, controller, original physical limits, workspace, banks, optimizer, evaluation workers, n_steps, schedules and cumulative 20M budget',
        curriculum_history=('same scene family; stage and history preserved' if same_family else
                            'same stage; old scene-family qualification history cleared'),created_at=time.time())
    write(output/'launch_manifest.json',manifest)
    write(output/'previous_training_stopped.json',stopped if stopped is not None else stop_selected(int(old['pid']),source))
    env=dict(os.environ,PYTHONPATH=str(root/'src'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl')
    with (output/'train.log').open('w') as log:
        process=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    manifest.update(pid=process.pid,launched_at=time.time())
    write(output/'launch_manifest.json',manifest)
    print(json.dumps(dict(pid=process.pid,output=str(output),commit=opts.expected_commit,
        checkpoint=str(checkpoint),starting_steps=start,defaults=TARGET_PPO_DEFAULTS),indent=2),flush=True)


if __name__=='__main__':
    main()
