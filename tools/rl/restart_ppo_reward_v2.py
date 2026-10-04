"""Replace the explicitly identified old PPO on Linux, preserving its artifacts."""
import argparse
from dataclasses import asdict, fields
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from stable_baselines3.common.save_util import load_from_zip_file

from panda_cable_grasp.rl.environment import RLConfig
from panda_cable_grasp.rl.workspace import WORKSPACE_SCENE_FAMILIES


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def identity(pid):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()
        return int(stat[1]), stat[19]
    except (OSError, IndexError):
        return None


def process_tree(pid):
    records = {int(p.name): identity(int(p.name)) for p in Path('/proc').iterdir() if p.name.isdigit()}
    descendants = {pid}
    while True:
        added = {p for p, item in records.items() if item and item[0] in descendants}
        if added <= descendants:
            return {p: records[p][1] for p in descendants if records.get(p)}
        descendants |= added


def stop_owned_tree(pid, source_run):
    cmdline = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\x00', b' ').decode()
    if str(source_run) not in cmdline or 'train_targetnode_full6_workspace' not in cmdline:
        raise RuntimeError('Refusing to stop a process outside the selected old PPO run')
    os.kill(pid, signal.SIGSTOP)
    tree = process_tree(pid)
    for child, birth in tree.items():
        if identity(child) and identity(child)[1] == birth:
            try:
                os.kill(child, signal.SIGTERM)
                os.kill(child, signal.SIGCONT)
            except ProcessLookupError:
                pass
    deadline = time.monotonic()+15.
    while time.monotonic() < deadline:
        alive = [p for p, birth in tree.items() if identity(p) and identity(p)[1] == birth]
        if not alive:
            break
        time.sleep(.2)
    killed = []
    for child, birth in tree.items():
        if identity(child) and identity(child)[1] == birth:
            try:
                os.kill(child, signal.SIGKILL)
                killed.append(child)
            except ProcessLookupError:
                pass
    return dict(parent_pid=pid, command=cmdline, descendants=sorted(tree), forced_kill=killed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stop-pid', type=int, required=True)
    parser.add_argument('--expected-commit', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=root, text=True).strip()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip()
    if branch != 'dev' or commit != args.expected_commit or dirty:
        raise RuntimeError('Launch requires the expected clean dev commit')
    source = args.source_run.resolve()
    config_path = source/'training_config_eval13.json'
    old = json.loads(config_path.read_text())
    mask = source/'workspace/workspace.npz'
    heldout_bank = Path(old['evaluation_bank'])
    training_bank = root/'outputs/motion_banks'
    if not mask.is_file() or not training_bank.is_dir() or not heldout_bank.is_dir():
        raise FileNotFoundError('Missing original workspace or replay banks')
    # Select a fully written checkpoint before stopping the existing process.
    selected = None
    for checkpoint in sorted((source/'checkpoints').glob('step_*.zip'), reverse=True):
        try:
            data, params, _ = load_from_zip_file(checkpoint, device='cpu',
                custom_objects={'learning_rate':0., 'lr_schedule':lambda _:0., 'clip_range':.2})
            selected = checkpoint
            break
        except (OSError, ValueError, EOFError):
            continue
    if selected is None:
        raise RuntimeError('No readable completed source checkpoint')
    start_steps = int(data['num_timesteps'])
    remaining = int(old['cumulative_target_timesteps'])-start_steps
    if remaining <= 0:
        raise RuntimeError('Original training budget already exhausted')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    reference = output/'reference_training_config.json'
    write(reference, old)
    accepted = {field.name for field in fields(RLConfig)}
    revised = {k:v for k,v in old['rl_config'].items() if k in accepted}
    revised.update(target_reward_profile='tolerance_progress_v2', target_node_tolerance=2,
                   target_position_reward_radius=4, target_wrong_event_penalty=2., target_wrong_event_cap=8.)
    actual = asdict(RLConfig(**revised))
    # A changed objective starts a new evaluation history in the SAME current
    # curriculum stage; historical qualification rates do not carry forward.
    old_stage = json.loads((source/'curriculum_state.json').read_text())['stage']
    (output/'evaluation').mkdir()
    write(output/'evaluation/workspace_curriculum.json',
          dict(stage=old_stage, history=[], scene_names=WORKSPACE_SCENE_FAMILIES['legacy_ppo_six']))
    command = [sys.executable, '-u', '-m', 'panda_cable_grasp.rl.train',
        '--resume', str(selected), '--resume-legacy-full6',
        '--rl-config-file', str(reference), '--target-node-task',
        '--target-reward-profile', 'tolerance_progress_v2', '--target-node-tolerance', '2',
        '--robot', 'nero', '--action-mode', 'task_space_vertical_down', '--observation-mode', 'state',
        '--controller-profile', 'enhanced6_bounded_dls_v1',
        '--target-workspace', str(mask), '--workspace-mask-key', 'interior',
        '--workspace-scene-family', 'legacy_ppo_six', '--workspace-curriculum',
        '--workspace-entry-seconds', '8', '--workspace-reset-attempts', '50',
        '--replay-bank-dir', str(training_bank), '--eval-replay-bank-dir', str(heldout_bank),
        '--eval-seed', '20260804', '--eval-seed-pool-size', '50',
        '--workers', str(old['workers']), '--eval-workers', str(old.get('eval_workers',13)),
        '--eval-episodes', str(old['promotion']['episodes']),
        '--eval-confirmation-episodes', str(old['promotion']['episodes']),
        '--eval-freq', str(old['promotion']['eval_every']),
        '--checkpoint-steps', str(old['promotion']['eval_every']),
        '--n-steps', str(old['n_steps']), '--batch-size', str(old['batch_size']),
        '--n-epochs', str(old['n_epochs']), '--timesteps', str(remaining),
        '--seed', str(old['seed']), '--device', old['training_device'], '--torch-threads', '1',
        '--episode-seconds', str(old['episode_seconds']), '--disable-singularity-avoidance',
        '--enable-safety-penalties', '--learning-rate-final', '.00003',
        '--entropy-coef-final', '.001', '--target-kl', '.03', '--output', str(output)]
    # Persist a recoverable version snapshot plus all configuration differences.
    subprocess.run(['git', 'archive', '--format=tar', '-o', str(output/'code_snapshot.tar'), 'HEAD'], cwd=root, check=True)
    launch = dict(branch=branch, commit=commit, source_run=str(source), source_checkpoint=str(selected),
        source_checkpoint_sha256=hashlib.sha256(selected.read_bytes()).hexdigest(),
        starting_steps=start_steps, starting_updates=int(data['_n_updates']),
        source_learning_rate=float(params['policy.optimizer']['param_groups'][0]['lr']),
        source_entropy_coef=float(data['ent_coef']), source_stage=old_stage,
        workspace_sha256=hashlib.sha256(mask.read_bytes()).hexdigest(), command=command,
        rl_config=actual, ignored_legacy_fields=sorted(set(old['rl_config'])-accepted),
        changed_existing_fields={k:dict(old=v,new=actual[k]) for k,v in old['rl_config'].items()
                                 if k in actual and actual[k] != v},
        additional_changes=['formal bounded DLS replaces frozen unconstrained DLS',
                            'physical_holding_v2 replaces command-dependent grasp judgement',
                            'formal best-model confirmation callback and independent evaluation history'],
        preserved=['old six scenes', 'original workspace', 'training and held-out replay banks',
                   '24 train / 13 evaluation workers', '109 state / 5 actions',
                   'optimizer and latest saved policy', 'remaining original 20M budget',
                   'source learning-rate and entropy continued to original endpoints',
                   'no table execution guard added'], created_at=time.time())
    write(output/'launch_manifest.json', launch)
    stopped = stop_owned_tree(args.stop_pid, source)
    write(output/'previous_training_stopped.json', stopped)
    env = dict(os.environ, PYTHONPATH=str(root/'src'), OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', MUJOCO_GL='egl')
    with (output/'train.log').open('w') as log:
        process = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    launch.update(pid=process.pid, launched_at=time.time())
    write(output/'launch_manifest.json', launch)
    print(json.dumps(dict(pid=process.pid, output=str(output), commit=commit,
                          checkpoint=str(selected), starting_steps=start_steps), indent=2))


if __name__ == '__main__':
    main()
