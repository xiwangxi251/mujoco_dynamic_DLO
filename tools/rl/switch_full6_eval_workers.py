"""Switch at a newly saved evaluation checkpoint; roll back if resume fails."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import psutil

RUN = Path(__file__).resolve().parent


def save(value):
    path = RUN / 'eval13_switch_status.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(updated_at=time.time(), **value), indent=2))
    temp.replace(path)


def read(name):
    return json.loads((RUN / name).read_text())


def main():
    assert read('eval13_preflight_verification.json')['passed']
    assert read('eval13_preflight_verification.json')['eval_episodes'] == 26
    old_pid = int((RUN / 'train.pid').read_text())
    old = psutil.Process(old_pid)
    assert 'train_targetnode_full6_workspace.py' in ' '.join(old.cmdline())
    old_created = old.create_time()
    last = read('curriculum_state.json')['steps']
    save(dict(phase='waiting_for_next_checkpoint', old_pid=old_pid,
              after_steps=last, eval_workers=13, eval_episodes=26))
    deadline = time.monotonic() + 10800
    while time.monotonic() < deadline:
        if not old.is_running() or old.create_time() != old_created:
            raise RuntimeError('Original trainer stopped before handoff')
        pending = RUN / 'evaluation_in_progress.json'
        if pending.exists():
            state = json.loads(pending.read_text())
            if state['steps'] > last and state.get('eval_workers', 8) == 8:
                checkpoint = RUN / f"checkpoints/step_{state['steps']:09d}.zip"
                if checkpoint.is_file():
                    break
        time.sleep(1)
    else:
        raise TimeoutError('No new evaluation checkpoint within three hours')
    # The old parent is waiting for evaluation, before optimizing this rollout.
    os.kill(old_pid, signal.SIGSTOP)
    if (not (RUN/'evaluation_in_progress.json').exists()
            or read('evaluation_in_progress.json')['steps'] != state['steps']):
        os.kill(old_pid, signal.SIGCONT)
        raise RuntimeError('Evaluation completed during handoff; original trainer kept running')
    children = old.children(recursive=True)
    frozen = [old] + children
    for process in children:
        if process.is_running():
            os.kill(process.pid, signal.SIGSTOP)
    new = None
    try:
        save(dict(phase='starting_resume', old_pid=old_pid,
                  checkpoint=str(checkpoint), eval_workers=13, eval_episodes=26))
        logfile = (RUN / 'train_eval13.log').open('a')
        new = subprocess.Popen([sys.executable, '-u',
            str(RUN/'train_targetnode_full6_workspace_eval13.py'),
            '--workers', '24', '--timesteps', '20000000',
            '--eval-workers', '13', '--eval-episodes', '26', '--resume', str(checkpoint)],
            cwd=RUN, stdout=logfile, stderr=subprocess.STDOUT,
            start_new_session=True, env=dict(os.environ, FULL6_PPO_RUN=str(RUN)))
        logfile.close()
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            if new.poll() is not None:
                raise RuntimeError(f'Resumed trainer exited: {new.returncode}')
            resumed_path = RUN / 'RESUMED_eval13.json'
            if resumed_path.exists():
                resumed = read('RESUMED_eval13.json')
                progress = read('evaluation_in_progress.json')
                if (resumed['pid'] == new.pid and progress.get('eval_workers') == 13
                        and progress.get('eval_episodes') == 26
                        and progress['steps'] == state['steps']):
                    assert resumed['optimizer_state_entries'] > 0
                    assert resumed['policy_parameters_finite']
                    break
            time.sleep(1)
        else:
            raise TimeoutError('Resume did not verify within four minutes')
        # The new model has restored the exact checkpoint and optimizer.
        # Retire only the original trainer tree after verified replacement.
        for process in reversed(frozen):
            if process.is_running():
                process.kill()
        psutil.wait_procs(frozen, timeout=10)
        (RUN/'train.pid').write_text(str(new.pid)+'\n')
        save(dict(phase='switched', old_pid=old_pid, new_pid=new.pid,
                  starting_steps=resumed['starting_steps'], checkpoint=str(checkpoint),
                  eval_workers=13, eval_episodes=26,
                  optimizer_state_restored=True, pending_rollout_discarded=True))
    except BaseException:
        if new is not None and new.poll() is None:
            os.killpg(new.pid, signal.SIGKILL)
        for process in frozen:
            if process.is_running():
                os.kill(process.pid, signal.SIGCONT)
        raise


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        save(dict(phase='failed_original_kept_if_alive', traceback=traceback.format_exc()))
        raise
