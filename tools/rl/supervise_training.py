"""Detached training supervision with atomic health/progress receipts."""
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


def utc(value=None):
    return dt.datetime.fromtimestamp(value or time.time(), dt.timezone.utc).isoformat()


def atomic(path, record):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(record, indent=2), encoding='utf-8')
    tmp.replace(path)


def birth(pid):
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]


def main():
    manifest_path = Path(sys.argv[1])
    manifest = json.loads(manifest_path.read_text())
    output = Path(manifest['output'])
    status_path = output/'status.json'
    status = json.loads(status_path.read_text())
    status.update(supervisor_pid=os.getpid(), supervisor_birth=birth(os.getpid()), supervisor_started_at=utc())
    child = None
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, resource.getrlimit(resource.RLIMIT_NOFILE)[1]))
    try:
        with (output/'stdout.log').open('ab',buffering=0) as out, (output/'stderr.log').open('ab',buffering=0) as err:
            child = subprocess.Popen(manifest['command'], cwd=manifest['cwd'], env=manifest['environment'],
                                     stdout=out, stderr=err, start_new_session=True)
            status.update(child_pid=child.pid, child_birth=birth(child.pid), child_started_at=utc())
            while True:
                now = time.time()
                status.update(heartbeat_at=utc(now), source_updated_at=utc(now), experiment_state=status['status'])
                health_path = output/'training_health.json'
                if health_path.exists():
                    health = json.loads(health_path.read_text())
                    old = status.get('completed',0)
                    status.update(completed=health['timesteps'], health=health,
                                  completed_episodes=health['completed_episodes'])
                    if health.get('worker_ready') and status['status']=='starting':
                        status.update(status='running', last_progress_at=utc(now),
                                      progress_deadline_at=utc(now+manifest['progress_timeout_seconds']))
                    if health['timesteps'] > old:
                        status.update(last_progress_at=utc(now), progress_deadline_at=utc(now+manifest['progress_timeout_seconds']))
                code = child.poll()
                if code is not None:
                    status.update(exit_code=code, child_exited_at=utc(now))
                    if code:
                        status.update(status='failed', error=(output/'stderr.log').read_text(errors='replace')[-4000:])
                    else:
                        model_path = output/'final_model.zip'
                        with zipfile.ZipFile(model_path) as z:
                            metadata = json.loads(z.read('data'))
                        if metadata['num_timesteps'] != manifest['budget']:
                            raise RuntimeError('Final checkpoint does not match exact budget')
                        if not status.get('health',{}).get('normal_training'):
                            raise RuntimeError('No finite normal PPO update recorded')
                        status.update(status='complete', completed=manifest['budget'],
                                      final_model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                                      last_progress_at=utc(now))
                    status['experiment_state'] = status['status']
                    atomic(status_path,status)
                    return
                status['experiment_state'] = status['status']
                atomic(status_path,status)
                time.sleep(manifest.get('source_interval_seconds',30))
    except BaseException:
        status.update(status='failed', experiment_state='failed', error=traceback.format_exc(),
                      child_alive=child is not None and child.poll() is None,
                      heartbeat_at=utc(), source_updated_at=utc())
        atomic(status_path,status)
        raise


if __name__=='__main__':
    main()
