"""Detached C training supervisor; progress counts additional samples."""
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
    return dt.datetime.fromtimestamp(time.time() if value is None else value, dt.timezone.utc).isoformat()


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def birth(pid):
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]


def main():
    m = json.loads(Path(sys.argv[1]).read_text())
    out = Path(m['output'])
    path = out/'status.json'
    s = json.loads(path.read_text())
    s.update(supervisor_pid=os.getpid(), supervisor_birth=birth(os.getpid()))
    child = None
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, resource.getrlimit(resource.RLIMIT_NOFILE)[1]))
    try:
        with (out/'stdout.log').open('ab', buffering=0) as stdout, (out/'stderr.log').open('ab', buffering=0) as stderr:
            child = subprocess.Popen(m['command'], cwd=m['cwd'], env=m['environment'],
                stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL, start_new_session=True)
            s.update(child_pid=child.pid, child_birth=birth(child.pid), child_started_at=utc())
            while True:
                now = time.time()
                s.update(source_updated_at=utc(now), heartbeat_at=utc(now))
                if m.get('purpose') == 'sampler_benchmark' and (out/'worker_ready.json').exists():
                    s['benchmark_worker_ready'] = json.loads((out/'worker_ready.json').read_text())
                    if s['status'] == 'starting':
                        s.update(status='running', last_progress_at=utc(now), progress_deadline_at=utc(now+1800))
                if (out/'training_health.json').exists():
                    h = json.loads((out/'training_health.json').read_text())
                    old = s['completed']
                    s.update(health=h, completed=h['c_added_steps'], model_global_steps=h['timesteps'],
                             completed_episodes=h['completed_episodes'])
                    if h.get('worker_ready') and s['status'] == 'starting':
                        s.update(status='running', last_progress_at=utc(now), progress_deadline_at=utc(now+1800))
                    if h['c_added_steps'] > old:
                        s.update(last_progress_at=utc(now), progress_deadline_at=utc(now+1800))
                if (out/'speed_curriculum_status.json').exists():
                    s['curriculum'] = json.loads((out/'speed_curriculum_status.json').read_text())
                code = child.poll()
                if code is not None:
                    s.update(exit_code=code, child_exited_at=utc(now))
                    if code:
                        s.update(status='failed', error=(out/'stderr.log').read_text(errors='replace')[-5000:])
                    else:
                        if m.get('purpose') == 'sampler_benchmark':
                            report = json.loads((out/'benchmark.json').read_text())
                            assert report['status'] == 'complete' and report['sampled_transitions'] == m['budget']
                            assert report['finite'] and report['policy_unchanged'] and report['formal_samples'] == 0
                            s.update(status='complete', experiment_state='complete', completed=m['budget'],
                                     benchmark=report, last_progress_at=utc(now))
                            atomic(path, s)
                            return
                        model = out/'final_model.zip'
                        with zipfile.ZipFile(model) as z:
                            data = json.loads(z.read('data'))
                        assert data['num_timesteps'] == 16997280+m['budget']
                        assert s.get('health', {}).get('normal_training') and s['health']['parameters_finite']
                        s.update(status='complete', completed=m['budget'], final_model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                                 last_progress_at=utc(now))
                    s['experiment_state'] = s['status']
                    atomic(path, s)
                    return
                s['experiment_state'] = s['status']
                atomic(path, s)
                time.sleep(10)
    except BaseException:
        s.update(status='failed', experiment_state='failed', error=traceback.format_exc(),
                 child_alive=child is not None and child.poll() is None, source_updated_at=utc(), heartbeat_at=utc())
        atomic(path, s)
        raise


if __name__ == '__main__':
    main()
