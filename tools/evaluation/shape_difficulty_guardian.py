"""Detached supervisor for the overnight coordinator.

It refreshes status.json independently of the coordinator and records a
terminal failed status when the child exits before reporting completion.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "shape_difficulty_overnight_20261010"
STATUS = OUT / "status.json"
STATUS_LOCK = OUT / "status.json.lock"
COORDINATOR = ROOT / "tools" / "evaluation" / "shape_difficulty_overnight.py"
STALL_TIMEOUT_SECONDS = 60 * 60


def stamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def deadline_after(seconds=STALL_TIMEOUT_SECONDS):
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_status():
    try:
        return json.loads(STATUS.read_text(encoding="utf-8-sig"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def merge_status(updates):
    """Serialize status updates shared with the coordinator process."""
    OUT.mkdir(parents=True, exist_ok=True)
    with STATUS_LOCK.open("a+b") as lock:
        lock.seek(0, os.SEEK_END)
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            current = read_status()
            current.update(updates)
            tmp = STATUS.with_suffix(".json.guardian.tmp")
            tmp.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp, STATUS)
        finally:
            lock.seek(0)
            if os.name == "nt":
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def log_event(log, event, **fields):
    record = {"at": stamp(), "event": event, **fields}
    log.write((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))
    log.flush()


def parse_deadline(value):
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except (AttributeError, ValueError):
        return time.time() + STALL_TIMEOUT_SECONDS


def terminate_process_tree(pid):
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=False)
    else:
        try:
            os.killpg(os.getpgid(pid), 9)
        except (ProcessLookupError, PermissionError):
            pass


def write_status(value):
    """Compatibility wrapper: merge supplied fields rather than replacing peers."""
    merge_status(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-episodes", type=int, default=0)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    guardian_log = (OUT / "guardian.log").open("ab", buffering=0)
    initial = read_status()
    write_status(dict(experiment="shape_difficulty_overnight_20261010", status="running",
                      stage="guardian_starting", updated_at=stamp(), heartbeat_at=stamp(),
                      progress_deadline=deadline_after(), progress_timeout_minutes=60,
                      guardian_pid=os.getpid(), guardian_heartbeat_at=stamp()))
    log_event(guardian_log, "guardian_start", pid=os.getpid(), python=sys.executable,
              root=str(ROOT), workers=args.workers, max_episodes=args.max_episodes)
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    env["OMP_NUM_THREADS"] = env["OPENBLAS_NUM_THREADS"] = env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["PYTHONPYCACHEPREFIX"] = str(ROOT / "outputs" / "tmp" / "shape_difficulty_overnight_20261010" / "pycache")
    log_out = (OUT / "coordinator.stdout.log").open("ab", buffering=0)
    log_err = (OUT / "coordinator.stderr.log").open("ab", buffering=0)
    command = [sys.executable, "-B", "-X", "utf8", str(COORDINATOR), "--workers", str(args.workers)]
    if args.max_episodes:
        command += ["--max-episodes", str(args.max_episodes)]
    child = None
    try:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log_out, stderr=log_err)
        started = time.time()
        last_stage = None
        last_completed = None
        log_event(guardian_log, "coordinator_started", coordinator_pid=child.pid,
                  command=command, cwd=str(ROOT))
        while child.poll() is None:
            state = read_status()
            stage = state.get("stage")
            completed = state.get("completed_episodes")
            progress_deadline = state.get("progress_deadline") or deadline_after()
            if stage != last_stage or completed != last_completed:
                last_stage, last_completed = stage, completed
                progress_deadline = deadline_after()
                state["progress_deadline"] = progress_deadline
                state["progress_timeout_minutes"] = STALL_TIMEOUT_SECONDS // 60
            now = stamp()
            write_status(dict(experiment="shape_difficulty_overnight_20261010",
                guardian_pid=os.getpid(), coordinator_pid=child.pid,
                guardian_heartbeat_at=now, heartbeat_at=now, updated_at=now,
                guardian_uptime_seconds=int(time.time() - started),
                progress_deadline=progress_deadline,
                progress_timeout_minutes=STALL_TIMEOUT_SECONDS // 60))
            if time.time() > parse_deadline(progress_deadline):
                log_event(guardian_log, "progress_deadline_expired", coordinator_pid=child.pid,
                          stage=stage, completed_episodes=completed, deadline=progress_deadline)
                terminate_process_tree(child.pid)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                write_status(dict(experiment="shape_difficulty_overnight_20261010", status="failed",
                    stage="progress_deadline_expired", updated_at=stamp(), heartbeat_at=stamp(),
                    guardian_pid=os.getpid(), coordinator_pid=child.pid,
                    message="Coordinator made no stage or episode progress before progress_deadline.",
                    failure_detected_by="shape_difficulty_guardian.py"))
                break
            time.sleep(10)
        code = child.returncode
        state = read_status()
        successful_terminal = state.get("status") == "complete"
        if state.get("stage") == "progress_deadline_expired":
            log_event(guardian_log, "coordinator_stopped_by_progress_deadline", exit_code=code)
        elif code != 0 or not successful_terminal:
            log_event(guardian_log, "coordinator_failed", exit_code=code,
                      status=state.get("status"), stage=state.get("stage"))
            state.update(experiment="shape_difficulty_overnight_20261010", status="failed",
                stage="guardian_detected_exit", updated_at=stamp(), heartbeat_at=stamp(),
                guardian_pid=os.getpid(), coordinator_pid=child.pid, coordinator_exit_code=code,
                message=f"Coordinator exited without a successful terminal status (exit code {code}).",
                failure_detected_by="shape_difficulty_guardian.py")
            write_status(state)
        else:
            state.update(guardian_pid=os.getpid(), coordinator_pid=child.pid,
                         guardian_exit_code=0, guardian_finished_at=stamp(),
                         heartbeat_at=stamp(), updated_at=stamp())
            write_status(state)
            log_event(guardian_log, "coordinator_complete", exit_code=code,
                      stage=state.get("stage"), completed=state.get("completed_episodes"))
        return int(code or 0)
    except BaseException as error:
        if child is not None and child.poll() is None:
            terminate_process_tree(child.pid)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        state = read_status()
        state.update(experiment="shape_difficulty_overnight_20261010", status="failed",
            stage="guardian_failed", updated_at=stamp(), heartbeat_at=stamp(),
            guardian_pid=os.getpid(), coordinator_pid=(child.pid if child else None),
            message=repr(error), traceback=traceback.format_exc(),
            failure_detected_by="shape_difficulty_guardian.py")
        write_status(state)
        log_event(guardian_log, "guardian_exception", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        log_out.close()
        log_err.close()
        guardian_log.close()


if __name__ == "__main__":
    raise SystemExit(main())
