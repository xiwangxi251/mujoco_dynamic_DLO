"""Speculate exactly one next reset in a Linux copy-on-write child.

The live environment always performs its own reset. A prediction is reused only
when the complete reset model/state, drive parameters and RNG states match.
Speculation never advances the live RNG, changes episode length, or batches away
terminal transitions. Curriculum/seed changes safely fall back to the oracle.
"""
import ctypes
import hashlib
import os
import pickle
import signal
import time
from multiprocessing.connection import Connection

import mujoco
import numpy as np


VERSION = 'exact_next_reset_prefetch_v1'


def initial_fingerprint(env):
    base = env.base_env
    kind = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(base.model, kind))
    mujoco.mj_getState(base.model, base.data, state, kind)
    drive = {}
    for name, value in base.__dict__.items():
        if isinstance(value, np.random.Generator):
            drive[name] = value.bit_generator.state
        elif (name.startswith(('_replay_', '_shape_', '_initial_shape_', '_rigid_'))
              or name in ('phase_offset', 'spatial_phase', 'target_body_id')):
            if isinstance(value, (np.ndarray, str, int, float, bool, tuple, type(None))):
                drive[name] = value
    # The replay driver reads an absent suspension-history flag as False.
    # It creates that attribute on the first physical step, even in static
    # scenes. Canonicalize the documented default without ignoring True.
    drive.setdefault('_replay_was_suspended', False)
    payload = (base.model, state, base.config, drive,
               env.np_random.bit_generator.state, env._target_object_slice())
    return hashlib.sha256(pickle.dumps(payload, protocol=5)).hexdigest()


class ResetPrefetch:
    def __init__(self):
        if not hasattr(os, 'fork'):
            raise RuntimeError('Future reset prefetch requires Linux fork')
        self.pid = None
        self.connection = None
        self.started = None
        self.hits = self.misses = self.errors = 0
        self.wait_seconds = 0.
        self.last_reason = 'cold_start'

    def start(self, env):
        if self.pid is not None:
            raise RuntimeError('Only one speculative reset per worker is allowed')
        # Fork only the single-threaded simulation worker, never the learner.
        if len(os.listdir('/proc/self/task')) != 1:
            raise RuntimeError('Future reset prefetch requires a single-threaded worker')
        read_fd, write_fd = os.pipe()
        parent = os.getpid()
        try:
            pid = os.fork()
        except OSError:
            os.close(read_fd)
            os.close(write_fd)
            raise
        if pid == 0:
            try:
                os.close(read_fd)
                libc = ctypes.CDLL(None)
                libc.prctl(1, signal.SIGKILL, 0, 0, 0)  # PR_SET_PDEATHSIG
                libc.prctl(15, b'ppo_future_pref', 0, 0, 0)
                if os.getppid() != parent:
                    os._exit(1)
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
                signal.alarm(120)
                # Do not keep the training worker's command sockets alive.
                for name in os.listdir('/proc/self/fd'):
                    fd = int(name)
                    if fd > 2 and fd != write_fd:
                        try:
                            os.close(fd)
                        except OSError:
                            pass
                connection = Connection(write_fd, readable=False)
                from .future_geometry import precompute
                try:
                    env.reset()
                    fingerprint = initial_fingerprint(env)
                    frames, seconds, mapping = precompute(env.base_env, env._target_object_slice())
                    result = dict(fingerprint=fingerprint, frames=frames, seconds=seconds, mapping=mapping)
                except BaseException as error:
                    result = dict(error=f'{type(error).__name__}: {error}')
                connection.send_bytes(pickle.dumps(result, protocol=5))
                connection.close()
                os._exit(0)
            except BaseException:
                os._exit(1)
        os.close(write_fd)
        self.connection = Connection(read_fd, writable=False)
        self.pid, self.started = pid, time.monotonic()

    def take(self, env, eligible=True):
        if self.pid is None:
            self.misses += 1
            self.last_reason = 'cold_start'
            return None
        began = time.monotonic()
        result = None
        try:
            if not eligible:
                self.last_reason = 'explicit_reset'
            elif self.connection.poll(max(0., 120. - (began-self.started))):
                result = pickle.loads(self.connection.recv_bytes(131072))
                if 'error' in result:
                    self.errors += 1
                    self.last_reason = result['error']
                    result = None
                elif result['fingerprint'] != initial_fingerprint(env):
                    self.last_reason = 'reset_fingerprint_changed'
                    result = None
                else:
                    self.last_reason = 'exact_match'
            else:
                self.last_reason = 'prefetch_timeout'
                self.errors += 1
        except (EOFError, OSError) as error:
            self.errors += 1
            self.last_reason = type(error).__name__
        finally:
            self.wait_seconds += time.monotonic()-began
            self.close()
        if result is None:
            self.misses += 1
        else:
            self.hits += 1
        return result

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if self.pid is not None:
            pid = self.pid
            self.pid = None
            try:
                done, _ = os.waitpid(pid, os.WNOHANG)
                if not done:
                    os.kill(pid, signal.SIGKILL)
                    os.waitpid(pid, 0)
            except (ProcessLookupError, ChildProcessError):
                pass

    def status(self):
        return dict(version=VERSION, hits=self.hits, misses=self.misses,
                    errors=self.errors, wait_seconds=self.wait_seconds,
                    last_reason=self.last_reason, pending_pid=self.pid)
