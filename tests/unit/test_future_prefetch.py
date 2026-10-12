"""Process isolation and invalidation gates for the optional Linux prefetcher."""
import os
import time
import unittest
from unittest.mock import patch

import numpy as np

from panda_cable_grasp.rl import future_geometry, future_prefetch




class ResetFixture:
    def __init__(self):
        self.rng = np.random.default_rng(1024)
        self.value = 0
        self.base_env = self

    def reset(self):
        self.value = int(self.rng.integers(1_000_000))

    def _target_object_slice(self):
        return slice(None)


@unittest.skipUnless(hasattr(os, 'fork'), 'Linux worker fork')
class TestResetPrefetch(unittest.TestCase):
    def setUp(self):
        self.env = ResetFixture()
        self.prefetch = future_prefetch.ResetPrefetch()
        self.patches = [patch.object(future_prefetch, 'initial_fingerprint', lambda e: str(e.value)),
            patch.object(future_geometry, 'precompute',
                lambda e, s: (np.full((30,14,3), e.value), 0., None))]
        for p in self.patches: p.start()

    def tearDown(self):
        self.prefetch.close()
        for p in reversed(self.patches): p.stop()

    def test_live_rng_untouched_and_next_reset_exact(self):
        original = repr(self.env.rng.bit_generator.state)
        self.prefetch.start(self.env)
        self.assertEqual(self.env.value, 0)
        self.assertEqual(repr(self.env.rng.bit_generator.state), original)
        self.env.reset()
        result = self.prefetch.take(self.env)
        self.assertIsNotNone(result)
        self.assertTrue(np.all(result['frames'] == self.env.value))
        self.assertEqual((self.prefetch.hits,self.prefetch.errors), (1,0))
        self.assertIsNone(self.prefetch.pid)

    def test_changed_reset_is_never_reused(self):
        self.prefetch.start(self.env)
        self.env.reset(); self.env.value += 1
        self.assertIsNone(self.prefetch.take(self.env))
        self.assertEqual((self.prefetch.hits,self.prefetch.misses), (0,1))

    def test_explicit_seed_discards_prediction(self):
        self.prefetch.start(self.env)
        self.env.reset()
        self.assertIsNone(self.prefetch.take(self.env, eligible=False))
        self.assertEqual(self.prefetch.last_reason, 'explicit_reset')

    def test_close_kills_and_reaps_busy_owned_helper(self):
        with patch.object(future_geometry,'precompute',lambda e,s: time.sleep(30)):
            self.prefetch.start(self.env)
            pid = self.prefetch.pid
            self.prefetch.close()
            with self.assertRaises(ChildProcessError): os.waitpid(pid,os.WNOHANG)

    def test_child_error_is_safe_miss(self):
        def fail(e,s): raise ValueError('test failure')
        with patch.object(future_geometry,'precompute',fail):
            self.prefetch.start(self.env); self.env.reset()
            self.assertIsNone(self.prefetch.take(self.env))
            self.assertEqual(self.prefetch.errors,1)
            self.assertIn('test failure',self.prefetch.last_reason)
