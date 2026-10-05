from types import SimpleNamespace
from unittest.mock import patch
import pytest
from panda_cable_grasp.env.environment import CableGraspEnv
from panda_cable_grasp.env.contact_cache import contact_read_scope, cached_contact_reads
from panda_cable_grasp.rl.affinity import physical_core_order, prepare_worker_affinity, bind_worker, bind_learner


class ContactReader:
    _pad_contact_samples = CableGraspEnv._pad_contact_samples
    _invalidate_contact_cache = CableGraspEnv._invalidate_contact_cache

    def __init__(self):
        self.config = SimpleNamespace(contact_read_cache_enabled=True)
        self._pad_contact_cache_active = False
        self._pad_contact_cache = None
        self.reads = 0
        self.force = 1.

    def _compute_pad_contact_samples(self):
        self.reads += 1
        return [(21, 8, self.force)]


def test_reads_only_reused_within_scope_and_return_values_cannot_mutate_cache():
    base = ContactReader()
    with contact_read_scope(base):
        first = base._pad_contact_samples()
        first.clear()
        assert base._pad_contact_samples() == [(21, 8, 1.)]
        assert base.reads == 1
        with contact_read_scope(base):
            assert base._pad_contact_samples() == [(21, 8, 1.)]
        assert base._pad_contact_cache_active
        base.force = 2.
        base._invalidate_contact_cache()
        assert base._pad_contact_samples() == [(21, 8, 2.)]
    base.force = 3.
    assert base._pad_contact_samples() == [(21, 8, 3.)]
    base.force = 4.
    assert base._pad_contact_samples() == [(21, 8, 4.)]
    assert not base._pad_contact_cache_active and base._pad_contact_cache is None


def test_scope_cleans_up_on_exception_and_disabled_path_reads_fresh():
    class Wrapper:
        def __init__(self): self.base_env = ContactReader()

        @cached_contact_reads
        def step(self):
            self.base_env._pad_contact_samples()
            raise ValueError('simulation failed')

    env = Wrapper()
    with pytest.raises(ValueError): env.step()
    assert not env.base_env._pad_contact_cache_active
    assert env.base_env._pad_contact_cache is None
    env.base_env.config.contact_read_cache_enabled = False
    with pytest.raises(ValueError): env.step()
    assert env.base_env.reads == 2


def test_numa_plan_balances_nodes_uses_distinct_physical_cores_and_keeps_spawn_plan(tmp_path):
    for cpu in range(8):
        directory = tmp_path/f'cpu{cpu}'
        (directory/'topology').mkdir(parents=True)
        # 0/4 and 1/5 share node 0 cores; 2/6 and 3/7 share node 1.
        node = (cpu % 4)//2
        (directory/f'node{node}').mkdir()
        (directory/'topology/physical_package_id').write_text(str(node))
        (directory/'topology/core_id').write_text(str(cpu % 2))
    assert physical_core_order(range(8),tmp_path) == [0,2,1,3]
    args = SimpleNamespace(worker_affinity='numa',workers=2,eval_workers=2)
    with patch('os.sched_getaffinity',return_value=set(range(8)),create=True), \
         patch('os.sched_setaffinity',create=True) as bind, \
         patch('panda_cable_grasp.rl.affinity.physical_core_order',return_value=[0,2,1,3]):
        args.worker_affinity_plan = prepare_worker_affinity(args)
        assert args.worker_affinity_plan['learner_cpu_id'] == 1
        bind.assert_not_called()
        bind_learner(args)
        bind.assert_called_with(0,{1})
        bind_worker(args,1)
        bind.assert_called_with(0,{2})
        bind_worker(args,0,evaluation=True)
        bind.assert_called_with(0,{0})


def test_affinity_none_never_changes_scheduling():
    args = SimpleNamespace(worker_affinity='none')
    with patch('os.sched_setaffinity',create=True) as bind:
        args.worker_affinity_plan = prepare_worker_affinity(args)
        bind_worker(args,0)
        bind.assert_not_called()
