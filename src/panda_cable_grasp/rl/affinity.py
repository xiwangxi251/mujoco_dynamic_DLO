"""Optional Linux worker placement, balanced across NUMA nodes and physical cores."""
import os
from pathlib import Path


def physical_core_order(allowed, sys_root=Path('/sys/devices/system/cpu')):
    cores = {}
    for cpu in sorted(allowed):
        directory = sys_root/f'cpu{cpu}'
        package = int((directory/'topology/physical_package_id').read_text())
        core = int((directory/'topology/core_id').read_text())
        nodes = sorted(directory.glob('node[0-9]*'))
        node = int(nodes[0].name[4:]) if nodes else package
        cores.setdefault((node, package, core), cpu)
    grouped = {}
    for (node, package, core), cpu in sorted(cores.items()):
        grouped.setdefault(node, []).append(cpu)
    order = []
    for index in range(max(map(len, grouped.values()))):
        for node in sorted(grouped):
            if index < len(grouped[node]):
                order.append(grouped[node][index])
    return order


def prepare_worker_affinity(args):
    if args.worker_affinity == 'none':
        return dict(mode='none', applied=False)
    if not hasattr(os, 'sched_setaffinity'):
        raise RuntimeError('NUMA worker affinity requires Linux')
    allowed = sorted(os.sched_getaffinity(0))
    cores = physical_core_order(allowed)
    if max(args.workers, args.eval_workers)+1 > len(cores):
        raise RuntimeError('Not enough distinct physical cores for workers and learner')
    plan = dict(mode='numa', applied=True, allowed_cpu_ids=allowed,
                train_cpu_ids=cores[:args.workers], eval_cpu_ids=cores[:args.eval_workers],
                learner_cpu_id=cores[max(args.workers, args.eval_workers)],
                memory_policy='first-touch after worker pinning; no forced membind')
    # Compute and pass the whole plan before narrowing the parent affinity.
    os.sched_setaffinity(0, {plan['learner_cpu_id']})
    return plan


def bind_worker(args, rank, evaluation=False):
    plan = getattr(args, 'worker_affinity_plan', {})
    if plan.get('applied'):
        cpu = plan['eval_cpu_ids' if evaluation else 'train_cpu_ids'][rank]
        os.sched_setaffinity(0, {cpu})
