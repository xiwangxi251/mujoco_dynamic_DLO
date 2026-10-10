"""Resolved defaults for new target-node PPO runs and saved evaluations."""
import json
import hashlib
from pathlib import Path

TARGET_PPO_DEFAULTS = {
    'robot': 'nero',
    'action_mode': 'task_space_vertical_down',
    'controller_profile': 'enhanced6_bounded_dls_v1',
    'table_execution_guard': True,
    'disable_table_finger_collision_filter': True,
    'workspace_scene_family': 'cable_x075_v1',
    'target_reward_profile': 'tolerance_progress_v6',
}


def checkpoint_configuration(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    for directory in (checkpoint.parent, *checkpoint.parents):
        path = directory/'training_config.json'
        if path.is_file():
            return json.loads(path.read_text(encoding='utf-8'))
    return {}


def configure_target_training_defaults(args, argv):
    """Explicit options remain authoritative; defaults are task scoped."""
    supplied = {item.split('=', 1)[0] for item in argv if item.startswith('--')}
    target = args.target_node_task
    generic = dict(robot='panda', action_mode='task_space',
                   controller_profile='legacy', table_execution_guard=False,
                   disable_table_finger_collision_filter=False,
                   target_reward_profile='legacy')
    for key in generic:
        if getattr(args, key) is None:
            setattr(args, key, TARGET_PPO_DEFAULTS[key] if target else generic[key])
    if (target and args.workspace_scene_family is None
            and not supplied.intersection({'--training-scenarios', '--eval-scenarios',
                                            '--training-distribution', '--eval-distribution'})):
        args.workspace_scene_family = TARGET_PPO_DEFAULTS['workspace_scene_family']
    if (target and args.target_workspace is None
            and args.workspace_scene_family == 'cable_x075_v1'):
        from .workspace import DEFAULT_TARGET_WORKSPACE
        args.target_workspace = DEFAULT_TARGET_WORKSPACE


def configure_saved_evaluation_defaults(args, argv):
    """Inherit the saved execution/scenes instead of replacing old experiments."""
    saved = checkpoint_configuration(args.model)
    target = bool(saved.get('target_node_task', saved.get('rl_config', {}).get('target_node_task', False)))
    if args.controller_profile is None:
        args.controller_profile = saved.get('controller_profile', 'legacy')
    if args.table_execution_guard is None:
        args.table_execution_guard = bool(saved.get('table_execution_guard_enabled', False))
    if args.disable_table_finger_collision_filter is None:
        args.disable_table_finger_collision_filter = not bool(
            saved.get('table_finger_collision_filter_enabled', True))
    if args.robot is None:
        args.robot = saved.get('robot', 'nero' if args.controller_profile != 'legacy' else 'panda')
    supplied = {item.split('=', 1)[0] for item in argv if item.startswith('--')}
    ordinary = bool(saved.get("ordinary_family_curriculum", False))
    if ((target or ordinary) and args.workspace_scene_family is None
            and not supplied.intersection({'--scenario', '--distribution'})):
        args.workspace_scene_family = saved.get('workspace_scene_family')
    args.saved_evaluation_config = saved
    args.saved_stage_evaluation = bool((target or ordinary)
        and getattr(args, 'evaluation_protocol', 'saved-stage') == 'saved-stage'
        and not supplied.intersection({'--scenario', '--distribution', '--workspace-scene-family'}))
    if not target and not ordinary:
        return
    workspace = saved.get('target_workspace')
    if "--workspace-start-protocol" not in supplied:
        # Old weights/configs must keep their archived conditioned protocol.
        args.workspace_start_protocol = saved.get("workspace_start_protocol",
            "physical_entry_v1" if workspace else "policy_from_initial_v2")
    if '--target-workspace' not in supplied and workspace:
        path = Path(workspace['path'])
        if not path.is_file():
            # The built-in rectangle can be relocated between Linux/Windows
            # only when its content hash matches the saved experiment.
            from .workspace import DEFAULT_TARGET_WORKSPACE
            if hashlib.sha256(DEFAULT_TARGET_WORKSPACE.read_bytes()).hexdigest() != workspace.get('sha256'):
                raise ValueError('Saved workspace unavailable; pass --target-workspace with the original NPZ')
            path = DEFAULT_TARGET_WORKSPACE
        args.target_workspace = path
    for key in ('workspace_entry_seconds', 'workspace_reset_attempts', 'episode_seconds', 'disturbance'):
        if '--'+key.replace('_', '-') not in supplied and key in saved:
            setattr(args, key, saved[key])
    if '--workspace-mask-key' not in supplied and workspace:
        args.workspace_mask_key = workspace.get('mask_key', 'interior')
    if '--replay-bank-dir' not in supplied:
        bank = saved.get('eval_replay_bank_dir') or saved.get('replay_bank_dir')
        if bank:
            if not Path(bank).is_dir():
                raise ValueError('Saved replay bank unavailable; pass --replay-bank-dir with the evaluation bank')
            args.replay_bank_dir = Path(bank)
    if '--replay-seed-fallback' not in supplied:
        args.replay_seed_fallback = 'error' if saved.get('eval_replay_bank_dir') else saved.get('replay_seed_fallback', 'resample')
    if '--geometric-safety' not in supplied:
        args.geometric_safety = bool(saved.get('geometric_safety_enabled', False))
    if '--seed' not in supplied and saved.get('eval_seed') is not None:
        args.seed = int(saved['eval_seed'])
    args.workspace_reset_seed_pool = ((int(saved['eval_seed']), int(saved['eval_seed_pool_size']))
        if saved.get('eval_seed_pool_size', 0) > 0 else None)


def saved_curriculum_stage(checkpoint, saved, model=None):
    stage = getattr(model, 'target_evaluation_stage', None)
    if stage is not None:
        return int(stage), 'checkpoint'
    # Old checkpoints lack stage metadata; explicitly report the fallback.
    for directory in (Path(checkpoint).resolve().parent, *Path(checkpoint).resolve().parents):
        if (directory/'training_config.json').is_file():
            archived = directory/'workspace_curriculum.json'
            if archived.is_file():
                return int(json.loads(archived.read_text(encoding='utf-8'))['stage']), 'archived_curriculum'
            path = directory/'evaluation/workspace_curriculum.json'
            if path.is_file():
                return int(json.loads(path.read_text(encoding='utf-8'))['stage']), 'run_curriculum_fallback'
            break
    raise ValueError('Saved curriculum stage unavailable; use --evaluation-protocol six-scenes for a distinct full-scene test')


def add_table_finger_filter_arguments(parser):
    options = parser.add_mutually_exclusive_group()
    options.add_argument('--disable-table-finger-collision-filter',
                         dest='disable_table_finger_collision_filter', action='store_true',
                         help='disable the legacy gripper/table target prediction filter')
    options.add_argument('--enable-table-finger-collision-filter',
                         dest='disable_table_finger_collision_filter', action='store_false',
                         help='explicitly restore the legacy gripper/table target prediction filter')
    parser.set_defaults(disable_table_finger_collision_filter=None)
