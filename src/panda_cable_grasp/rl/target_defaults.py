"""Resolved defaults for new target-node PPO runs and saved evaluations."""
import json
from pathlib import Path

TARGET_PPO_DEFAULTS = {
    'robot': 'nero',
    'action_mode': 'task_space_vertical_down',
    'controller_profile': 'enhanced6_bounded_dls_v1',
    'table_execution_guard': True,
    'disable_table_finger_collision_filter': True,
    'workspace_scene_family': 'cable_x075_v1',
    'target_reward_profile': 'tolerance_progress_v2',
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
    if (target and args.workspace_scene_family is None
            and not supplied.intersection({'--scenario', '--distribution'})):
        args.workspace_scene_family = saved.get('workspace_scene_family')


def add_table_finger_filter_arguments(parser):
    options = parser.add_mutually_exclusive_group()
    options.add_argument('--disable-table-finger-collision-filter',
                         dest='disable_table_finger_collision_filter', action='store_true',
                         help='disable the legacy gripper/table target prediction filter')
    options.add_argument('--enable-table-finger-collision-filter',
                         dest='disable_table_finger_collision_filter', action='store_false',
                         help='explicitly restore the legacy gripper/table target prediction filter')
    parser.set_defaults(disable_table_finger_collision_filter=None)
