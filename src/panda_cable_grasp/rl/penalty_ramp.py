"""Checkpointed global-sample penalty ramps; evaluation never advances them."""
VERSION = 'cumulative_samples_v1'


def validate_schedule(schedule):
    if schedule.get('version') != VERSION:
        raise ValueError('Unsupported penalty ramp schedule version')
    for key in ('reference_workers', 'height_full_samples', 'safety_full_samples'):
        value = schedule[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError('Invalid penalty ramp schedule: '+key)
    if schedule['reference_workers'] < 1:
        raise ValueError('Penalty ramp reference worker count must be positive')


def resolve_schedule(model, rl_config, workers, saved=None):
    saved = saved or {}
    schedule = getattr(model, 'penalty_ramp_schedule', None) or saved.get('penalty_ramp_schedule')
    if schedule is not None:
        validate_schedule(schedule)
        schedule = dict(schedule)
        for kind in ('height', 'safety'):
            # An explicit new ramp duration remains effective without resetting
            # the cumulative progress or changing the reference worker count.
            schedule[kind+'_full_samples'] = int(getattr(rl_config,
                kind+'_penalty_ramp_env_steps')) * schedule['reference_workers']
        return schedule
    # Migration from checkpoints predating the ramp record. The saved worker
    # count defines the original duration, not the new invocation's parallelism.
    reference = int(saved.get('workers', workers))
    schedule = dict(version=VERSION, reference_workers=reference,
        height_full_samples=int(rl_config.height_penalty_ramp_env_steps)*reference,
        safety_full_samples=int(rl_config.safety_penalty_ramp_env_steps)*reference)
    validate_schedule(schedule)
    return schedule


def apply_progress(env, model, schedule, *, training_workers=0):
    values = (int(model.num_timesteps), int(training_workers), schedule)
    if hasattr(env, 'env_method'):
        env.env_method('set_penalty_ramp_progress', *values)
    else:
        env.set_penalty_ramp_progress(*values)
