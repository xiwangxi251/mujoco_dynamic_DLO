"""Same node-feedback trials, saving native states for later video rendering."""
from __future__ import annotations
from dataclasses import asdict, replace
from pathlib import Path
import sys
import time
import mujoco
import numpy as np
import run_node_feedback_family as runner
from record_scripted_scene_family import project_camera_manifest

STATE = mujoco.mjtState.mjSTATE_INTEGRATION
OUTPUT = None
OriginalEnv = runner.CableGraspEnv
OriginalSelector = runner.GeometrySelector
original_prepare = runner.prepare
original_coordinate = runner.coordinate
original_supervise = runner.supervise


class StateRecordingEnv(OriginalEnv):
    def __init__(self, config):
        self.saved_states = []
        self.saved_times = []
        self.saved_targets = []
        self.saved_cable = []
        self.saved_actions = []
        self.saved_quats = []
        self.saved_phases = []
        self.record_policy = None
        self.states_written = False
        super().__init__(replace(config, dynamicvla_cameras_enabled=True))
        self.record_kinematics = mujoco.MjData(self.model)

    def capture(self):
        state = np.empty(mujoco.mj_stateSize(self.model, STATE))
        mujoco.mj_getState(self.model, self.data, state, STATE)
        self.record_kinematics.qpos[:] = self.data.qpos
        mujoco.mj_kinematics(self.model, self.record_kinematics)
        p = self.record_policy
        if p.locked_segment_index is None:
            target = self.record_kinematics.xpos[p.selected_target_body_id].copy()
        else:
            i, a = p.locked_segment_index, p.locked_segment_alpha
            ids = p.locked_node_ids or self.cable_ids
            target = ((1-a)*self.record_kinematics.xpos[ids[i]]
                      + a*self.record_kinematics.xpos[ids[i+1]])
        self.saved_states.append(state)
        self.saved_times.append(float(self.data.time))
        self.saved_cable.append(self.record_kinematics.xpos[self.cable_ids].copy())
        self.saved_targets.append(target.copy())
        self.saved_quats.append(p.desired_quat.copy())
        self.saved_phases.append(p.phase.name)

    def step(self, action):
        result = super().step(action)
        self.saved_actions.append(np.asarray(action).copy())
        self.capture()
        return result

    def close(self):
        if self.saved_states and not self.states_written:
            dest = OUTPUT/'episodes'/self.config.scenario_name/f'seed_{self.episode_seed}'
            dest.mkdir(parents=True, exist_ok=True)
            path = dest/'trajectory.npz'
            np.savez_compressed(path, states=np.asarray(self.saved_states),
                frame_times=self.saved_times, frame_indices=np.arange(len(self.saved_states)),
                cable_xyz=np.asarray(self.saved_cable), tracked_target_xyz=self.saved_targets,
                native_actions=self.saved_actions, desired_quats=self.saved_quats,
                phases=self.saved_phases, cable_body_ids=self.cable_ids, state_kind=int(STATE))
            manifest = runner.load(OUTPUT/'manifest.json')
            baseline_path = (Path(manifest['source_run'])/'episodes'/self.config.scenario_name
                             /f'seed_{self.episode_seed}'/'result.json')
            current = runner.load(dest/'result.json') if (dest/'result.json').exists() else None
            baseline = runner.load(baseline_path) if baseline_path.exists() else None
            matching = (current is not None and baseline is not None and
                all(current[key] == baseline[key] for key in
                    ('task_success', 'duration_s', 'steps', 'retries', 'last_attempt_failure', 'phase_changes')))
            runner.save(dest/'state_metadata.json', dict(env_config=asdict(self.config),
                policy_config=asdict(self.record_policy.config), frames=len(self.saved_states),
                state_kind=int(STATE), state_description='mjSTATE_INTEGRATION: physical state, controls, applied forces, warm start and user state',
                cameras=project_camera_manifest(self.config),
                camera_rendering='Use named cameras; states captured at native 50Hz controller boundaries; no video rendered yet',
                trajectory_sha256=runner.sha(path), source_sha256=manifest['source_sha256'],
                original_result=str(baseline_path), original_result_available=baseline is not None,
                original_outcome_and_phase_history_match=matching))
            self.states_written = True
        super().close()


class StateRecordingSelector(OriginalSelector):
    def __init__(self, env, config=None):
        super().__init__(env, config)
        env.base.record_policy = self
        env.base.capture()


def prepare(out, smoke=False):
    original_prepare(out, smoke)
    manifest = runner.load(out/'manifest.json')
    if manifest.get('states_saved'):
        return
    source = Path(__file__)
    target = out/'source'/source.relative_to(runner.ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())
    manifest['source_sha256'][str(source.relative_to(runner.ROOT))] = runner.sha(source)
    helper = runner.ROOT/'tools/evaluation/run_node_feedback_family.py'
    helper_target = out/'source'/helper.relative_to(runner.ROOT)
    helper_target.write_bytes(helper.read_bytes())
    manifest['source_sha256'][str(helper.relative_to(runner.ROOT))] = runner.sha(helper)
    manifest.update(states_saved=True, state_kind=int(STATE),
        source_run=str(runner.ROOT/'outputs/scripted_node_feedback_seven_20261010'),
        cameras=project_camera_manifest(runner.EnvConfig(robot='nero', dynamicvla_cameras_enabled=True)))
    runner.save(out/'manifest.json', manifest)


def coordinate(args):
    original_coordinate(args)
    manifest = runner.load(args.output/'manifest.json')
    metadata = [runner.load(p) for p in args.output.glob('episodes/*/*/state_metadata.json')]
    assert len(metadata) == len(manifest['jobs']), 'Missing saved simulation states'
    result = runner.load(args.output/'summary.json')
    result.update(state_files=len(metadata), states_verified=True,
        original_matching_episodes=sum(m['original_outcome_and_phase_history_match'] for m in metadata),
        original_comparable_episodes=sum(m['original_result_available'] for m in metadata))
    runner.save(args.output/'summary.json', result)
    report = args.output/'report.md'
    with report.open('a', encoding='utf-8') as stream:
        stream.write(f"\n完整状态：{len(metadata)}回合，保存原生状态、动作、时间戳、当前目标、夹爪目标姿态及相机配置。后续视频可直接重渲染，无需重跑策略。\n")
        stream.write(f"与前次结果及阶段历史一致：{result['original_matching_episodes']}/{result['original_comparable_episodes']}。\n")


def supervise(args):
    original_supervise(args)
    status = runner.load(args.output/'status.json')
    result = runner.load(args.output/'summary.json')
    status['results'].update({key: result[key] for key in
        ('state_files', 'states_verified', 'original_matching_episodes', 'original_comparable_episodes')})
    runner.save(args.output/'status.json', status)


def main():
    global OUTPUT
    OUTPUT = Path(sys.argv[sys.argv.index('--output')+1]).resolve()
    runner.CableGraspEnv = StateRecordingEnv
    runner.GeometrySelector = StateRecordingSelector
    runner.prepare = prepare
    runner.coordinate = coordinate
    runner.supervise = supervise
    # Coordinator and supervisor must invoke this entrypoint, not the old worker.
    runner.__file__ = __file__
    runner.main()


if __name__ == '__main__':
    main()
