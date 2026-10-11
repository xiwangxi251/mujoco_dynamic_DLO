"""Remove only the workspace harness reference-speed ramp; retain native limits."""
from __future__ import annotations
import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time
import scan_full6_enhanced_workspace as enhanced
import numpy as np
import mujoco

OUT = enhanced.OUT / 'no_extra_motion_limit'
_RIG = None


class NoExtraSpeedRig(enhanced.EnhancedRig):
    def __init__(self):
        super().__init__()
        self.audit.update(reference_speed_ramp_enabled=False,
            waypoint_mode='direct position target; approach then descent',
            approach_completion_position_tolerance_m=.005,
            original_motion_limits_preserved=True)

    def trial(self, x, y):
        self.reset()
        target = np.array([x, y, enhanced.native.Z], dtype=float)
        approach = np.array([x, y, .10])
        descending = False
        descent_start = None
        held = 0
        min_floor, min_self = 1., 1.
        max_tilt = max_pose = 0.
        reason = 'timeout'
        started = time.monotonic()
        for step in range(800):
            # Feed the waypoint directly. All velocity limiting remains inside
            # the native JIK and environment. Advance after actual arrival.
            reference = target if descending else approach
            self.control(reference, enhanced.native.Phase.INTERCEPT if descending
                         else enhanced.native.Phase.APPROACH)
            if not descending and np.linalg.norm(self.env.hand_position - approach) <= .005:
                descending = True
                descent_start = float(self.data.time)
            measurement = self.measure(target, collision=(step % 5 == 0 or held > 0))
            min_floor = min(min_floor, measurement[3])
            min_self = min(min_self, measurement[5])
            max_tilt = max(max_tilt, measurement[1])
            max_pose = max(max_pose, measurement[2])
            finger_floor = min(float(self.geometry.geom_vertices(g)[:, 2].min())
                               for g in self.geometry.finger_geoms)
            acceptable = (
                measurement[0] <= .005 and measurement[2] <= 2.
                and .0002 <= finger_floor <= .008
                and measurement[3] >= .0002 and measurement[4] >= 0.
                and measurement[5] >= .001
            )
            held = held + 1 if acceptable else 0
            if held >= 100:
                reason = 'endpoint_held_2s'
                break
            if measurement[3] < -.001:
                reason = 'floor_penetration'
                break
            if measurement[5] < -.001:
                reason = 'self_collision'
                break
            if not np.all(np.isfinite(self.data.qpos)):
                reason = 'invalid_nonfinite'
                break
        final = self.measure(target, collision=True)
        velocity_ratio = float(np.max(self.peak_velocity / self.env._arm_velocity_limits))
        endpoint_pass = reason == 'endpoint_held_2s'
        return dict(
            x_m=float(x), y_m=float(y), z_m=enhanced.native.Z,
            endpoint_pass=endpoint_pass,
            motion_pass=bool(endpoint_pass and max_pose <= 5. and velocity_ratio <= 1.01),
            reason=reason, steps=step+1, sim_seconds=float(self.data.time),
            wall_seconds=time.monotonic()-started,
            position_error_mm=final[0]*1000, tilt_deg=final[1], pose_error_deg=final[2],
            robot_floor_mm=final[3]*1000, finger_floor_mm=finger_floor*1000,
            joint_margin_deg=final[4], self_distance_mm=final[5]*1000,
            min_path_floor_mm=min_floor*1000, min_path_self_mm=min_self*1000,
            max_path_tilt_deg=max_tilt, max_path_pose_deg=max_pose,
            peak_velocity_to_limit=velocity_ratio,
            velocity_exceedance_substep_ratio=self.over_steps/max(1,self.velocity_steps),
            peak_joint_velocity_rad_s=self.peak_velocity.tolist(),
            table_proximity_limited_steps=int(self.env._table_proximity_limited_steps),
            max_gravity_ff_nm=self.max_ff.tolist(), max_combined_torque_nm=self.max_torque.tolist(),
            torque_saturation_substeps=self.saturation_steps, velocity_guard_substeps=self.guard_steps,
            descent_started=descending, descent_start_s=descent_start)


def worker(job):
    global _RIG
    if _RIG is None:
        _RIG = NoExtraSpeedRig()
    return _RIG.trial(*job)


def audit_limits(rig):
    original = enhanced.EnhancedRig()
    for key in ('uniform_cartesian_linear_cap_m_s', 'ik_target_horizon_s',
                'arm_acceleration_limit_enabled', 'env_cartesian_cap_enabled',
                'table_proximity_velocity_limit_enabled'):
        assert rig.audit[key] == original.audit[key], key
    for key in ('arm_joint_velocity_limits', 'hand_linear_velocity_limit',
                'hand_cartesian_velocity_limit_enabled', 'arm_acceleration_limit_enabled'):
        assert getattr(rig.env.config,key) == getattr(original.env.config,key), key
    assert np.array_equal(rig.model.jnt_range, original.model.jnt_range)
    return dict(original_controller_audit=original.audit,
                original_motion_limits_identical=True,
                arm_joint_velocity_limits_rad_s=rig.env._arm_velocity_limits.tolist(),
                reference_speed_ramp_removed_m_s=[.15,.03])


def scan(workers):
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((enhanced.OUT/'manifest.json').read_text())
    rig=NoExtraSpeedRig()
    limits=audit_limits(rig)
    sources=[Path(__file__),Path(enhanced.__file__)]+[
        Path(p) for p in manifest['source_sha256'] if Path(p)!=Path(enhanced.__file__)]
    manifest.update(controller=rig.audit, approach_speed_m_s=None, descent_speed_m_s=None,
        reference_speed_ramp_enabled=False,
        waypoint_transition='actual TCP within 5 mm of approach at z=0.10 m, then send final waypoint',
        same_protocol_as='previous enhanced full6 scan except removal of reference-speed ramp',
        original_motion_limits_audit=limits,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    mujoco.mj_saveModel(rig.model,str(OUT/'isolated_model.mjb'),None)
    xs=np.asarray(manifest['grid_x_m']);ys=np.asarray(manifest['grid_y_m'])
    file=OUT/'cells.jsonl'
    rows=[json.loads(line) for line in file.read_text().splitlines()] if file.exists() else []
    done={(r['x_m'],r['y_m']) for r in rows}
    jobs=[(float(x),float(y)) for y in ys for x in xs if (float(x),float(y)) not in done]
    start=time.monotonic()
    with file.open('a',encoding='utf-8') as stream,mp.get_context('spawn').Pool(workers) as pool:
        for row in pool.imap_unordered(worker,jobs,chunksize=1):
            rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
            if len(rows)%50==0:
                print(json.dumps(dict(done=len(rows),total=961,wall_s=round(time.monotonic()-start,1))),flush=True)
    enhanced.OUT=OUT
    enhanced.summarize(rows,xs,ys,.04)


def videos(side):
    from record_retest_136_videos import capture
    def factory(directory):
        rig=NoExtraSpeedRig()
        return rig,rig.env,enhanced.native
    sign=1 if side=='positive' else -1
    for x,y,name in ((.67,.28,'limit_pair'),(.55,.28,'endpoint_pair')):
        capture(6,x,sign*y,name+'_'+side,rig_factory=factory,
            output_root=OUT/'videos',scan_directory=OUT,
            label='6+重力前馈+增强伺服+增益10 · 取消测试额外限速')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('scan','videos'),default='scan')
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--side',choices=('positive','negative'),default='positive')
    args=parser.parse_args()
    if args.mode=='scan':scan(args.workers)
    else:videos(args.side)
