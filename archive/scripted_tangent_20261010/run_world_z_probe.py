"""Record an end-effector world-Z yaw step using the normal enhanced6 controller."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/scripted_world_z_rotation_20261010'
SOURCE = OUT/'source'

def save(name, value):
    path = OUT/name
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    save('status.json', dict(status='starting'))
    try:
        if not SOURCE.exists():
            for folder in ('src', 'assets'):
                shutil.copytree(ROOT/folder, SOURCE/folder,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            target = SOURCE/'tools/recording'
            target.mkdir(parents=True)
            for name in ('record_scripted_scene_family.py', 'target_node_annotation.py'):
                shutil.copy2(ROOT/'tools/recording'/name, target/name)
        provenance = {str(p.relative_to(SOURCE)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in SOURCE.rglob('*') if p.is_file() and p.suffix in ('.py', '.npz')}
        save('source_provenance.json', dict(source_sha256=provenance,
            task='Fixed grasp-center position, R_world_Z(+90deg) @ initial hand rotation'))
        sys.path.insert(0, str(SOURCE/'src'))
        sys.path.insert(0, str(SOURCE/'tools/recording'))
        import numpy as np
        import record_scripted_scene_family as recording
        from panda_cable_grasp.env.kinematics import rotation_to_quat
        samples = []
        probe = {}
        OriginalEnv = recording.CableGraspEnv
        OriginalPolicy = recording.DynamicCableGraspPolicy
        class InstrumentedEnv(OriginalEnv):
            def _physics_step(self):
                result = super()._physics_step()
                _, jacr = self._hand_jacobian()
                rotation = self.data.xmat[self.hand_id].reshape(3, 3).copy()
                closing = rotation @ self.gripper_closing_axis_local
                samples.append([float(self.data.time), *self.hand_position,
                    float(np.arctan2(closing[1], closing[0])), *(jacr @ self.data.qvel),
                    *rotation.ravel()])
                return result
        class WorldZProbe(OriginalPolicy):
            def __init__(self, env, config):
                super().__init__(env, config)
                self.probe_position = env.hand_position.copy()
                self.probe_rotation = env.data.xmat[env.hand_id].reshape(3, 3).copy()
                self.goal_rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]) @ self.probe_rotation
                self.settled_at = None
                probe.update(position=self.probe_position, rotation=self.probe_rotation,
                    closing_axis=env.gripper_closing_axis_local.copy())
            def action(self):
                now = float(self.env.data.time)
                active = now >= 1.
                self.phase = SimpleNamespace(name='WORLD_Z_PLUS_90' if active else 'HOLD_INITIAL_POSE')
                self.desired_quat = rotation_to_quat(self.goal_rotation if active else self.probe_rotation)
                error_angle = self._orientation_error()[1]
                position_error = np.linalg.norm(self.env.hand_position-self.probe_position)
                if active and error_angle <= np.radians(2.) and position_error <= .005:
                    if self.settled_at is None:
                        self.settled_at = now
                else:
                    self.settled_at = None
                if now >= 12. or (self.settled_at is not None and now-self.settled_at >= 1.):
                    self.finished = True
                    self.result = 'world_z_probe_complete'
                return self._ik_action(self.probe_position, self.env.gripper_open_ctrl)
        recording.CableGraspEnv = InstrumentedEnv
        recording.DynamicCableGraspPolicy = WorldZProbe
        save('status.json', dict(status='recording'))
        row = recording.record(dict(scene='dev_static_x075_v1', seed=20260804, episode=1,
            output=str(OUT/'rotation'), bank=str(ROOT/'outputs/rl/train/ppo_nero_targetnode_state6_warmstart3m_20260928/eval_banks'),
            controller_profile='enhanced6_bounded_dls_v1', table_guard=True, gpu=0,
            protocol_sha256='world_z_plus90_fixed_grasp_center_v1', rule_config={}))
        raw = np.asarray(samples)
        t = raw[:, 0]
        yaw = np.unwrap(raw[:, 4])
        initial_closing = probe['rotation'] @ probe['closing_axis']
        initial_yaw = np.arctan2(initial_closing[1], initial_closing[0])
        delta_yaw = np.degrees(yaw-initial_yaw)
        after = t >= 1.
        error = np.abs(delta_yaw-90.)
        first = np.flatnonzero(after & (error <= 2.))
        stable = next((i for i in np.flatnonzero(after) if np.all(error[i:] <= 2.)), None)
        # Relative world rotation: its vertical basis must remain unchanged.
        relative = raw[:,8:].reshape(-1,3,3) @ probe['rotation'].T
        tilt_change = np.degrees(np.arccos(np.clip(relative[:,2,2], -1., 1.)))
        stats = dict(requested_rotation_deg=90., axis='world Z (vertical)', measurement_hz=500,
            actual_final_yaw_change_deg=float(delta_yaw[-1]),
            peak_actual_world_z_speed_deg_s=float(np.max(np.abs(np.degrees(raw[after,7])))),
            first_within_2deg_seconds=None if not len(first) else float(t[first[0]]-1.),
            stable_within_2deg_seconds=None if stable is None else float(t[stable]-1.),
            max_grasp_center_displacement_mm=float(1000*np.max(np.linalg.norm(raw[:,1:4]-probe['position'],axis=1))),
            max_tilt_change_deg=float(tilt_change.max()),
            duration_s=row['duration_s'], directory=row['directory'],
            control='Enhanced6 bounded DLS; native limits and table protection; fixed grasp-center target, coordinated arm joints.')
        np.savetxt(OUT/'world_z_500hz.csv', raw, delimiter=',',
            header='time,x,y,z,closing_yaw_rad,omega_x,omega_y,omega_z,'+','.join(f'R{i}' for i in range(9)), comments='')
        save('world_z_speed.json', stats)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2,1,figsize=(8,5),sharex=True)
        axes[0].plot(t,delta_yaw,label='actual yaw change')
        axes[0].plot(t,np.where(after,90.,0.),'--',label='world-Z yaw command')
        axes[0].set_ylabel('Yaw change (deg)'); axes[0].legend(); axes[0].grid(alpha=.3)
        axes[1].plot(t,np.degrees(raw[:,7])); axes[1].set_ylabel('World-Z speed (deg/s)')
        axes[1].set_xlabel('Episode time (s)'); axes[1].grid(alpha=.3)
        fig.tight_layout(); fig.savefig(OUT/'world_z_speed.png',dpi=160); plt.close(fig)
        make_page(row,stats)
        save('status.json',dict(status='complete',world_z_speed=stats,
            gallery='http://127.0.0.1:8783/world_z.html'))
    except Exception as error:
        save('status.json',dict(status='failed',error=f'{type(error).__name__}: {error}'))
        raise

def make_page(row, stats):
    import re
    page=(ROOT/'outputs/scripted_seven_default_20261009/index.html').read_text(encoding='utf-8')
    row.update(rotation_probe=True, directory='../scripted_world_z_rotation_20261010/rotation/'+row['directory'])
    page=re.sub(r'<script id="data" type="application/json">.*?</script>',lambda _: '<script id="data" type="application/json">'+json.dumps([row],ensure_ascii=False)+'</script>',page,flags=re.S)
    page=re.sub(r'<select id="scene">.*?</select>', '<select id="scene"><option value="dev_static_x075_v1">夹爪绕世界竖直轴转90°</option></select>', page,flags=re.S)
    page=re.sub(r'<select id="goal">.*?</select>', '<select id="goal"><option value="20260804">旋转测速 · seed 20260804</option></select>',page,flags=re.S)
    page=re.sub(r'<h1>.*?</h1>','<h1>夹爪绕世界竖直轴旋转90°</h1>',page,count=1)
    page=re.sub(r'<table>.*?</table>','',page,count=1,flags=re.S)
    page=page.replace('scripted_seven_default_20261009','scripted_world_z_rotation_20261010')
    page=page.replace("e.task_success?'成功':'失败'", "'旋转测速'")
    note=f"绕世界Z轴旋转；固定夹爪中心目标；增强6＋有界DLS，限速与防撞保护保留。实际最终转角 {stats['actual_final_yaw_change_deg']:.1f}°；峰值Z轴角速度 {stats['peak_actual_world_z_speed_deg_s']:.1f}°/s；进入目标±2°用时 {stats['first_within_2deg_seconds']}s。"
    page=re.sub(r'<p class="note">此页为原七场景.*?</p>',lambda _: '<p class="note">'+note+'</p>',page,count=1,flags=re.S)
    page=re.sub(r'<p id="note".*?</p>', '<p id="note" class="note">蓝线为当前环形扇区工作区；黄圈为绳缆材料点，本测速没有执行抓取。</p>',page,count=1,flags=re.S)
    page=page.replace('<footer>','<details><summary>世界竖直轴角度与速度曲线（500 Hz）</summary><img src="../scripted_world_z_rotation_20261010/world_z_speed.png" style="max-width:100%"></details><footer>')
    page=re.sub(r'<footer>.*?</footer>','<footer><a href="../scripted_world_z_rotation_20261010/world_z_speed.json">测速数据</a><a href="../scripted_world_z_rotation_20261010/world_z_500hz.csv">500Hz记录</a></footer>',page,flags=re.S)
    (OUT/'index.html').write_text(page.replace('../scripted_world_z_rotation_20261010/',''),encoding='utf-8')
    # Existing server serves only the original experiment root.
    served=ROOT/'outputs/scripted_tangent_20261010'
    # Relative parent URLs would be outside that HTTP root, so link the result directory.
    link=served/'world_z_result'
    if not link.exists():
        import subprocess
        subprocess.run(['cmd','/c','mklink','/J',str(link),str(OUT)],check=True,capture_output=True)
    (served/'world_z.html').write_text(page.replace('../scripted_world_z_rotation_20261010/','world_z_result/'),encoding='utf-8')

if __name__=='__main__':
    main()
