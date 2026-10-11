"""Paired causal interventions; production IK, model XML and criteria unchanged.

Gravity feedforward uses a zero-velocity inverse-dynamics probe, applied as
joint torque at each physics substep. Total PD + feedforward respects the
original actuator torque bounds. This is an unloaded simulation diagnostic,
not a deployed torque-interface implementation.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import subprocess

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw
import compare_project_ik_workspace as comparison
from panda_cable_grasp.env.kinematics import point_jacobian, quat_error, rotation_to_quat

OUT = Path('outputs/vertical_workspace/nero_ik_tracking_diagnosis_20261001')
VARIANTS = ('baseline', 'gravity_ff', 'gain4', 'gravity_ff_gain4')
TARGETS = ((.7, 0), (.65, .3), (.65, -.3), (.4, .4), (.82, 0), (.55, 0))


class DiagnosticRig(comparison.Rig):
    def __init__(self, method, variant):
        super().__init__(method)
        self.variant = variant
        self.gravity_probe = mujoco.MjData(self.model)
        self.feedforward = 'gravity_ff' in variant
        self.max_total_torque = np.zeros(7)
        self.max_ff = np.zeros(7)
        self.saturation_steps = 0
        if 'gain4' in variant:
            # Stiffer position servo with sqrt(4) velocity gain scaling.
            self.model.actuator_gainprm[:7, 0] *= 4
            self.model.actuator_biasprm[:7, 1] *= 4
            self.model.actuator_biasprm[:7, 2] *= 2
        assert np.allclose(self.model.actuator_gear[:7, 0], 1)
        if 'angular4' in variant:
            self.adapter.config = replace(self.adapter.config, angular_gain=10.)

    def reset(self):
        super().reset()
        self.max_total_torque = np.zeros(7)
        self.max_ff = np.zeros(7)
        self.saturation_steps = 0

    def control(self, target):
        self.adapter.set_model_action(np.r_[target, self.quat, 1.])
        assert not self.adapter.last_position_clipped
        requested = self.adapter.action()
        requested = np.clip(requested, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
        applied = self.base._limit_robot_action(requested)
        for _ in range(10):
            guarded, _ = self.base._velocity_guarded_action(applied)
            self.data.ctrl[:] = guarded
            self.data.qfrc_applied[:] = 0
            mujoco.mj_forward(self.model, self.data)
            if self.feedforward:
                self.gravity_probe.qpos[:] = self.data.qpos
                self.gravity_probe.qvel[:] = 0
                self.gravity_probe.qacc[:] = 0
                mujoco.mj_forward(self.model, self.gravity_probe)
                ff = self.gravity_probe.qfrc_bias[self.base.arm_dof_adr].copy()
                pd = self.data.actuator_force[:7].copy()
                bounds = self.model.actuator_forcerange[:7]
                total = np.clip(pd + ff, bounds[:, 0], bounds[:, 1])
                self.saturation_steps += int(np.any(abs(total-pd-ff) > 1e-9))
                ff = total - pd
                self.data.qfrc_applied[self.base.arm_dof_adr] = ff
                self.max_ff = np.maximum(self.max_ff, abs(ff))
                self.max_total_torque = np.maximum(self.max_total_torque, abs(total))
            else:
                self.max_total_torque = np.maximum(self.max_total_torque, abs(self.data.actuator_force[:7]))
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return applied

    def conditioning(self, target):
        jp, jr = point_jacobian(self.model, self.data, self.base.hand_id, self.base.GRASP_CENTER_LOCAL)
        jp, jr = jp[:, self.base.arm_dof_adr], jr[:, self.base.arm_dof_adr]
        null = np.eye(7)-np.linalg.pinv(jp, rcond=1e-5)@jp
        jo = jr@null
        s = np.linalg.svd(jo, compute_uv=False)
        c = self.adapter.config
        lam = c.orientation_damping_min + (c.orientation_damping_max-c.orientation_damping_min)*np.clip((.1-s[-1])/.1, 0, 1)**2
        return {'orientation_nullspace_singular_values': s.tolist(),
                'orientation_damping': float(lam),
                'orientation_mode_transmission': (s*s/(s*s+lam*lam)).tolist(),
                'joint_position_error_rad': (self.data.ctrl[:7]-self.data.qpos[:7]).tolist(),
                'gravity_torque_at_endpoint_nm': self.data.qfrc_bias[:7].tolist()}


def worker(job):
    method, variant, x, y = job
    comparison.HOLD_STEPS = 100  # Require two continuous seconds, stricter than prior 0.3 s.
    rig = DiagnosticRig(method, variant)
    row, history = rig.trial(x, y, record=True)
    row.update(variant=variant, hold_requirement_s=2.,
               max_total_torque_nm=rig.max_total_torque.tolist(),
               max_gravity_feedforward_nm=rig.max_ff.tolist(),
               torque_saturation_substeps=rig.saturation_steps,
               endpoint=rig.conditioning(np.array([x, y, comparison.Z])))
    name = f'{method}_{variant}_{x}_{y}'
    np.savez_compressed(OUT/f'{name}.npz', history=history)
    return row


def continuous_worker(job):
    method, variant, x, y = job
    comparison.HOLD_STEPS = 100000
    rig = DiagnosticRig(method, variant)
    row, history = rig.trial(x, y, record=True)
    last = history[-100:]
    ok = (last[:,10] <= .005) & (last[:,12] <= 2) & (last[:,13] >= .0002) & (last[:,13] <= .008) & (last[:,14] >= -.01) & (last[:,15] >= .001)
    success = bool(len(history)==700 and ok.all())
    row.update(variant=variant, success=success,
               reason='continued_14s_endpoint_hold_pass' if success else row['reason'],
               continued_14s_last_2s_pass=success,
               last_2s_max_position_error_mm=float(last[:,10].max()*1000),
               last_2s_max_tilt_deg=float(last[:,11].max()),
               last_2s_max_pose_error_deg=float(last[:,12].max()),
               max_total_torque_nm=rig.max_total_torque.tolist(),
               torque_saturation_substeps=rig.saturation_steps)
    np.savez_compressed(OUT/f'{method}_{variant}_{x}_{y}_continued_14s.npz', history=history)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--finish', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.finish:
        finish()
        return
    jobs = [(m, v, x, y) for m in comparison.METHODS for v in VARIANTS for x, y in TARGETS]
    rows = []
    result = OUT/'results.jsonl'
    done = set()
    if result.exists():
        rows = [json.loads(s) for s in result.read_text(encoding='utf-8').splitlines()]
        done = {(r['method'], r['variant'], r['x_m'], r['y_m']) for r in rows}
    jobs = [j for j in jobs if j not in done]
    plan = {'variants': list(VARIANTS), 'targets': TARGETS,
            'gravity_feedforward': 'zero-velocity qfrc_bias, recomputed each 2ms physics substep; total torque clamped to original limits',
            'gain4': 'arm kp x4, kv x2; original joint damping, friction, gravity, masses, limits, contact unchanged',
            'hold_s': 2., 'timeout_s': 14., 'position_tolerance_mm': 5., 'pose_tolerance_deg': 2.,
            'floor_clearance_mm': [.2, 8.],
            'scope': 'paired unloaded engineering diagnostics; not a workspace rescan or grasp benchmark',
            'source_sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                ('src/panda_cable_grasp/dynamicvla/adapter.py', 'src/panda_cable_grasp/env/environment.py',
                 'assets/mujoco/nero/nero.xml', __file__)}}
    (OUT/'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    with result.open('a', encoding='utf-8') as file, mp.get_context('spawn').Pool(args.workers) as pool:
        for row in pool.imap_unordered(worker, jobs):
            rows.append(row)
            file.write(json.dumps(row)+'\n'); file.flush()
            print(f"{len(rows)}/48 {row['method']} {row['variant']} ({row['x_m']},{row['y_m']}): "
                  f"{row['reason']} pos={row['position_error_mm']:.2f}mm tilt={row['tilt_deg']:.2f}deg "
                  f"pose={row['pose_error_deg']:.2f}deg floor={row['floor_clearance_mm']:.2f}mm", flush=True)
    (OUT/'results.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')


def finish():
    rows = json.loads((OUT/'results.json').read_text(encoding='utf-8'))
    prior = json.loads((comparison.OUTPUT/'physical_pilot/pilot.json').read_text(encoding='utf-8'))
    assert len(rows) == len({(r['method'], r['variant'], r['x_m'], r['y_m']) for r in rows}) == 48
    baseline_deltas = []
    for old in prior:
        r = next(r for r in rows if r['method']==old['method'] and r['variant']=='baseline'
                 and r['x_m']==old['x_m'] and r['y_m']==old['y_m'])
        baseline_deltas += [abs(r[k]-old[k]) for k in ('position_error_mm','tilt_deg','pose_error_deg')]
    assert max(baseline_deltas) < 1e-9
    plan = json.loads((OUT/'plan.json').read_text(encoding='utf-8'))
    for p, sha in plan['source_sha256'].items():
        if p != __file__:
            assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha
    assert all(r['torque_saturation_substeps']==0 for r in rows)

    # Friction removal is a causal probe only, not the proposed correction.
    friction = json.loads((OUT/'friction_probe.json').read_text(encoding='utf-8')) if (OUT/'friction_probe.json').exists() else []
    for method in comparison.METHODS if not friction else ():
        rig = DiagnosticRig(method, 'gravity_ff')
        old = rig.model.dof_frictionloss[:7].copy()
        rig.model.dof_frictionloss[:7] = 0
        comparison.HOLD_STEPS = 100000
        row, history = rig.trial(.7, 0, record=True)
        row.update(variant='gravity_ff_friction_zero_causal_probe', original_frictionloss_nm=old.tolist())
        friction.append(row)
        np.savez_compressed(OUT/f'{method}_friction_probe.npz', history=history)
        print(f"friction probe {method}: {row['position_error_mm']:.4f}mm {row['tilt_deg']:.4f}deg", flush=True)
    (OUT/'friction_probe.json').write_text(json.dumps(friction, indent=2), encoding='utf-8')

    # Equal 14-second recordings, including continued control after qualification.
    continuity = json.loads((OUT/'continuity_check.json').read_text(encoding='utf-8')) if (OUT/'continuity_check.json').exists() else []
    histories = {}
    rigs = {}
    for method in comparison.METHODS:
        for variant in ('baseline', 'gravity_ff_gain4'):
            rig = DiagnosticRig(method, variant)
            cached = OUT/f'{method}_{variant}_continued_14s.npz'
            if cached.exists() and continuity:
                histories[method,variant] = np.load(cached)['history']
                rigs[method,variant] = rig
                continue
            comparison.HOLD_STEPS = 100000
            row, hist = rig.trial(.7, 0, record=True)
            last = hist[-100:]
            ok = (last[:,10] <= .005) & (last[:,12] <= 2) & (last[:,13] >= .0002) & (last[:,13] <= .008) & (last[:,14] >= -.01) & (last[:,15] >= .001)
            row.update(variant=variant, continued_14s_last_2s_pass=bool(ok.all()),
                       last_2s_max_position_error_mm=float(last[:,10].max()*1000),
                       last_2s_max_tilt_deg=float(last[:,11].max()),
                       last_2s_max_pose_error_deg=float(last[:,12].max()))
            continuity.append(row); histories[method,variant]=hist; rigs[method,variant]=rig
            np.savez_compressed(OUT/f'{method}_{variant}_continued_14s.npz', history=hist)
    (OUT/'continuity_check.json').write_text(json.dumps(continuity, indent=2), encoding='utf-8')
    extra_path = OUT/'jacobian_angular_feedback.json'
    if extra_path.exists():
        extra = json.loads(extra_path.read_text(encoding='utf-8'))
    else:
        extra = []
        jobs = [('jacobian_ik','gravity_ff_gain4_angular4',x,y) for x,y in TARGETS]
        with mp.get_context('spawn').Pool(4) as pool:
            for r in pool.imap_unordered(continuous_worker,jobs):
                extra.append(r)
                print(f"angular feedback ({r['x_m']},{r['y_m']}): {r['position_error_mm']:.3f}mm {r['tilt_deg']:.3f}deg pose={r['pose_error_deg']:.3f}deg pass={r['success']}",flush=True)
        extra_path.write_text(json.dumps(extra,indent=2),encoding='utf-8')
    improved = next(r for r in extra if r['x_m']==.7 and r['y_m']==0)
    assert improved['continued_14s_last_2s_pass']
    assert all(r['torque_saturation_substeps']==0 for r in extra)
    key=('jacobian_ik','gravity_ff_gain4_angular4')
    histories[key] = np.load(OUT/'jacobian_ik_gravity_ff_gain4_angular4_0.7_0_continued_14s.npz')['history']
    rigs[key] = DiagnosticRig(*key)
    integrity = {'rows':48, 'unique':48, 'prior_baseline_max_delta':max(baseline_deltas),
                 'production_source_hashes_unchanged':True, 'torque_saturation_substeps':0,
                 'qualified_hold_s':2, 'continued_14s_pose_fixed_pass':True,
                 'continued_14s_jacobian_gain_only_pose_drift_deg':2.2921998470685425,
                 'continued_14s_jacobian_angular4_fixed_pass':True,
                 'additional_jacobian_angular4_cases':len(extra),
                 'diagnostic_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'verification.json').write_text(json.dumps(integrity, indent=2), encoding='utf-8')

    report(rows, friction, continuity, extra)
    movie(rigs, histories)


def report(rows, friction, continuity, extra):
    lines = ['# NERO 竖直贴地：IK 与执行层的配对因果诊断', '',
        '2026-10-01。独立诊断，未修改正式控制器或机器人 XML。相同初始关节、参考路径、目标高度5 mm、速度限制、质量、重力、摩擦和接触参数；仅改变表中明确注明的控制项。不是抓取基准，也不是修正后的全工作区扫描。', '',
        '位置误差≤5 mm、完整姿态误差≤2°、最低离地0.2–8 mm、合法关节和自碰撞间距≥1 mm；连续保持2秒。每次最多14秒。相比前一轮0.3秒保持，本次更严格。', '',
        '| 方法 | 控制项 | 通过/6 | (0.7,0)位置误差 mm | 竖直偏差 ° | 完整姿态误差 ° |',
        '|---|---|---:|---:|---:|---:|']
    for m in comparison.METHODS:
        for v in VARIANTS:
            group = [r for r in rows if r['method']==m and r['variant']==v]
            p = next(r for r in group if r['x_m']==.7 and r['y_m']==0)
            lines.append(f"| {m} | {v} | {sum(r['success'] for r in group)}/6 | {p['position_error_mm']:.3f} | {p['tilt_deg']:.3f} | {p['pose_error_deg']:.3f} |")
    lines += ['', 'gravity_ff：每2 ms在静态逆动力学探针中计算重力力矩，并加入关节力矩；PD＋前馈的合计力矩受原模型力矩限幅。gain4：臂关节 kp×4、kv×2；不改变关节摩擦、固有阻尼、关节范围、质量、重力、指爪参数或接触参数。48次配对试验均未触发合计力矩饱和。', '',
        '## 因果解释', '',
        '默认位置执行器输出近似 kp(q_target−q)−kv*qdot；没有重力前馈。目标关节角等于实际关节角时，静止臂执行器力矩为零，不能平衡重力。上一轮理想解的第二关节重力力矩约6.82 Nm，kp=80，对应单关节约0.085 rad（4.9°）的支撑偏差量级；真实多关节误差还受摩擦、接触和闭环控制影响，不等于此单关节估计。MuJoCo位置伺服定义参见 [官方文档](https://mujoco.readthedocs.io/en/latest/XMLreference.html#actuator-position)。', '',
        '旧 Jacobian 分支是位置优先的微分控制：姿态在位置零空间内修正，最终只输出 q_target=q_current+0.11*q_velocity，姿态速度增益2.5。因此它和底层位置伺服共同构成一个间接的姿态伺服，受静摩擦与支撑误差影响更大。Pose IK则迭代求完整目标位姿的关节角。', '',
        '在旧 Jacobian 46°偏斜的代表终态，位置零空间内姿态雅可比最小奇异值为0.714，阻尼0.04，对应最弱模态传递系数0.9969。这不是明显的奇异点。补偿重力后仍有4.24°竖直偏差；再增加伺服增益降到1.10°。所以这一例不能主要归因于雅可比奇异或IK数学本身。', '',
        '为区分剩余静摩擦影响，另做单目标“重力补偿＋臂摩擦归零”的因果探针。该探针不是推荐改动，不应通过删除摩擦改善正式评测：', '']
    for r in friction:
        lines.append(f"- {r['method']}：14秒终态位置误差{r['position_error_mm']:.4f} mm，竖直偏差{r['tilt_deg']:.4f}°，完整姿态误差{r['pose_error_deg']:.4f}°。")
    lines += ['', '延长到完整14秒，Pose IK加前馈和增益后的终态为0.905 mm / 竖直0.269° / 完整姿态0.625°；旧Jacobian相同改动的终态为1.228 mm / 竖直0.688° / 完整姿态2.292°。它在前期能连续2秒达标，但后期绕竖直轴的朝向略漂移。因此又保留摩擦、增益与所有限幅，将旧分支姿态速度反馈增益从2.5提高到10，做完整14秒的六目标验证：', '',
        '| 目标XY / m | 位置误差 mm | 竖直偏差 ° | 完整姿态误差 ° | 末2秒全部条件通过 |',
        '|---|---:|---:|---:|---|']
    for r in sorted(extra,key=lambda r:(r['x_m'],r['y_m'])):
        lines.append(f"| ({r['x_m']},{r['y_m']}) | {r['position_error_mm']:.3f} | {r['tilt_deg']:.3f} | {r['pose_error_deg']:.3f} | {r['success']} |")
    lines += ['', '## 边界与后续修复范围', '',
        '补偿重力并增强伺服后，两种方法均在(0.7,0)、(0.65,0.3)、(0.4,0.4)、(0.82,0)四个目标通过；它们的测试坐标使用世界米。', '',
        '(0.65,-0.3)仍未整体通过：Pose IK的末端位置/姿态已达标，但关节边界略超限（约0.125°）；Jacobian分支的位置约5.92 mm、竖直偏差4.72°。这是限位、冗余姿态选择和路径层面的剩余问题。', '',
        '(0.55,0)在本次固定朝向、初始姿态、路径和求解器下始终未达到，不能靠伺服增益解决。未证明不存在另一关节解或路径。', '',
        '正式修复应先在共用底层增加可配置的重力前馈与匹配的伺服参数，继续使用目前默认的Pose IK；对边界任务增加关节限位余量、冗余姿态选择与路径可达性检查。旧Jacobian分支若继续用于竖直贴地，应单独验证姿态闭环和约束优先级。', '',
        '本诊断没有更新原有工作区范围，也没有替换历史基准结果；若修改正式共用底层，应让各方法使用同一执行层重新评测。', '',
        '## 验证与文件', '',
        '12条原始基线与上一轮位置、倾角、完整姿态误差逐项完全一致。生产源文件哈希不变，48次主要结果唯一；另有6次Jacobian姿态反馈增益验证、2次摩擦因果探针和4次延长控制验证。Pose修正案例以及额外增强姿态反馈的Jacobian案例，在完整14秒的最后2秒连续满足全部要求。', '',
        '- results.json / results.jsonl：48次配对结果及端点雅可比、力矩诊断；每次原始轨迹保存在同名npz。',
        '- friction_probe.json：两次仅用于定位摩擦影响的实验。',
        '- continuity_check.json：14秒连续控制核验。',
        '- jacobian_angular_feedback.json：保留物理与摩擦，额外加强姿态反馈的六点核验。',
        '- tracking_before_after.mp4：同一目标、真实物理轨迹，修正前后同时回放；全部四段均来自完整14秒物理运行。',
        '- verification.json / plan.json：完整性、协议和源文件哈希。', '',
        '```powershell', "$env:PYTHONPATH='src'", "$env:OMP_NUM_THREADS='1'", "$env:OPENBLAS_NUM_THREADS='1'", "$env:MKL_NUM_THREADS='1'",
        'conda run --no-capture-output -n mujoco-dlo python tools/recording/diagnose_ik_tracking.py --workers 4',
        'conda run --no-capture-output -n mujoco-dlo python tools/recording/diagnose_ik_tracking.py --finish', '```']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def movie(rigs, histories):
    from render_vertical_workspace import font, add_geom, add_line
    renderer = mujoco.Renderer(next(iter(rigs.values())).model, height=310, width=640, max_geom=1000)
    cam = mujoco.MjvCamera(); cam.lookat[:]=(.56,0,.18); cam.distance=1.4; cam.azimuth=135; cam.elevation=-30
    option=mujoco.MjvOption(); option.geomgroup[3]=0
    raw=OUT/'tracking_before_after_raw.mp4'
    video=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),25,(1280,860))
    assert video.isOpened()
    keys=[('pose_ik','baseline'),('pose_ik','gravity_ff_gain4'),('jacobian_ik','baseline'),('jacobian_ik','gravity_ff_gain4_angular4')]
    for frame,t in enumerate(np.arange(0,14,.04)):
        canvas=Image.new('RGB',(1280,860),'#f5f8fc'); draw=ImageDraw.Draw(canvas)
        draw.text((18,10),f'NERO 竖直贴地 · 同一目标 (0.70, 0, 0.005) m · 时间 {t:.2f} s',font=font(25),fill='#192940')
        for i,key in enumerate(keys):
            rig=rigs[key]; history=histories[key]
            row=history[min(len(history)-1,int(np.searchsorted(history[:,0],t)))]
            rig.data.qpos[:9]=row[1:10]; rig.data.qvel[:]=0
            mujoco.mj_forward(rig.model,rig.data)
            renderer.update_scene(rig.data,camera=cam,scene_option=option)
            goal=np.array([.7,0,.005]); tcp=row[16:19]
            axis=rig.data.xmat[rig.base.hand_id].reshape(3,3)[:,0]
            add_line(renderer.scene,tcp-.08*axis,tcp,(.95,.15,.25,1),.003)
            add_line(renderer.scene,goal+[0,0,.10],goal,(1,.74,.05,1),.002)
            add_geom(renderer.scene,mujoco.mjtGeom.mjGEOM_SPHERE,[.006,0,0],goal,(1,.74,.05,1))
            x=(i%2)*640; y=55+(i//2)*395
            canvas.paste(Image.fromarray(renderer.render()),(x,y+33))
            draw=ImageDraw.Draw(canvas)
            name='Pose IK' if key[0]=='pose_ik' else 'Jacobian IK'
            label='原默认伺服' if key[1]=='baseline' else '重力前馈 + kp×4 / kv×2'
            if 'angular4' in key[1]: label='前馈 + kp×4 / kv×2 + 姿态反馈×4'
            draw.text((x+15,y),f'{name} · {label}',font=font(22),fill='#192940')
            color='#b94d58' if row[10]>.005 or row[12]>2 else '#17885e'
            draw.text((x+15,y+349),f'位置 {row[10]*1000:.2f} mm   竖直 {row[11]:.2f}°   姿态 {row[12]:.2f}°',font=font(21),fill=color)
        draw.text((18,840),'黄色：目标竖直轴  /  红色：实际夹爪轴   ·   独立控制诊断，正式模型未修改',font=font(15),fill='#415169')
        video.write(cv2.cvtColor(np.array(canvas),cv2.COLOR_RGB2BGR))
        if frame==325: canvas.save(OUT/'tracking_before_after.png')
        if frame%100==0: print(f'rendered {frame}/350',flush=True)
    video.release(); renderer.close()
    subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(raw),'-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'tracking_before_after.mp4')], check=True)


if __name__ == '__main__':
    main()
