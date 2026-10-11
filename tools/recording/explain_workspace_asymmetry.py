"""Separate controller-from-home failure from fixed-pose geometric feasibility."""
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from PIL import Image,ImageDraw

import record_improved_workspace as w
import render_improved_workspace as visual
from panda_cable_grasp.env.kinematics import rotation_to_quat,quat_error
from render_vertical_workspace import font

OUT=w.OUT


def probe():
    rig=w.ImprovedRig();geometry=rig.geometry;goal_quat=rig.quat
    old=np.load(w.ROOT/'outputs/vertical_workspace/nero_20261001/workspace.npz')
    scan={(r['x_m'],r['y_m']):r for r in json.loads((OUT/'cells.json').read_text())}
    rng=np.random.default_rng(20261001);results=[]
    for y in (-.28,.28):
        target=np.array([.67,y,.005])
        ix=np.argmin(abs(old['xs']-.67));iy=np.argmin(abs(old['ys']-y))
        starts=[geometry.home,old['q'][iy,ix]]+[rng.uniform(geometry.lo+.03,geometry.hi-.03) for _ in range(14)]
        def residual(q):
            geometry.set(q)
            error=quat_error(rotation_to_quat(geometry.data.xmat[geometry.hand].reshape(3,3)),goal_quat)
            return np.r_[geometry.tcp()-target,.18*error]
        found=None
        for start_index,start in enumerate(starts):
            if not np.isfinite(start).all():continue
            fit=least_squares(residual,np.clip(start,geometry.lo+1e-6,geometry.hi-1e-6),
                bounds=(geometry.lo,geometry.hi),max_nfev=250,gtol=1e-9,ftol=1e-8,xtol=1e-8)
            residual(fit.x)
            geometry.data.qpos[rig.base.finger_qpos_adr]=(.05,-.05)
            mujoco.mj_forward(geometry.model,geometry.data)
            measurement=rig.measure(target,collision=True)
            if measurement[0]<.001 and measurement[2]<.6 and measurement[3]>.0002 and measurement[5]>=.001:
                found={'q_rad':fit.x.tolist(),'position_error_mm':float(measurement[0]*1000),
                    'full_pose_error_deg':float(measurement[2]),'robot_floor_mm':float(measurement[3]*1000),
                    'joint_margin_deg':float(measurement[4]),'self_distance_mm':float(measurement[5]*1000),
                    'start_index':start_index}
                break
        rig.reset();row,_=rig.trial(.67,y,False)
        margin=np.minimum(rig.data.qpos[rig.base.arm_qpos_adr]-geometry.lo,geometry.hi-rig.data.qpos[rig.base.arm_qpos_adr])
        worst=int(np.argmin(margin))
        result={'target_m':target.tolist(),'fixed_full_pose_geometric_solution':found,
            'frozen_scan_result':scan[.67,y],
            'repeat_from_home_matches_scan':all(row[k]==scan[.67,y][k] for k in ('endpoint_pass','motion_pass','steps','reason')),
            'near_limit_joint_from_home':rig.base.robot_spec.arm_joint_names[worst],
            'joint_margin_deg_from_home':float(np.degrees(margin[worst]))}
        if found:
            # This is only a local hold diagnostic. Teleporting to an IK solution
            # cannot establish a safe trajectory and does not modify any map.
            rig.reset();rig.data.qpos[rig.base.arm_qpos_adr]=found['q_rad'];rig.data.qvel[:]=0
            rig.data.ctrl[:7]=found['q_rad'];rig.base._last_applied_action[:]=rig.data.ctrl
            rig.base._last_requested_action[:]=rig.data.ctrl;rig.base._previous_arm_command_velocity[:]=0
            mujoco.mj_forward(rig.model,rig.data)
            metrics=[]
            for _ in range(150):rig.control(target);metrics.append(rig.measure(target,collision=True))
            tail=np.asarray(metrics[-100:])
            hold=bool(np.all((tail[:,0]<=.005)&(tail[:,2]<=2)&(tail[:,3]>=.0002)&(tail[:,3]<=.008)&(tail[:,4]>=0)&(tail[:,5]>=.001)))
            result['solution_initialized_hold_diagnostic']={'passed_last_2s':hold,
                'max_position_error_mm':float(tail[:,0].max()*1000),'max_pose_error_deg':float(tail[:,2].max()),
                'initialization':'IK solution placed directly; local hold only, not a movement test'}
        results.append(result);print(json.dumps(result,indent=2),flush=True)
    evidence={'grid_y_symmetric':True,'grid_x_symmetric_about_base':True,'mirrored_targets':results,
        'method':'bounded multistart SciPy fixed full pose, same z=5mm and opening, followed by local hold diagnostic',
        'original_motion_scan_unchanged':True,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'asymmetry_diagnostic.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')


def figure():
    d,summary,_=visual.load()
    old=np.load(w.ROOT/'outputs/vertical_workspace/nero_20261001/workspace.npz')
    oldmap={'xs':old['xs'],'ys':old['ys'],'endpoint_pass':old['reachable'],
        'motion_pass':np.zeros_like(old['reachable']),'interior':np.zeros_like(old['reachable'])}
    image=Image.new('RGB',(1440,990),'#f5f8fc');dd=ImageDraw.Draw(image)
    dd.text((25,17),'两张图测的是不同范围：几何可达 vs 当前控制器试跑通过',font=font(30),fill='#182b43')
    dd.text((30,79),'旧 SciPy：允许水平转向、多初值找解',font=font(25),fill='#287ca4')
    dd.text((740,79),'新 Jacobian：固定朝向、同一起点',font=font(25),fill='#188569')
    image.paste(visual.map_image(oldmap,700),(0,125));image.paste(visual.map_image(d,700),(720,125))
    dd=ImageDraw.Draw(image)
    lines=[('蓝色圆环：存在合格姿态解；未验证实际进场路径。', '#287ca4'),
        ('绿色：当前固定朝向、home、路径和时间预算下，进场与保持通过。','#188569'),
        ('新图空白不代表几何不可达；机器人没有因为提高反馈增益而变成偏心结构。','#415169'),
        ('旧图：TCP 2.5 mm / 2 cm网格 / yaw自由；新图：TCP 5 mm / 4 cm网格 / yaw固定。','#415169')]
    for i,(txt,color) in enumerate(lines):dd.text((28,825+i*37),txt,font=font(23),fill=color)
    image.save(OUT/'workspace_scope_comparison.png')


if __name__=='__main__':probe();figure()
