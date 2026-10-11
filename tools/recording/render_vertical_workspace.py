"""Measure and record NERO's near-floor, vertically downward workspace.

Standalone kinematic scene; never changes the grasp environment or its physics.
Run from the repository root with PYTHONPATH=src.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy import ndimage

from panda_cable_grasp.env import ROBOT_SPECS
from panda_cable_grasp.env.environment import XML_PATH, _robot_assets
from panda_cable_grasp.env.kinematics import point_jacobian


def make_model():
    robot = ROBOT_SPECS['nero']
    root = ET.fromstring(XML_PATH.read_text(encoding='utf-8'))
    for tag in ('extension', 'contact'):
        for item in root.findall(tag):
            root.remove(item)
    world = root.find('worldbody')
    for tag in ('composite', 'site'):
        for item in world.findall(tag):
            world.remove(item)
    spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding='unicode'),
        include={'robot.xml': robot.xml_path.read_bytes()},
        assets=_robot_assets(str(robot.asset_dir)))
    next(b for b in spec.bodies if b.name == robot.base_body_name).pos[:] += robot.base_offset
    for tex in spec.textures:
        if tex.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            tex.rgb2[:] = (0.16, 0.23, 0.32)
        elif tex.name == 'table_tex':
            tex.rgb1[:] = (0.43, 0.46, 0.48)
            tex.rgb2[:] = (0.36, 0.39, 0.42)
    model = spec.compile()
    model.vis.global_.offwidth = 1280
    model.vis.global_.offheight = 720
    model.vis.headlight.diffuse[:] = (0.8, 0.8, 0.8)
    return model


class Workspace:
    def __init__(self):
        self.model = make_model()
        self.data = mujoco.MjData(self.model)
        self.spec = ROBOT_SPECS['nero']
        self.joints = np.array([self.model.joint(n).id for n in self.spec.arm_joint_names])
        self.qadr = self.model.jnt_qposadr[self.joints]
        self.dadr = self.model.jnt_dofadr[self.joints]
        self.lo, self.hi = self.model.jnt_range[self.joints].T
        self.hand = self.model.body(self.spec.hand_body_name).id
        self.offset = np.array(self.spec.grasp_center_local)
        self.visual_geoms = [i for i in range(self.model.ngeom)
            if self.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
            and self.model.geom_group[i] == 1]
        self.vertices = {}
        for g in self.visual_geoms:
            mid = self.model.geom_dataid[g]
            a, n = self.model.mesh_vertadr[mid], self.model.mesh_vertnum[mid]
            self.vertices[g] = self.model.mesh_vert[a:a+n].copy()
        self.finger_geoms = [g for g in self.visual_geoms
            if self.model.body(self.model.geom_bodyid[g]).name.startswith('gripper_link')]
        self.home = np.array(self.spec.ready_arm_qpos)
        self.set(self.home)

    def set(self, q):
        self.data.qpos[self.qadr] = q
        for n, val in zip(self.spec.finger_joint_names, (0.035, -0.035)):
            self.data.qpos[self.model.jnt_qposadr[self.model.joint(n).id]] = val
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def tcp(self):
        r = self.data.xmat[self.hand].reshape(3, 3)
        return self.data.xpos[self.hand] + r @ self.offset

    def geom_vertices(self, g):
        return self.vertices[g] @ self.data.geom_xmat[g].reshape(3, 3).T + self.data.geom_xpos[g]

    def rotation(self, yaw=0.):
        # Mesh measurement: wrist is at negative local X relative to the TCP,
        # and the lowest finger vertices are at local X ~= 0. +X must face down.
        r = np.array([[0., -1., 0.], [0., 0., 1.], [-1., 0., 0.]])
        return Rotation.from_euler('z', yaw).as_matrix() @ r

    def jacobian(self):
        jp, jr = point_jacobian(self.model, self.data, self.hand, self.offset)
        a = self.data.xmat[self.hand].reshape(3, 3)[:, 0]
        ja = np.cross(jr[:, self.dadr].T, a).T
        return np.vstack((jp[:, self.dadr], 0.18 * ja))

    def solve(self, target, starts, attempts=3):
        def fun(q):
            self.set(q)
            a = self.data.xmat[self.hand].reshape(3, 3)[:, 0]
            return np.r_[self.tcp() - target, 0.18 * (a - [0., 0., -1.])]
        for start in starts[:attempts]:
            result = least_squares(fun, np.clip(start, self.lo + 1e-7, self.hi - 1e-7),
                jac=lambda q: self.jac_at(q), bounds=(self.lo, self.hi),
                max_nfev=65, ftol=2e-5, xtol=2e-5, gtol=1e-7)
            self.set(result.x)
            error = float(np.linalg.norm(self.tcp() - target))
            tilt = math.degrees(math.acos(np.clip(-self.data.xmat[self.hand].reshape(3, 3)[2, 0], -1, 1)))
            if error < 0.001 and tilt < 0.6:
                floor = self.floor_clearance()
                # Solve position tightly; near-floor clearance must still be positive.
                if floor < 0.0005:
                    continue
                if self.self_collision():
                    continue
                s = np.linalg.svd(self.jacobian(), compute_uv=False)
                margin = float(np.min(np.minimum(result.x-self.lo, self.hi-result.x)))
                return result.x, [error, tilt, floor, math.degrees(margin), float(s[-2])]
        return None, None

    def jac_at(self, q):
        self.set(q)
        return self.jacobian()

    def refine(self, d, out):
        """Use redundant joints to improve margins, then recheck real geometry.

        A small centering penalty selects among near-equivalent IK solutions;
        the original position/tilt/collision thresholds are still enforced.
        """
        center=(self.lo+self.hi)/2
        width=(self.hi-self.lo)/2
        count=0
        for iy,ix in zip(*np.where(d['reachable'])):
            oldq=d['q'][iy,ix].copy()
            oldm=d['metrics'][iy,ix].copy()
            if oldm[3]>=8 and oldm[4]>=.05:
                continue
            target=np.array([d['xs'][ix],d['ys'][iy],float(d['z'])])
            weight=.0007
            def fun(q):
                self.set(q)
                a=self.data.xmat[self.hand].reshape(3,3)[:,0]
                return np.r_[self.tcp()-target,.18*(a-[0.,0.,-1.]),weight*(q-center)/width]
            def jac(q):
                self.set(q)
                return np.vstack([self.jacobian(),weight*np.diag(1/width)])
            res=least_squares(fun,oldq,jac=jac,bounds=(self.lo,self.hi),
                max_nfev=85,ftol=1e-6,xtol=1e-6,gtol=1e-9)
            self.set(res.x)
            err=np.linalg.norm(self.tcp()-target)
            tilt=math.degrees(math.acos(np.clip(-self.data.xmat[self.hand].reshape(3,3)[2,0],-1,1)))
            margin=math.degrees(float(np.min(np.minimum(res.x-self.lo,self.hi-res.x))))
            sing=np.linalg.svd(self.jacobian(),compute_uv=False)[-2]
            score=min(margin/5,sing/.035)
            oldscore=min(oldm[3]/5,oldm[4]/.035)
            if err<.001 and tilt<.6 and score>oldscore and self.floor_clearance()>.0005 and not self.self_collision():
                d['q'][iy,ix]=res.x
                d['metrics'][iy,ix]=[err,tilt,self.floor_clearance(),margin,sing]
                count+=1
            if (iy*len(d['xs'])+ix)%200==0:
                print(f'margin refinement: {count} improved',flush=True)
        step=float(d['xs'][1]-d['xs'][0])
        edge=ndimage.distance_transform_edt(d['reachable'])*step
        d['recommended']=d['reachable'] & (edge>=.04) & (d['metrics'][:,:,3]>=5) & (d['metrics'][:,:,4]>=.035)
        np.savez_compressed(out/'workspace.npz',**d)
        print(f'margin refinement done: {count} improved; {d["recommended"].sum()} recommended',flush=True)

    def floor_clearance(self):
        return min(float(self.geom_vertices(g)[:, 2].min()) for g in self.visual_geoms
            if self.model.body(self.model.geom_bodyid[g]).name != 'base_link')

    def self_collision(self):
        # Distance queries deliberately bypass the stock model's disabled arm
        # collision masks. Mesh distances use MuJoCo's convex mesh hulls.
        for i, g in enumerate(self.visual_geoms):
            b1 = self.model.geom_bodyid[g]
            for h in self.visual_geoms[i+1:]:
                b2 = self.model.geom_bodyid[h]
                a, b = int(b1), int(b2)
                da, db = 0, 0
                ancestors = {a: 0}
                while a:
                    a = int(self.model.body_parentid[a]); da += 1; ancestors[a] = da
                while b not in ancestors:
                    b = int(self.model.body_parentid[b]); db += 1
                if db + ancestors[b] <= 2:
                    continue
                distance = mujoco.mj_geomDistance(self.model, self.data, g, h, 0.002, None)
                if distance < 0.001:
                    return True
        return False

    def scan(self, out, step=0.02):
        rng = np.random.default_rng(20261001)
        xs = np.round(np.arange(-0.55, 1.251, step), 5)
        ys = np.round(np.arange(-0.90, 0.901, step), 5)
        qmap = np.full((len(ys), len(xs), 7), np.nan)
        metrics = np.full((len(ys), len(xs), 5), np.nan)
        z = 0.0025
        start_time = time.monotonic()
        for iy, y in enumerate(ys):
            order = range(len(xs)) if iy % 2 == 0 else range(len(xs)-1, -1, -1)
            last = None
            for ix in order:
                x = xs[ix]
                if math.hypot(x-0.35, y) > 0.85:
                    continue
                radial = self.home.copy()
                radial[0] = np.clip(self.home[0] + math.atan2(y, x-0.35), self.lo[0]+0.01, self.hi[0]-0.01)
                starts = []
                if last is not None:
                    starts.append(last)
                if iy and np.isfinite(qmap[iy-1, ix]).all():
                    starts.append(qmap[iy-1, ix])
                starts += [radial, self.home, rng.uniform(self.lo+0.02, self.hi-0.02)]
                q, m = self.solve(np.array([x, y, z]), starts, attempts=4)
                if q is not None:
                    qmap[iy, ix] = q
                    metrics[iy, ix] = m
                    last = q
            if iy % 5 == 0:
                print(f'row {iy+1}/{len(ys)}: {np.isfinite(qmap[:,:,0]).sum()} reachable; {time.monotonic()-start_time:.1f}s', flush=True)
                np.savez_compressed(out/'scan_partial.npz', xs=xs, ys=ys, q=qmap, metrics=metrics, z=z)
        mask = np.isfinite(qmap[:, :, 0])
        edge_distance = ndimage.distance_transform_edt(mask) * step
        robust = mask & (edge_distance >= 0.04) & (metrics[:, :, 3] >= 5.0) & (metrics[:, :, 4] >= 0.035)
        np.savez_compressed(out/'workspace.npz', xs=xs, ys=ys, q=qmap, metrics=metrics,
            z=z, reachable=mask, recommended=robust, edge_distance=edge_distance)
        return dict(xs=xs, ys=ys, q=qmap, metrics=metrics, z=z, reachable=mask, recommended=robust)


def font(size):
    return ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', size)


def bounds(d, mask):
    iy, ix = np.where(mask)
    if len(ix) == 0:
        return None
    radial=np.hypot(d['xs'][ix]-.35,d['ys'][iy])
    return {'x_m': [float(d['xs'][ix].min()), float(d['xs'][ix].max())],
            'y_m': [float(d['ys'][iy].min()), float(d['ys'][iy].max())],
            'base_radial_m': [float(radial.min()),float(radial.max())],
            'area_m2': float(mask.sum() * (d['xs'][1]-d['xs'][0])**2), 'cells': int(mask.sum())}


def largest_rectangle(mask):
    heights = np.zeros(mask.shape[1], int)
    best, rect = 0, None
    for y, row in enumerate(mask):
        heights = np.where(row, heights+1, 0)
        stack = []
        for x in range(len(heights)+1):
            h = heights[x] if x < len(heights) else 0
            left = x
            while stack and stack[-1][1] > h:
                l, hh = stack.pop()
                if hh*(x-l) > best:
                    best = hh*(x-l); rect = (l, x-1, y-hh+1, y)
                left = l
            if not stack or stack[-1][1] < h:
                stack.append((left, h))
    return rect


def plan_image(d, size=600, rect=None, point=None, path=None):
    im = Image.new('RGB', (size, size), '#f5f8fc')
    draw = ImageDraw.Draw(im)
    pad = int(size*0.105)
    scale = (size-2*pad)/1.8
    def xy(x,y):
        return pad+(x+.55)*scale, size-pad-(y+.9)*scale
    step = float(d['xs'][1]-d['xs'][0])
    for iy, ix in zip(*np.where(d['reachable'])):
        x,y = float(d['xs'][ix]), float(d['ys'][iy])
        p1=xy(x-step/2,y+step/2); p2=xy(x+step/2,y-step/2)
        draw.rectangle([p1,p2], fill='#4bc4eb' if not d['recommended'][iy,ix] else '#39b98c')
    small = font(max(10,int(size*.025)))
    for v in np.arange(-.4,1.21,.2):
        a,b=xy(v,-.9),xy(v,.9)
        draw.line([a,b],fill='#bec8d4',width=1)
        draw.text((a[0]-12,a[1]+5),f'{v:.1f}',font=small,fill='#445064')
    for v in np.arange(-.8,.81,.2):
        a,b=xy(-.55,v),xy(1.25,v)
        draw.line([a,b],fill='#bec8d4',width=1)
        draw.text((4,a[1]-8),f'{v:.1f}',font=small,fill='#445064')
    contours,_ = cv2.findContours(d['reachable'].astype('uint8'),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        pts=[xy(d['xs'][x],d['ys'][y]) for x,y in contour[:,0]]
        if len(pts)>1:
            draw.line(pts+[pts[0]],fill='#1685b3',width=max(1,int(size*.003)))
    b=xy(.35,0); rad=size*.021
    draw.ellipse((b[0]-rad,b[1]-rad,b[0]+rad,b[1]+rad),fill='#26324d')
    draw.text((b[0]+rad,b[1]-rad), '底座',font=small,fill='#26324d')
    if rect is not None:
        x0,x1,y0,y1=rect
        draw.rectangle([xy(x0,y1),xy(x1,y0)],outline='#e0a320',width=max(2,int(size*.006)))
    if point is not None:
        a=xy(*point[:2]); r=size*.012
        draw.ellipse((a[0]-r,a[1]-r,a[0]+r,a[1]+r),fill='#ef486b',outline='white',width=2)
    if path is not None:
        draw.line([xy(p[0],p[1]) for p in path],fill='#e0a320',width=max(2,int(size*.005)))
    draw.text((size//2-38,size-25),'世界 X / m',font=small,fill='#27374b')
    draw.text((8,5),'世界 Y / m',font=small,fill='#27374b')
    return im


def make_path(w, d, rect):
    # A broad front-side annular sweep shows sideways travel and both near/far
    # reaches without pretending a bounding rectangle is fully reachable.
    coarse=[]
    for radius,angles in [(.49,np.linspace(-80,80,120)),(.33,np.linspace(80,-80,120))]:
        for angle in angles:
            a=math.radians(angle)
            coarse.append([.35+radius*math.cos(a),radius*math.sin(a)])
    # Close the loop and sample uniformly by arc length (2 traversals in video).
    coarse=np.array(coarse+[coarse[0]])
    lengths=np.r_[0,np.cumsum(np.linalg.norm(np.diff(coarse,axis=0),axis=1))]
    s=np.linspace(0,lengths[-1],600)
    xy=np.column_stack([np.interp(s,lengths,coarse[:,i]) for i in (0,1)])
    ix=int(np.argmin(abs(d['xs']-xy[0,0]))); iy=int(np.argmin(abs(d['ys']-xy[0,1])))
    previous=d['q'][iy,ix]
    qs=[]; actual=[]; ms=[]
    for i,p in enumerate(xy):
        ix=int(np.argmin(abs(d['xs']-p[0]))); iy=int(np.argmin(abs(d['ys']-p[1])))
        q,m=w.solve(np.r_[p,float(d['z'])],[previous,d['q'][iy,ix],w.home],attempts=3)
        if q is None:
            raise RuntimeError(f'continuous near-floor path failed at {p}')
        if qs and np.max(np.abs(q-previous))>.13:
            raise RuntimeError(f'IK branch jump at {p}: {np.max(np.abs(q-previous))}')
        qs.append(q); actual.append(w.tcp().copy()); ms.append(m); previous=q
    return np.array(qs),np.array(actual),np.array(ms)


def add_geom(scene, kind, size, pos, rgba, mat=None):
    g=scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g,kind,np.asarray(size,float),np.asarray(pos,float),
        np.eye(3).reshape(-1) if mat is None else mat,np.asarray(rgba,float))
    scene.ngeom+=1


def add_line(scene, a, b, color, width=.0025):
    g=scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),
        np.eye(3).reshape(-1),np.array(color,float))
    mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_CAPSULE,width,np.asarray(a,float),np.asarray(b,float))
    scene.ngeom+=1


def render(w, d, out):
    step=float(d['xs'][1]-d['xs'][0])
    # Place the illustrative, fully checked rectangle on the task-facing side.
    rectmask=d['recommended'].copy()
    rectmask[:,d['xs']<.43]=False
    r=largest_rectangle(rectmask)
    if r is None:
        raise RuntimeError('No recommended front workspace found')
    ix0,ix1,iy0,iy1=r
    rect=[float(d['xs'][ix0]),float(d['xs'][ix1]),float(d['ys'][iy0]),float(d['ys'][iy1])]
    qs,actual,ms=make_path(w,d,rect)
    np.savez_compressed(out/'demonstration_path.npz', q=qs, tcp=actual, metrics=ms, rectangle=rect)
    summary={'robot':'NERO','world_base_xyz_m':list(w.spec.base_offset),'plane_z_m':0.0,
        'tcp_z_m':float(d['z']),'gripper_opening_command_m':.035,
        'grid_step_m':step,'yaw':'free, +X of link7 constrained vertically downward',
        'constraints':{'position_error_m':.001,'tilt_error_deg':.6,'floor_clearance_m':.0005,
            'nonadjacent_convex_mesh_separation_m':.001,'self_collision_skip':'body tree distance <= 2'},
        'recommended_rule':{'distance_to_sampled_edge_m':.04,'joint_limit_margin_deg':5.,
            'weighted_5d_jacobian_min_singular_value':.035,'orientation_weight_m':.18},
        'reachable':bounds(d,d['reachable']),'recommended':bounds(d,d['recommended']),
        'largest_scanned_recommended_front_rectangle_world_xy_m':rect,
        'demonstration_path_world_xy_bounds_m':[[float(actual[:,0].min()),float(actual[:,0].max())],
            [float(actual[:,1].min()),float(actual[:,1].max())]],
        'demonstration':{'frames':len(qs),'max_position_error_mm':float(ms[:,0].max()*1000),
            'max_tilt_deg':float(ms[:,1].max()),'min_floor_clearance_mm':float(ms[:,2].min()*1000),
            'min_joint_limit_margin_deg':float(ms[:,3].min()),
            'max_joint_frame_delta_deg':float(np.degrees(np.max(np.abs(np.diff(qs,axis=0)))))} ,
        'scope':'Sampled kinematic workspace, not a guaranteed dynamic/real-hardware workspace; bounding boxes include holes and must not be treated as entirely reachable.',
        'method':'Bounded multi-start 5-DoF IK (free yaw); official mesh vertices for floor; explicit convex mesh distance for non-neighbor collisions. Numerical failures may miss feasible configurations.'}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'reachable_samples.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f)
        writer.writerow(['x_m','y_m','tcp_z_m','recommended','position_error_m','tilt_deg','floor_clearance_m','joint_margin_deg','min_singular_value',*[f'joint{i}_rad' for i in range(1,8)]])
        for iy,ix in zip(*np.where(d['reachable'])):
            writer.writerow([d['xs'][ix],d['ys'][iy],d['z'],int(d['recommended'][iy,ix]),*d['metrics'][iy,ix],*d['q'][iy,ix]])
    plan_image(d,900,path=actual).save(out/'workspace_top.png')
    image=Image.new('RGB',(1280,1000),'#f5f8fc')
    image.paste(plan_image(d,900,path=actual),(0,70))
    draw=ImageDraw.Draw(image)
    draw.text((30,16),'NERO · 夹爪竖直贴地工作区域',font=font(32),fill='#15253c')
    for i,(txt,col) in enumerate([('绿色：留有余量','#39b98c'),('蓝色：采样可达','#24a9dc'),('黄色：视频演示路径','#c48a0c')]):
        draw.text((925,110+i*45),txt,font=font(22),fill=col)
    for i,txt in enumerate(['指尖离地约 2.5 mm','竖直向下；水平转向自由',f'网格：{step*100:.0f} cm','检查关节限位 / 穿地', '检查非相邻连杆碰撞','底座世界坐标 (0.35, 0)', '边界为数值采样近似', '包围框内包含不可达空洞']):
        draw.text((925,280+i*38),txt,font=font(18),fill='#33435c')
    for i,key in enumerate(['reachable','recommended']):
        b=summary[key]; label='可达包围范围' if key=='reachable' else '余量区域包围范围'
        lines=[label,f"X {b['x_m'][0]:+.2f} ~ {b['x_m'][1]:+.2f} m",f"Y {b['y_m'][0]:+.2f} ~ {b['y_m'][1]:+.2f} m",f"面积约 {b['area_m2']:.3f} m²"]
        for j,txt in enumerate(lines):
            draw.text((925,650+i*145+j*30),txt,font=font(18),fill='#33435c')
    image.save(out/'workspace_map.png')
    renderer=mujoco.Renderer(w.model,height=720,width=1280,max_geom=12000)
    cam=mujoco.MjvCamera()
    cam.lookat[:]=(.35,0,.15); cam.distance=2.18; cam.azimuth=135; cam.elevation=-38
    option=mujoco.MjvOption(); option.geomgroup[3]=0
    fps=24
    raw=out/'nero_vertical_workspace_raw.mp4'
    writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1280,720))
    if not writer.isOpened():
        raise RuntimeError('Video writer failed')
    map_base=plan_image(d,340,path=actual)
    contour,_=cv2.findContours(d['reachable'].astype('uint8'),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for frame in range(720):
        idx=min(599,max(0,frame-48))
        w.set(qs[idx]); mujoco.mj_forward(w.model,w.data)
        renderer.update_scene(w.data,camera=cam,scene_option=option)
        scene=renderer.scene
        # Discrete tiles show the actual nonconvex workspace including holes.
        for iy,ix in zip(*np.where(d['reachable'])):
            color=(.12,.76,.48,1) if d['recommended'][iy,ix] else (.15,.64,.9,1)
            add_geom(scene,mujoco.mjtGeom.mjGEOM_BOX,[step*.46,step*.46,.00035],
                [d['xs'][ix],d['ys'][iy],.0007],color)
        for c in contour:
            pts=c[:,0]
            for a,b in zip(pts,np.roll(pts,-1,axis=0)):
                add_line(scene,[d['xs'][a[0]],d['ys'][a[1]],.003],
                    [d['xs'][b[0]],d['ys'][b[1]],.003],(.0,.3,.58,1),.0015)
        # Coordinate grid and the measured demonstration trajectory.
        for v in np.arange(-.8,.81,.2):
            add_line(scene,[-.5,v,.0017],[1.2,v,.0017],(.8,.85,.9,1),.0007)
        for v in np.arange(-.4,1.21,.2):
            add_line(scene,[v,-.85,.0017],[v,.85,.0017],(.8,.85,.9,1),.0007)
        for a,b in zip(actual[::12],actual[12::12]):
            add_line(scene,[a[0],a[1],.006],[b[0],b[1],.006],(1,.72,.05,1),.003)
        add_geom(scene,mujoco.mjtGeom.mjGEOM_SPHERE,[.008,0,0],actual[idx]+[0,0,.005],(1,.2,.35,1))
        rgb=renderer.render()
        im=Image.fromarray(rgb)
        overlay=Image.new('RGBA',im.size,(0,0,0,0)); dd=ImageDraw.Draw(overlay)
        dd.rounded_rectangle((16,15,896,114),radius=13,fill=(248,251,255,235))
        dd.text((30,23),'NERO 竖直贴地：可达范围与余量区域',font=font(28),fill='#15253c')
        dd.text((30,65),'指尖离地约 2.5 mm  |  水平转向自由  |  底座 (0.35, 0) m',font=font(21),fill='#33435c')
        dd.rounded_rectangle((918,15,1271,707),radius=14,fill=(248,251,255,248))
        im=Image.alpha_composite(im.convert('RGBA'),overlay).convert('RGB')
        im.paste(map_base,(925,65))
        dd=ImageDraw.Draw(im)
        dd.text((940,25),'俯视范围 · 世界坐标 / m',font=font(20),fill='#15253c')
        p=actual[idx]; px=925+.105*340+(p[0]+.55)*(340*.79/1.8); py=65+340-.105*340-(p[1]+.9)*(340*.79/1.8)
        dd.ellipse((px-4,py-4,px+4,py+4),fill='#ef486b',outline='white',width=1)
        lines=[('绿色  留有余量的区域','#179369'),('蓝色  采样可达的区域','#1685b3'),('黄色  已验证演示路径','#b28110'),
            (f'X {p[0]:+.3f} m   Y {p[1]:+.3f} m','#27374b'),
            (f'实际竖直误差 {ms[idx,1]:.2f}°','#27374b'),
            (f'最低几何离地 {ms[idx,2]*1000:.2f} mm','#27374b'),
            ('2 cm 网格 · 运动学回放','#536479'),
            ('范围包含空洞，请以色块为准','#536479')]
        for j,(txt,col) in enumerate(lines):
            dd.text((935,422+j*33),txt,font=font(19 if j<6 else 17),fill=col)
        dd.rounded_rectangle((22,658,891,707),radius=10,fill='#f5f8fc')
        dd.text((38,668),'绿色标准：距边界 ≥4 cm，关节余量 ≥5°，避开低灵活度姿态',font=font(21),fill='#27374b')
        writer.write(cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR))
        if frame in (0,250,550):
            im.save(out/f'checkframe_{frame:04d}.png')
        if frame%120==0:
            print(f'video frame {frame}/720',flush=True)
    writer.release(); renderer.close()
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

    def probe(self):
        tcp = self.tcp()
        r = self.data.xmat[self.hand].reshape(3, 3)
        out = {'tcp_home': tcp.tolist(), 'rotation_home': r.tolist(), 'geometry': []}
        for g in self.visual_geoms:
            verts = (self.geom_vertices(g) - tcp) @ r
            out['geometry'].append({'body': self.model.body(self.model.geom_bodyid[g]).name,
                'local_min_from_tcp': verts.min(0).tolist(), 'local_max_from_tcp': verts.max(0).tolist()})
        return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--probe', action='store_true')
    p.add_argument('--scan', action='store_true')
    p.add_argument('--render', action='store_true')
    p.add_argument('--refine', action='store_true')
    p.add_argument('--output', type=Path, default=Path('outputs/vertical_workspace/nero_20261001'))
    p.add_argument('--step', type=float, default=0.02)
    args = p.parse_args()
    w = Workspace()
    if args.probe:
        print(json.dumps(w.probe(), indent=2))
    if args.scan:
        args.output.mkdir(parents=True, exist_ok=True)
        w.scan(args.output, args.step)
    if args.render or args.refine:
        with np.load(args.output/'workspace.npz') as data:
            d={k:data[k] for k in data.files}
        if args.refine:
            w.refine(d,args.output)
        if args.render:
            render(w,d,args.output)


if __name__ == '__main__':
    main()
