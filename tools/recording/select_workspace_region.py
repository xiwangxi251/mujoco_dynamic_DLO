"""Select a conservative rectangle and measure specific continuous witnesses."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import record_improved_workspace as w
import render_improved_workspace as visual
from render_vertical_workspace import font

OUT=w.OUT/'recommended_rectangle'
BOUNDS=(.67,.83,-.12,.12)

def main():
    OUT.mkdir(exist_ok=True)
    d,_,plan=visual.load()
    visual.verify(plan)
    rows=json.loads((w.OUT/'cells.json').read_text())
    x0,x1,y0,y1=BOUNDS
    mask=(d['xs'][None,:]>=x0)&(d['xs'][None,:]<=x1)&(d['ys'][:,None]>=y0)&(d['ys'][:,None]<=y1)
    assert mask.sum()==35 and d['interior'][mask].all()
    selected=[r for r in rows if x0<=r['x_m']<=x1 and y0<=r['y_m']<=y1]
    corners=np.array([[x0,y0,w.Z],[x1,y0,w.Z],[x1,y1,w.Z],[x0,y1,w.Z]])
    # Both perimeter directions, then both diagonals; these are finite witnesses.
    goals=np.array([corners[i] for i in (1,2,3,0,3,2,1,0,2,1,3)])
    rig=w.ImprovedRig()
    initial,h=rig.trial(x0,y0,True)
    assert initial['motion_pass'],initial
    reference=corners[0].copy(); history=list(h);stage=[0]*len(h)
    valid=True;failure=None
    def step(ref,phase):
        rig.control(ref)
        m=rig.measure(ref,collision=True)
        finger=min(float(rig.geometry.geom_vertices(g)[:,2].min()) for g in rig.geometry.finger_geoms)
        history.append(np.r_[rig.data.time,rig.data.qpos[:9],m,ref,ref,finger]);stage.append(phase)
        return bool(m[0]<=.005 and m[2]<=2 and .0002<=finger<=.008 and m[3]>=.0002 and m[4]>=0 and m[5]>=.001)
    for index,goal in enumerate(goals):
        while np.linalg.norm(goal-reference)>1e-9:
            delta=goal-reference
            reference+=delta*min(1.,.02*w.DT/max(np.linalg.norm(delta),1e-12))
            if not step(reference,1):
                valid=False;failure={'segment':index,'reference_m':reference.tolist(),'record_index':len(history)-1};break
        if not valid:break
    if valid:
        for _ in range(100):
            if not step(reference,2):valid=False;failure={'stage':'final_hold'};break
    hist=np.asarray(history);stages=np.array(stage);ground=hist[stages>0]
    peak=float(np.max(rig.peak_velocity/rig.base._arm_velocity_limits))
    valid=valid and peak<=1.01
    result={'world_x_m':[x0,x1],'world_y_m':[y0,y1],'world_tcp_z_m':w.Z,
        'base_offset_m':[.35,0,0],'rectangle_area_m2':(x1-x0)*(y1-y0),
        'grid_samples':len(selected),'all_grid_samples_motion_and_interior_pass':True,
        'minimum_endpoint_joint_margin_deg':min(r['joint_margin_deg'] for r in selected),
        'grid_step_m':.04,'interior_margin_definition':'EDT of sampled motion mask >= 6 cm; no continuum guarantee',
        'continuous_witness_pass':bool(valid),'continuous_failure':failure,
        'continuous_reference_speed_m_s':.02,'sim_seconds':float(hist[-1,0]),
        'witness_description':'clockwise and counterclockwise perimeter plus both diagonals, same fixed-yaw frozen Jacobian controller',
        'ground_max_position_error_mm':float(ground[:,10].max()*1000),
        'ground_max_full_pose_error_deg':float(ground[:,12].max()),
        'ground_finger_clearance_mm':[float(ground[:,25].min()*1000),float(ground[:,25].max()*1000)],
        'ground_min_joint_margin_deg':float(ground[:,14].min()),
        'peak_velocity_to_limit':peak,'actual_velocity_exceedance_substeps':rig.over_steps,
        'scope':'unloaded fixed-yaw simulation; finite samples and paths, not a hardware or dynamic grasp guarantee'}
    np.savez_compressed(OUT/'continuous_witness.npz',history=hist,stage=stages,goals=goals)
    (OUT/'selection.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)
    image=Image.new('RGB',(1080,890),'#f5f8fc');dd=ImageDraw.Draw(image)
    dd.text((25,18),'建议先用：Y=0 对称的内部矩形',font=font(29),fill='#182b43')
    size=760;chart=visual.map_image(d,size);draw=ImageDraw.Draw(chart)
    pad=.12*size;scale=.76*size/1.36
    def xy(x,y):return pad+(x+.33)*scale,size-pad-(y+.68)*scale
    draw.line([xy(x,y) for x,y,_ in np.vstack((corners,corners[0]))],fill='#d94670',width=5)
    draw.text(xy(.46,-.20),'推荐矩形',font=font(23),fill='#c72c56')
    image.paste(chart,(0,75));dd=ImageDraw.Draw(image)
    for i,line in enumerate(['世界坐标 / m',f'X：{x0:.2f} ～ {x1:.2f}',f'Y：{y0:+.2f} ～ {y1:+.2f}', 'TCP Z：0.005', '尺寸：16 × 24 cm','面积：0.0384 m²','35 个网格点均通过','边界双向 + 两条对角线', '连续补查：'+('通过' if valid else '失败'), '参考速度：2 cm/s']):
        dd.text((755,160+i*43),line,font=font(22),fill='#273c53')
    dd.text((25,849),'当前固定朝向、空载控制器的实验候选区；离散采样不保证每条连续轨迹。',font=font(22),fill='#455266')
    image.save(OUT/'recommended_rectangle.png')

if __name__=='__main__':main()
