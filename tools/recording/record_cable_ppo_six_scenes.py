"""Correct the scene list to the frozen PPO trainer's six scenes.

Reuse four identical nominal recordings, add the two missing bank scenes.
Never alter the original recorder, scene physics, training or cable placement.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import multiprocessing as mp
from pathlib import Path
import re
import urllib.request
import numpy as np
import record_cable_workspace_scenes as base

ROOT=base.ROOT
PREVIOUS=base.OUT
OUT=ROOT/'outputs/vertical_workspace/scripted_136_retest_20261004/cable_ppo_six_scenes'
TRAIN=ROOT/'outputs/rl/train/ppo_targetnode_full6_workspace_20261004'
BANK=ROOT/'outputs/rl/train/ppo_nero_targetnode_state6_warmstart3m_20260928/eval_banks'
TRAIN_BANK=TRAIN/'frozen_code/outputs/motion_banks/replay_src_shape_nominal_v1.npz'
SCENES=('id_static','id_rigid_l1_nominal','id_shape_nominal_current','id_combined_l1_nominal','id_static_midshape_v1','id_rigid_replay_shape_nominal')
LABELS=('静态','整体运动 L1','局部形变','运动＋形变 L1','复杂形状静态','形变轨迹刚体回放')
ADDED=SCENES[4:]
original_sources=base.source_hashes
base.OUT=OUT;base.CORE_SCENARIO_NAMES=SCENES;base.NAMES=LABELS

def sources():
    extra=(Path(__file__),TRAIN/'train_targetnode_full6_workspace_eval13.py',
        BANK/'replay_src_shape_nominal_v1.npz',BANK/'replay_src_shape_nominal_v1.manifest.json')
    return {**original_sources(),**{str(p):base.sha(p) for p in extra}}

def env_with_bank(scene):
    registered=base.get_scenario(scene)
    config=base.EnvConfig(robot='nero',seed=base.SEED,scenario_name=scene,
        scenario_id=registered.scenario_id,scenario_split=registered.split.value,
        replay_bank_dir=str(BANK),replay_seed_fallback='error',**registered.to_env_overrides())
    env=base.CableGraspEnv(config);env.reset(seed=base.SEED)
    return env

base.source_hashes=sources;base.make_env=env_with_bank

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    trainer=(TRAIN/'train_targetnode_full6_workspace_eval13.py').read_text()
    import ast
    tree=ast.parse(trainer)
    actual=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SCENES' for t in n.targets))
    assert tuple(actual)==SCENES
    manifest=dict(scenes=SCENES,labels=LABELS,seed=base.SEED,workspace_x_m=[.6,.84],workspace_y_m=[-.22,.3],
        trainer_source=str(TRAIN/'train_targetnode_full6_workspace_eval13.py'),
        reused_scenes=SCENES[:4],new_recordings=ADDED,
        bank_for_paired_seed=str(BANK/'replay_src_shape_nominal_v1.npz'),
        motion_difficulty='full nominal, not a particular training curriculum stage',
        placement='original, no translation; no policy/workspace-conditioned reset rejection',
        source_sha256=sources())
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')

def gallery():
    rows=[]
    for i,s in enumerate(SCENES):
        folder=OUT/s if s in ADDED else PREVIOUS/s
        meta=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
        assert (folder/'video_verification.json').exists()
        rows.append(dict(meta,label=LABELS[i],directory=s if s in ADDED else f'../{PREVIOUS.name}/{s}',
            capture_origin='new physical simulation' if s in ADDED else 'reused exact same nominal scene/seed recording'))
    (OUT/'gallery_manifest.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
    html=base.GALLERY.replace('__DATA__',json.dumps(rows,ensure_ascii=False))
    html=html.replace('线缆六场景与候选工作区','PPO训练六场景与候选工作区').replace('线缆六场景与工作区','PPO训练六场景与工作区')
    html=html.replace('原目标节点','本回合节点').replace('红点为该回合原本选中的目标节点','红点为本次环境默认抽到的材料节点，并非PPO筛选后的训练目标')
    html=html.replace('静态和形变最多15秒；整体运动场景沿用越过原出口线即结束','静态、形变和轨迹回放最多15秒；L1沿用越过原出口线即结束')
    html=re.sub(r'<select id="group">.*?</select>',
        '<select id="group"><option value="all">PPO全部六场景</option><option value="shape">静态 / 形变</option><option value="rigid">整体运动L1 / 叠加运动L1</option><option value="combined">复杂静态 / 刚体回放</option></select>',html,count=1)
    html=html.replace('shape:[0,1],rigid:[2,3],combined:[4,5]','shape:[0,2],rigid:[1,3],combined:[4,5]')
    cards=''.join(f'<article class="player" id="card-{i}"><h2>{r["label"]}</h2><video id="video-{i}" controls playsinline preload="auto" aria-label="{r["label"]}录像"></video><p id="meta-{i}"></p></article>' for i,r in enumerate(rows))
    html=html.replace('__PLAYERS__',cards)
    html=html.replace('场景与来源记录</a>','场景与来源记录</a> · <a href="x_placement_analysis.json" target="_blank">X摆位统计</a>')
    (OUT/'index.html').write_text(html,encoding='utf-8')
    report(rows)

def report(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import Rectangle
    points=[np.load(OUT/r['directory']/'trajectory.npz')['cable_xyz'] for r in rows]
    lo=np.min([p[...,:2].min(axis=(0,1)) for p in points],axis=0)
    hi=np.max([p[...,:2].max(axis=(0,1)) for p in points],axis=0)
    lo=np.floor((lo-.06)*10)/10;hi=np.ceil((hi+.06)*10)/10
    fig,axes=plt.subplots(2,3,figsize=(16,11),sharex=True,sharey=True)
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc',size=13)
    stats=[];checks=[]
    for row,p,ax in zip(rows,points,axes.flat):
        node=row['target_node'];ax.scatter(p[::5,:,0].ravel(),p[::5,:,1].ravel(),s=2,alpha=.13,color='#1686bb')
        ax.plot(p[:,node,0],p[:,node,1],color='#e85039',lw=1.4)
        ax.add_patch(Rectangle((.6,-.22),.24,.52,fill=False,edgecolor='#c49a00',lw=2.5))
        ax.scatter(.35,0,color='#333',marker='s',s=25);ax.set_title(row['label'],fontproperties=font)
        ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_aspect('equal');ax.grid(alpha=.2)
        ax.tick_params(labelbottom=True,labelleft=True);ax.set_xlabel('World X (m)');ax.set_ylabel('World Y (m)')
        stats.append(dict(scene=row['scene'],initial_node_mean_x_m=float(p[0,:,0].mean()),
            time_node_mean_x_m=float(p[:,:,0].mean()),center_alignment_offset_x_m=.72-float(p[:,:,0].mean())))
        folder=OUT/row['directory'];v=json.loads((folder/'video_verification.json').read_text())
        assert v['rendered_fullphysics_state_max_difference']==row['robot_qpos_max_change']==row['robot_link_position_max_change_m']==0
        assert row['robot_cable_contact_substep_pairs']==0
        for view in base.VIEWS:
            url=urllib.request.urljoin('http://127.0.0.1:8766/cable_ppo_six_scenes/',row['directory']+'/'+view+'.mp4')
            with urllib.request.urlopen(urllib.request.Request(url,headers={'Range':'bytes=0-15'})) as response:
                assert response.status==206 and len(response.read())==16
                checks.append(dict(scene=row['scene'],view=view,http_status=206))
    fig.suptitle('PPO训练六场景：蓝色为整缆节点轨迹；红色为本次抽到的材料节点；黄色为候选工作区\n机械臂固定；seed 20260804；原始摆位；XY投影',fontproperties=FontProperties(fname='C:/Windows/Fonts/msyh.ttc',size=16))
    fig.tight_layout();fig.savefig(OUT/'motion_overview.png',dpi=160);plt.close(fig)
    with np.load(TRAIN_BANK) as bank:
        entries=bank['positions_xy'].copy();valid=bank['valid_steps'].copy();initial=bank['placed_com_xy'][:,0].copy()
    means=np.array([p[:int(n),:,0].mean() for p,n in zip(entries,valid)])
    frontier=[]
    for offset in np.round(np.arange(0,.301,.01),2):
        fraction=float(np.mean([((p[:int(n),:,0]+offset>=.6)&(p[:int(n),:,0]+offset<=.84)&
            (p[:int(n),:,1]>=-.22)&(p[:int(n),:,1]<=.30)).mean() for p,n in zip(entries,valid)]))
        frontier.append(dict(offset_x_m=float(offset),mean_node_frame_xy_inside_fraction=fraction))
    bank_summary=dict(bank=str(TRAIN_BANK),sha256=base.sha(TRAIN_BANK),entries=len(entries),
        initial_com_x_mean_m=float(initial.mean()),time_node_x_mean_m=float(means.mean()),
        episode_mean_x_q10_q50_q90_m=np.quantile(means,[.1,.5,.9]).tolist(),
        center_alignment_offset_x_m=.72-float(means.mean()),
        best_geometric_offset=max(frontier,key=lambda r:r['mean_node_frame_xy_inside_fraction']),
        offset_frontier=frontier,
        scope='training bank source-shape trajectories, equal entry weights; not all six trained episodes or task success',
        shift_test='offline XY translation only, not a physical rerun')
    (OUT/'x_placement_analysis.json').write_text(json.dumps(dict(workspace_center_x_m=.72,
        single_seed_scene_statistics=stats,training_shape_bank_statistics=bank_summary),indent=2),encoding='utf-8')
    (OUT/'verification.json').write_text(json.dumps(dict(correct_six_scenes=SCENES,clips=18,
        four_scenes_reused=True,two_bank_scenes_new=True,all_robot_poses_fixed=True,all_rendered_states_exact=True,
        byte_range_checks=checks),indent=2),encoding='utf-8')
    print(json.dumps(dict(correct_scenes=6,clips=18,training_bank=bank_summary)),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('prepare','render','gallery','all'),default='all');args=parser.parse_args()
    if args.mode in ('prepare','all'):
        prepare()
        with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as pool:list(pool.map(base.simulate,ADDED))
    if args.mode in ('render','all'):
        with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as pool:list(pool.map(base.render,ADDED))
    if args.mode in ('gallery','all'):gallery()

if __name__=='__main__':main()
