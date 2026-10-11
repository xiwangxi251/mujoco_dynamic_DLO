"""120 fresh cable-only episodes for the explicit cable_x075_v1 family.

Reuse the verified native 1280x960/50 fps recording implementation, with
per-process scene/seed/output adapters. Original galleries stay intact.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import multiprocessing as mp
from pathlib import Path
import time
import urllib.request
import numpy as np
import record_cable_workspace_scenes as base
from panda_cable_grasp.scenarios.registry import CABLE_X075_SCENARIO_NAMES

ROOT = base.ROOT
OUT = ROOT / 'outputs/vertical_workspace/scripted_136_retest_20261004/cable_x075_20episodes'
BANK = ROOT / 'outputs/rl/train/ppo_nero_targetnode_state6_warmstart3m_20260928/eval_banks'
SCENES = CABLE_X075_SCENARIO_NAMES
LABELS = ('静态', '整体运动 L1', '局部形变', '运动＋形变 L1', '复杂形状静态', '形变轨迹刚体回放')
SEEDS = tuple(range(20260804, 20260824))
ORIGINAL_SOURCES = base.source_hashes
ORIGINAL_GET_SCENARIO = base.get_scenario


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def sources():
    extra = (Path(__file__), BANK/'replay_src_shape_nominal_v1.npz', BANK/'replay_src_shape_nominal_v1.manifest.json')
    return {**ORIGINAL_SOURCES(), **{str(p): base.sha(p) for p in extra}}


def job_directory(scene, episode):
    return OUT / scene / f'episode_{episode:02d}'


def record_job(job):
    scene, episode, seed = job
    dest = job_directory(scene, episode)
    scene_config = ORIGINAL_GET_SCENARIO(scene)
    label = LABELS[SCENES.index(scene)] + f' · 回合 {episode:02d}/20 · X基准0.75m'
    key = f'episode_{episode:02d}'
    base.OUT = OUT/scene
    base.SEED = seed
    base.CORE_SCENARIO_NAMES = (key,)
    base.NAMES = (label,)
    base.source_hashes = sources
    base.get_scenario = lambda name: scene_config

    def make_env(name):
        config = base.EnvConfig(robot='nero', seed=seed, scenario_name=scene,
            scenario_id=scene_config.scenario_id, scenario_split=scene_config.split.value,
            replay_bank_dir=str(BANK), replay_seed_fallback='error', **scene_config.to_env_overrides())
        env = base.CableGraspEnv(config)
        env.reset(seed=seed)
        return env
    base.make_env = make_env
    verification = dest/'video_verification.json'
    if verification.exists():
        meta = json.loads((dest/'metadata.json').read_text(encoding='utf-8'))
        if meta['source_sha256'] == sources() and all((dest/f'{v}.mp4').exists() for v in base.VIEWS):
            return {'scene': scene, 'episode': episode, 'seed': seed, 'resumed': True}
        raise RuntimeError(f'Existing recording has different provenance: {dest}')
    start = time.monotonic()
    meta = base.simulate(key)
    meta.update(scene=scene, episode=episode, family='cable_x075_v1',
        directory=f'{scene}/{key}', initial_cable_center_x_m=0.75,
        paired_scenario_seed=True, ready=True)
    saved = np.load(dest/'trajectory.npz')
    points = saved['cable_xyz']
    assert np.all(np.isfinite(points))
    meta.update(initial_center_xyz=points[0].mean(axis=0).tolist(),
        initial_xy_min=points[0,:,:2].min(axis=0).tolist(), initial_xy_max=points[0,:,:2].max(axis=0).tolist(),
        cable_node_z_min_m=float(points[...,2].min()), cable_node_z_max_m=float(points[...,2].max()),
        initial_nodes_inside=int(base.inside(points[0]).sum()),
        finite_cable_states=True)
    write_json(dest/'metadata.json', meta)
    base.render(key)
    return {'scene': scene, 'episode': episode, 'seed': seed, 'wall_s': round(time.monotonic()-start, 1)}


def rows():
    result = []
    for episode, seed in enumerate(SEEDS, 1):
        for scene, label in zip(SCENES, LABELS):
            dest = job_directory(scene, episode)
            if (dest/'video_verification.json').exists():
                row = json.loads((dest/'metadata.json').read_text(encoding='utf-8'))
            else:
                row = dict(scene=scene, label=label+f' · 回合 {episode:02d}/20', seed=seed,
                    episode=episode, directory=f'{scene}/episode_{episode:02d}', ready=False,
                    duration_s=15., target_node=0, target_inside_recorded_frame_fraction=0., mean_nodes_inside_fraction=0.)
            result.append(row)
    return result


def gallery():
    data = rows()
    ready = [r for r in data if r['ready']]
    write_json(OUT/'gallery_manifest.json', data)
    html = base.GALLERY
    html = html.replace('线缆六场景与工作区', 'X=0.75 场景族 · 120回合检查')
    html = html.replace('线缆六场景与候选工作区', 'X=0.75 场景族 · 每场景20回合')
    html = html.replace('机械臂固定在原 ready 姿态、夹爪张开；线缆沿用当前场景的位置、驱动和材料。',
        '新场景族 cable_x075_v1：X基准0.750m，保留±0.100m随机偏移。机械臂固定在原 ready 姿态、夹爪张开。')
    html = html.replace('每个场景同一 seed，', '同一回合六场景使用配对seed，20组seed=20260804～20260823；')
    html = html.replace('整体运动 L1 / L2', '整体运动 L1 / 叠加运动 L1').replace('叠加运动 L1 / L2', '复杂静态 / 刚体回放')
    html = html.replace('这里展示一个种子的活动范围，不是全部随机回合的包络。',
        '共120个新仿真回合。红点是环境原始随机节点，不是PPO工作区筛选后的训练目标。选择不同回合逐一检查。')
    html = html.replace('六场景完整运动范围俯视图', '120回合初始位置及录制检查数据').replace('motion_overview.png', 'summary.json')
    options = ''.join(f'<option value="{i}">回合 {i:02d}/20 · seed {seed}</option>' for i,seed in enumerate(SEEDS,1))
    html = html.replace('<div class="toolbar">', '<p id="batch-status">已完成 '+str(len(ready))+'/120 回合</p><div class="toolbar"><label>回合 <select id="episode">'+options+'</select></label>')
    html = html.replace("const D=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id),groups={all:[0,1,2,3,4,5],shape:[0,1],rigid:[2,3],combined:[4,5]};",
        "const ALL=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id),groups={all:[0,1,2,3,4,5],shape:[0,2],rigid:[1,3],combined:[4,5]};let D=ALL.filter(e=>e.episode===1);")
    html = html.replace("if(ids.includes(i)){v.src=", "if(ids.includes(i)&&e.ready){v.src=")
    html = html.replace("$('meta-'+i).innerHTML=`seed", "if(!e.ready){v.removeAttribute('src');v.removeAttribute('poster');v.load();}$('meta-'+i).innerHTML=!e.ready?'本回合正在录制，完成后刷新页面':`seed")
    html = html.replace("u.searchParams.set('group',$('group').value);", "u.searchParams.set('episode',$('episode').value);u.searchParams.set('group',$('group').value);")
    html = html.replace("$('group').addEventListener('change',load);", "$('episode').addEventListener('change',()=>{D=ALL.filter(e=>e.episode===Number($('episode').value));load()});$('group').addEventListener('change',load);")
    html = html.replace("load();tick();", "if(ALL.some(e=>e.episode===Number(params.get('episode')))){$('episode').value=params.get('episode');D=ALL.filter(e=>e.episode===Number($('episode').value));}load();tick();")
    cards = ''.join(f'<article class="player" id="card-{i}"><h2>{label}</h2><video id="video-{i}" controls playsinline preload="auto" aria-label="{label}录像"></video><p id="meta-{i}"></p></article>' for i,label in enumerate(LABELS))
    html = html.replace('__DATA__', json.dumps(data, ensure_ascii=False)).replace('__PLAYERS__', cards)
    temporary = OUT/'index.html.tmp';temporary.write_text(html, encoding='utf-8');temporary.replace(OUT/'index.html')
    summary = dict(family='cable_x075_v1',expected_episodes=120,completed_episodes=len(ready),expected_clips=360,completed_clips=len(ready)*3,
        scenes=SCENES, seeds=SEEDS, native_resolution=[1280,960],fps=50,
        fixed_robot=True, no_grasp_policy=True, workspace_x_m=[.6,.84],workspace_y_m=[-.22,.30],
        original_scene_definitions_unchanged=True,
        per_scene=[dict(scene=s,completed=sum(r['scene']==s for r in ready)) for s in SCENES],
        episodes=[{k:r[k] for k in ('scene','episode','seed','initial_center_xyz','initial_target_xyz','whole_cable_xy_min','whole_cable_xy_max','cable_node_z_min_m','cable_node_z_max_m','robot_qpos_max_change','robot_link_position_max_change_m','robot_cable_contact_substep_pairs','termination','frames','directory')} for r in ready])
    write_json(OUT/'summary.json', summary)
    return summary


def final_verify():
    data = rows();assert len(data)==120 and all(r['ready'] for r in data)
    checks=[]
    for row in data:
        dest=OUT/row['directory']
        meta=json.loads((dest/'metadata.json').read_text(encoding='utf-8'))
        report=json.loads((dest/'video_verification.json').read_text())
        assert meta['robot_qpos_max_change']==meta['robot_link_position_max_change_m']==0
        assert meta['finite_cable_states'] and meta['initial_cable_center_x_m']==.75
        assert meta['source_sha256']==sources() and base.sha(dest/'trajectory.npz')==report['trajectory_sha256']
        assert report['rendered_fullphysics_state_max_difference']==0
        assert len(report['encoding'])==3 and all(v['maximum_gop_frames']<=12 and v['uniform_20ms'] for v in report['encoding'])
        for view in base.VIEWS:
            url='http://127.0.0.1:8766/cable_x075_20episodes/'+row['directory']+'/'+view+'.mp4'
            req=urllib.request.Request(url,headers={'Range':'bytes=0-15'})
            with urllib.request.urlopen(req, timeout=20) as response:
                assert response.status==206 and len(response.read())==16
                assert response.headers['Content-Range'].startswith('bytes 0-15/')
                checks.append(dict(scene=row['scene'],episode=row['episode'],view=view,status=206))
    write_json(OUT/'verification.json', dict(episodes=120,clips=360,all_states_finite=True,
        all_robot_poses_fixed=True,all_rendered_states_exact=True,all_encoding_verified=True,byte_range_checks=checks))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--limit',type=int);parser.add_argument('--mode',choices=('all','gallery','verify'),default='all');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.mode=='all':
        bank=np.load(BANK/'replay_src_shape_nominal_v1.npz');assert set(SEEDS)<=set(map(int,bank['seeds']))
        jobs=[(scene,episode,seed) for episode,seed in enumerate(SEEDS,1) for scene in SCENES]
        if args.limit:jobs=jobs[:args.limit]
        write_json(OUT/'manifest.json',dict(family='cable_x075_v1',scenes=SCENES,seeds=SEEDS,
            expected_episodes=120,expected_clips=360,workers=args.workers,source_sha256=sources(),
            fixed_robot='9 qpos restored after each 2ms step; robot velocities and warmstart zero; cable DOFs unchanged',
            capture='fresh simulation per episode; independent native render of saved real 20ms states',
            no_policy=True,no_workspace_conditioned_reset=True,bank_seed_fallback='error'))
        gallery();start=time.monotonic()
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool:
            futures=[pool.submit(record_job,job) for job in jobs]
            for future in as_completed(futures):
                result=future.result();summary=gallery()
                print(json.dumps(dict(completed=summary['completed_episodes'],of=120,wall_s=round(time.monotonic()-start,1),last=result)),flush=True)
        if not args.limit:final_verify()
    elif args.mode=='verify':final_verify()
    summary=gallery();print(json.dumps(dict(done=summary['completed_episodes'],clips=summary['completed_clips'])),flush=True)


if __name__=='__main__':main()
