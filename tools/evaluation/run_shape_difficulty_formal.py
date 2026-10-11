"""Frozen, resumable complex-static shape study; no speculative rollouts."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools/evaluation'), str(ROOT/'tools/recording')]
import mujoco
import numpy as np
from panda_cable_grasp.env.environment import CableGraspEnv, EnvConfig
from panda_cable_grasp.env.control import configure_profile
from panda_cable_grasp.env.reference_shape import set_current_cable_reference
from panda_cable_grasp.policies.scripted import PolicyConfig
from panda_cable_grasp.scenarios.registry import get_scenario
from shape_selection_pilot import GeometrySelector, features, save, load, sha
from run_node_feedback_family import NodeOnlyEnv
from shape_difficulty_overnight import max_nonadjacent_capsule_penetration
from record_scripted_scene_family import project_camera_manifest
SCENE = 'dev_static_midshape_v1_x075_v1'
BANK = ROOT/'outputs/rl/train/ppo_targetnode_full6_workspace_20261004/frozen_code/outputs/motion_banks'


def now():
    return datetime.now(timezone.utc)


def prepare(out, count, regression=False):
    if (out/'manifest.json').exists():
        raise RuntimeError('Existing manifest; use its frozen entrypoint to resume')
    out.mkdir(parents=True, exist_ok=True)
    with np.load(BANK/'replay_src_shape_nominal_v1.npz', allow_pickle=False) as bank:
        seeds, valid = bank['seeds'].copy(), bank['valid_steps'].copy()
        bank_meta = json.loads(str(bank['meta_json']))
    assert len(seeds) == len(np.unique(seeds)) == 300
    rng = np.random.default_rng(2026101001)
    order = rng.permutation(300)
    jobs, used = [], set()
    for i in range(count):
        entry = int(order[i % 300])
        while True:
            fraction = float(rng.uniform(.30, .70))
            frame = min(int(fraction*int(valid[entry])), int(valid[entry])-1)
            if (entry, frame) not in used:
                used.add((entry, frame)); break
        jobs.append(dict(sample_id=f'sample_{i:04d}', source_entry=entry, seed=int(seeds[entry]),
                         snapshot_fraction=fraction, snapshot_step=frame, bank=str(BANK)))
    if regression:
        jobs[0].update(seed=20260805, snapshot_fraction=.35, snapshot_step=262,
                       bank=str(ROOT/'outputs/rl/train/ppo_nero_targetnode_state6_warmstart3m_20260928/eval_banks'),
                       regression_case=True)
    policy = asdict(PolicyConfig(preserve_selected_material_target=True))
    source = out/'source'
    for directory in ('src', 'assets', 'tools/evaluation', 'tools/recording'):
        shutil.copytree(ROOT/directory, source/directory,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'), dirs_exist_ok=True)
    hashes = {p.relative_to(source).as_posix(): sha(p) for p in source.rglob('*') if p.is_file()}
    protocol = dict(jobs=jobs, bank=str(BANK), bank_sha256=sha(BANK/'replay_src_shape_nominal_v1.npz'),
                    bank_entries=300, bank_meta=bank_meta, sampling_seed=2026101001,
                    sampling='Balanced 3/4 per source, uniform fraction in [0.30,0.70); unique source/frame',
                    policy_config=policy, controller='enhanced6_bounded_dls_v1', table_guard=True,
                    arm_wait_s=.82, episode_seconds=15, penetration_filter_mm=5,
                    penetration_measurement='Actual transformed capsule surfaces, ignore index gap <=2',
                    reference='Current shape as zero-stress elastic reference; gravity/contact/stiffness unchanged',
                    displacement_screen=False, recording='All passing episodes: integration states, actions, targets, reference model, named cameras. Representative videos after analysis.',
                    source_hashes=hashes, commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT,text=True).strip(),
                    worktree_dirty=True, regression=regression,
                    stages=['sampling_and_penetration_screen', 'rule_grasp_with_saved_states',
                            'trajectory_grouped_difficulty_analysis', 'representative_dual_view_videos'])
    protocol['configuration_hash'] = hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    save(out/'manifest.json', protocol)
    save(out/'status.json', dict(status='starting', run=str(out), planned=count, completed=0,
        configuration_hash=protocol['configuration_hash'], heartbeat_at=now().isoformat(),
        last_progress_at=now().isoformat(), progress_deadline_at=(now()+timedelta(minutes=10)).isoformat()))


def destination(out, job):
    return out/'episodes'/SCENE/job['sample_id']


def worker(out, index):
    manifest = load(out/'manifest.json'); job = manifest['jobs'][index]
    dest = destination(out, job); dest.mkdir(parents=True, exist_ok=True)
    if (dest/'result.json').exists():
        result = load(dest/'result.json')
        assert result['configuration_hash'] == manifest['configuration_hash']
        return result
    started = time.monotonic()
    overrides = get_scenario(SCENE).to_env_overrides()
    overrides['initial_shape_time_range'] = (job['snapshot_fraction'],)*2
    cfg = configure_profile(EnvConfig(robot='nero', seed=job['seed'], scenario_name=SCENE,
        replay_bank_dir=job['bank'], replay_seed_fallback='error', episode_seconds=15,
        **overrides), 'enhanced6_bounded_dls_v1')
    cfg.table_execution_guard_enabled = True
    env = CableGraspEnv(cfg)
    try:
        env.reset(seed=job['seed'])
        assert env._initial_shape_step == job['snapshot_step'], (env._initial_shape_step, job)
        if manifest.get('factor_extension'):
            from shape_difficulty_factors import transform_snapshot
            transform_snapshot(env, job.get('layout', {}))
        penetration = max_nonadjacent_capsule_penetration(env)
        result = dict(**job, scene=SCENE, directory=dest.relative_to(out).as_posix(),
            configuration_hash=manifest['configuration_hash'], initial_penetration_mm=1000*penetration,
            filtered=penetration>.005, initial_features=features(env.data.xpos[env.cable_ids]))
        save(dest/'ready.json', dict(pid=os.getpid(), ready_at=now().isoformat(), sample_id=job['sample_id'],
                                    configuration_hash=manifest['configuration_hash']))
        if result['filtered']:
            result.update(task_success=None, wall_seconds=time.monotonic()-started)
            save(dest/'result.json', result); return result
        baked = set_current_cable_reference(env)
        np.savez_compressed(dest/'reference_model.npz', body_quat=env.model.body_quat.copy())
        if manifest.get('factor_extension'):
            from shape_difficulty_factors import StudySelector
            policy = StudySelector(NodeOnlyEnv(env), PolicyConfig(**manifest['policy_config']), job)
        else:
            policy = GeometrySelector(NodeOnlyEnv(env), PolicyConfig(**manifest['policy_config']))
        states, times, cable, targets, phases, actions, quats, selected, locked = ([] for _ in range(9))
        kind = mujoco.mjtState.mjSTATE_INTEGRATION
        changes, settled = [], None
        def capture():
            mujoco.mj_forward(env.model, env.data)
            state = np.empty(mujoco.mj_stateSize(env.model, kind))
            mujoco.mj_getState(env.model, env.data, state, kind)
            states.append(state); times.append(float(env.data.time))
            cable.append(env.data.xpos[env.cable_ids].copy())
            ids = policy.locked_node_ids or env.cable_ids
            if policy.locked_segment_index is None:
                point = env.data.xpos[policy.selected_target_body_id].copy(); material = float(policy.selected_target_index)
            else:
                j, a = policy.locked_segment_index, policy.locked_segment_alpha
                point = (1-a)*env.data.xpos[ids[j]]+a*env.data.xpos[ids[j+1]]; material = j+a
                assert abs(material-policy.selected_target_index)<1e-9, 'Unannounced material target switch'
            targets.append(point.copy()); phases.append(policy.phase.name)
            quats.append(policy.desired_quat.copy()); selected.append(policy.selected_target_index); locked.append(material)
            if not changes or changes[-1]['phase'] != policy.phase.name:
                changes.append(dict(time_s=times[-1], phase=policy.phase.name,
                    selected_node=policy.selected_target_index, material_point=material))
        capture()
        while not policy.finished:
            action = policy.action(); actions.append(action.copy())
            _, _, terminated, truncated, _ = env.step(action); capture()
            if settled is None and float(env.data.time)>=.82-1e-9:
                settled = features(cable[-1])
            if terminated or truncated: break
        np.savez_compressed(dest/'trajectory.npz', states=states, frame_times=times,
            frame_indices=np.arange(len(states)), cable_xyz=cable, tracked_target_xyz=targets,
            native_actions=actions, desired_quats=quats, phases=phases, selected_nodes=selected,
            locked_material_points=locked, cable_body_ids=env.cable_ids, state_kind=int(kind))
        initial_candidates = policy.selection_events[0]['candidates']
        good = [c for c in initial_candidates if c['in_workspace'] and c['in_execution_limits']
                and c['swept_mesh_box_clearance_mm']>=1 and c['branches']<=1 and c['crossing_angle_deg']>=60]
        result.update(task_success=bool(env.ever_success), duration_s=float(env.data.time),
            retries=policy.retry_count, policy_result=policy.result, failure_reason=policy.last_attempt_failure,
            phase_changes=changes, selection_events=policy.selection_events, settled_features=settled,
            choice_features=dict(good_candidate_fraction=len(good)/len(initial_candidates),
                good_candidate_count=len(good), good_material_span=(max(c['node'] for c in good)-min(c['node'] for c in good))/(len(env.cable_ids)-1) if good else 0.,
                eligible_fraction=sum(c['in_workspace'] and c['in_execution_limits'] for c in initial_candidates)/len(initial_candidates)),
            wall_seconds=time.monotonic()-started, privilege_guard='passed', material_lock_guard='passed')
        save(dest/'state_metadata.json', dict(env_config=asdict(cfg), policy_config=asdict(policy.config),
            cameras=project_camera_manifest(cfg), trajectory_sha256=sha(dest/'trajectory.npz'),
            reference_model_file='reference_model.npz', reference_model_sha256=sha(dest/'reference_model.npz'),
            stress_free_setup=baked, configuration_hash=manifest['configuration_hash']))
        save(dest/'result.json', result)
        return result
    finally:
        env.close()


def auc(y, score):
    pos, neg = score[y==1], score[y==0]
    return None if not len(pos) or not len(neg) else float(np.mean((pos[:,None]>neg)+.5*(pos[:,None]==neg)))


def fit_predict(x, y, test):
    mean, scale = x.mean(0), np.maximum(x.std(0), 1e-8)
    a, b = np.c_[np.ones(len(x)),(x-mean)/scale], np.c_[np.ones(len(test)),(test-mean)/scale]
    weights = np.zeros(a.shape[1]); penalty = np.eye(len(weights)); penalty[0,0]=0.
    for _ in range(35):
        p = 1/(1+np.exp(-np.clip(a@weights,-30,30)))
        gradient = a.T@(p-y)+penalty@weights
        hessian = (a.T*(p*(1-p)))@a+penalty+np.eye(len(weights))*1e-8
        step = np.linalg.solve(hessian, gradient); weights -= step
        if np.max(np.abs(step))<1e-7: break
    return 1/(1+np.exp(-np.clip(b@weights,-30,30)))


def analyze(out, results):
    passed = [r for r in results if not r['filtered']]
    summary = dict(candidates=len(results), filtered=len(results)-len(passed), eligible=len(passed),
        successes=sum(r['task_success'] for r in passed), source_trajectories=len({r['seed'] for r in passed}),
        state_files=len(passed), episodes=passed)
    rows = [dict(sample_id=r['sample_id'], seed=r['seed'], fraction=r['snapshot_fraction'],
        filtered=r['filtered'], success=r.get('task_success'), penetration_mm=r['initial_penetration_mm'],
        duration_s=r.get('duration_s'), retries=r.get('retries'), **(r.get('settled_features') or r['initial_features']),
        **r.get('choice_features',{})) for r in results]
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (out/'per_sample.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    text = ['# 复杂静态：完整形状难度实验', '',
        f"候选 {len(results)}；初始穿透＞5 mm 筛除 {summary['filtered']}；合格 {len(passed)}，抓取成功 {summary['successes']}/{len(passed)}。",
        '', '300条源轨迹均衡抽样，30%–70%随机时刻。采用无应力参考形状，保留重力、接触和弹性；未做3秒位移筛选。',
        '按目标切线旋转、固定材料目标直到重试；增强6有界DLS、碰桌保护、0.82秒等待、15秒预算。全部合格回合保存可重渲染状态。',
        '', '以下是该规则策略和工作区下的经验难度，不能直接推广为形状的固有难度。']
    if passed:
        feature_keys = list(passed[0]['settled_features'])+list(passed[0]['choice_features'])
        x = np.array([[{**r['settled_features'],**r['choice_features']}[k] for k in feature_keys] for r in passed])
        y = np.array([not r['task_success'] for r in passed], dtype=float)
        groups = np.array([r['seed'] for r in passed]); sources = np.unique(groups)
        shuffled = np.random.default_rng(2026101002).permutation(sources)
        folds = {int(seed):i%5 for i,seed in enumerate(shuffled)}
        assignment=np.array([folds[int(g)] for g in groups]); predictions=np.full(len(y),np.nan)
        if len(np.unique(y))>1 and len(sources)>=10:
            for fold in range(5):
                test=assignment==fold;train=~test
                predictions[test]=fit_predict(x[train],y[train],x[test])
            summary.update(oof_failure_auc=auc(y,predictions),oof_brier=float(np.mean((y-predictions)**2)))
            text += ['', f"按源轨迹隔离的5折预测：失败AUC={summary['oof_failure_auc']:.3f}，Brier={summary['oof_brier']:.3f}。评分列为折外失败概率，未用自身源轨迹拟合。"]
            comparisons = {}
            global_count = len(passed[0]['settled_features'])
            for label, columns in (('global_shape_only',slice(0,global_count)),
                                   ('candidate_choice_only',slice(global_count,None))):
                score = np.full(len(y),np.nan)
                for fold in range(5):
                    test=assignment==fold; train=~test
                    score[test]=fit_predict(x[train,columns],y[train],x[test,columns])
                comparisons[label] = dict(auc=auc(y,score),brier=float(np.mean((y-score)**2)))
            summary['feature_group_comparison'] = comparisons
            text += [f"仅全局形状因素AUC={comparisons['global_shape_only']['auc']:.3f}；仅候选选择余量AUC={comparisons['candidate_choice_only']['auc']:.3f}。"]
            if summary['oof_failure_auc'] < .60:
                text += ['折外区分能力弱；当前评分不足以作为可靠的形状难度等级，只保留因素分析与原始结果。']
        else:
            text += ['', '成功/失败或独立源轨迹不足，不能建立经过验证的难度评分。']
        rating=[dict(sample_id=r['sample_id'],source_seed=r['seed'],fold=int(assignment[i]),
                task_success=r['task_success'],oof_failure_probability=float(predictions[i]) if np.isfinite(predictions[i]) else None) for i,r in enumerate(passed)]
        save(out/'difficulty_scores.json',rating)
        text += ['', '| 因素 | 最低四分位失败率 | 最高四分位失败率 |', '|---|---:|---:|']
        associations=[]
        rng=np.random.default_rng(2026101003)
        for j,k in enumerate(feature_keys):
            lo,hi=np.quantile(x[:,j],[.25,.75]); low=x[:,j]<=lo;high=x[:,j]>=hi
            difference=float(y[high].mean()-y[low].mean())
            # Bootstrap source trajectories, keeping their snapshots together.
            group_indices=[np.flatnonzero(groups==g) for g in sources];boot=[]
            for _ in range(300):
                indices=np.concatenate([group_indices[g] for g in rng.integers(0,len(sources),len(sources))])
                a,b=low[indices],high[indices]
                if a.any() and b.any():boot.append(float(y[indices][b].mean()-y[indices][a].mean()))
            ci=np.quantile(boot,[.025,.975]).tolist()
            associations.append(dict(feature=k,low_failure=float(y[low].mean()),high_failure=float(y[high].mean()),difference=difference,source_bootstrap_ci95=ci))
            text.append(f'| {k} | {y[low].mean():.1%} ({int(low.sum())}) | {y[high].mean():.1%} ({int(high.sum())}) |')
        save(out/'factor_associations.json',associations)
        summary['strongest_associations']=sorted(associations,key=lambda r:abs(r['difference']),reverse=True)[:4]
        text += ['', '分位组可因离散值而重叠；关联及评分不证明因果。样本来自300条轨迹，不能按1000条独立轨迹计算置信度。',
            '局部候选数量/分布与全局折叠、空间占据、拥挤、工作区覆盖同时分析。材料参考曲率已改变，结论限定于此复杂静态构造。']
    (out/'report.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    summary.update(report=str(out/'report.md'),per_sample_csv=str(out/'per_sample.csv'))
    save(out/'summary.json',summary)
    return summary


def progress(out,stage,completed,planned,**extra):
    save(out/'progress.json',dict(stage=stage,completed=completed,planned=planned,
        last_progress_at=now().isoformat(),**extra))


def coordinate(args):
    out=args.output;manifest=load(out/'manifest.json');jobs=manifest['jobs'];results=[]
    pending=[]
    for i,job in enumerate(jobs):
        path=destination(out,job)/'result.json'
        if path.exists():
            result=load(path);assert result['configuration_hash']==manifest['configuration_hash'];results.append(result)
        else:pending.append(i)
    progress(out,'evaluation',len(results),len(jobs))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        active={}
        while pending or active:
            while pending and len(active)<args.workers:
                i=pending.pop(0);active[pool.submit(worker,out,i)]=i
            done,_=wait(active,timeout=2,return_when=FIRST_COMPLETED)
            for future in done:
                active.pop(future);results.append(future.result())
                progress(out,'evaluation',len(results),len(jobs))
    results.sort(key=lambda r:r['sample_id'])
    progress(out,'analysis',len(results),len(jobs))
    summary=analyze(out,results)
    # Regression/preflight results are never included in the formal sample.
    if not manifest['regression']:
        from render_saved_node_family import render
        eligible=summary['episodes']
        successes=[i for i,r in enumerate(eligible) if r['task_success']]
        failures=[i for i,r in enumerate(eligible) if not r['task_success']]
        chosen=(successes[:1]+failures[:1]) or list(range(min(2,len(eligible))))
        progress(out,'recording',len(results),len(jobs),selected_videos=chosen)
        for index in chosen:
            render(out,index)
            progress(out,'recording',len(results),len(jobs),rendered_sample=eligible[index]['sample_id'])
        video_rows=[dict(sample_id=eligible[i]['sample_id'],seed=eligible[i]['seed'],directory=eligible[i]['directory'],success=eligible[i]['task_success']) for i in chosen]
        build_page(out,video_rows)
        summary.update(representative_videos=video_rows,gallery=str(out/'index.html'))
        save(out/'summary.json',summary)
    progress(out,'complete',len(results),len(jobs))


def build_page(out,rows):
    data=json.dumps(rows,ensure_ascii=False)
    page='''<!doctype html><meta charset="utf-8"><title>完整形状难度正式实验</title><style>body{font:16px system-ui;margin:24px;background:#edf1f5}video{width:48%;background:#111}input{width:70%}</style><h2>完整形状难度正式实验</h2><select id="s"></select><button id="p">播放/暂停</button><button id="c">复制标识</button><span id="ref"></span><p><input id="seek" type="range" min="0" max="15" step=".02"></p><video id="g" preload="metadata"></video><video id="w" preload="metadata"></video><p><a href="report.md">报告</a> · <a href="per_sample.csv">逐样本结果</a></p><script>const D=DATA,s=document.querySelector('#s'),g=document.querySelector('#g'),w=document.querySelector('#w'),seek=document.querySelector('#seek');D.forEach((x,i)=>s.add(new Option(`${x.sample_id} | seed=${x.seed} | success=${x.success}`,i)));function load(){let x=D[+s.value];g.src=x.directory+'/global.mp4';w.src=x.directory+'/wrist.mp4';seek.value=0} s.onchange=load;document.querySelector('#p').onclick=()=>{let play=g.paused;[g,w].forEach(v=>{if(play){v.removeAttribute('controls');v.play()}else{v.pause();v.controls=true}})};seek.oninput=()=>[g,w].forEach(v=>{v.pause();v.currentTime=+seek.value});function tick(){if(!g.paused){seek.value=g.currentTime;if(Math.abs(g.currentTime-w.currentTime)>.06&&!w.seeking)w.currentTime=g.currentTime}let x=D[+s.value];document.querySelector('#ref').textContent=`RUN | ${x.sample_id} | seed=${x.seed} | t=${(+seek.value).toFixed(2)}s`;requestAnimationFrame(tick)}document.querySelector('#c').onclick=()=>navigator.clipboard.writeText(document.querySelector('#ref').textContent);load();tick()</script>'''.replace('DATA',data).replace('RUN',out.name)
    (out/'index.html').write_text(page,encoding='utf8')


def supervise(args):
    out=args.output;manifest=load(out/'manifest.json');start=now();child=None;restarts=0
    log_path=out/'coordinator.log'
    def status(state,**extra):
        p=load(out/'progress.json') if (out/'progress.json').exists() else {}
        last=datetime.fromisoformat(p.get('last_progress_at',start.isoformat()))
        stage=p.get('stage','starting');deadline=last+timedelta(minutes=12 if stage=='recording' else 8)
        save(out/'status.json',dict(status=state,stage=stage,run=str(out),configuration_hash=manifest['configuration_hash'],
            planned=len(manifest['jobs']),completed=p.get('completed',0),started_at=start.isoformat(),
            supervisor_pid=os.getpid(),supervisor_start_at=start.isoformat(),child_pid=child.pid if child else None,
            child_start_at=child_started.isoformat() if child else None,heartbeat_at=now().isoformat(),
            heartbeat_timeout_seconds=180,last_progress_at=last.isoformat(),progress_deadline_at=deadline.isoformat(),
            ready_workers=len(list(out.glob('episodes/*/*/ready.json'))),recovery_attempts=restarts,
            report=str(out/'report.md'),summary=str(out/'summary.json'),**extra))
        return deadline
    child_started=start
    status('starting')
    try:
        while True:
            child_started=now()
            with log_path.open('ab') as log:
                command=[sys.executable,'-B','-X','utf8',str(Path(__file__).resolve()),'--output',str(out),'--coordinate','--workers',str(args.workers)]
                if args.fail_probe: command.append('--fail-probe')
                child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=log)
                save(out/'launch_receipt.json',dict(configuration_hash=manifest['configuration_hash'],command=command,
                    supervisor_pid=os.getpid(),child_pid=child.pid,child_started_at=child_started.isoformat(),source_commit=manifest['commit']))
                while child.poll() is None:
                    deadline=status('running' if (out/'progress.json').exists() else 'starting')
                    if now()>deadline:
                        subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                        raise TimeoutError('Real progress deadline exceeded; saved results preserved')
                    time.sleep(3)
            if child.returncode==0:
                p=load(out/'progress.json');assert p['stage']=='complete'
                summary=load(out/'summary.json')
                compact={k:v for k,v in summary.items() if k!='episodes'}
                status('complete',results=compact);return
            error=log_path.read_text(encoding='utf8',errors='replace')[-3500:]
            if args.fail_probe or restarts>=1:raise RuntimeError(f'Coordinator exit {child.returncode}: {error}')
            restarts+=1
            save(out/'recovery.json',dict(at=now().isoformat(),attempt=restarts,error=error,
                action='One resume from matching completed result records; configuration unchanged'))
            progress(out,'recovering',load(out/'progress.json').get('completed',0),len(manifest['jobs']))
    except BaseException:
        if child and child.poll() is None:
            subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        status('failed',error=traceback.format_exc(),exit_code=child.returncode if child else None)
        raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--count',type=int,default=1000)
    parser.add_argument('--regression',action='store_true');parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--coordinate',action='store_true');parser.add_argument('--fail-probe',action='store_true')
    args=parser.parse_args();args.output=args.output.resolve()
    if args.prepare:prepare(args.output,args.count,args.regression)
    elif args.coordinate:
        if args.fail_probe:raise RuntimeError('Intentional supervisor failure probe')
        coordinate(args)
    else:supervise(args)


if __name__=='__main__':main()
