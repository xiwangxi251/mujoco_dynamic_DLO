"""Build a two-camera gallery and concise results from a scripted-family run."""
import argparse
import csv
import json
from pathlib import Path
import re

SCENE_LABELS = {
    'dev_static_x075_v1': '简单初始形状＋无整体运动＋无形状变化',
    'dev_rigid_l1_nominal_x075_v1': '简单初始形状＋简单整体运动＋无形状变化',
    'dev_shape_nominal_current_x075_v1': '简单初始形状＋无整体运动＋形状变化',
    'dev_combined_l1_nominal_x075_v1': '简单初始形状＋简单整体运动＋形状变化',
    'dev_static_midshape_v1_x075_v1': '复杂初始形状＋无整体运动＋无形状变化',
    'dev_rigid_replay_shape_nominal_x075_v1': '简单初始形状＋复杂整体运动＋无形状变化',
    'dev_rigid_l1_midshape_v1_x075_v1': '复杂初始形状＋简单整体运动＋无形状变化',
    'dev_rigid_replay_midshape_v1_x075_v1': '复杂初始形状＋复杂整体运动＋无形状变化（历史）',
}
HISTORICAL_NOTE = (
    '此页为原七场景的历史录像；第七项是复杂初始形状＋复杂整体运动＋无形状变化。'
    '当前场景族第七项已改为复杂初始形状＋简单整体运动＋无形状变化，尚无该新场景的评测录像。'
)

PAGE = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>规则方法 · 七场景</title><style>
*{box-sizing:border-box}body{margin:0;background:#eef2f7;color:#193047;font:16px/1.6 system-ui,"Microsoft YaHei",sans-serif}
main{max-width:1800px;margin:auto;padding:22px}h1{font-size:27px;margin:0 0 8px}h2{font-size:18px;margin:0}
.subtitle,.note{color:#4a6078}.toolbar{display:flex;gap:15px;flex-wrap:wrap;align-items:center;margin:15px 0}
button,select{font:inherit;padding:6px 12px;border:1px solid #b9c7d7;border-radius:6px;background:white;cursor:pointer}
button.primary{background:#1766a3;color:white}table{border-collapse:collapse;background:white;width:100%;max-width:980px;margin:16px 0}
th,td{text-align:left;padding:7px 14px;border-bottom:1px solid #e0e6ee}.players{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}
.player{background:white;border:1px solid #d9e2ec;border-radius:9px;overflow:hidden}.player h2,.meta{padding:10px 15px;margin:0}
video{display:block;width:100%;background:#0a1724;aspect-ratio:4/3}.meta{font-size:14px;color:#52677e}.badge{float:right;color:#a34835}.badge.pass{color:#167448}
.timeline{display:flex;gap:15px;align-items:center;background:white;padding:14px;margin-top:12px;border-radius:8px}#seek{flex:1;min-width:180px}
.reference{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:10px 0}.reference input{flex:1;min-width:240px;padding:8px;font:13px/1.5 monospace;border:1px solid #b9c7d7;border-radius:6px}
.focus{position:fixed;inset:0;background:#eef2f7;z-index:1000;overflow:auto;padding:18px}.focus video{max-height:75vh}.focus .players{max-width:2000px;margin:auto}
footer{margin-top:18px;display:flex;gap:22px}a{color:#1766a3}.note{margin:14px 0}@media(max-width:850px){.players{grid-template-columns:1fr}}
</style><main><h1>默认规则方法 · 七个场景 × 十回合</h1>
<div class="subtitle">录制时 dev 源码快照 · NERO · 增强6＋有界DLS · 同种子配对 · 原生 1280×960 / 50 fps</div>
__TABLE__
<div id="viewing"><div class="toolbar">
<label>场景 <select id="scene">__SCENES__</select></label><label>回合 <select id="goal">__SEEDS__</select></label>
<label>视角 <select id="view"><option value="both">全局＋腕部</option><option value="global">全局</option><option value="wrist">腕部</option></select></label>
<label>速度 <select id="rate"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></label>
<button class="primary" id="play">同步播放</button><button id="pause">暂停</button><button id="restart">从头播放</button><button id="expand">放大对比</button>
</div><div class="reference"><label for="episode-reference">回合标识</label><input id="episode-reference" readonly aria-label="可复制回合标识"><button id="copy-reference">复制标识＋时间</button><span id="copy-status" role="status"></span></div><div class="players">__PLAYERS__</div><div class="timeline"><span id="clock">0.00 s</span><input id="seek" type="range" min="0" max="15.02" step="0.02" value="0"></div></div>
<p id="note" class="note">蓝线：环形扇区工作区，圆心(0.35,0)m，半径0.28～0.56m，角度−30°～70°（世界+X为0°、向+Y为正）；仅显示，不限制选点。黄圈：规则策略当前跟踪的材料点；初始从中点开始，保留默认最近点锁定，因此跟踪点可改变。完整回合按公共物理成功条件计数，所有成功和失败均保留。两路使用同一仿真状态和时间；短片结束后停在末帧。</p>
<footer><a href="report.md" target="_blank">结果记录</a><a href="summary.json" target="_blank">逐回合结果</a><a href="manifest.json" target="_blank">配置与来源</a><a href="episodes.csv">下载 CSV</a></footer>
<script id="data" type="application/json">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent),ids=['global','wrist'],$=id=>document.getElementById(id),vs=()=>ids.map(id=>$('video-'+id));
const runName=__RUN_NAME__;
function reference(){const t=scrubbing||pending!==null?Number($('seek').value):Math.max(...vs().map(v=>v.currentTime||0));return `${runName} | ${$('scene').value} | seed=${$('goal').value} | t=${(Math.round(t/.02)*.02).toFixed(2)}s`}
$('episode-reference').addEventListener('click',e=>e.target.select());
$('copy-reference').addEventListener('click',async()=>{const text=reference();$('episode-reference').value=text;try{await navigator.clipboard.writeText(text);$('copy-status').textContent='已复制'}catch{$('episode-reference').focus();$('episode-reference').select();$('copy-status').textContent='请按 Ctrl+C 复制'}});
let scrubbing=false,pointerActive=false,resumeAfter=false,pending=null,timer=0,lastDispatch=0,generation=0;
function cancel(){generation++;clearTimeout(timer);timer=0;scrubbing=false;pointerActive=false;resumeAfter=false;pending=null}
function load(){cancel();const e=D.find(e=>e.scene===$('scene').value&&String(e.seed)===$('goal').value),view=$('view').value;ids.forEach(id=>{const v=$('video-'+id);v.pause();v.src=e.directory+'/'+id+'.mp4';v.poster=e.directory+'/'+id+'_poster.png';v.playbackRate=Number($('rate').value);v.load();$('player-'+id).hidden=view!=='both'&&view!==id;$('badge-'+id).textContent=e.task_success?'成功':'失败';$('badge-'+id).className='badge'+(e.task_success?' pass':'');$('meta-'+id).innerHTML=`初始节点 ${e.goal_node} · 时长 ${e.duration_s.toFixed(2)} s · ${e.task_failure_type}<br>最高抬起比例 ${(100*e.max_lifted_fraction).toFixed(0)}% · 最长成功保持 ${e.max_success_hold_s.toFixed(3)} s<br><span id="frame-clock-${id}">t=0.00 s</span> · <a href="${e.directory}/metadata.json" target="_blank">完整记录</a>`});document.querySelector('.players').style.gridTemplateColumns=view==='both'?'':'1fr';$('seek').value=0;$('seek').max=e.duration_s+.02;const u=new URL(location.href);u.searchParams.set('scene',e.scene);u.searchParams.set('seed',e.seed);history.replaceState(null,'',u)}
async function play(){const results=await Promise.allSettled(vs().filter(v=>v.currentTime<v.duration-.021).map(v=>v.play()));if(results.some(r=>r.status==='rejected'))$('note').textContent='请点击同步播放。'}
function pause(){vs().forEach(v=>v.pause())}
function finish(){if(pending!==null||vs().some(v=>v.seeking))return;if(!scrubbing&&resumeAfter){resumeAfter=false;play()}}
function pump(){if(timer||pending===null||vs().some(v=>v.seeking||v.readyState<1)){finish();return}const delay=Math.max(0,80-(performance.now()-lastDispatch));if(delay){const g=generation;timer=setTimeout(()=>{timer=0;if(g===generation)pump()},delay);return}const t=pending;pending=null;lastDispatch=performance.now();$('seek').dataset.appliedTarget=String(t);$('seek').dataset.dispatches=String(Number($('seek').dataset.dispatches||0)+1);vs().forEach(v=>{const target=Math.min(t,Math.max(0,v.duration-.02));if(Math.abs(v.currentTime-target)>.009)v.currentTime=target});finish()}
function begin(){if(scrubbing)return;scrubbing=true;resumeAfter=vs().some(v=>!v.paused&&!v.ended);pause()}
function request(){begin();pending=Number($('seek').value);$('seek').dataset.target=String(pending);pump()}
function end(){scrubbing=false;pump();finish()}
$('scene').addEventListener('change',load);$('goal').addEventListener('change',load);$('view').addEventListener('change',load);$('play').addEventListener('click',play);$('pause').addEventListener('click',()=>{resumeAfter=false;pause()});$('restart').addEventListener('click',()=>{pause();resumeAfter=true;pending=0;scrubbing=false;pump()});$('rate').addEventListener('change',()=>vs().forEach(v=>v.playbackRate=Number($('rate').value)));
$('seek').addEventListener('pointerdown',()=>{pointerActive=true;begin()});$('seek').addEventListener('input',request);for(const event of ['change','blur'])$('seek').addEventListener(event,()=>{if(!pointerActive)end()});for(const event of ['pointerup','pointercancel'])$('seek').addEventListener(event,()=>{pointerActive=false;end()});
vs().forEach(v=>{v.addEventListener('seeked',()=>{pump();finish()});v.addEventListener('loadedmetadata',pump);v.addEventListener('loadeddata',()=>v.removeAttribute('poster'));v.addEventListener('error',()=>{$('note').textContent='录像加载失败。'})});
$('expand').addEventListener('click',()=>{$('viewing').classList.toggle('focus');$('expand').textContent=$('viewing').classList.contains('focus')?'退出大画面':'放大对比'});document.addEventListener('keydown',e=>{if(e.key==='Escape')$('viewing').classList.remove('focus')});
function tick(){const t=Math.max(...vs().map(v=>v.currentTime||0));ids.forEach(id=>$('frame-clock-'+id).textContent=`t=${($('video-'+id).currentTime||0).toFixed(2)} s`);if(!scrubbing&&pending===null&&!vs().some(v=>v.seeking))$('seek').value=t;$('clock').textContent=`${Number($('seek').value).toFixed(2)} / ${Number($('seek').max).toFixed(2)} s`;if(document.activeElement!==$('episode-reference'))$('episode-reference').value=reference();requestAnimationFrame(tick)}window.addEventListener('pointerup',()=>{if(pointerActive){pointerActive=false;end()}});const params=new URLSearchParams(location.search);if(D.some(e=>e.scene===params.get('scene')))$('scene').value=params.get('scene');if(D.some(e=>String(e.seed)===params.get('seed')))$('goal').value=params.get('seed');load();tick();
</script>__VISIBILITY__<script>
// Restore controls after a drag which started while the videos were paused.
(function(){
  const restore=()=>vs().forEach(v=>v.controls=v.paused&&!scrubbing);
  for(const event of ['change','pointerup','pointercancel','blur'])$('seek').addEventListener(event,restore);
  window.addEventListener('pointerup',restore);
})();
</script></main></html>'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--visibility-template', type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.output/'summary.json').read_text(encoding='utf-8'))
    manifest = json.loads((args.output/'manifest.json').read_text(encoding='utf-8'))
    assert summary['status'] == 'complete'
    scenes, seeds = manifest['scenes'], manifest['seeds']
    rows = sorted(summary['episodes'], key=lambda r: (scenes.index(r['scene']), r['seed']))
    assert len(rows) == len(scenes)*len(seeds)
    assert len({(r['scene'], r['seed']) for r in rows}) == len(rows)
    label = {name: SCENE_LABELS.get(name, name) for name in scenes}
    historical = 'dev_rigid_replay_midshape_v1_x075_v1' in scenes
    results = [(name, sum(bool(r['task_success']) for r in rows if r['scene']==name)) for name in scenes]
    table = '<table><tr><th>场景</th><th>成功数</th><th>成功率</th></tr>' + ''.join(
        f'<tr><td>{label[name]}</td><td>{n}/{len(seeds)}</td><td>{n/len(seeds):.0%}</td></tr>' for name, n in results)+'</table>'
    template = args.visibility_template.read_text(encoding='utf-8')
    visibility = next(script for script in re.findall(r'<script>.*?</script>', template, re.S)
                      if 'function showControls(video,show)' in script)
    page = PAGE.replace('__TABLE__', table).replace('__VISIBILITY__', visibility)
    if manifest.get('workspace',{}).get('kind')!='annular_sector':
        workspace_note = ('本组录像不绘制工作区边界。' if manifest.get('run_type') == 'scripted_ppo_motion_difficulty_video_grid'
                          else '蓝框：本次录像沿用历史矩形工作区 X=0.60～0.84m、Y=−0.22～0.30m')
        page=page.replace('蓝线：环形扇区工作区，圆心(0.35,0)m，半径0.28～0.56m，角度−30°～70°（世界+X为0°、向+Y为正）', workspace_note)
    if historical:
        page = page.replace('<div id="viewing">', f'<p class="note">{HISTORICAL_NOTE}</p><div id="viewing">')
    page = page.replace('<title>规则方法 · 七场景</title>', f'<title>规则方法 · {len(scenes)} 个场景</title>')
    page = page.replace('__RUN_NAME__', json.dumps(args.output.resolve().name))
    page = page.replace('七个场景 × 十回合', f'{len(scenes)} 个场景 × {len(seeds)} 回合')
    page = page.replace('__SCENES__', ''.join(f'<option value="{name}">{label[name]}</option>' for name in scenes))
    page = page.replace('__SEEDS__', ''.join(f'<option value="{seed}">回合 {i} · seed {seed}</option>' for i,seed in enumerate(seeds,1)))
    page = page.replace('__PLAYERS__', ''.join(f'<article class="player" id="player-{view}"><h2>{title}<span class="badge" id="badge-{view}"></span></h2><video id="video-{view}" controls playsinline preload="auto"></video><p class="meta" id="meta-{view}"></p></article>' for view,title in [('global','全局'),('wrist','腕部')]))
    page = page.replace('__DATA__', json.dumps(rows, ensure_ascii=False).replace('</','<\/'))
    (args.output/'index.html').write_text(page, encoding='utf-8')
    with (args.output/'episodes.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        fields = ['scene','seed','task_success','task_failure_type','policy_result','duration_s','steps','goal_node','directory']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)
    robot = str(manifest.get('robot', 'NERO')).upper()
    report = [f"# 默认规则方法 {len(scenes)} 个条件评测（{manifest.get('run_name', args.output.name)}）", '',
        f"{robot}；{manifest['controller_profile']}；TableExecutionGuard={manifest['table_execution_guard_enabled']}。",
        'PolicyConfig 默认值；初始中点＋最近点锁定；15 秒上限；工作区仅用于显示；公共物理成功判定。',
        f"配对种子：{seeds[0]}～{seeds[-1]}；源轨迹库：`{manifest.get('bank', 'not recorded')}`。", '',
        '| 场景 | 成功数 | 成功率 |', '| --- | --- | --- |']
    if historical:
        report[2:2] = [HISTORICAL_NOTE, '']
    report.extend(f'| {label[name]} | {n}/{len(seeds)} | {n/len(seeds):.0%} |' for name,n in results)
    total = sum(n for _,n in results)
    report += ['', f'总计：{total}/{len(rows)}（{total/len(rows):.1%}）。每个条件 {len(seeds)} 次。', '',
        f'{len(rows)*2} 路录像按实际控制步状态原生渲染，1280×960、50 fps；H.264 CRF16、GOP≤12、无B帧。',
        '常规控制步为 20 ms；边界终止可能发生在最后一个控制步内，末帧保留真实终止状态。视频为恒定 50 fps，真实仿真时间保存在 trajectory.npz 的 frame_times，例外见 verification.json。',
        '逐视频帧数、时间戳、关键帧间隔及校验和保存在 metadata.json；渲染前后完整物理状态一致。',
        '黄圈标记策略选取/锁定的当前材料点；工作区边界是否绘制以录像配置为准。',
        f"运行配置和来源提交见 manifest.json；源提交：`{manifest.get('source_git_head', 'not recorded')}`。"]
    (args.output/'report.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    print(json.dumps(dict(episodes=len(rows), successes=total, by_scene=summary['by_scene']), ensure_ascii=False))


if __name__ == '__main__':
    main()
