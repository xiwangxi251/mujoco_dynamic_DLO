"""Keep pending and finished node-feedback recordings on one live page."""
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/evaluation'),str(ROOT/'tools/recording')]
from run_node_feedback_family import load, save
from build_scripted_family_gallery import PAGE, SCENE_LABELS
COMPLEX='dev_static_midshape_v1_x075_v1'


def build_gallery(out):
    manifest=load(out/'manifest.json')
    data=[]
    for row in load(out/'summary.json')['episodes']:
        dest=out/'episodes'/row['scene']/f"seed_{row['seed']}"
        data.append(dict(scene=row['scene'],seed=row['seed'],task_success=row['task_success'],
            duration_s=row['duration_s'],directory=str(dest.relative_to(out)).replace('\\','/'),
            ready=(dest/'video_metadata.json').exists(),
            goal_node=row['selection_events'][0]['chosen']['node']))
    table='<table><tr><th>场景</th><th>成功数</th><th>视频完成</th></tr>'+''.join(
        f"<tr><td>{SCENE_LABELS[scene]}</td><td>{sum(r['task_success'] for r in data if r['scene']==scene)}/10</td><td id='count-{scene}'>{sum(r['ready'] for r in data if r['scene']==scene)}/10</td></tr>"
        for scene in manifest['scenes'])+'</table>'
    source=(ROOT.parent/'archive/reward_v2_20261005/reward_v6_gallery_template.html').read_text(encoding='utf-8')
    visibility=next(s for s in re.findall(r'<script>.*?</script>',source,re.S) if 'function showControls(video,show)' in s)
    page=PAGE.replace('__TABLE__',table).replace('__VISIBILITY__',visibility)
    page=page.replace('__RUN_NAME__',json.dumps(out.name))
    scene_order=[COMPLEX]+[s for s in manifest['scenes'] if s!=COMPLEX]
    page=page.replace('__SCENES__',''.join(f'<option value="{s}">{SCENE_LABELS[s]}</option>' for s in scene_order))
    page=page.replace('__SEEDS__',''.join(f'<option value="{s}">seed {s}</option>' for s in manifest['seeds']))
    page=page.replace('__PLAYERS__',''.join(f'<article class="player" id="player-{view}"><h2>{title}<span class="badge" id="badge-{view}"></span></h2><video id="video-{view}" controls playsinline preload="auto"></video><p class="meta" id="meta-{view}"></p></article>' for view,title in [('global','项目全局'),('wrist','项目腕部')]))
    page=page.replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('</','<\\/'))
    page=page.replace('const D=JSON.parse','let D=JSON.parse')
    page=page.replace('默认规则方法 · 七个场景 × 十回合','规则方法 · 保存状态录像')
    page=re.sub(r'<div class="subtitle">.*?</div>','<div class="subtitle">优先生成复杂静态；未完成视频留空，完成后自动补入本页。保存状态直接重渲染；1280×960 / 50 fps。</div>',page,count=1)
    page=re.sub(r'<p id="note" class="note">.*?</p>','<p id="note" class="note">黄圈：策略当时跟踪的材料点；蓝线：实测工作区。几何选点＋切线旋转；闭爪0.8秒后抬升，1秒后升高不足3厘米重试。未完成的回合显示空位，本页每10秒自动更新。</p>',page,count=1)
    page=page.replace('<a href="episodes.csv">下载 CSV</a>','')
    load_js=r"""function load(){cancel();const e=D.find(e=>e.scene===$('scene').value&&String(e.seed)===$('goal').value),view=$('view').value;if(!e)return;ids.forEach(id=>{const v=$('video-'+id);v.pause();v.removeAttribute('src');v.removeAttribute('poster');if(e.ready){v.src=e.directory+'/'+id+'.mp4';v.poster=e.directory+'/'+id+'_poster.png'}v.playbackRate=Number($('rate').value);v.load();$('player-'+id).hidden=view!=='both'&&view!==id;$('badge-'+id).textContent=e.ready?(e.task_success?'成功':'失败'):'视频待生成';$('badge-'+id).className='badge'+(e.ready&&e.task_success?' pass':'');$('meta-'+id).innerHTML=(e.ready?'视频已完成':'此处留空，生成后自动补上')+' · 初始节点 '+e.goal_node+' · 仿真时长 '+e.duration_s.toFixed(2)+' s<br><span id="frame-clock-'+id+'">t=0.00 s</span>'+(e.ready?' · <a href="'+e.directory+'/video_metadata.json" target="_blank">录像记录</a>':'')});document.querySelector('.players').style.gridTemplateColumns=view==='both'?'':'1fr';$('seek').value=0;$('seek').max=e.duration_s+.02;$('play').disabled=!e.ready;$('restart').disabled=!e.ready;$('seek').disabled=!e.ready;const u=new URL(location.href);u.searchParams.set('scene',e.scene);u.searchParams.set('seed',e.seed);history.replaceState(null,'',u)}"""
    page=re.sub(r'function load\(\)\{.*?(?=\nasync function play)',lambda _:load_js,page,count=1,flags=re.S)
    ref_js=r"""function reference(){const t=scrubbing||pending!==null?Number($('seek').value):Math.max(...vs().map(v=>v.currentTime||0));return runName+' | '+$('scene').value+' | seed='+$('goal').value+' | view='+$('view').value+' | t='+(Math.round(t/.02)*.02).toFixed(2)+'s'}"""
    page=re.sub(r'function reference\(\)\{[^\n]+',lambda _:ref_js,page,count=1)
    poll=r"""<script>setInterval(async()=>{try{const response=await fetch('gallery_data.json',{cache:'no-store'});if(!response.ok)return;const next=await response.json();const old=D.find(e=>e.scene===$('scene').value&&String(e.seed)===$('goal').value);D=next;const current=D.find(e=>e.scene===$('scene').value&&String(e.seed)===$('goal').value);for(const scene of [...new Set(D.map(e=>e.scene))]){const cell=$('count-'+scene);if(cell)cell.textContent=D.filter(e=>e.scene===scene&&e.ready).length+'/10'}if(current?.ready&&!old?.ready)load()}catch{}},10000)</script>"""
    page=page.replace('</main>',poll+'</main>')
    tmp=out/'index.html.tmp'
    tmp.write_text(page,encoding='utf-8')
    tmp.replace(out/'index.html')
    save(out/'gallery_data.json',data)
