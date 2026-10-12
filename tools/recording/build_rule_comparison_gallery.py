"""Progressive two-rule comparison and individual two-camera galleries."""
import json
from pathlib import Path
import re
from build_scripted_family_gallery import PAGE
from run_node_feedback_family import save, load

LABELS = {
    'dev_static_x075_v1':'普通静态',
    'dev_static_midshape_v1_x075_v1':'复杂初始静态',
    'dev_rigid_l1_nominal_x075_v1':'简单初态整体运动',
    'dev_rigid_l1_midshape_v1_x075_v1':'复杂初态整体运动',
    'dev_rigid_replay_shape_nominal_x075_v1':'Shape replay',
    'dev_shape_nominal_current_x075_v1':'形变',
    'dev_combined_l1_nominal_x075_v1':'组合运动',
}


def build_gallery(out):
    manifest=load(out/'manifest.json')
    data=[]
    for row in load(out/'summary.json')['episodes']:
        item={k:row[k] for k in ('scene','seed','task_success','duration_s','video_directory','arm','source_run')}
        item['ready']=(out/row['video_directory']/'video_metadata.json').exists()
        data.append(item)
    counts=[]
    for scene,label in LABELS.items():
        arms=[]
        for arm in ('material','nearest'):
            subset=[r for r in data if r['scene']==scene and r['arm']==arm]
            arms.append(f"{sum(r['task_success'] for r in subset)}/{len(subset)}" if subset else '—')
        counts.append(f'<tr><td>{label}</td><td>{arms[0]}</td><td>{arms[1]}</td></tr>')
    table='<table><tr><th>场景</th><th>固定材料点</th><th>最近线段</th></tr>'+''.join(counts)+'</table>'
    frozen_template=out/'source/visibility_template.html'
    template_path=frozen_template if frozen_template.exists() else next(
        p/'archive/reward_v2_20261005/reward_v6_gallery_template.html'
        for p in Path(__file__).resolve().parents
        if (p/'archive/reward_v2_20261005/reward_v6_gallery_template.html').exists())
    template=template_path.read_text(encoding='utf-8')
    visibility=next(s for s in re.findall(r'<script>.*?</script>',template,re.S) if 'function showControls(video,show)' in s)
    page=PAGE.replace('__TABLE__',table).replace('__VISIBILITY__',visibility)
    page=page.replace('<title>规则方法 · 七场景</title>','<title>规则测试录像与选点对照</title>')
    page=page.replace('默认规则方法 · 七个场景 × 十回合','规则测试录像与选点对照')
    page=re.sub(r'<div class="subtitle">.*?</div>', '<div class="subtitle">450回合、900段双视角视频 · 原生1280×960 / 50 fps · 两批测试的实际保存状态 · 成功与失败全部保留</div><p id="progress">视频生成中</p>',page,count=1)
    page=page.replace('<label>场景 <select', '<label>模式 <select id="mode"><option value="compare">两种规则对比</option><option value="material">固定材料点：七场景</option><option value="nearest">最近线段：形变与组合</option></select></label><label>场景 <select')
    page=page.replace('<label>视角 <select id="view"><option value="both">全局＋腕部</option><option value="global">全局</option><option value="wrist">腕部</option></select></label>', '<select id="view" hidden><option value="both">全局＋夹爪</option><option value="global">全局</option><option value="wrist">夹爪（腕部）</option></select><div class="view-switch" role="group" aria-label="视角"><span>视角</span><button id="view-global" type="button">全局视角</button><button id="view-wrist" type="button">夹爪视角</button><button id="view-both" type="button">全局＋夹爪</button></div>')
    page=page.replace('</style>', '.view-switch{display:flex;gap:6px;align-items:center;flex-wrap:wrap}.view-switch button[aria-pressed="true"]{background:#1766a3;color:white;border-color:#1766a3}.view-switch button:disabled{opacity:.45;cursor:default}#goal{max-width:100%}</style>')
    page=page.replace('__SCENES__',''.join(f'<option value="{s}">{label}</option>' for s,label in LABELS.items()))
    page=page.replace('__SEEDS__',''.join(f'<option value="{s}">seed {s}</option>' for s in manifest['seeds']))
    page=page.replace('__PLAYERS__',''.join(f'<article class="player" id="player-{slot}"><h2><span id="title-{slot}"></span><span class="badge" id="badge-{slot}"></span></h2><video id="video-{slot}" controls playsinline preload="auto"></video><p class="meta" id="meta-{slot}"></p></article>' for slot in ('global','wrist')))
    page=page.replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('</','<\\/')).replace('__RUN_NAME__',json.dumps(out.name))
    page=page.replace('const D=JSON.parse','let D=JSON.parse')
    page=re.sub(r'<p id="note" class="note">.*?</p>','<p id="note" class="note">黄圈为当时跟踪的材料点，蓝线为实测工作区。对照仅关闭材料点保持，几何选点、切线跟随、节点反馈相同。时间轴使用实际仿真秒；短回合停在末帧。未完成视频显示待生成。</p>',page,count=1)
    page=re.sub(r'<footer>.*?</footer>','<footer><a href="video_report.md" target="_blank">录像记录</a><a href="manifest.json" target="_blank">来源</a><a href="../scripted_material50_20261012/report.md" target="_blank">七场景测试结果</a><a href="../scripted_nearest50_20261012/report.md" target="_blank">选点配对结果</a></footer>',page,count=1)
    reference=r"""function reference(){const t=scrubbing||pending!==null?Number($('seek').value):Math.max(...vs().map(v=>v.currentTime||0));const mode=$('mode').value,runs=mode==='compare'?'scripted_material50_20261012 + scripted_nearest50_20261012':mode==='material'?'scripted_material50_20261012':'scripted_nearest50_20261012';return runs+' | '+$('scene').value+' | seed='+$('goal').value+' | view='+$('view').value+' | t='+(Math.round(t/.02)*.02).toFixed(2)+'s'}"""
    page=re.sub(r'function reference\(\)\{[^\n]+',lambda _:reference,page,count=1)
    load_js=r"""function sceneOptions(){const mode=$('mode').value;for(const option of $('scene').options)option.hidden=mode!=='material'&&!['dev_shape_nominal_current_x075_v1','dev_combined_l1_nominal_x075_v1'].includes(option.value);if($('scene').selectedOptions[0]?.hidden)$('scene').value='dev_shape_nominal_current_x075_v1';$('view').querySelector('option[value="both"]').disabled=mode==='compare';if(mode==='compare'&&$('view').value==='both')$('view').value='global'}
function load(){cancel();sceneOptions();const mode=$('mode').value,view=$('view').value,scene=$('scene').value,seed=Number($('goal').value);const selected=D.filter(e=>e.scene===scene&&e.seed===seed);let longest=0,available=false;ids.forEach((id,i)=>{const arm=mode==='compare'?(i===0?'material':'nearest'):mode,e=selected.find(e=>e.arm===arm),camera=mode==='compare'?view:id,v=$('video-'+id);v.pause();v.removeAttribute('src');v.removeAttribute('poster');if(e?.ready){v.src=e.video_directory+'/'+camera+'.mp4';v.poster=e.video_directory+'/'+camera+'_poster.png';available=true}v.playbackRate=Number($('rate').value);v.load();$('player-'+id).hidden=mode!=='compare'&&view!=='both'&&view!==id;$('title-'+id).textContent=(arm==='material'?'固定材料点':'最近线段')+' · '+(camera==='global'?'全局':'腕部');$('badge-'+id).textContent=e?.ready?(e.task_success?'成功':'失败'):'视频待生成';$('badge-'+id).className='badge'+(e?.ready&&e.task_success?' pass':'');$('meta-'+id).innerHTML=(e?e.source_run+' · 时长 '+e.duration_s.toFixed(2)+' s':'无此测试')+'<br><span id="frame-clock-'+id+'">t=0.00 s</span>'+(e?.ready?' · <a href="'+e.video_directory+'/video_metadata.json" target="_blank">录像来源</a>':'');longest=Math.max(longest,e?.duration_s||0)});document.querySelector('.players').style.gridTemplateColumns=mode==='compare'||view==='both'?'':'1fr';$('seek').value=0;$('seek').max=longest+.02;$('play').disabled=!available;$('restart').disabled=!available;$('seek').disabled=!available;const u=new URL(location.href);for(const [k,value] of Object.entries({mode,scene,seed,view}))u.searchParams.set(k,value);history.replaceState(null,'',u)}"""
    page=re.sub(r'function load\(\)\{.*?(?=\nasync function play)',lambda _:load_js,page,count=1,flags=re.S)
    seed_js=r"""function seedOptions(){const mode=$('mode').value,scene=$('scene').value,rows=D.filter(e=>e.scene===scene),arms=mode==='compare'?['material','nearest']:[mode],result=e=>e?(e.task_success?'成功':'失败'):'无测试';for(const option of $('goal').options){const entries=arms.map(arm=>rows.find(e=>e.arm===arm&&e.seed===Number(option.value)));option.textContent='seed '+option.value+' · '+(mode==='compare'?'固定：'+result(entries[0])+' ｜ 最近：'+result(entries[1]):result(entries[0]));option.style.color=entries.every(e=>e?.task_success)?'#167448':entries.every(e=>e&&!e.task_success)?'#a34835':'#76580a'}}
function viewButtons(){for(const view of ['global','wrist','both']){const button=$('view-'+view);button.setAttribute('aria-pressed',String($('view').value===view));button.disabled=view==='both'&&$('mode').value==='compare';button.title=button.disabled?'对照模式中，两种规则使用同一视角；可选择全局或夹爪':'切换到'+button.textContent}}
"""
    page=page.replace('function sceneOptions(){',seed_js+'function sceneOptions(){')
    page=page.replace('cancel();sceneOptions();const mode=', 'cancel();sceneOptions();seedOptions();viewButtons();const mode=')
    page=page.replace("camera==='global'?'全局':'腕部'", "camera==='global'?'全局':'夹爪（腕部）'")
    page=page.replace("$('view').addEventListener('change',load);", "$('view').addEventListener('change',load);for(const view of ['global','wrist','both'])$('view-'+view).addEventListener('click',()=>{$('view').value=view;load()});")
    # Only loaded players participate in seeking; pending slots cannot block.
    page=page.replace("vs=()=>ids.map(id=>$('video-'+id))", "vs=()=>ids.map(id=>$('video-'+id)).filter(v=>v.hasAttribute('src'))")
    page=page.replace("const t=Math.max(...vs().map(v=>v.currentTime||0));", "const t=Math.max(0,...vs().map(v=>v.currentTime||0));")
    page=page.replace("Math.max(...vs().map(v=>v.currentTime||0))", "Math.max(0,...vs().map(v=>v.currentTime||0))")
    page=page.replace("$('scene').addEventListener('change',load);", "$('mode').addEventListener('change',load);$('scene').addEventListener('change',load);")
    # Visibility handlers must attach to both elements even when initially pending.
    page=page.replace('const videos=vs();',"const videos=ids.map(id=>$('video-'+id));")
    page=page.replace("vs().forEach(v=>{v.addEventListener('seeked'", "ids.map(id=>$('video-'+id)).forEach(v=>{v.addEventListener('seeked'")
    page=page.replace("load();tick();", "if(params.has('mode'))$('mode').value=params.get('mode');if(params.has('view'))$('view').value=params.get('view');load();tick();")
    progress=r"""<script>function progress(){const ready=D.filter(e=>e.ready).length;$('progress').textContent='视频已完成 '+ready+'/'+D.length+' 回合（'+(ready*2)+'/'+(D.length*2)+' 段）'}progress();setInterval(async()=>{try{const response=await fetch('gallery_data.json',{cache:'no-store'});if(!response.ok)return;const next=await response.json(),mode=$('mode').value,selected=(mode==='compare'?['material','nearest']:[mode]);const before=D.filter(e=>e.scene===$('scene').value&&e.seed===Number($('goal').value)&&selected.includes(e.arm)).filter(e=>e.ready).length;D=next;const after=D.filter(e=>e.scene===$('scene').value&&e.seed===Number($('goal').value)&&selected.includes(e.arm)).filter(e=>e.ready).length;progress();if(after>before)load()}catch{}},10000)</script>"""
    progress=progress.replace('progress();if(after>before)load()', 'progress();seedOptions();if(after>before)load()')
    page=page.replace('</main>',progress+'</main>')
    page=page.replace("v.controls=v.paused&&!scrubbing", "v.controls=(v.paused||v.ended)&&!scrubbing")
    save(out/'gallery_data.json',data)
    temporary=out/'index.html.tmp'
    temporary.write_text(page,encoding='utf-8')
    temporary.replace(out/'index.html')
