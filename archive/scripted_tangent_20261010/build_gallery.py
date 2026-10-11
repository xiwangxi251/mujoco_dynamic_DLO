import json
import re
from pathlib import Path

NAMES={
    'dev_shape_nominal_current_x075_v1':'简单初始形状＋无整体运动＋形状变化',
    'dev_static_midshape_v1_x075_v1':'复杂初始形状＋无整体运动＋无形状变化',
    'joint7_minus90':'第7关节单独反向转90°',
}

def build(root, first_n=None):
    root=Path(root)
    summary=json.loads((root/'scenes/summary.json').read_text(encoding='utf-8'))
    stats=json.loads((root/'joint7_speed.json').read_text(encoding='utf-8'))
    page=(root/'gallery_template.html').read_text(encoding='utf-8')
    rows=summary['episodes']
    if first_n is not None:
        rows=[row for row in rows if row['seed'] < 20260804+first_n]
        summary['episodes']=rows
    summary['by_scene']={key:dict(episodes=len(selected),
        successes=sum(bool(row['task_success']) for row in selected),
        success_rate=sum(bool(row['task_success']) for row in selected)/len(selected))
        for key in NAMES if key!='joint7_minus90'
        and (selected:=[row for row in rows if row['scene']==key])}
    for row in rows: row['directory']='scenes/'+row['directory']
    rotation=json.loads((root/stats['directory']/'metadata.json').read_text(encoding='utf-8'))['episode_row']
    rotation.update(scene='joint7_minus90',directory=stats['directory'],rotation_probe=True)
    rows.append(rotation)
    page=re.sub(r'<script id="data" type="application/json">.*?</script>',lambda m:'<script id="data" type="application/json">'+json.dumps(rows,ensure_ascii=False)+'</script>',page,flags=re.S)
    page=page.replace('const runName="scripted_seven_default_20261009";','const runName="scripted_tangent_20261010";')
    available={r['scene'] for r in rows}
    page=re.sub(r'<select id="scene">.*?</select>',lambda m:'<select id="scene">'+''.join(f'<option value="{key}">{value}</option>' for key,value in NAMES.items() if key in available)+'</select>',page,flags=re.S)
    table='<table><tr><th>场景</th><th>成功数</th><th>成功率</th></tr>'
    for key,value in summary['by_scene'].items():
        table+=f"<tr><td>{NAMES[key]}</td><td>{value['successes']}/{value['episodes']}</td><td>{value['success_rate']:.0%}</td></tr>"
    table+='</table>'
    page=re.sub(r'<table>.*?</table>',lambda m:table,page,count=1,flags=re.S)
    page=re.sub(r'<h1>.*?</h1>','<h1>规则方法跟随目标切线旋转 · 两场景各10回合＋第7关节测速</h1>',page,count=1)
    if first_n is not None:
        page=page.replace('两场景各10回合','两场景各'+str(first_n)+'回合（阶段结果）')
    note='当前dev场景，统一等待0.82秒。复杂静态沿用现有快照，5mm初始穿透筛选尚未实施。'
    if first_n is not None:
        note='本页仅含两场景前'+str(first_n)+'回合；其余回合继续后台录制。'+note
    elif summary['status']!='complete':
        note='抓取测试正在录制，当前先显示第7关节测速。'+note
    speed=f"第7关节：反向转90°，峰值实际角速度 {stats['peak_actual_speed_deg_s']:.1f}°/s；首次进入目标±2°用时 {stats['first_within_2deg_seconds']} s。"
    page=re.sub(r'<p class="note">此页为原七场景.*?</p>',lambda m:'<p class="note">'+note+' '+speed+'</p>',page,count=1,flags=re.S)
    page=page.replace('<footer>', '<details><summary>第7关节角度与速度曲线（500 Hz）</summary><img src="joint7_speed.png" style="max-width:100%"><p><a href="joint7_speed.json">测速记录</a> · <a href="joint7_500hz.csv">500 Hz原始数据</a></p></details><footer>')
    page=page.replace('录制时 dev 源码快照 · NERO · 增强6＋有界DLS · 同种子配对','当前 dev 源码冻结 · NERO · 增强6＋有界DLS · 碰桌保护开启 · 切线跟随开启')
    page=page.replace("function load(){cancel();const e=", "function load(){cancel();const allowed=D.filter(e=>e.scene===$('scene').value).map(e=>String(e.seed));for(const option of $('goal').options)option.disabled=!allowed.includes(option.value);if(!allowed.includes($('goal').value))$('goal').value=allowed[0];const e=")
    page=page.replace("e.task_success?'成功':'失败'", "e.rotation_probe?'测速':(e.task_success?'成功':'失败')")
    for filename in ('summary.json','manifest.json','episodes.csv'):
        page=page.replace(f'href="{filename}"',f'href="scenes/{filename}"')
    stem='first'+str(first_n) if first_n is not None else 'index'
    report='first'+str(first_n)+'.md' if first_n is not None else 'report.md'
    page=page.replace('href="report.md"','href="'+report+'"')
    (root/(stem+'.html')).write_text(page,encoding='utf-8')
    (root/report).write_text('# 切线跟随测试\n\n'+note+'\n\n'+speed+'\n\n'+json.dumps(summary['by_scene'],ensure_ascii=False,indent=2),encoding='utf-8')
    if first_n is not None:
        (root/(stem+'.json')).write_text(json.dumps(dict(by_scene=summary['by_scene'],
            seeds=list(range(20260804,20260804+first_n))),ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':
    import sys
    build(sys.argv[1],int(sys.argv[2]) if len(sys.argv)>2 else None)
