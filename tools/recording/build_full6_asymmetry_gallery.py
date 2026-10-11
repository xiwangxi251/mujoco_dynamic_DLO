"""Reuse the verified synchronized player for positive/negative target pairs."""
import json
import re
import subprocess
from pathlib import Path
import build_retest_136_gallery as base

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/vertical_workspace/scripted_136_retest_20261004/full6_gravity_servo_gain10/asymmetry/videos'

def main():
    episodes=[];checks=[]
    for pair in ('limit_pair','endpoint_pair'):
        for side in ('positive','negative'):
            directory=OUT/'version_6'/f'{pair}_{side}'
            metadata=json.loads((directory/'metadata.json').read_text())
            metadata.update(directory=str(directory.relative_to(OUT)).replace('\\','/'),side=1 if side=='positive' else -1,pair_name=pair)
            episodes.append(metadata)
            assert metadata['previous_saved_joint_state_max_difference']==0
            for view in ('global','wrist'):
                file=directory/f'{view}.mp4'
                info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_streams','-show_frames',
                    '-show_entries','stream=width,height,avg_frame_rate,nb_frames,codec_name:frame=key_frame,best_effort_timestamp_time,pict_type',
                    '-of','json',str(file)],text=True))
                stream=info['streams'][0];frames=info['frames']
                assert (stream['width'],stream['height'],stream['avg_frame_rate'],stream['codec_name'])==(1280,960,'50/1','h264')
                assert len(frames)==metadata['frames']==int(stream['nb_frames'])
                assert max(abs(float(f['best_effort_timestamp_time'])-i/50) for i,f in enumerate(frames))<1e-6
                keys=[i for i,f in enumerate(frames) if f['key_frame']]
                gaps=[b-a for a,b in zip(keys,keys[1:])]+[len(frames)-keys[-1]]
                assert max(gaps)<=12 and all(f['pict_type']!='B' for f in frames)
                checks.append(dict(file=str(file.relative_to(OUT)).replace('\\','/'),frames=len(frames),fps=50,
                    resolution=[1280,960],max_gop=max(gaps),b_frames=0,trajectory_max_difference=0))
    html=base.HTML.replace('版本 1、3、6 · 工作范围实际录像','增强版本 6 · 正负 Y 对照录像')
    html=html.replace('相同 home、目标、末端朝向和测试轨迹。','同一组合：完整 6D DLS + 重力前馈 + kp×4、kv×2 + 姿态增益 10。相同 home、固定朝向；目标 Y 取正负镜像。')
    html=re.sub(r'<select id="goal">.*?</select>',
        '<select id="goal"><option value="limit_pair">限位与超时：(0.67, ±0.28, 0.005) m</option><option value="endpoint_pair">终点与全程的区别：(0.55, ±0.28, 0.005) m</option></select>',html,count=1)
    html=html.replace('ids=[1,3,6]','ids=[1,-1]').replace('e.version===id','e.side===id')
    html=html.replace('e.goal_name===goal','e.pair_name===goal').replace("e.goal_name===$('goal').value","e.pair_name===$('goal').value")
    html=html.replace('三路同步进度','正负 Y 同步进度').replace('暂停三路','暂停两路')
    html=html.replace('°${floor}', '°<br>全程最大姿态误差 ${e.result.max_path_pose_deg.toFixed(2)}°${floor}')
    oldnote="'每张卡片下方是结束帧统计，视频内 HUD 随当前播放帧变化。先以 0.15 m/s 接近目标上方 10 cm，再以 0.03 m/s 下降，检查终点保持 2 秒。短录像停在末帧，其余继续。拖动时间轴会暂停两路，定位完成后恢复原播放状态。'"
    newnote="(goal==='limit_pair'?'左侧正 Y 全程通过；右侧负 Y 的关节 6 到达下限，16 秒超时。':'两侧终点都通过；负 Y 途中最大姿态误差 11.58°，超过全程 5° 门限，因此只算终点通过。')+' 终点通过：位置≤5 mm、完整姿态≤2°，满足离桌/关节/自碰撞要求并保持 2 秒。全程通过再要求途中姿态≤5°、实际关节速度合规。卡片数值是结束统计；当前数值见视频内 HUD。短录像停在末帧。'"
    assert oldnote in html
    html=html.replace(oldnote,newnote)
    html=re.sub(r'<footer>.*?</footer>',
        '<footer><a href="../../workspace_map.png" target="_blank">范围图</a><a href="../diagnostic.json" target="_blank">对称目标诊断</a><a href="video_verification.json" target="_blank">录像核查</a></footer>',html,count=1)
    players=''.join(f'<article class="player"><div class="head"><strong>{label}</strong><span id="badge-{side}" class="badge"></span></div><video id="video-{side}" controls playsinline preload="auto" aria-label="{label}录像"></video><div class="meta" id="meta-{side}"></div></article>' for side,label in ((1,'正 Y 目标'),(-1,'负 Y 目标')))
    html=html.replace('repeat(3,minmax(0,1fr))','repeat(2,minmax(0,1fr))')
    (OUT/'index.html').write_text(html.replace('__PLAYERS__',players).replace('__DATA__',json.dumps(episodes,ensure_ascii=False)),encoding='utf-8')
    (OUT/'gallery_manifest.json').write_text(json.dumps(episodes,indent=2,ensure_ascii=False),encoding='utf-8')
    (OUT/'video_verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(dict(clips=len(checks),all_checks_passed=True)),flush=True)

if __name__=='__main__':main()
