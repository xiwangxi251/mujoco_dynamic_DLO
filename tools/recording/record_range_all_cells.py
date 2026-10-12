"""Frozen-policy full range test: 400 actual episodes, all dual-camera videos."""
import os,sys,json,time,subprocess,hashlib,shutil,zipfile,traceback
from pathlib import Path
SOURCE=Path('/root/rivermind-data/ppo_cpu_project/range-ppo-20261012')
RUN=Path('/root/rivermind-data/runs/ppo_range_session_height_scratch_20m_20261012')
OUT=Path(os.environ.get('RANGE_VIDEO_OUT',str(RUN/'all_cells_video400_20261012')))
COMMIT='44af6e30f9ae95de99924776d0c3814174f8bdfa'
os.environ['PATH']=str(RUN/'video_bin')+os.pathsep+os.environ.get('PATH','')
os.environ.update(PYTHONPATH=str(SOURCE/'src'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',LP_NUM_THREADS='1',MUJOCO_GL='osmesa',PYOPENGL_PLATFORM='osmesa',PYTHONPYCACHEPREFIX=str(OUT/'pycache'))
sys.path.insert(0,str(SOURCE/'src'))
def load(p):return json.loads(Path(p).read_text())
def dump(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
 tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda v:v.tolist() if hasattr(v,'tolist') else str(v)));tmp.replace(p)
def identity(p):
 try:
  s=Path(f'/proc/{p}/stat').read_text().rsplit(')',1)[1].split()
  return dict(pid=p,birth=s[19],state=s[0],command=Path(f'/proc/{p}/cmdline').read_bytes().decode().strip('\0').split('\0'),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
 except FileNotFoundError:return None
def worker(index,smoke=False):
 import numpy as np,torch,mujoco
 from PIL import Image,ImageDraw,ImageFont
 from stable_baselines3 import PPO
 from stable_baselines3.common.utils import set_random_seed
 from panda_cable_grasp.rl.environment import RLConfig,make_rl_env
 from dataclasses import asdict
 torch.set_num_threads(1)
 cfg=load(OUT/'training_config.json');manifest=load(OUT/'manifest.json')
 import importlib.util
 spec=importlib.util.spec_from_file_location('original_runner',str(RUN/'runner.py'));runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
 train,args=runner.parsed(OUT/'factory_probe',workers=2,steps=6,nsteps=3,epochs=1,batch=2)
 original_make=train.make_rl_env
 def camera_factory(**kwargs):
  kwargs['dynamicvla_cameras_enabled']=True
  return original_make(**kwargs)
 train.make_rl_env=camera_factory
 env=train.make_worker(index,args)()
 train.make_rl_env=original_make
 env.set_penalty_ramp_progress(manifest['checkpoint_steps'],48,cfg['penalty_ramp_schedule'])
 assert asdict(env.rl_config)==cfg['rl_config']
 assert env.rl_config.ordinary_reward_profile=='session_height_v1'
 assert hashlib.sha256((OUT/'checkpoint.zip').read_bytes()).hexdigest()==manifest['checkpoint_sha256']
 model=PPO.load(OUT/'checkpoint.zip',device='cpu',custom_objects={'learning_rate':0.0,'lr_schedule':lambda _:0.0})
 assert model.num_timesteps==manifest['checkpoint_steps'] and all(torch.isfinite(v).all() for v in model.policy.parameters())
 from panda_cable_grasp.rl.range_curriculum import cells
 jobs=manifest['jobs'][::20] if smoke else manifest['jobs']
 jobs=jobs[index::8]
 b=env.base_env;b.model.vis.global_.offwidth=max(1280,b.model.vis.global_.offwidth);b.model.vis.global_.offheight=max(960,b.model.vis.global_.offheight)
 renderer=mujoco.Renderer(b.model,height=960,width=1280)
 cams={'global':'dynamicvla_opst_camera','wrist':'dynamicvla_wrist_camera'}
 camera_meta={}
 for view,name in cams.items():
  cid=mujoco.mj_name2id(b.model,mujoco.mjtObj.mjOBJ_CAMERA,name);assert cid>=0
  camera_meta[view]=dict(name=name,position=b.model.cam_pos[cid].tolist(),quaternion=b.model.cam_quat[cid].tolist(),fovy=float(b.model.cam_fovy[cid]))
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
 def state():
  s=np.empty(mujoco.mj_stateSize(b.model,mujoco.mjtState.mjSTATE_FULLPHYSICS));mujoco.mj_getState(b.model,b.data,s,mujoco.mjtState.mjSTATE_FULLPHYSICS);return s
 dump(OUT/f'worker_{index}_ready.json',dict(at=time.time(),worker=index,configuration_verified=True,checkpoint_steps=model.num_timesteps,camera=camera_meta,jobs=len(jobs)))
 try:
  for offset,job in enumerate(jobs):
   scene=job['scenario'];cell=job['cell'];seed=job['seed'];set_random_seed(seed)
   spec={scene:dict(probabilities=[float(c==cell) for c in cells(scene)])}
   env.set_range_curriculum(6,(scene,),(1.,),spec)
   obs,initial=env.reset(seed=seed)
   assert obs.shape==(99,) and env.action_space.shape==(5,)
   assert b.config.scenario_name==scene and initial['range_episode']['cell']==cell
   if b.config.initial_shape_bank:assert b._initial_shape_source_seed is not None
   if b.config.replay_bank:assert b._replay_entry_seed is not None
   if smoke:
    plain=train.make_worker(index,args)()
    plain.set_penalty_ramp_progress(manifest['checkpoint_steps'],48,cfg['penalty_ramp_schedule'])
    plain.set_range_curriculum(6,(scene,),(1.,),spec)
    po,pi=plain.reset(seed=seed)
    np.testing.assert_array_equal(po,obs)
    assert pi['range_episode']==initial['range_episode']
   directory=OUT/('smoke' if smoke else 'episodes')/job['id'];directory.mkdir(parents=True,exist_ok=True)
   assert not (directory/'metadata.json').exists(),'Refusing duplicate episode'
   writers={v:subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgb24','-video_size','1280x960','-framerate','50','-i','-','-an','-c:v','libx264','-preset','veryfast','-crf','18','-threads','1','-g','12','-keyint_min','12','-sc_threshold','0','-bf','0','-pix_fmt','yuv420p','-movflags','+faststart',str(directory/f'{v}.mp4')],stdin=subprocess.PIPE) for v in cams}
   rows=[];states=[state()];actions=[];observations=[obs.copy()];start=float(b.data.time);wall=time.time();render_delta=0.0
   def frame(info):
    nonlocal render_delta
    before=state()
    for view,name in cams.items():
     renderer.update_scene(b.data,camera=name);im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im)
     draw.rectangle((0,0,1280,76),fill='#101b29');draw.text((14,8),f'PPO {model.num_timesteps:,} | {scene} | cell {cell or "static"} | seed {seed} | {view} | t={b.data.time-start:.2f}s',font=font,fill='white')
     draw.text((14,42),f'confirmed={bool(b.grasp_confirmed)} | secured={bool(info.get("secured_grasp",False))} | hold={b.success_hold:.3f}s | success={bool(info.get("task_success",False))}',font=font,fill='white')
     writers[view].stdin.write(np.asarray(im).tobytes())
     if not rows:im.save(directory/f'{view}_poster.png')
    render_delta=max(render_delta,float(np.max(np.abs(state()-before))));assert render_delta==0.0
   info=initial;frame(initial)
   try:
    while True:
     action,_=model.predict(obs,deterministic=True);obs,reward,done,truncated,info=env.step(action)
     if smoke:
      pobs,pr,pdone,ptrunc,pinfo=plain.step(action)
      np.testing.assert_array_equal(pobs,obs);assert pr==reward and pdone==done and ptrunc==truncated
      np.testing.assert_array_equal(plain.base_env.data.qpos,b.data.qpos);np.testing.assert_array_equal(plain.base_env.data.qvel,b.data.qvel)
     actions.append(action.copy());observations.append(obs.copy());states.append(state())
     rows.append(dict(time=float(b.data.time-start),reward=float(reward),action=action.tolist(),confirmed=bool(b.grasp_confirmed),secured=bool(info.get('secured_grasp',False)),success_hold=float(b.success_hold),info={k:v for k,v in info.items() if isinstance(v,(int,float,bool,str,np.number))}))
     frame(info)
     if len(rows)%25==0:dump(OUT/f'worker_{index}_progress.json',dict(at=time.time(),episode=offset,seed=seed,steps=len(rows),scene=scene))
     if done or truncated or (smoke and len(rows)>=3):break
   finally:
    for w in writers.values():
     w.stdin.close()
     if w.wait()!=0:raise RuntimeError('ffmpeg failed')
   encoding={}
   for view in cams:
    import cv2
    cap=cv2.VideoCapture(str(directory/f'{view}.mp4'))
    probe=dict(width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),fps=float(cap.get(cv2.CAP_PROP_FPS)),frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    decoded,frame0=cap.read();cap.release()
    assert decoded and (probe['width'],probe['height'],probe['fps'],probe['frames'])==(1280,960,50.,len(rows)+1)
    encoding[view]=probe
   result=dict(seed=seed,scenario_name=scene,scene_index=job['scene_index'],cell=cell,parameters=initial['range_episode']['parameters'],job_id=job['id'],directory=str(directory.relative_to(OUT)),task_success=bool(info.get('task_success',False)),loaded_lift=bool(info.get('loaded_lift',False)),secured=bool(info.get('ever_grasped',False)),lift_attempt=bool(info.get('lift_attempt',False)),simulation_duration_s=float(b.data.time-start),steps=len(rows),episode_return=sum(r['reward'] for r in rows),wall_seconds=time.time()-wall,checkpoint_steps=model.num_timesteps,checkpoint_sha256=manifest['checkpoint_sha256'],source_commit=COMMIT,deterministic=True,difficulty='continuous_range',render_max_state_difference=render_delta,cameras=camera_meta,encoding=encoding,penalty_scales={'height':env._last_height_penalty_scale,'safety':env._last_safety_penalty_scale},diagnostics=dict(loaded_lift_frames=sum(bool(r['info'].get('loaded_lift',False)) for r in rows),secured_frames=sum(r['secured'] for r in rows),max_success_hold_s=max(r['success_hold'] for r in rows)),initial_info=initial,final_info=info)
   if smoke:plain.close()
   assert result['penalty_scales']=={'height':1.0,'safety':1.0}
   np.savez_compressed(directory/'trajectory.npz',states=np.array(states),actions=np.array(actions),observations=np.array(observations),times=np.arange(len(states))*.02)
   dump(directory/'metrics.json',rows);dump(directory/'metadata.json',result)
   dump(OUT/f'worker_{index}_progress.json',dict(at=time.time(),completed=offset+1,seed=seed,scene=scene))
 finally:renderer.close();env.close()
def aggregate(episodes):
 result=[]
 for job in load(OUT/'manifest.json')['jobs'][::20]:
  es=[e for e in episodes if e['scenario_name']==job['scenario'] and e['cell']==job['cell']]
  result.append(dict(scenario=job['scenario'],cell=job['cell'],episodes=len(es),success=sum(e['task_success'] for e in es),loaded=sum(e['loaded_lift'] for e in es),secured=sum(e['secured'] for e in es),attempt=sum(e['lift_attempt'] for e in es)))
 return result
def supervise(smoke=False,fault=False):
 summary=load(OUT/'summary.json');jobs=[]
 summary.pop('error',None);summary.pop('traceback',None);summary['worker_identities']=[]
 summary.update(supervisor=identity(os.getpid()),status='preflight' if smoke else 'starting',source_updated_at=time.time(),last_progress_at=time.time(),progress_deadline_at=time.time()+1200)
 dump(OUT/'summary.json',summary)
 try:
  for i in range(8):
   command=[sys.executable,'-c','import sys;sys.exit(7)'] if fault else [sys.executable,__file__,'worker',str(i)]+(['smoke'] if smoke else [])
   with (OUT/f'worker_{i}.log').open('w') as f:p=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,env=os.environ.copy(),cwd=SOURCE)
   jobs.append((i,p));summary.setdefault('worker_identities',[]).append(identity(p.pid))
  dump(OUT/'summary.json',summary)
  while True:
   now=time.time();episodes=[load(p) for p in sorted((OUT/('smoke' if smoke else 'episodes')).glob('*/metadata.json'))]
   for i,p in list(jobs):
    if p.poll() is not None:
     if p.returncode:raise RuntimeError(f'worker {i} exit {p.returncode}: '+(OUT/f'worker_{i}.log').read_text()[-1800:])
     jobs.remove((i,p))
   progress=max([summary['last_progress_at']]+[p.stat().st_mtime for p in OUT.glob('worker_*_progress.json')]+[p.stat().st_mtime for p in OUT.glob('worker_*_ready.json')])
   ready=all((OUT/f'worker_{i}_ready.json').exists() for i in range(8))
   summary.update(status='preflight' if smoke else ('recording' if ready else 'starting'),source_updated_at=now,heartbeat_at=now,last_progress_at=progress,progress_deadline_at=progress+1200,completed_episodes=len(episodes),groups=aggregate(episodes),worker_ready=ready)
   dump(OUT/'summary.json',summary)
   if not jobs:break
   if now>summary['progress_deadline_at']:raise RuntimeError('No recording progress for 1200 seconds')
   time.sleep(2 if fault else 10)
  assert len(episodes)==(20 if smoke else 400)
  assert all(e['render_max_state_difference']==0 for e in episodes)
  if not smoke:assert all(g['episodes']==20 for g in summary['groups'])
  dump(OUT/('smoke_results.json' if smoke else 'results.json'),dict(episodes=episodes,groups=summary['groups']))
  summary.update(status='preflight_complete' if smoke else 'complete',completed_at=time.time(),source_updated_at=time.time(),children_exited=True,result_path=str(OUT/'results.json'))
 except BaseException as e:
  summary.update(status='failed',error=str(e),traceback=traceback.format_exc(),source_updated_at=time.time())
  for _,p in jobs:
   if p.poll() is None:p.terminate()
 finally:dump(OUT/'summary.json',summary)
def prepare():
 assert not OUT.exists(),'Refusing duplicate test'
 source_manifest=load(RUN/'launch_manifest.json')
 for rel,sha in source_manifest['source_hashes'].items():assert hashlib.sha256((SOURCE/rel).read_bytes()).hexdigest()==sha
 pause=load(RUN/'pause_for_video400_receipt.json');assert pause['status']=='paused_by_user_for_video400'
 candidates=[]
 for p in [Path(pause['checkpoint']['path'])]:
  try:
   with zipfile.ZipFile(p) as z:meta=json.loads(z.read('data'));assert z.testzip() is None
   candidates.append((int(meta['num_timesteps']),p))
  except (OSError,zipfile.BadZipFile,KeyError,json.JSONDecodeError):pass
 assert candidates,'No valid checkpoint'
 steps,path=max(candidates,key=lambda x:x[0]);OUT.mkdir()
 shutil.copy2(path,OUT/'checkpoint.zip');shutil.copy2(RUN/'training_config.json',OUT/'training_config.json')
 with zipfile.ZipFile(OUT/'checkpoint.zip') as z:assert int(json.loads(z.read('data'))['num_timesteps'])==steps
 from panda_cable_grasp.rl.range_curriculum import ALL_SCENES,cells,BOUNDS
 jobs=[dict(id=f's{si}_c{cell or "static"}_seed{seed}',scene_index=si,scenario=n,cell=cell,seed=seed) for si,n in enumerate(ALL_SCENES) for cell in cells(n) for seed in range(20260804,20260824)]
 assert len(jobs)==400
 manifest=dict(source_run=str(RUN),source=str(SOURCE),source_commit=COMMIT,source_hashes=source_manifest['source_hashes'],checkpoint_steps=steps,checkpoint_sha256=hashlib.sha256((OUT/'checkpoint.zip').read_bytes()).hexdigest(),checkpoint_source=str(path),jobs=jobs,bounds=BOUNDS,expected_episodes=400,workers=8,deterministic=True,video='native named global/wrist cameras1280x960/50fps; every execution',seed_protocol='same 20 reset seeds per cell; original training bank fallback/resampling; actual bank source seeds in initial_info; continuous uniform inside each cell, not adaptive; bank entries may overlap training',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 dump(OUT/'manifest.json',manifest);dump(OUT/'summary.json',dict(status='prepared',source_updated_at=time.time(),last_progress_at=time.time(),progress_deadline_at=time.time()+1200,completed_episodes=0,expected_episodes=400,checkpoint_steps=steps,checkpoint_sha256=manifest['checkpoint_sha256'],run=str(OUT),workers=8));print(json.dumps({k:manifest[k] for k in ('checkpoint_steps','checkpoint_sha256','expected_episodes')}))
def launch():
 assert load(OUT/'summary.json')['status']=='preflight_complete';assert not (OUT/'launch_receipt.json').exists()
 smoke=load(OUT/'smoke_results.json')['episodes'];assert len(smoke)==20 and all(e['steps']==3 and e['render_max_state_difference']==0 for e in smoke)
 summary=load(OUT/'summary.json');summary.update(status='starting',worker_identities=[],worker_ready=False,completed_episodes=0,source_updated_at=time.time(),last_progress_at=time.time(),progress_deadline_at=time.time()+1200);dump(OUT/'summary.json',summary)
 for p in OUT.glob('worker_*_ready.json'):p.rename(p.with_suffix('.smoke.json'))
 for p in OUT.glob('worker_*_progress.json'):p.rename(p.with_suffix('.smoke.json'))
 with (OUT/'supervisor.log').open('w') as f:p=subprocess.Popen([sys.executable,__file__,'supervise'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL,cwd=SOURCE,env=os.environ.copy())
 dump(OUT/'launch_receipt.json',dict(**identity(p.pid),started_at=time.time(),output=str(OUT)));print(json.dumps(load(OUT/'launch_receipt.json')))
if __name__=='__main__':
 mode=sys.argv[1]
 if mode=='prepare':prepare()
 elif mode=='worker':worker(int(sys.argv[2]),len(sys.argv)>3)
 elif mode=='smoke':supervise(True)
 elif mode=='supervise':supervise()
 elif mode=='launch':launch()
 elif mode=='fault-test':
  test=OUT/'fault_test';test.mkdir();dump(test/'summary.json',load(OUT/'summary.json'));shutil.copy2(OUT/'manifest.json',test/'manifest.json')
  original=OUT;OUT=test;supervise(True,True);r=load(OUT/'summary.json');assert r['status']=='failed' and 'exit 7' in r['error'];dump(original/'fault_receipt.json',dict(passed=True,error=r['error'],at=time.time()));print('intentional exit7 recorded as failed')
