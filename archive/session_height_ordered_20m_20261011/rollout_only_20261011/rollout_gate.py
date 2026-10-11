"""Training-episode curriculum gates; no independent evaluation environments."""
import csv
import json
from collections import deque
from pathlib import Path
import time

VERSION='ordered_rollout_gate_v1'
PROTOCOL=dict(version=VERSION,target_episodes=100,replay_episodes=50,consecutive_windows=2,
    success=.6,loaded=.8,secured=.6,replay_success=.4,nonoverlapping=True,
    replay_difficulty='mastered: static 0, motion 1',all_replay_scenes_gated=True,
    independent_evaluation=False,statistics='stochastic training episodes; not held-out deterministic evaluation')

class Windows:
    def __init__(self, targets, replay, state=None):
        self.targets=tuple(targets);self.replay=tuple(replay)
        state=state or {}
        self.queues={n:deque(state.get('queues',{}).get(n,[]),maxlen=100 if n in targets else 50)
                     for n in (*self.targets,*self.replay)}
        self.fresh={n:state.get('fresh',{}).get(n,0) for n in self.queues}
        self.consecutive=state.get('consecutive',0)

    def add(self, name, result):
        if name in self.queues:
            self.queues[name].append(result);self.fresh[name]+=1

    def state(self):
        return dict(queues={n:list(q) for n,q in self.queues.items()},fresh=self.fresh.copy(),consecutive=self.consecutive)

    def decide(self):
        required={n:100 if n in self.targets else 50 for n in self.queues}
        if any(len(self.queues[n])<k or self.fresh[n]<k for n,k in required.items()): return None
        scenes=[];passed=True
        for n,k in required.items():
            q=list(self.queues[n]);counts={key:sum(r[key] for r in q) for key in ('success','loaded','secured','attempt')}
            ok=counts['success']/k >= (.6 if n in self.targets else .4)
            if n in self.targets: ok=ok and counts['loaded']/k>=.8 and counts['secured']/k>=.6
            passed=passed and ok
            scenes.append(dict(scenario=n,role='target' if n in self.targets else 'replay',episodes=k,counts=counts,passed=ok))
            self.fresh[n]=0
        self.consecutive=self.consecutive+1 if passed else 0
        return dict(scenes=scenes,passed=passed,consecutive=self.consecutive,promote=self.consecutive>=2)

def install_env_protocol():
    from panda_cable_grasp.rl.environment import RLCableGraspEnv
    if hasattr(RLCableGraspEnv,'set_rollout_curriculum'): return
    original_select=RLCableGraspEnv._select_training_scenario
    original_info=RLCableGraspEnv._augment_info
    def configure(self,phase,names,probabilities,difficulties):
        self.set_training_scenario_distribution(names,probabilities)
        self._rollout_pending_phase=int(phase)
        self._rollout_difficulties=dict(difficulties)
    def select(self,explicit_seed=None):
        scene=original_select(self,explicit_seed)
        if scene is not None and hasattr(self,'_rollout_difficulties'):
            difficulty=self._rollout_difficulties[scene.name]
            self.set_motion_difficulty(difficulty)
            # Same existing difficulty formula; choose the mastered value for replay.
            self.base_env.config.disturbance_strength=scene.disturbance_strength*(.5+(self._final_disturbance_scale-.5)*difficulty)
            self.base_env.config.motion_frequency_scale=scene.frequency_scale*(2.+difficulty)/3.
            self._rollout_episode_phase=self._rollout_pending_phase
            self._rollout_episode_difficulty=difficulty
        return scene
    def info(self,*args,**kwargs):
        out=original_info(self,*args,**kwargs)
        out['rollout_episode_phase']=getattr(self,'_rollout_episode_phase',-1)
        out['rollout_episode_difficulty']=getattr(self,'_rollout_episode_difficulty',-1.)
        return out
    RLCableGraspEnv.set_rollout_curriculum=configure
    RLCableGraspEnv._select_training_scenario=select
    RLCableGraspEnv._augment_info=info

def difficulties(phase):
    from panda_cable_grasp.rl.ordered_curriculum import GROUPS,PHASES
    group,difficulty=PHASES[phase]
    return {n:float(difficulty if g==group else 0. if g<2 else 1.)
            for g,names in enumerate(GROUPS[:group+1]) for n in names}

def configure_env(env,phase):
    from panda_cable_grasp.rl.ordered_curriculum import MIXES,PHASES
    mix=MIXES[PHASES[phase][0]]
    env.set_rollout_curriculum(phase,tuple(n for n,p in mix),tuple(p for n,p in mix),difficulties(phase))

def install(train,output,initial_phase):
    from panda_cable_grasp.rl.ordered_curriculum import OrderedMotionCurriculum,GROUPS
    output=Path(output)
    original_worker=train.make_worker
    def make_worker(rank,args):
        factory=original_worker(rank,args)
        def initialize():
            install_env_protocol()
            env=factory();configure_env(env,initial_phase)
            return env
        return initialize
    train.make_worker=make_worker

    class RolloutCurriculum(OrderedMotionCurriculum):
        def _apply(self):
            mix=self.current_training_mix
            self.training_env.env_method('set_rollout_curriculum',self.phase_index,
                tuple(n for n,p in mix),tuple(p for n,p in mix),difficulties(self.phase_index))
            self.model.ordinary_evaluation_phase=self.phase_index
            self.model.ordinary_evaluation_difficulty=self.difficulty
            self.model.ordinary_evaluation_scenarios=self.current_scenarios
            self._last_update=self.num_timesteps
            for k,v in dict(phase=self.phase_index,stage=self.stage_index,motion_difficulty=self.difficulty,
                qualifying_evaluations=self.windows.consecutive).items(): self.logger.record('curriculum/'+k,v)

        def _new_windows(self,state=None):
            replay=tuple(n for names in GROUPS[:self.stage_index] for n in names)
            self.windows=Windows(self.current_scenarios,replay,state)

        def _on_training_start(self):
            saved=getattr(self.model,'rollout_gate_state',{})
            self.phase_index=int(saved.get('phase',initial_phase))
            self._new_windows(saved.get('windows'))
            self.episode_counter=int(saved.get('episode_counter',0))
            self._apply()
            output.mkdir(parents=True,exist_ok=True)
            p=output/'rollout_episodes.csv'
            exists=p.exists()
            self.file=p.open('a',newline='');self.writer=csv.DictWriter(self.file,fieldnames=[
                'episode','timesteps','scenario_name','phase_at_completion','difficulty_at_completion',
                'eligible_for_current_gate','success','lift_attempt','loaded_lift','ever_grasped','ever_aligned_pinch'])
            if not exists:self.writer.writeheader();self.file.flush()
            self._save_state()

        def _save_state(self,persist=True):
            state=dict(version=VERSION,phase=self.phase_index,windows=self.windows.state(),episode_counter=self.episode_counter)
            self.model.rollout_gate_state=state
            if not persist:return
            record=dict(version=VERSION,phase=self.phase_index,consecutive=self.windows.consecutive,
                timesteps=self.num_timesteps,updated_at=time.time(),protocol=PROTOCOL,
                episode_counter=self.episode_counter,target_scenes=self.current_scenarios,
                difficulties=difficulties(self.phase_index),scenes=[dict(scenario=n,
                    role='target' if n in self.windows.targets else 'replay',episodes=len(q),
                    fresh_episodes=self.windows.fresh[n],required_episodes=100 if n in self.windows.targets else 50,
                    counts={k:sum(r[k] for r in q) for k in ('success','loaded','secured','attempt')})
                    for n,q in self.windows.queues.items()])
            p=output/'rollout_gate_status.json';tmp=p.with_suffix('.tmp')
            tmp.write_text(json.dumps(record));tmp.replace(p)

        def _on_step(self):
            for done,info in zip(self.locals['dones'],self.locals['infos']):
                if not done:continue
                episode=info.get('episode',{})
                if not episode:continue
                name=info['scenario_name'];phase=int(info['rollout_episode_phase']);difficulty=float(info['rollout_episode_difficulty'])
                eligible=(phase==self.phase_index and name in self.windows.queues
                    and abs(difficulty-difficulties(self.phase_index)[name])<1e-9)
                result=dict(success=int(bool(episode.get('success',info.get('success',False)))),
                    loaded=int(bool(episode.get('loaded_lift',info.get('loaded_lift',False)))),
                    secured=int(bool(episode.get('ever_grasped',info.get('ever_grasped',False)))),
                    attempt=int(bool(episode.get('lift_attempt',info.get('lift_attempt',False)))))
                self.episode_counter+=1
                self.writer.writerow(dict(episode=self.episode_counter,timesteps=self.num_timesteps,scenario_name=name,
                    phase_at_completion=phase,difficulty_at_completion=difficulty,eligible_for_current_gate=int(eligible),
                    success=result['success'],lift_attempt=result['attempt'],loaded_lift=result['loaded'],ever_grasped=result['secured'],
                    ever_aligned_pinch=int(bool(episode.get('ever_aligned_pinch',info.get('ever_aligned_pinch',False))))))
                if eligible:self.windows.add(name,result)
            self.file.flush();self._save_state(persist=False)
            return True

        def _on_rollout_end(self):
            decision=self.windows.decide()
            if decision:
                record=dict(timesteps=self.num_timesteps,phase=self.phase_index,difficulty=self.difficulty,**decision)
                if decision['promote'] and self.phase_index<len(self.phases)-1:
                    self.phase_index+=1;self._new_windows();self._apply()
                record['next_phase']=self.phase_index
                with (output/'rollout_gate_rounds.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                print('rollout_curriculum '+json.dumps(record),flush=True)
            self._save_state()

        def _on_training_end(self):
            self._save_state();self.file.close()

    import panda_cable_grasp.rl.ordered_curriculum as ordered
    ordered.OrderedMotionCurriculum=RolloutCurriculum
    ordered.PROTOCOL=dict(PROTOCOL,phases=ordered.PHASES,groups=ordered.GROUPS,training_mixes=ordered.MIXES)
    return RolloutCurriculum
