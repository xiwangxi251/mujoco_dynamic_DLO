"""Training-episode curriculum gates; no independent evaluation environments."""
import csv
import json
from collections import deque
from pathlib import Path
import time
import zipfile

from .ordered_curriculum import GROUPS, PHASES, MIXES, OrderedMotionCurriculum, structure_for_family

VERSION='ordered_rollout_gate_v1'
PROTOCOL=dict(version=VERSION,target_episodes=100,replay_episodes=50,consecutive_windows=2,
    success=.6,loaded=.8,secured=.6,replay_success=.4,nonoverlapping=True,
    replay_difficulty='mastered: static 0, motion 1',all_replay_scenes_gated=True,
    independent_evaluation=False,statistics='stochastic training episodes; not held-out deterministic evaluation')
FAST_PROTOCOL = dict(PROTOCOL, version='ordered_rollout_gate_fast_v2', replay_episodes=20,
    consecutive_windows=1, success=.4, loaded=.5, secured=.4, replay_success=.3)
PROFILES = {'strict': PROTOCOL, 'fast': FAST_PROTOCOL}
ALL_SCENES = tuple(n for group in GROUPS for n in group)
CONSOLIDATION_MIX = tuple((n, .3 if n in (GROUPS[1][0], GROUPS[3][0]) else .08)
                          for n in ALL_SCENES)
CONSOLIDATION_PROTOCOL = dict(FAST_PROTOCOL, version='ordered_rollout_consolidation_v3',
    terminal_stage='all scenes at nominal difficulty; optimize to original budget',
    terminal_training_mix=CONSOLIDATION_MIX, terminal_episodes_per_scene=100)
PROFILES['consolidate'] = CONSOLIDATION_PROTOCOL


def protocol_for(profile, scene_family='cable_x075_v1'):
    result = dict(PROFILES[profile])
    if profile == 'consolidate' and scene_family != 'cable_x075_v1':
        result['terminal_training_mix'] = structure_for(profile, scene_family)[2][-1]
    return result


def structure_for(profile, scene_family='cable_x075_v1'):
    groups, mixes = structure_for_family(scene_family)
    if profile == 'consolidate':
        scenes = tuple(n for group in groups for n in group)
        final_mix = tuple((n, .3 if n in (groups[1][0], groups[3][0]) else .08) for n in scenes)
        return groups + (scenes,), PHASES + ((6, 1.),), mixes + (final_mix,)
    return groups, PHASES, mixes


def restored_phase(state, default, profile):
    phase = int(state.get('phase', default))
    if profile == 'consolidate' and phase == 13 and state.get('windows', {}).get('consecutive', 0) >= 1:
        # The existing checkpoint already contains a completed passing final-stage window.
        phase = 14
    return phase

class Windows:
    def __init__(self, targets, replay, state=None, protocol=None):
        self.targets=tuple(targets);self.replay=tuple(replay)
        self.protocol = dict(protocol or PROTOCOL)
        state=state or {}
        self.queues={n:deque(state.get('queues',{}).get(n,[]),maxlen=self.required(n))
                     for n in (*self.targets,*self.replay)}
        self.fresh={n:state.get('fresh',{}).get(n,0) for n in self.queues}
        self.consecutive=state.get('consecutive',0)

    def required(self, name):
        return self.protocol['target_episodes' if name in self.targets else 'replay_episodes']

    def add(self, name, result):
        if name in self.queues:
            self.queues[name].append(result);self.fresh[name]+=1

    def state(self):
        return dict(queues={n:list(q) for n,q in self.queues.items()},fresh=self.fresh.copy(),consecutive=self.consecutive)

    def decide(self):
        required={n:self.required(n) for n in self.queues}
        if any(len(self.queues[n])<k or self.fresh[n]<k for n,k in required.items()): return None
        scenes=[];passed=True
        for n,k in required.items():
            q=list(self.queues[n]);counts={key:sum(r[key] for r in q) for key in ('success','loaded','secured','attempt')}
            ok=counts['success']/k >= self.protocol['success' if n in self.targets else 'replay_success']
            if n in self.targets:
                ok=ok and counts['loaded']/k>=self.protocol['loaded'] and counts['secured']/k>=self.protocol['secured']
            passed=passed and ok
            scenes.append(dict(scenario=n,role='target' if n in self.targets else 'replay',episodes=k,counts=counts,passed=ok))
            self.fresh[n]=0
        self.consecutive=self.consecutive+1 if passed else 0
        return dict(scenes=scenes,passed=passed,consecutive=self.consecutive,
                    promote=self.consecutive>=self.protocol['consecutive_windows'])

def difficulties(phase, groups=GROUPS):
    if phase == 14:
        return {n:float(0. if g < 2 else 1.) for g,names in enumerate(groups) for n in names}
    group,difficulty=PHASES[phase]
    return {n:float(difficulty if g==group else 0. if g<2 else 1.)
            for g,names in enumerate(groups[:group+1]) for n in names}

def configure_env(env,phase,scene_family='cable_x075_v1'):
    groups, mixes = structure_for_family(scene_family)
    mix=structure_for('consolidate',scene_family)[2][-1] if phase == 14 else mixes[PHASES[phase][0]]
    env.set_rollout_curriculum(phase,tuple(n for n,p in mix),tuple(p for n,p in mix),difficulties(phase,groups))

def checkpoint_phase(path, gate_profile='strict', scene_family='cable_x075_v1'):
    """Read JSON curriculum identity before spawned workers perform their first reset."""
    if path is None:return 0
    path=Path(path)
    if not path.is_file():path=Path(str(path)+'.zip')
    with zipfile.ZipFile(path) as z:data=json.loads(z.read('data'))
    state=data.get('rollout_gate_state',{})
    from .target_defaults import checkpoint_configuration
    saved_family = state.get('scene_family', checkpoint_configuration(path).get('workspace_scene_family', 'cable_x075_v1'))
    if saved_family != scene_family:
        raise ValueError('Rollout curriculum checkpoint uses a different scene family')
    if state and state.get('version') not in {p['version'] for p in PROFILES.values()}:
        raise ValueError('Incompatible rollout curriculum checkpoint version')
    if state.get('version') == CONSOLIDATION_PROTOCOL['version'] and gate_profile != 'consolidate':
        raise ValueError('Consolidation checkpoint requires consolidate gate profile')
    phase=restored_phase(state,data.get('ordinary_evaluation_phase',0),gate_profile)
    if not 0<=phase<len(structure_for(gate_profile)[1]):raise ValueError('Invalid checkpoint curriculum phase')
    return phase

class RolloutMotionCurriculum(OrderedMotionCurriculum):
    def __init__(self, output_dir, initial_phase=0, gate_profile='strict', scene_family='cable_x075_v1', **kwargs):
        self.gate_profile = gate_profile
        self.scene_family = scene_family
        groups, phases, mixes = structure_for(gate_profile, scene_family)
        if gate_profile == 'consolidate':
            kwargs.update(stages=groups, training_mixes=mixes)
        super().__init__(**kwargs)
        self.phases = phases
        self.protocol = protocol_for(gate_profile, scene_family)
        self.output=Path(output_dir)
        self.initial_phase=int(initial_phase)
        if not 0<=self.initial_phase<len(self.phases):raise ValueError('Invalid initial curriculum phase')

    def _apply(self):
        mix=self.current_training_mix
        self.training_env.env_method('set_rollout_curriculum',self.phase_index,
            tuple(n for n,p in mix),tuple(p for n,p in mix),difficulties(self.phase_index,self.stages[:6]))
        self.model.ordinary_evaluation_phase=self.phase_index
        self.model.ordinary_evaluation_difficulty=self.difficulty
        self.model.ordinary_evaluation_scenarios=self.current_scenarios
        self._last_update=self.num_timesteps
        for k,v in dict(phase=self.phase_index,stage=self.stage_index,motion_difficulty=self.difficulty,
            qualifying_evaluations=self.windows.consecutive).items(): self.logger.record('curriculum/'+k,v)

    def _new_windows(self,state=None):
        replay=() if self.phase_index == 14 else tuple(n for names in self.stages[:self.stage_index] for n in names)
        self.windows=Windows(self.current_scenarios,replay,state,self.protocol)

    def _on_training_start(self):
        saved=getattr(self.model,'rollout_gate_state',{})
        if saved and saved.get('scene_family','cable_x075_v1') != self.scene_family:
            raise ValueError('Incompatible rollout curriculum scene family')
        if saved and saved.get('version') not in {p['version'] for p in PROFILES.values()}:
            raise ValueError('Incompatible rollout curriculum state')
        self.phase_index=restored_phase(saved,self.initial_phase,self.gate_profile)
        changed = bool(saved and (saved.get('version') != self.protocol['version']
                                  or saved.get('phase') != self.phase_index))
        self._new_windows(None if changed else saved.get('windows'))
        if changed:
            self.model.rollout_gate_migration = dict(from_version=saved['version'],
                to_version=self.protocol['version'], phase=self.phase_index,
                timesteps=self.num_timesteps, windows_reset=True)
        self.episode_counter=int(saved.get('episode_counter',0))
        self._apply()
        self.output.mkdir(parents=True,exist_ok=True)
        p=self.output/'rollout_episodes.csv'
        self.extra_episode_fields = ('rigid_motion_nominal_speed', 'motion_frequency_scale',
            'initial_sector_coverage', 'initial_sector_fraction_used', 'initial_shape_source_seed',
            'initial_cable_dx', 'initial_cable_dy', 'motion_profile_hash') if self.scene_family in {'cable_x075_sector_speed_v1', 'cable_x075_sector_speed_v2', 'cable_x075_sector_speed_v3'} else ()
        if self.scene_family in {'cable_x075_sector_speed_v2', 'cable_x075_sector_speed_v3'}:
            self.extra_episode_fields += ('initial_sector_yaw_deg',)
        if self.scene_family == 'cable_x075_sector_speed_v3':
            self.extra_episode_fields += ('initial_rigid_yaw_deg',)
        exists=p.exists()
        self.file=p.open('a',newline='');self.writer=csv.DictWriter(self.file,fieldnames=[
            'episode','timesteps','scenario_name','phase_at_completion','difficulty_at_completion',
            'eligible_for_current_gate','success','lift_attempt','loaded_lift','ever_grasped','ever_aligned_pinch'] + list(self.extra_episode_fields))
        if not exists:self.writer.writeheader();self.file.flush()
        self._save_state()

    def _save_state(self,persist=True):
        state=dict(version=self.protocol['version'],phase=self.phase_index,windows=self.windows.state(),episode_counter=self.episode_counter,
                   scene_family=self.scene_family)
        self.model.rollout_gate_state=state
        if not persist:return
        record=dict(version=self.protocol['version'],phase=self.phase_index,consecutive=self.windows.consecutive,
            timesteps=self.num_timesteps,updated_at=time.time(),protocol=self.protocol,
            episode_counter=self.episode_counter,target_scenes=self.current_scenarios,
            difficulties=difficulties(self.phase_index,self.stages[:6]),scenes=[dict(scenario=n,
                role='target' if n in self.windows.targets else 'replay',episodes=len(q),
                fresh_episodes=self.windows.fresh[n],required_episodes=self.windows.required(n),
                counts={k:sum(r[k] for r in q) for k in ('success','loaded','secured','attempt')})
                for n,q in self.windows.queues.items()])
        p=self.output/'rollout_gate_status.json';tmp=p.with_suffix('.tmp')
        tmp.write_text(json.dumps(record));tmp.replace(p)

    def _on_step(self):
        for done,info in zip(self.locals['dones'],self.locals['infos']):
            if not done:continue
            episode=info.get('episode',{})
            if not episode:continue
            name=info['scenario_name'];phase=int(info['rollout_episode_phase']);difficulty=float(info['rollout_episode_difficulty'])
            eligible=(phase==self.phase_index and name in self.windows.queues
                and abs(difficulty-difficulties(self.phase_index,self.stages[:6])[name])<1e-9)
            result=dict(success=int(bool(episode.get('success',info.get('success',False)))),
                loaded=int(bool(episode.get('loaded_lift',info.get('loaded_lift',False)))),
                secured=int(bool(episode.get('ever_grasped',info.get('ever_grasped',False)))),
                attempt=int(bool(episode.get('lift_attempt',info.get('lift_attempt',False)))))
            self.episode_counter+=1
            self.writer.writerow(dict(episode=self.episode_counter,timesteps=self.num_timesteps,scenario_name=name,
                phase_at_completion=phase,difficulty_at_completion=difficulty,eligible_for_current_gate=int(eligible),
                success=result['success'],lift_attempt=result['attempt'],loaded_lift=result['loaded'],ever_grasped=result['secured'],
                ever_aligned_pinch=int(bool(episode.get('ever_aligned_pinch',info.get('ever_aligned_pinch',False)))),
                **{key: info.get(key) for key in self.extra_episode_fields}))
            if eligible:self.windows.add(name,result)
        self.file.flush();self._save_state(persist=False)
        return True

    def _on_rollout_end(self):
        decision=self.windows.decide()
        if decision:
            if self.phase_index == 14:
                decision['promote'] = False
            record=dict(timesteps=self.num_timesteps,phase=self.phase_index,difficulty=self.difficulty,**decision)
            if decision['promote'] and self.phase_index<len(self.phases)-1:
                self.phase_index+=1;self._new_windows();self._apply()
            record['next_phase']=self.phase_index
            with (self.output/'rollout_gate_rounds.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print('rollout_curriculum '+json.dumps(record),flush=True)
        self._save_state()

    def _on_training_end(self):
        self._save_state();self.file.close()
