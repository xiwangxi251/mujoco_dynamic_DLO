"""Continuous motion sampling with per-scene, per-cell training gates."""
from collections import deque
import csv
import itertools
import json
from pathlib import Path
import time
import zipfile

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from .ordered_curriculum import GROUPS, MIXES
from ..scenarios.registry import get_scenario

VERSION = 'range_ability_rollout_speed025_v2'
ALL_SCENES = tuple(n for group in GROUPS for n in group)
RIGID_SPEED = (0.25 * 7 / 9, 0.25)
SHAPE_STRENGTH = (1.0, 1.5)
SHAPE_FREQUENCY = (7 / 9, 1.0)
PROTOCOL = dict(version=VERSION, groups=GROUPS, budget=20000000,
    static_episodes=100, cell_episodes=50, success=.4, loaded=.5, secured=.4,
    replay_episodes=20, replay_success=.3, consecutive_windows=1,
    uniform_fraction=.2, adaptive_fraction=.8, independent_evaluation=False,
    rigid_target_speed=RIGID_SPEED, shape_drive_strength=SHAPE_STRENGTH,
    shape_frequency=SHAPE_FREQUENCY, replay_time_scale=SHAPE_FREQUENCY,
    sampling='continuous uniform inside binary parameter cells, one draw per episode',
    adaptation='quality=min(success/.4,loaded/.5,secured/.4)*min(n/50,1); '
               'readiness=min(quality of immediate easier neighbours); '
               'priority=(.02+readiness)*(.25+1-quality); .2 uniform+.8 normalized priority',
    replay_sampling='same full range, own saved per-cell performance',
    final_training='all seven scenes equally after group 6 passes',
    statistics='stochastic training episodes; not independent evaluation')

def axes(name):
    if _OUTPUT_RESPONSE is not None and name in _OUTPUT_RESPONSE.tables:
        return _OUTPUT_RESPONSE.axes(name)
    scenario = get_scenario(name)
    if scenario.motion_type.value == 'static':
        return ()
    if scenario.replay_bank:
        return ('replay_time_scale',)
    if scenario.motion_type.value == 'rigid':
        return ('rigid_speed',)
    if scenario.motion_type.value == 'shape':
        return ('shape_strength', 'shape_frequency')
    return ('rigid_speed', 'shape_strength', 'shape_frequency')

def cells(name):
    return tuple(''.join(map(str, x)) for x in itertools.product((0, 1), repeat=len(axes(name)))) or ('',)

BOUNDS = dict(rigid_speed=RIGID_SPEED, shape_strength=SHAPE_STRENGTH,
              shape_frequency=SHAPE_FREQUENCY, replay_time_scale=SHAPE_FREQUENCY)

_OUTPUT_RESPONSE = None

def configure_output_response(path):
    """Opt in to independently accepted actual-output ranges, before reset."""
    global _OUTPUT_RESPONSE, VERSION
    from .motion_response import MotionResponse
    response=MotionResponse.load(path)
    required={n for n in ALL_SCENES if get_scenario(n).motion_type.value in {'shape','combined'}}
    if not required <= response.tables.keys():
        raise ValueError('Measured response must cover both shape and combined scenes')
    _OUTPUT_RESPONSE=response
    VERSION='range_actual_amplitude_speed_v1_'+response.sha256[:12]
    PROTOCOL.update(version=VERSION, measured_response_sha256=response.sha256,
        measured_response_file=str(Path(path).resolve()),
        measured_ranges={n:response.bounds(n) for n in response.tables},
        shape_axes='actual amplitude RMS mm and residual node speed m/s',
        response_validation=response.profile['validation'],
        runtime='original continuous deformation drive; no extra pose hold or airborne shape gate')

def sample_episode(scenario, rng, spec, phase):
    names=axes(scenario.name); available=cells(scenario.name)
    probabilities=np.asarray(spec['probabilities'], dtype=float)
    if len(probabilities)!=len(available) or not np.isclose(probabilities.sum(),1) or np.any(probabilities<0):
        raise ValueError('Invalid continuous sampler probabilities')
    cell=available[int(rng.choice(len(available),p=probabilities))]
    params={}
    response=_OUTPUT_RESPONSE if _OUTPUT_RESPONSE is not None and scenario.name in _OUTPUT_RESPONSE.tables else None
    bounds=response.bounds(scenario.name) if response else BOUNDS
    for axis,bit in zip(names,cell):
        lo,hi=bounds[axis];mid=(lo+hi)/2
        params[axis]=float(rng.uniform(lo,mid) if bit=='0' else rng.uniform(mid,hi))
    overrides=dict(rigid_frequency_scale=None, replay_time_scale=1.0)
    if 'rigid_speed' in params:
        overrides.update(rigid_motion_nominal_speed=params['rigid_speed'], rigid_frequency_scale=1.0)
    if 'shape_strength' in params:
        overrides.update(disturbance_strength=params['shape_strength'], motion_frequency_scale=params['shape_frequency'])
    if 'replay_time_scale' in params:
        overrides['replay_time_scale']=params['replay_time_scale']
    if response:
        drivers=response.solve(scenario.name,params)
        allowed={'disturbance_strength','motion_frequency_scale','rigid_motion_nominal_speed'}
        if set(drivers)-allowed: raise ValueError('Calibration may only adjust original motion driver parameters')
        overrides.update(drivers)
        if 'rigid_motion_nominal_speed' in drivers: overrides['rigid_frequency_scale']=1.
    return dict(version=VERSION, scenario=scenario.name, phase=int(phase), cell=cell,
                axes=names, parameters=params, overrides=overrides,
                sampling_probabilities=probabilities.tolist(),
                **(dict(measured_response_sha256=response.sha256,
                         internal_driver_parameters=drivers) if response else {}))

def counts(queue):
    return {k:sum(r[k] for r in queue) for k in ('success','loaded','secured','attempt')}

def qualified(queue):
    n=len(queue)
    if not n:return False
    c=counts(queue)
    return all(c[k]/n>=PROTOCOL[k] for k in ('success','loaded','secured'))

class Evidence:
    def __init__(self,state=None):
        state=state or {}
        self.queues={n:{cell:deque(state.get('queues',{}).get(n,{}).get(cell,[]),
            maxlen=100 if not axes(n) else 50) for cell in cells(n)} for n in ALL_SCENES}
        self.fresh={n:{cell:int(state.get('fresh',{}).get(n,{}).get(cell,0)) for cell in cells(n)} for n in ALL_SCENES}
        self.retention={n:deque(state.get('retention',{}).get(n,[]),maxlen=20) for n in ALL_SCENES}
        self.retention_fresh={n:int(state.get('retention_fresh',{}).get(n,0)) for n in ALL_SCENES}

    def add(self,name,cell,result):
        self.queues[name][cell].append(result);self.fresh[name][cell]+=1
        self.retention[name].append(result);self.retention_fresh[name]+=1

    def state(self):
        return dict(queues={n:{c:list(q) for c,q in d.items()} for n,d in self.queues.items()},
            fresh=self.fresh,retention={n:list(q) for n,q in self.retention.items()},retention_fresh=self.retention_fresh)

    def reset_gate(self):
        for n in ALL_SCENES:
            self.retention_fresh[n]=0
            for c in cells(n):self.fresh[n][c]=0

    def probabilities(self,name):
        cs=cells(name)
        if len(cs)==1:return [1.0]
        quality={}
        for c in cs:
            q=self.queues[name][c];total=counts(q);n=len(q)
            quality[c]=min(1.,min((total[k]/max(n,1))/PROTOCOL[k] for k in ('success','loaded','secured')))*min(n/50,1.)
        priority=[]
        for c in cs:
            easier=[c[:i]+'0'+c[i+1:] for i,b in enumerate(c) if b=='1']
            ready=min((quality[x] for x in easier),default=1.)
            priority.append((.02+ready)*(.25+1-quality[c]))
        p=np.asarray(priority,dtype=float);p/=p.sum()
        return (.2/len(cs)+.8*p).tolist()

    def rows(self,phase):
        target=ALL_SCENES if phase==6 else GROUPS[phase]
        previous=() if phase==6 else tuple(n for g in GROUPS[:phase] for n in g)
        result=[]
        for name in target:
            required=100 if not axes(name) else 50
            for c,q in self.queues[name].items():
                ready=len(q)>=required and self.fresh[name][c]>=required
                result.append(dict(scenario=name,role='target',cell=c,axes=axes(name),episodes=len(q),
                    required=required,fresh=self.fresh[name][c],counts=counts(q),ready=ready,passed=ready and qualified(q)))
        for name in previous:
            q=self.retention[name];ready=len(q)==20 and self.retention_fresh[name]>=20
            result.append(dict(scenario=name,role='replay',episodes=len(q),required=20,
                fresh=self.retention_fresh[name],counts=counts(q),ready=ready,
                passed=ready and counts(q)['success']>=6))
        return result

    def decide(self,phase):
        rows=self.rows(phase)
        if not all(r['ready'] for r in rows):return None
        self.reset_gate()
        return dict(scenes=rows,passed=all(r['passed'] for r in rows))

def specs_for(evidence):
    return {n:dict(probabilities=evidence.probabilities(n)) for n in ALL_SCENES}

def configure_env(env,phase,state=None):
    evidence=Evidence(state)
    mix=MIXES[phase] if phase<6 else tuple((n,1/len(ALL_SCENES)) for n in ALL_SCENES)
    env.set_range_curriculum(phase,tuple(n for n,p in mix),tuple(p for n,p in mix),specs_for(evidence))

def checkpoint_state(path):
    if path is None:return {}
    p=Path(path)
    if not p.is_file():p=Path(str(p)+'.zip')
    with zipfile.ZipFile(p) as z:state=json.loads(z.read('data')).get('range_curriculum_state',{})
    if state.get('version')!=VERSION:raise ValueError('Not a compatible range PPO checkpoint')
    if not 0<=state['phase']<=6:raise ValueError('Invalid range phase')
    return state

class RangeMotionCurriculum(BaseCallback):
    def __init__(self,output_dir,**kwargs):
        super().__init__();self.output=Path(output_dir)

    def _apply(self):
        mix=MIXES[self.phase] if self.phase<6 else tuple((n,1/len(ALL_SCENES)) for n in ALL_SCENES)
        self.training_env.env_method('set_range_curriculum',self.phase,
            tuple(n for n,p in mix),tuple(p for n,p in mix),specs_for(self.evidence))
        self.model.ordinary_evaluation_phase=self.phase
        self.model.ordinary_evaluation_difficulty=None
        self.model.ordinary_evaluation_scenarios=GROUPS[self.phase] if self.phase<6 else ALL_SCENES

    def _on_training_start(self):
        saved=getattr(self.model,'range_curriculum_state',{})
        if saved and saved.get('version')!=VERSION:raise ValueError('Range checkpoint version mismatch')
        self.phase=int(saved.get('phase',0));self.evidence=Evidence(saved.get('evidence'))
        self.episode_counter=int(saved.get('episode_counter',0));self._apply()
        self.output.mkdir(parents=True,exist_ok=True)
        p=self.output/'range_episodes.csv';exists=p.exists()
        self.file=p.open('a',newline='',encoding='utf-8')
        self.writer=csv.DictWriter(self.file,fieldnames=['episode','timesteps','scenario_name','seed','phase',
            'cell','eligible','parameters','precontact_mean_speed','precontact_speed_samples',
            'success','loaded','secured','attempt'])
        if not exists:self.writer.writeheader()
        self._save()

    def _on_step(self):
        for done,info in zip(self.locals['dones'],self.locals['infos']):
            if not done:continue
            if not info.get('episode'):continue
            sample=info['range_episode'];name=info['scenario_name'];ep=info['episode']
            result={k:int(bool(ep.get(field,info.get(field,False)))) for k,field in
                dict(success='success',loaded='loaded_lift',secured='ever_grasped',attempt='lift_attempt').items()}
            eligible=int(sample['phase']==self.phase)
            if eligible:self.evidence.add(name,sample['cell'],result)
            self.episode_counter+=1
            self.writer.writerow(dict(episode=self.episode_counter,timesteps=self.num_timesteps,
                scenario_name=name,seed=info.get('episode_seed'),phase=sample['phase'],cell=sample['cell'],eligible=eligible,
                parameters=json.dumps(sample['parameters']),precontact_mean_speed=sample['precontact_mean_speed'],
                precontact_speed_samples=sample['precontact_speed_samples'],**result))
        self.file.flush();self._save(persist=False)
        return True

    def _save(self,persist=True):
        self.model.range_curriculum_state=dict(version=VERSION,phase=self.phase,
            evidence=self.evidence.state(),episode_counter=self.episode_counter)
        if persist:
            record=dict(version=VERSION,phase=self.phase,timesteps=self.num_timesteps,updated_at=time.time(),
                consolidation=self.phase==6,protocol=PROTOCOL,scenes=self.evidence.rows(self.phase),
                sampling_specs=specs_for(self.evidence),episode_counter=self.episode_counter)
            p=self.output/'range_gate_status.json';tmp=p.with_suffix('.tmp')
            tmp.write_text(json.dumps(record),encoding='utf-8');tmp.replace(p)

    def _on_rollout_end(self):
        decision=self.evidence.decide(self.phase) if self.phase<6 else None
        if decision:
            record=dict(timesteps=self.num_timesteps,phase=self.phase,**decision)
            if decision['passed']:self.phase+=1
            record['next_phase']=self.phase
            with (self.output/'range_gate_rounds.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
        self._apply();self._save()

    def _on_training_end(self):
        self._save();self.file.close()
