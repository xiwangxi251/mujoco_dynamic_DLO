"""Joint inverse of measured motion response, without live object feedback."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import Delaunay, QhullError


class MotionResponse:
    def __init__(self, profile: dict):
        if profile.get('version') != 'measured_motion_response_v1':
            raise ValueError('Unsupported measured motion response version')
        validation=profile.get('validation',{})
        if validation.get('status') != 'passed' or not validation.get('independent_physics_validation'):
            raise ValueError('Motion response requires independent physical acceptance before training')
        self.profile=profile
        self.sha256=hashlib.sha256(json.dumps(profile,sort_keys=True).encode()).hexdigest()
        self.tables={}
        for name, table in profile['scenes'].items():
            axes=tuple(table['axes']); names=tuple(table['driver_parameters'])
            if len(set(axes))!=len(axes) or not set(axes)<= {'shape_amplitude_mm','shape_speed_m_s','rigid_speed_m_s'}:
                raise ValueError('Only distinct actual amplitude and speed axes are supported')
            if not set(names)<= {'disturbance_strength','motion_frequency_scale','rigid_motion_nominal_speed'}:
                raise ValueError('Response map may only adjust original driver parameters')
            output=np.asarray(table['measured_outputs'],dtype=float)
            inputs=np.asarray(table['driver_values'],dtype=float)
            if output.ndim!=2 or output.shape[1]!=len(axes) or inputs.shape!=(len(output),len(names)):
                raise ValueError(f'Mismatched calibration dimensions for {name}')
            if not np.isfinite(output).all() or not np.isfinite(inputs).all():
                raise ValueError('Calibration must be finite')
            if np.any(inputs<=0):
                raise ValueError('Calibrated drive parameters must be positive')
            origin=output.min(axis=0); scale=np.ptp(output,axis=0)
            if np.any(scale<=0) or len(output)<len(axes)+1:
                raise ValueError('Calibration must cover independent actual output axes')
            bounds={axis:tuple(table['bounds'][axis]) for axis in axes}
            if any(len(b)!=2 or not 0<b[0]<b[1] for b in bounds.values()):
                raise ValueError('Actual motion bounds must be positive ordered pairs')
            triangles=None
            if len(axes)==1:
                order=np.argsort(output[:,0]); output=output[order]; inputs=inputs[order]
                if np.any(np.diff(output[:,0])<=0):
                    raise ValueError('One-dimensional response must have distinct measured outputs')
                triangles=output[:,0]
            else:
                try: triangles=Delaunay((output-origin)/scale)
                except QhullError as error: raise ValueError('Calibration output axes are not independently covered') from error
            self.tables[name]=(axes,names,inputs,origin,scale,triangles,bounds)
            # Full requested box must lie within measured coverage. Never
            # extrapolate missing corners or silently clip a requested target.
            import itertools
            for corner in itertools.product(*(bounds[a] for a in axes)):
                self.solve(name,dict(zip(axes,corner)))

    @classmethod
    def load(cls,path):
        return cls(json.loads(Path(path).read_text(encoding='utf-8-sig')))

    def axes(self,name): return self.tables[name][0]
    def bounds(self,name): return self.tables[name][-1]

    def solve(self,name,targets):
        axes,names,inputs,origin,scale,triangles,bounds=self.tables[name]
        target=np.asarray([targets[a] for a in axes],dtype=float)
        if not np.isfinite(target).all(): raise ValueError('Targets must be finite')
        if any(not bounds[a][0]-1e-10<=targets[a]<=bounds[a][1]+1e-10 for a in axes):
            raise ValueError('Target outside accepted motion range')
        if len(axes)==1:
            if target[0]<triangles[0]-1e-10 or target[0]>triangles[-1]+1e-10:
                raise ValueError('Actual motion target is not covered by calibration')
            return {n:float(np.interp(target[0],triangles,inputs[:,i])) for i,n in enumerate(names)}
        point=(target-origin)/scale
        index=int(triangles.find_simplex(point,tol=1e-9))
        if index<0: raise ValueError(f'Actual motion target is not covered by calibration: {name} {targets}')
        transform=triangles.transform[index]
        first=transform[:-1]@(point-transform[-1])
        weights=np.r_[first,1-first.sum()]
        value=weights@inputs[triangles.simplices[index]]
        return dict(zip(names,map(float,value)))
