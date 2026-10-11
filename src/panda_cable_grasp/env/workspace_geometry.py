"""Measured workspace geometry shared by diagnostics and video annotations."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import numpy as np

BUILTIN_WORKSPACE_DIR = Path(__file__).resolve().parents[1] / 'rl/workspaces'
DEFAULT_TARGET_WORKSPACE = BUILTIN_WORKSPACE_DIR / 'near_table_sector_inner15_5_20261009.npz'
LEGACY_RECTANGLE_WORKSPACE = BUILTIN_WORKSPACE_DIR / 'rectangle_060_084_m022_030.npz'


@dataclass(frozen=True)
class AnnularSector:
    center: tuple[float, float]
    inner_radius: float
    outer_radius: float
    angle_min_deg: float
    angle_max_deg: float

    def __post_init__(self):
        values=(*self.center,self.inner_radius,self.outer_radius,self.angle_min_deg,self.angle_max_deg)
        if (len(self.center)!=2 or not np.all(np.isfinite(values))
                or not 0 <= self.inner_radius < self.outer_radius
                or not -180 <= self.angle_min_deg < self.angle_max_deg <= 180):
            raise ValueError('Invalid annular workspace sector')

    @classmethod
    def from_array(cls, values):
        values=np.asarray(values,dtype=float)
        if values.shape!=(6,):
            raise ValueError('Sector requires center XY, radii and angle bounds')
        return cls(tuple(values[:2]),*map(float,values[2:]))

    def contains_xy(self, positions):
        delta=np.asarray(positions,dtype=float)-np.asarray(self.center)
        radius=np.linalg.norm(delta,axis=-1)
        angle=np.degrees(np.arctan2(delta[...,1],delta[...,0]))
        return ((radius>=self.inner_radius-1e-12)&(radius<=self.outer_radius+1e-12)
                &(angle>=self.angle_min_deg-1e-10)&(angle<=self.angle_max_deg+1e-10))

    def boundary(self, z=.007, arc_step_deg=2.):
        angles=np.radians(np.linspace(self.angle_min_deg,self.angle_max_deg,
            int(np.ceil((self.angle_max_deg-self.angle_min_deg)/arc_step_deg))+1))
        unit=np.column_stack((np.cos(angles),np.sin(angles)))
        xy=np.concatenate((self.outer_radius*unit,self.inner_radius*unit[::-1]))+np.asarray(self.center)
        return np.column_stack((xy,np.full(len(xy),z)))

    def manifest(self):
        return dict(kind='annular_sector',center_xy_m=list(self.center),
                    radius_range_m=[self.inner_radius,self.outer_radius],
                    angle_range_deg=[self.angle_min_deg,self.angle_max_deg],
                    angle_zero='world +X, positive toward +Y')


@lru_cache(maxsize=1)
def default_workspace_sector():
    with np.load(DEFAULT_TARGET_WORKSPACE,allow_pickle=False) as data:
        return AnnularSector.from_array(data['annular_sector_xy'])
