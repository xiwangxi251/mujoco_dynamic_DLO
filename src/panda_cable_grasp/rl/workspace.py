"""Workspace-conditioned goals and the explicit seven-stage PPO curriculum."""
import hashlib
import json
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from .target_node import TargetNodeTask
from ..scenarios.registry import CABLE_X075_SOURCE_NAMES, SCENARIO_FAMILIES

WORKSPACE_SCENE_FAMILIES = {
    "legacy_ppo_six": CABLE_X075_SOURCE_NAMES,
    "cable_x075_v1": SCENARIO_FAMILIES["cable_x075_v1"],
}
DEFAULT_TARGET_WORKSPACE = Path(__file__).with_name("workspaces") / "rectangle_060_084_m022_030.npz"
TARGET_WORKSPACE_XY_BOUNDS = (.60, .84, -.22, .30)
WORKSPACE_STAGES = (
    ("static_workspace_all_nodes", (1,0,0,0,0,0), 0.),
    ("complex_static_workspace", (.3,0,0,0,.7,0), 0.),
    ("component_motion_low", (.1,.4,.4,0,.1,0), 1/3),
    ("component_motion_full", (.1,.4,.4,0,.1,0), 1.),
    ("combined_motion", (.1,.1,.1,.6,.1,0), 1.),
    ("six_scenes_replay_low", (.1,.1,.1,.1,.3,.3), 1/3),
    ("six_scenes_full", (.1,.1,.1,.1,.3,.3), 1.),
)


class NoWorkspaceTarget(ValueError):
    pass


class WorkspaceMask:
    def __init__(self, path, key="interior"):
        self.path = Path(path).resolve()
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.key = key
        with np.load(self.path, allow_pickle=False) as data:
            self.xs, self.ys = data["xs"].copy(), data["ys"].copy()
            self.mask = data[key].astype(bool)
            self.xy_bounds = (data["rectangle_xy_bounds"].copy()
                              if "rectangle_xy_bounds" in data else None)
        if (self.xs.ndim != 1 or self.ys.ndim != 1 or
                len(self.xs) < 2 or len(self.ys) < 2 or
                self.mask.shape != (len(self.ys), len(self.xs))):
            raise ValueError("Workspace must contain a rectangular XY grid")
        for axis in (self.xs, self.ys):
            if (not np.all(np.isfinite(axis)) or np.any(np.diff(axis) <= 0)
                    or not np.allclose(np.diff(axis), axis[1]-axis[0])):
                raise ValueError("Workspace axes must be finite uniform increasing grids")
        if not np.any(self.mask):
            raise ValueError("Workspace mask has no accepted cells")
        if self.xy_bounds is not None:
            bounds = self.xy_bounds
            if (bounds.shape != (4,) or not np.all(np.isfinite(bounds))
                    or bounds[0] >= bounds[1] or bounds[2] >= bounds[3]
                    or not np.all(self.mask)):
                raise ValueError("Rectangle workspace requires finite bounds and a full mask")
            edges = [self.xs[0]-(self.xs[1]-self.xs[0])/2,
                     self.xs[-1]+(self.xs[1]-self.xs[0])/2,
                     self.ys[0]-(self.ys[1]-self.ys[0])/2,
                     self.ys[-1]+(self.ys[1]-self.ys[0])/2]
            if not np.allclose(bounds, edges, atol=1e-12, rtol=0):
                raise ValueError("Rectangle bounds must match the XY cell edges")

    def contains(self, positions):
        positions = np.atleast_2d(np.asarray(positions, dtype=float))
        if positions.shape[1] != 3 or not np.all(np.isfinite(positions)):
            raise ValueError("Expected finite XYZ positions")
        ix = np.rint((positions[:,0]-self.xs[0])/(self.xs[1]-self.xs[0])).astype(int)
        iy = np.rint((positions[:,1]-self.ys[0])/(self.ys[1]-self.ys[0])).astype(int)
        valid = ((ix >= 0) & (ix < len(self.xs)) & (iy >= 0) & (iy < len(self.ys))
                 & (positions[:,2] >= -.005) & (positions[:,2] <= .10))
        if self.xy_bounds is not None:
            xmin,xmax,ymin,ymax = self.xy_bounds
            # Include exact cell-edge corners despite round-to-even and float
            # roundoff. XY membership uses explicit rectangle bounds.
            valid = ((positions[:,0] >= xmin-1e-12) & (positions[:,0] <= xmax+1e-12)
                     & (positions[:,1] >= ymin-1e-12) & (positions[:,1] <= ymax+1e-12)
                     & (positions[:,2] >= -.005) & (positions[:,2] <= .10))
            ix = np.clip(ix, 0, len(self.xs)-1)
            iy = np.clip(iy, 0, len(self.ys)-1)
        result = np.zeros(len(positions), dtype=bool)
        result[valid] = self.mask[iy[valid], ix[valid]]
        return result

    def manifest(self):
        result = dict(path=str(self.path), sha256=self.sha256, mask_key=self.key,
                      accepted_cells=int(self.mask.sum()), z_range_m=[-.005, .10])
        if self.xy_bounds is not None:
            result.update(kind="rectangle", xy_bounds_m=self.xy_bounds.tolist())
        return result


class WorkspaceTargetNode(TargetNodeTask):
    """Select a material node once; workspace membership is diagnostic in v2."""
    def __init__(self, wrapper, seed, workspace):
        super().__init__(wrapper, seed)
        self.workspace = workspace
        self.start_protocol = wrapper._workspace_start_protocol
        self.ever_inside = False
        self.first_entry_seconds = None
        self.candidates = np.array([], dtype=int)
        self.first_grasp_correct = None
        self.first_grasp_index = None

    def reset(self, seed, options):
        self.candidates = (np.arange(len(self.base.cable_ids))
            if self.start_protocol == "policy_from_initial_v2" else
            np.flatnonzero(self.workspace.contains(self.base.data.xpos[self.base.cable_ids])))
        if not len(self.candidates):
            raise NoWorkspaceTarget("No material node inside the selected workspace")
        fixed = (options or {}).get("goal_node_index", self.fixed_index)
        if fixed is not None and (isinstance(fixed, bool) or not isinstance(fixed, (int, np.integer))):
            raise ValueError("goal_node_index must be an integer")
        if fixed is not None and not 0 <= fixed < len(self.base.cable_ids):
            raise ValueError("goal_node_index outside the cable")
        if fixed is not None and fixed not in self.candidates:
            raise NoWorkspaceTarget("Explicit goal is outside the selected workspace")
        if seed is not None:
            self.rng = np.random.default_rng(int(seed) ^ 0x607A19)
        goal = int(fixed) if fixed is not None else int(self.rng.choice(self.candidates))
        super().reset(seed, dict(options or {}, goal_node_index=goal))
        self.first_grasp_correct = self.first_grasp_index = None
        self.ever_inside = False
        self.first_entry_seconds = None
        self.observe_workspace_goal()

    def observe_workspace_goal(self):
        inside = bool(self.workspace.contains(
            self.base.data.xpos[self.base.cable_ids[self.index]])[0])
        if inside and not self.ever_inside:
            self.first_entry_seconds = float(self.base.data.time)
        self.ever_inside |= inside
        return inside

    def info(self):
        info = super().info()
        info.update(workspace_candidate_count=len(self.candidates),
                    workspace_start_protocol=self.start_protocol,
                    workspace_goal_ever_inside=self.ever_inside,
                    workspace_goal_first_entry_seconds=self.first_entry_seconds,
                    workspace_goal_current_inside=bool(self.workspace.contains(
                        self.base.data.xpos[self.base.cable_ids[self.index]])[0]),
                    first_grasp_index=self.first_grasp_index,
                    first_grasp_correct=self.first_grasp_correct)
        return info


def prepare_workspace_entry(base, workspace, maximum_seconds, episode_seconds):
    """Physically advance only the entry prefix; no PPO actions or rewards."""
    base.config.episode_seconds = max(float(episode_seconds), maximum_seconds + .02)
    steps = int(np.floor(maximum_seconds/(base.model.opt.timestep*base.config.frame_skip)))
    for _ in range(steps):
        if np.any(workspace.contains(base.data.xpos[base.cable_ids])):
            break
        _, _, done, truncated, _ = base.step(base.ready_ctrl.copy())
        if done or truncated or base.grasp_confirmed:
            raise NoWorkspaceTarget("Workspace entry prefix terminated or grasped the cable")
    prefix = float(base.data.time)
    base.config.episode_seconds = float(episode_seconds) + prefix
    if not np.any(workspace.contains(base.data.xpos[base.cable_ids])):
        raise NoWorkspaceTarget("No workspace target after physical entry wait")
    return prefix


class WorkspaceCurriculumCallback(BaseCallback):
    """Promote on first-correct >=60% and success >=40%, two of last three."""
    def __init__(self, scene_names, state_path, *, resume=False):
        super().__init__()
        if len(scene_names) != 6:
            raise ValueError("Workspace curriculum requires six ordered scenes")
        self.scene_names = tuple(scene_names)
        self.state_path = Path(state_path)
        self.phase_index = 0
        self.history = []
        if resume and self.state_path.exists():
            state = json.loads(self.state_path.read_text())
            if tuple(state["scene_names"]) != self.scene_names:
                raise ValueError("Workspace curriculum scene family changed")
            self.phase_index, self.history = int(state["stage"]), state["history"]
        if not 0 <= self.phase_index < len(WORKSPACE_STAGES):
            raise ValueError("Invalid saved workspace stage")

    @property
    def stage_index(self):
        return self.phase_index

    @property
    def difficulty(self):
        return WORKSPACE_STAGES[self.stage_index][2]

    def _apply(self):
        # Serialized with each PPO checkpoint, so an older checkpoint is not
        # evaluated using a later stage from the run's mutable curriculum file.
        self.model.target_evaluation_stage = self.stage_index
        self.training_env.env_method("set_training_scenario_distribution", self.scene_names,
                                     WORKSPACE_STAGES[self.stage_index][1])
        self.training_env.env_method("set_motion_difficulty", self.difficulty)
        self.logger.record("curriculum/stage", self.stage_index)

    def configure_evaluation_env(self, env):
        env.env_method("set_training_scenario_distribution", self.scene_names,
                       WORKSPACE_STAGES[self.stage_index][1])
        env.env_method("set_motion_difficulty", self.difficulty)

    def restore_phase(self, phase_index):
        if not 0 <= int(phase_index) < len(WORKSPACE_STAGES):
            raise ValueError("Invalid workspace curriculum phase")
        if self.phase_index != int(phase_index):
            self.history = []
        self.phase_index = int(phase_index)
        self._apply()

    def observe_evaluation(self, metrics):
        evaluated_stage = self.stage_index
        self.logger.record("eval/first_grasp_correct_rate", metrics["first_grasp_correct_rate"])
        passed = metrics["first_grasp_correct_rate"] >= .60 and metrics["success_rate"] >= .40
        self.history = (self.history + [bool(passed)])[-3:]
        if sum(self.history) >= 2 and self.stage_index < len(WORKSPACE_STAGES)-1:
            self.phase_index += 1
            self.history = []
            self._apply()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(dict(scene_names=self.scene_names, stage=self.stage_index,
                                               history=self.history, last_evaluation=dict(stage=evaluated_stage, **metrics))), encoding="utf-8")
        temporary.replace(self.state_path)

    def _on_training_start(self):
        self._apply()

    def _on_step(self):
        return True


def add_workspace_arguments(parser):
    parser.add_argument("--target-workspace", type=Path, help="NPZ containing xs, ys and the selected XY mask")
    parser.add_argument("--workspace-mask-key", default="interior")
    parser.add_argument("--workspace-start-protocol",
                        choices=("policy_from_initial_v2", "physical_entry_v1"),
                        default="policy_from_initial_v2",
                        help="v2: policy starts immediately, all material nodes eligible; v1: archived entry-conditioned protocol")
    parser.add_argument("--workspace-entry-seconds", type=float, default=8.,
                        help="legacy physical_entry_v1 only; v2 never performs an entry wait")
    parser.add_argument("--workspace-reset-attempts", type=int, default=50,
                        help="maximum conditioned reset attempts; use 1 to prohibit seed fallback")
    parser.add_argument("--replay-bank-dir", type=Path)
    parser.add_argument("--replay-seed-fallback", choices=("error", "resample"), default="resample")
