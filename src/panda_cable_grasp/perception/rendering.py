"""Observation-only gripper-visibility intervention helpers.

Gripper hiding is implemented as a *runtime* ``model.geom_group`` toggle
around ``renderer.update_scene``: the gripper-mesh geoms are moved to a
render group that MuJoCo's default ``mjvOption.geomgroup`` keeps disabled
(groups 3-5), the scene is built, and the original groups are restored.

This keeps ``nero.xml`` untouched — every other render path in the codebase
(dataset collection, evaluation videos, debug gifs) keeps the stock model
and is unaffected.  ``geom_group`` is a visualization-only attribute:
contacts and dynamics read ``contype``/``conaffinity``, so physics state
(``qpos``/``qvel``/``data.contact``) is bitwise unchanged.
"""

from __future__ import annotations

import mujoco
import numpy as np


# Render group the gripper geoms are parked in while hidden.  MuJoCo's
# default mjvOption.geomgroup enables only groups 0-2, so anything parked
# here is invisible under the stock option used by every renderer.
GRIPPER_RENDER_GROUP = 4

GRIPPER_MESH_NAMES = frozenset(
    {"gripper_flange", "gripper_base", "gripper_link1", "gripper_link2"}
)


def gripper_geom_ids(model: mujoco.MjModel) -> list[int]:
    """Return geom ids whose mesh belongs to the gripper visual set."""
    ids: list[int] = []
    for geom_id in range(model.ngeom):
        mesh_id = int(model.geom_dataid[geom_id])
        if mesh_id < 0:
            continue
        mesh_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_MESH, mesh_id
        )
        if mesh_name in GRIPPER_MESH_NAMES:
            ids.append(geom_id)
    return ids


class GripperVisibilityToggle:
    """Context manager hiding gripper geoms inside a render capture.

    Usage::

        toggle = GripperVisibilityToggle(model)
        with toggle.hidden():
            renderer.update_scene(data, camera=cam)  # default scene option
            image = renderer.render()
    """

    def __init__(self, model: mujoco.MjModel) -> None:
        self.geom_ids = np.asarray(gripper_geom_ids(model), dtype=np.int64)
        if self.geom_ids.size == 0:
            raise RuntimeError("no gripper mesh geoms found in this model")
        self._original_groups = model.geom_group[self.geom_ids].copy()
        self._model = model

    class _Hidden:
        def __init__(self, outer: "GripperVisibilityToggle") -> None:
            self._outer = outer

        def __enter__(self) -> None:
            self._outer._model.geom_group[self._outer.geom_ids] = (
                GRIPPER_RENDER_GROUP
            )

        def __exit__(self, *exc_info: object) -> None:
            self._outer._model.geom_group[self._outer.geom_ids] = (
                self._outer._original_groups
            )

    def hidden(self) -> "GripperVisibilityToggle._Hidden":
        return GripperVisibilityToggle._Hidden(self)
