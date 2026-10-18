"""Measured polar workspaces constrain requested TCP motion before IK.

Legacy maps use measured cells; optional regular maps use selected layer sectors
with conservative transfer bands. Neither certifies unsampled poses.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

DEFAULT_ACTION_WORKSPACE = Path(__file__).with_name('workspaces') / 'workspace3d_20261010.json'
REGULAR_ACTION_WORKSPACE = Path(__file__).with_name('workspaces') / 'workspace3d_regular_sectors_20261014.json'


def load_action_workspace(path, expected_sha256):
    """Resolve a saved map by content hash, including relocated bundled maps."""
    candidates = dict.fromkeys((Path(path), DEFAULT_ACTION_WORKSPACE, REGULAR_ACTION_WORKSPACE))
    for candidate in candidates:
        if candidate.is_file() and hashlib.sha256(candidate.read_bytes()).hexdigest() == expected_sha256:
            return ActionWorkspace(candidate)
    raise ValueError('Action workspace hash mismatch: no matching supplied or bundled map')


class ActionWorkspace:
    def __init__(self, path):
        self.path = Path(path)
        raw = self.path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.metadata = json.loads(raw)
        version = self.metadata['version']
        if version not in {'measured_polar_cells_v1', 'measured_layer_sectors_v1'}:
            raise ValueError('Unsupported action workspace')
        self.center = np.asarray(self.metadata['center_xy_m'], dtype=float)
        self.cells = np.asarray(self.metadata['cells_r_theta_z'], dtype=float)
        if self.cells.ndim != 2 or self.cells.shape[1] != 6 or not len(self.cells):
            raise ValueError('Workspace must contain polar cells')
        if (not np.isfinite(self.cells).all() or
                np.any(self.cells[:, ::2] >= self.cells[:, 1::2])):
            raise ValueError('Invalid workspace cell bounds')
        self.layers = None
        if version == 'measured_layer_sectors_v1':
            self.layers = np.asarray([
                [row['height_m'], row['inner_radius_m'], row['outer_radius_m'],
                 row['angle_min_deg'], row['angle_max_deg']]
                for row in self.metadata['layer_sectors']], dtype=float)
            if (len(self.layers) < 2 or not np.isfinite(self.layers).all() or
                    np.any(np.diff(self.layers[:, 0]) <= 0) or
                    np.any(self.layers[:, 1] >= self.layers[:, 2]) or
                    np.any(self.layers[:, 3] >= self.layers[:, 4])):
                raise ValueError('Invalid workspace layer sectors')
        self.radii = np.unique(self.cells[:, :2])
        self.angles = np.radians(np.unique(self.cells[:, 2:4]))
        self.heights = np.unique(self.cells[:, 4:])

    def contains(self, point):
        xy = np.asarray(point)[:2] - self.center
        p = np.array([np.linalg.norm(xy), np.degrees(np.arctan2(xy[1], xy[0])), point[2]])
        if self.layers is not None:
            exact = self.layers[np.abs(self.layers[:, 0] - point[2]) <= 1e-10]
            if len(exact):
                layer = exact[0]
                return bool(layer[1] - 1e-10 <= p[0] <= layer[2] + 1e-10 and
                            layer[3] - 1e-10 <= p[1] <= layer[4] + 1e-10)
        return bool(np.any(np.all((p >= self.cells[:, ::2] - 1e-10) &
                                  (p <= self.cells[:, 1::2] + 1e-10), axis=1)))

    def nearest(self, point):
        point = np.asarray(point, dtype=float)
        xy = point[:2] - self.center
        theta = np.degrees(np.arctan2(xy[1], xy[0]))
        # Closest point on each annular wedge, including its curved boundaries.
        cells = self.cells
        if self.layers is not None:
            # Measured planes retain their full sector; transfer bands are intersections.
            planes = np.column_stack((self.layers[:, 1:5], self.layers[:, 0], self.layers[:, 0]))
            cells = np.concatenate((cells, planes))
        a = np.radians(np.clip(theta, cells[:, 2], cells[:, 3]))
        direction = np.column_stack((np.cos(a), np.sin(a)))
        r = np.clip(direction @ xy, cells[:, 0], cells[:, 1])
        candidates = np.column_stack((self.center + r[:, None] * direction,
                                      np.clip(point[2], cells[:, 4], cells[:, 5])))
        return candidates[np.argmin(np.sum((candidates - point) ** 2, axis=1))]

    def constrain(self, position, delta):
        """Limit the requested displacement; outside recovery keeps its length cap.

        Never teleport to the map. Physical tracking and final safety remain the
        controller's responsibility. Return a request, not a policy action.
        """
        position = np.asarray(position, dtype=float)
        delta = np.asarray(delta, dtype=float)
        budget = np.linalg.norm(delta)
        if budget == 0:
            return delta.copy()
        target = position + delta
        if not self.contains(target):
            target = self.nearest(target)
        executed = target - position
        length = np.linalg.norm(executed)
        if length > budget:
            executed *= budget / length
        if not self.contains(position):
            # Entry is gradual at the original action limit, with no hidden reset.
            return executed
        # Split at every polar cell boundary and check each interval. This stops
        # motion at its first exit rather than bridging a failed/unknown hole.
        xy = position[:2] - self.center
        v = executed[:2]
        cuts = [0., 1.]
        qa = float(v @ v)
        if qa > 1e-20:
            qb = 2 * float(xy @ v)
            qc = float(xy @ xy) - self.radii ** 2
            disc = qb * qb - 4 * qa * qc
            valid = disc >= 0
            for sign in (-1, 1):
                cuts.extend(((-qb + sign * np.sqrt(disc[valid])) / (2 * qa)).tolist())
        directions = np.column_stack((np.cos(self.angles), np.sin(self.angles)))
        denom = v[0] * directions[:, 1] - v[1] * directions[:, 0]
        numer = xy[0] * directions[:, 1] - xy[1] * directions[:, 0]
        valid = np.abs(denom) > 1e-20
        cuts.extend((-numer[valid] / denom[valid]).tolist())
        if abs(executed[2]) > 1e-20:
            cuts.extend(((self.heights - position[2]) / executed[2]).tolist())
        cuts = sorted(set(t for t in cuts if 0 <= t <= 1))
        for start, end in zip(cuts[:-1], cuts[1:]):
            if not self.contains(position + executed * ((start + end) / 2)):
                return executed * max(0., start - 1e-9)
        return executed
