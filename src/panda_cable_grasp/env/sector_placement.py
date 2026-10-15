"""Translate existing cable geometry into the user-selected near-table sector B."""
import numpy as np
from .workspace_geometry import AnnularSector

STATIC_SECTOR = AnnularSector((.35, 0.), .28, .56, -30., 70.)


def sample_translation(xy, rng, table_min, table_max, fraction=.5, fallback=.4):
    xy = np.asarray(xy, dtype=float)
    count = 2048
    radius = np.sqrt(rng.uniform(.28**2, .56**2, count))
    angle = rng.uniform(np.radians(-30.), np.radians(70.), count)
    targets = np.column_stack((.35 + radius * np.cos(angle), radius * np.sin(angle)))
    anchors = xy[rng.integers(0, len(xy), count)]
    shifts = targets - anchors
    candidates = xy[None] + shifts[:, None]
    on_table = np.all((candidates >= table_min) & (candidates <= table_max), axis=(1, 2))
    coverage = STATIC_SECTOR.contains_xy(candidates).mean(axis=1)
    accepted = np.flatnonzero(on_table & (coverage >= fraction - 1e-12))
    if len(accepted):
        chosen, used = int(accepted[0]), fraction
    else:
        score = np.where(on_table, coverage, -1.)
        chosen = int(np.argmax(score))
        if score[chosen] < fallback - 1e-12:
            raise ValueError('No static sector placement satisfies the fallback node fraction')
        used = fallback
    return shifts[chosen], float(coverage[chosen]), float(used)


def sample_rotated_translation(xy, rng, table_min, table_max, fraction=.3):
    """Draw one yaw per episode, then translate it without relaxing coverage.

    Fix the yaw before rejection sampling so placement rejection does not
    preferentially select an orientation. Rotation preserves the existing shape.
    """
    xy = np.asarray(xy, dtype=float)
    yaw = float(rng.uniform(-np.pi, np.pi))
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)],
                         [np.sin(yaw), np.cos(yaw)]])
    center = xy.mean(axis=0)
    rotated = (xy-center) @ rotation.T + center
    shift, coverage, used = sample_translation(
        rotated, rng, table_min, table_max, fraction, fraction)
    return yaw, shift, coverage, used
