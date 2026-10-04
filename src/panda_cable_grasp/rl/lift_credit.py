"""Episode-wide lift credit, independent of grasp sessions and material switches."""
import numpy as np


class HighWaterLiftCredit:
    def __init__(self, heights, fraction):
        self.height_high_water = float(np.max(heights))
        self.fraction_high_water = float(fraction)
        self.previous_heights = np.asarray(heights, dtype=float).copy()
        self.previous_fraction = float(fraction)
        self.previous_held_index = None

    def update(self, heights, fraction, held_index, positive_scale):
        heights = np.asarray(heights, dtype=float)
        fraction = float(fraction)
        lift = cable = 0.0
        eligible = held_index is not None and positive_scale > 0.
        if eligible:
            # Only new episode-wide height can earn positive credit. A different
            # node, a release or a larger reward weight cannot reset this ceiling.
            lift = positive_scale * max(0., float(heights[held_index]) - self.height_high_water)
            cable = positive_scale * max(0., fraction - self.fraction_high_water)
            if self.previous_held_index == held_index:
                lift += min(0., float(heights[held_index] - self.previous_heights[held_index]))
            if self.previous_held_index is not None:
                cable += min(0., fraction - self.previous_fraction)
        # Unheld external motion also consumes observed progress; catching an
        # already raised cable must not turn that motion into policy credit.
        self.height_high_water = max(self.height_high_water, float(np.max(heights)))
        self.fraction_high_water = max(self.fraction_high_water, fraction)
        self.previous_heights = heights.copy()
        self.previous_fraction = fraction
        self.previous_held_index = held_index if eligible else None
        return float(lift), float(cable)
