"""Episode-wide lift credit, independent of grasp sessions and material switches."""
import numpy as np


class HighWaterLiftCredit:
    def __init__(self, heights, fraction, *, bounded_debits=False):
        self.height_high_water = float(np.max(heights))
        self.fraction_high_water = float(fraction)
        self.previous_heights = np.asarray(heights, dtype=float).copy()
        self.previous_fraction = float(fraction)
        self.previous_held_index = None
        self.bounded_debits = bool(bounded_debits)
        self.height_debit_remaining = 0.
        self.fraction_debit_remaining = 0.

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
            height_drop = fraction_drop = 0.
            if self.previous_held_index == held_index:
                height_drop = max(0., float(self.previous_heights[held_index] - heights[held_index]))
            if self.previous_held_index is not None:
                fraction_drop = max(0., self.previous_fraction - fraction)
            if self.bounded_debits:
                # A credited increment can be debited only once, including
                # across release, regrasp, node switches and reward-scale changes.
                # Only newly paid positive progress replenishes this budget.
                self.height_debit_remaining += lift
                self.fraction_debit_remaining += cable
                height_drop = min(height_drop, self.height_debit_remaining)
                fraction_drop = min(fraction_drop, self.fraction_debit_remaining)
                self.height_debit_remaining = max(0., self.height_debit_remaining-height_drop)
                self.fraction_debit_remaining = max(0., self.fraction_debit_remaining-fraction_drop)
            lift -= height_drop
            cable -= fraction_drop
        # Unheld external motion also consumes observed progress; catching an
        # already raised cable must not turn that motion into policy credit.
        self.height_high_water = max(self.height_high_water, float(np.max(heights)))
        self.fraction_high_water = max(self.fraction_high_water, fraction)
        self.previous_heights = heights.copy()
        self.previous_fraction = fraction
        self.previous_held_index = held_index if eligible else None
        return float(lift), float(cable)
