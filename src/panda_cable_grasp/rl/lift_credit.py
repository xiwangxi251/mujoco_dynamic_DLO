"""Episode-wide lift credit, independent of grasp sessions and material switches."""
import numpy as np


class HighWaterLiftCredit:
    def __init__(self, heights, fraction, *, bounded_debits=False, release_debits=False,
                 first_holding_baseline=False):
        self.height_high_water = float(np.max(heights))
        self.fraction_high_water = float(fraction)
        self.previous_heights = np.asarray(heights, dtype=float).copy()
        self.previous_fraction = float(fraction)
        self.previous_held_index = None
        self.bounded_debits = bool(bounded_debits)
        self.height_debit_remaining = 0.
        self.fraction_debit_remaining = 0.
        if release_debits and not bounded_debits:
            raise ValueError('Release debits require bounded reward credit')
        self.release_debits = bool(release_debits)
        # Spawn height is not earned lift. Latch once, using the physical
        # node's previous control-step height so an already raised cable is
        # not paid at acquisition. Never rebase after a release or node switch.
        self.first_holding_baseline = bool(first_holding_baseline)
        self.height_baseline_locked = not self.first_holding_baseline
        # Paid intervals belong to the physical material node, not the next
        # grasp session. Store the original discount so only paid credit returns.
        self.height_paid_intervals = [[] for _ in heights]
        self.fraction_paid_intervals = []

    @staticmethod
    def _pay(intervals, lower, upper, scale):
        if upper <= lower or scale <= 0.:
            return 0.
        if intervals and abs(intervals[-1][1] - lower) < 1e-12 and intervals[-1][2] == scale:
            intervals[-1][1] = upper
        else:
            intervals.append([lower, upper, scale])
        return scale * (upper - lower)

    @staticmethod
    def _debit(intervals, value):
        debit = 0.
        retained = []
        for lower, upper, scale in intervals:
            remaining_upper = min(upper, max(lower, value))
            debit += scale * (upper - remaining_upper)
            if remaining_upper > lower:
                retained.append([lower, remaining_upper, scale])
        intervals[:] = retained
        return debit

    def _update_release_debits(self, heights, fraction, held_index, positive_scale):
        lift = cable = 0.
        if held_index is not None and positive_scale > 0.:
            lift = self._pay(self.height_paid_intervals[held_index],
                             self.height_high_water, float(heights[held_index]), positive_scale)
            cable = self._pay(self.fraction_paid_intervals,
                              self.fraction_high_water, fraction, positive_scale)
        # Negative progress follows every previously credited node even after
        # opening, contact loss, an ineligible grasp or a material-node switch.
        height_drop = sum(self._debit(intervals, float(height))
                          for intervals, height in zip(self.height_paid_intervals, heights))
        fraction_drop = self._debit(self.fraction_paid_intervals, fraction)
        self.height_debit_remaining = max(0., self.height_debit_remaining + lift - height_drop)
        self.fraction_debit_remaining = max(0., self.fraction_debit_remaining + cable - fraction_drop)
        return lift - height_drop, cable - fraction_drop

    def update(self, heights, fraction, held_index, positive_scale):
        heights = np.asarray(heights, dtype=float)
        fraction = float(fraction)
        lift = cable = 0.0
        eligible = held_index is not None and positive_scale > 0.
        if eligible and not self.height_baseline_locked:
            self.height_high_water = float(self.previous_heights[held_index])
            self.height_baseline_locked = True
        if self.release_debits:
            lift, cable = self._update_release_debits(heights, fraction, held_index, positive_scale)
        elif eligible:
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
