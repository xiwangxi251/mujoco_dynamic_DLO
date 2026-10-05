"""Episode-wide reach/alignment progress across pinch eligibility changes."""
import numpy as np


class ApproachProgressCredit:
    def __init__(self, distance, alignment_potential):
        self.previous = np.array([-distance, alignment_potential], dtype=float)
        self.high_water = self.previous.copy()
        self.debit_remaining = np.zeros(2)

    def update(self, distance, alignment_potential, *, eligible):
        current = np.array([-distance, alignment_potential], dtype=float)
        reward = np.zeros(2)
        if eligible:
            positive = np.maximum(0., current-self.high_water)
            self.debit_remaining += positive
            debit = np.minimum(np.maximum(0., self.previous-current), self.debit_remaining)
            self.debit_remaining = np.maximum(0., self.debit_remaining-debit)
            reward = positive-debit
        # Physical pinch pauses shaping but never resets the observed ceilings.
        self.high_water = np.maximum(self.high_water, current)
        self.previous = current
        return tuple(float(value) for value in reward)
