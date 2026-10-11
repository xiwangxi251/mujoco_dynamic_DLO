"""Reversible lift credit belonging to one continuous holding session."""
import numpy as np


class SessionLiftCredit:
    def __init__(self, heights, ceiling):
        self.previous_heights = np.asarray(heights, dtype=float).copy()
        self.ceiling = float(ceiling)
        self.held_index = None
        self.balance = 0.0
        self.sessions = 0

    def update(self, heights, held_index):
        heights = np.asarray(heights, dtype=float)
        change = 0.0
        # A material switch settles the old account before opening another;
        # a different node's absolute height cannot manufacture positive lift.
        if self.held_index is not None and held_index != self.held_index:
            change -= self.balance
            self.balance = 0.0
        if held_index is not None:
            if held_index != self.held_index:
                self.sessions += 1
            delta = min(float(heights[held_index]), self.ceiling) - min(
                float(self.previous_heights[held_index]), self.ceiling)
            next_balance = max(0.0, self.balance + delta)
            change += next_balance - self.balance
            self.balance = next_balance
        self.held_index = held_index
        self.previous_heights = heights.copy()
        return float(change)
