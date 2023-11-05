from graph_tool.all import *
import numpy as np


class State():

    def __init__(self, n, id, ncg):
        self.NCG = ncg

        self._scores = np.zeros(n)
        self.parent_id = -1
        self._id = id
        self.mean_value = 0
        self.visits = 0
        self.is_terminal = False

    @property
    def get_state(self) -> Graph:
        """Returns state (directed graph)."""
        return self.NCG.network.ownership

    def set_state(self, ncg):
        self.NCG = ncg

    @property
    def get_scores(self) -> np.ndarray:
        """Returns scores (n-tuple indexed by agent)."""
        return self._scores

    def set_scores(self, val):
        """Sets scores (n-tuple indexed by agent)."""
        self._scores = val

    def set_mean_value(self, val):
        """Sets mean value."""
        self.mean_value = val

    @property
    def get_mean_value(self) -> float:
        """Returns mean value."""
        return self.mean_value

    def update_mean_value(self, val):
        self.mean_value = np.max([val, self.mean_value])

    @property
    def get_parent_id(self) -> int:
        """Returns _id."""
        return self.parent_id

    def set_parent_id(self, val):
        self.parent_id = val

    @property
    def get_id(self) -> int:
        """Returns _id."""
        return self._id

    @property
    def get_visits(self) -> int:
        """Returns visits."""
        return self.visits

    def set_visits(self, val):
        self.visits = val

    def incr_visits(self):
        self.visits += 1

    def set_terminal(self):
        self.is_terminal = True
