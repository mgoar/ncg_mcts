import numpy as np

class Agent():
    history: list[float]
    cost_history: list[set]
    strategy: set

    def __init__(self, _id):
        self._id = _id
        self.history = []
        self.cost_history = []
        self.strategy = tuple()
        self.cost = np.inf

    def append_cost(self, c_i):
        self.cost_history.append(c_i)