# Lint as python3
"""Utils module for network creation game"""

from __future__ import annotations

from typing import List
from itertools import chain, combinations

import graph_tool as gt
from graph_tool.all import *
import numpy as np

# Action 0 is reserved to encode no possible action as requested by Open Spiel.
NO_POSSIBLE_ACTION = 0


class Network:

    def __init__(self, A, V, is_minimum_spanning=False):
        self._V = V
        if(A.size == 0):
            # Empty graph
            self.G = Graph(directed=False)
            self.ownership = Graph(directed=True)

            # Add vertices
            self.G.add_vertex(len(self._V))
            self.ownership.add_vertex(len(self._V))

        else:
            # Use upper triangular matrix (undirected graph) from provided adjacency matrix
            _A = np.triu(A)
            edge_list = np.nonzero(_A)
            self.G = Graph(
                np.array([edge_list[0], edge_list[1], _A[edge_list]]).T, directed=False)
            if(is_minimum_spanning):
                # If min spanning, ownership is trivially set
                self.ownership = self.G.copy()
                self.ownership.set_directed(True)
            else:
                # Otherwise, make empty directed graph
                self.ownership = Graph(directed=True)

    def num_actions(self) -> int:
        """Returns the number of possible actions.

        Equal to 2^{number of agents} + 1 (no move).
        """
        return 2**self._V + 1

    def find_shortest_path(self, s, d):
        # Iterate on undirected graph
        ug = self.ownership.copy()
        ug.set_directed(False)
        _, path_edge_list = shortest_path(ug, s, d)
        self.diameter = [len(path_edge_list) if len(
            path_edge_list) != 0 else np.inf][0]

    def add_edge_to_ownership(self, source, target):
        self.ownership.add_edge(source, target)

    def remove_edge_from_ownership(self, source, target):
        self.ownership.remove_edge(self.ownership.edge(source, target))

    def get_edges_vertex_v(self, v):
        return list(self.ownership.iter_out_neighbors(v))

    def get_edges_vertex_v_as_array(self, v):
        return np.array([self.G.vertex_index[v] for v in list(self.ownership.iter_out_neighbors(v))])

    def adj_matrix(self, graph):
        return gt.spectral.adjacency(graph)

    def list_subsets_i(self, agent_idx):

        return list(chain.from_iterable(combinations([v_ for v_ in self.G.vertices() if self.vertex_index[v_] is not self.get_vertex_from_index(agent_idx)], r)
                                        for r in range(len(list(self.G.vertices())))))

    def get_vertex_from_agent(self, agent):

        return self.G.vertex_index[agent._idx]

    def get_vertex_from_index(self, agent_idx):

        return self.G.vertex(agent_idx)

    def compute_cost_player_i(self, s_i, i):

        dij = []
        for j in np.delete(np.arange(self.network.V_), self.get_vertex_from_agent(i)):
            self.network.find_shortest_path(
                self.get_vertex_from_agent(i), j)
            dij.append(self.network.diameter)

        if(s_i is not None):
            c_i = self.alpha * len(s_i) + np.sum(dij)
        else:
            # Current cost
            c_i = self.alpha * \
                len(self.network.get_edges_vertex_v(i._id)) + np.sum(dij)

        return c_i

    def strategic_play_player_i(self, s_i, i):
        # Save current state
        i.state = self.network.get_edges_vertex_v(i)

        # Remove former edges
        for j in i.state:
            self.network.remove_edge_from_ownership(
                i, j)

        for j in s_i:
            self.play_strategy(j, i)

    def play_strategy(self, s, i):
        self.network.add_edge_to_ownership(i, s)


class Agent():

    def __init__(self, _idx):
        self._idx = _idx        # Vertex index
        self.history = []
        self.state = []
        self.cost_history = []

    @property
    def id(self) -> int:
        """Returns agent's id."""
        return self._id

    @property
    def history(self) -> List[int]:
        """Returns agent's history (past actions)."""
        return self.history

    @property
    def state(self) -> List[int]:
        """Returns agent's state."""
        return self.state
