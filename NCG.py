from graph_tool.all import *
from itertools import chain, combinations
import numpy as np
from typing import List, Tuple, Union

import Agent


class NCG():

    def __init__(self, G, alpha, is_SumGame=True):
        self.network = G
        self.alpha = alpha

        self.is_SumGame = is_SumGame
        self.agents = [Agent.Agent(this_id)
                       for this_id in self.network.G.get_vertices()]

        self.n = len(self.agents)

        self._set_strategies()
        self._set_costs()

    def _set_strategies(self):
        for agent in self.agents:
            agent.strategy = tuple(
                self.network.ownership.get_out_neighbors(agent._id))

    def _set_costs(self):
        for agent in self.agents:
            agent.cost = self._compute_cost_player_i(agent)

    def _play_strategy(self, s, i):
        self.network.add_edge_to_ownership(i._id, s)

    def _append_history(self, s_i, i):
        i.history.append(s_i)
        i.cost_history.append(i.cost)

    def _pop_history(self, i):
        return i.history.pop()

    def append_cost(self, c_i, i):
        i.cost_history.append(c_i)

    def pop_cost(self, i):
        return i.cost_history.pop()

    def set_state(self, s, i):
        i.strategy = tuple(s)

    def get_vertex_from_agent(self, agent):

        return self.network.G.vertex_index[agent._id]

    def set_ownership(self, A, S):
        for _ in np.arange(len(A)):
            self.network.add_vertex_to_ownership()
        for actions in S:
            self.network.add_edge_to_ownership(actions[0], actions[1])

    def list_subsets_i(self, agent):

        return list(chain.from_iterable(combinations([v_ for v_ in self.network.G.vertices() if self.network.G.vertex_index[v_] is not self.get_vertex_from_agent(agent)], r)
                                        for r in range(len(list(self.network.G.vertices())))))

    def _change_strategy(self, s: List, i: Agent.Agent):
        self._append_history(
            tuple(self.network.ownership.get_out_neighbors(i._id)), i)
        # Remove former edges
        for j in i.strategy:
            self.network.remove_edge_from_ownership(
                i._id, j)
        for j in s:
            self._play_strategy(j, i)

        i.strategy = s

        self._set_costs()

    def _undo_change_strategy(self, i):
        # Remove former edges
        for j in i.strategy:
            self.network.remove_edge_from_ownership(
                i._id, j)
        for j in i.history[-1]:
            self._play_strategy(j, i)

        i.strategy = i.history[-1]

        _ = self._pop_history(i)
        _ = self.pop_cost(i)

        self._set_costs()

    def _compute_cost_player_i(self, agent):

        if(self.is_SumGame):
            dij = []
            for j in np.delete(np.arange(self.n), self.get_vertex_from_agent(agent)):
                self.network._find_shortest_path(
                    self.get_vertex_from_agent(agent), j)
                dij.append(self.network.diameter)

            c_i = self.alpha * len(agent.strategy) + np.sum(dij)

        return c_i

    def _is_better_response(self, s_i: Tuple[Tuple], agent: Agent.Agent) -> Union[bool, float]:
        self._change_strategy(s_i, agent)
        cost = agent.cost
        self._undo_change_strategy(agent)
        return agent.cost > cost, cost

    def _legal_actions(self, agent):
        return list(chain.from_iterable(combinations(list(np.delete(np.arange(self.n), agent._id)), r) for r in range(len(list(np.arange(self.n)))+1)))

    def _legal_k_length_actions(self, agent, k):
        return [action for action in self._legal_actions(agent) if len(action) <= k and action != agent.strategy]

    def _legal_swap_actions(self, agent, k):
        return [action for action in self._legal_actions(agent) if len(action) == k and action != agent.strategy]
