from graph_tool.all import *
import copy
from functools import partial
import logging
import multiprocessing
import numpy as np
from typing import List, Tuple

import Agent
import State
import utils

MAX_DEPTH = 10**4

INIT_BRANCHING = 10**2
EXPANSION_BRANCHING = 10**2
INIT_DEPTH = 2

C = 1

DISCOUNT_FACTOR = 0.8


class MCTS(object):
    def __init__(self, s0: State.State, budget=np.inf, swap_only=False):
        self.tree = Graph(directed=True)

        # For polynomial expansion
        self.k = budget

        # Greedy (swap) equilibrium
        self.swap_eq = swap_only

        # Metadata
        self.s0_prop = self.tree.new_vp("object")
        v0 = self.tree.add_vertex()
        self.s0_prop[v0] = s0

        self._set_value(s0)

        # Set to terminal if value equals number of agents (NE)
        if s0.get_mean_value == 1.0:
            s0.set_terminal()

        # List of NE
        self.ne = []

        self.initialize()

    def initialize(self):
        logging.info("Performing recursive computation of better responses. INIT_BRANCHING/INIT_DEPTH/k {}/{}/{}".format(
            INIT_BRANCHING, INIT_DEPTH, self.k))
        self.recursive_best_response(self.s0_prop[0])

    def recursive_best_response(self, start: State.State, visited=None, expansion=False):

        if visited is None:
            visited = set()

        visited.add(start.get_id)

        if start.is_terminal:
            pass
        else:
            if (len(self._find_backtrace(start)) < INIT_DEPTH) and (not expansion):
                # Sequential play of better responses
                for agent in start.NCG.agents:
                    # List all legal actions
                    if not self.swap_eq:
                        actions = start.NCG._legal_k_length_actions(
                            agent, self.k)
                    else:
                        actions = start.NCG._legal_swap_actions(agent, self.k)

                    if (len(self.tree.get_out_neighbors(start.get_id)) < INIT_BRANCHING):
                        # Fetch costs of actions
                        better_responses = self._fetch_costs(
                            agent, actions, start)

                        for br in better_responses:
                            if (len(self.tree.get_out_neighbors(start.get_id)) < INIT_BRANCHING):
                                _ = self._append_tree_node(
                                    start, agent, br[0], False, True)

            elif expansion:
                # Sequential play of better responses
                for agent in start.NCG.agents:
                    # List all legal actions
                    if not self.swap_eq:
                        actions = start.NCG._legal_k_length_actions(
                            agent, self.k)
                    else:
                        actions = start.NCG._legal_swap_actions(agent, self.k)

                    if (len(self.tree.get_out_neighbors(start.get_id)) < EXPANSION_BRANCHING):
                        # Fetch costs of actions
                        better_responses = self._fetch_costs(
                            agent, actions, start)

                        for br in better_responses:
                            if (len(self.tree.get_out_neighbors(start.get_id)) < EXPANSION_BRANCHING):
                                _ = self._append_tree_node(
                                    start, agent, br[0], False, True)

        for next in set(self.tree.get_out_neighbors(start.get_id)) - visited:
            self.recursive_best_response(self.s0_prop[next], visited)

    def selection(self, start: State.State, visited=None) -> State.State:

        logging.info("Performing selection from State Id: {}".format(
            start.get_id))

        if visited is None:
            visited = set()

        visited.add(start.get_id)

        assert len(self._find_backtrace(start)) < MAX_DEPTH

        start.incr_visits()

        # Look-ahead states
        states_id = self.tree.get_out_neighbors(start.get_id)

        states = [self.s0_prop[_id] for _id in states_id]

        if states_id.size == 0:
            if start.is_terminal:
                # Continue recursive selection
                logging.info("Selection ended in terminal state")
                for next in set(self.tree.get_out_neighbors(self.s0_prop[0].get_id)) - visited:
                    return self.selection(self.s0_prop[next], visited)
            else:
                # Expand
                return self.expansion(start)

        else:
            # Compute UCB
            ucb = self._compute_ucb(states, start)
            logging.debug("UCB: {}".format(ucb))
            return self.selection(states[np.argmax(ucb)], visited)

    def expansion(self, state: State.State):
        logging.info(
            "Performing expansion at State Id: {}".format(state.get_id))
        if (state.get_parent_id == 0) and (state.get_visits == 1):
            # Perform simulation. Skip
            # Set to terminal if value equals number of agents (NE)
            if state.get_mean_value == 1.0:
                state.set_terminal()

        else:
            # Expand random better response(s)
            self.recursive_best_response(state, None, True)

            if len(self.tree.get_out_neighbors(state.get_id)) > 0:
                child = self.tree.get_out_neighbors(state.get_id)
                mean_values = [self.s0_prop[n].get_mean_value for n in child]
                kk = np.argmax(mean_values)
                state = self.s0_prop[self.tree.get_out_neighbors(state.get_id)[kk]]

                state.incr_visits()

                # Set to terminal if value equals number of agents (NE)
                if state.get_mean_value == 1.0:
                    state.set_terminal()

            else:
                logging.error(
                    "Expansion. No expanded node found from State Id: {}".format(state.get_id))

        return state

    def simulation(self, state: State.State) -> State.State:

        if state is not None:
            logging.info("Simulation. State Id: {}".format(state.get_id))
            if not state.is_terminal:
                # Default policy
                s = self._apply_default_policy(state, "random_drawn")

                logging.info(
                    "Simulation. State mean value/Scores: {}/{}".format(np.round(s.get_mean_value,3), s.get_scores))

                return s
            else:
                return state
        else:
            logging.error("Simulation. No available states. Exiting")
            pass

    def _apply_default_policy(self, initial_state: State.State, heuristic="random_drawn", discount=True) -> State.State:
        if heuristic == "random_drawn":
            # Playout of individuals randomly drawn
            n = len(initial_state.NCG.agents)
            order = np.random.choice(np.arange(n), size=n**2, replace=True)

            for i in order:
                a = initial_state.NCG.agents[i]

                # List all legal actions
                if not self.swap_eq:
                    actions = initial_state.NCG._legal_k_length_actions(
                        a, self.k)
                else:
                    actions = initial_state.NCG._legal_swap_actions(a, self.k)

                # Fetch better responses
                better_responses = self._fetch_costs(
                    a, actions, initial_state)

                if len(better_responses) > 0:
                    jj = np.random.choice(np.arange(len(better_responses)))
                    v = self._append_tree_node(
                        initial_state, a, better_responses[jj][0], False, True)
                    initial_state = self.s0_prop[v]

                    # Save NE
                    if initial_state.is_terminal:
                        self.ne.append(initial_state)
                        break
        
        if discount:
            reward = initial_state.get_mean_value
            if not utils._is_tree(initial_state):
                logging.info("Simulation. Non-tree NE found. State Id: {}".format(initial_state.get_id))
            else:
                initial_state.set_mean_value(pow(self._get_discount_factor(), len(self._find_backtrace(initial_state))) * reward)

        return initial_state

    def backpropagation(self, initial_state: State.State, end_state: State.State, keep_branch=False):

        # Find backtrace from simulated state
        backtrace = self._find_backtrace(end_state)

        if end_state.is_terminal and keep_branch:
            # Update visits
            for state in backtrace:
                if state.get_id > initial_state.get_id:
                    state.incr_visits()
                state.update_mean_value(end_state.get_mean_value)
        else:
            # Remove from terminal_state to child of initial_state
            for this_state in backtrace[0:backtrace.index(initial_state)]:
                if this_state.get_id != initial_state.get_id:
                    self.tree.remove_vertex(this_state.get_id)

            # Update mean value
            for state in self._find_backtrace(initial_state):
                state.update_mean_value(end_state.get_mean_value)

    def _append_tree_node(self, s: State.State, agent: Agent.Agent, action: Tuple, is_terminal=False, set_value=False) -> graph_tool.libgraph_tool_core.Vertex:

        s.NCG._change_strategy(action, agent)

        v = self.tree.add_vertex()

        ncg = copy.deepcopy(s.NCG)
        child = State.State(len(ncg.agents),
                            self.tree.vertex_index[v], ncg)
        child.set_parent_id(s.get_id)

        # Update scores
        self._update_scores(child)

        if (set_value):
            self._set_value(child)

        if is_terminal:
            child.is_terminal = True

        self.s0_prop[v] = child

        # Set to terminal if value equals number of agents (NE)
        if child.get_mean_value == 1.0:
            child.set_terminal()

        self.tree.add_edge(s.get_id, v)

        s.NCG._undo_change_strategy(agent)

        return v

    def _remove_tree_node(self, v):
        self.tree.remove_vertex(v)

    def _set_value(self, node: State.State):
        value = 0
        for a in node.NCG.agents:
            # List all legal actions
            if not self.swap_eq:
                actions = node.NCG._legal_k_length_actions(a, self.k)
            else:
                actions = node.NCG._legal_swap_actions(a, self.k)

            if not self._exists_better_response(a, actions, node):
                value += 1

        node.set_mean_value(value/node.NCG.n)

    def _update_scores(self, state):
        state.NCG._set_costs()
        val = [a.cost for a in state.NCG.agents]
        state.set_scores(val)

    def _compute_ucb(self, child, parent) -> np.ndarray:
        ucb = np.zeros(len(child))
        for j, this_state in enumerate(child):
            try:
                ucb[j] = this_state.get_mean_value + \
                    C * np.sqrt(
                        2*np.log(parent.get_visits/this_state.get_visits))
            except ZeroDivisionError:
                ucb[j] = np.inf

        return ucb

    def _get_states_at_depth(self, d: int) -> List[State.State]:
        return [self.s0_prop[state] for state in self.tree.get_vertices() if self.s0_prop[state].get_parent_id == d-1]

    def _find_backtrace(self, state: State.State) -> List[State.State]:

        # Make undirected
        ut = copy.deepcopy(self.tree)
        ut.set_directed(False)

        v_, _ = shortest_path(ut, ut.vertex(state.get_id), 0)

        return [state] + [self.s0_prop[v] for v in v_[1:]]

    def _fetch_costs(self, agent: Agent.Agent, actions: List[Tuple], s: State.State, stop=False) -> List[Tuple[Tuple, float]]:
        results = []
        with multiprocessing.Pool() as pool:
            for result in pool.map(partial(s.NCG._is_better_response, agent=agent), actions):
                results.append(result)
                if result[0] and stop:
                    break

        return [(action, cost) for ((is_better_response, cost), action) in zip(results, actions) if is_better_response]

    def _exists_better_response(self, agent: Agent.Agent, actions: List[Tuple], s: State.State) -> bool:
        with multiprocessing.Pool() as pool:
            for result in pool.map(partial(s.NCG._is_better_response, agent=agent), actions):
                if result[0]:
                    return True
        return False

    def _recursive_dfs_mean_value(self, s: State.State):

        child = self.tree.get_out_neighbors(s.get_id)

        if len(child) > 0:
            mean_vals = [self.s0_prop[ch].get_mean_value for ch in child]
            return self._return_max_child(self.s0_prop[child[np.argmax(mean_vals)]])
        else:
            return s

    def _return_max_child(self, s) -> State.State:

        return self._recursive_dfs_mean_value(s)
    
    def _get_discount_factor(self) -> float:
        return DISCOUNT_FACTOR
