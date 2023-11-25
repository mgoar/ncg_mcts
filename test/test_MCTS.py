import graph_tool as gt
import networkx as nx
import numpy as np
import unittest

import MCTS
import State
import NCG
import Network
import Agent
import utils


class TestMCTS(unittest.TestCase):
    def test_initialization_n_q(self):
        # (n,q)-graph
        q = 4
        n = 2*q + 3
        alpha = 2*q

        ug, dg = utils._create_n_q_graph(q)

        graph = Network.Network(gt.spectral.adjacency(ug).todense(), n)
        graph.set_ownership(dg)

        # Create NCG
        ncg = NCG.NCG(graph, alpha, True)

        # Create initial state
        state_0 = State.State(n, 0, ncg)

        # Update scores
        val = np.zeros(n)
        for ii in np.arange(n):
            agent = Agent.Agent(ii)
            val[ii] = ncg._compute_cost_player_i(agent)

        state_0.set_scores(val)

        gt.draw.graph_draw(
            ncg.network.G, vertex_text=ncg.network.G.vertex_index, output="fig/initial_n_q_graph.pdf")
        gt.draw.graph_draw(
            ncg.network.ownership, vertex_text=ncg.network.ownership.vertex_index, output="fig/initial_n_q_graph_own.pdf")

        # Create instance of MCTS
        mcts = MCTS.MCTS(state_0, np.floor(n/2).astype(int))

        # Check is NE
        self.assertEqual(len(mcts.tree.get_vertices()), 1)

    def test_homs_messegue(self):
        # n = 20 (single-arm star rooted at each node)
        dict_own = {0: [1, 5], 1: [2, 6], 4: [0], 5: [7, 8], 2: [3],
                    3: [4, 7], 6: [9], 7: [6], 9: [4, 8], 8: [2]}
        H_temp = nx.DiGraph()
        for key, val in dict_own.items():
            for dst in val:
                H_temp.add_edge(key, dst)

        n = 20
        alpha = 2*n/5

        total_n = 10
        for x in np.arange(total_n):
            nx.add_star(H_temp, np.hstack((x, total_n+x)))

        dg = nx.adjacency_matrix(H_temp)
        ug = nx.adjacency_matrix(H_temp.to_undirected())

        graph = Network.Network(ug.todense(), n)
        graph.set_ownership(gt.Graph(np.array(np.nonzero(dg)).T))

        # Create NCG
        ncg = NCG.NCG(graph, alpha, True)

        # Create initial state
        state_0 = State.State(n, 0, ncg)

        # Update scores
        val = np.zeros(n)
        for ii in np.arange(n):
            agent = Agent.Agent(ii)
            val[ii] = ncg._compute_cost_player_i(agent)

        state_0.set_scores(val)

        gt.draw.graph_draw(
            ncg.network.G, vertex_text=ncg.network.G.vertex_index, output="fig/initial_pet_graph.pdf")
        gt.draw.graph_draw(
            ncg.network.ownership, vertex_text=ncg.network.ownership.vertex_index, output="fig/initial_pet_graph_own.pdf")

        # Create instance of MCTS
        # Assuming Greedy equilibrium
        k = 5
        mcts = MCTS.MCTS(state_0, k)

        # Check is NE
        self.assertEqual(len(mcts.tree.get_vertices()), 1)
