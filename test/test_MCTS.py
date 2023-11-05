import graph_tool as gt
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
