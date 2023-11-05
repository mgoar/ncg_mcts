import graph_tool as gt
import numpy as np
import unittest

import State
import NCG
import Network
import utils


class TestNetwork(unittest.TestCase):
    def test_find_shortest_path(self):
        # Construct min spanning tree on Petersen graph from adjacency matrix
        pet = gt.collection.petersen_graph()
        tree = gt.topology.min_spanning_tree(pet, root=pet.vertex(0))
        u = gt.GraphView(pet, efilt=tree)
        A = gt.spectral.adjacency(u).todense()

        gt.draw.graph_draw(u,
                           vertex_text=u.vertex_index, output="fig/Petersen_minspan.pdf")

        graph = Network.Network(A, A.shape[0], True)

        # Find shortest path
        graph.find_shortest_path(0, 9)
        self.assertEqual(graph.diameter, 3)


class TestNCG(unittest.TestCase):
    def test_compute_cost_player_i(self):
        A = np.array([])
        graph = Network.Network(A, 3)
        ncg = NCG.NCG(graph, .5, True)
        c_i = ncg._compute_cost_player_i(ncg.agents[0])
        self.assertEqual(c_i, np.inf)

    def test_play(self):
        # Create (n,q)-graph
        q = 4
        n = 2*q + 3
        alpha = 2*q

        ug, dg = utils._create_n_q_graph(q)

        graph = Network.Network(gt.spectral.adjacency(ug).todense(), n)
        graph.set_ownership(dg)

        # Create NCG
        ncg = NCG.NCG(graph, alpha, True)

        responses = []
        for agent in ncg.agents:
            # List all k=1 legal actions
            actions = ncg._legal_k_length_actions(agent, 1)

            # Fetch costs of actions
            for a in actions:
                is_br, _ = ncg._is_better_response(a, agent)

                if is_br:
                    responses.append(a)

        # Check is NE
        self.assertTrue(not any(responses))
