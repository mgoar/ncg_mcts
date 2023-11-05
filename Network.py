import graph_tool as gt
from graph_tool.all import *
import numpy as np


class Network():

    def __init__(self, A, V_, is_minimum_spanning=False):
        self.instantiate(A, V_, is_minimum_spanning)

    def instantiate(self, A, V, is_minimum_spanning):
        self._V = V
        if(A.size == 0):
            # Empty graph
            self.G = Graph(directed=False)
            self.ownership = Graph(directed=True)
            for _ in np.arange(self._V):
                self.G.add_vertex()
                self.ownership.add_vertex()

        else:
            A_ = np.triu(A)
            edge_list = np.nonzero(A_)
            self.G = Graph(
                np.array([edge_list[0], edge_list[1], A_[edge_list]]).T, directed=False)
            if(is_minimum_spanning):
                self.ownership = self.G.copy()
                self.ownership.set_directed(True)
            else:
                self.ownership = Graph(directed=True)

    def find_shortest_path(self, s, d):
        # Iterate on undirected graph
        ug = self.ownership.copy()
        ug.set_directed(False)
        _, path_edge_list = shortest_path(ug, s, d)
        self.diameter = [len(path_edge_list) if len(
            path_edge_list) != 0 else np.inf][0]
        
    def _find_shortest_path(self, s, d):
        # Iterate on undirected graph
        ug = self.ownership.copy()
        ug.set_directed(False)
        _, path_edge_list = shortest_path(ug, s, d)
        self.diameter = [len(path_edge_list) if len(
            path_edge_list) != 0 else np.inf][0]

    def add_vertex_to_ownership(self):
        self.ownership.add_vertex()
    
    def add_edge_to_ownership(self, source, target):
        self.ownership.add_edge(source, target)

    def remove_edge_from_ownership(self, source, target):
        self.ownership.remove_edge(self.ownership.edge(source, target))

    def get_edges_vertex_v(self, v):
        return list(self.ownership.iter_out_neighbors(v))

    def update_graph(self, set, v):
        for s in set:
            self.G.add_edge(v, s)

        gt.generation.remove_parallel_edges(self.G)

    def adj_matrix(self, graph):
        return gt.spectral.adjacency(graph)
    
    def set_ownership(self, dgraph):
        self.ownership = dgraph