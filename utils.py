from graph_tool.all import *
import graph_tool as gt
import networkx as nx
import numpy as np
import copy


def _create_petersen():

    return nx.adjacency_matrix(nx.petersen_graph())


def _create_extended_petersen(order, diam):

    pet = nx.petersen_graph()

    # Subgraph last layer
    bfs = _create_bfs_petersen()

    x = np.round((order-len(pet.nodes()))/(diam-1)).astype(int)

    layer = pet.subgraph(bfs[2][0:x])

    P_n = nx.path_graph(diam-1)

    keys = np.arange(diam-1)
    vals = len(pet.nodes())+keys
    dict_labels = dict(zip(keys, vals))

    nx.relabel_nodes(P_n, dict_labels, copy=False)

    g = nx.Graph(layer)
    nx.add_path(g, P_n.nodes())
    extended = nx.compose(pet, g)

    return nx.adjacency_matrix(extended)


def _create_heawood():

    return nx.adjacency_matrix(nx.heawood_graph())


def _create_mcgee():

    cicle = nx.cycle_graph(24)

    edges_dict = {0: (0, 12),
                  1: (1, 8),
                  2: (2, 19),
                  3: (3, 15),
                  4: (4, 11),
                  5: (5, 22),
                  6: (6, 18),
                  7: (7, 14),
                  8: (9, 21),
                  9: (10, 17),
                  10: (13, 20),
                  11: (16, 23)}

    cicle.add_edges_from(edges_dict.values())

    return nx.adjacency_matrix(cicle)


def _create_ring(order: int):

    cicle = nx.cycle_graph((order/2).astype(int))
    nx.add_cycle(cicle, list(np.arange((order/2).astype(int)-1, order)))

    return nx.adjacency_matrix(cicle)


def _create_bfs_petersen():

    pet = nx.petersen_graph()

    return dict(enumerate(nx.bfs_layers(pet, [0])))


def _find_cut_vertices(g):
    _, art, _ = gt.topology.label_biconnected_components(g)
    return art.a


def _create_n_q_graph(q):
    # Manually create (n,q)-graph
    q = q
    n = 2*q + 3

    ug = gt.Graph(directed=False)

    for _ in np.arange(n):
        ug.add_vertex()

    # Left side, L owner-vertex index: q-1
    for l in np.arange(0, q-1):
        ug.add_edge(q-1, l)

    # Right side, R owner-vertex index: q
    for r in np.arange(0, q-1):
        ug.add_edge(q, q+r+1)

    # Bridge vertex index: 2q
    ug.add_edge(2*q, q-1)
    ug.add_edge(2*q, q)

    # Alternative vertices to bridge vertex
    ug.add_edge(2*q+1, q-1)
    ug.add_edge(2*q+1, q)

    ug.add_edge(2*q+2, q-1)
    ug.add_edge(2*q+2, q)

    dg = gt.Graph(directed=True)

    # Add vertices to ownership graph
    for _ in np.arange(n):
        dg.add_vertex()

    # Ownership relationships - L
    for l in np.arange(q-1):
        dg.add_edge(
            q-1, l)

    # Ownership relationships - R
    for r in np.arange(start=q+1, stop=2*q):
        dg.add_edge(q, r)

    # Ownership relationships - Bridge vertex
    dg.add_edge(2*q, q-1)
    dg.add_edge(2*q, q)

    # Ownsership relationships - Alternative vertices to bridge vertex
    dg.add_edge(2*q+1, q-1)
    dg.add_edge(2*q+1, q)

    dg.add_edge(2*q+2, q-1)
    dg.add_edge(2*q+2, q)

    gt.draw.graph_draw(ug, vertex_text=ug.vertex_index,
                       output="fig/n_q-graph.pdf")
    gt.draw.graph_draw(dg, vertex_text=dg.vertex_index,
                       output="fig/n_q-graph_own.pdf")

    return ug, dg


def _edge_percolation(dg, edges):
    # Edges defined on ownership graph (dg)
    for e in edges:
        dg.remove_edge(e)

    ug = copy.deepcopy(dg)

    ug.set_directed(False)

    return ug, dg


def _get_index_from_vertex(g, vertex):
    return g.vertex_index[vertex]
