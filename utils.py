from graph_tool.all import *
import graph_tool as gt
import copy
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pickle

import Network


def _is_tree(node):
    g = nx.DiGraph(gt.spectral.adjacency(node.NCG.network.ownership))
    return nx.is_tree(g)


def _create_d_regular_random_graph(order, d):
    return nx.adjacency_matrix(nx.random_regular_graph(d, order))


def p_erdos_renyi(n, mode):
    if mode == "components":
        # Sharp threshold: ln(n)/n
        eps = 1/4
        return np.random.uniform(low=(1+eps)*np.log(n)/n,
                                 high=2*np.log(n)/n)
    elif mode == "connectivity":
        eps = 1/n
        return np.random.uniform(low=1/n,
                                 high=(1-eps)*np.log(n)/n)


def _create_random_graph(order, sparse=False):

    G = nx.empty_graph(order)
    if sparse:
        while not nx.is_connected(G):
            p = p_erdos_renyi(order, "connectivity")
            G = nx.fast_gnp_random_graph(order, p)

        return nx.adjacency_matrix(G), p
    else:
        while not nx.is_connected(G):
            p = p_erdos_renyi(order, "components")
            G = nx.gnp_random_graph(order, p)

        return nx.adjacency_matrix(G), p


def _create_petersen():

    return nx.adjacency_matrix(nx.petersen_graph())


def _create_extended_petersen(order, diam):

    pet = nx.petersen_graph()

    # Subgraph last layer
    bfs = _create_bfs_petersen()

    avail = order - len(pet.nodes())
    counter = len(pet.nodes())

    # Length of paths
    k = diam - nx.diameter(pet)
    n_paths = np.round(avail/k).astype(int)
    for i in np.arange(n_paths):
        idx = i
        root = bfs[2][idx]
        for _ in np.arange(k):
            pet.add_node(counter)
            pet.add_edge(root, counter)
            root = counter
            counter += 1

    return nx.adjacency_matrix(pet)


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


def _create_graph(adj, n):

    # Create arbitrary ownership
    m = np.triu(adj.todense())
    es = np.nonzero(m)
    ug = gt.Graph(
        np.array([es[0], es[1], m[es]]).T, directed=False)

    graph = Network.Network(gt.spectral.adjacency(ug).todense(), n)
    dg = copy.deepcopy(ug)
    dg.set_directed(True)
    graph.set_ownership(dg)

    return graph


def _generate_graph_collection():

    collection = []

    # G(n,p) random graph. Orders
    n_ = np.arange(10, 16, dtype=int)
    for n in n_:
        adj, _ = _create_random_graph(n, False)
        collection.append(_create_graph(adj, n))

    # Regular random graph
    n = 10
    d_ = np.array([3, 4])
    for d in d_:
        adj = _create_d_regular_random_graph(n, d)
        collection.append(_create_graph(adj, n))

    # Random dense small graphs
    n_ = np.arange(5, 9, dtype=int)

    for n in n_:
        adj, _ = _create_random_graph(n, False)
        collection.append(_create_graph(adj, n))

    return collection


def _plot_collection():

    collection = _generate_graph_collection()

    # Save to pickle
    with open('collection.pkl', 'wb') as f:
        pickle.dump(collection, f)

    plt.switch_backend("cairo")

    rows = 2

    fig, ax = plt.subplots(rows, int(len(collection)/rows))
    for i, this_ax in enumerate(ax.reshape(-1)):
        this_ax.axis("off")
        gt.draw.graph_draw(collection[i].ownership,
                           mplfig=this_ax)

    fig.savefig("collection.pdf")
