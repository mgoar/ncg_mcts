import graph_tool as gt
import logging
import numpy as np
import pickle

import MCTS
import State
import NCG
import Network
import utils

MAX_LOOPS = 1000

# Logging config
format = "%(asctime)s: %(message)s"
logging.basicConfig(filename='log_biconnected_n_q.txt',
                    filemode='a', format=format, level=logging.DEBUG,
                    datefmt="%d/%m/%Y %H:%M:%S")

#############################################################################################
#############################################################################################
# (n,q)-graph
q = 4
n = 2*q + 3

ug, dg = utils._create_n_q_graph(q)

# Percolation
# ug_p, dg_p = utils._edge_percolation(dg, [[8, 3], [9, 4]])

# graph = Network.Network(gt.spectral.adjacency(ug_p).todense(), n)
graph = Network.Network(gt.spectral.adjacency(ug).todense(), n)
# graph.set_ownership(dg_p)
graph.set_ownership(dg)

alphas_ = np.logspace(np.log10(n), np.log10(4*n), 10, dtype=float)[1:]
logging.info("NCG. alphas_ {}".format(alphas_))

for ll, alpha in enumerate(alphas_):

    logging.info("NCG. alpha/n {}/{}".format('{0:.2f}'.format(alpha), n))

    #############################################################################################
    #############################################################################################

    # Create NCG
    ncg = NCG.NCG(graph, alpha, True)

    # Create initial state
    state_0 = State.State(n, 0, ncg)

    # Update scores
    val = np.array([agent.cost for agent in ncg.agents])
    state_0.set_scores(val)

    gt.draw.graph_draw(
        ncg.network.ownership, vertex_text=ncg.network.ownership.vertex_index, output="fig/initial_mcts_own_biconnected_n_q.pdf")

    # Create instance of MCTS
    # Budget
    k = 2
    mcts = MCTS.MCTS(state_0, k)

    # MCTS loop
    for _ in np.arange(MAX_LOOPS):
        logging.info("MCTS iteration: {}/{}".format(_+1, MAX_LOOPS))

        # Save MCTS tree
        with open('mcts_'+str('{0:.2f}'.format(alpha))+'_temp.pkl', 'wb') as file:
            pickle.dump(mcts, file)

        s = mcts.selection(state_0)
        t = mcts.simulation(s)

        if t is None:
            logging.info(
                "Stopping after simulation")
            break

        elif t.is_terminal:
            logging.info(
                "NE found. State Id: {}/{}".format(t.get_id, t.get_scores))

        mcts.backpropagation(s, t)

        states = [mcts.s0_prop[v] for v in mcts.tree.get_vertices()]
        for this in states:
            logging.info("State Id: {} (parent: {}, terminal: {}). Scores/mean val/visits count: {}/{}/{}".format(this.get_id, this.get_parent_id, this.is_terminal,
                                                                                                                  this.get_scores, this.get_mean_value, this.get_visits))

    # Housekeeping
    from pathlib import Path
    from datetime import date
    src = Path()
    dir = Path(str(ll)+'_biconnected_n_q_' + date.today().strftime("%Y%m%d"))
    dir.mkdir(parents=True, exist_ok=True)
    for file in src.glob("*.pkl"):
        file.replace(dir / file.name)
