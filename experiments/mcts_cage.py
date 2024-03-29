import copy
import graph_tool as gt
import logging
import numpy as np
import pickle
import utils

import MCTS
import State
import NCG
import Network

MAX_LOOPS = 1000

#############################################################################################
#############################################################################################

# (3,5)-cage (Petersen)
# n = 10
# l = n
# adj = utils._create_petersen()

# Extended Petersen
# order = 12
# n = order
# l = n/2
# diam = 4
# adj = utils._create_extended_petersen(order, diam)

# (3,6)-cage (Heawood)
n = 14
l = n/2
adj = utils._create_heawood()

# (3,7)-cage (McGee)
# n = 24
# l = n/2
# adj = utils._create_mcgee()

alphas_ = np.logspace(np.log10(n/l), np.log10(2*n), 10, dtype=float)[-6:-1]

m = np.triu(adj.todense())
es = np.nonzero(m)
ug = gt.Graph(np.array([es[0], es[1], m[es]]).T, directed=False)

graph = Network.Network(gt.spectral.adjacency(ug).todense(), n)
dg = copy.deepcopy(ug)
dg.set_directed(True)
graph.set_ownership(dg)

for _, alpha in enumerate(alphas_):

    # Logging config
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(filename='log_mcts_extended_12_4.txt',
                        filemode='a', format=format, level=logging.DEBUG,
                        datefmt="%d/%m/%Y %H:%M:%S")

    logging.info("NCG. alphas_ {}".format(alphas_))

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
        ncg.network.ownership, vertex_text=ncg.network.ownership.vertex_index, output="fig/initial_mcts_own_extended_12_4_"+str(n)+"_"+'{0:.2f}'.format(alpha)+".pdf")

    # Create instance of MCTS
    # Budget
    k = 2
    mcts = MCTS.MCTS(state_0, k)

    # MCTS loop
    for _ in np.arange(MAX_LOOPS):
        logging.info("MCTS iteration: {}/{}".format(_+1, MAX_LOOPS))

        # Save MCTS tree
        with open('mcts_'+str(n)+'_'+str('{0:.2f}'.format(alpha))+'_temp.pkl', 'wb') as file:
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
    dir = Path('{0:.2f}'.format(alpha)+'_extended_12_4' +
               date.today().strftime("%Y%m%d"))
    dir.mkdir(parents=True, exist_ok=True)
    for pkl in src.glob("*.pkl"):
        pkl.replace(dir / pkl.name)
