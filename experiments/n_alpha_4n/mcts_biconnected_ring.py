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

n_ = (np.ceil(np.logspace(np.log10(10), np.log10(100), 5) / 2)*2).astype(int)

for ii, n in enumerate(n_):
    # Logging config
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(filename='log_biconnected_ring_'+str(n)+'.txt',
                        filemode='a', format=format, level=logging.DEBUG,
                        datefmt="%d/%m/%Y %H:%M:%S")

    alphas_ = np.logspace(np.log10(n), np.log10(4*n), 5, dtype=float)[1:]
    logging.info("NCG. alphas_ {}".format(alphas_))

    adj = utils._create_ring(n)

    m = np.triu(adj.todense())
    es = np.nonzero(m)
    ug = gt.Graph(np.array([es[0], es[1], m[es]]).T, directed=False)

    graph = Network.Network(gt.spectral.adjacency(ug).todense(), n)
    dg = copy.deepcopy(ug)
    dg.set_directed(True)
    graph.set_ownership(dg)

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
            ncg.network.ownership, vertex_text=ncg.network.ownership.vertex_index, output="fig/initial_mcts_own_biconnected_ring_"+str(n)+"_"+str(ll)+"_.pdf")

        # Create instance of MCTS
        # Budget
        k = np.round(n / 2).astype(int)
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
        dir = Path(str(ll)+'_biconnected_ring_' +
                   date.today().strftime("%Y%m%d"))
        dir.mkdir(parents=True, exist_ok=True)
        for file in src.glob("*.pkl"):
            file.replace(dir / file.name)
