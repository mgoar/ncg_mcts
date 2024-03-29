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

# Order (even)
n_ = (np.ceil(np.logspace(np.log10(10), np.log10(100), 10) / 2)*2).astype(int)

for n in n_:
    alphas = np.hstack([n, 
                    np.logspace(np.log10(n), np.log10(2*n), 10, dtype=float)[1:3], 
                    np.logspace(np.log10(n), np.log10(n/2), 10, dtype=float)[1:3]])
    for _ in np.arange(n):
        # Logging config
        format = "%(asctime)s: %(message)s"
        logging.basicConfig(filename='log_biconnected_ring_'+str(n)+'.txt',
                            filemode='a', format=format, level=logging.DEBUG,
                            datefmt="%d/%m/%Y %H:%M:%S")
        logging.info("NCG. n_ {}".format(n_))
        logging.info("NCG. alphas_ {}".format(alphas))

        adj = utils._create_ring(n)

        m = np.triu(adj.todense())
        es = np.nonzero(m)
        ug = gt.Graph(np.array([es[0], es[1], m[es]]).T, directed=False)

        graph = Network.Network(gt.spectral.adjacency(ug).todense(), n)
        dg = copy.deepcopy(ug)
        dg.set_directed(True)
        graph.set_ownership(dg)

        for ll, alpha in enumerate(alphas):

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
                ncg.network.ownership, vertex_text=ncg.network.ownership.vertex_index, output="fig/initial_mcts_own_biconnected_ring_"+str(n)+"_"+'{0:.2f}'.format(alpha)+".pdf")

            # Create instance of MCTS
            # Budget
            k = np.round(n / 2).astype(int)
            mcts = MCTS.MCTS(state_0, k)

            # MCTS loop
            for _ in np.arange(MAX_LOOPS):
                logging.info("MCTS iteration: {}/{}".format(_+1, MAX_LOOPS))

                # Check number of NE found so far and break execution if necessary
                if len(mcts.ne) > mcts.tree.get_out_neighbors(state_0.get_id)**2:
                    logging.info("MCTS: number of NE found ({}) exceeded.".format(len(mcts.ne)))
                    break
                
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
            src = Path()
            dir = Path('mcts_biconnected_ring_' + str(ll))
            dir.mkdir(parents=True, exist_ok=True)
            for file in src.glob("*.pkl"):
                file.replace(dir / file.name)