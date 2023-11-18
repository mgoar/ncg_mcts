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

# G(n,p) random graph. Orders
n_ = np.arange(10, 16)
sparse = False
regular = False

for n in n_:
    alphas = np.hstack([n, 
                    np.logspace(np.log10(n), np.log10(2*n), 10, dtype=float)[1:3], 
                    np.logspace(np.log10(n), np.log10(n/2), 10, dtype=float)[1:3]])
    for _ in np.arange(n):
        # Logging config
        format = "%(asctime)s: %(message)s"
        logging.basicConfig(filename='log_random_'+str(n)+'_'+str(_)+'.txt',
                            filemode='a', format=format, level=logging.DEBUG,
                            datefmt="%d/%m/%Y %H:%M:%S")
        logging.info("NCG. n_ {}".format(n_))
        logging.info("NCG. alphas {}".format(np.round(alphas, 2)))

        for ll, alpha in enumerate(alphas):
            logging.info(
                "NCG. alpha/n {}/{}".format('{0:.2f}'.format(alpha), n))
            for ii in np.arange(n).astype(int):
                if not regular:
                    adj, p = utils._create_random_graph(n, sparse)
                    logging.info("utils. p={}".format('{0:.4f}'.format(p)))
                else:
                    d = 3
                    adj = utils._create_d_regular_random_graph(n, d)

                # Create arbitrary ownership
                m = np.triu(adj.todense())
                es = np.nonzero(m)
                ug = gt.Graph(np.array([es[0], es[1], m[es]]).T, directed=False)

                graph = Network.Network(gt.spectral.adjacency(ug).todense(), n)
                dg = copy.deepcopy(ug)
                dg.set_directed(True)
                graph.set_ownership(dg)

                # Create NCG
                ncg = NCG.NCG(graph, alpha, True)

                # Create initial state
                state_0 = State.State(n, 0, ncg)

                # Update scores
                val = np.array([agent.cost for agent in ncg.agents])
                state_0.set_scores(val)

                gt.draw.graph_draw(
                    ncg.network.ownership, vertex_text=ncg.network.ownership.vertex_index, output="fig/initial_mcts_own_random_"+str(n)+"_"+'{0:.2f}'.format(alpha)+"_"+str(ii)+".pdf")

                # Create instance of MCTS
                # Budget
                if not regular:
                    k = 2
                else:
                    k = d-1
                mcts = MCTS.MCTS(state_0, k)

                # MCTS loop
                for _ in np.arange(MAX_LOOPS):
                    logging.info("MCTS iteration: {}/{}".format(_+1, MAX_LOOPS))

                    # Check number of NE found so far and break execution if necessary
                    if len(mcts.ne) > len(mcts.tree.get_out_neighbors(state_0.get_id))**2:
                        logging.info("MCTS: number of NE found ({}) exceeded.".format(len(mcts.ne)))
                        break

                    # Save MCTS tree
                    with open('mcts_random_'+str(n)+'_'+str('{0:.2f}'.format(alpha))+'_'+str(ii)+'_temp.pkl', 'wb') as file:
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
                                                                                                                            this.get_scores, np.round(this.get_mean_value,2), this.get_visits))

                # Housekeeping
                from pathlib import Path
                src = Path()
                dir = Path('mcts_random_'+str(n)+'_'+str('{0:.2f}'.format(alpha))+'_'+str(ii))
                dir.mkdir(parents=True, exist_ok=True)
                for file in src.glob("*.pkl"):
                    file.replace(dir / file.name)
