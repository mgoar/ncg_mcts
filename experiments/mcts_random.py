from pathlib import Path
import graph_tool as gt
import logging
import numpy as np
import pickle
import yaml

import MCTS
import State
import NCG

MAX_LOOPS = 1000

# Load collection
with open('collection.pkl', 'rb') as f:
    collection = pickle.load(f)

with open(f'experiments/experiments.yaml', 'r') as yamlf:
    exp = yaml.safe_load(yamlf)

for this_exp in exp:
    # Get id
    id = this_exp['id']

    # Get graph
    graph = collection[this_exp['index']]

    # n, alpha
    n = this_exp['n']
    alpha = this_exp['alpha']

    # Greedy
    if this_exp['greedy']:
        k_ = this_exp['k']
    else:
        k_ = n

    # Iterate over k
    for k in k_:
        # Create NCG
        ncg = NCG.NCG(graph, alpha, True)

        # Create initial state
        state_0 = State.State(n, 0, ncg)

        # Update scores
        val = np.array([agent.cost for agent in ncg.agents])
        state_0.set_scores(val)

        gt.draw.graph_draw(
            ncg.network.ownership, vertex_text=ncg.network.ownership.vertex_index, output="fig/mcts_"+id+".pdf")

        # Logging config
        format = "%(asctime)s: %(message)s"
        logging.basicConfig(filename="log_"+id+".txt",
                            filemode='a', format=format, level=logging.DEBUG,
                            datefmt="%d/%m/%Y %H:%M:%S")

        mcts = MCTS.MCTS(state_0, k)

        logging.info(
            "NCG. alpha/n {}/{}".format('{0:.2f}'.format(alpha), n))

        # MCTS loop
        for _ in np.arange(MAX_LOOPS):
            logging.info(
                "MCTS iteration: {}/{}".format(_+1, MAX_LOOPS))

            # Check number of NE found so far and break execution if necessary
            if (len(mcts.ne) > len(mcts.tree.get_out_neighbors(state_0.get_id))**2) or (len(mcts.ne) > MAX_LOOPS/2):
                logging.info(
                    "MCTS: number of NE found ({}) exceeded.".format(len(mcts.ne)))
                break

            # Save MCTS tree
            with open('temp.pkl', 'wb') as file:
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
                                                                                                                      this.get_scores, np.round(this.get_mean_value, 3), this.get_visits))

        # Housekeeping
        src = Path()
        dir = Path('mcts_'+id)
        dir.mkdir(parents=True, exist_ok=True)
        for file in src.glob("temp.pkl"):
            file.replace(dir / file.name)
