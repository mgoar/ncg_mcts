import pickle
from graph_tool.all import *
import graph_tool as gt
import matplotlib.pyplot as plt
import networkx as nx

case = 'mcts_random-dense-0_k_2'
with open('temp.pkl', 'rb') as file:
    mcts = pickle.load(file)

tree = nx.Graph(gt.spectral.adjacency(mcts.tree))
pos = nx.nx_agraph.graphviz_layout(tree, prog="dot")
nx.draw(tree, pos, node_size=2)
plt.savefig(case+'-tree.pdf')
plt.show()

terminal_nodes = []
nontrees = []
for node in mcts.ne:
    if node.is_terminal:
        terminal_nodes.append(node)

        print("Terminal child. State Id: {}/{}/{}".format(node.get_id,
                                                          node.get_scores, node.get_visits))

        g = nx.DiGraph(gt.spectral.adjacency(node.NCG.network.ownership))
        if not nx.is_tree(g):
            print("NE non-tree.")
            nontrees.append(node.NCG.network)
            nx.draw(g)
            plt.show()

            # Double-check it is NE
            for a in node.NCG.agents:
                # List all legal actions
                actions = node.NCG._legal_k_length_actions(a, mcts.k)
                if mcts._exists_better_response(a, actions, node):
                    print("BR found.")
                else:
                    print("No BR found.")     

        else:
            print("NE tree.")

# Plot all non-tree NE
rows = 2

plt.switch_backend("cairo")

if(len(nontrees) != 0):
    fig, ax = plt.subplots(rows, int(len(nontrees)/rows))
    for i, this_ax in enumerate(ax.reshape(-1)):
        this_ax.axis("off")
        gt.draw.graph_draw(nontrees[i].ownership,
                            mplfig=this_ax)

    fig.savefig(case+"-non_trees.pdf")

    max = mcts._return_max_child(mcts.s0_prop[0])
    print("Max child. State Id: {}/{}/{}/{}".format(max.get_id,
                                                    max.get_scores, max.get_visits, max.get_mean_value))
