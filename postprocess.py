import pickle
from graph_tool.all import *
import graph_tool as gt
import matplotlib.pyplot as plt
import networkx as nx

with open('mcts_d_regular_random_10_5.83_temp.pkl', 'rb') as file:
    mcts = pickle.load(file)

tree = nx.Graph(gt.spectral.adjacency(mcts.tree))
pos = nx.nx_agraph.graphviz_layout(tree, prog="dot")
nx.draw(tree, pos, node_size=2)
plt.show()

terminal_nodes = []
for node in mcts.s0_prop:
    if node.is_terminal:
        terminal_nodes.append(node)

        print("Terminal child. State Id: {}/{}/{}".format(node.get_id,
                                                          node.get_scores, node.get_visits))

        g = nx.DiGraph(gt.spectral.adjacency(node.NCG.network.ownership))
        if not nx.is_tree(g):
            print("NE non-tree.")
            nx.draw(g)
            plt.show()

        else:
            print("NE tree.")
            pos = nx.nx_agraph.graphviz_layout(g, prog="dot")
            nx.draw(g, pos, node_size=2)
            plt.show()

        for a in node.NCG.agents:
            # List all legal actions
            actions = node.NCG._legal_k_length_actions(a, mcts.k)
            if mcts._exists_better_response(a, actions, node):
                print("BR found.")
            else:
                print("No BR found.")

max = mcts._return_max_child(mcts.s0_prop[0])
print("Max child. State Id: {}/{}/{}/{}".format(max.get_id,
                                                max.get_scores, max.get_visits, max.get_mean_value))
