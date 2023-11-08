import pickle
from graph_tool.all import *
import graph_tool as gt
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout


#with open('experiments/alpha_n/results/petersen_k_3/3.50_cage_20231106/mcts_10_3.50_temp.pkl', 'rb') as file:
with open('mcts_12_2.85_temp.pkl', 'rb') as file:
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
            nx.draw_circular(g)
            plt.show()

            for a in node.NCG.agents:
                # List all legal actions
                actions = node.NCG._legal_k_length_actions(a, mcts.k)
                if mcts._exists_better_response(a, actions, node):
                    print("BR found.")
                else:
                    print("No BR found.")
        else:
            print("NE tree.")
            nx.draw_circular(g)

max = mcts._return_max_child(mcts.s0_prop[0])
print("Max child. State Id: {}/{}/{}".format(max.get_id,
                                             max.get_scores, max.get_visits))
