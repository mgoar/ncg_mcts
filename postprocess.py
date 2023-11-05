import pickle

with open('mcts_10_14.14_temp.pkl', 'rb') as file:
    mcts = pickle.load(file)

from graph_tool.all import *

terminal = mcts.s0_prop[65]

print("Terminal child. State Id: {}/{}/{}".format(terminal.get_id,
                                                  terminal.get_scores, terminal.get_visits))

graph_draw(terminal.NCG.network.ownership)

if terminal.is_terminal:
    max = mcts._return_max_child(mcts.s0_prop[0])
    print("Max child. State Id: {}/{}/{}".format(max.get_id,
          max.get_scores, max.get_visits))

    for a in terminal.NCG.agents:
        # List all legal actions
        actions = terminal.NCG._legal_k_length_actions(a, mcts.k)
        if mcts._exists_better_response(a, actions, terminal):
            print("BR found.")
        else:
            print("No BR found.")
