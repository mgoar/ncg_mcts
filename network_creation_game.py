from graph_tool.all import *

from typing import Any, Iterable, List, Mapping, Optional

import numpy as np
from open_spiel.python.games import network_creation_game_data
import network_creation_game_utils
from open_spiel.python.observation import IIGObserverForPublicInfoGame
import pyspiel

_DEFAULT_PARAMS = {
    "alpha": 1,
    "players": 2
}
_GAME_TYPE = pyspiel.GameType(
    short_name="python_networ_creation_game",
    long_name="Python Network Creation Game",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.GENERAL_SUM,
    reward_model=pyspiel.GameType.RewardModel.REWARDS,
    max_num_players=10**10,
    min_num_players=2,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    default_loadable=True,
    provides_factored_observation_string=True,
    parameter_specification=_DEFAULT_PARAMS)


class NetworkCreationGameGame(pyspiel.Game):
    network: network_creation_game_utils.Network
    _agents: List[network_creation_game_utils.Agent]

    def __init__(self,
                 params: Mapping[str, Any],
                 network: Optional[network_creation_game_utils.Network] = None,
                 agents: Optional[List[network_creation_game_utils.Agent]] = None):
        """ Initialize the game.

        Args:
            params: game parameters.
            network: the network of the game.
            agents: a list of agents. Equals the number of vertices in the network.
        """

        max_num_time_step = params["max_num_time_step"]

        self.network = network if network else network_creation_game_data.EMPTY_GRAPH
        game_info = pyspiel.GameInfo(
            num_distinct_actions=self.network.num_actions(),
            max_chance_outcomes=0,      # Game is deterministic
            num_players=len(self.network._V),
            min_utility=0,
            max_utility=np.inf,
            max_game_length=max_num_time_step)

        self.network = network
        self.agents = agents

        super().__init__(_GAME_TYPE, game_info, params or dict())

    def new_initial_state(self) -> "NetworkCreationGameState":
        """Returns the state corresponding to the start of a game."""
        return NetworkCreationGameState(self, self._agents)

    def make_py_observer(self, iig_obs_type=None, params=None):
        """Returns an object used for observing game state."""
        if ((iig_obs_type is None) or
                (iig_obs_type.public_info and not iig_obs_type.perfect_recall)):
            return NetworkObserver(self.num_players(), self.max_game_length())
        else:
            # Else default observation type for imperfect information games.
            return IIGObserverForPublicInfoGame(iig_obs_type, params)


class NetworkCreationGameState(pyspiel.State):
    """A Python version of the Network Creation Game state."""

    def __init__(self, game: NetworkCreationGameGame,
                 agents: Iterable[network_creation_game_utils.Agent]):
        """Constructor; should only be called by Game.new_initial_state."""
        super().__init__(game)
        self._current_time_step = 0
        self._is_terminal = False
        self._agent_without_legal_actions = set()
        self.running_cost = [0 for _ in agents]

    @property
    def current_time_step(self) -> int:
        """Return current time step."""
        return self._current_time_step

    def current_player(self) -> pyspiel.PlayerId:
        """Returns id of the next player to move, or TERMINAL if game is over."""
        return pyspiel.PlayerId.TERMINAL if self._is_terminal else pyspiel.PlayerId.DEFAULT_PLAYER_ID

    def _legal_actions(self, agent_idx: int) -> List[int]:
        """Return the legal actions of an agent.

        Legal actions are all the subsets of the rest of the agents.
        Args:
            agent: the agent id.

        Returns:
            list_legal_actions: a list of legal actions. If the game is finished then
                the list is empty. If the agent has no legal actions, an empty list is returned.
        """
        if self._is_terminal:
            return []
        elif agent_idx in self._agent_without_legal_actions:
            return [network_creation_game_utils.NO_POSSIBLE_ACTION]
        else:
            actions = [self.get_game().network.list_subsets_i(agent_idx)]
            return sorted(actions)

    def _apply_action(self, actions):
        """Applies the specified action to the state. Actions are indexed by vertex/agent"""
        for player_i, a in enumerate(actions):
            self.get_game().network.strategic_play_player_i(a, player_i)

    def _action_to_string(self, player, action) -> str:
        """Action -> string."""
        return "{}({})".format(self.get_game().network.get_vertex_from_agent(player), action)

    def is_terminal(self):
        """Returns True if the game is over."""
        return self._is_terminal

    def rewards(self):
        """Reward at the previous step."""
        reward = [self.get_game().network.compute_cost_player_i(
            agent.state, agent._idx) for agent in self.get_game().agents]
        return reward

    def returns(self):
        """Total reward for each player over the course of the game so far."""
        return [np.sum(agent.cost_history) for agent in self.get_game().agents]

    def __str__(self):
        """String for debug purposes. No particular semantics are required."""
        return "".join(self._action_to_string(p, p.history()) for p in self.agents)

    def action_history_string(self, player):
        return "".join(
            self._action_to_string(player, player.history()))

    def get_state(self, agent: int) -> np.ndarray(dtype=bool):
        """Get agent's state."""
        return self.get_game().network.get_edges_vertex_v_as_array(
            agent)

    def get_current_agents_states(self) -> List[np.array]:
        """Get state of all agents for the observation tensor."""
        return [
            self.get_state(x)
            for x in range(self.get_game().num_players())
        ]


class NetworkObserver:
    """Network observer used by the learning algorithm.

    The state string is the state history string. The state tensor is an array
    of size (max_game_length, num_players, possible where each element is the game state at that time.
    Attributes:
        dict: dictionary {"observation": tensor}.
        tensor: game tensor for each time step.
    """

    def __init__(self, num_agents: int, num_time: int, num_actions: int):
        """Initializes an empty observation tensor."""
        shape = (num_time + 1, num_agents + 1, num_actions + 1)
        self.tensor = np.zeros(np.prod(shape), np.float32)
        self.dict = {"observation": np.reshape(self.tensor, shape)}

    def set_from(self, state, player):
        """Update the state tensor.

        Put the state of each player in the tensor row corresponding to
        the current time step.
        Args:
            state: the state,
            player: the player.
        """
        game_state = state.get_current_agents_states()
        game_state.insert(0, state.get_state(player))
        self.dict["observation"][state.current_time_step, :] = game_state

    def string_from(self, state, player):
        """Return the state history string."""
        return f"{player}: {state.action_history_string(player)}"


# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, NetworkCreationGameGame)
