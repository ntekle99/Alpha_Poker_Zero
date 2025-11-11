"""Monte Carlo Tree Search for poker."""

import numpy as np
import math
from copy import deepcopy
from config import PokerConfig as cfg
from poker_game import PokerGame, PokerGameState


class PokerNode:
    """Node in the MCTS tree for poker."""

    def __init__(self, prior_prob: float, player: int, parent=None):
        """
        Initialize a node.

        Args:
            prior_prob: Prior probability from neural network
            player: Player to act at this node (1 or -1)
            parent: Parent node
        """
        self.state: PokerGameState = None
        self.player = player
        self.total_visits_N = 0
        self.total_action_value_W = 0
        self.mean_action_value_Q = 0
        self.prior_prob_P = prior_prob
        self.children = {}  # action_index -> PokerNode
        self.parent = parent

    def set_state(self, state: PokerGameState):
        """Set the state for this node."""
        self.state = state

    def expand(self, action_probs: np.ndarray, valid_actions: np.ndarray, next_player: int):
        """
        Expand the node with children for all valid actions.

        Args:
            action_probs: Action probabilities from neural network
            valid_actions: Binary mask of valid actions
            next_player: Player to act after this node's actions
        """
        for action_idx in range(cfg.NUM_ACTIONS):
            if valid_actions[action_idx] > 0:
                prior = action_probs[action_idx]
                self.children[action_idx] = PokerNode(prior, next_player, self)

    def is_leaf_node(self) -> bool:
        """Check if this is a leaf node."""
        return len(self.children) == 0

    def select_best_child(self) -> tuple:
        """
        Select the best child using UCB score.

        Returns:
            Tuple of (action_index, child_node)
        """
        best_score = float('-inf')
        best_action = None
        best_child = None

        c_puct = 1.0  # Exploration constant

        for action_idx, child in self.children.items():
            # UCB score: Q + c * P * sqrt(N_parent) / (1 + N_child)
            q_value = child.mean_action_value_Q
            u_value = c_puct * child.prior_prob_P * math.sqrt(self.total_visits_N) / (1 + child.total_visits_N)
            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_action = action_idx
                best_child = child

        return best_action, best_child


class PokerMCTS:
    """Monte Carlo Tree Search for poker."""

    def __init__(self, game: PokerGame, policy_value_network):
        """
        Initialize MCTS.

        Args:
            game: PokerGame instance
            policy_value_network: Neural network that returns (value, policy)
        """
        self.game = game
        self.policy_value_network = policy_value_network

    def init_root_node(self, state: PokerGameState, player: int) -> PokerNode:
        """
        Initialize the root node for MCTS.

        Args:
            state: Initial game state
            player: Current player

        Returns:
            Root node
        """
        root_node = PokerNode(prior_prob=0, player=player)
        root_node.set_state(state)
        return root_node

    def run_simulation(self, root_node: PokerNode, num_simulations: int = None) -> PokerNode:
        """
        Run MCTS simulations from the root node.

        Args:
            root_node: Starting node
            num_simulations: Number of simulations to run

        Returns:
            Root node with updated statistics
        """
        if num_simulations is None:
            num_simulations = cfg.NUM_SIMULATIONS

        # Get neural network predictions for root node
        state_vector = self.game.get_canonical_state(root_node.state, root_node.player)
        value, action_probs = self.policy_value_network(state_vector, root_node.player)

        # Mask invalid actions
        valid_actions = self.game.get_valid_moves(root_node.state, root_node.player)
        action_probs = action_probs * valid_actions
        if action_probs.sum() > 0:
            action_probs = action_probs / action_probs.sum()
        else:
            # All actions invalid? Shouldn't happen, but handle it
            action_probs = valid_actions / valid_actions.sum()

        # Expand root node
        next_player = -root_node.player
        root_node.expand(action_probs, valid_actions, next_player)

        # Run simulations
        for _ in range(num_simulations):
            self._run_single_simulation(root_node)

        return root_node

    def _run_single_simulation(self, root_node: PokerNode):
        """Run a single MCTS simulation."""
        node = root_node
        search_path = [node]

        # Selection: traverse tree until we reach a leaf
        while not node.is_leaf_node():
            action, node = node.select_best_child()
            search_path.append(node)

        # Get the parent node and action that led to this leaf
        parent = search_path[-2] if len(search_path) > 1 else None
        if parent is None:
            # Root is leaf (shouldn't happen after expansion)
            value = 0
        else:
            # Find which action led to this node
            action = None
            for act, child in parent.children.items():
                if child is node:
                    action = act
                    break

            # Apply action to get the leaf node's state
            new_state = parent.state.copy()
            self.game.apply_action(new_state, action, parent.player)
            node.set_state(new_state)

            # Expansion and evaluation
            if self.game.is_terminal(new_state):
                # Terminal node - use actual reward
                value = self.game.get_reward(new_state, root_node.player)
            else:
                # Non-terminal - evaluate with neural network and expand
                state_vector = self.game.get_canonical_state(new_state, node.player)
                value, action_probs = self.policy_value_network(state_vector, node.player)

                # Mask invalid actions
                valid_actions = self.game.get_valid_moves(new_state, node.player)
                action_probs = action_probs * valid_actions
                if action_probs.sum() > 0:
                    action_probs = action_probs / action_probs.sum()
                else:
                    action_probs = valid_actions / valid_actions.sum()

                # Expand this node
                next_player = -node.player
                node.expand(action_probs, valid_actions, next_player)

        # Backup: propagate value up the tree
        self._backup(search_path, value, root_node.player)

    def _backup(self, search_path: list, value: float, root_player: int):
        """
        Backup the value through the search path.

        Args:
            search_path: List of nodes from root to leaf
            value: Value to backup (from root player's perspective)
            root_player: Player at root
        """
        for node in reversed(search_path):
            node.total_visits_N += 1

            # Value is from root player's perspective
            # Convert to current node's player perspective
            if node.player == root_player:
                node_value = value
            else:
                node_value = -value

            node.total_action_value_W += node_value
            node.mean_action_value_Q = node.total_action_value_W / node.total_visits_N

    def select_action(self, node: PokerNode, temperature: float = 1.0) -> tuple:
        """
        Select an action from the node based on visit counts.

        Args:
            node: Node to select action from
            temperature: Temperature for action selection (higher = more exploration)

        Returns:
            Tuple of (action_index, action_probs)
        """
        # Get visit counts
        visits = np.zeros(cfg.NUM_ACTIONS)
        for action, child in node.children.items():
            visits[action] = child.total_visits_N

        # Apply temperature
        if temperature == 0:
            # Greedy selection
            action = np.argmax(visits)
            action_probs = np.zeros(cfg.NUM_ACTIONS)
            action_probs[action] = 1.0
        else:
            # Stochastic selection with temperature
            visits_temp = visits ** (1.0 / temperature)
            if visits_temp.sum() > 0:
                action_probs = visits_temp / visits_temp.sum()
            else:
                # No visits? Shouldn't happen
                valid_actions = self.game.get_valid_moves(node.state, node.player)
                action_probs = valid_actions / valid_actions.sum()

            action = np.random.choice(cfg.NUM_ACTIONS, p=action_probs)

        return action, action_probs


if __name__ == "__main__":
    # Test MCTS
    print("Testing PokerMCTS...")

    from value_policy_network import ValuePolicyNetworkWrapper

    game = PokerGame()
    vpn = ValuePolicyNetworkWrapper()
    mcts = PokerMCTS(game, vpn.get_vp)

    # Initialize a new hand
    state = game.init_new_hand()
    print(f"Initial state: Player {state.current_player} to act")
    print(f"Pot: ${state.pot}, P1 stack: ${state.player1_stack}, P2 stack: ${state.player2_stack}")

    # Create root node and run MCTS
    root = mcts.init_root_node(state, state.current_player)
    print(f"\nRunning {cfg.NUM_SIMULATIONS} MCTS simulations...")

    root = mcts.run_simulation(root, num_simulations=50)
    print(f"Root visits: {root.total_visits_N}")
    print(f"Number of children: {len(root.children)}")

    # Select action
    action, action_probs = mcts.select_action(root, temperature=1.0)
    print(f"\nSelected action: {action}")
    print(f"Action probabilities: {action_probs}")

    print("\nMCTS test complete!")
