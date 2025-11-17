"""Configuration for poker self-play and training."""

class PokerConfig:
    # Game settings
    STARTING_CHIPS = 1000
    SMALL_BLIND = 5
    BIG_BLIND = 10

    # Self-play settings
    SELFPLAY_GAMES = 1000
    NUM_SIMULATIONS = 100  # MCTS simulations per action

    # Action space - discretized betting actions
    # Actions: fold, check/call, bet/raise (small, medium, large, all-in)
    ACTION_FOLD = 0
    ACTION_CHECK_CALL = 1
    ACTION_BET_SMALL = 2   # 0.5x pot
    ACTION_BET_MEDIUM = 3  # 1x pot
    ACTION_BET_LARGE = 4   # 2x pot
    ACTION_ALL_IN = 5
    NUM_ACTIONS = 6

    # State representation
    # 52 cards for 2 hole cards (one-hot) = 52
    # 52 cards × 5 for up to 5 community cards (one-hot each) = 260
    # + pot size (normalized) = 1
    # + player stack (normalized) = 1
    # + opponent stack (normalized) = 1
    # + current bet (normalized) = 1
    # + game stage (one-hot: preflop, flop, turn, river) = 4
    STATE_SIZE = 52 + (52 * 5) + 4 + 4  # = 52 + 260 + 4 + 4 = 320

    # Training settings
    BATCH_SIZE = 128  # Increased batch size
    EPOCHS = 200  # More epochs for better convergence
    LEARNING_RATE = 0.0001  # Lower learning rate for stability
    WEIGHT_DECAY = 1e-4  # L2 regularization
    VALUE_LOSS_WEIGHT = 1.0  # Weight for value loss
    POLICY_LOSS_WEIGHT = 1.0  # Weight for policy loss
    
    # Paths
    SAVE_MODEL_PATH = "poker-rl/output/models"
    SAVE_PICKLES = "poker-rl/output/pickles"
    DATASET_PATH = "poker_training_dataset.pkl"

    DATASET_QUEUE_SIZE = 10000  # Increased dataset size

    TEMPERATURE = 1.0  # For exploration during self-play
    
    # PPO Training settings (IMPROVED)
    PPO_LEARNING_RATE = 1e-4  # Lower for more stability
    PPO_CLIP_EPSILON = 0.2
    PPO_VALUE_COEF = 1.0  # Increased - value function is important
    PPO_ENTROPY_COEF = 0.075  # Balanced - enough exploration without too much randomness
    PPO_MAX_GRAD_NORM = 0.5
    PPO_GAMMA = 0.99
    PPO_GAE_LAMBDA = 0.95
    PPO_EPOCHS = 8  # More epochs for better policy updates
    PPO_BATCH_SIZE = 128  # Larger batches for stability
    PPO_ROLLOUT_LENGTH = 4096  # Longer rollouts for better gradients
    PPO_UPDATE_FREQUENCY = 4096  # Update after this many steps
    PPO_OPPONENT_POOL_SIZE = 8  # More diverse opponents
    PPO_OPPONENT_UPDATE_FREQ = 5  # Update more frequently
    PPO_RANDOM_OPPONENT_PROB = 0.5  # Balanced: 50% random, 50% PPO opponents
    
    # Evaluation settings
    HANDS_PER_GAME = 10  # Number of hands per game/match
    
    # DQN Training settings
    DQN_LEARNING_RATE = 1e-4
    DQN_GAMMA = 0.99
    DQN_EPSILON_START = 1.0
    DQN_EPSILON_END = 0.01
    DQN_EPSILON_DECAY = 0.995
    DQN_BATCH_SIZE = 64
    DQN_TARGET_UPDATE_FREQ = 100
    DQN_REPLAY_BUFFER_SIZE = 100000
    DQN_RANDOM_OPPONENT_PROB = 0.5  # Probability of random opponent