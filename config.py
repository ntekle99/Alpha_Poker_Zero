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
    BATCH_SIZE = 64
    EPOCHS = 100
    LEARNING_RATE = 0.001

    # Paths
    SAVE_MODEL_PATH = "poker-rl/output/models"
    SAVE_PICKLES = "poker-rl/output/pickles"
    DATASET_PATH = "poker_training_dataset.pkl"

    DATASET_QUEUE_SIZE = 1000

    TEMPERATURE = 1.0