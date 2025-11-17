# Alpha-Poker-Zero

A Texas Hold'em poker AI training system with two approaches: MCTS-based self-play and PPO reinforcement learning.

## Installation

Install required dependencies:

```bash
pip3 install -r requirements.txt
```

Required packages:
- `torch` - PyTorch for neural networks
- `numpy` - Numerical operations
- `tqdm` - Progress bars

## Project Structure

### Core Game Files
- `card.py` - Card and Deck classes
- `hand_evaluator.py` - Poker hand evaluation logic
- `player.py` - Player classes (Human and Random AI)
- `poker_game.py` - Game engine for training (MCTS/PPO compatible)
- `texas_holdem.py` - Interactive game engine
- `main.py` - Interactive game entry point

### Training Files
- `config.py` - Configuration and hyperparameters
- `value_policy_network.py` - Neural network architecture
- `poker_mcts.py` - Monte Carlo Tree Search implementation
- `selfplay.py` - MCTS-based self-play data generation
- `train.py` - MCTS model training script
- `evaluate.py` - MCTS model evaluation
- `ppo_trainer.py` - PPO algorithm implementation
- `train_ppo.py` - PPO training script
- `evaluate_ppo.py` - PPO model evaluation
- `compare_ppo_mcts.py` - Compare PPO vs MCTS bots

## Training Methods

This project supports two training approaches:

1. **MCTS-based Training** (Older method)
   - Uses Monte Carlo Tree Search for self-play
   - Generates training data through tree search
   - Trains neural network via supervised learning
   - Good for learning basic strategies

2. **PPO Training** (Recommended for competitive bots)
   - Uses Proximal Policy Optimization (reinforcement learning)
   - Direct policy optimization through self-play
   - Better suited for imperfect information games like poker
   - Produces stronger competitive bots

---

## MCTS-Based Training (Step-by-Step)

### Step 1: Generate Self-Play Data

Generate training data using MCTS self-play:

```bash
# Basic: Generate 1000 games of self-play data
python3 selfplay.py --games 1000

# Advanced: More simulations per action (slower but better quality)
python3 selfplay.py --games 500 --simulations 200

# Resume from existing model (iterative improvement)
python3 selfplay.py --games 5000 --model poker-rl/output/models/poker_model_best.pt
```

**What happens:**
- MCTS plays against itself to generate (state, action_probs, reward) tuples
- Data is saved to `poker_training_dataset.pkl`
- More games = more diverse training data
- More simulations = better action probabilities

### Step 2: Train the Neural Network

Train the neural network on collected self-play data:

```bash
# Basic training
python3 train.py

# Custom parameters
python3 train.py --epochs 300 --batch-size 256 --lr 0.00005

# Continue training from existing model
python3 train.py --model poker-rl/output/models/poker_model_best.pt
```

**What happens:**
- Loads training data from `poker_training_dataset.pkl`
- Trains policy and value networks via supervised learning
- Saves checkpoints to `poker-rl/output/models/`
- Best model saved as `poker_model_best.pt`

### Step 3: Evaluate the Model

Test your trained model:

```bash
# Evaluate via self-play (model vs itself)
python3 evaluate.py --model poker-rl/output/models/poker_model_best.pt --games 100

# Test against random player (shows if training worked!)
python3 evaluate.py --model poker-rl/output/models/poker_model_best.pt --vs-random --games 100

# Compare two models
python3 evaluate.py --model model1.pt --compare model2.pt --games 200
```

**What to expect:**
- Win rate > 50% vs random = model is learning
- Win rate > 55% = model is strong
- Win rate < 45% = needs more training

### Step 4: Iterative Improvement (Optional)

Improve the model through iterative training:

```bash
# Step 1: Generate data with current model
python3 selfplay.py --games 5000 --model poker-rl/output/models/poker_model_best.pt

# Step 2: Train on accumulated data
python3 train.py --model poker-rl/output/models/poker_model_best.pt

# Step 3: Evaluate
python3 evaluate.py --model poker-rl/output/models/poker_model_best.pt --vs-random --games 100

# Repeat steps 1-3 for further improvement
```

**Iterative process:**
1. Use trained model to generate better self-play data
2. Train on accumulated data (old + new)
3. Evaluate and repeat

---

## PPO Training (Step-by-Step)

### Step 1: Train with PPO

Train a competitive bot using Proximal Policy Optimization:

```bash
# Basic training (1000 updates)
python3 train_ppo.py --updates 1000

# Continue training from existing model
python3 train_ppo.py --updates 1000 --model poker-rl/output/models/ppo_poker_model_best.pt

# Custom training length
python3 train_ppo.py --updates 2000
```

**What happens:**
- PPO agent plays against random opponents (50%) and past PPO models (50%)
- Collects rollouts of (state, action, reward) experiences
- Updates policy using PPO clipped objective
- Maintains opponent pool for diverse training
- Saves checkpoints automatically
- Best model saved as `ppo_poker_model_best.pt`

**Training details:**
- Rollout length: 4096 steps per update
- PPO epochs: 8 per update
- Batch size: 128
- Learning rate: 1e-4
- Automatically deletes old checkpoints (keeps 5 most recent)

### Step 2: Evaluate PPO Model

Evaluate your trained PPO model:

```bash
# Basic evaluation (100 games, 10 hands per game = 1000 total hands)
python3 evaluate_ppo.py --model poker-rl/output/models/ppo_poker_model_best.pt --games 100

# Custom: 100 games, 20 hands per game (2000 total hands)
python3 evaluate_ppo.py --model poker-rl/output/models/ppo_poker_model_best.pt --games 100 --hands-per-game 20

# Against random player (default)
python3 evaluate_ppo.py --model poker-rl/output/models/ppo_poker_model_best.pt --games 100 --vs-random

# Self-play evaluation
python3 evaluate_ppo.py --model poker-rl/output/models/ppo_poker_model_best.pt --games 100 --self-play

# Stochastic policy (sample instead of greedy)
python3 evaluate_ppo.py --model poker-rl/output/models/ppo_poker_model_best.pt --games 100 --stochastic
```

**Evaluation structure:**
- A "game" = multiple hands (default: 10 hands)
- Winner determined by final chip count after all hands
- 100 games = 100 matches = 1000 total hands (with 10 hands/game)

**What to expect:**
- Win rate > 50% = model is learning
- Win rate > 55% = model is strong
- Positive average chips per game = model is profitable
- Win rate < 45% = needs more training

### Step 3: Compare PPO vs MCTS

Compare your PPO bot against an MCTS-trained bot:

```bash
python3 compare_ppo_mcts.py \
    --ppo-model poker-rl/output/models/ppo_poker_model_best.pt \
    --mcts-model poker-rl/output/models/poker_model_best.pt \
    --games 100
```

**What this shows:**
- Direct comparison of PPO (policy-based) vs MCTS (tree search)
- Helps determine which approach works better for your use case

---

## Configuration

Edit `config.py` to adjust:

### Game Settings
- `STARTING_CHIPS` - Starting chip count (default: 1000)
- `SMALL_BLIND` - Small blind amount (default: 5)
- `BIG_BLIND` - Big blind amount (default: 10)

### MCTS Training
- `SELFPLAY_GAMES` - Number of games for self-play (default: 1000)
- `NUM_SIMULATIONS` - MCTS simulations per action (default: 100)
- `BATCH_SIZE` - Training batch size (default: 128)
- `EPOCHS` - Training epochs (default: 200)
- `LEARNING_RATE` - Learning rate (default: 0.0001)

### PPO Training
- `PPO_LEARNING_RATE` - PPO learning rate (default: 1e-4)
- `PPO_CLIP_EPSILON` - PPO clip parameter (default: 0.2)
- `PPO_ENTROPY_COEF` - Entropy bonus coefficient (default: 0.075)
- `PPO_ROLLOUT_LENGTH` - Steps per rollout (default: 4096)
- `PPO_BATCH_SIZE` - PPO batch size (default: 128)
- `PPO_RANDOM_OPPONENT_PROB` - Probability of random opponent (default: 0.5)
- `HANDS_PER_GAME` - Hands per game in evaluation (default: 10)

---

## Interactive Game

Play interactively against a random AI:

```bash
python main.py
```

**Gameplay:**
1. Enter your name and configure starting chips/blinds
2. Receive 2 hole cards each hand
3. Make decisions during betting rounds:
   - **Check**: Pass without betting (when no bet required)
   - **Call**: Match current bet
   - **Raise**: Increase bet amount
   - **Fold**: Give up the hand
4. Game progresses through: Pre-flop → Flop → Turn → River
5. Best 5-card hand wins at showdown

---

## Hand Rankings (Highest to Lowest)

1. Royal Flush
2. Straight Flush
3. Four of a Kind
4. Full House
5. Flush
6. Straight
7. Three of a Kind
8. Two Pair
9. Pair
10. High Card

---

## Training Tips

### MCTS Training
- Start with 1000-5000 games for initial training
- Use 100-200 simulations per action for good quality
- Iterative training improves model over time
- More training data = better generalization

### PPO Training
- Start with 1000+ updates for good performance
- Model improves gradually - be patient
- Check evaluation metrics regularly
- More updates = stronger bot (up to a point)
- Random opponent training prevents overfitting

### General
- Monitor win rates vs random player
- Positive chip average = model is profitable
- Win rate > 50% = model is learning
- Win rate > 55% = model is strong
- If performance plateaus, try adjusting hyperparameters

---

## Troubleshooting

### Model not learning
- Increase training time (more games/updates)
- Check learning rate (too high = unstable, too low = slow)
- Ensure enough training data
- Try different hyperparameters

### State size mismatch
- Model was trained with different state representation
- Solution: Retrain model with current codebase

### Low win rate
- Model needs more training
- Try adjusting hyperparameters
- Check if reward function is correct
- Ensure evaluation uses correct structure (multiple hands per game)

---

## Files Generated

### MCTS Training
- `poker_training_dataset.pkl` - Training data from self-play
- `poker-rl/output/models/poker_model.pt` - Latest model
- `poker-rl/output/models/poker_model_best.pt` - Best model
- `poker-rl/output/models/poker_model_epoch_*.pt` - Epoch checkpoints

### PPO Training
- `poker-rl/output/models/ppo_poker_model.pt` - Latest model
- `poker-rl/output/models/ppo_poker_model_best.pt` - Best model
- `poker-rl/output/models/ppo_poker_model_update_*.pt` - Update checkpoints

---

## License

This project is open source and available for educational and research purposes.
