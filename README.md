# Texas Hold'em Poker Game

A command-line Texas Hold'em poker game where you can play heads-up against a random AI opponent.

## Features

- Full Texas Hold'em implementation with proper betting rounds
- Pre-flop, Flop, Turn, and River gameplay
- Accurate hand evaluation (Royal Flush, Straight Flush, Four of a Kind, etc.)
- Heads-up (2 player) game
- Random AI opponent that makes random decisions
- Dealer button rotation
- Small and Big blinds
- Configurable starting chips and blind amounts

## Files

### Interactive Game
- `card.py` - Card and Deck classes
- `hand_evaluator.py` - Poker hand evaluation logic
- `player.py` - Player classes (Human and Random AI)
- `texas_holdem.py` - Main game engine
- `main.py` - Game entry point

### Self-Play & Training Pipeline
- `config.py` - Configuration and hyperparameters
- `selfplay.py` - Self-play data generation script

## How to Run

### Play Interactively
```bash
python main.py
```

### Generate Training Data (Self-Play)
```bash
# Generate 1000 games of self-play data
python selfplay.py --games 1000

# Generate with more MCTS simulations per action (slower but better quality)
python selfplay.py --games 500 --simulations 200

# Resume from a trained model
python selfplay.py --games 1000 --model output/models/best_model.pt
```

### Train the Neural Network

??? need to implement


```

## How to Play

1. Run the game with `python main.py`
2. Enter your name and configure starting chips and blinds
3. Each hand, you'll receive 2 hole cards
4. Make decisions during each betting round:
   - **Check**: Pass the action without betting (when no bet is required)
   - **Call**: Match the current bet
   - **Raise**: Increase the bet amount
   - **Fold**: Give up the hand

5. The game progresses through:
   - Pre-flop (after receiving hole cards)
   - Flop (3 community cards revealed)
   - Turn (4th community card)
   - River (5th community card)

6. At showdown, the best 5-card hand wins the pot

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
# Alpha-Poker-Zero
