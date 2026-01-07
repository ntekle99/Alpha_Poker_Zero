# 🎰 Alpha Poker Zero

<div align="center">

**Texas Hold'em poker AI comparing MCTS, DQN, and PPO algorithms with opponent modeling**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[Quick Start](#-quick-start) • [Results](#-results) • [Training](#-training) • [Docs](TRAINING_GUIDE.md)

</div>

---

## 📋 What is this?

A poker AI research project that compares different reinforcement learning approaches for imperfect information games. Key finding: **Deep RL (DQN/PPO) vastly outperforms MCTS** in poker due to hidden information (97.5% win rate).

**Features:**
- 🤖 Three AI algorithms: MCTS, DQN, PPO
- 🧠 LSTM-based opponent modeling
- 🎮 Full-stack web app with AI advisor
- 📊 Comprehensive evaluation framework

---

## 🚀 Quick Start

```bash
# Clone and install
git clone https://github.com/yourusername/Alpha-Poker-Zero.git
cd Alpha-Poker-Zero
pip install -r requirements.txt

# Run the web app
python start.py
# Open http://localhost:3000 to play with AI advice
```

**Or train your own models:**
```bash
# Train DQN (recommended)
python train_dqn.py --episodes 10000

# Train PPO
python train_ppo.py --updates 1000

# Evaluate
python run_all_evaluations.py --games 100
```

📖 See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for detailed instructions

---

## 🏗️ Architecture

```
React Frontend ←→ Flask API ←→ AI Models (DQN/PPO/MCTS + Opponent Modeling)
```

**Models:**
- **DQN**: Experience replay + target network (1.05M params)
- **PPO**: Actor-critic with GAE (1.07M params)  
- **MCTS**: Neural-guided tree search (1.05M params)
- **Psychology Network**: LSTM opponent modeling (332K params)

**State**: 320-dim vector (cards, pot, stacks, stage)  
**Actions**: 6 discrete (fold, check/call, bet small/medium/large, all-in)

---

## 📈 Results

| Algorithm | vs Random | vs MCTS | vs PPO | vs DQN |
|-----------|-----------|---------|--------|--------|
| **MCTS** | 2.5% | - | 8.0% | 2.5% |
| **PPO** | 62.5% | 92.0% | - | 47.0% |
| **DQN** | **60.5%** | **97.5%** | 53.0% | - |

**Key Finding:** Deep RL (DQN/PPO) vastly outperforms MCTS in imperfect information games. MCTS struggles because tree search assumes observable states—hidden cards break this assumption.

📊 See [RESEARCH_PROGRESSION.md](RESEARCH_PROGRESSION.md) for detailed analysis

---

## 🎓 Training

All three algorithms support self-play training and comprehensive evaluation:

**Quick training commands:**
```bash
# Train DQN (recommended - best performance)
python train_dqn.py --episodes 10000

# Train PPO (alternative approach)
python train_ppo.py --updates 1000

# Train MCTS (baseline for comparison)
python selfplay.py --games 5000 && python train.py
```

**Evaluate and compare:**
```bash
# Run all evaluations (generates comparison matrix)
python run_all_evaluations.py --games 100 \
    --dqn-model poker-rl/output/models/dqn_poker_model_best.pt \
    --ppo-model poker-rl/output/models/ppo_poker_model_best.pt \
    --mcts-model poker-rl/output/models/poker_model_best.pt
```

📖 **For detailed training instructions, hyperparameter tuning, and best practices, see:**
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Complete training tutorials
- [config.py](config.py) - All hyperparameters and settings

---

## 📚 Documentation

- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Step-by-step training for all algorithms
- **[RESEARCH_PROGRESSION.md](RESEARCH_PROGRESSION.md)** - Research narrative and findings
- **[config.py](config.py)** - Hyperparameters and configuration

---

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Statistical significance testing
- Additional poker variants
- Improved opponent modeling
- Performance optimizations

---

## 📄 License

MIT License - See LICENSE for details

---

## 🎯 Citation

If you use this project in your research, please cite:

```bibtex
@software{alpha_poker_zero,
  title={Alpha Poker Zero: Comparing MCTS and Deep RL in Imperfect Information Games},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/Alpha-Poker-Zero}
}
```
