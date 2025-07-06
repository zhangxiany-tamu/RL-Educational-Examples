# Tic-Tac-Toe Q-Learning

## Overview
This example demonstrates Q-Learning fundamentals using the classic game of tic-tac-toe. Perfect for understanding basic reinforcement learning concepts.

## Learning Objectives
- Understand Q-Learning algorithm and Q-table structure
- Learn epsilon-greedy exploration vs exploitation
- See how states are represented (Markovian property)
- Experience training vs playing behavior differences

## Files
- `tic_tac_toe_rl.ipynb` - Interactive Jupyter notebook with detailed explanations
- `tic_tac_toe_game.py` - Standalone Python script for terminal play

## Quick Start

### Option 1: Jupyter Notebook (Recommended)
```bash
jupyter notebook tic_tac_toe_rl.ipynb
```

### Option 2: Play Against the AI
```bash
python tic_tac_toe_game.py
```

The Python script will:
- Train a Q-learning agent (takes a few minutes)
- Let you play interactive games against the trained AI
- You can choose to go first or let the AI go first
- Enter positions 1-9 to make your moves

## Key Concepts Covered

### Q-Learning Algorithm
- Q-table: Stores quality values for state-action pairs
- Bellman equation: Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
- Temporal difference learning: Learn from prediction errors

### State Representation
- Markovian states: Only current board matters (not move history)
- Sparse Q-table: Only stores encountered states (~3000 out of 19,683 possible)
- State format: String representation of 3x3 board

### Exploration vs Exploitation
- Epsilon-greedy policy: Balance learning new strategies vs using known good moves
- Epsilon decay: Start with 10% exploration, decay to 1% over time
- Training vs Playing: Exploration during training, pure greedy when playing

## Training Process
1. Random starting player: Agent learns both offensive and defensive strategies
2. Experience collection: Store (state, action, player) tuples during gameplay
3. Backward learning: Update Q-values for all agent moves after game ends
4. Progressive improvement: ~80% win rate against random opponent after 10,000 episodes

## Results
- Final win rate: ~80% against random opponent
- Balanced learning: ~50% first player, ~50% second player during training
- Strategic knowledge: Learns center/corner preferences, blocking, winning moves

## Educational Value
This example provides a concrete, visual introduction to reinforcement learning that's easy to understand and experiment with. The game rules are simple, but the learning process demonstrates all key RL concepts.

## Extensions to Try
- Train against different opponents (minimax, human players)
- Experiment with different hyperparameters (α, γ, ε)
- Implement other RL algorithms (SARSA, Monte Carlo)
- Add more sophisticated state representations