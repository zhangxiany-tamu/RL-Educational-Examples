# Reinforcement Learning Educational Examples

Educational implementations of reinforcement learning algorithms with clear explanations and runnable code.

## Current Examples

<details>
<summary><strong>01 - Tic-Tac-Toe Q-Learning</strong></summary>

Q-Learning implementation using the classic game of tic-tac-toe to demonstrate basic reinforcement learning concepts.

**Key Learning Concepts:**
- State representation in discrete environments
- Q-table initialization and updates
- Epsilon-greedy exploration strategy
- Reward engineering for game environments
- Training against different opponent strategies

**Files:**
- `tic_tac_toe_game.py` - Interactive game against trained AI
- `tic_tac_toe_rl.ipynb` - Step-by-step tutorial notebook
- `README.md` - Detailed learning objectives and instructions

**Usage:**
1. Navigate to `01-tic-tac-toe-q-learning`
2. Run the Jupyter notebook for interactive learning
3. Or run the Python script to play against the trained AI:
   ```bash
   python tic_tac_toe_game.py
   ```

**Playing Against the AI:**
The Python script will:
- Train a Q-learning agent (takes a few minutes)
- Let you play interactive games against the trained AI
- You can choose to go first or let the AI go first
- Enter positions 1-9 to make your moves

</details>

## Upcoming Examples

More examples will be added to cover additional reinforcement learning algorithms and environments across different complexity levels and problem domains.

## Getting Started

### Prerequisites
```bash
pip install numpy matplotlib jupyter
```

### Usage
1. Clone this repository
2. Navigate to the desired example directory
3. Run the Jupyter notebook for interactive learning
4. Or run the Python script to interact with the trained AI

## Structure

Each example includes:
- README with learning objectives
- Jupyter notebook with explanations
- Standalone Python implementation

## License

MIT License