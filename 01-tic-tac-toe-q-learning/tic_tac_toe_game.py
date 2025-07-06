import numpy as np
import random
from collections import defaultdict

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # 1 for X, -1 for O
        
    def reset(self, random_start=False):
        self.board = np.zeros((3, 3), dtype=int)
        if random_start:
            self.current_player = random.choice([1, -1])
        else:
            self.current_player = 1
        return self.get_state()
    
    def get_state(self):
        # Convert board to string representation for Q-table key
        # State contains ONLY current board configuration (not history)
        # Format: "[0 0 0 0 1 0 0 0 0]" where 0=empty, 1=X, -1=O
        # This is sufficient because tic-tac-toe is Markovian
        return str(self.board.flatten())
    
    def get_valid_actions(self):
        valid_actions = []
        for i in range(3):
            for j in range(3):
                if self.board[i, j] == 0:
                    valid_actions.append((i, j))
        return valid_actions
    
    def make_move(self, action):
        row, col = action
        if self.board[row, col] != 0:
            return False
        self.board[row, col] = self.current_player
        return True
    
    def check_winner(self):
        for i in range(3):
            if abs(sum(self.board[i, :])) == 3:
                return self.board[i, 0]
            if abs(sum(self.board[:, i])) == 3:
                return self.board[0, i]
        
        if abs(sum([self.board[i, i] for i in range(3)])) == 3:
            return self.board[0, 0]
        if abs(sum([self.board[i, 2-i] for i in range(3)])) == 3:
            return self.board[0, 2]
        
        return 0
    
    def is_game_over(self):
        return self.check_winner() != 0 or len(self.get_valid_actions()) == 0
    
    def get_reward(self, player):
        winner = self.check_winner()
        if winner == player:
            return 1
        elif winner == -player:
            return -1
        else:
            return 0
    
    def display(self):
        symbols = {0: ' ', 1: 'X', -1: 'O'}
        print("\n   0   1   2")
        for i in range(3):
            print(f"{i}  {symbols[self.board[i,0]]} | {symbols[self.board[i,1]]} | {symbols[self.board[i,2]]}")
            if i < 2:
                print("  -----------")
        print()

class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.learning_rate = learning_rate    # α: How much new info overrides old (0.1)
        self.discount_factor = discount_factor # γ: Importance of future rewards (0.9)
        self.epsilon = epsilon                # ε: Exploration probability (starts 0.1)
        
        # Q-table: Q[state][action] = expected_reward
        # - Stores quality values (NOT probabilities)
        # - Starts empty, grows during training
        # - Only contains valid, reachable game states
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.player = 1  # This agent plays as X
        
    def get_action(self, state, valid_actions):
        # Epsilon-greedy policy: balance exploration vs exploitation
        if random.random() < self.epsilon:
            # EXPLORATION: Choose random action to discover new strategies
            return random.choice(valid_actions)
        else:
            # EXPLOITATION: Choose action with highest Q-value
            q_values = [self.q_table[state][action] for action in valid_actions]
            max_q = max(q_values)
            # If multiple actions have same max Q-value, choose randomly among them
            best_actions = [action for action, q in zip(valid_actions, q_values) if q == max_q]
            return random.choice(best_actions)
    
    def update_q_table(self, state, action, reward, next_state, next_valid_actions):
        # Q-Learning update rule: Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
        current_q = self.q_table[state][action]
        
        if next_valid_actions:
            # Game continues: consider future rewards
            max_next_q = max([self.q_table[next_state][next_action] for next_action in next_valid_actions])
        else:
            # Game over: no future rewards
            max_next_q = 0
        
        # Temporal difference learning: adjust Q-value based on prediction error
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action] = new_q
    
    def decay_epsilon(self, decay_rate=0.995):
        # Reduce exploration over time: ε = ε × 0.995
        # Starts at 0.1 (10% random), decays to minimum 0.01 (1% random)
        # This shifts from exploration (learning) to exploitation (using knowledge)
        self.epsilon = max(0.01, self.epsilon * decay_rate)

class RandomAgent:
    def __init__(self):
        self.player = -1
    
    def get_action(self, state, valid_actions):
        return random.choice(valid_actions)

def train_agent(episodes=10000):
    """Train Q-learning agent through self-play against random opponent.
    
    Training Process:
    1. Random starting player (~50% agent first, ~50% agent second)
    2. Play complete games, storing agent's moves
    3. At game end, update Q-values for all agent moves
    4. Decay epsilon to reduce exploration over time
    
    Agent learns:
    - Opening strategies (when going first)
    - Defensive responses (when going second)
    - Tactical patterns (blocking, winning, forks)
    """
    env = TicTacToe()
    q_agent = QLearningAgent()
    random_agent = RandomAgent()
    
    wins = 0
    losses = 0
    draws = 0
    first_player_games = 0  # Track when Q-agent goes first
    
    for episode in range(episodes):
        state = env.reset(random_start=True)  # Randomly choose starting player
        game_history = []  # Store (state, action, player) for Q-table updates
        
        # Track if Q-agent starts first (important for balanced learning)
        if env.current_player == q_agent.player:
            first_player_games += 1
        
        while not env.is_game_over():
            valid_actions = env.get_valid_actions()
            
            if env.current_player == q_agent.player:
                action = q_agent.get_action(state, valid_actions)
                env.make_move(action)
                game_history.append((state, action, env.current_player))
            else:
                action = random_agent.get_action(state, valid_actions)
                env.make_move(action)
            
            env.current_player *= -1
            state = env.get_state()
        
        # Game over: assign rewards and update Q-table
        final_reward = env.get_reward(q_agent.player)  # +1 win, -1 loss, 0 draw
        
        # Backward learning: update Q-values for all moves made by Q-agent
        # Each move gets same final reward (win/loss/draw)
        for i, (hist_state, hist_action, player) in enumerate(game_history):
            if player == q_agent.player:
                if i < len(game_history) - 1:
                    next_state = game_history[i + 1][0] if i + 1 < len(game_history) else state
                    next_valid_actions = env.get_valid_actions() if not env.is_game_over() else []
                else:
                    next_state = state
                    next_valid_actions = []
                
                q_agent.update_q_table(hist_state, hist_action, final_reward, next_state, next_valid_actions)
        
        if final_reward == 1:
            wins += 1
        elif final_reward == -1:
            losses += 1
        else:
            draws += 1
        
        # Decay epsilon: gradually shift from exploration to exploitation
        q_agent.decay_epsilon()
        
        if (episode + 1) % 2000 == 0:
            win_rate = wins / (episode + 1)
            first_player_rate = first_player_games / (episode + 1)
            print(f"Episode {episode + 1}: Win rate = {win_rate:.3f}, First player rate = {first_player_rate:.3f}")
    
    print(f"\nTraining completed after {episodes} episodes:")
    print(f"Wins: {wins} ({wins/episodes:.1%})")
    print(f"Losses: {losses} ({losses/episodes:.1%})")
    print(f"Draws: {draws} ({draws/episodes:.1%})")
    print(f"Q-agent went first: {first_player_games} times ({first_player_games/episodes:.1%})")
    
    return q_agent

def play_against_agent(agent):
    """Interactive game between human and trained agent.
    
    Key difference from training:
    - Agent uses pure greedy policy (epsilon = 0)
    - No exploration/randomness during gameplay
    - Agent plays optimally to maximize challenge
    """
    env = TicTacToe()
    original_epsilon = agent.epsilon
    agent.epsilon = 0  # Pure greedy: no exploration when playing
    
    print("🎮 Welcome to Tic-Tac-Toe!")
    print("You are O (circles), the agent is X (crosses)")
    print("\nBoard positions:")
    print("  1 | 2 | 3")
    print("  ---------")
    print("  4 | 5 | 6") 
    print("  ---------")
    print("  7 | 8 | 9")
    print("\nJust enter a number 1-9 to make your move!\n")
    
    # Ask who should go first
    while True:
        first_choice = input("Who should go first? (y)ou or (a)gent: ").lower()
        if first_choice in ['y', 'you']:
            user_goes_first = True
            break
        elif first_choice in ['a', 'agent']:
            user_goes_first = False
            break
        else:
            print("Please enter 'y' for you or 'a' for agent.")
    
    pos_map = {
        1: (0,0), 2: (0,1), 3: (0,2),
        4: (1,0), 5: (1,1), 6: (1,2),
        7: (2,0), 8: (2,1), 9: (2,2)
    }
    
    reverse_map = {v: k for k, v in pos_map.items()}
    
    # Set starting player based on choice
    if user_goes_first:
        env.reset()
        env.current_player = -1  # User (O) goes first
        print("\n🎯 You go first!")
    else:
        env.reset()
        env.current_player = 1   # Agent (X) goes first
        print("\n🤖 Agent goes first!")
    
    state = env.get_state()
    env.display()
    
    while not env.is_game_over():
        valid_actions = env.get_valid_actions()
        
        if env.current_player == agent.player:  # Agent's turn (X)
            print("🤖 Agent's turn...")
            action = agent.get_action(state, valid_actions)
            env.make_move(action)
            agent_pos = reverse_map[action]
            print(f"Agent chose position {agent_pos}")
            
        else:  # Human player's turn (O)
            print("👤 Your turn!")
            valid_numbers = [reverse_map[action] for action in valid_actions]
            print(f"Available positions: {sorted(valid_numbers)}")
            
            try:
                move_input = input("Enter position (1-9): ")
                position = int(move_input)
                
                if position in valid_numbers:
                    action = pos_map[position]
                    env.make_move(action)
                else:
                    print("❌ Invalid move! Position already taken or out of range.")
                    continue
                    
            except ValueError:
                print("❌ Invalid input! Please enter a number 1-9.")
                continue
        
        env.current_player *= -1
        state = env.get_state()
        env.display()
    
    winner = env.check_winner()
    if winner == agent.player:
        print("🤖 Agent wins!")
    elif winner == -agent.player:
        print("🎉 You win!")
    else:
        print("🤝 It's a draw!")
    
    agent.epsilon = original_epsilon

if __name__ == "__main__":
    print("Training Q-learning agent...")
    trained_agent = train_agent(500000)
    
    print("\n" + "="*50)
    print("Training complete! Let's play!")
    print("="*50)
    
    while True:
        play_against_agent(trained_agent)
        
        play_again = input("\nPlay again? (y/n): ").lower()
        if play_again != 'y':
            break
    
    print("Thanks for playing! 🎮")