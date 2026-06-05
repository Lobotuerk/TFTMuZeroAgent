# Monte Carlo Tree Search C++ Implementation & Python Bindings

**Monte Carlo Tree Search (MCTS)** is a probabilistic search algorithm that uses random simulations to selectively (i.e. in an imbalanced manner) grow a game tree. MCTS has been particularly successful in domains with vast search spaces (i.e. game trees with high branching factor) where deterministic algorithms such as minimax (or alpha-beta pruning) have struggled.

This project contains a fast **C++** implementation of the vanilla MCTS algorithm with **Python bindings** (PyMCTS) that allow you to implement new games entirely in Python while leveraging the high-performance C++ MCTS core.

## ✨ **New Features & Updates**

### 🔄 **Multi-Player Support & Interface Changes**
- **Flexible Player System**: Support for 1 vs N player scenarios beyond traditional 2-player games
- **Minimal Interface**: Single `is_self_side_turn()` method for turn determination
- **⚠️ BREAKING CHANGE**: `player1_turn()` method has been removed - use `is_self_side_turn()` instead
- **Migration Required**: All existing code must be updated to use the new interface
- **Enhanced Capabilities**: Supports battle royale, team vs team, and complex game modes

### 🎮 **Complete Python Game Implementations**
- **Connect Four**: Full 6x7 board game with win detection and visual interface
- **Multiple Demo Games**: Coin flip, number guessing, and more learning examples
- **Pure Python**: Implement complex games without touching C++ code

### 🧪 **Modern Testing Framework**
- **PyTest Integration**: Professional test suite with fixtures and markers
- **Comprehensive Coverage**: Core functionality, parallel processing, and Python inheritance tests
- **Test Organization**: Clean separation with dedicated `tests/` folder

### 📁 **Organized Project Structure**
- **demo/**: Complete game demonstrations and examples
- **tests/**: Comprehensive test suite with documentation
- **Improved Documentation**: Step-by-step guides and implementation templates

### ⚡ **Enhanced Performance**
- **Configurable Multi-threading**: Adjust thread count for optimal performance
- **Hardware Detection**: Automatic CPU core detection and optimization
- **Memory Management**: Improved C++ object lifecycle and cleanup

### 🧠 **Heuristic Rollout Enhancement**
- **Smart Rollouts**: Enhanced simulations using domain knowledge instead of pure random play
- **Move Evaluation**: Quantitative assessment of move quality (0.0 to 1.0 scoring)
- **Position Analysis**: Game state evaluation based on strategic principles
- **Configurable Strategy**: Mix heuristic and random rollouts for optimal exploration
- **Game-Specific Intelligence**: Win detection, threat blocking, and strategic positioning

## 🚀 Quick Start

### Python Usage (PyMCTS) - **Recommended**
Implement games in pure Python while leveraging high-performance C++ MCTS:

```python
import sys
sys.path.append('build')  # Add build directory to path
import pymcts

# Step 1: Define your move class
class MyGameMove(pymcts.MCTS_move):
    def __init__(self, move_data):
        super().__init__()
        self.move_data = move_data
    
    def __eq__(self, other):
        return self.move_data == other.move_data
    
    def sprint(self):
        return f"Move: {self.move_data}"

# Step 2: Define your game state class
class MyGameState(pymcts.MCTS_state):
    def __init__(self):
        super().__init__()
        # Initialize your game state here
    
    def actions_to_try(self):
        # Return list of possible moves
        return [MyGameMove(move) for move in possible_moves]
    
    def next_state(self, move):
        # Return new state after applying move
        new_state = MyGameState()
        # Apply move logic...
        return new_state
    
    def rollout(self):
        # Simulate random game to completion
        # Return score: 1.0 (Player 1 wins), 0.0 (Player 2 wins), 0.5 (draw)
        return random_simulation_result
    
    # Optional: Enhanced heuristic rollout for smarter play
    def heuristic_rollout(self):
        # Simulate game using domain knowledge
        # Prioritize: Win > Block > Strategic positions > Random
        return smart_simulation_result
    
    def evaluate_move(self, move):
        # Optional: Evaluate move quality (0.0 to 1.0)
        # Higher scores for better moves
        return move_quality_score
    
    def evaluate_position(self):
        # Optional: Evaluate current position (0.0 to 1.0)
        # 1.0 = favorable for Player 1, 0.0 = favorable for Player 2
        return position_score
    
    def is_terminal(self):
        # Return True if game is over
        return game_over_check()
    
    def player1_turn(self):
        # Return True if it's Player 1's turn
        return self.current_player == 1

# Step 3: Use with MCTS
initial_state = MyGameState()
agent = pymcts.MCTSAgent(initial_state, max_iter=1000)
best_move = agent.genmove(None)
print(f"Best move: {best_move.sprint()}")
```

### C++ Usage (Advanced)
For maximum performance, implement games directly in C++:

```cpp
#include "mcts/include/mcts.h"

class MyMove : public MCTS_move { /* implement virtual methods */ };
class MyState : public MCTS_state { /* implement virtual methods */ };

int main() {
    auto initial_state = std::make_shared<MyState>();
    MCTS_agent agent(initial_state, 1000);
    auto best_move = agent.genmove(nullptr);
    return 0;
}
```

## 📁 Project Structure

```
MonteCarloTreeSearch/
├── 📂 mcts/                   # Core C++ MCTS implementation
│   ├── include/               # Header files (mcts.h, state.h, JobScheduler.h)
│   └── src/                   # Implementation files (.cpp)
├── 📂 examples/               # C++ example games (reference implementations)
│   ├── TicTacToe/            # Simple C++ TicTacToe (3x3 grid)
│   └── Quoridor/             # Complex C++ Quoridor (strategy game)
├── 📂 demo/                   # 🆕 Python game demonstrations
│   ├── connect_four_python.py    # Complete Connect Four (6x7 board)
│   ├── simple_python_games.py   # Learning examples (coin flip, etc.)
│   ├── demo_pymcts.py            # Basic PyMCTS usage guide
│   └── README.md                 # Demo documentation
├── 📂 tests/                  # 🆕 Comprehensive test suite
│   ├── test_core_minimal.py     # Core functionality tests (safe)
│   ├── test_parallel.py         # Multi-threading tests
│   ├── test_python_inheritance.py # Python game tests
│   ├── conftest.py              # PyTest fixtures
│   ├── pytest.ini              # Test configuration
│   └── README.md                # Testing documentation
├── 📂 build/                  # Compiled binaries (pymcts module)
├── 📂 docs/                   # Additional documentation
├── 🔧 setup.py               # Python build configuration
├── 🔧 Makefile               # C++ build configuration
└── 📖 README.md              # This file
```

## 🎮 Example Games & Demos

### 🐍 Python Examples (Pure Python Implementation) - **Start Here!**

#### 1. **Connect Four** (`demo/connect_four_python.py`)
- **Complete 6x7 board game** with win detection
- Visual board display and move validation
- Demonstrates complex game logic implementation
- Perfect example for learning advanced patterns

#### 2. **Simple Games** (`demo/simple_python_games.py`)
- **Coin Flip Game**: Minimal implementation for learning
- **Number Guessing Game**: Shows state transitions
- Great starting point for understanding the basics

#### 3. **PyMCTS Demo** (`demo/demo_pymcts.py`)
- Basic usage patterns and configuration
- Shows how to use built-in C++ TicTacToe
- Threading and performance examples

### ⚙️ C++ Examples (Built-in Reference)

#### 1. **TicTacToe** (`examples/TicTacToe/`)
- Simple 3x3 grid implementation
- Excellent for debugging and development
- Reference for C++ implementation patterns

#### 2. **Quoridor** (`examples/Quoridor/`)
- Complex strategy game with high branching factor
- Demonstrates advanced MCTS capabilities
- Showcases performance on difficult problems

### 🚀 How to Run Examples

```bash
# Python demos (from project root)
cd demo
python connect_four_python.py      # Play Connect Four
python simple_python_games.py      # Try simple games
python demo_pymcts.py              # Learn basic usage

# C++ examples (build required)
make all                           # Build C++ examples
./TicTacToe                        # Run C++ TicTacToe
./Quoridor                         # Run C++ Quoridor
```

## 🔧 Build Instructions

### Prerequisites
- **C++14 compatible compiler**: Visual Studio 2022, GCC 7+, or Clang 5+
- **Python 3.7+** with development headers
- **pybind11**: For Python bindings

### 🐍 Building PyMCTS (Recommended)

```bash
# Step 1: Install Python dependencies
pip install pybind11 pytest

# Step 2: Build the Python module
python setup.py build_ext --inplace

# Step 3: Verify installation
python tests/quick_test.py
```

**If build fails on Windows:**
```bash
# Ensure Visual Studio build tools are installed
# Try with specific Python version
py -3.11 setup.py build_ext --inplace
```

### ⚙️ Building C++ Examples

```bash
# Linux/macOS
make all

# Windows with Visual Studio
# Open Developer Command Prompt and run:
cl examples\TicTacToe\*.cpp mcts\src\*.cpp /I mcts\include /EHsc
```

### 🧪 Running Tests

```bash
# Run all working tests
python -m pytest tests/test_core_minimal.py tests/test_parallel.py tests/test_python_inheritance.py -v

# Quick functionality test
python tests/quick_test.py

# Specific test modules
python -m pytest tests/test_core_minimal.py -v    # Core functionality
python -m pytest tests/test_parallel.py -v       # Multi-threading
python -m pytest tests/test_python_inheritance.py -v  # Python games
```

### 🎯 Verification Steps

After successful build, you should be able to:

```python
import sys
sys.path.append('build')
import pymcts

# Check basic functionality
print(f"Hardware info: {pymcts.hardware_info()}")
print(f"Thread count: {pymcts.getNumThreads()}")

# Test with built-in TicTacToe
state = pymcts.cpp_TicTacToeState()
agent = pymcts.MCTSAgent(state, max_iter=100)
print("✅ PyMCTS is working correctly!")
```

## 🎯 **How to Implement a New Game**

This guide walks you through creating your own game using PyMCTS. We'll implement a simple "Number Battle" game as an example.

### 📋 **Step 1: Game Design**

First, define your game rules:
- **Number Battle**: Two players pick numbers 1-10
- **Winning**: Player with higher number wins
- **Ties**: Numbers are equal → draw
- **Goal**: Use MCTS to pick optimal numbers

### 🏗️ **Step 2: Project Setup**

Create your game file:

```bash
# Create your game file
touch my_number_battle.py

# Add the PyMCTS import path
```

```python
import sys
sys.path.append('build')  # Point to PyMCTS module
import pymcts
import random
```

### 🎮 **Step 3: Implement the Move Class**

```python
class NumberBattleMove(pymcts.MCTS_move):
    """Represents a player's number choice (1-10)"""
    
    def __init__(self, number):
        super().__init__()
        self.number = number
    
    def __eq__(self, other):
        """Required: Check if two moves are equal"""
        return isinstance(other, NumberBattleMove) and self.number == other.number
    
    def sprint(self):
        """Required: String representation of the move"""
        return f"Choose {self.number}"
```

### 🎲 **Step 4: Implement the State Class**

```python
class NumberBattleState(pymcts.MCTS_state):
    """Represents the current game state"""
    
    def __init__(self):
        super().__init__()
        self.player1_number = None  # Player 1's chosen number
        self.player2_number = None  # Player 2's chosen number
        self.current_player = 1     # Whose turn (1 or 2)
    
    def actions_to_try(self):
        """Required: Return list of possible moves"""
        if self.is_terminal():
            return []
        
        # Current player can choose any number 1-10
        return [NumberBattleMove(i) for i in range(1, 11)]
    
    def next_state(self, move):
        """Required: Return new state after applying move"""
        new_state = NumberBattleState()
        new_state.player1_number = self.player1_number
        new_state.player2_number = self.player2_number
        new_state.current_player = self.current_player
        
        # Apply the move
        if self.current_player == 1:
            new_state.player1_number = move.number
            new_state.current_player = 2
        else:
            new_state.player2_number = move.number
            # Game ends after both players move
        
        return new_state
    
    def rollout(self):
        """Required: Simulate random game to completion"""
        state = self
        
        # If game not finished, simulate random moves
        if state.player1_number is None:
            state = state.next_state(NumberBattleMove(random.randint(1, 10)))
        if state.player2_number is None:
            state = state.next_state(NumberBattleMove(random.randint(1, 10)))
        
        # Determine winner and return score
        if state.player1_number > state.player2_number:
            return 1.0  # Player 1 wins
        elif state.player1_number < state.player2_number:
            return 0.0  # Player 2 wins
        else:
            return 0.5  # Draw
    
    def is_terminal(self):
        """Required: Check if game is over"""
        return self.player1_number is not None and self.player2_number is not None
    
    def player1_turn(self):
        """Required: Check whose turn it is"""
        return self.current_player == 1
    
    def print(self):
        """Optional: Display current state"""
        p1 = self.player1_number if self.player1_number else "?"
        p2 = self.player2_number if self.player2_number else "?"
        print(f"Player 1: {p1}, Player 2: {p2}, Turn: Player {self.current_player}")
```

### 🤖 **Step 5: Create the Game Loop**

```python
def play_number_battle():
    """Main game function"""
    print("🎯 Number Battle Game!")
    print("Each player picks a number 1-10. Highest number wins!")
    
    # Create initial game state
    state = NumberBattleState()
    
    # Create MCTS agents
    player1_agent = pymcts.MCTSAgent(state, max_iter=1000)
    player2_agent = pymcts.MCTSAgent(state, max_iter=1000)
    
    print("\nGame starts!")
    state.print()
    
    # Game loop
    while not state.is_terminal():
        if state.player1_turn():
            print("\n🤖 Player 1 (MCTS) is thinking...")
            move = player1_agent.genmove(state)
            print(f"Player 1 chose: {move.sprint()}")
        else:
            print("\n🤖 Player 2 (MCTS) is thinking...")
            move = player2_agent.genmove(state)
            print(f"Player 2 chose: {move.sprint()}")
        
        # Apply move and update state
        state = state.next_state(move)
        state.print()
    
    # Game over - announce winner
    print("\n🏁 Game Over!")
    if state.player1_number > state.player2_number:
        print("🏆 Player 1 wins!")
    elif state.player1_number < state.player2_number:
        print("🏆 Player 2 wins!")
    else:
        print("🤝 It's a draw!")

if __name__ == "__main__":
    play_number_battle()
```

### 🚀 **Step 6: Run Your Game**

```bash
python my_number_battle.py
```

Expected output:
```
🎯 Number Battle Game!
Each player picks a number 1-10. Highest number wins!

Game starts!
Player 1: ?, Player 2: ?, Turn: Player 1

🤖 Player 1 (MCTS) is thinking...
Player 1 chose: Choose 8
Player 1: 8, Player 2: ?, Turn: Player 2

🤖 Player 2 (MCTS) is thinking...
Player 2 chose: Choose 9
Player 1: 8, Player 2: 9, Turn: Player 2

🏁 Game Over!
🏆 Player 2 wins!
```

### 🎯 **Implementation Checklist**

When implementing your own game, ensure you have:

#### ✅ **Move Class (`MCTS_move`)**
- [ ] `__init__(self, ...)` - Constructor with move data
- [ ] `__eq__(self, other)` - Equality comparison
- [ ] `sprint(self)` - String representation

#### ✅ **State Class (`MCTS_state`)**
- [ ] `__init__(self)` - Constructor with game state
- [ ] `actions_to_try(self)` - List of possible moves
- [ ] `next_state(self, move)` - New state after move
- [ ] `rollout(self)` - Random simulation (return 0.0-1.0)
- [ ] `is_terminal(self)` - Game over check
- [ ] `player1_turn(self)` - Current player check
- [ ] `print(self)` - Optional: state visualization

#### ✅ **Game Logic**
- [ ] Clear win/lose/draw conditions
- [ ] Proper state transitions
- [ ] Valid move generation
- [ ] Correct player turn handling

### 💡 **Advanced Tips**

#### **Performance Optimization**
```python
# Use threading for better performance
pymcts.setNumThreads(4)  # Use 4 CPU cores

# Adjust MCTS iterations based on game complexity
agent = pymcts.MCTSAgent(state, max_iter=5000)  # More iterations = stronger play
```

#### **Debugging Your Implementation**
```python
# Add debug prints to understand MCTS behavior
def rollout(self):
    print(f"Rollout called for state: {self}")
    result = self.simulate_to_end()
    print(f"Rollout result: {result}")
    return result
```

#### **Common Patterns**

**Board Games:**
```python
class BoardGameState(pymcts.MCTS_state):
    def __init__(self, board=None, player=1):
        super().__init__()
        self.board = board or self.create_empty_board()
        self.current_player = player
```

**Card Games:**
```python
class CardGameState(pymcts.MCTS_state):
    def __init__(self):
        super().__init__()
        self.deck = self.create_shuffled_deck()
        self.player_hands = [[], []]
        self.table_cards = []
```

**Abstract Strategy Games:**
```python
class AbstractGameState(pymcts.MCTS_state):
    def __init__(self):
        super().__init__()
        self.position = self.initial_position()
        self.move_history = []
```

### 🧠 **Enhanced Rollouts with Heuristics**

Beyond random simulation, you can implement intelligent rollouts using domain knowledge:

#### **Heuristic Rollout Interface**
```python
class SmartGameState(pymcts.MCTS_state):
    def heuristic_rollout(self):
        """Smart simulation using game knowledge (optional)"""
        while not self.is_terminal():
            # Use evaluate_move() to pick good moves
            best_move = self.choose_smart_move()
            self = self.next_state(best_move)
        return self.get_reward()
    
    def evaluate_move(self, move):
        """Rate move quality: 0.0 (bad) to 1.0 (excellent)"""
        # Implement your move evaluation logic
        return self.calculate_move_value(move)
    
    def evaluate_position(self):
        """Rate current position: 0.0 (losing) to 1.0 (winning)"""
        # Implement your position evaluation
        return self.calculate_position_strength()
```

#### **TicTacToe Heuristic Example**
```python
class TicTacToeHeuristic(pymcts.MCTS_state):
    def heuristic_rollout(self):
        while not self.is_terminal():
            moves = self.actions_to_try()
            
            # 1. Try to win immediately
            for move in moves:
                if self.creates_win(move):
                    self = self.next_state(move)
                    break
            else:
                # 2. Block opponent wins
                for move in moves:
                    if self.blocks_opponent_win(move):
                        self = self.next_state(move)
                        break
                else:
                    # 3. Take center, then corners, then edges
                    preferred = self.prioritize_moves(moves)
                    self = self.next_state(preferred[0])
        
        return 1.0 if self.player1_wins() else 0.0
    
    def evaluate_move(self, move):
        """Rate TicTacToe moves by strategic value"""
        if self.creates_win(move): return 1.0
        if self.blocks_opponent_win(move): return 0.9
        if self.is_center(move): return 0.7
        if self.is_corner(move): return 0.6
        return 0.3  # Edge positions
```

#### **Configure Rollout Strategy**
```python
# Set global rollout strategy (C++ level)
pymcts.setRolloutStrategy('HEURISTIC')  # Pure heuristic
pymcts.setRolloutStrategy('MIXED')      # 70% heuristic, 30% random
pymcts.setRolloutStrategy('HEAVY')      # 90% heuristic, 10% random
pymcts.setRolloutStrategy('RANDOM')     # Default: pure random

# Use in your MCTS agent
agent = pymcts.MCTSAgent(smart_state, max_iter=1000)
best_move = agent.get_action()  # Uses configured rollout strategy
```

### 🔗 **Next Steps**

1. **Study Examples**: Look at `demo/connect_four_python.py` for a complex implementation
2. **Run Tests**: Use `python -m pytest tests/` to validate your implementation
3. **Experiment**: Try different MCTS parameters and game variations
4. **Share**: Contribute your game back to the community!

### 🆘 **Troubleshooting**

**Common Issues:**

1. **Import Error**: Make sure `build/` directory exists and contains `pymcts` module
2. **Infinite Loops**: Check that `is_terminal()` correctly detects game end
3. **Crashes**: Ensure `rollout()` always returns a float between 0.0 and 1.0
4. **Poor AI**: Increase `max_iter` parameter for stronger play

**Getting Help:**
- Check `tests/` folder for working examples
- Review `demo/` folder for complete implementations
- Look at error messages carefully - they often indicate the exact issue

## ⚡ Performance Features & Configuration

### 🚀 **Multi-threading Support**
```python
import pymcts

# Check available CPU cores
print(f"Hardware info: {pymcts.hardware_info()}")

# Configure thread count (default: auto-detect CPU cores)
pymcts.setNumThreads(4)              # Use 4 threads for rollouts
current_threads = pymcts.getNumThreads()
print(f"Using {current_threads} threads")
```

### 📊 **Performance Tuning**

#### **MCTS Iterations**
```python
# Quick decisions (fast, less accurate)
agent = pymcts.MCTSAgent(state, max_iter=100)

# Balanced performance (recommended)
agent = pymcts.MCTSAgent(state, max_iter=1000)

# Strong play (slow, high accuracy)
agent = pymcts.MCTSAgent(state, max_iter=10000)
```

#### **Memory Management**
- **Smart Pointers**: Automatic C++ object cleanup
- **Python GC Integration**: Proper memory management across languages
- **Efficient Tree Storage**: Optimized node allocation and deallocation

### 🎯 **Platform Support**
- ✅ **Windows**: Visual Studio 2022, MinGW
- ✅ **Linux**: GCC 7+, Clang 5+
- ✅ **macOS**: Xcode 10+, Homebrew GCC
- ✅ **Python**: 3.7, 3.8, 3.9, 3.10, 3.11+

### 📈 **Benchmarking**

```python
import time
import pymcts

def benchmark_game(state, iterations):
    """Benchmark MCTS performance"""
    start_time = time.time()
    
    agent = pymcts.MCTSAgent(state, max_iter=iterations)
    move = agent.genmove(state)
    
    end_time = time.time()
    print(f"Iterations: {iterations}")
    print(f"Time: {end_time - start_time:.3f}s")
    print(f"Rate: {iterations / (end_time - start_time):.0f} iter/sec")
    
    return move
```


## 🧠 Implementation Details & Algorithm Overview

### 🎯 **MCTS Algorithm Fundamentals**

The MCTS algorithm selectively builds a game tree where each node stores:
1. **Game State**: The current position/configuration
2. **Move History**: The move that led to this state from parent
3. **Visit Count**: Number of simulations performed from this node
4. **Score Sum**: Accumulated scores from all simulations
5. **Win Rate**: Score sum divided by visit count (estimated winning probability)

**Score Convention**: Results must be in range [0, 1]
- `1.0` = Player 1 wins
- `0.0` = Player 2 wins  
- `0.5` = Draw/tie

### 🔄 **MCTS Four Phases**

#### 1. **Selection** 🎯
- Traverse tree using **UCT (Upper Confidence Trees)** policy
- Balances *exploration* (trying new moves) vs *exploitation* (using known good moves)
- Formula: `UCT = win_rate + C * sqrt(ln(parent_visits) / child_visits)`
- Selects promising node that isn't fully expanded

#### 2. **Expansion** 🌱
- Add a new child node to the selected node
- Choose an untried move from the selected state
- Create new node representing the resulting game state

#### 3. **Simulation/Rollout** 🎲
- Simulate random game from new node to completion
- Can use domain knowledge or pure random play
- Returns final game result (0.0, 0.5, or 1.0)

#### 4. **Backpropagation** ⬆️
- Update all ancestor nodes with simulation result
- Increment visit counts and add score to sum
- Continues recursively up to root node

![MCTS phases from wikipedia](https://i.stack.imgur.com/wZAqy.png "MCTS phases")

### 🔁 **Algorithm Loop**

```python
# Pseudo-code for MCTS
def mcts_search(root_state, iterations):
    for i in range(iterations):
        # 1. Selection: Find promising leaf node
        node = select_promising_node(root_state)
        
        # 2. Expansion: Add child if not terminal
        if not node.is_terminal():
            node = expand_node(node)
        
        # 3. Simulation: Random rollout
        result = simulate_random_game(node.state)
        
        # 4. Backpropagation: Update ancestors
        backpropagate_result(node, result)
    
    # Return best move based on visit counts or win rates
    return select_best_child(root_state)
```

### 🧵 **Parallel Rollouts Architecture**

This implementation uses **thread pools** for embarrassingly parallel rollout execution:

#### **JobScheduler Design**
- **Thread Pool**: Pre-allocated worker threads (avoids thread creation overhead)
- **Task Queue**: Thread-safe job distribution using mutexes and condition variables
- **POSIX Threads**: Cross-platform threading with pthreads
- **Configurable**: 1-N threads based on CPU cores

#### **Parallel Execution Flow**
```cpp
// C++ parallel rollout pseudo-code
class JobScheduler {
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex task_mutex;
    std::condition_variable condition;
    
    void worker_thread() {
        while (running) {
            // Wait for task
            std::unique_lock<std::mutex> lock(task_mutex);
            condition.wait(lock, [this] { return !tasks.empty() || !running; });
            
            // Execute task
            if (!tasks.empty()) {
                auto task = tasks.front();
                tasks.pop();
                lock.unlock();
                task();  // Execute rollout
            }
        }
    }
};
```

#### **Thread Safety**
- **Independent Rollouts**: Each simulation is completely independent
- **No Shared State**: Rollouts don't modify the search tree during execution
- **Atomic Updates**: Final results are safely aggregated
- **Memory Isolation**: Each thread works with separate game state copies

### 🏗️ **Architecture Overview**

```
┌─────────────────────────────────────────────────────────────┐
│                     Python Layer                           │
├─────────────────────────────────────────────────────────────┤
│  Your Game (MyGameState, MyGameMove)                       │
│  └─ Inherits from pymcts.MCTS_state, pymcts.MCTS_move      │
├─────────────────────────────────────────────────────────────┤
│                   PyBind11 Bindings                        │
│  └─ Python ↔ C++ interface                                 │
├─────────────────────────────────────────────────────────────┤
│                    C++ MCTS Core                           │
│  ├─ MCTS_agent: Main algorithm controller                  │
│  ├─ MCTS_node: Tree node with UCT selection               │
│  ├─ MCTS_state: Abstract state interface                  │
│  └─ MCTS_move: Abstract move interface                    │
├─────────────────────────────────────────────────────────────┤
│                   JobScheduler                             │
│  └─ Thread pool for parallel rollouts                     │
└─────────────────────────────────────────────────────────────┘
```

### 💾 **Memory Management**

#### **C++ Side**
- **Smart Pointers**: `std::shared_ptr` for automatic cleanup
- **RAII**: Resource Acquisition Is Initialization pattern
- **Tree Cleanup**: Automatic node deallocation when tree is destroyed

#### **Python Side**
- **Reference Counting**: Python manages object lifetimes
- **PyBind11 Integration**: Automatic conversion between Python/C++ objects
- **Garbage Collection**: Python GC handles cleanup of unused objects

#### **Cross-Language**
- **Object Ownership**: Clear ownership rules between Python and C++
- **Exception Safety**: Proper cleanup even when exceptions occur
- **Memory Leaks**: Extensive testing to prevent leaks

### 🔧 **Customization Points**

#### **UCT Parameters**
```cpp
// Exploration constant (higher = more exploration)
const double UCT_C = sqrt(2.0);  // Theoretical optimum

// Custom UCT calculation
double custom_uct_value(Node* node, Node* parent) {
    double exploitation = node->win_rate();
    double exploration = UCT_C * sqrt(log(parent->visits) / node->visits);
    return exploitation + exploration;
}
```

#### **Rollout Policies**
```python
def smart_rollout(self):
    """Custom rollout with domain knowledge"""
    state = self
    while not state.is_terminal():
        moves = state.actions_to_try()
        
        # Use heuristics instead of pure random
        if self.has_winning_move(moves):
            move = self.get_winning_move(moves)
        elif self.has_blocking_move(moves):
            move = self.get_blocking_move(moves)
        else:
            move = random.choice(moves)
        
        state = state.next_state(move)
    
    return state.get_result()
```

### 📊 **Performance Characteristics**

#### **Time Complexity**
- **Per Iteration**: O(log n) for selection + O(1) for expansion + O(d) for rollout + O(log n) for backprop
- **Total**: O(iterations × (log n + rollout_depth))
- **Parallel**: Linear speedup with number of cores (for rollout phase)

#### **Space Complexity**
- **Tree Size**: O(iterations) nodes maximum
- **Memory per Node**: ~64-128 bytes (pointers, counters, state reference)
- **Peak Usage**: Depends on branching factor and game complexity

### 📊 **Performance Characteristics**

#### **Time Complexity**
- **Per Iteration**: O(log n) for selection + O(1) for expansion + O(d) for rollout + O(log n) for backprop
- **Total**: O(iterations × (log n + rollout_depth))
- **Parallel**: Linear speedup with number of cores (for rollout phase)

#### **Space Complexity**
- **Tree Size**: O(iterations) nodes maximum
- **Memory per Node**: ~64-128 bytes (pointers, counters, state reference)
- **Peak Usage**: Depends on branching factor and game complexity

#### **Scaling Factors**
- **Branching Factor**: Higher = more memory, slightly slower selection
- **Game Depth**: Affects rollout time
- **State Complexity**: Impacts memory usage and copy operations
- **Thread Count**: Scales rollout performance up to CPU core count

## 🧪 Testing & Validation

### 🔬 **Test Suite**
Our comprehensive test suite ensures code quality and reliability:

```bash
# Run all tests
python -m pytest tests/ -v

# Specific test categories
python -m pytest tests/test_core_minimal.py -v       # Core functionality
python -m pytest tests/test_parallel.py -v          # Multi-threading
python -m pytest tests/test_python_inheritance.py -v # Python game implementations
```

### 📋 **Test Coverage**
- ✅ **Core MCTS**: Algorithm correctness, tree building, UCT selection
- ✅ **Multi-threading**: Parallel rollouts, thread safety, performance
- ✅ **Python Bindings**: Interface correctness, memory management
- ✅ **Game Implementations**: Complete Python game examples
- ✅ **Platform Compatibility**: Windows, Linux, macOS testing

### 🐛 **Known Issues**
- **C++ Agent Creation**: Some Windows configurations may experience memory issues with certain C++ MCTS agent patterns (workaround: use minimal test suite)
- **Thread Count**: Performance may degrade with excessive thread counts (recommendation: use CPU core count)

### ⚠️ **Breaking Changes**

**IMPORTANT**: The `player1_turn()` method has been removed. All code must be updated to use the new interface:

```cpp
// ❌ REMOVED (will cause compilation errors)
bool player1_turn() const override;

// ✅ REQUIRED (new interface)
bool is_self_side_turn() const override;
```

**Migration Required:**
- **All existing code** must be updated to use the new interface
- **Compilation will fail** until migration is complete
- **No backward compatibility** - breaking change for cleaner API

**Migration Steps:**
1. Replace `player1_turn()` with `is_self_side_turn()`
2. Update rollout methods to return self-side win probability
3. See [`MULTI_PLAYER_INTERFACE.md`](MULTI_PLAYER_INTERFACE.md) for examples

## 📚 Resources & Learning

### 📖 **Documentation**
- **API Reference**: See `mcts/include/` for C++ headers with documentation
- **Python Examples**: Complete implementations in `demo/` folder
- **Test Examples**: Working code patterns in `tests/` folder

### 🎓 **Learning Path**
1. **Beginner**: Start with `demo/demo_pymcts.py` and `demo/simple_python_games.py`
2. **Intermediate**: Study `demo/connect_four_python.py` for complex game implementation
3. **Advanced**: Explore C++ examples in `examples/` and modify MCTS parameters
4. **Expert**: Contribute new games, optimize performance, extend C++ core

### 🤝 **Contributing**
We welcome contributions! Areas of interest:
- **New Game Implementations**: Add your favorite games to `demo/`
- **Performance Improvements**: Optimize C++ core or Python bindings
- **Platform Support**: Test and improve compatibility
- **Documentation**: Improve guides, add examples, fix typos

### 💬 **Community**
- **Issues**: Report bugs and request features via GitHub issues
- **Discussions**: Share your games and get help from the community
- **Examples**: Submit your game implementations for others to learn from

## 📄 References & Citations

### 📚 **Academic References**

1. **Coulom, R.** (2006). *Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search*. Computers and Games, 5th International Conference, CG 2006.

2. **Kocsis, L., & Szepesvári, C.** (2006). *Bandit based Monte-Carlo Planning*. 17th European Conference on Machine Learning (ECML-06).

3. **Browne, C., et al.** (2012). *A Survey of Monte Carlo Tree Search Methods*. IEEE Transactions on Computational Intelligence and AI in Games, 4(1), 1-43.

4. **Max Magnuson.** (2015). *Monte Carlo Tree Search and Its Applications*. https://digitalcommons.morris.umn.edu/horizons/vol2/iss2/4/

5. **Massagué Respall, Victor & Brown, Joseph & Aslam, Hamna.** (2018). *Monte Carlo Tree Search for Quoridor*. https://www.researchgate.net/publication/327679826_Monte_Carlo_Tree_Search_for_Quoridor

### 🔧 **Technical References**

- **PyBind11**: https://pybind11.readthedocs.io/ - Python/C++ binding library
- **UCT Algorithm**: Upper Confidence bounds applied to Trees
- **POSIX Threads**: Multi-threading implementation for parallel rollouts

### 🎮 **Game Theory & AI**

- **Alpha-Beta Pruning**: Traditional minimax enhancement for comparison
- **Game Tree Search**: Classical algorithms that MCTS often outperforms
- **Reinforcement Learning**: MCTS as a component in modern RL systems

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏷️ Version

**Current Version**: 2.0.0 (Major update with Python game support, test suite, and improved documentation)

**Last Updated**: October 2025

---

*Made with ❤️ for the AI and game development community. Happy coding! 🎮🤖*
