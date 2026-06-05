# 🎮 PyMCTS Demo Games

This folder contains complete game implementations written in pure Python that demonstrate how to use the PyMCTS library.

## 📁 Files

### `connect_four_python.py`
- **Complete Connect Four implementation** 
- Shows how to implement a complex board game in Python
- Demonstrates proper inheritance from `MCTS_move` and `MCTS_state`
- Features:
  - 6x7 game board
  - Win detection (4 in a row)
  - Move validation
  - Game state visualization

### `simple_python_games.py`
- **Simple game examples** for learning
- Contains multiple mini-games:
  - Coin flip game
  - Number guessing game
- Shows minimal implementation patterns
- Great for understanding the basic structure

### `demo_pymcts.py`
- **Basic usage demonstration**
- Shows how to use the built-in C++ TicTacToe
- Demonstrates MCTS agent configuration
- Parallel rollout examples

## 🚀 How to Run

```bash
# From the root directory
cd demo

# Run Connect Four demo
python connect_four_python.py

# Run simple games
python simple_python_games.py

# Run basic PyMCTS demo
python demo_pymcts.py
```

## 📚 Learning Path

1. **Start with** `demo_pymcts.py` - Learn basic PyMCTS usage
2. **Study** `simple_python_games.py` - Understand the Python implementation patterns
3. **Explore** `connect_four_python.py` - See a complete, complex game implementation

## 🎯 Implementation Template

All Python games follow this pattern:

```python
import sys
sys.path.append('../build')
import pymcts

class MyGameMove(pymcts.MCTS_move):
    def __eq__(self, other): ...
    def sprint(self): ...

class MyGameState(pymcts.MCTS_state):
    def actions_to_try(self): ...
    def next_state(self, move): ...
    def rollout(self): ...
    def is_terminal(self): ...
    def player1_turn(self): ...
    def print(self): ...  # optional

# Usage
initial_state = MyGameState()
agent = pymcts.MCTS_agent(initial_state, max_iter=1000)
best_move = agent.genmove(None)
```

## 💡 Tips

- **Parallel Rollouts**: Use `pymcts.set_rollout_threads(4)` for better performance
- **Debug Mode**: Add print statements in your methods to understand the flow
- **Quick Testing**: Start with small `max_iter` values during development
- **Game Balance**: Adjust rollout logic to create interesting gameplay

## 🔗 Related Files

- `../tests/` - Test files for validation
- `../PYTHON_GAMES_GUIDE.md` - Comprehensive implementation guide
- `../examples/TicTacToe/` - C++ reference implementation