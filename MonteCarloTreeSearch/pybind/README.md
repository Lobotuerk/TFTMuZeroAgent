# PyMCTS - Python Bindings for Monte Carlo Tree Search

This directory contains Python bindings for the MCTS C++ library using pybind11.

## Requirements

- Python 3.6 or higher
- pybind11 (will be installed automatically)
- C++11 compatible compiler
- CMake (optional, for CMake-based build)

## Installation

### Method 1: Using setup.py (Recommended)

```bash
# Install pybind11 if not already installed
pip install pybind11

# Build and install the module
python setup.py build_ext --inplace

# Or install system-wide
pip install .
```

### Method 2: Using CMake

```bash
mkdir build
cd build
cmake ..
make
```

## Usage

### Basic TicTacToe Example

```python
import pymcts

# Create a new TicTacToe game
state = pymcts.TicTacToe_state()
state.print()

# Create an MCTS agent
agent = pymcts.MCTS_agent(state, max_iter=1000, max_seconds=5)

# Generate a move
move = agent.genmove(None)
print(f"Agent chose: {move}")

# Get current state
current_state = agent.get_current_state()
current_state.print()
```

### Creating Custom Games

To create your own game, inherit from `MCTS_move` and `MCTS_state`:

```python
class MyGameMove(pymcts.MCTS_move):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y
    
    def __eq__(self, other):
        return isinstance(other, MyGameMove) and self.x == other.x and self.y == other.y
    
    def sprint(self):
        return f"Move({self.x}, {self.y})"

class MyGameState(pymcts.MCTS_state):
    def __init__(self):
        super().__init__()
        # Initialize your game state
    
    def actions_to_try(self):
        # Return list of possible moves
        return [MyGameMove(x, y) for x, y in self.get_valid_positions()]
    
    def next_state(self, move):
        # Create and return new state after applying move
        new_state = MyGameState()
        # Apply move logic
        return new_state
    
    def rollout(self):
        # Return win probability for player 1 (0.0 to 1.0)
        return self.simulate_random_game()
    
    def is_terminal(self):
        # Return True if game is over
        return self.check_game_over()
    
    def player1_turn(self):
        # Return True if it's player 1's turn
        return self.current_player == 1
    
    def print(self):
        # Print current state
        print("Current game state...")
```

## API Reference

### Core Classes

#### `MCTS_move` (Abstract Base Class)
- `__eq__(other)`: Equality comparison
- `sprint()`: String representation

#### `MCTS_state` (Abstract Base Class)
- `actions_to_try()`: Returns list of possible moves
- `next_state(move)`: Returns new state after applying move
- `rollout()`: Returns win probability for player 1 (0.0-1.0)
- `is_terminal()`: Returns True if game is over
- `player1_turn()`: Returns True if it's player 1's turn
- `print()`: Print the current state

#### `MCTS_agent` (High-level Interface)
- `__init__(starting_state, max_iter=100000, max_seconds=30)`
- `genmove(enemy_move=None)`: Generate next move
- `get_current_state()`: Get current game state
- `feedback()`: Print thinking statistics

#### `MCTS_tree` (Low-level Interface)
- `__init__(starting_state)`
- `grow_tree(max_iter, max_time_in_seconds)`: Expand the search tree
- `select_best_child()`: Get the best move
- `advance_tree(move)`: Apply a move to the tree
- `get_size()`: Get tree size
- `print_stats()`: Print tree statistics

### TicTacToe Classes

#### `TicTacToe_move`
- `__init__(x, y, player)`: Create move at position (x,y) for player
- `x`, `y`: Coordinates (0-2)
- `player`: Player character ('x' or 'o')

#### `TicTacToe_state`
- `__init__()`: Create new game
- `get_turn()`: Get current player ('x' or 'o')
- `get_winner()`: Get winner ('x', 'o', 'd' for draw, ' ' for ongoing)

## Testing

Run the test script to verify everything works:

```bash
python test_pymcts.py
```

## Memory Management

The Python bindings handle memory management automatically:
- States and moves are properly cleaned up
- C++ objects are wrapped with appropriate lifetime management
- No manual memory management required from Python side

## Performance Notes

- The bindings disable parallel rollouts by default for simplicity
- For maximum performance, consider using the C++ library directly
- Python overhead is minimal for the tree search algorithm itself
- Most computation happens in C++, so performance is very good

## Troubleshooting

### Build Issues

1. **pybind11 not found**: Install with `pip install pybind11`
2. **Compiler errors**: Ensure you have a C++11 compatible compiler
3. **Threading issues**: The bindings disable parallel rollouts by default

### Runtime Issues

1. **Import errors**: Make sure the module was built correctly and is in Python path
2. **Segmentation faults**: Usually indicate incorrect virtual method implementation

## Examples

See `test_pymcts.py` for complete working examples including:
- Basic TicTacToe usage
- MCTS agent gameplay
- Custom game implementation framework