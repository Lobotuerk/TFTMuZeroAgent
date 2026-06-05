"""
Pytest configuration and shared fixtures for PyMCTS tests.
"""
import sys
import os
import pytest

# Add the parent directory to sys.path to import pymcts
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import pymcts
except ImportError as e:
    pytest.skip(f"pymcts module not available: {e}", allow_module_level=True)

@pytest.fixture(scope="session")
def pymcts_module():
    """Provide the pymcts module to all tests."""
    return pymcts

@pytest.fixture
def thread_configs():
    """Common thread configurations for parallel testing."""
    return [1, 2, 4, min(8, pymcts.get_hardware_concurrency())]

@pytest.fixture(autouse=True)
def reset_thread_config():
    """Reset thread configuration before each test."""
    original_threads = pymcts.get_rollout_threads()
    # Set to single thread for safety
    pymcts.set_rollout_threads(1)
    yield
    # Restore original configuration
    pymcts.set_rollout_threads(original_threads)

# Simple test classes for Python inheritance testing
class SimpleMove(pymcts.MCTS_move):
    def __init__(self, value):
        super().__init__()
        self.value = value
    
    def __eq__(self, other):
        return isinstance(other, SimpleMove) and self.value == other.value
    
    def sprint(self):
        return f"Move({self.value})"

class SimpleState(pymcts.MCTS_state):
    def __init__(self, turn=0, moves_made=0):
        super().__init__()
        self.turn = turn
        self.moves_made = moves_made
    
    def actions_to_try(self):
        if self.is_terminal():
            return []
        return [SimpleMove(0), SimpleMove(1)]
    
    def next_state(self, move):
        return SimpleState((self.turn + 1) % 2, self.moves_made + 1)
    
    def rollout(self):
        return 0.5
    
    def is_terminal(self):
        return self.moves_made >= 3
    
    def is_self_side_turn(self):
        return self.turn == 0
    
    def clone(self):
        """Create a deep copy of this state for C++ ownership."""
        return SimpleState(self.turn, self.moves_made)
    
    def print(self):
        pass

@pytest.fixture
def simple_python_state():
    """Create a simple Python game state for testing inheritance."""
    return SimpleState()

@pytest.fixture
def simple_python_move():
    """Create a simple Python move for testing."""
    return SimpleMove(0)

@pytest.fixture
def tictactoe_state():
    """Create a TicTacToe state for testing."""
    return pymcts.TicTacToe_state()

@pytest.fixture
def mcts_agent_factory(pymcts_module):
    """Factory for creating MCTS agents with proper cleanup."""
    agents = []
    
    def create_agent(state, max_iter=100, max_seconds=2):
        """Create an MCTS agent and register it for cleanup."""
        agent = pymcts_module.MCTS_agent(state, max_iter, max_seconds)
        agents.append(agent)
        return agent
    
    yield create_agent
    
    # Cleanup all agents when fixture scope ends
    for agent in agents:
        try:
            del agent
        except:
            pass  # Ignore cleanup errors
    agents.clear()