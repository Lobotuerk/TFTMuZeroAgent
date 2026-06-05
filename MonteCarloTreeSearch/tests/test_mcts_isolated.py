"""
Working MCTS agent tests using subprocess isolation.
This approach runs MCTS tests in separate processes to avoid C++ destructor issues.
"""
import pytest
import subprocess
import sys
import os
import tempfile
import time


class TestMCTSAgentIsolated:
    """Test MCTS agent functionality in isolated subprocesses."""
    
    def test_single_agent_basic(self, pymcts_module):
        """Test basic MCTS agent functionality in subprocess."""
        test_script = '''
import sys
import os
# Add the MonteCarloTreeSearch directory to Python path
sys.path.insert(0, os.environ.get('MCTS_DIR', '.'))
import pymcts

# Test basic MCTS agent functionality
state = pymcts.TicTacToe_state()
agent = pymcts.MCTS_agent(state, 20, 1)
move = agent.genmove(None)

print(f"SUCCESS: Generated move {move.sprint()}")
exit(0)
'''
        
        # Write test script to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_script)
            script_path = f.name
        
        try:
            # Run in subprocess with environment variable
            env = os.environ.copy()
            env['MCTS_DIR'] = os.path.dirname(os.path.dirname(__file__))
            
            result = subprocess.run([
                sys.executable, script_path
            ], capture_output=True, text=True, timeout=10, env=env)
            
            assert result.returncode == 0, f"Test failed: {result.stderr}"
            assert "SUCCESS" in result.stdout
            print(f"Subprocess output: {result.stdout.strip()}")
            
        finally:
            os.unlink(script_path)
    
    def test_parallel_rollouts_basic(self, pymcts_module):
        """Test parallel rollouts in subprocess."""
        test_script = '''
import sys
import os
# Add the MonteCarloTreeSearch directory to Python path
sys.path.insert(0, os.environ.get('MCTS_DIR', '.'))
import pymcts
import time

# Test parallel vs single thread
pymcts.set_rollout_threads(1)
state1 = pymcts.TicTacToe_state()
agent1 = pymcts.MCTS_agent(state1, 30, 1)

start = time.time()
move1 = agent1.genmove(None)
single_time = time.time() - start

# Change to multi-thread
pymcts.set_rollout_threads(2)
state2 = pymcts.TicTacToe_state()
agent2 = pymcts.MCTS_agent(state2, 30, 1)

start = time.time()
move2 = agent2.genmove(None)
multi_time = time.time() - start

print(f"SUCCESS: Single: {single_time:.3f}s, Multi: {multi_time:.3f}s")
print(f"Move1: {move1.sprint()}, Move2: {move2.sprint()}")
exit(0)
'''
        
        # Write test script to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_script)
            script_path = f.name
        
        try:
            # Run in subprocess with environment variable
            env = os.environ.copy()
            env['MCTS_DIR'] = os.path.dirname(os.path.dirname(__file__))
            
            result = subprocess.run([
                sys.executable, script_path
            ], capture_output=True, text=True, timeout=15, env=env)
            
            assert result.returncode == 0, f"Test failed: {result.stderr}"
            assert "SUCCESS" in result.stdout
            print(f"Subprocess output: {result.stdout.strip()}")
            
        finally:
            os.unlink(script_path)
    
    def test_python_inheritance_with_mcts(self, pymcts_module):
        """Test Python inheritance with MCTS in subprocess."""
        test_script = '''
import sys
import os
# Add the MonteCarloTreeSearch directory to Python path
sys.path.insert(0, os.environ.get('MCTS_DIR', '.'))
import pymcts

# Define Python classes
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
        return True
    
    def clone(self):
        return SimpleState(self.turn, self.moves_made)
    

# Test Python classes with MCTS using SerializedPythonState wrapper
state = SimpleState()
wrapped_state = pymcts.SerializedPythonState(state)
agent = pymcts.MCTS_agent(wrapped_state, 10, 1)
move = agent.genmove(None)

print(f"SUCCESS: Python inheritance works! Move: {move.sprint()}")
exit(0)
'''
        
        # Write test script to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_script)
            script_path = f.name
        
        try:
            # Run in subprocess with environment variable
            env = os.environ.copy()
            env['MCTS_DIR'] = os.path.dirname(os.path.dirname(__file__))
            
            result = subprocess.run([
                sys.executable, script_path
            ], capture_output=True, text=True, timeout=10, env=env)
            
            assert result.returncode == 0, f"Test failed: {result.stderr}"
            assert "SUCCESS" in result.stdout
            print(f"Subprocess output: {result.stdout.strip()}")
            
        finally:
            os.unlink(script_path)


class TestMCTSAgentDirectLimited:
    """Limited direct tests that avoid destructor issues."""
    
    def test_agent_creation_only(self, pymcts_module):
        """Test that agent can be created without calling genmove."""
        state = pymcts_module.TicTacToe_state()
        
        # Just create agent - don't call genmove or store reference
        agent = pymcts_module.MCTS_agent(state, 5, 1)
        
        # Test that we can access basic properties
        current_state = agent.get_current_state()
        assert current_state is not None
        
        # Don't store agent reference or call del - let Python handle cleanup
        print("Agent creation test passed")
    
    def test_single_move_generation(self, pymcts_module):
        """Test single move generation without cleanup issues."""
        state = pymcts_module.TicTacToe_state()
        
        # Create agent and generate one move
        agent = pymcts_module.MCTS_agent(state, 10, 1)
        move = agent.genmove(None)
        
        assert move is not None
        assert hasattr(move, 'sprint')
        
        move_str = move.sprint()
        assert isinstance(move_str, str)
        assert len(move_str) > 0
        
        print(f"Single move test passed: {move_str}")
        # Let Python handle cleanup automatically
    
    def test_thread_configuration_only(self, pymcts_module):
        """Test thread configuration without creating agents."""
        # Test thread setting
        original_threads = pymcts_module.get_rollout_threads()
        
        pymcts_module.set_rollout_threads(1)
        assert pymcts_module.get_rollout_threads() == 1
        
        pymcts_module.set_rollout_threads(2)
        assert pymcts_module.get_rollout_threads() == 2
        
        # Restore original
        pymcts_module.set_rollout_threads(original_threads)
        
        print("Thread configuration test passed")