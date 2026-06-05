#!/usr/bin/env python3
"""
Comprehensive standalone test suite for PyMCTS library.
This suite proves that the MCTS library functions correctly outside of pytest.

Run this script to verify all MCTS functionality:
    python test_mcts_comprehensive.py
"""
import sys
import time
sys.path.append('..')
import pymcts

class ComprehensiveTestRunner:
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.failed_tests = []
    
    def run_test(self, test_func, test_name):
        """Run a single test function."""
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print('='*60)
        
        try:
            test_func()
            print(f"✅ PASSED: {test_name}")
            self.tests_passed += 1
        except Exception as e:
            print(f"❌ FAILED: {test_name}")
            print(f"Error: {e}")
            self.failed_tests.append((test_name, str(e)))
        
        self.tests_run += 1
    
    def print_summary(self):
        """Print test summary."""
        print(f"\n{'='*60}")
        print("TEST SUMMARY")
        print('='*60)
        print(f"Tests run: {self.tests_run}")
        print(f"Tests passed: {self.tests_passed}")
        print(f"Tests failed: {len(self.failed_tests)}")
        
        if self.failed_tests:
            print("\nFailed tests:")
            for test_name, error in self.failed_tests:
                print(f"  - {test_name}: {error}")
        else:
            print("\n🎉 All tests passed!")

def test_basic_imports():
    """Test that all required modules can be imported."""
    # These should not raise exceptions
    state = pymcts.TicTacToe_state()
    move = pymcts.TicTacToe_move(0, 0, 'x')
    
    # Test basic properties
    assert hasattr(state, 'actions_to_try')
    assert hasattr(state, 'next_state')
    assert hasattr(state, 'rollout')
    assert hasattr(state, 'is_terminal')
    assert hasattr(state, 'is_self_side_turn')
    
    print("✓ All imports successful")
    print("✓ Basic object creation works")

def test_tictactoe_state_functionality():
    """Test TicTacToe state basic functionality."""
    state = pymcts.TicTacToe_state()
    
    # Initial state tests
    assert not state.is_terminal(), "Initial state should not be terminal"
    assert state.is_self_side_turn(), "Self side should go first"
    
    # Move generation
    moves = state.actions_to_try()
    assert len(moves) == 9, f"Expected 9 moves, got {len(moves)}"
    
    # State transitions
    first_move = moves[0]
    new_state = state.next_state(first_move)
    assert not new_state.is_self_side_turn(), "Should switch to other side"
    
    new_moves = new_state.actions_to_try()
    assert len(new_moves) == 8, f"Expected 8 moves after first move, got {len(new_moves)}"
    
    print("✓ TicTacToe state functionality verified")

def test_rollout_functionality():
    """Test rollout simulations."""
    state = pymcts.TicTacToe_state()
    
    # Test multiple rollouts
    for i in range(10):
        result = state.rollout()
        assert 0.0 <= result <= 1.0, f"Rollout result {result} not in [0,1]"
    
    print("✓ Rollout functionality verified")

def test_mcts_agent_basic():
    """Test basic MCTS agent functionality."""
    state = pymcts.TicTacToe_state()
    agent = pymcts.MCTS_agent(state, 50, 1)  # Small tree for quick test
    
    # Test agent creation
    current_state = agent.get_current_state()
    assert current_state is not None, "Agent should have a current state"
    
    print("✓ MCTS agent creation successful")
    
    # Clean up
    del agent
    print("✓ MCTS agent destruction successful")

def test_mcts_agent_move_generation():
    """Test MCTS agent move generation."""
    state = pymcts.TicTacToe_state()
    agent = pymcts.MCTS_agent(state, 100, 2)
    
    # Generate move
    move = agent.genmove(None)
    assert move is not None, "Agent should generate a move"
    assert hasattr(move, 'sprint'), "Move should have sprint method"
    
    move_str = move.sprint()
    assert isinstance(move_str, str), "Move string should be a string"
    assert len(move_str) > 0, "Move string should not be empty"
    
    print(f"✓ Generated move: {move_str}")
    
    # Clean up
    del agent
    print("✓ Agent with move generation cleaned up successfully")

def test_mcts_agent_game_progression():
    """Test MCTS agent through a few moves of a game."""
    state = pymcts.TicTacToe_state()
    agent = pymcts.MCTS_agent(state, 50, 1)
    
    moves_played = 0
    max_moves = 5  # Play a few moves
    
    while not agent.get_current_state().is_terminal() and moves_played < max_moves:
        current_state = agent.get_current_state()
        
        if current_state.is_self_side_turn():
            # MCTS agent's turn
            move = agent.genmove(None)
            assert move is not None, f"Agent should generate move on turn {moves_played + 1}"
            print(f"  Turn {moves_played + 1}: MCTS chose {move.sprint()}")
        else:
            # Simulate opponent - take first available move
            possible_moves = current_state.actions_to_try()
            if possible_moves:
                opponent_move = possible_moves[0]
                agent.genmove(opponent_move)
                print(f"  Turn {moves_played + 1}: Opponent chose {opponent_move.sprint()}")
            else:
                break
        
        moves_played += 1
    
    print(f"✓ Played {moves_played} moves successfully")
    
    # Clean up
    del agent
    print("✓ Game progression agent cleaned up successfully")

def test_multiple_agents():
    """Test creating and destroying multiple agents."""
    agents = []
    
    # Create multiple agents
    for i in range(3):
        state = pymcts.TicTacToe_state()
        agent = pymcts.MCTS_agent(state, 20, 1)
        agents.append(agent)
        print(f"✓ Created agent {i + 1}")
    
    # Use each agent
    for i, agent in enumerate(agents):
        move = agent.genmove(None)
        print(f"✓ Agent {i + 1} generated move: {move.sprint()}")
    
    # Clean up all agents
    for i, agent in enumerate(agents):
        del agent
        print(f"✓ Destroyed agent {i + 1}")
    
    print("✓ Multiple agents test completed successfully")

def test_agent_performance_characteristics():
    """Test MCTS agent performance with different parameters."""
    state = pymcts.TicTacToe_state()
    
    test_configs = [
        (10, 1, "Quick test"),
        (50, 1, "Medium test"),
        (100, 2, "Larger test")
    ]
    
    for iterations, max_time, description in test_configs:
        agent = pymcts.MCTS_agent(state, iterations, max_time)
        
        start_time = time.time()
        move = agent.genmove(None)
        elapsed = time.time() - start_time
        
        assert move is not None, f"Agent should generate move for {description}"
        assert elapsed < max_time + 1.0, f"Agent took too long for {description}"
        
        print(f"✓ {description}: {elapsed:.3f}s, move: {move.sprint()}")
        
        del agent
    
    print("✓ Performance characteristics verified")

def test_thread_configuration():
    """Test thread configuration functionality."""
    # Test getting current thread count
    original_threads = pymcts.get_rollout_threads()
    print(f"✓ Original thread count: {original_threads}")
    
    # Test setting thread count
    pymcts.set_rollout_threads(1)
    assert pymcts.get_rollout_threads() == 1, "Thread count should be 1"
    print("✓ Single thread configuration works")
    
    # Test with agent
    state = pymcts.TicTacToe_state()
    agent = pymcts.MCTS_agent(state, 20, 1)
    move = agent.genmove(None)
    print(f"✓ Single-threaded agent works: {move.sprint()}")
    
    del agent
    
    # Restore original configuration
    pymcts.set_rollout_threads(original_threads)
    print("✓ Thread configuration test completed")

def main():
    """Run all tests."""
    print("PyMCTS Comprehensive Test Suite")
    print("===============================")
    print("This suite verifies that the MCTS library works correctly")
    print("outside of pytest environment.\n")
    
    runner = ComprehensiveTestRunner()
    
    # Run all tests
    runner.run_test(test_basic_imports, "Basic Imports and Object Creation")
    runner.run_test(test_tictactoe_state_functionality, "TicTacToe State Functionality")
    runner.run_test(test_rollout_functionality, "Rollout Functionality")
    runner.run_test(test_mcts_agent_basic, "MCTS Agent Basic Operations")
    runner.run_test(test_mcts_agent_move_generation, "MCTS Agent Move Generation")
    runner.run_test(test_mcts_agent_game_progression, "MCTS Agent Game Progression")
    runner.run_test(test_multiple_agents, "Multiple Agents Lifecycle")
    runner.run_test(test_agent_performance_characteristics, "Agent Performance Characteristics")
    runner.run_test(test_thread_configuration, "Thread Configuration")
    
    # Print final summary
    runner.print_summary()
    
    return len(runner.failed_tests) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)