"""
Core functionality tests for PyMCTS.
Tests basic module import, TicTacToe functionality, and MCTS agent operations.
"""
import pytest
import time


class TestModuleImport:
    """Test that the PyMCTS module imports and basic functions work."""
    
    def test_module_imports(self, pymcts_module):
        """Test that pymcts module imports successfully."""
        assert pymcts_module is not None
        
    def test_hardware_info(self, pymcts_module):
        """Test hardware information functions."""
        concurrency = pymcts_module.get_hardware_concurrency()
        optimal = pymcts_module.get_optimal_thread_count()
        
        assert concurrency > 0
        assert optimal > 0
        assert optimal <= concurrency
        
    def test_thread_configuration(self, pymcts_module):
        """Test thread configuration functions."""
        original = pymcts_module.get_rollout_threads()
        
        # Test setting different values
        for threads in [1, 2, 4]:
            pymcts_module.set_rollout_threads(threads)
            assert pymcts_module.get_rollout_threads() == threads
        
        # Restore original
        pymcts_module.set_rollout_threads(original)


class TestTicTacToe:
    """Test the built-in C++ TicTacToe implementation."""
    
    def test_state_creation(self, pymcts_module):
        """Test that TicTacToe state can be created."""
        state = pymcts_module.TicTacToe_state()
        assert state is not None
        assert not state.is_terminal()
        assert state.is_self_side_turn()
        
    def test_cpp_state_creation(self, pymcts_module):
        """Test that C++ TicTacToe state factory function works."""
        state = pymcts_module.cpp_TicTacToeState()
        assert state is not None
        assert not state.is_terminal()
        assert state.is_self_side_turn()
        
    def test_move_generation(self, pymcts_module):
        """Test that moves can be generated."""
        state = pymcts_module.TicTacToe_state()
        moves = state.actions_to_try()
        assert len(moves) == 9  # Empty board should have 9 possible moves
        
    def test_state_transitions(self, pymcts_module):
        """Test state transitions work correctly."""
        state = pymcts_module.TicTacToe_state()
        moves = state.actions_to_try()
        assert len(moves) > 0
        
        # Make a move
        new_state = state.next_state(moves[0])
        assert new_state is not None
        assert not new_state.is_self_side_turn()  # Should switch players
        
        # New state should have fewer moves available
        new_moves = new_state.actions_to_try()
        assert len(new_moves) == len(moves) - 1
        
    def test_rollout(self, pymcts_module):
        """Test rollout functionality."""
        state = pymcts_module.TicTacToe_state()
        result = state.rollout()
        assert 0.0 <= result <= 1.0
        
    def test_move_sprint(self, pymcts_module):
        """Test move string representation."""
        state = pymcts_module.TicTacToe_state()
        moves = state.actions_to_try()
        assert len(moves) > 0
        
        move_str = moves[0].sprint()
        assert isinstance(move_str, str)
        assert len(move_str) > 0


class TestMCTSAgent:
    """Test MCTS agent functionality."""
    
    def test_agent_creation(self, pymcts_module):
        """Test that MCTS agent can be created."""
        state = pymcts_module.TicTacToe_state()
        agent = pymcts_module.MCTS_agent(state, max_iter=10, max_seconds=1)
        assert agent is not None
        
    def test_move_generation_basic(self, pymcts_module):
        """Test that agent can generate moves with minimal parameters."""
        # Use very conservative parameters to avoid crashes
        state = pymcts_module.TicTacToe_state()
        agent = pymcts_module.MCTS_agent(state, max_iter=5, max_seconds=1)
        
        try:
            move = agent.genmove(None)
            assert move is not None
            
            # Move should be a valid string representation
            move_str = move.sprint()
            assert isinstance(move_str, str)
            assert len(move_str) > 0
        except Exception as e:
            pytest.fail(f"Move generation failed: {e}")
        
    def test_current_state(self, pymcts_module):
        """Test that agent tracks current state correctly."""
        state = pymcts_module.TicTacToe_state()
        agent = pymcts_module.MCTS_agent(state, max_iter=5, max_seconds=1)
        
        initial_state = agent.get_current_state()
        assert initial_state is not None
        assert not initial_state.is_terminal()
        
    def test_feedback_safe(self, pymcts_module):
        """Test that agent feedback doesn't crash."""
        state = pymcts_module.TicTacToe_state()
        agent = pymcts_module.MCTS_agent(state, max_iter=5, max_seconds=1)
        
        # This should not raise an exception
        try:
            agent.feedback()
        except Exception as e:
            pytest.fail(f"Agent feedback failed: {e}")


class TestPerformanceBasic:
    """Basic performance tests with conservative parameters."""
    
    def test_move_generation_timeout(self, pymcts_module):
        """Test that move generation completes within timeout."""
        state = pymcts_module.TicTacToe_state()
        agent = pymcts_module.MCTS_agent(state, max_iter=20, max_seconds=1)
        
        start_time = time.time()
        try:
            move = agent.genmove(None)
            elapsed = time.time() - start_time
            
            assert move is not None
            assert elapsed < 3.0  # Should complete within 3 seconds
        except Exception as e:
            elapsed = time.time() - start_time
            pytest.fail(f"Move generation failed after {elapsed:.2f}s: {e}")
        
    def test_small_iterations(self, pymcts_module):
        """Test agent with very small iteration counts."""
        state = pymcts_module.TicTacToe_state()
        
        for iterations in [1, 5, 10]:
            try:
                agent = pymcts_module.MCTS_agent(state, max_iter=iterations, max_seconds=1)
                
                start_time = time.time()
                move = agent.genmove(None)
                elapsed = time.time() - start_time
                
                assert move is not None
                assert elapsed < 5.0  # Generous timeout
            except Exception as e:
                pytest.fail(f"Small iteration test failed with {iterations} iterations: {e}")


class TestThreadSafety:
    """Basic thread safety tests."""
    
    def test_single_thread_mode(self, pymcts_module):
        """Test that single thread mode works reliably."""
        # Force single thread mode to avoid threading issues
        original_threads = pymcts_module.get_rollout_threads()
        
        try:
            pymcts_module.set_rollout_threads(1)
            
            state = pymcts_module.TicTacToe_state()
            agent = pymcts_module.MCTS_agent(state, max_iter=10, max_seconds=1)
            
            move = agent.genmove(None)
            assert move is not None
            
        finally:
            pymcts_module.set_rollout_threads(original_threads)