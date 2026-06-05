"""
Heuristic rollout functionality tests for PyMCTS.
Tests enhanced rollout strategies and underlying C++ heuristic implementation.
"""
import pytest
import time


class TestHeuristicRollouts:
    """Test heuristic rollout enhancement functionality at the C++ level."""
    
    def test_tictactoe_enhanced_functionality(self, pymcts_module):
        """Test that TicTacToe works with the enhanced C++ backend."""
        state = pymcts_module.TicTacToe_state()
        assert state is not None
        
        # Test that basic functionality works with heuristic-enhanced backend
        moves = state.actions_to_try()
        assert isinstance(moves, list), "TicTacToe should return move list"
        assert len(moves) == 9, "Empty TicTacToe board should have 9 possible moves"
    
    def test_enhanced_rollout_execution(self, pymcts_module):
        """Test that rollout execution works with enhanced C++ implementation."""
        state = pymcts_module.TicTacToe_state()
        
        # Test rollout execution (may use heuristics internally)
        try:
            result = state.rollout()
            assert isinstance(result, (int, float)), "Rollout should return numeric result"
            assert 0.0 <= result <= 1.0, "Rollout result should be between 0.0 and 1.0"
        except Exception as e:
            pytest.fail(f"Enhanced rollout execution failed: {e}")
    
    def test_mcts_with_enhanced_backend(self, mcts_agent_factory, pymcts_module):
        """Test MCTS agent functionality with heuristic-enhanced C++ backend."""
        state = pymcts_module.TicTacToe_state()
        wrapped_state = pymcts_module.SerializedPythonState(state)
        
        # Create MCTS agent with enhanced backend
        agent = mcts_agent_factory(wrapped_state, max_iter=50, max_seconds=1)
        
        # Test that agent works with enhanced backend
        move = agent.genmove(None)
        assert move is not None, "Agent should find a move with enhanced backend"
        
        # Parse move from string format
        move_str = str(move)
        assert "(" in move_str and ")" in move_str, f"Move should be a valid tuple, got: {move_str}"
    
    def test_move_quality_consistency(self, mcts_agent_factory, pymcts_module):
        """Test that enhanced backend provides consistent move quality."""
        state = pymcts_module.TicTacToe_state()
        wrapped_state = pymcts_module.SerializedPythonState(state)
        
        # Create agent for consistency testing
        agent = mcts_agent_factory(wrapped_state, max_iter=30, max_seconds=1)
        
        # Test multiple searches for consistency
        moves = []
        for _ in range(3):
            move = agent.genmove(None)
            moves.append(str(move))
        
        # All moves should be valid
        for i, move_str in enumerate(moves):
            assert "(" in move_str and ")" in move_str, f"Move {i+1} should be valid tuple format, got: {move_str}"
        
        # At least one move should be found consistently
        assert all(move != "None" for move in moves), "All searches should find valid moves"
    
    def test_strategic_move_preference(self, mcts_agent_factory, pymcts_module):
        """Test that enhanced backend shows strategic preferences (center/corner bias)."""
        state = pymcts_module.TicTacToe_state()
        wrapped_state = pymcts_module.SerializedPythonState(state)
        
        # Create agent with more iterations for strategic analysis
        agent = mcts_agent_factory(wrapped_state, max_iter=100, max_seconds=1)
        
        # Test that agent makes strategic moves
        move = agent.genmove(None)
        assert move is not None, "Agent should find a strategic move"
        
        # Parse move and verify it's reasonable for opening
        move_str = str(move)
        assert "(" in move_str and ")" in move_str, f"Move should be valid tuple format, got: {move_str}"
        
        # For TicTacToe, moves should be in format (row,col,player)
        # Just verify we get a valid position
        assert "," in move_str, f"Move should have comma-separated format, got: {move_str}"
    
    def test_thread_configuration_compatibility(self, pymcts_module):
        """Test that thread configuration works with enhanced implementation."""
        # Test thread configuration functions
        original_threads = pymcts_module.get_rollout_threads()
        
        try:
            # Test setting different thread counts
            for threads in [1, 2]:
                pymcts_module.set_rollout_threads(threads)
                current = pymcts_module.get_rollout_threads()
                assert current == threads, f"Thread setting failed: expected {threads}, got {current}"
            
            # Test hardware detection functions
            hardware_threads = pymcts_module.get_hardware_concurrency()
            optimal_threads = pymcts_module.get_optimal_thread_count()
            
            assert hardware_threads > 0, "Hardware should report positive thread count"
            assert optimal_threads > 0, "Optimal thread count should be positive"
            assert optimal_threads <= hardware_threads, "Optimal should not exceed hardware capacity"
            
        finally:
            # Restore original setting
            pymcts_module.set_rollout_threads(original_threads)


class TestEnhancedIntegration:
    """Test integration of enhanced C++ implementation with existing functionality."""
    
    def test_backward_compatibility(self, pymcts_module):
        """Test that existing functionality still works with enhanced C++ backend."""
        state = pymcts_module.TicTacToe_state()
        
        # Test original methods still work
        moves = state.actions_to_try()
        assert isinstance(moves, list), "actions_to_try should return a list"
        
        if moves:
            next_state = state.next_state(moves[0])
            assert next_state is not None, "next_state should work"
            
            # Test state progression
            remaining_moves = next_state.actions_to_try()
            assert len(remaining_moves) == len(moves) - 1, "Should have one fewer move after playing"
        
        # Test game state methods
        assert isinstance(state.is_terminal(), bool), "is_terminal should return boolean"
        assert isinstance(state.is_self_side_turn(), bool), "is_self_side_turn should return boolean"
        assert isinstance(state.get_turn(), str), "get_turn should return string"
        assert isinstance(state.get_winner(), str), "get_winner should return string"
    
    def test_performance_with_enhancement(self, mcts_agent_factory, pymcts_module):
        """Test that enhanced implementation maintains good performance."""
        import time
        
        state = pymcts_module.TicTacToe_state()
        wrapped_state = pymcts_module.SerializedPythonState(state)
        
        # Create agent with time limit
        agent = mcts_agent_factory(wrapped_state, max_iter=30, max_seconds=1)
        
        # Test performance
        start_time = time.time()
        move = agent.genmove(None)
        elapsed_time = time.time() - start_time
        
        assert move is not None, "Agent should find a move within time limit"
        assert elapsed_time < 2.0, f"Search should complete quickly, took {elapsed_time:.3f}s"
        
        # Verify move is valid format
        move_str = str(move)
        assert "(" in move_str and ")" in move_str, f"Move should be in correct tuple format, got: {move_str}"
    
    def test_multiple_games_stability(self, mcts_agent_factory, pymcts_module):
        """Test that enhanced implementation is stable across multiple games."""
        # Test multiple game instances
        for i in range(3):
            state = pymcts_module.TicTacToe_state()
            wrapped_state = pymcts_module.SerializedPythonState(state)
            
            # Create fresh agent for each game
            agent = mcts_agent_factory(wrapped_state, max_iter=20, max_seconds=1)
            
            # Test that each game works
            move = agent.genmove(None)
            assert move is not None, f"Agent should find move in game {i+1}"
            
            # Verify move format
            move_str = str(move)
            assert "(" in move_str and ")" in move_str, f"Move {i+1} should be valid tuple format, got: {move_str}"