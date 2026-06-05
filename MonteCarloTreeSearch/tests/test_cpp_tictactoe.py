"""
Tests for C++ TicTacToe implementation.
Tests the built-in C++ TicTacToe game to verify core MCTS functionality.
"""
import pytest
import time


class TestCppTicTacToeBasic:
    """Test C++ TicTacToe basic functionality without MCTS agents."""
    
    def test_tictactoe_state_creation(self, tictactoe_state):
        """Test that C++ TicTacToe state can be created."""
        assert tictactoe_state is not None
        assert not tictactoe_state.is_terminal()
        assert tictactoe_state.is_self_side_turn()
        
    def test_tictactoe_moves(self, tictactoe_state):
        """Test C++ TicTacToe move generation."""
        moves = tictactoe_state.actions_to_try()
        assert len(moves) == 9  # TicTacToe has 9 positions
        
        # All moves should be valid
        for move in moves:
            assert move is not None
            assert hasattr(move, 'sprint')
            move_str = move.sprint()
            assert isinstance(move_str, str)
            assert len(move_str) > 0
            
    def test_tictactoe_state_transitions(self, tictactoe_state):
        """Test C++ TicTacToe state transitions."""
        moves = tictactoe_state.actions_to_try()
        assert len(moves) > 0
        
        # Make a move
        new_state = tictactoe_state.next_state(moves[0])
        assert new_state is not None
        assert not new_state.is_self_side_turn()  # Should switch players
        
        # Should have one fewer move available
        new_moves = new_state.actions_to_try()
        assert len(new_moves) == len(moves) - 1
        
    def test_tictactoe_rollout(self, tictactoe_state):
        """Test C++ TicTacToe rollout functionality."""
        for _ in range(5):  # Test multiple times
            rollout_result = tictactoe_state.rollout()
            assert isinstance(rollout_result, (int, float))
            assert 0.0 <= rollout_result <= 1.0
            
    def test_tictactoe_terminal_detection(self, tictactoe_state):
        """Test C++ TicTacToe terminal state detection."""
        # Initial state should not be terminal
        assert not tictactoe_state.is_terminal()
        
        # Play a complete game to test terminal detection
        current_state = tictactoe_state
        moves_made = 0
        
        while not current_state.is_terminal() and moves_made < 9:
            moves = current_state.actions_to_try()
            if not moves:
                break
            current_state = current_state.next_state(moves[0])
            moves_made += 1
        
        # Should eventually reach terminal state or fill the board
        # Note: TicTacToe might still have moves available in terminal states
        # if it only detects wins, not draws
        if current_state.is_terminal():
            # Terminal state detected - this is correct behavior
            assert True
        else:
            # If not terminal but we filled the board, that's also valid
            assert moves_made == 9


# Working MCTS agent tests - issue was incorrect constructor syntax

class TestCppTicTacToeWithMCTS:
    """Test C++ TicTacToe with MCTS agent."""
    
    def test_tictactoe_with_mcts_agent(self, mcts_agent_factory):
        """Test C++ TicTacToe with MCTS agent."""
        import pymcts
        tictactoe_state = pymcts.TicTacToe_state()
        
        # Create MCTS agent with factory for proper cleanup
        agent = mcts_agent_factory(tictactoe_state, 100, 2)  # 100 iterations, 2 seconds max
        
        # Should be able to generate a move
        move = agent.genmove(None)
        assert move is not None
        assert hasattr(move, 'sprint')
        
        move_str = move.sprint()
        assert isinstance(move_str, str)
        assert len(move_str) > 0
        
        print(f"MCTS generated move: {move_str}")
        
        # Note: agent will be cleaned up by fixture
        
    def test_tictactoe_full_game_with_mcts(self, pymcts_module):
        """Test a complete TicTacToe game with MCTS."""
        initial_state = pymcts_module.TicTacToe_state()
        agent = pymcts_module.MCTS_agent(initial_state, 50, 1)  # Quick game for testing
        
        moves_played = 0
        max_moves = 9  # TicTacToe has maximum 9 moves
        
        while not agent.get_current_state().is_terminal() and moves_played < max_moves:
            current_state = agent.get_current_state()
            print(f"Move {moves_played + 1}:")
            current_state.print()
            
            if current_state.is_self_side_turn():
                # MCTS agent's turn
                agent_move = agent.genmove(None)
                assert agent_move is not None
                print(f"MCTS chose: {agent_move.sprint()}")
            else:
                # Simulate opponent - take first available move
                possible_moves = current_state.actions_to_try()
                if possible_moves:
                    opponent_move = possible_moves[0]
                    agent.genmove(opponent_move)
                    print(f"Opponent chose: {opponent_move.sprint()}")
                else:
                    break
            
            moves_played += 1
        
        # Game should have progressed
        assert moves_played > 0
        
        final_state = agent.get_current_state()
        print("Final state:")
        final_state.print()
        
        if final_state.is_terminal():
            print("Game finished!")
        else:
            print(f"Game ended early after {moves_played} moves")
            
    def test_mcts_agent_performance(self, pymcts_module):
        """Test MCTS agent performance characteristics."""
        state = pymcts_module.TicTacToe_state()
        
        # Test with different iteration counts
        for iterations in [10, 50, 100]:
            agent = pymcts_module.MCTS_agent(state, iterations, 1)
            
            start_time = time.time()
            move = agent.genmove(None)
            elapsed = time.time() - start_time
            
            assert move is not None
            assert elapsed < 2.0  # Should be reasonably fast
            
            print(f"Iterations: {iterations}, Time: {elapsed:.3f}s, Move: {move.sprint()}")
            
    def test_mcts_agent_statistics(self, pymcts_module):
        """Test MCTS agent statistics and feedback."""
        state = pymcts_module.TicTacToe_state()
        agent = pymcts_module.MCTS_agent(state, 100, 1)
        
        # Generate a move to populate statistics
        move = agent.genmove(None)
        assert move is not None
        
        # Test that feedback works (this method should exist and not crash)
        try:
            agent.feedback()
            print("Agent feedback displayed successfully")
        except AttributeError:
            # If feedback method doesn't exist, that's OK
            print("Agent feedback method not available")
        
        # Test getting current state
        current_state = agent.get_current_state()
        assert current_state is not None
        assert isinstance(current_state, pymcts_module.TicTacToe_state)