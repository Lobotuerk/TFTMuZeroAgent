#!/usr/bin/env python3
"""
Test suite for TFT MCTS implementation using TDD approach.

This test file defines the expected behavior of TFT MCTS integration
before implementing the actual functionality.
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Test imports - these will initially fail until we implement them
try:
    from Models.tft_mcts import TFTMove, TFTState
    from TFTSet4Gym.tft_set4_gym.tft_simulator import parallel_env
    MCTS_AVAILABLE = True
except ImportError:
    MCTS_AVAILABLE = False

# Skip all tests if MCTS components not available
pytestmark = pytest.mark.skipif(not MCTS_AVAILABLE, reason="TFT MCTS components not implemented yet")


class TestTFTMove:
    """Test the TFT move class for MCTS integration."""
    
    def test_move_creation(self):
        """Test that we can create different types of TFT moves."""
        # Test shop purchase move
        buy_move = TFTMove(action_type="buy", shop_index=0, player_id="player_0")
        assert buy_move.action_type == "buy"
        assert buy_move.shop_index == 0
        assert buy_move.player_id == "player_0"
        
        # Test sell move
        sell_move = TFTMove(action_type="sell", board_index=5, player_id="player_0")
        assert sell_move.action_type == "sell"
        assert sell_move.board_index == 5
        
        # Test move units on board
        move_unit = TFTMove(action_type="move", from_index=0, to_index=7, player_id="player_0")
        assert move_unit.action_type == "move"
        assert move_unit.from_index == 0
        assert move_unit.to_index == 7
        
        # Test level up move
        level_move = TFTMove(action_type="level", player_id="player_0")
        assert level_move.action_type == "level"
        
        # Test reroll move
        reroll_move = TFTMove(action_type="reroll", player_id="player_0")
        assert reroll_move.action_type == "reroll"

    def test_move_equality(self):
        """Test that move equality works correctly."""
        move1 = TFTMove(action_type="buy", shop_index=0, player_id="player_0")
        move2 = TFTMove(action_type="buy", shop_index=0, player_id="player_0")
        move3 = TFTMove(action_type="buy", shop_index=1, player_id="player_0")
        
        assert move1 == move2
        assert move1 != move3

    def test_move_string_representation(self):
        """Test that moves have meaningful string representations."""
        buy_move = TFTMove(action_type="buy", shop_index=2, player_id="player_0")
        move_str = str(buy_move)
        assert "buy" in move_str.lower()
        assert "2" in move_str


class TestTFTState:
    """Test the TFT game state class for MCTS integration."""
    
    def test_state_creation_from_env(self):
        """Test that we can create a TFT state from environment observation."""
        env = parallel_env()
        observations, infos = env.reset()
        
        # Get observation for first player
        first_player = list(observations.keys())[0]
        observation = observations[first_player]
        
        # Create TFT state from observation
        tft_state = TFTState(observations=observations, current_player=first_player)
        
        assert tft_state.current_player == first_player
        assert tft_state.observations is not None
        assert len(tft_state.observations) == len(observations)

    def test_state_actions_to_try(self):
        """Test that the state can generate valid moves."""
        env = parallel_env()
        observations, infos = env.reset()
        first_player = list(observations.keys())[0]
        
        tft_state = TFTState(observations=observations, current_player=first_player)
        moves = tft_state.actions_to_try()
        
        # Should return a list of TFTMove objects
        assert isinstance(moves, list)
        assert len(moves) > 0
        assert all(isinstance(move, TFTMove) for move in moves)
        
        # Should include basic actions like reroll, level, buy
        action_types = [move.action_type for move in moves]
        assert "reroll" in action_types or "level" in action_types

    def test_state_next_state(self):
        """Test that we can apply a move to get the next state."""
        env = parallel_env()
        observations, infos = env.reset()
        first_player = list(observations.keys())[0]
        
        tft_state = TFTState(observations=observations, current_player=first_player)
        moves = tft_state.actions_to_try()
        
        if moves:
            # Apply first valid move
            next_state = tft_state.next_state(moves[0])
            
            assert isinstance(next_state, TFTState)
            assert next_state != tft_state  # Should be different state
            # Player might change to next player or stay same depending on game phase
            assert next_state.current_player is not None

    def test_state_is_terminal(self):
        """Test terminal state detection."""
        env = parallel_env()
        observations, infos = env.reset()
        first_player = list(observations.keys())[0]
        
        tft_state = TFTState(observations=observations, current_player=first_player)
        
        # Fresh game should not be terminal
        assert not tft_state.is_terminal()

    def test_state_is_self_side_turn(self):
        """Test turn detection for MCTS."""
        env = parallel_env()
        observations, infos = env.reset()
        first_player = list(observations.keys())[0]
        
        tft_state = TFTState(observations=observations, current_player=first_player)
        
        # Should return boolean
        result = tft_state.is_self_side_turn()
        assert isinstance(result, bool)

    def test_state_rollout(self):
        """Test random rollout simulation."""
        env = parallel_env()
        observations, infos = env.reset()
        first_player = list(observations.keys())[0]
        
        tft_state = TFTState(observations=observations, current_player=first_player)
        
        # Rollout should return a probability between 0 and 1
        result = tft_state.rollout()
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


class TestTFTMCTSIntegration:
    """Test integration of TFT with PyMCTS library."""
    
    @pytest.mark.skipif(False, reason="Testing PyMCTS integration")
    def test_pymcts_integration(self):
        """Test that TFT classes work with PyMCTS library."""
        # Import PyMCTS for integration testing
        try:
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MonteCarloTreeSearch'))
            import pymcts
            
            # Create TFT state
            env = parallel_env()
            observations, infos = env.reset()
            first_player = list(observations.keys())[0]
            
            tft_state = TFTState(observations=observations, current_player=first_player)
            
            # Test that we can create PyMCTS wrapper (future integration)
            # For now, just test that our classes have the right interface
            
            # Test that actions_to_try returns a list
            moves = tft_state.actions_to_try()
            assert isinstance(moves, list)
            assert len(moves) > 0
            
            # Test that next_state works
            if moves:
                next_state = tft_state.next_state(moves[0])
                assert isinstance(next_state, TFTState)
            
            # Test rollout
            result = tft_state.rollout()
            assert 0.0 <= result <= 1.0
            
            print("✅ PyMCTS interface compatibility confirmed")
            
        except ImportError:
            pytest.skip("PyMCTS not available for integration testing")

    def test_performance_basic(self):
        """Test basic performance characteristics."""
        env = parallel_env()
        observations, infos = env.reset()
        first_player = list(observations.keys())[0]
        
        tft_state = TFTState(observations=observations, current_player=first_player)
        
        # Generating moves should be reasonably fast
        import time
        start_time = time.time()
        moves = tft_state.actions_to_try()
        elapsed = time.time() - start_time
        
        assert elapsed < 1.0, f"Move generation took too long: {elapsed:.3f}s"
        assert len(moves) > 0, "Should generate at least some moves"


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])