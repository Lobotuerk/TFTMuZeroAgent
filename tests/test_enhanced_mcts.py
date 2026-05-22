#!/usr/bin/env python3
"""
Test script for enhanced MCTS implementation
"""

import numpy as np
import sys
import os
import torch

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import config
from Models.MCTS_torch import EnhancedMCTS, TFTMove, TFTState, create_enhanced_mcts


class MockNetwork:
    """Mock network for testing"""
    
    def __init__(self):
        self.training_steps_count = 0
    
    def initial_inference(self, observation):
        batch_size = observation.shape[0] if observation.ndim > 1 else 1
        
        # Mock network outputs
        output = {
            "reward": torch.tensor(np.random.rand(batch_size)).float(),
            "value": torch.tensor(np.random.rand(batch_size)).float(), 
            "policy_logits": [torch.tensor(np.random.rand(10)).float() for _ in range(batch_size)],
            "hidden_state": torch.tensor(np.random.rand(batch_size, config.HIDDEN_STATE_SIZE)).float()
        }
        
        directive = torch.tensor(np.random.rand(batch_size, 64)).float()
        board_distribution = torch.tensor(np.random.rand(batch_size, 4, 7, 8)).float()
        
        return output, directive, board_distribution
    
    def recurrent_inference(self, hidden_state, action):
        batch_size = hidden_state.shape[0]
        
        output = {
            "reward": torch.tensor(np.random.rand(batch_size)).float(),
            "value": torch.tensor(np.random.rand(batch_size)).float(),
            "policy_logits": torch.tensor(np.random.rand(batch_size, 10)).float(),
            "hidden_state": torch.tensor(np.random.rand(batch_size, config.HIDDEN_STATE_SIZE)).float()
        }
        
        return output
    
    def training_steps(self):
        return self.training_steps_count


def test_pymcts_integration():
    """Test PyMCTS integration specifically"""
    print("Testing PyMCTS integration...")
    
    try:
        import pymcts
        
        # Test basic PyMCTS functionality
        print("  - Testing basic PyMCTS classes...")
        
        # Test TicTacToe example (built into PyMCTS)
        ttt_state = pymcts.TicTacToe_state()
        ttt_agent = pymcts.MCTS_agent(ttt_state, max_iter=10, max_seconds=1)
        
        print(f"  - TicTacToe state created: {ttt_state}")
        print(f"  - MCTS agent created: {ttt_agent}")
        
        # Test a few moves
        move = ttt_agent.genmove()
        print(f"  - Generated move: {move}")
        
        print("✓ PyMCTS integration working")
        return True
        
    except Exception as e:
        print(f"✗ PyMCTS integration failed: {e}")
        return False


def test_tft_move():
    """Test TFTMove functionality"""
    print("Testing TFTMove...")
    
    # Test basic move creation
    move = TFTMove(1, 2, 3, 4)
    assert move.action_type == 1
    assert move.target_1 == 2
    assert move.target_2 == 3
    assert move.target_3 == 4
    assert move.action_string == "1_2_3_4"
    
    # Test equality
    move2 = TFTMove(1, 2, 3, 4)
    assert move == move2
    
    move3 = TFTMove(1, 2, 3, 5)
    assert move != move3
    
    # Test string representation
    assert "TFT_move(1, 2, 3, 4)" in str(move)
    
    # Test environment action conversion
    env_action = move.to_env_action()
    assert env_action == [1, 2, 3, 4]
    
    # Test from_string creation
    move4 = TFTMove.from_string("1_2_3_4")
    assert move4 == move
    
    print("✓ TFTMove tests passed")


def test_tft_state():
    """Test TFTState functionality"""
    print("Testing TFTState...")
    
    # Create mock observation
    observation = np.random.rand(config.OBSERVATION_SIZE)
    mask = np.ones((13, 100), dtype=bool)
    
    # Create state
    state = TFTState(observation, mask, True, MockNetwork())
    
    # Test basic properties
    assert state.player_turn == True
    assert state.health > 0
    assert state.round_num >= 1
    assert state.level >= 1
    
    # Test actions
    actions = state.actions_to_try()
    assert len(actions) > 0
    assert all(isinstance(action, TFTMove) for action in actions)
    print(f"  - Generated {len(actions)} legal actions")
    
    # Test terminal check
    terminal = state.is_terminal()
    assert isinstance(terminal, bool)
    print(f"  - Terminal state: {terminal}")
    
    # Test next state
    if actions:
        next_state = state.next_state(actions[0])
        assert isinstance(next_state, TFTState)
        assert next_state.player_turn != state.player_turn
        print(f"  - Next state created successfully")
    
    # Test rollout
    rollout_result = state.rollout()
    assert 0.0 <= rollout_result <= 1.0
    print(f"  - Rollout result: {rollout_result:.3f}")
    
    # Test print functionality
    print("  - State info:")
    state.print()
    
    print("✓ TFTState tests passed")


def test_enhanced_mcts_basic():
    """Test basic EnhancedMCTS functionality"""
    print("Testing Enhanced MCTS basic functionality...")
    
    # Create mock network
    network = MockNetwork()
    
    # Create MCTS with minimal simulations for speed
    mcts = create_enhanced_mcts(
        sample_size=5,  # Reduced for speed
        action_size=4,
        action_limits=[7, 37, 10, 5],
        policy_size=100,  # Reduced for speed
        network=network
    )
    
    # Test basic properties
    assert mcts.action_size == 4
    assert len(mcts.action_limits) == 4
    assert mcts.policy_size == 100
    assert mcts.network == network
    
    print(f"  - MCTS created with {mcts.action_size} action dimensions")
    
    # Test stats
    stats = mcts.get_stats()
    assert 'total_generations' in stats
    assert 'pymcts_generations' in stats
    assert 'pymcts_available' in stats
    print(f"  - Initial stats: {stats}")
    
    # Test metadata
    metadata = mcts.fill_metadata()
    assert 'network_id' in metadata
    assert 'pymcts_available' in metadata
    print(f"  - Metadata: {metadata}")
    
    print("✓ Enhanced MCTS basic tests passed")


def test_enhanced_mcts_action_generation():
    """Test MCTS action generation"""
    print("Testing Enhanced MCTS action generation...")
    
    # Create network and MCTS
    network = MockNetwork()
    mcts = create_enhanced_mcts(
        sample_size=5,
        action_size=4,
        action_limits=[7, 37, 10, 5],
        policy_size=100,
        network=network
    )
    
    # Test with small batch for speed
    batch_size = 2
    observation = np.random.rand(batch_size, config.OBSERVATION_SIZE)
    mask = np.ones((batch_size, 13, 100), dtype=bool)
    
    print(f"  - Testing with batch size: {batch_size}")
    print(f"  - Observation shape: {observation.shape}")
    print(f"  - Mask shape: {mask.shape}")
    
    # Generate actions with minimal simulations
    try:
        actions, policies = mcts.generate_action(2, observation, mask)  # Only 2 simulations for speed
        
        # Check outputs
        print(f"  - Generated actions shape: {actions.shape}")
        print(f"  - Generated policies shape: {policies.shape}")
        print(f"  - Actions: {actions}")
        
        assert actions.shape[0] == batch_size
        assert policies.shape[0] == batch_size
        assert all(isinstance(action, (str, np.str_)) for action in actions)
        
        print("✓ Action generation successful")
        
    except Exception as e:
        print(f"✗ Action generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test stats after generation
    final_stats = mcts.get_stats()
    print(f"  - Final stats: {final_stats}")
    
    print("✓ Enhanced MCTS action generation tests passed")
    return True


def test_observation_schema_integration():
    """Test integration with new observation schema"""
    print("Testing observation schema integration...")
    
    # Test with realistic observation
    observation = np.random.rand(config.OBSERVATION_SIZE)
    mask = np.ones((13, 100), dtype=bool)
    
    # Create state and test field extraction
    state = TFTState(observation, mask, True, MockNetwork())
    
    # Verify that extraction works (even if it falls back to defaults)
    assert state.health >= 0
    assert state.round_num >= 0
    assert state.level >= 0
    assert state.turns_for_combat >= 0
    
    print(f"  - Extracted health: {state.health}")
    print(f"  - Extracted round: {state.round_num}")
    print(f"  - Extracted level: {state.level}")
    print(f"  - Extracted turns_for_combat: {state.turns_for_combat}")
    
    print("✓ Observation schema integration tests passed")


def test_mcts_with_tft_states():
    """Test MCTS using TFT states directly"""
    print("Testing MCTS with TFT states...")
    
    try:
        import pymcts
        
        # Create a TFT state
        observation = np.random.rand(config.OBSERVATION_SIZE)
        mask = np.ones((13, 100), dtype=bool)
        tft_state = TFTState(observation, mask, True, MockNetwork())
        
        print(f"  - Created TFT state with {len(tft_state.actions_to_try())} actions")
        
        # Create MCTS agent with TFT state
        agent = pymcts.MCTS_agent(tft_state, max_iter=5, max_seconds=1)
        print(f"  - Created MCTS agent: {agent}")
        
        # Generate a move
        move = agent.genmove()
        print(f"  - Generated move: {move}")
        
        if move:
            print(f"  - Move type: {type(move)}")
            if hasattr(move, 'action_string'):
                print(f"  - Action string: {move.action_string}")
        
        print("✓ MCTS with TFT states working")
        return True
        
    except Exception as e:
        print(f"✗ MCTS with TFT states failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=== Enhanced MCTS Test Suite ===\n")
    
    try:
        # Test PyMCTS integration first
        if not test_pymcts_integration():
            print("\n❌ PyMCTS integration failed - cannot continue")
            return False
            
        print()
        test_tft_move()
        print()
        test_tft_state()
        print()
        test_enhanced_mcts_basic()
        print()
        
        # Test action generation (this might be slow)
        if not test_enhanced_mcts_action_generation():
            print("\n❌ Action generation test failed")
            return False
            
        print()
        test_observation_schema_integration()
        print()
        
        # Test direct MCTS usage
        if not test_mcts_with_tft_states():
            print("\n❌ Direct MCTS test failed")
            return False
        
        print("\n🎉 All tests passed successfully!")
        
        # Final integration test
        print("\n=== Final Integration Test ===")
        network = MockNetwork()
        mcts = create_enhanced_mcts(
            sample_size=3,
            action_size=4,
            action_limits=[7, 37, 10, 5],
            policy_size=50,
            network=network
        )
        
        # Test with single observation
        observation = np.random.rand(config.OBSERVATION_SIZE)
        mask = np.ones((13, 100), dtype=bool)
        
        print(f"Final test - single observation shape: {observation.shape}")
        
        # Wrap in batch format
        obs_batch = observation.reshape(1, -1)
        mask_batch = mask.reshape(1, 13, 100)
        
        actions, policies = mcts.generate_action(1, obs_batch, mask_batch)
        print(f"Final test - Generated action: {actions[0]}")
        print(f"Final test - Policy shape: {policies[0].shape}")
        
        # Show final stats
        final_stats = mcts.get_stats()
        print(f"Final stats: {final_stats}")
        
        print("\n✅ Enhanced MCTS is working correctly!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)