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
    
    def parameters(self):
        # Mock parameters to get device
        class MockParam:
            def __init__(self):
                self.device = torch.device('cpu')
        return iter([MockParam()])
        
    def initial_inference(self, observation):
        batch_size = observation.shape[0] if observation.ndim > 1 else 1
        
        # Mock network outputs
        output = {
            "reward": torch.tensor(np.random.rand(batch_size)).float(),
            "value": torch.tensor(np.random.rand(batch_size)).float(), 
            "policy_logits": torch.tensor(np.random.rand(batch_size, 1134)).float(),
            "hidden_state": torch.tensor(np.random.rand(batch_size, config.HIDDEN_STATE_SIZE)).float()
        }
        
        return output
    
    def recurrent_inference(self, hidden_state, action):
        batch_size = hidden_state.shape[0]
        
        output = {
            "reward": torch.tensor(np.random.rand(batch_size)).float(),
            "value": torch.tensor(np.random.rand(batch_size)).float(),
            "policy_logits": torch.tensor(np.random.rand(batch_size, 1134)).float(),
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
        
        print(f"  - TicTacToe state created")
        print(f"  - MCTS agent created")
        
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
    assert move.index == 4
    
    # Test equality
    move2 = TFTMove(1, 2, 3, 4)
    assert move == move2
    
    move3 = TFTMove(1, 2, 3, 5)
    assert move != move3
    
    # Test string representation
    assert "TFTMove" in str(move)
    
    # Test environment action conversion
    env_action = move.to_env_action()
    assert env_action == [1, 2, 3]
    
    print("✓ TFTMove tests passed")


def test_tft_state():
    """Test TFTState functionality"""
    print("Testing TFTState...")
    
    # Create mock observation
    observation = np.ones(config.OBSERVATION_SIZE)
    mask = np.ones((54,), dtype=bool)
    
    # Create state
    state = TFTState(observation, mask, network=MockNetwork())
    
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
        sample_size=5,
        action_size=3,
        action_limits=[7, 37, 10],
        policy_size=1134,
        network=network
    )
    
    # Test basic properties
    assert mcts.action_size == 3
    assert len(mcts.action_limits) == 3
    assert mcts.policy_size == 1134
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
        action_size=3,
        action_limits=[7, 37, 10],
        policy_size=1134,
        network=network
    )
    
    # Test with single observation
    observation = np.random.rand(config.OBSERVATION_SIZE)
    mask = np.ones((54,), dtype=bool)
    
    print(f"  - Observation shape: {observation.shape}")
    print(f"  - Mask shape: {mask.shape}")
    
    # Generate actions with minimal simulations
    try:
        # genmove in EnhancedMCTS expects single observation and mask
        # but it appends to obs_queue.
        env_move, action_vector = mcts.generate_action(2, observation, mask)
        
        # Check outputs
        print(f"  - Generated env move: {env_move}")
        print(f"  - Generated action vector length: {len(action_vector)}")
        
        assert len(env_move) == 3
        assert len(action_vector) == 1134
        
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
    mask = np.ones((54,), dtype=bool)
    
    # Create state and test field extraction
    state = TFTState(observation, mask, network=MockNetwork())
    
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
        mask = np.ones((54,), dtype=bool)
        tft_state = TFTState(observation, mask, network=MockNetwork())
        
        print(f"  - Created TFT state with {len(tft_state.actions_to_try())} actions")
        
        # Create MCTS agent with TFT state
        agent = pymcts.MCTS_agent(pymcts.SerializedPythonState(tft_state), max_iter=5, max_seconds=1)
        print(f"  - Created MCTS agent")
        
        # Generate a move
        move = agent.genmove(None)
        print(f"  - Generated move: {move}")
        
        if move:
            print(f"  - Move type: {type(move)}")
            print(f"  - Sprint: {move.sprint()}")
        
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
            # If pymcts is not available, we can still test some parts
            # But EnhancedMCTS requires it.
        
        print()
        test_tft_move()
        print()
        test_tft_state()
        print()
        test_enhanced_mcts_basic()
        print()
        
        # Test action generation
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
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
