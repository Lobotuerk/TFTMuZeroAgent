#!/usr/bin/env python3
"""
Test the updated MCTS torch implementation with neural network integration.
"""

import numpy as np
import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Models.MCTS_torch import MCTS, EnhancedTFTState


class MockNetwork(nn.Module):
    """Mock neural network for testing."""
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
        self.training_steps_val = 0
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))
    
    def value_function(self, x):
        return self.forward(x)
    
    def training_steps(self):
        return self.training_steps_val


def test_mcts_initialization():
    """Test MCTS initialization with network."""
    print("Testing MCTS initialization...")
    
    network = MockNetwork()
    mcts = MCTS(
        network=network,
        sample_size=50,
        action_size=3,
        policy_size=100,
        max_simulations=25
    )
    
    assert mcts.network == network
    assert mcts.sample_size == 50
    assert mcts.action_size == 3
    assert mcts.policy_size == 100
    assert mcts.max_simulations == 25
    
    print("✅ MCTS initialization test passed")


def test_enhanced_tft_state():
    """Test enhanced TFT state with network rollout."""
    print("Testing enhanced TFT state...")
    
    network = MockNetwork()
    observations = {
        "player_0": np.random.random((10,)),
        "player_1": np.random.random((10,))
    }
    
    state = EnhancedTFTState(
        observations=observations,
        current_player="player_0",
        network=network
    )
    
    # Test rollout
    rollout_value = state.rollout()
    assert 0.0 <= rollout_value <= 1.0
    
    print(f"✅ Enhanced TFT state rollout value: {rollout_value}")


def test_legacy_generate_action():
    """Test legacy action generation."""
    print("Testing legacy action generation...")
    
    network = MockNetwork()
    mcts = MCTS(network=network, policy_size=100)
    
    observations = {
        "player_0": np.random.random((10,))
    }
    
    # Test legacy fallback
    action, policy, info = mcts._legacy_generate_action(observations, None, "player_0")
    
    assert action in ['reroll', 'level', 'pass']
    assert policy.shape == (100, 1)
    assert info['move_type'] == 'legacy_fallback'
    
    print(f"✅ Legacy action: {action}, confidence: {info['confidence']}")


def test_move_parsing():
    """Test MCTS move parsing."""
    print("Testing move parsing...")
    
    network = MockNetwork()
    mcts = MCTS(network=network)
    
    # Test different move strings
    test_moves = [
        "reroll_action",
        "level_up",
        "buy_champion_0",
        "unknown_action"
    ]
    
    for move_str in test_moves:
        action_info = mcts._parse_mcts_move(move_str)
        assert 'action' in action_info
        assert 'type' in action_info
        assert 'confidence' in action_info
        print(f"  Move '{move_str}' -> {action_info}")
    
    print("✅ Move parsing tests passed")


def test_policy_compatibility():
    """Test legacy policy method compatibility."""
    print("Testing legacy policy compatibility...")
    
    network = MockNetwork()
    mcts = MCTS(network=network, policy_size=100)
    
    # Test with numpy observation
    observation = np.random.random((10,))
    mask = np.ones(10, dtype=bool)
    
    result = mcts.policy(observation, mask)
    
    # Should return (actions, target_policy, string_samples, board_distribution, directive)
    assert len(result) == 5
    actions, target_policy, string_samples, board_dist, directive = result
    
    assert len(actions) == 1
    assert len(target_policy) == 1
    assert target_policy[0].shape == (100, 1)
    
    print("✅ Legacy policy compatibility test passed")


def test_statistics():
    """Test statistics tracking."""
    print("Testing statistics tracking...")
    
    network = MockNetwork()
    mcts = MCTS(network=network)
    
    # Initial statistics
    stats = mcts.get_statistics()
    assert stats['total_actions'] == 0
    
    # Update state to increment action count
    mcts.num_actions = 5
    stats = mcts.get_statistics()
    assert stats['total_actions'] == 5
    
    print(f"✅ Statistics: {stats}")


def test_network_info():
    """Test network info retrieval."""
    print("Testing network info...")
    
    network = MockNetwork()
    network.training_steps_val = 42
    mcts = MCTS(network=network)
    
    info = mcts.get_info()
    assert 'network_id' in info
    print(f"✅ Network info: {info}")


def run_all_tests():
    """Run all tests."""
    print("Running MCTS torch implementation tests...\n")
    
    try:
        test_mcts_initialization()
        test_enhanced_tft_state()
        test_legacy_generate_action()
        test_move_parsing()
        test_policy_compatibility()
        test_statistics()
        test_network_info()
        
        print("\n🎉 All tests passed! Updated MCTS torch implementation is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)