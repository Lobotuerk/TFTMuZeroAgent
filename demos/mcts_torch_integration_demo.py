#!/usr/bin/env python3
"""
Demo showing the updated MCTS torch implementation with TFT MCTS bridge integration.
"""

import numpy as np
import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Models.MCTS_torch import MCTS, EnhancedTFTState
from Models.tft_mcts import create_tft_state_from_env


class DemoNetwork(nn.Module):
    """Demo neural network with multiple inference methods."""
    
    def __init__(self, obs_size=128, hidden_size=64):
        super().__init__()
        self.obs_size = obs_size
        self.hidden_size = hidden_size
        
        # Value function head
        self.value_net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Policy head
        self.policy_net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 10)  # 10 action types
        )
        
        self._training_steps = 100
    
    def forward(self, x):
        """Simple forward pass returns value."""
        return torch.sigmoid(self.value_net(x))
    
    def value_function(self, x):
        """Dedicated value function."""
        return torch.sigmoid(self.value_net(x))
    
    def initial_inference(self, x):
        """MuZero-style initial inference."""
        value = self.value_function(x)
        policy_logits = self.policy_net(x)
        
        return {
            'value': value,
            'policy_logits': [policy_logits],
            'reward': torch.zeros_like(value),
            'hidden_state': x
        }
    
    def training_steps(self):
        """Get training steps."""
        return self._training_steps


def demo_basic_functionality():
    """Demo basic MCTS functionality."""
    print("=== DEMO: Basic MCTS Functionality ===\n")
    
    # Create network and MCTS
    network = DemoNetwork(obs_size=128)
    mcts = MCTS(
        network=network,
        sample_size=30,
        action_size=5,
        policy_size=200,
        max_simulations=10
    )
    
    print(f"🔧 Initialized MCTS with:")
    print(f"   - Network: {type(network).__name__}")
    print(f"   - Sample size: {mcts.sample_size}")
    print(f"   - Max simulations: {mcts.max_simulations}")
    print(f"   - Policy size: {mcts.policy_size}")
    
    # Test enhanced state rollout
    observations = {
        "player_0": np.random.random(128),
        "player_1": np.random.random(128)
    }
    
    enhanced_state = EnhancedTFTState(
        observations=observations,
        current_player="player_0",
        network=network
    )
    
    rollout_value = enhanced_state.rollout()
    print(f"\n🎲 Enhanced rollout value: {rollout_value:.4f}")
    
    # Test statistics
    stats = mcts.get_statistics()
    print(f"📊 Initial statistics: {stats}")
    
    print("✅ Basic functionality demo completed\n")


def demo_action_generation():
    """Demo action generation with different scenarios."""
    print("=== DEMO: Action Generation ===\n")
    
    network = DemoNetwork()
    mcts = MCTS(network=network, max_simulations=5)
    
    # Mock TFT observations
    observations = {
        "player_0": np.random.random(128),
        "player_1": np.random.random(128),
        "player_2": np.random.random(128)
    }
    
    print("🎮 Generating actions for multiple scenarios...")
    
    # Scenario 1: Single player action
    action, policy, info = mcts.generate_action(observations, player_id="player_0")
    print(f"\n   Player 0 Action: {action}")
    print(f"   Policy shape: {policy.shape}")
    print(f"   Info: {info}")
    
    # Scenario 2: Different player
    action, policy, info = mcts.generate_action(observations, player_id="player_1")
    print(f"\n   Player 1 Action: {action}")
    print(f"   Move type: {info['move_type']}")
    print(f"   Confidence: {info['confidence']}")
    
    # Test move parsing
    print("\n🔍 Testing move parsing:")
    test_moves = ['reroll_shop', 'level_experience', 'buy_unit_3', 'sell_unit_1', 'unknown']
    for move in test_moves:
        parsed = mcts._parse_mcts_move(move)
        print(f"   '{move}' -> {parsed['action']} (type: {parsed['type']}, conf: {parsed['confidence']})")
    
    print("\n✅ Action generation demo completed\n")


def demo_network_integration():
    """Demo different network integration patterns."""
    print("=== DEMO: Network Integration Patterns ===\n")
    
    # Pattern 1: Value function network
    class ValueNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.value_head = nn.Linear(128, 1)
        
        def value_function(self, x):
            return torch.sigmoid(self.value_head(x))
    
    # Pattern 2: MuZero-style network
    class MuZeroNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.value_head = nn.Linear(128, 1)
            self.policy_head = nn.Linear(128, 10)
        
        def initial_inference(self, x):
            return {
                'value': torch.sigmoid(self.value_head(x)),
                'policy_logits': [self.policy_head(x)],
                'reward': torch.zeros(x.shape[0], 1)
            }
    
    # Pattern 3: Simple network
    class SimpleNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(128, 1)
        
        def forward(self, x):
            return torch.sigmoid(self.net(x))
    
    networks = [
        ("Value Function", ValueNetwork()),
        ("MuZero Style", MuZeroNetwork()),
        ("Simple Forward", SimpleNetwork())
    ]
    
    observations = {
        "player_0": np.random.random(128)
    }
    
    for name, network in networks:
        print(f"🧠 Testing {name} network:")
        mcts = MCTS(network=network, max_simulations=3)
        
        # Create enhanced state and test rollout
        state = EnhancedTFTState(
            observations=observations,
            current_player="player_0",
            network=network
        )
        
        rollout_value = state.rollout()
        print(f"   Rollout value: {rollout_value:.4f}")
        
        # Test action generation
        try:
            action, policy, info = mcts.generate_action(observations)
            print(f"   Generated action: {action}")
            print(f"   Confidence: {info.get('confidence', 'N/A')}")
        except Exception as e:
            print(f"   Action generation: {str(e)[:50]}...")
        
        print()
    
    print("✅ Network integration demo completed\n")


def demo_performance_tracking():
    """Demo performance tracking and statistics."""
    print("=== DEMO: Performance Tracking ===\n")
    
    network = DemoNetwork()
    mcts = MCTS(network=network)
    
    observations = {"player_0": np.random.random(128)}
    
    print("📈 Running multiple actions to track performance:")
    
    for i in range(5):
        action, policy, info = mcts.generate_action(observations)
        temp = mcts.visit_softmax_temperature()
        
        print(f"   Action {i+1}: {action} (temp: {temp:.3f}, confidence: {info.get('confidence', 'N/A')})")
    
    # Get final statistics
    stats = mcts.get_statistics()
    print(f"\n📊 Final statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n✅ Performance tracking demo completed\n")


def main():
    """Run all demos."""
    print("🚀 MCTS Torch Implementation Demo\n")
    print("This demo shows the updated MCTS implementation with:")
    print("- Neural network enhanced rollouts")
    print("- TFT MCTS bridge integration")
    print("- Multiple network architectures\n")
    
    try:
        demo_basic_functionality()
        demo_action_generation()
        demo_network_integration()
        demo_performance_tracking()
        
        print("🎉 All demos completed successfully!")
        print("\nThe updated MCTS torch implementation is ready for use with:")
        print("✅ Neural network enhanced rollouts")
        print("✅ TFT MCTS bridge integration")
        print("✅ Comprehensive action generation")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)