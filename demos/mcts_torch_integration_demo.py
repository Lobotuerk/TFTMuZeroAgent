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

import config

from Models.MCTS_torch import EnhancedMCTS, TFTState
from Models.tft_mcts import create_tft_state_from_env


class DemoNetwork(nn.Module):
    """Demo neural network with multiple inference methods."""
    
    def __init__(self, obs_size=config.OBSERVATION_SIZE, hidden_size=64):
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
            nn.Linear(hidden_size, 1134)  # Matches ACTION_CONCAT_SIZE
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
        # Handle sequence of observations if necessary
        if len(x.shape) == 3: # (batch, seq, obs)
            x = x[:, -1, :] # Take last observation
            
        value = self.value_function(x)
        policy_logits = self.policy_net(x)
        
        return {
            'value': value,
            'policy_logits': policy_logits,
            'reward': torch.zeros_like(value),
            'hidden_state': x
        }
    
    def recurrent_inference(self, hidden_state, action):
        """MuZero-style recurrent inference."""
        return {
            'value': self.value_function(hidden_state),
            'policy_logits': self.policy_net(hidden_state),
            'reward': torch.zeros((hidden_state.shape[0], 1)),
            'hidden_state': hidden_state
        }
    
    def prediction(self, x):
        """Prediction for hidden states."""
        return self.policy_net(x), self.value_function(x)
    
    def training_steps(self):
        """Get training steps."""
        return self._training_steps


def demo_basic_functionality():
    """Demo basic MCTS functionality."""
    print("=== DEMO: Basic MCTS Functionality ===\n")
    
    # Create network and MCTS
    network = DemoNetwork()
    mcts = EnhancedMCTS(
        sample_size=30,
        action_size=3,
        action_limits=config.ACTION_DIM,
        policy_size=1134,
        network=network
    )
    
    print(f"🔧 Initialized MCTS with:")
    print(f"   - Network: {type(network).__name__}")
    print(f"   - Sample size: {mcts.sample_size}")
    print(f"   - Action size: {mcts.action_size}")
    print(f"   - Policy size: {mcts.policy_size}")
    
    # Test enhanced state rollout
    observation = np.random.random(config.OBSERVATION_SIZE)
    mask = np.zeros(sum(config.ACTION_DIM), dtype=bool)
    
    enhanced_state = TFTState(
        observation=observation,
        mask=mask,
        network=network
    )
    
    rollout_value = enhanced_state.rollout()
    print(f"\n🎲 Enhanced rollout value: {rollout_value:.4f}")
    
    # Test statistics
    stats = mcts.get_stats()
    print(f"📊 Initial statistics: {stats}")
    
    print("✅ Basic functionality demo completed\n")


def demo_action_generation():
    """Demo action generation with different scenarios."""
    print("=== DEMO: Action Generation ===\n")
    
    network = DemoNetwork()
    mcts = EnhancedMCTS(
        sample_size=30,
        action_size=3,
        action_limits=config.ACTION_DIM,
        policy_size=1134,
        network=network
    )
    
    # Mock TFT observation and mask
    observation = np.random.random(config.OBSERVATION_SIZE)
    mask = np.zeros(sum(config.ACTION_DIM), dtype=bool)
    
    print("🎮 Generating actions...")
    
    # Scenario: Generate action
    action, policy = mcts.generate_action(n_simulations=5, observation=observation, mask=mask)
    policy = np.array(policy)
    print(f"\n   Generated Action: {action}")
    print(f"   Policy shape: {policy.shape}")
    
    # Get stats
    stats = mcts.get_stats()
    print(f"\n📊 Generation statistics: {stats}")
    
    print("\n✅ Action generation demo completed\n")


def demo_network_integration():
    """Demo different network integration patterns."""
    print("=== DEMO: Network Integration Patterns ===\n")
    
    # Pattern 1: MuZero-style network (compatible with TFTState)
    class MuZeroNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.value_head = nn.Linear(config.OBSERVATION_SIZE, 1)
            self.policy_head = nn.Linear(config.OBSERVATION_SIZE, 1134)
            self.hidden_head = nn.Linear(config.OBSERVATION_SIZE, config.OBSERVATION_SIZE)
        
        def initial_inference(self, x):
            if len(x.shape) == 3: x = x[:, -1, :]
            return {
                'value': torch.sigmoid(self.value_head(x)),
                'policy_logits': self.policy_head(x),
                'reward': torch.zeros(x.shape[0], 1),
                'hidden_state': self.hidden_head(x)
            }
        
        def recurrent_inference(self, hidden_state, action):
            return {
                'value': torch.sigmoid(self.value_head(hidden_state)),
                'policy_logits': self.policy_head(hidden_state),
                'reward': torch.zeros(hidden_state.shape[0], 1),
                'hidden_state': self.hidden_head(hidden_state)
            }

    # Pattern 2: Prediction-style network
    class PredictionNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.value_head = nn.Linear(config.OBSERVATION_SIZE, 1)
            self.policy_head = nn.Linear(config.OBSERVATION_SIZE, 1134)
        
        def prediction(self, x):
            return self.policy_head(x), torch.sigmoid(self.value_head(x))
            
        def initial_inference(self, x):
            if len(x.shape) == 3: x = x[:, -1, :]
            return {
                'value': torch.sigmoid(self.value_head(x)),
                'policy_logits': self.policy_head(x),
                'reward': torch.zeros(x.shape[0], 1),
                'hidden_state': x
            }
            
        def recurrent_inference(self, hidden_state, action):
            return {'hidden_state': hidden_state}

    networks = [
        ("MuZero Style", MuZeroNetwork()),
        ("Prediction Style", PredictionNetwork())
    ]
    
    observation = np.random.random(config.OBSERVATION_SIZE)
    mask = np.zeros(sum(config.ACTION_DIM), dtype=bool)
    
    for name, network in networks:
        print(f"🧠 Testing {name} network:")
        mcts = EnhancedMCTS(
            sample_size=10, 
            action_size=3, 
            action_limits=config.ACTION_DIM, 
            policy_size=1134, 
            network=network
        )
        
        # Create state and test rollout
        state = TFTState(
            observation=observation,
            mask=mask,
            network=network
        )
        
        rollout_value = state.rollout()
        print(f"   Rollout value: {rollout_value:.4f}")
        
        # Test action generation
        try:
            action, policy = mcts.generate_action(n_simulations=3, observation=observation, mask=mask)
            print(f"   Generated action: {action}")
        except Exception as e:
            print(f"   Action generation failed: {str(e)[:100]}")
        
        print()
    
    print("✅ Network integration demo completed\n")


def demo_performance_tracking():
    """Demo performance tracking and statistics."""
    print("=== DEMO: Performance Tracking ===\n")
    
    network = DemoNetwork()
    mcts = EnhancedMCTS(
        sample_size=30,
        action_size=3,
        action_limits=config.ACTION_DIM,
        policy_size=1134,
        network=network
    )
    
    observation = np.random.random(config.OBSERVATION_SIZE)
    mask = np.zeros(sum(config.ACTION_DIM), dtype=bool)
    
    print("📈 Running multiple actions to track performance:")
    
    for i in range(5):
        action, policy = mcts.generate_action(n_simulations=2, observation=observation, mask=mask)
        print(f"   Action {i+1}: {action}")
    
    # Get final statistics
    stats = mcts.get_stats()
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