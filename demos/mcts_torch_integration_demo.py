#!/usr/bin/env python3
"""
Demo showing the updated MCTS torch implementation with TFT MCTS bridge integration.
"""

import numpy as np
import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Models.MCTS_torch import EnhancedMCTS, TFTState


class DemoNetwork(nn.Module):
    """Demo neural network with multiple inference methods."""

    def __init__(self, obs_size=5152, hidden_size=64):
        super().__init__()
        self.obs_size = obs_size
        self.hidden_size = hidden_size

        self.value_net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.policy_net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 111)
        )

        self._training_steps = 100

    def forward(self, x):
        return torch.sigmoid(self.value_net(x))

    def value_function(self, x):
        return torch.sigmoid(self.value_net(x))

    def initial_inference(self, x):
        value = self.value_function(x)
        policy_logits = self.policy_net(x)

        return {
            'value': value,
            'policy_logits': policy_logits,
            'reward': torch.zeros_like(value),
            'hidden_state': x
        }

    def recurrent_inference(self, x, action_encoding):
        return self.initial_inference(x)

    def training_steps(self):
        return self._training_steps


def demo_basic_functionality():
    """Demo basic MCTS functionality."""
    print("=== DEMO: Basic MCTS Functionality ===\n")

    network = DemoNetwork(obs_size=5152)
    mcts = EnhancedMCTS(
        sample_size=30,
        action_size=3,
        action_limits=[7, 37, 10],
        policy_size=111,
        network=network
    )

    print(f"Initialized EnhancedMCTS with:")
    print(f"   - Network: {type(network).__name__}")
    print(f"   - Sample size: {mcts.sample_size}")
    print(f"   - Action limits: {mcts.action_limits}")

    observation = np.random.random(5152).astype(np.float32)
    mask = np.ones(54, dtype=bool)

    tft_state = TFTState(
        observation=observation,
        mask=mask,
        network=network
    )

    rollout_value = tft_state.rollout()
    print(f"\nRollout value: {rollout_value}")

    stats = mcts.get_stats()
    print(f"Initial statistics: {stats}")

    print("Basic functionality demo completed\n")


def demo_action_generation():
    """Demo action generation with different scenarios."""
    print("=== DEMO: Action Generation ===\n")

    network = DemoNetwork()
    mcts = EnhancedMCTS(
        sample_size=16,
        action_size=3,
        action_limits=[7, 37, 10],
        policy_size=111,
        network=network
    )

    observation = np.random.random(5152).astype(np.float32)
    mask = np.ones(54, dtype=bool)

    print("Generating actions...")

    actions, target_policies = mcts.generate_action(
        n_simulations=5,
        observation=observation,
        mask=mask
    )

    print(f"\nActions: {actions}")
    print(f"Target policies (first 10): {target_policies[:10]}...")

    stats = mcts.get_stats()
    print(f"\nStatistics: {stats}")

    print("\nAction generation demo completed\n")


def demo_network_integration():
    """Demo different network integration patterns."""
    print("=== DEMO: Network Integration Patterns ===\n")

    class ValueNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.value_head = nn.Linear(5152, 1)

        def value_function(self, x):
            return torch.sigmoid(self.value_head(x))

        def initial_inference(self, x):
            return {
                'value': self.value_function(x),
                'policy_logits': torch.zeros(x.shape[0], 111),
                'reward': torch.zeros(x.shape[0], 1),
                'hidden_state': x
            }

        def recurrent_inference(self, x, action_encoding):
            return self.initial_inference(x)

    class SimpleNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(5152, 1)

        def forward(self, x):
            return torch.sigmoid(self.net(x))

        def initial_inference(self, x):
            return {
                'value': torch.sigmoid(self.net(x)),
                'policy_logits': torch.zeros(x.shape[0], 111),
                'reward': torch.zeros(x.shape[0], 1),
                'hidden_state': x
            }

        def recurrent_inference(self, x, action_encoding):
            return self.initial_inference(x)

    networks = [
        ("Value Function", ValueNetwork()),
        ("Simple Forward", SimpleNetwork())
    ]

    observation = np.random.random(5152).astype(np.float32)
    mask = np.ones(54, dtype=bool)

    for name, network in networks:
        print(f"Testing {name} network:")
        mcts = EnhancedMCTS(
            sample_size=16,
            action_size=3,
            action_limits=[7, 37, 10],
            policy_size=111,
            network=network
        )

        state = TFTState(
            observation=observation,
            mask=mask,
            network=network
        )

        rollout_value = state.rollout()
        print(f"   Rollout value: {rollout_value}")

        try:
            actions, target_policies = mcts.generate_action(
                n_simulations=3,
                observation=observation,
                mask=mask
            )
            print(f"   Generated actions: {actions}")
        except Exception as e:
            print(f"   Action generation: {str(e)[:50]}...")

        print()

    print("Network integration demo completed\n")


def demo_performance_tracking():
    """Demo performance tracking and statistics."""
    print("=== DEMO: Performance Tracking ===\n")

    network = DemoNetwork()
    mcts = EnhancedMCTS(
        sample_size=16,
        action_size=3,
        action_limits=[7, 37, 10],
        policy_size=111,
        network=network
    )

    observation = np.random.random(5152).astype(np.float32)
    mask = np.ones(54, dtype=bool)

    print("Running multiple actions to track performance:")

    for i in range(3):
        try:
            actions, target_policies = mcts.generate_action(
                n_simulations=5,
                observation=observation,
                mask=mask
            )
            print(f"   Generation {i+1}: actions={actions}")
        except Exception as e:
            print(f"   Generation {i+1}: {str(e)[:50]}...")

    stats = mcts.get_stats()
    print(f"\nFinal statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print("\nPerformance tracking demo completed\n")


def main():
    """Run all demos."""
    print("MCTS Torch Implementation Demo\n")
    print("This demo shows the updated MCTS implementation with:")
    print("- Neural network enhanced rollouts")
    print("- TFT MCTS bridge integration")
    print("- Multiple network architectures\n")

    try:
        demo_basic_functionality()
        demo_action_generation()
        demo_network_integration()
        demo_performance_tracking()

        print("All demos completed successfully!")
        print("\nThe updated MCTS torch implementation is ready for use with:")
        print("- Neural network enhanced rollouts")
        print("- TFT MCTS bridge integration")
        print("- Comprehensive action generation")

    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
