#!/usr/bin/env python3
"""
Integration test for MuZeroNetwork recurrent_inference end-to-end.

Validates that the DynNetwork dimension mismatch fix (TFT-195) works correctly
by exercising the full recurrent_inference pipeline with dummy inputs.
"""

import sys
import os
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import config
from Models.MuZero_torch_model import MuZeroNetwork


def test_recurrent_inference_no_dim_mismatch():
    """Verify recurrent_inference does not raise dimension mismatch in DynNetwork."""
    torch.manual_seed(42)
    np.random.seed(42)

    network = MuZeroNetwork()
    network.eval()

    batch_size = 2
    obs = torch.randn(batch_size, config.OBSERVATION_SIZE)

    result = network.initial_inference(obs)
    hidden_state = result["hidden_state"]
    assert hidden_state.shape == (batch_size, config.HIDDEN_STATE_SIZE), \
        f"Expected hidden_state shape {(batch_size, config.HIDDEN_STATE_SIZE)}, got {hidden_state.shape}"

    action = np.zeros((batch_size, 3), dtype=np.float32)

    rec_result = network.recurrent_inference(hidden_state, action)

    assert "value" in rec_result
    assert "reward" in rec_result
    assert "policy_logits" in rec_result
    assert "hidden_state" in rec_result

    assert rec_result["hidden_state"].shape == (batch_size, config.HIDDEN_STATE_SIZE), \
        f"Expected next_hidden_state shape {(batch_size, config.HIDDEN_STATE_SIZE)}, got {rec_result['hidden_state'].shape}"

    assert rec_result["reward"].shape == (batch_size, 1), \
        f"Expected reward shape {(batch_size, 1)}, got {rec_result['reward'].shape}"

    print(f"PASS: recurrent_inference succeeded with batch_size={batch_size}")
    print(f"  hidden_state: {rec_result['hidden_state'].shape}")
    print(f"  reward: {rec_result['reward'].shape}")
    print(f"  value: {rec_result['value'].shape}")
    print(f"  policy_logits: {rec_result['policy_logits'].shape}")


def test_recurrent_inference_multiple_steps():
    """Run multiple recurrent steps to verify DynNetwork stability across rollouts."""
    torch.manual_seed(123)
    np.random.seed(123)

    network = MuZeroNetwork()
    network.eval()

    batch_size = 1
    obs = torch.randn(batch_size, config.OBSERVATION_SIZE)

    result = network.initial_inference(obs)
    hidden_state = result["hidden_state"]

    for step in range(5):
        action = np.zeros((batch_size, 3), dtype=np.float32)
        result = network.recurrent_inference(hidden_state, action)
        hidden_state = result["hidden_state"]
        assert hidden_state.shape == (batch_size, config.HIDDEN_STATE_SIZE)

    print(f"PASS: 5 recurrent steps completed without error")


if __name__ == "__main__":
    test_recurrent_inference_no_dim_mismatch()
    test_recurrent_inference_multiple_steps()
    print("\nAll tests passed.")
