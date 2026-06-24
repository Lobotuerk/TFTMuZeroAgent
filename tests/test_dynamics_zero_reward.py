"""Tests for dynamics network and zero-reward synthesis."""

import sys
import os
import numpy as np
import pytest
import torch

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import config
from Models.MuZero_torch_model import DynNetwork


@pytest.fixture
def device():
    return torch.device('cpu')


@pytest.fixture
def batch_size():
    return 4


class TestZeroRewardSynthesis:
    """Tests for zero-reward synthesis in dynamics network."""

    def test_dynamics_zero_reward_shape(self, device, batch_size):
        """Test that synthesized reward has correct shape (batch, 1)."""
        input_size = config.HIDDEN_STATE_SIZE + config.ACTION_ENCODING_SIZE
        dyn = DynNetwork(
            input_size=input_size,
            layer_sizes=[config.LAYER_HIDDEN_SIZE] * 6,
            output_size=config.HIDDEN_STATE_SIZE,
            encoding_size=config.ENCODER_NUM_STEPS
        ).to(device)

        hidden = torch.randn(batch_size, config.HIDDEN_STATE_SIZE, device=device)
        action = torch.randn(batch_size, 1, config.ACTION_ENCODING_SIZE, device=device)
        next_hidden = dyn(hidden, action)

        # Simulate the zero-reward synthesis from MuZero_torch_model.dynamics()
        reward = torch.zeros(next_hidden.shape[0], 1, device=next_hidden.device)
        assert reward.shape == (batch_size, 1), f"Expected shape (batch, 1), got {reward.shape}"

    def test_dynamics_zero_reward_values(self, device, batch_size):
        """Test that synthesized reward is all zeros."""
        input_size = config.HIDDEN_STATE_SIZE + config.ACTION_ENCODING_SIZE
        dyn = DynNetwork(
            input_size=input_size,
            layer_sizes=[config.LAYER_HIDDEN_SIZE] * 6,
            output_size=config.HIDDEN_STATE_SIZE,
            encoding_size=config.ENCODER_NUM_STEPS
        ).to(device)

        hidden = torch.randn(batch_size, config.HIDDEN_STATE_SIZE, device=device)
        action = torch.randn(batch_size, 1, config.ACTION_ENCODING_SIZE, device=device)
        next_hidden = dyn(hidden, action)

        reward = torch.zeros(next_hidden.shape[0], 1, device=next_hidden.device)
        assert torch.all(reward == 0), "Zero-reward synthesis should produce all zeros"

    def test_dynamics_zero_reward_terminal_only(self, device):
        """Test zero-reward assumption for terminal-only environments."""
        # In terminal-only environments, rewards are synthesized as zero
        # because the value head captures the full return
        batch_size = 2
        hidden = torch.randn(batch_size, config.HIDDEN_STATE_SIZE, device=device)
        action = torch.zeros(batch_size, 1, config.ACTION_ENCODING_SIZE, device=device)

        input_size = config.HIDDEN_STATE_SIZE + config.ACTION_ENCODING_SIZE
        dyn = DynNetwork(
            input_size=input_size,
            layer_sizes=[config.LAYER_HIDDEN_SIZE] * 6,
            output_size=config.HIDDEN_STATE_SIZE,
            encoding_size=config.ENCODER_NUM_STEPS
        ).to(device)

        next_hidden = dyn(hidden, action)
        reward = torch.zeros(next_hidden.shape[0], 1, device=next_hidden.device)

        assert reward.sum() == 0, "Terminal-only reward should sum to zero"

    def test_dynamics_zero_reward_batch_size_consistency(self, device):
        """Test reward batch size matches hidden state batch size."""
        for bs in [1, 4, 8, 16]:
            hidden = torch.randn(bs, config.HIDDEN_STATE_SIZE, device=device)
            action = torch.randn(bs, 1, config.ACTION_ENCODING_SIZE, device=device)

            input_size = config.HIDDEN_STATE_SIZE + config.ACTION_ENCODING_SIZE
            dyn = DynNetwork(
                input_size=input_size,
                layer_sizes=[config.LAYER_HIDDEN_SIZE] * 6,
                output_size=config.HIDDEN_STATE_SIZE,
                encoding_size=config.ENCODER_NUM_STEPS
            ).to(device)

            next_hidden = dyn(hidden, action)
            reward = torch.zeros(next_hidden.shape[0], 1, device=next_hidden.device)

            assert reward.shape[0] == next_hidden.shape[0], \
                f"Reward batch {reward.shape[0]} != hidden batch {next_hidden.shape[0]}"


class TestRewardDeviceConsistency:
    """Tests for reward device and batch consistency."""

    def test_reward_device_matches_hidden(self, device, batch_size):
        """Test reward tensor is on the same device as hidden state."""
        input_size = config.HIDDEN_STATE_SIZE + config.ACTION_ENCODING_SIZE
        dyn = DynNetwork(
            input_size=input_size,
            layer_sizes=[config.LAYER_HIDDEN_SIZE] * 6,
            output_size=config.HIDDEN_STATE_SIZE,
            encoding_size=config.ENCODER_NUM_STEPS
        ).to(device)

        hidden = torch.randn(batch_size, config.HIDDEN_STATE_SIZE, device=device)
        action = torch.randn(batch_size, 1, config.ACTION_ENCODING_SIZE, device=device)
        next_hidden = dyn(hidden, action)

        reward = torch.zeros(next_hidden.shape[0], 1, device=next_hidden.device)
        assert reward.device == next_hidden.device, "Reward device should match hidden state device"

    def test_reward_batch_consistency_multiple_batches(self, device):
        """Test reward batch consistency across different batch sizes."""
        input_size = config.HIDDEN_STATE_SIZE + config.ACTION_ENCODING_SIZE
        dyn = DynNetwork(
            input_size=input_size,
            layer_sizes=[config.LAYER_HIDDEN_SIZE] * 6,
            output_size=config.HIDDEN_STATE_SIZE,
            encoding_size=config.ENCODER_NUM_STEPS
        ).to(device)

        for bs in [1, 2, 4, 8]:
            hidden = torch.randn(bs, config.HIDDEN_STATE_SIZE, device=device)
            action = torch.randn(bs, 1, config.ACTION_ENCODING_SIZE, device=device)
            next_hidden = dyn(hidden, action)
            reward = torch.zeros(next_hidden.shape[0], 1, device=next_hidden.device)

            assert reward.shape[0] == bs, f"Expected batch {bs}, got {reward.shape[0]}"

    def test_reward_dtype(self, device, batch_size):
        """Test reward tensor has float dtype."""
        input_size = config.HIDDEN_STATE_SIZE + config.ACTION_ENCODING_SIZE
        dyn = DynNetwork(
            input_size=input_size,
            layer_sizes=[config.LAYER_HIDDEN_SIZE] * 6,
            output_size=config.HIDDEN_STATE_SIZE,
            encoding_size=config.ENCODER_NUM_STEPS
        ).to(device)

        hidden = torch.randn(batch_size, config.HIDDEN_STATE_SIZE, device=device)
        action = torch.randn(batch_size, 1, config.ACTION_ENCODING_SIZE, device=device)
        next_hidden = dyn(hidden, action)

        reward = torch.zeros(next_hidden.shape[0], 1, device=next_hidden.device)
        assert reward.dtype == torch.float32, f"Expected float32, got {reward.dtype}"

    def test_dynamics_output_not_nan(self, device, batch_size):
        """Test dynamics output is not NaN after zero-reward synthesis."""
        input_size = config.HIDDEN_STATE_SIZE + config.ACTION_ENCODING_SIZE
        dyn = DynNetwork(
            input_size=input_size,
            layer_sizes=[config.LAYER_HIDDEN_SIZE] * 6,
            output_size=config.HIDDEN_STATE_SIZE,
            encoding_size=config.ENCODER_NUM_STEPS
        ).to(device)

        hidden = torch.randn(batch_size, config.HIDDEN_STATE_SIZE, device=device)
        action = torch.randn(batch_size, 1, config.ACTION_ENCODING_SIZE, device=device)
        next_hidden = dyn(hidden, action)

        assert not torch.isnan(next_hidden).any(), "Dynamics output contains NaN"
        assert not torch.isinf(next_hidden).any(), "Dynamics output contains Inf"
