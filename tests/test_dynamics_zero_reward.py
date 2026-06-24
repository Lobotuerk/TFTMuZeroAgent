"""Tests for zero-reward synthesis in dynamics network (PR #12 / TFT-194)."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import config
import torch
from Models.MuZero_torch_model import MuZeroNetwork, DynNetwork


class TestZeroRewardSynthesis:
    """Verify rewards are synthesized as zeros in terminal-only environment."""

    def test_dynamics_returns_hidden_state_only(self):
        """DynNetwork.forward should return only hidden state, no reward."""
        input_size = config.HIDDEN_STATE_SIZE + config.ACTION_ENCODING_SIZE
        dyn = DynNetwork(
            input_size=input_size,
            layer_sizes=[config.LAYER_HIDDEN_SIZE] * 6,
            output_size=config.HIDDEN_STATE_SIZE,
            encoding_size=config.ACTION_CONCAT_SIZE,
        )
        batch_size = 4
        x = torch.randn(batch_size, input_size)
        action = torch.randn(batch_size, 1, config.ACTION_CONCAT_SIZE)
        output = dyn(x, action)
        assert output.shape == (batch_size, config.HIDDEN_STATE_SIZE)

    def test_muzero_dynamics_reward_is_zero(self):
        """MuZeroNetwork.dynamics should return zero reward."""
        model = MuZeroNetwork()
        batch_size = 4
        hidden = torch.randn(batch_size, config.HIDDEN_STATE_SIZE)
        action = np.random.randint(0, 10, (batch_size, 1, 3)).astype(np.float32)
        next_hidden, reward = model.dynamics(hidden, action)
        assert reward.shape == (batch_size, 1)
        assert torch.all(reward == 0)

    def test_initial_inference_reward_is_zero_array(self):
        """MuZeroNetwork.initial_inference should return zero reward array."""
        model = MuZeroNetwork()
        batch_size = 4
        obs = np.random.rand(batch_size, config.OBSERVATION_SIZE).astype(np.float32)
        outputs = model.initial_inference(obs)
        reward = outputs["reward"]
        assert isinstance(reward, np.ndarray)
        assert reward.shape == (batch_size,)
        assert np.all(reward == 0)

    def test_recurrent_inference_reward_is_zero(self):
        """MuZeroNetwork.recurrent_inference should return zero reward."""
        model = MuZeroNetwork()
        batch_size = 4
        obs = np.random.rand(batch_size, config.OBSERVATION_SIZE).astype(np.float32)
        init = model.initial_inference(obs)
        hidden = init["hidden_state"]
        action = np.random.randint(0, 10, (batch_size, 1, 3)).astype(np.float32)
        rec = model.recurrent_inference(hidden, action)
        reward = rec["reward"]
        if isinstance(reward, torch.Tensor):
            assert torch.all(reward == 0)
        else:
            assert np.all(reward == 0)

    def test_reward_device_matches_hidden_state(self):
        """Reward tensor/device should match hidden state device."""
        model = MuZeroNetwork()
        batch_size = 4
        hidden = torch.randn(batch_size, config.HIDDEN_STATE_SIZE)
        action = np.random.randint(0, 10, (batch_size, 1, 3)).astype(np.float32)
        next_hidden, reward = model.dynamics(hidden, action)
        assert reward.device == next_hidden.device

    def test_reward_batch_size_matches(self):
        """Reward batch size should match hidden state batch size."""
        model = MuZeroNetwork()
        for batch_size in [1, 4, 16]:
            hidden = torch.randn(batch_size, config.HIDDEN_STATE_SIZE)
            action = np.random.randint(0, 10, (batch_size, 1, 3)).astype(np.float32)
            next_hidden, reward = model.dynamics(hidden, action)
            assert reward.shape[0] == batch_size


class TestTerminalOnlyEnvironment:
    """Verify terminal-only environment assumptions."""

    def test_no_intermediate_rewards(self):
        """In terminal-only env, rewards should be zero at every step."""
        model = MuZeroNetwork()
        batch_size = 4
        obs = np.random.rand(batch_size, config.OBSERVATION_SIZE).astype(np.float32)
        outputs = model.initial_inference(obs)
        assert np.all(outputs["reward"] == 0)

    def test_config_reward_bounds(self):
        """MINIMUM_REWARD and MAXIMUM_REWARD should be set for terminal-only."""
        assert config.MINIMUM_REWARD == 0.0
        assert config.MAXIMUM_REWARD == 250.0

    def test_reward_head_removed_from_dyn_network(self):
        """DynNetwork should not have a reward output head."""
        input_size = config.HIDDEN_STATE_SIZE + config.ACTION_ENCODING_SIZE
        dyn = DynNetwork(
            input_size=input_size,
            layer_sizes=[config.LAYER_HIDDEN_SIZE] * 6,
            output_size=config.HIDDEN_STATE_SIZE,
            encoding_size=config.ACTION_CONCAT_SIZE,
        )
        # DynNetwork should only have linear layers for hidden state transformation
        layer_count = sum(1 for attr in dir(dyn) if attr.startswith('dense'))
        assert layer_count > 0
        # Should not have a separate reward head
        assert not hasattr(dyn, 'reward_head')
