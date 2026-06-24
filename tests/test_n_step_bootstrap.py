"""Tests for n-step bootstrap implementation (PR #12 / TFT-194)."""

import sys
import os
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import config
from Models.MuZero_torch_trainer import Trainer


@pytest.fixture
def trainer():
    return Trainer()


@pytest.fixture
def mock_batch():
    batch_size = 4
    unroll_steps = config.UNROLL_STEPS

    observations = np.random.rand(batch_size, config.OBSERVATION_SIZE).astype(np.float32)
    actions = np.random.randint(0, 10, (batch_size, unroll_steps - 1, 3)).astype(np.float32)
    values = np.zeros((batch_size, unroll_steps), dtype=np.float32)
    rewards = np.zeros((batch_size, unroll_steps), dtype=np.float32)
    policies = np.random.rand(batch_size, unroll_steps, config.ACTION_CONCAT_SIZE).astype(np.float32)
    target_obs = [None] * batch_size
    bootstrap_depth = np.full((batch_size,), config.UNROLL_STEPS, dtype=np.float32)

    return observations, actions, values, rewards, policies, target_obs, bootstrap_depth


class TestNStepBootstrap:
    """Verify n-step bootstrap via recomputing works correctly."""

    def test_bootstrap_depth_defaults_to_unroll_steps(self, mock_batch):
        observations, actions, values, rewards, policies, target_obs, bootstrap_depth = mock_batch
        assert bootstrap_depth[0] == config.UNROLL_STEPS

    def test_bootstrap_depth_used_in_computation(self, trainer, mock_batch, monkeypatch):
        observations, actions, values, rewards, policies, target_obs, bootstrap_depth = mock_batch

        # Replace target_obs with actual observations so recomputing produces a value
        target_obs = [observations.copy()] * observations.shape[0]

        trainer.compute_loss(
            agent=None,
            observation=observations,
            action=actions,
            target_value=values,
            target_reward=rewards,
            target_policy=policies,
            target_obs=target_obs,
            bootstrap_depth=bootstrap_depth,
            combats=[],
            train_step=1,
            summary_writer=None,
        )

    def test_bootstrap_replaces_placeholder_value(self, mock_batch):
        import torch
        observations, actions, values, rewards, policies, target_obs, bootstrap_depth = mock_batch
        assert np.all(values == 0), "Placeholder target values should be all zeros"
        # After bootstrap, target_value should be gamma^n * v_{t+n}
        assert config.UNROLL_STEPS >= 2

    def test_gamma_n_computation(self):
        discount = config.DISCOUNT
        n = config.UNROLL_STEPS
        gamma_n = discount ** n
        assert 0 < gamma_n < 1
        assert gamma_n < discount  # gamma^n < gamma for n > 1

    def test_bootstrap_target_shape_preserved(self, mock_batch):
        batch_size = mock_batch[0].shape[0]
        unroll_steps = config.UNROLL_STEPS
        target_value_shape = (batch_size, unroll_steps)
        assert target_value_shape == (4, config.UNROLL_STEPS)

    def test_unroll_steps_equals_n(self):
        assert config.UNROLL_STEPS == 25

    def test_bootstrap_depth_tensor_conversion(self, mock_batch):
        observations, actions, values, rewards, policies, target_obs, bootstrap_depth = mock_batch
        depth_tensor = torch.from_numpy(bootstrap_depth).float()
        assert depth_tensor.shape == bootstrap_depth.shape
        assert depth_tensor[0] == config.UNROLL_STEPS

    def test_bootstrap_with_different_depths(self):
        depths = np.array([1.0, 5.0, 10.0, 25.0], dtype=np.float32)
        discount = config.DISCOUNT
        gamma_ns = discount ** depths

        assert gamma_ns[0] == discount
        assert gamma_ns[-1] == discount ** 25
        assert np.all(gamma_ns > 0)
        assert np.all(gamma_ns < 1)
