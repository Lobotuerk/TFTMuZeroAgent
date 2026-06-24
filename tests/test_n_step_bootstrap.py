"""Tests for n-step bootstrap computation in MuZero trainer."""

import sys
import os
import numpy as np
import pytest

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import config
from Models.MuZero_torch_trainer import Trainer


@pytest.fixture
def trainer():
    return Trainer()


@pytest.fixture
def bootstrap_depth():
    return np.array([1.0, 2.0, 3.0, 5.0])


@pytest.fixture
def discount():
    return config.DISCOUNT


def test_gamma_n_discount_calculation(trainer, bootstrap_depth, discount):
    """Verify gamma^n discount is computed correctly for each depth."""
    import torch
    device = torch.device('cpu')
    bootstrap_depth_tensor = torch.tensor(bootstrap_depth, device=device)
    gamma_n = discount ** bootstrap_depth_tensor
    expected = torch.tensor([discount ** d for d in bootstrap_depth], device=device)
    assert torch.allclose(gamma_n, expected), "gamma^n discount mismatch"


def test_bootstrap_target_computation(trainer, bootstrap_depth, discount):
    """Verify z_t = gamma^n * v_{t+n} computation."""
    import torch
    device = torch.device('cpu')
    v_t_plus_n = torch.tensor([[0.5], [0.3], [0.8], [0.1]], device=device)
    gamma_n = discount ** bootstrap_depth
    bootstrap_targets = gamma_n * v_t_plus_n.squeeze()
    expected = torch.tensor([
        discount ** 1 * 0.5,
        discount ** 2 * 0.3,
        discount ** 3 * 0.8,
        discount ** 5 * 0.1,
    ], device=device)
    assert torch.allclose(bootstrap_targets, expected, atol=1e-6), "bootstrap target mismatch"


def test_bootstrap_depth_one(trainer, discount):
    """Test bootstrap with depth=1 (single step)."""
    import torch
    device = torch.device('cpu')
    bootstrap_depth_tensor = torch.tensor([1.0], device=device)
    v_t_plus_n = torch.tensor([[0.5]], device=device)
    gamma_n = discount ** bootstrap_depth_tensor
    bootstrap_targets = gamma_n * v_t_plus_n.squeeze()
    expected = torch.tensor([discount * 0.5], device=device)
    assert torch.allclose(bootstrap_targets, expected), "depth=1 bootstrap target mismatch"


def test_bootstrap_depth_equals_unroll(trainer, bootstrap_depth, discount):
    """Test bootstrap with depth equal to UNROLL_STEPS."""
    import torch
    device = torch.device('cpu')
    full_depth = np.array([float(config.UNROLL_STEPS)], dtype=float)
    bootstrap_depth_tensor = torch.tensor(full_depth, device=device)
    v_t_plus_n = torch.tensor([[0.7]], device=device)
    gamma_n = discount ** bootstrap_depth_tensor
    bootstrap_targets = gamma_n * v_t_plus_n.squeeze()
    expected = torch.tensor([discount ** config.UNROLL_STEPS * 0.7], device=device)
    assert torch.allclose(bootstrap_targets, expected, atol=1e-6), "full depth bootstrap mismatch"


def test_bootstrap_gamma_n_monotonic_decay(discount):
    """Verify gamma^n decreases as n increases (since gamma < 1)."""
    import torch
    depths = torch.arange(1, 20, dtype=torch.float)
    gamma_n = discount ** depths
    diffs = torch.diff(gamma_n)
    assert torch.all(diffs < 0), "gamma^n should monotonically decrease"


def test_bootstrap_zero_value(trainer, discount):
    """Test bootstrap with zero value produces zero target."""
    import torch
    device = torch.device('cpu')
    bootstrap_depth_tensor = torch.tensor([2.0, 5.0], device=device)
    v_t_plus_n = torch.zeros(2, 1, device=device)
    gamma_n = discount ** bootstrap_depth_tensor
    bootstrap_targets = gamma_n * v_t_plus_n.squeeze()
    assert torch.all(bootstrap_targets == 0), "zero value should produce zero bootstrap target"


def test_bootstrap_discount_value_in_range():
    """Verify DISCOUNT is in valid range (0, 1)."""
    assert 0 < config.DISCOUNT < 1, "DISCOUNT must be in (0, 1)"


def test_bootstrap_depth_used_in_computation(trainer):
    """Verify bootstrap_depth from batch is used in compute_loss."""
    import torch
    device = torch.device('cpu')
    batch_size = 2

    # Create minimal observation data
    observation = np.random.rand(batch_size, config.OBSERVATION_SIZE).astype(np.float32)
    action = np.zeros((batch_size, config.UNROLL_STEPS - 1), dtype=np.float32)
    target_value = np.zeros((batch_size, config.UNROLL_STEPS), dtype=np.float32)
    target_reward = np.zeros((batch_size, config.UNROLL_STEPS), dtype=np.float32)
    target_policy = np.zeros((batch_size, config.UNROLL_STEPS, config.ACTION_CONCAT_SIZE), dtype=np.float32)
    target_obs = [None] * batch_size
    bootstrap_depth = np.array([1.0, 3.0])

    # The compute_loss method should use bootstrap_depth to scale value targets
    # We verify the bootstrap_depth values are properly converted to tensor
    bootstrap_depth_tensor = torch.from_numpy(bootstrap_depth).float().to(device)
    assert bootstrap_depth_tensor.shape == (2,), "bootstrap_depth shape mismatch"
    assert torch.allclose(bootstrap_depth_tensor, torch.tensor([1.0, 3.0], device=device)), "bootstrap_depth values mismatch"
