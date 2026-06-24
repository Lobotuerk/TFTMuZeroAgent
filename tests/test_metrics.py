"""Tests for metrics: policy entropy and value MAE."""

import sys
import os
import numpy as np
import pytest
import torch

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import config


def policy_entropy(logits):
    """Compute policy entropy: H(p) = -sum(p * log(p))."""
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-10)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy


def value_mae(value, target):
    """Compute value Mean Absolute Error."""
    return torch.mean(torch.abs(torch.squeeze(value) - torch.squeeze(target)))


class TestPolicyEntropy:
    """Tests for policy entropy computation."""

    def test_entropy_non_negative(self):
        """Policy entropy should always be non-negative."""
        for _ in range(10):
            logits = torch.randn(32)
            ent = policy_entropy(logits)
            assert ent >= 0, f"Entropy {ent} should be non-negative"

    def test_entropy_uniform_distribution(self):
        """Uniform distribution should produce maximum entropy."""
        num_actions = 81
        logits = torch.zeros(num_actions)
        ent = policy_entropy(logits)
        expected = np.log(num_actions)
        assert torch.allclose(ent, torch.tensor(expected, dtype=torch.float32), atol=1e-4), \
            f"Uniform entropy {ent} should be ~{expected}"

    def test_entropy_deterministic_distribution(self):
        """Deterministic distribution (one hot) should produce near-zero entropy."""
        logits = torch.tensor([-100.0] * 80 + [0.0])
        ent = policy_entropy(logits)
        assert ent < 0.01, f"Deterministic entropy {ent} should be near zero"

    def test_entropy_more_actions_higher(self):
        """More actions with uniform distribution should give higher entropy."""
        ent_10 = policy_entropy(torch.zeros(10))
        ent_100 = policy_entropy(torch.zeros(100))
        assert ent_100 > ent_10, "More uniform actions should yield higher entropy"

    def test_entropy_batch_mean(self):
        """Test entropy computation over a batch with mean aggregation."""
        batch_logits = torch.randn(4, 81)
        entropies = policy_entropy(batch_logits)
        mean_entropy = entropies.mean()
        assert mean_entropy >= 0, "Batch mean entropy should be non-negative"

    def test_entropy_continuous_with_temperature(self):
        """Entropy should increase as logits approach zero (higher temperature)."""
        base_logits = torch.randn(32)
        ent_cold = policy_entropy(base_logits * 10)
        ent_warm = policy_entropy(base_logits)
        assert ent_warm > ent_cold, "Warmer distribution should have higher entropy"

    def test_entropy_shape(self):
        """Entropy should return 1D tensor matching batch size."""
        logits = torch.randn(5, 81)
        ent = policy_entropy(logits)
        assert ent.shape == (5,), f"Expected shape (5,), got {ent.shape}"


class TestValueMAE:
    """Tests for value Mean Absolute Error computation."""

    def test_value_mae_perfect_prediction(self):
        """MAE should be zero for perfect predictions."""
        value = torch.tensor([[0.5], [0.3], [0.8]])
        target = torch.tensor([[0.5], [0.3], [0.8]])
        mae = value_mae(value, target)
        assert torch.allclose(mae, torch.tensor(0.0)), "Perfect prediction MAE should be 0"

    def test_value_mae_constant_error(self):
        """MAE should equal the constant error value."""
        value = torch.tensor([[0.5], [0.5], [0.5]])
        target = torch.tensor([[0.7], [0.7], [0.7]])
        mae = value_mae(value, target)
        assert torch.allclose(mae, torch.tensor(0.2)), f"Expected MAE 0.2, got {mae}"

    def test_value_mae_symmetric(self):
        """MAE should be symmetric around zero error."""
        value = torch.tensor([[0.5]])
        mae_plus = value_mae(value, torch.tensor([[0.7]]))
        mae_minus = value_mae(torch.tensor([[0.7]]), value)
        assert torch.allclose(mae_plus, mae_minus), "MAE should be symmetric"

    def test_value_mae_batch(self):
        """MAE should correctly average over a batch."""
        value = torch.tensor([[0.0], [1.0]])
        target = torch.tensor([[1.0], [0.0]])
        mae = value_mae(value, target)
        # Errors are 1.0 and 1.0, mean = 1.0
        assert torch.allclose(mae, torch.tensor(1.0)), f"Expected MAE 1.0, got {mae}"

    def test_value_mae_range(self):
        """MAE should be in a reasonable range for bounded values."""
        value = torch.rand(10, 1)
        target = torch.rand(10, 1)
        mae = value_mae(value, target)
        assert 0 <= mae <= 1.0, f"MAE {mae} should be in [0, 1] for values in [0, 1]"
