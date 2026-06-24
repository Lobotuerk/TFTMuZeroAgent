"""Tests for PredNetwork, DynNetwork, and RepNetwork constructor-based refactoring."""

import sys
import os
import numpy as np
import pytest
import torch

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import config
from Models.MuZero_torch_model import PredNetwork, DynNetwork, RepNetwork


@pytest.fixture
def device():
    return torch.device('cpu')


@pytest.fixture
def batch_size():
    return 2


class TestPredNetwork:
    """Tests for PredNetwork constructor parameters and output shapes."""

    def test_pred_network_instantiation(self, device):
        """Test PredNetwork can be instantiated with constructor params."""
        model = PredNetwork(
            input_size=config.HIDDEN_STATE_SIZE,
            layer_sizes=[config.LAYER_HIDDEN_SIZE] * 6,
            output_size=1,
            encoding_size=config.ENCODER_NUM_STEPS
        ).to(device)
        assert model is not None

    def test_pred_network_value_output_shape(self, device, batch_size):
        """Test PredNetwork value head outputs shape (batch, 1)."""
        model = PredNetwork(
            input_size=config.HIDDEN_STATE_SIZE,
            layer_sizes=[config.LAYER_HIDDEN_SIZE] * 6,
            output_size=1,
            encoding_size=config.ENCODER_NUM_STEPS
        ).to(device)
        x = torch.randn(batch_size, config.HIDDEN_STATE_SIZE, device=device)
        policy, value = model(x)
        assert value.shape == (batch_size, 1), f"Expected value shape (batch, 1), got {value.shape}"

    def test_pred_network_policy_output_shape(self, device, batch_size):
        """Test PredNetwork policy head outputs ACTION_CONCAT_SIZE."""
        model = PredNetwork(
            input_size=config.HIDDEN_STATE_SIZE,
            layer_sizes=[config.LAYER_HIDDEN_SIZE] * 6,
            output_size=1,
            encoding_size=config.ENCODER_NUM_STEPS
        ).to(device)
        x = torch.randn(batch_size, config.HIDDEN_STATE_SIZE, device=device)
        policy, value = model(x)
        assert policy.shape[-1] == config.ACTION_CONCAT_SIZE, \
            f"Expected policy last dim {config.ACTION_CONCAT_SIZE}, got {policy.shape[-1]}"

    def test_pred_network_residual_layers(self, device):
        """Test PredNetwork creates correct number of residual layers."""
        layer_sizes = [config.LAYER_HIDDEN_SIZE] * 6
        model = PredNetwork(
            input_size=config.HIDDEN_STATE_SIZE,
            layer_sizes=layer_sizes,
            output_size=1,
            encoding_size=config.ENCODER_NUM_STEPS
        ).to(device)
        assert len(model.res_layers) == 5, \
            f"Expected 5 residual layers for 6 layer_sizes, got {len(model.res_layers)}"
        assert len(model.res_lns) == 5, \
            f"Expected 5 layer norms for 6 layer_sizes, got {len(model.res_lns)}"

    def test_pred_network_forward_pass(self, device, batch_size):
        """Test PredNetwork forward pass produces valid outputs."""
        model = PredNetwork(
            input_size=config.HIDDEN_STATE_SIZE,
            layer_sizes=[config.LAYER_HIDDEN_SIZE] * 6,
            output_size=1,
            encoding_size=config.ENCODER_NUM_STEPS
        ).to(device)
        x = torch.randn(batch_size, config.HIDDEN_STATE_SIZE, device=device)
        policy, value = model(x)
        assert not torch.isnan(policy).any(), "Policy contains NaN"
        assert not torch.isnan(value).any(), "Value contains NaN"

    def test_pred_network_empty_layer_sizes(self, device):
        """Test PredNetwork defaults layer_sizes when empty."""
        model = PredNetwork(
            input_size=config.HIDDEN_STATE_SIZE,
            layer_sizes=[],
            output_size=1,
            encoding_size=config.ENCODER_NUM_STEPS
        ).to(device)
        x = torch.randn(1, config.HIDDEN_STATE_SIZE, device=device)
        policy, value = model(x)
        assert value.shape == (1, 1)

    def test_pred_network_embedding_tables(self, device):
        """Verify PredNetwork has no embedding tables (uses direct linear layers)."""
        model = PredNetwork(
            input_size=config.HIDDEN_STATE_SIZE,
            layer_sizes=[config.LAYER_HIDDEN_SIZE] * 6,
            output_size=1,
            encoding_size=config.ENCODER_NUM_STEPS
        ).to(device)
        has_embeddings = any(isinstance(m, torch.nn.Embedding) for m in model.modules())
        assert not has_embeddings, "PredNetwork should not contain Embedding layers"


class TestDynNetwork:
    """Tests for DynNetwork constructor parameters and output shapes."""

    def test_dyn_network_instantiation(self, device):
        """Test DynNetwork can be instantiated with constructor params."""
        input_size = config.HIDDEN_STATE_SIZE + config.ACTION_ENCODING_SIZE
        model = DynNetwork(
            input_size=input_size,
            layer_sizes=[config.LAYER_HIDDEN_SIZE] * 6,
            output_size=config.HIDDEN_STATE_SIZE,
            encoding_size=config.ENCODER_NUM_STEPS
        ).to(device)
        assert model is not None

    def test_dyn_network_output_shape(self, device, batch_size):
        """Test DynNetwork output matches HIDDEN_STATE_SIZE."""
        input_size = config.HIDDEN_STATE_SIZE + config.ACTION_ENCODING_SIZE
        model = DynNetwork(
            input_size=input_size,
            layer_sizes=[config.LAYER_HIDDEN_SIZE] * 6,
            output_size=config.HIDDEN_STATE_SIZE,
            encoding_size=config.ENCODER_NUM_STEPS
        ).to(device)
        hidden = torch.randn(batch_size, config.HIDDEN_STATE_SIZE, device=device)
        action = torch.randn(batch_size, 1, config.ACTION_ENCODING_SIZE, device=device)
        output = model(hidden, action)
        assert output.shape[1] == config.HIDDEN_STATE_SIZE, \
            f"Expected output dim {config.HIDDEN_STATE_SIZE}, got {output.shape[1]}"

    def test_dyn_network_consistent_hidden_dim(self, device):
        """Test DynNetwork residual layers maintain consistent hidden dimension."""
        input_size = config.HIDDEN_STATE_SIZE + config.ACTION_ENCODING_SIZE
        model = DynNetwork(
            input_size=input_size,
            layer_sizes=[config.LAYER_HIDDEN_SIZE] * 6,
            output_size=config.HIDDEN_STATE_SIZE,
            encoding_size=config.ENCODER_NUM_STEPS
        ).to(device)
        assert model.dense1.out_features == config.HIDDEN_STATE_SIZE

    def test_dyn_network_forward_pass(self, device, batch_size):
        """Test DynNetwork forward pass produces valid outputs."""
        input_size = config.HIDDEN_STATE_SIZE + config.ACTION_ENCODING_SIZE
        model = DynNetwork(
            input_size=input_size,
            layer_sizes=[config.LAYER_HIDDEN_SIZE] * 6,
            output_size=config.HIDDEN_STATE_SIZE,
            encoding_size=config.ENCODER_NUM_STEPS
        ).to(device)
        hidden = torch.randn(batch_size, config.HIDDEN_STATE_SIZE, device=device)
        action = torch.randn(batch_size, 1, config.ACTION_ENCODING_SIZE, device=device)
        output = model(hidden, action)
        assert not torch.isnan(output).any(), "DynNetwork output contains NaN"


class TestRepNetwork:
    """Tests for RepNetwork constructor parameters and output shapes."""

    def test_rep_network_instantiation(self, device):
        """Test RepNetwork can be instantiated with constructor params."""
        model = RepNetwork(
            input_size=config.OBSERVATION_SIZE,
            layer_sizes=[config.HIDDEN_STATE_SIZE] * 5,
            output_size=config.HIDDEN_STATE_SIZE,
            encoding_size=1
        ).to(device)
        assert model is not None

    def test_rep_network_output_shape(self, device, batch_size):
        """Test RepNetwork output matches HIDDEN_STATE_SIZE."""
        model = RepNetwork(
            input_size=config.OBSERVATION_SIZE,
            layer_sizes=[config.HIDDEN_STATE_SIZE] * 5,
            output_size=config.HIDDEN_STATE_SIZE,
            encoding_size=1
        ).to(device)
        x = torch.randn(batch_size, config.OBSERVATION_SIZE, device=device)
        output = model(x)
        assert output.shape[1] == config.HIDDEN_STATE_SIZE, \
            f"Expected output dim {config.HIDDEN_STATE_SIZE}, got {output.shape[1]}"

    def test_rep_network_embedding_tables(self, device):
        """Verify RepNetwork has correct embedding tables."""
        model = RepNetwork(
            input_size=config.OBSERVATION_SIZE,
            layer_sizes=[config.HIDDEN_STATE_SIZE] * 5,
            output_size=config.HIDDEN_STATE_SIZE,
            encoding_size=1
        ).to(device)
        assert hasattr(model, 'champion_embedding'), "Missing champion_embedding"
        assert hasattr(model, 'item_embedding'), "Missing item_embedding"
        assert hasattr(model, 'trait_embedding'), "Missing trait_embedding"
        assert hasattr(model, 'origin_embedding'), "Missing origin_embedding"

    def test_rep_network_embedding_dimensions(self, device):
        """Verify embedding table dimensions."""
        model = RepNetwork(
            input_size=config.OBSERVATION_SIZE,
            layer_sizes=[config.HIDDEN_STATE_SIZE] * 5,
            output_size=config.HIDDEN_STATE_SIZE,
            encoding_size=1
        ).to(device)
        assert model.champion_embedding.embedding_dim == 32
        assert model.champion_embedding.num_embeddings == 58
        assert model.item_embedding.embedding_dim == 24
        assert model.item_embedding.num_embeddings == 37
        assert model.trait_embedding.embedding_dim == 8
        assert model.trait_embedding.num_embeddings == 20
        assert model.origin_embedding.embedding_dim == 8
        assert model.origin_embedding.num_embeddings == 10

    def test_rep_network_forward_pass(self, device, batch_size):
        """Test RepNetwork forward pass produces valid outputs."""
        model = RepNetwork(
            input_size=config.OBSERVATION_SIZE,
            layer_sizes=[config.HIDDEN_STATE_SIZE] * 5,
            output_size=config.HIDDEN_STATE_SIZE,
            encoding_size=1
        ).to(device)
        x = torch.randn(batch_size, config.OBSERVATION_SIZE, device=device)
        output = model(x)
        assert not torch.isnan(output).any(), "RepNetwork output contains NaN"

    def test_rep_network_residual_layers(self, device):
        """Test RepNetwork creates correct number of residual layers."""
        layer_sizes = [config.HIDDEN_STATE_SIZE] * 5
        model = RepNetwork(
            input_size=config.OBSERVATION_SIZE,
            layer_sizes=layer_sizes,
            output_size=config.HIDDEN_STATE_SIZE,
            encoding_size=1
        ).to(device)
        assert len(model.res_layers) == 5
        assert len(model.res_lns) == 5

    def test_rep_network_dense1_input(self, device):
        """Test RepNetwork dense1 takes embedded_dim as input."""
        model = RepNetwork(
            input_size=config.OBSERVATION_SIZE,
            layer_sizes=[config.HIDDEN_STATE_SIZE] * 5,
            output_size=config.HIDDEN_STATE_SIZE,
            encoding_size=1
        ).to(device)
        assert model.dense1.in_features == model.embedded_dim
