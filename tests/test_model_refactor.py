"""Tests for model constructor refactoring (PR #12 / TFT-194)."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import config
import torch
from Models.MuZero_torch_model import PredNetwork, DynNetwork, RepNetwork, MuZeroNetwork


class TestPredNetworkConstructor:
    """Verify PredNetwork uses constructor parameters."""

    def test_pred_network_accepts_parameters(self):
        net = PredNetwork(
            input_size=config.HIDDEN_STATE_SIZE,
            layer_sizes=[config.LAYER_HIDDEN_SIZE] * 6,
            output_size=config.ACTION_CONCAT_SIZE,
            encoding_size=config.ACTION_CONCAT_SIZE,
        )
        assert net is not None

    def test_pred_network_output_shape(self):
        net = PredNetwork(
            input_size=config.HIDDEN_STATE_SIZE,
            layer_sizes=[config.LAYER_HIDDEN_SIZE] * 6,
            output_size=config.ACTION_CONCAT_SIZE,
            encoding_size=config.ACTION_CONCAT_SIZE,
        )
        batch_size = 4
        x = torch.randn(batch_size, config.HIDDEN_STATE_SIZE)
        policy, value = net(x)
        assert policy.shape == (batch_size, config.ACTION_CONCAT_SIZE)
        assert value.shape == (batch_size, 1)

    def test_pred_network_default_layer_sizes(self):
        hidden = config.HIDDEN_STATE_SIZE
        net = PredNetwork(
            input_size=hidden,
            layer_sizes=[],
            output_size=10,
            encoding_size=10,
        )
        assert len(net.res_layers) > 0

    def test_pred_network_value_head_size(self):
        net = PredNetwork(
            input_size=config.HIDDEN_STATE_SIZE,
            layer_sizes=[config.LAYER_HIDDEN_SIZE] * 6,
            output_size=1,
            encoding_size=config.ACTION_CONCAT_SIZE,
        )
        batch_size = 2
        x = torch.randn(batch_size, config.HIDDEN_STATE_SIZE)
        _, value = net(x)
        assert value.shape == (batch_size, 1)


class TestDynNetworkConstructor:
    """Verify DynNetwork uses constructor parameters and has no reward head."""

    def test_dyn_network_accepts_parameters(self):
        input_size = config.HIDDEN_STATE_SIZE
        net = DynNetwork(
            input_size=input_size,
            layer_sizes=[config.LAYER_HIDDEN_SIZE] * 6,
            output_size=config.HIDDEN_STATE_SIZE,
            encoding_size=config.ACTION_CONCAT_SIZE,
        )
        assert net is not None

    def test_dyn_network_output_is_hidden_state_only(self):
        input_size = config.HIDDEN_STATE_SIZE
        net = DynNetwork(
            input_size=input_size,
            layer_sizes=[config.LAYER_HIDDEN_SIZE] * 6,
            output_size=config.HIDDEN_STATE_SIZE,
            encoding_size=config.ACTION_CONCAT_SIZE,
        )
        batch_size = 4
        x = torch.randn(batch_size, input_size)
        action = torch.randn(batch_size, 1, config.ACTION_CONCAT_SIZE)
        output = net(x, action)
        # DynNetwork returns only hidden state, no reward
        assert output.shape == (batch_size, config.HIDDEN_STATE_SIZE)

    def test_dyn_network_no_reward_head_attribute(self):
        input_size = config.HIDDEN_STATE_SIZE
        net = DynNetwork(
            input_size=input_size,
            layer_sizes=[config.LAYER_HIDDEN_SIZE] * 6,
            output_size=config.HIDDEN_STATE_SIZE,
            encoding_size=config.ACTION_CONCAT_SIZE,
        )
        assert not hasattr(net, 'reward_head')


class TestRepNetworkConstructor:
    """Verify RepNetwork uses constructor parameters with embeddings."""

    def test_rep_network_accepts_parameters(self):
        net = RepNetwork(
            input_size=config.OBSERVATION_SIZE,
            layer_sizes=[config.HIDDEN_STATE_SIZE] * 5,
            output_size=config.HIDDEN_STATE_SIZE,
            encoding_size=122,
        )
        assert net is not None

    def test_rep_network_has_embedding_tables(self):
        net = RepNetwork(
            input_size=config.OBSERVATION_SIZE,
            layer_sizes=[config.HIDDEN_STATE_SIZE] * 5,
            output_size=config.HIDDEN_STATE_SIZE,
            encoding_size=122,
        )
        assert hasattr(net, 'champion_embedding')
        assert hasattr(net, 'item_embedding')
        assert hasattr(net, 'trait_embedding')
        assert hasattr(net, 'origin_embedding')

    def test_rep_network_embedding_sizes(self):
        net = RepNetwork(
            input_size=config.OBSERVATION_SIZE,
            layer_sizes=[config.HIDDEN_STATE_SIZE] * 5,
            output_size=config.HIDDEN_STATE_SIZE,
            encoding_size=122,
        )
        assert net.champion_embedding.num_embeddings == 58
        assert net.champion_embedding.embedding_dim == 32
        assert net.item_embedding.num_embeddings == 37
        assert net.item_embedding.embedding_dim == 24

    def test_rep_network_output_shape(self):
        net = RepNetwork(
            input_size=config.OBSERVATION_SIZE,
            layer_sizes=[config.HIDDEN_STATE_SIZE] * 5,
            output_size=config.HIDDEN_STATE_SIZE,
            encoding_size=122,
        )
        batch_size = 2
        x = torch.randn(batch_size, config.OBSERVATION_SIZE)
        output = net(x)
        assert output.shape == (batch_size, config.HIDDEN_STATE_SIZE)


class TestMuZeroNetworkIntegration:
    """Verify full MuZeroNetwork works with refactored constructor-based networks."""

    def test_muzero_network_creation(self):
        model = MuZeroNetwork()
        assert model is not None
        assert model.representation_network is not None
        assert model.dynamics_network is not None
        assert model.prediction_network is not None

    def test_muzero_initial_inference_shapes(self):
        model = MuZeroNetwork()
        batch_size = 4
        obs = np.random.rand(batch_size, config.OBSERVATION_SIZE).astype(np.float32)
        outputs = model.initial_inference(obs)
        assert outputs["value"].shape == (batch_size, 1)
        assert outputs["policy_logits"].shape == (batch_size, config.ACTION_CONCAT_SIZE)
        assert outputs["hidden_state"].shape == (batch_size, config.HIDDEN_STATE_SIZE)

    def test_muzero_recurrent_inference_shapes(self):
        model = MuZeroNetwork()
        batch_size = 4
        obs = np.random.rand(batch_size, config.OBSERVATION_SIZE).astype(np.float32)
        init_outputs = model.initial_inference(obs)
        hidden = init_outputs["hidden_state"]
        action = np.random.randint(0, 10, (batch_size, 1, 3)).astype(np.float32)
        rec_outputs = model.recurrent_inference(hidden, action)
        assert rec_outputs["value"].shape == (batch_size, 1)
        assert rec_outputs["policy_logits"].shape == (batch_size, config.ACTION_CONCAT_SIZE)
        assert rec_outputs["hidden_state"].shape == (batch_size, config.HIDDEN_STATE_SIZE)

    def test_muzero_reward_is_zero_tensor(self):
        model = MuZeroNetwork()
        batch_size = 4
        obs = np.random.rand(batch_size, config.OBSERVATION_SIZE).astype(np.float32)
        outputs = model.initial_inference(obs)
        reward = outputs["reward"]
        assert isinstance(reward, np.ndarray)
        assert np.all(reward == 0)

    def test_muzero_recurrent_reward_is_zero(self):
        model = MuZeroNetwork()
        batch_size = 4
        obs = np.random.rand(batch_size, config.OBSERVATION_SIZE).astype(np.float32)
        init_outputs = model.initial_inference(obs)
        hidden = init_outputs["hidden_state"]
        action = np.random.randint(0, 10, (batch_size, 1, 3)).astype(np.float32)
        rec_outputs = model.recurrent_inference(hidden, action)
        reward = rec_outputs["reward"]
        if isinstance(reward, torch.Tensor):
            assert torch.all(reward == 0)
        else:
            assert np.all(reward == 0)
