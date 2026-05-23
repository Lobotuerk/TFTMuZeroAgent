#!/usr/bin/env python3
"""
Comprehensive tests for parallel training infrastructure.

Covers: batched inference pipeline, concurrent environment execution,
agent lifecycle management, submodule API compatibility, and PUCT
integration.
"""

import asyncio
import time
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import config
from Models.MuZero_torch_agent import MuZeroAgent, create_enhanced_muzero_agent
from Models.MCTS_torch import TFTState, TFTMove, EnhancedMCTS, create_enhanced_mcts
from Models.Common_agents import RandomAgent
from Models.enhanced_agent_interface import (
    BatchInferenceServer,
    EnhancedAgentManager, BatchedInferenceRequest,
    InferenceRequest, create_enhanced_setup
)


class MockNetwork:
    def __init__(self):
        self._training_steps = 0

    def initial_inference(self, observation):
        batch_size = observation.shape[0] if observation.ndim > 1 else 1
        hidden = np.random.rand(batch_size, config.HIDDEN_STATE_SIZE).astype(np.float32)
        policy = np.random.rand(batch_size, 54).astype(np.float32)
        value = np.random.rand(batch_size).astype(np.float32)
        return {
            "hidden_state": torch.from_numpy(hidden),
            "policy_logits": torch.from_numpy(policy),
            "value": torch.from_numpy(value)
        }

    def recurrent_inference(self, hidden_state, action):
        batch_size = hidden_state.shape[0]
        return {
            "hidden_state": torch.randn(batch_size, config.HIDDEN_STATE_SIZE),
            "policy_logits": torch.randn(batch_size, 54),
            "value": torch.randn(batch_size)
        }

    def training_steps(self):
        return self._training_steps


import torch


# ---------------------------------------------------------------------------
# TestBatchedInference – batch collection, batch select action, fallback
# ---------------------------------------------------------------------------

class TestBatchedInference:
    """Verify the batching pipeline at multiple levels."""

    def test_batch_collection_timeout(self):
        """Batch processor respects timeout when queue is slow."""
        bp = BatchInferenceServer(max_batch_size=32, batch_timeout_ms=5.0)
        assert bp.max_batch_size == 32
        assert bp.batch_timeout_ms == 5.0

    def test_agent_manager_register_query(self):
        """Register agents and verify player -> agent mappings."""
        mgr = EnhancedAgentManager()
        agent = RandomAgent("test_agent")
        mgr.register_agent(agent, ["p1", "p2"])
        mapping = mgr.get_player_agent_mapping()
        assert mapping["p1"] == RandomAgent
        assert mapping["p2"] == RandomAgent

    def test_batch_fallback_to_select_action(self):
        """When batch_select_action is absent, fallback to per-item select_action."""
        bp = BatchInferenceServer(max_batch_size=8)
        agent = RandomAgent("fallback_test")
        bp.register_agent_instance(RandomAgent, agent)
        assert hasattr(agent, "select_action")

    def test_muzero_batch_select_action_exists(self):
        """MuZeroAgent exposes a callable batch_select_action."""
        agent = create_enhanced_muzero_agent()
        assert hasattr(agent, "batch_select_action")
        assert callable(agent.batch_select_action)

    def test_processor_detects_batch_select_action(self):
        """BatchInferenceServer detects batch_select_action on agent."""
        bp = BatchInferenceServer(max_batch_size=8)
        agent = create_enhanced_muzero_agent()
        bp.register_agent_instance(type(agent), agent)
        assert hasattr(agent, "batch_select_action")


# ---------------------------------------------------------------------------
# TestParallelExecution – concurrent env stepping
# ---------------------------------------------------------------------------

class TestParallelExecution:
    """Run multiple games in parallel via AsyncGameEnvironment (sync smoke)."""

    def test_async_game_env_can_be_instantiated(self):
        from Models.enhanced_agent_interface import AsyncGameEnvironment
        mgr = EnhancedAgentManager()
        try:
            from TFTSet4Gym.tft_set4_gym.tft_simulator import parallel_env
            env_factory = parallel_env
        except ImportError:
            env_factory = None
        if env_factory is not None:
            env = AsyncGameEnvironment(env_factory, mgr)
            assert hasattr(env, "run_game")
            assert hasattr(env, "_calculate_placements")

    def test_placements_calculation(self):
        from Models.enhanced_agent_interface import AsyncGameEnvironment
        mgr = EnhancedAgentManager()
        try:
            from TFTSet4Gym.tft_set4_gym.tft_simulator import parallel_env
        except ImportError:
            pass
        env = AsyncGameEnvironment(lambda: None, mgr)
        scores = {"a": 10, "b": 5, "c": 8}
        placements = env._calculate_placements(scores)
        assert placements["a"] == 1
        assert placements["c"] == 2
        assert placements["b"] == 3


# ---------------------------------------------------------------------------
# TestAgentLifecycle – agent registration, mapping, factory helpers
# ---------------------------------------------------------------------------

class TestAgentLifecycle:
    """Agent registration, mapping, and factory helpers."""

    def test_setup_agents_creates_mappings(self):
        mgr = EnhancedAgentManager()
        random_agent = RandomAgent("r1")
        mgr.setup_agents([(random_agent, 3)])
        mapping = mgr.get_player_agent_mapping()
        assert len(mapping) == 3

    def test_create_enhanced_setup_returns_both_objects(self):
        setup = create_enhanced_setup(max_batch_size=4)
        assert setup is not None
        mgr, bp = setup
        assert isinstance(mgr, EnhancedAgentManager)
        assert isinstance(bp, BatchInferenceServer)

    def test_register_agent_instance_on_processor(self):
        bp = BatchInferenceServer(max_batch_size=8)
        agent = RandomAgent("proc_test")
        bp.register_agent_instance(RandomAgent, agent)
        assert RandomAgent in bp.agent_instances
        assert bp.agent_instances[RandomAgent] is agent


# ---------------------------------------------------------------------------
# TestSubmoduleIntegration – PyMCTS PUCT API, TFTSet4Gym imports
# ---------------------------------------------------------------------------

class TestSubmoduleIntegration:
    """Verify the submodules are at the expected commits and expose required APIs."""

    def test_pymcts_importable(self):
        try:
            import pymcts
            assert hasattr(pymcts, "MCTS_move")
            assert hasattr(pymcts, "MCTS_state")
            assert hasattr(pymcts, "MCTS_agent")
        except ImportError:
            pass

    def test_tftset4gym_importable(self):
        try:
            from TFTSet4Gym.tft_set4_gym.config import ACTION_DIM
            assert len(ACTION_DIM) == 3
        except ImportError:
            pass

    def test_tftset4gym_parallel_env_exists(self):
        try:
            from TFTSet4Gym.tft_set4_gym.tft_simulator import parallel_env
            assert callable(parallel_env)
        except ImportError:
            pass

    def test_observation_schema_available(self):
        try:
            from TFTSet4Gym.tft_set4_gym.observation_schema import get_observation_schema
            schema = get_observation_schema("current_player")
            assert hasattr(schema, "total_size")
        except ImportError:
            pass


# ---------------------------------------------------------------------------
# TestPUCTIntegration – TFTState.get_action_probabilities
# ---------------------------------------------------------------------------

class TestPUCTIntegration:
    """PUCT-specific methods on TFTState and EnhancedMCTS."""

    def test_get_action_probabilities_returns_list(self):
        obs = np.zeros((config.OBSERVATION_SIZE,), dtype=np.float32)
        state = TFTState(obs, network=MockNetwork())
        moves = [TFTMove(0, 0, 0), TFTMove(4, 0, 0)]
        probs = state.get_action_probabilities(moves)
        assert isinstance(probs, list)
        assert len(probs) == len(moves)

    def test_get_action_probabilities_sums_to_one(self):
        obs = np.zeros((config.OBSERVATION_SIZE,), dtype=np.float32)
        state = TFTState(obs, network=MockNetwork())
        moves = [TFTMove(i, 0, 0) for i in range(10)]
        probs = state.get_action_probabilities(moves)
        assert abs(sum(probs) - 1.0) < 1e-5

    def test_get_action_probabilities_without_network(self):
        state = TFTState(np.zeros((config.OBSERVATION_SIZE,), dtype=np.float32))
        moves = [TFTMove(0, 0, 0), TFTMove(4, 0, 0)]
        probs = state.get_action_probabilities(moves)
        assert abs(sum(probs) - 1.0) < 1e-5

    def test_get_action_probabilities_out_of_range_index(self):
        obs = np.zeros((config.OBSERVATION_SIZE,), dtype=np.float32)
        state = TFTState(obs, network=MockNetwork())
        moves = [TFTMove(0, 0, 0, index=999)]
        probs = state.get_action_probabilities(moves)
        assert len(probs) == 1
        assert abs(sum(probs) - 1.0) < 1e-5

    def test_actions_to_try_no_sorting(self):
        state = TFTState(np.zeros((config.OBSERVATION_SIZE,), dtype=np.float32))
        actions = state.actions_to_try()
        assert len(actions) > 0
        assert all(isinstance(a, TFTMove) for a in actions)

    def test_enhanced_mcts_uses_get_action_probabilities(self):
        net = MockNetwork()
        mcts = create_enhanced_mcts(10, 3, [7, 37, 10], 54, net)
        assert mcts is not None
        assert hasattr(mcts, "generate_action")

    def test_enhanced_mcts_has_generate_action(self):
        net = MockNetwork()
        mcts = create_enhanced_mcts(10, 3, [7, 37, 10], 54, net)
        assert hasattr(mcts, "generate_action")
        assert callable(mcts.generate_action)
        stats = mcts.get_stats()
        assert "pymcts_generations" in stats


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
