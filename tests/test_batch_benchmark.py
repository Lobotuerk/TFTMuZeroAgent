#!/usr/bin/env python3
"""
Benchmark test: verify that batched GPU inference is faster than sequential.

Directly measures the model-level speedup: one batched initial_inference
call vs N individual calls with the same total observations. The
BatchInferenceServer integration checks are in test_parallel_training.py.
"""

import time
import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import config


class BenchmarkModel(torch.nn.Module):
    """Lightweight model that simulates real model compute cost."""

    def __init__(self):
        super().__init__()
        # Add a small linear layer to make inference non-trivial
        self.fc = torch.nn.Linear(config.OBSERVATION_SIZE, config.HIDDEN_STATE_SIZE)

    def initial_inference(self, observation):
        device = observation.device
        self.fc = self.fc.to(device)
        h = self.fc(observation.float())
        batch_size = observation.shape[0]
        return {
            'hidden_state': h,
            'policy_logits': torch.randn(batch_size, 54, device=device),
            'value': torch.randn(batch_size, device=device),
        }


class TestBatchedInferenceSpeedup:
    """Benchmark: batched model inference must be faster than sequential."""

    def test_model_batching_speedup(self):
        """One batched initial_inference must be faster than N sequential calls."""
        model = BenchmarkModel().to('cuda' if torch.cuda.is_available() else 'cpu')
        device = next(model.parameters()).device

        batch_size = 8
        obs = torch.randn(batch_size, config.OBSERVATION_SIZE, device=device)

        # Warm-up
        for _ in range(5):
            model.initial_inference(obs)
            for i in range(batch_size):
                model.initial_inference(obs[i:i+1])

        n_trials = 20

        # Batched: single call
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start = time.perf_counter()
        for _ in range(n_trials):
            model.initial_inference(obs)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        batched = (time.perf_counter() - start) / n_trials

        # Sequential: N individual calls
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start = time.perf_counter()
        for _ in range(n_trials):
            for i in range(batch_size):
                model.initial_inference(obs[i:i+1])
        torch.cuda.synchronize() if device.type == 'cuda' else None
        sequential = (time.perf_counter() - start) / n_trials

        ratio = sequential / batched
        speedup_msg = (
            f"batch_size={batch_size}: batched={batched*1000:.3f}ms "
            f"sequential={sequential*1000:.3f}ms "
            f"speedup={ratio:.1f}x on device={device}"
        )
        print(f"\n{speedup_msg}")

        # On GPU, batched should be clearly faster (speedup >= 1.5x).
        # On CPU, the ratio is closer to 1x (batching mainly helps GPU parallelism).
        # We still enforce that batched is NOT significantly slower (>2x).
        if device.type == 'cuda':
            assert ratio >= 1.5, f"Expected >=1.5x speedup on GPU, got {ratio:.1f}x. {speedup_msg}"
        else:
            # On CPU, batching overhead may make it slightly slower;
            # flag if >2x slower (indicates a regression)
            assert batched / sequential < 2.0, (
                f"Batched is {batched/sequential:.1f}x slower than sequential on CPU. {speedup_msg}"
            )

    def test_batched_runs_single_initial_inference(self):
        """Verify BatchInferenceServer runs model.initial_inference once per batch."""
        from Models.agent_manager import (
            BatchInferenceServer, InferenceRequest,
        )
        from Models.Common_agents import BaseAgent

        call_count = [0]

        class TrackingModel(torch.nn.Module):
            def initial_inference(self, observation):
                call_count[0] += 1
                batch_size = observation.shape[0]
                return {
                    'hidden_state': torch.zeros(batch_size, config.HIDDEN_STATE_SIZE),
                    'policy_logits': torch.zeros(batch_size, 54),
                    'value': torch.zeros(batch_size),
                }

        class TrackingAgent(BaseAgent):
            def __init__(self):
                super().__init__("tracking")
                self.model = TrackingModel()
                self.batch_select_action_called = False

            def select_action(self, observation, mask, reward=None, terminated=None):
                return [0, 0, 0]

            def batch_select_action(self, observations, masks, rewards=None, terminated=None, precomputed_results=None, player_ids=None, **kwargs):
                self.batch_select_action_called = True
                return [[0, 0, 0] for _ in observations]

        agent = TrackingAgent()
        bp = BatchInferenceServer(max_batch_size=16, batch_timeout_ms=100.0)
        bp.register_agent_instance(type(agent), agent)

        batch_size = 4
        requests = [
            InferenceRequest(
                player_id=f"p{i}",
                observation=np.random.randn(config.OBSERVATION_SIZE).astype(np.float32),
                mask=np.ones(55, dtype=bool),
                reward=0.0,
                terminated=False,
            )
            for i in range(batch_size)
        ]

        bp._infer_sync(agent, requests)
        assert call_count[0] == 1, (
            f"Expected 1 initial_inference call for batch of {batch_size}, "
            f"got {call_count[0]}"
        )
        assert agent.batch_select_action_called

    def test_precomputed_results_passed_to_agent(self):
        """Pre-computed results are passed through to batch_select_action."""
        from Models.agent_manager import (
            BatchInferenceServer, InferenceRequest,
        )
        from Models.Common_agents import BaseAgent

        class CaptureAgent(BaseAgent):
            def __init__(self):
                super().__init__("capture")
                self.model = BenchmarkModel()
                self.precomputed_received = 0

            def select_action(self, observation, mask, reward=None, terminated=None):
                return [0, 0, 0]

            def batch_select_action(self, observations, masks, rewards=None, terminated=None, precomputed_results=None, player_ids=None, **kwargs):
                if precomputed_results is not None:
                    self.precomputed_received += len(precomputed_results)
                return [[0, 0, 0] for _ in observations]

        agent = CaptureAgent()
        bp = BatchInferenceServer(max_batch_size=16, batch_timeout_ms=100.0)
        bp.register_agent_instance(type(agent), agent)

        batch_size = 4
        requests = [
            InferenceRequest(
                player_id=f"p{i}",
                observation=np.random.randn(config.OBSERVATION_SIZE).astype(np.float32),
                mask=np.ones(55, dtype=bool),
                reward=0.0,
                terminated=False,
            )
            for i in range(batch_size)
        ]

        bp._infer_sync(agent, requests)
        assert agent.precomputed_received == batch_size, (
            f"Expected {batch_size} precomputed items, got {agent.precomputed_received}"
        )


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
