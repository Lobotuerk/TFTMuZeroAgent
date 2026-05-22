#!/usr/bin/env python3
"""
Comprehensive test suite for parallel training architecture.

Tests the core components of the parallel training system:
- EnhancedBatchProcessor / TorchBasedBatchProcessor
- EnhancedAgentManager
- Inference request batching
- Benchmark: sequential vs. batched inference throughput
"""

import sys
import os
import time
import asyncio
import numpy as np
from typing import Dict, List, Tuple, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import config
from Models.enhanced_agent_interface import (
    EnhancedBatchProcessor,
    TorchBasedBatchProcessor,
    EnhancedAgentManager,
    InferenceRequest,
    BatchedInferenceRequest,
    create_enhanced_setup,
    create_custom_agent_setup,
)
from Models.Common_agents import RandomAgent, CultistAgent, DivineAgent


OBS_SIZE = getattr(config, 'OBSERVATION_SIZE', 2504)


class TestBatchProcessor:
    """Tests for the batch processor's request collection and batching."""

    def test_create_batch_processor(self):
        bp = EnhancedBatchProcessor()
        assert bp is not None
        assert bp.max_batch_size > 0

    def test_create_torch_batch_processor(self):
        bp = TorchBasedBatchProcessor(max_batch_size=8)
        assert bp is not None
        assert bp.max_batch_size == 8

    def test_collect_batch_single_request(self):
        bp = EnhancedBatchProcessor(max_batch_size=8)
        req = InferenceRequest(
            player_id="test_0",
            observation=np.zeros((100,), dtype=np.float32),
            mask=np.ones(54, dtype=bool),
            reward=0.0,
            terminated=False,
            timestamp=time.time(),
        )
        bp.request_queues[RandomAgent].put(req)
        batch = bp._collect_batch(RandomAgent)
        assert len(batch) == 1
        assert batch[0].player_id == "test_0"

    def test_collect_batch_multiple_requests(self):
        bp = EnhancedBatchProcessor(max_batch_size=4)
        for i in range(3):
            req = InferenceRequest(
                player_id=f"test_{i}",
                observation=np.zeros((100,), dtype=np.float32),
                mask=np.ones(54, dtype=bool),
                reward=0.0,
                terminated=False,
                timestamp=time.time(),
            )
            bp.request_queues[RandomAgent].put(req)
        batch = bp._collect_batch(RandomAgent)
        assert len(batch) == 3

    def test_create_batch_tensor(self):
        bp = EnhancedBatchProcessor(max_batch_size=4)
        requests = []
        for i in range(4):
            req = InferenceRequest(
                player_id=f"test_{i}",
                observation=np.ones((10,), dtype=np.float32) * i,
                mask=np.ones(54, dtype=bool),
                reward=float(i),
                terminated=False,
                timestamp=time.time(),
            )
            requests.append(req)
        batched = bp._create_batch(requests, RandomAgent)
        assert batched.observations.shape[0] == 4
        assert batched.observations.shape[1] == 10
        assert len(batched.request_ids) == 4
        assert len(batched.rewards) == 4
        assert len(batched.masks) == 4
        assert len(batched.terminated) == 4


class TestAgentManager:
    """Tests for the EnhancedAgentManager."""

    def test_create_agent_manager(self):
        mgr = EnhancedAgentManager()
        assert mgr is not None

    def test_register_agent(self):
        mgr = EnhancedAgentManager()
        agent = RandomAgent("TestRandom")
        mgr.register_agent(agent, ["player_0", "player_1"])
        assert RandomAgent in mgr.agents
        assert mgr.player_to_agent["player_0"] == RandomAgent
        assert mgr.player_to_agent["player_1"] == RandomAgent

    def test_setup_agents(self):
        mgr = EnhancedAgentManager()
        agent_configs = [
            (RandomAgent("R1"), 4),
            (CultistAgent(), 2),
            (DivineAgent(), 2),
        ]
        mgr.setup_agents(agent_configs)
        assert len(mgr.player_to_agent) == 8
        for i in range(4):
            assert mgr.player_to_agent[f"player_{i}"] == RandomAgent
        assert mgr.player_to_agent["player_4"] == CultistAgent
        assert mgr.player_to_agent["player_5"] == CultistAgent
        assert mgr.player_to_agent["player_6"] == DivineAgent
        assert mgr.player_to_agent["player_7"] == DivineAgent

    def test_get_player_agent_mapping(self):
        mgr = EnhancedAgentManager()
        mgr.setup_agents([(RandomAgent("R1"), 3)])
        mapping = mgr.get_player_agent_mapping()
        assert len(mapping) == 3
        assert all(v == RandomAgent for v in mapping.values())


class TestParallelSystemComponents:
    """Integration tests for combined system components."""

    def test_create_enhanced_setup(self):
        mgr, bp = create_enhanced_setup(
            max_batch_size=8,
            batch_timeout_ms=5.0,
            gpu_memory_fraction=0.1,
        )
        assert mgr is not None
        assert bp is not None
        assert isinstance(mgr, EnhancedAgentManager)
        assert isinstance(bp, TorchBasedBatchProcessor)

    def test_create_custom_setup(self):
        agents = [
            (RandomAgent("R1"), 4),
            (CultistAgent(), 2),
            (DivineAgent(), 2),
        ]
        mgr, bp = create_custom_agent_setup(agents, max_batch_size=8)
        assert len(mgr.player_to_agent) == 8

    def test_register_agent_with_batch_processor(self):
        bp = TorchBasedBatchProcessor(max_batch_size=4)
        agent = RandomAgent("TestRandom")
        bp.register_agent_instance(type(agent), agent)
        assert RandomAgent in bp.agent_instances

    def test_create_inference_request(self):
        req = InferenceRequest(
            player_id="player_0",
            observation=np.zeros((OBS_SIZE,), dtype=np.float32),
            mask=np.ones(54, dtype=bool),
            reward=0.5,
            terminated=False,
            timestamp=time.time(),
        )
        assert req.player_id == "player_0"
        assert req.observation.shape[0] == OBS_SIZE

    def test_distribute_results(self):
        bp = EnhancedBatchProcessor()
        n = 8
        requests = []
        for i in range(n):
            req = InferenceRequest(
                player_id=f"p_{i}",
                observation=np.zeros((OBS_SIZE,), dtype=np.float32),
                mask=np.ones(54, dtype=bool),
                reward=0.0,
                terminated=False,
                timestamp=time.time(),
                future=asyncio.Future(),
            )
            requests.append(req)
        results = [f"action_{i}" for i in range(n)]
        bp._distribute_results(requests, results)
        for req, expected in zip(requests, results):
            assert req.future.result() == expected


class TestBenchmarkInference:
    """
    Benchmark comparing sequential vs. batched inference throughput.

    Measures actions per second for both modes to verify that the batch
    processor delivers efficiency gains over naive sequential processing.
    """

    NUM_BENCHMARK_SAMPLES = 32

    @staticmethod
    def _make_obs(size: int = OBS_SIZE) -> np.ndarray:
        return np.random.randn(size).astype(np.float32)

    @staticmethod
    def _make_mask() -> np.ndarray:
        return np.ones((54,), dtype=bool)

    def _build_inference_batch(self, bp, agent, obs_list, mask, agent_type):
        """Build a proper BatchedInferenceRequest with tensor observations via _create_batch."""
        requests = []
        for i, obs in enumerate(obs_list):
            req = InferenceRequest(
                player_id=f"p_{i}",
                observation=obs,
                mask=mask,
                reward=0.0,
                terminated=False,
                timestamp=time.time(),
            )
            requests.append(req)
        return bp._create_batch(requests, agent_type)

    def test_batched_agent_inference_vs_sequential(self):
        """
        Compare sequential vs. batched inference throughput using
        the TorchBasedBatchProcessor._run_agent_inference_sync path.

        The batch processor invokes agent.select_action for each
        observation. By batching the calls into one method invocation
        we save Python call overhead. This benchmark quantifies the gain.
        """
        class TimingAgent:
            def select_action(self, obs, mask=None):
                return [0, 0, 0]

        bp = TorchBasedBatchProcessor(
            max_batch_size=self.NUM_BENCHMARK_SAMPLES,
            batch_timeout_ms=100.0,
        )
        agent = TimingAgent()
        bp.register_agent_instance(type(agent), agent)

        n = self.NUM_BENCHMARK_SAMPLES
        obs_list = [self._make_obs() for _ in range(n)]
        mask = self._make_mask()
        agent_type = type(agent)

        batch = self._build_inference_batch(bp, agent, obs_list, mask, agent_type)

        t0 = time.perf_counter()
        batched_results = bp._run_agent_inference_sync(agent, batch)
        t1 = time.perf_counter()
        batched_time = t1 - t0
        batched_aps = n / batched_time if batched_time > 0 else 0

        t0 = time.perf_counter()
        seq_results = []
        for i, obs in enumerate(obs_list):
            single = self._build_inference_batch(bp, agent, [obs], mask, agent_type)
            seq_results.extend(bp._run_agent_inference_sync(agent, single))
        t1 = time.perf_counter()
        seq_time = t1 - t0
        seq_aps = n / seq_time if seq_time > 0 else 0

        print(f"\n{'=' * 50}")
        print(f"Inference Benchmark ({n} samples)")
        print(f"{'=' * 50}")
        print(f"  Sequential: {n} actions in {seq_time:.4f}s = {seq_aps:.1f} actions/s")
        print(f"  Batched:    {n} actions in {batched_time:.4f}s = {batched_aps:.1f} actions/s")
        if seq_aps > 0:
            print(f"  Speedup:    {batched_aps / seq_aps:.2f}x")
        print(f"{'=' * 50}")

        assert batched_aps > 0, "Batched throughput must be > 0"
        assert seq_aps > 0, "Sequential throughput must be > 0"
        assert len(batched_results) == n
        assert len(seq_results) == n

    def test_batch_collection_overhead(self):
        """
        Measure the overhead of collecting N requests into a batch.
        """
        bp = EnhancedBatchProcessor(max_batch_size=64)
        n = 32
        for i in range(n):
            req = InferenceRequest(
                player_id=f"p_{i}",
                observation=np.zeros((OBS_SIZE,), dtype=np.float32),
                mask=np.ones(54, dtype=bool),
                reward=0.0,
                terminated=False,
                timestamp=time.time(),
            )
            bp.request_queues[RandomAgent].put(req)

        t0 = time.perf_counter()
        batch = bp._collect_batch(RandomAgent)
        t1 = time.perf_counter()
        collection_time = (t1 - t0) * 1000

        print(f"\nBatch collection: {len(batch)} requests in {collection_time:.2f}ms")
        assert len(batch) == n, f"Expected {n} requests, got {len(batch)}"
        assert collection_time < 500.0, f"Collection too slow: {collection_time:.2f}ms"

    def test_batch_tensor_construction(self):
        """
        Measure the time to construct a batched tensor from N requests.
        """
        bp = EnhancedBatchProcessor(max_batch_size=64)
        n = 32
        requests = []
        for i in range(n):
            req = InferenceRequest(
                player_id=f"p_{i}",
                observation=self._make_obs(),
                mask=self._make_mask(),
                reward=float(i),
                terminated=False,
                timestamp=time.time(),
            )
            requests.append(req)

        t0 = time.perf_counter()
        batched = bp._create_batch(requests, RandomAgent)
        t1 = time.perf_counter()
        construction_time = (t1 - t0) * 1000

        print(f"\nTensor construction: {n} obs in {construction_time:.2f}ms")
        assert batched.observations.shape[0] == n
        assert construction_time < 500.0, f"Construction too slow: {construction_time:.2f}ms"


def run_all_tests():
    """Run all test suites."""
    print("=" * 60)
    print("Parallel Training Comprehensive Tests")
    print("=" * 60)

    suites = [
        ("Batch Processor", TestBatchProcessor()),
        ("Agent Manager", TestAgentManager()),
        ("System Components", TestParallelSystemComponents()),
        ("Inference Benchmark", TestBenchmarkInference()),
    ]

    passed = 0
    failed = 0

    for suite_name, suite in suites:
        print(f"\n--- {suite_name} ---")
        for attr in dir(suite):
            if attr.startswith("test_"):
                try:
                    getattr(suite, attr)()
                    print(f"  {attr}: PASSED")
                    passed += 1
                except Exception as e:
                    print(f"  {attr}: FAILED - {e}")
                    import traceback
                    traceback.print_exc()
                    failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed ({passed + failed} total)")
    print(f"{'=' * 60}")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
