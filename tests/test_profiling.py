import time
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from utils.profiling import EnvironmentBenchmark, MetricsCollector, BenchmarkResults
from Models.agent_manager import (
    BatchInferenceServer, EnhancedAgentManager, AsyncGameEnvironment,
)
from utils.profiling import MetricsCollector as MC
from Models.Common_agents import BaseAgent


class TestMetricsCollector:
    def test_record_and_get_stats(self):
        mc = MetricsCollector(window_size=100)
        mc.record("step_time", 0.05)
        mc.record("step_time", 0.10)
        mc.record("step_time", 0.15)

        stats = mc.get_stats("step_time")
        assert stats["count"] == 3
        assert abs(stats["mean_ms"] - 100.0) < 0.01
        assert abs(stats["median_ms"] - 100.0) < 0.01
        assert abs(stats["min_ms"] - 50.0) < 0.01
        assert abs(stats["max_ms"] - 150.0) < 0.01

    def test_multiple_metrics(self):
        mc = MetricsCollector(window_size=100)
        mc.record("env_step", 0.02)
        mc.record("env_step", 0.03)
        mc.record("gpu_infer", 0.50)

        all_stats = mc.all_stats()
        assert "env_step" in all_stats
        assert "gpu_infer" in all_stats
        assert all_stats["env_step"]["count"] == 2
        assert all_stats["gpu_infer"]["count"] == 1

    def test_window_size(self):
        mc = MetricsCollector(window_size=3)
        for i in range(10):
            mc.record("rolling", float(i))
        stats = mc.get_stats("rolling")
        assert stats["count"] == 3
        assert abs(stats["min_ms"] - 7000.0) < 0.01
        assert abs(stats["max_ms"] - 9000.0) < 0.01

    def test_empty_stats(self):
        mc = MetricsCollector()
        stats = mc.get_stats("nonexistent")
        assert stats == {}

    def test_clear(self):
        mc = MetricsCollector()
        mc.record("a", 1.0)
        mc.record("b", 2.0)
        mc.clear()
        assert mc.get_stats("a") == {}
        assert mc.get_stats("b") == {}

    def test_log_summary(self):
        mc = MetricsCollector()
        mc.record("test_metric", 0.1)
        mc.record("test_metric", 0.2)
        stats = mc.log_summary()
        assert "test_metric" in stats
        assert stats["test_metric"]["count"] == 2

    def test_thread_safety(self):
        import threading
        mc = MetricsCollector(window_size=1000)
        errors = []

        def record_many():
            try:
                for _ in range(100):
                    mc.record("thread_safe", 0.01)
                    mc.get_stats("thread_safe")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = mc.get_stats("thread_safe")
        assert stats["count"] == 400


class TestBenchmarkResults:
    def test_summary_basic(self):
        step_times = np.array([0.01, 0.02, 0.03, 0.04, 0.05], dtype=np.float64)
        results = BenchmarkResults(step_times=step_times, reset_time=0.5, num_steps=5)
        s = results.summary()
        assert s["num_steps"] == 5
        assert abs(s["step_time_avg_ms"] - 30.0) < 0.01
        assert abs(s["reset_time_ms"] - 500.0) < 0.01

    def test_summary_empty(self):
        results = BenchmarkResults(step_times=np.array([]), reset_time=0.0, num_steps=0)
        s = results.summary()
        assert s["num_steps"] == 0
        assert s["step_time_avg_ms"] == 0.0


class TestEnvironmentBenchmark:
    def test_run_with_mock_env(self):
        class MockEnv:
            def __init__(self):
                self.possible_agents = [f"p{i}" for i in range(8)]
                self._step_count = 0

            def reset(self):
                self._step_count = 0
                obs = {p: {"tensor": np.zeros((2504,)), "action_mask": np.ones(54, dtype=bool)} for p in self.possible_agents}
                return obs, {}

            def step(self, actions):
                self._step_count += 1
                terminated = {p: self._step_count >= 50 for p in self.possible_agents}
                rewards = {p: 1.0 if t else 0.0 for p, t in terminated.items()}
                obs = {p: {"tensor": np.zeros((2504,)), "action_mask": np.ones(54, dtype=bool)} for p in self.possible_agents}
                return obs, rewards, terminated, {}, {}

        def env_factory():
            return MockEnv()

        bench = EnvironmentBenchmark(env_factory)
        results = bench.run(num_steps=100)
        assert results.num_steps == 50
        assert results.reset_time > 0
        assert len(results.step_times) == 50

    def test_run_multiple(self):
        class MockEnv:
            def __init__(self):
                self.possible_agents = [f"p{i}" for i in range(8)]
                self._step_count = 0

            def reset(self):
                self._step_count = 0
                obs = {p: {"tensor": np.zeros((2504,)), "action_mask": np.ones(54, dtype=bool)} for p in self.possible_agents}
                return obs, {}

            def step(self, actions):
                self._step_count += 1
                terminated = {p: self._step_count >= 20 for p in self.possible_agents}
                rewards = {p: 1.0 if t else 0.0 for p, t in terminated.items()}
                obs = {p: {"tensor": np.zeros((2504,)), "action_mask": np.ones(54, dtype=bool)} for p in self.possible_agents}
                return obs, rewards, terminated, {}, {}

        def env_factory():
            return MockEnv()

        bench = EnvironmentBenchmark(env_factory)
        results_list = bench.run_multiple(num_episodes=3, steps_per_episode=100)
        assert len(results_list) == 3
        for r in results_list:
            assert r.num_steps == 20


class TestBatchInferenceServerMetrics:
    def test_inference_timing_collected(self):
        import torch

        class TestModel:
            def initial_inference(self, batch_tensor):
                batch_size = batch_tensor.shape[0]
                device = batch_tensor.device
                return {
                    'hidden_state': torch.zeros(batch_size, 64, device=device),
                    'policy_logits': torch.zeros(batch_size, 54, device=device),
                    'value': torch.zeros(batch_size, device=device),
                }

        class TestAgent(BaseAgent):
            def __init__(self):
                super().__init__("test_agent")
                self.model = TestModel()

            def select_action(self, observation, mask, reward=None, terminated=None):
                return [0, 0, 0]

            def batch_select_action(self, observations, masks, rewards=None, terminated=None,
                                    precomputed_results=None, player_ids=None, **kwargs):
                return [[0, 0, 0] for _ in observations]

        mc = MetricsCollector(window_size=100)
        server = BatchInferenceServer(max_batch_size=16, batch_timeout_ms=100.0,
                                      metrics_collector=mc)
        agent = TestAgent()
        server.register_agent_instance(type(agent), agent)

        from Models.agent_manager import InferenceRequest
        import numpy as np
        import config

        batch_size = 4
        requests = [
            InferenceRequest(
                player_id=f"p{i}",
                observation=np.random.randn(config.OBSERVATION_SIZE).astype(np.float32),
                mask=np.ones(54, dtype=bool),
                reward=0.0,
                terminated=False,
            )
            for i in range(batch_size)
        ]

        server._infer_sync(agent, requests)

        stats = mc.get_stats("gpu_forward_pass")
        assert stats["count"] == 1
        assert stats["mean_ms"] >= 0

        stats = mc.get_stats("gpu_sync")
        assert stats["count"] == 1


class TestEnhancedAgentManagerMetrics:
    @pytest.mark.asyncio
    async def test_get_actions_records_timing(self):
        mc = MetricsCollector(window_size=100)
        server = BatchInferenceServer(max_batch_size=16, batch_timeout_ms=10.0,
                                      metrics_collector=mc)
        manager = EnhancedAgentManager(server, metrics_collector=mc)

        class FixedActionAgent(BaseAgent):
            def __init__(self):
                super().__init__("fixed")

            def select_action(self, observation, mask, reward=None, terminated=None):
                return [0, 0, 0]

            def batch_select_action(self, observations, masks, rewards=None, terminated=None,
                                    precomputed_results=None, player_ids=None, **kwargs):
                return [[0, 0, 0] for _ in observations]

        agent = FixedActionAgent()
        manager.setup_agents([(agent, 8)])

        observations = {
            f"player_{i}": {
                "tensor": np.zeros((2504,), dtype=np.float32),
                "action_mask": np.ones(54, dtype=bool)
            }
            for i in range(8)
        }
        rewards = {f"player_{i}": 0.0 for i in range(8)}
        terminated = {f"player_{i}": False for i in range(8)}

        await manager.get_actions(observations, rewards, terminated)

        stats = mc.get_stats("get_actions_total")
        assert stats["count"] >= 1
        assert stats["mean_ms"] >= 0


class TestAsyncGameEnvironmentMetrics:
    @pytest.mark.asyncio
    async def test_env_step_timing(self):
        mc = MetricsCollector(window_size=100)
        server = BatchInferenceServer(max_batch_size=16, batch_timeout_ms=10.0,
                                      metrics_collector=mc)
        manager = EnhancedAgentManager(server, metrics_collector=mc)

        class FixedActionAgent(BaseAgent):
            def __init__(self):
                super().__init__("fixed")

            def select_action(self, observation, mask, reward=None, terminated=None):
                return [0, 0, 0]

            def batch_select_action(self, observations, masks, rewards=None, terminated=None,
                                    precomputed_results=None, player_ids=None, **kwargs):
                return [[0, 0, 0] for _ in observations]

        agent = FixedActionAgent()
        manager.setup_agents([(agent, 8)])

        class MockEnv:
            def __init__(self):
                self.possible_agents = [f"player_{i}" for i in range(8)]
                self._step = 0

            def reset(self):
                self._step = 0
                obs = {p: {"tensor": np.zeros((2504,)), "action_mask": np.ones(54, dtype=bool)} for p in self.possible_agents}
                return obs, {}

            def step(self, actions):
                self._step += 1
                terminated = {p: self._step >= 3 for p in self.possible_agents}
                rewards = {p: 1.0 if t else 0.0 for p, t in terminated.items()}
                obs = {p: {"tensor": np.zeros((2504,)), "action_mask": np.ones(54, dtype=bool)} for p in self.possible_agents}
                return obs, rewards, terminated, {}, {}

        def env_factory():
            return MockEnv()

        env = AsyncGameEnvironment(env_factory, manager, metrics_collector=mc)
        result = await env.run_game("test_metrics")

        step_stats = mc.get_stats("env_step")
        assert step_stats["count"] == 3
        assert step_stats["mean_ms"] >= 0
        assert "game_id" in result
        assert "scores" in result
        assert "duration" in result


class TestTrainingOrchestratorProfiling:
    @pytest.mark.asyncio
    async def test_benchmark_and_metrics_integration(self):
        from training_orchestrator import TrainingOrchestrator, TrainingConfig

        cfg = TrainingConfig(
            concurrent_games=1,
            evaluation_games=1,
            evaluation_concurrent=1,
            max_batch_size=8,
            batch_timeout_ms=100.0,
            sync_steps=9999,
        )

        orch = TrainingOrchestrator(cfg)

        assert orch.metrics_collector is not None
        assert hasattr(orch, 'benchmark')
        assert hasattr(orch, '_run_benchmark')
        assert hasattr(orch, '_log_metrics')

        stats = orch.metrics_collector.all_stats()
        assert isinstance(stats, dict)

    def test_run_benchmark_standalone(self):
        from training_orchestrator import TrainingOrchestrator, TrainingConfig

        cfg = TrainingConfig(concurrent_games=1)
        orch = TrainingOrchestrator(cfg)

        assert orch.metrics_collector is not None
        assert hasattr(orch, 'benchmark')
        assert hasattr(orch, '_run_benchmark')
        assert hasattr(orch, '_log_metrics')


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
