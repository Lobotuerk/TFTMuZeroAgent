"""Test that benchmark_training.py imports and runs without errors."""

import asyncio
import pytest
from unittest.mock import MagicMock, patch, ANY
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from training_orchestrator import TrainingOrchestrator, TrainingConfig, ProfilingTracker
import benchmark_training


class TestBenchmarkScriptImports:
    def test_imports(self):
        assert hasattr(benchmark_training, 'build_config')
        assert hasattr(benchmark_training, 'run_benchmark')
        assert hasattr(benchmark_training, 'main')

    def test_build_config_defaults(self):
        class Args:
            steps = 100
            concurrent = 18
            eval_games = 10
            eval_concurrent = 10
            sync_steps = 100
        cfg = benchmark_training.build_config(Args())
        assert cfg.concurrent_games == 18
        assert cfg.evaluation_games == 10
        assert cfg.evaluation_concurrent == 10

    def test_build_config_custom(self):
        class Args:
            steps = 50
            concurrent = 4
            eval_games = 5
            eval_concurrent = 2
            sync_steps = 25
        cfg = benchmark_training.build_config(Args())
        assert cfg.concurrent_games == 4
        assert cfg.evaluation_games == 5
        assert cfg.evaluation_concurrent == 2


@pytest.mark.asyncio
async def test_benchmark_runs_without_error():
    """Verify the benchmark orchestrator runs and profiling data is collected."""
    with patch('training_orchestrator.Trainer') as MockTrainer, \
         patch('training_orchestrator.GlobalBuffer') as MockBuffer, \
         patch('training_orchestrator.MuZeroAgent') as MockAgent, \
         patch('training_orchestrator.create_custom_agent_setup') as MockSetup, \
         patch('training_orchestrator._ThreadEnvManager') as MockEnvMgr, \
         patch('training_orchestrator._MultiProcessEnvManager') as MockMPEnvMgr, \
         patch('training_orchestrator.SummaryWriter') as MockWriter, \
         patch('training_orchestrator.torch.save') as mock_torch_save:

        mock_buffer = MockBuffer.return_value
        mock_buffer.available_gameplay_batch.side_effect = [True] * 150 + [False] * 100
        mock_buffer.read_gameplay_batch.return_value = (
            MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        mock_buffer.available_combat_batch.return_value = False
        mock_trainer = MockTrainer.return_value
        mock_env_mgr = MockEnvMgr.return_value
        MockMPEnvMgr.return_value = mock_env_mgr

        async def mock_run_continuously(agent_mgr, on_game_done):
            for _ in range(5):
                await on_game_done(MagicMock())
                await asyncio.sleep(0.01)

        mock_env_mgr.run_continuously = mock_run_continuously
        MockSetup.return_value = (MagicMock(), MagicMock())
        MockAgent.return_value.get_weights.return_value = {}

        class Args:
            steps = 10
            concurrent = 4
            eval_games = 2
            eval_concurrent = 2
            sync_steps = 5

        success = await benchmark_training.run_benchmark(Args())
        assert success is True


class TestProfilingTracker:
    def test_record_and_summary(self):
        pt = ProfilingTracker()
        pt.record_inference(0.5)
        pt.record_inference(0.3)
        pt.record_env_step(0.2)
        pt.record_env_step(0.4)
        pt.record_train_step(0.1)
        pt.record_idle(0.8)

        s = pt.summary()
        assert s['inference_count'] == 2
        assert s['env_step_count'] == 2
        assert s['train_step_count'] == 1
        assert abs(s['inference_wait_time'] - 0.8) < 1e-6
        assert abs(s['env_step_time'] - 0.6) < 1e-6
        assert abs(s['train_time'] - 0.1) < 1e-6
        assert abs(s['idle_time'] - 0.8) < 1e-6
        assert s['total_time'] > 0

    def test_empty_tracker(self):
        pt = ProfilingTracker()
        s = pt.summary()
        assert s['inference_count'] == 0
        assert s['total_time'] == 0.0

    def test_percentages(self):
        pt = ProfilingTracker()
        pt.record_inference(1.0)
        pt.record_env_step(3.0)
        pt.record_train_step(0.5)
        pt.record_idle(0.5)
        s = pt.summary()
        assert abs(s['inference_pct'] - 20.0) < 0.01
        assert abs(s['env_step_pct'] - 60.0) < 0.01
        assert abs(s['train_pct'] - 10.0) < 0.01
        assert abs(s['idle_pct'] - 10.0) < 0.01

    def test_thread_safety(self):
        import threading
        pt = ProfilingTracker()
        n = 100
        threads = []
        results = []

        def worker():
            for _ in range(n):
                pt.record_inference(0.01)
                pt.record_env_step(0.02)

        for _ in range(4):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        s = pt.summary()
        assert s['inference_count'] == 4 * n
        assert s['env_step_count'] == 4 * n
