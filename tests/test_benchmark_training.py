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
            games = 2
            steps = None
            concurrent = 18

        cfg = benchmark_training.build_config(Args())
        assert cfg.concurrent_games == 18
        assert cfg.evaluation_games == 0
        assert cfg.evaluation_concurrent == 0

    def test_build_config_custom(self):
        class Args:
            games = 5
            steps = None
            concurrent = 4

        cfg = benchmark_training.build_config(Args())
        assert cfg.concurrent_games == 4
        assert cfg.evaluation_games == 0
        assert cfg.evaluation_concurrent == 0


@pytest.mark.asyncio
async def test_benchmark_runs_without_error():
    """Verify the benchmark orchestrator runs and profiling data is collected."""
    with patch('training_orchestrator.Trainer') as MockTrainer, \
         patch('training_orchestrator.GlobalBuffer') as MockBuffer, \
         patch('training_orchestrator.MuZeroAgent') as MockAgent, \
         patch('training_orchestrator.create_custom_agent_setup') as MockSetup, \
         patch('training_orchestrator._ThreadEnvManager') as MockEnvMgr, \
         patch('training_orchestrator.SummaryWriter') as MockWriter, \
         patch('training_orchestrator.torch.save') as mock_torch_save:

        mock_env_mgr = MockEnvMgr.return_value

        async def mock_run_fixed_games(agent_mgr, num_games):
            return []

        mock_env_mgr.run_fixed_games = mock_run_fixed_games
        MockSetup.return_value = (MagicMock(), MagicMock())
        MockAgent.return_value.get_weights.return_value = {}

        class Args:
            games = 2
            steps = None
            concurrent = 4

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
        pt.record_round(1.5)
        pt.record_round(2.5)
        pt.record_game(45.0)

        s = pt.summary()
        assert s['inference_count'] == 2
        assert s['env_step_count'] == 2
        assert s['train_step_count'] == 1
        assert s['round_count'] == 2
        assert s['game_count'] == 1
        assert abs(s['inference_wait_time'] - 0.8) < 1e-6
        assert abs(s['env_step_time'] - 0.6) < 1e-6
        assert abs(s['train_time'] - 0.1) < 1e-6
        assert abs(s['idle_time'] - 0.8) < 1e-6
        assert abs(s['avg_round_time'] - 2.0) < 1e-6
        assert abs(s['avg_game_time'] - 45.0) < 1e-6
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
                pt.record_round(0.05)
                pt.record_game(1.0)

        for _ in range(4):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        s = pt.summary()
        assert s['inference_count'] == 4 * n
        assert s['env_step_count'] == 4 * n
        assert s['round_count'] == 4 * n
        assert s['game_count'] == 4 * n


def test_prepopulate_buffer_logic():
    """Verify that synthetic experiences can be stored in the GlobalBuffer."""
    import numpy as np
    import config
    from Models.global_buffer import GlobalBuffer
    from Models.action_conversion import action_3d_to_policy

    buf = GlobalBuffer(config.BATCH_SIZE, action_to_policy=action_3d_to_policy)
    synthetic_experiences = []
    for _ in range(config.BATCH_SIZE * 4):
        synthetic_experiences.append([
            np.zeros(config.OBSERVATION_SIZE),
            [np.zeros(3, dtype=np.int32) for _ in range(config.UNROLL_STEPS - 1)],
            [0.0] * config.UNROLL_STEPS,
            [0.0] * config.UNROLL_STEPS,
            [np.zeros(config.ACTION_CONCAT_SIZE) for _ in range(config.UNROLL_STEPS)],
        ])
    buf.store_episode(synthetic_experiences)
    assert buf.get_gameplay_buffer_size() == config.BATCH_SIZE * 4
