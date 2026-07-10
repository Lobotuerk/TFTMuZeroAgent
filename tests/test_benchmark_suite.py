import numpy as np
import pytest
import json
import os
from typing import Dict, Any

from benchmarks.core import SystemMetrics, BenchmarkMockEnv, MCTSProfiler, BenchmarkRunner
from benchmarks.report import BenchmarkReport


class TestSystemMetrics:
    def test_system_metrics_retrieval(self):
        mem_info = SystemMetrics.get_process_memory_info()
        assert 'rss_mb' in mem_info
        assert 'vms_mb' in mem_info
        assert isinstance(mem_info['rss_mb'], float)
        assert isinstance(mem_info['vms_mb'], float)
        assert mem_info['rss_mb'] > 0
        assert mem_info['vms_mb'] > 0

    def test_system_memory_percent(self):
        pct = SystemMetrics.get_system_memory_percent()
        assert isinstance(pct, float)
        assert 0.0 <= pct <= 100.0

    def test_gpu_memory_info_no_cuda(self):
        info = SystemMetrics.get_gpu_memory_info()
        assert 'allocated_mb' in info
        assert 'max_allocated_mb' in info
        assert isinstance(info['allocated_mb'], float)
        assert isinstance(info['max_allocated_mb'], float)


class TestBenchmarkMockEnv:
    def test_benchmark_mock_env_pettingzoo_compliance(self):
        import config
        env = BenchmarkMockEnv(num_players=8, max_steps=10)

        observations, info = env.reset()
        assert len(observations) == 8

        for pid, obs in observations.items():
            assert 'tensor' in obs
            assert 'action_mask' in obs
            assert obs['tensor'].shape == (config.OBSERVATION_SIZE,)
            assert obs['action_mask'].shape == (sum(config.ACTION_DIM),)

        actions = {pid: [0, 0, 0] for pid in env.possible_agents}
        observations, rewards, terminated, truncated, info = env.step(actions)

        assert len(rewards) == 8
        assert len(terminated) == 8

        step_count = 0
        observations, _ = env.reset()
        actions = {pid: [0, 0, 0] for pid in env.possible_agents}
        while not all(terminated.get(pid, False) for pid in env.possible_agents):
            observations, rewards, terminated, truncated, info = env.step(actions)
            step_count += 1
            if step_count > 20:
                break

        assert step_count == 10


class TestMCTSProfiler:
    def test_mcts_profiler_activation_and_cleanup(self):
        try:
            from Models.MCTS_torch import EnhancedMCTS
            from Models.batched_inference import BlockingBatchInferenceQueue
        except ImportError:
            pytest.skip("EnhancedMCTS dependencies (pymcts/TFTSet4Gym) not available")

        original_generate = EnhancedMCTS.generate_action
        original_run_batch = BlockingBatchInferenceQueue._run_batch

        profiler = MCTSProfiler()
        profiler.__enter__()

        assert EnhancedMCTS.generate_action is not original_generate
        assert BlockingBatchInferenceQueue._run_batch is not original_run_batch

        profiler.__exit__(None, None, None)

        assert EnhancedMCTS.generate_action is original_generate
        assert BlockingBatchInferenceQueue._run_batch is original_run_batch

    def test_mcts_profiler_metrics_cleanup_on_multiple_entries(self):
        profiler = MCTSProfiler()
        metrics = profiler.get_metrics()
        assert isinstance(metrics, dict)


class TestBenchmarkRunner:
    def test_benchmark_runner_schema_validation(self):
        from benchmarks.core import BenchmarkRunner

        schema_fields = {'metadata', 'system', 'performance', 'agents'}
        metadata_fields = {'git_commit', 'git_branch', 'timestamp', 'args'}
        system_fields = {'rss_mb_start', 'rss_mb_end', 'vms_mb_start', 'vms_mb_end',
                         'system_memory_percent_avg', 'gpu_memory_allocated_mb_peak',
                         'gpu_memory_max_allocated_mb_peak'}
        perf_fields = {'total_duration_s', 'env_step_time_ms_avg', 'env_step_time_ms_median',
                       'get_actions_time_ms_avg'}

        runner = BenchmarkRunner(
            use_mock_env=True,
            num_games=1,
            steps_per_game=5,
            deep_mcts=False,
        )

        meta = runner._build_metadata()
        assert set(meta.keys()) == metadata_fields

        system_start = SystemMetrics.get_process_memory_info()
        assert set(system_start.keys()) == {'rss_mb', 'vms_mb'}
        assert system_start['rss_mb'] > 0


class TestBenchmarkReport:
    def test_benchmark_report_markdown_and_diff(self):
        current = {
            'metadata': {
                'git_commit': 'abc123',
                'git_branch': 'main',
                'timestamp': '2026-01-01T00:00:00Z',
                'args': {'num_games': 1, 'steps_per_game': 10},
            },
            'system': {
                'rss_mb_start': 100.0,
                'rss_mb_end': 110.0,
                'system_memory_percent_avg': 40.0,
            },
            'performance': {
                'total_duration_s': 5.0,
                'env_step_time_ms_avg': 2.0,
            },
            'agents': {
                'RandomAgent': {
                    'total_actions': 80,
                    'time_per_action_ms_avg': 0.5,
                },
            },
        }
        reference = {
            'system': {
                'rss_mb_start': 90.0,
                'rss_mb_end': 100.0,
                'system_memory_percent_avg': 35.0,
            },
            'performance': {
                'total_duration_s': 4.0,
                'env_step_time_ms_avg': 1.5,
            },
            'agents': {
                'RandomAgent': {
                    'total_actions': 80,
                    'time_per_action_ms_avg': 0.4,
                },
            },
        }

        report = BenchmarkReport()
        diff = report.compare(current, reference)

        assert 'system' in diff
        assert 'performance' in diff
        assert 'agent_RandomAgent' in diff or 'agents' in diff or 'agent_Random' in diff

        rss_diff = None
        for section_key, section_data in diff.items():
            if isinstance(section_data, dict) and 'rss_mb_start' in section_data:
                rss_diff = section_data['rss_mb_start']
                break

        if rss_diff:
            assert abs(rss_diff['delta'] - 10.0) < 0.01
            assert abs(rss_diff['delta_pct'] - 11.1111) < 0.1

        markdown = report.generate_markdown(current, reference=reference)
        assert isinstance(markdown, str)
        assert len(markdown) > 0
        assert 'Benchmark Report' in markdown
        assert 'Comparison vs Reference' in markdown

    def test_report_save_and_load(self, tmp_path):
        data = {'test': 'value', 'number': 42}
        report = BenchmarkReport()
        filepath = os.path.join(tmp_path, 'test_results.json')

        report.save(data, filepath)
        assert os.path.exists(filepath)

        loaded = report.load(filepath)
        assert loaded == data
