"""Unit tests for training_orchestrator components: _GameWorker and TrainingConfig."""

import sys
import os
import pytest
from dataclasses import fields

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training_orchestrator import TrainingConfig, _GameWorker, GameResult
import config


class TestTrainingConfig:
    def test_default_creation(self):
        cfg = TrainingConfig()
        assert cfg.starting_train_step == 0
        assert cfg.run_name == ""
        assert cfg.concurrent_games == config.CONCURRENT_GAMES
        assert cfg.evaluation_games == config.EVALUATION_GAMES
        assert cfg.evaluation_concurrent == config.EVALUATION_CONCURRENT_GAMES
        assert cfg.max_batch_size == config.BATCH_SIZE
        assert cfg.batch_timeout_ms == 5.0
        assert cfg.gpu_memory_fraction == 0.8
        assert hasattr(cfg, 'sync_steps')
        assert hasattr(cfg, 'results_path')
        assert cfg.sync_steps == config.SYNC_STEPS
        assert cfg.results_path == config.RESULTS_PATH

    def test_custom_values(self):
        cfg = TrainingConfig(
            starting_train_step=100,
            run_name="test_run",
            concurrent_games=8,
            evaluation_games=20,
            evaluation_concurrent=4,
            max_batch_size=32,
            batch_timeout_ms=10.0,
            gpu_memory_fraction=0.5,
            sync_steps=2,
            results_path="./test_results"
        )
        assert cfg.starting_train_step == 100
        assert cfg.run_name == "test_run"
        assert cfg.concurrent_games == 8
        assert cfg.evaluation_games == 20
        assert cfg.evaluation_concurrent == 4
        assert cfg.max_batch_size == 32
        assert cfg.batch_timeout_ms == 10.0
        assert cfg.gpu_memory_fraction == 0.5
        assert cfg.sync_steps == 2
        assert cfg.results_path == "./test_results"

    def test_partial_custom_values(self):
        cfg = TrainingConfig(starting_train_step=50, concurrent_games=2)
        assert cfg.starting_train_step == 50
        assert cfg.concurrent_games == 2
        assert cfg.evaluation_games == 10

    def test_all_fields_have_defaults(self):
        cfg = TrainingConfig()
        for f in fields(TrainingConfig):
            assert hasattr(cfg, f.name)


class TestGameWorker:
    def test_worker_initialization(self):
        worker = _GameWorker(worker_id=0)
        assert worker.worker_id == 0
        assert worker.games_completed == 0

    def test_worker_initialization_multiple_ids(self):
        for i in range(5):
            worker = _GameWorker(i)
            assert worker.worker_id == i
            assert worker.games_completed == 0

    def test_worker_games_completed_tracking(self):
        worker = _GameWorker(0)
        assert worker.games_completed == 0
        worker.games_completed += 1
        assert worker.games_completed == 1
        worker.games_completed = 10
        assert worker.games_completed == 10


class TestGameResult:
    def test_game_result_creation(self):
        result = GameResult(
            game_id="test_0",
            placements={"player_0": 1},
            scores={"player_0": 100.0},
            duration=30.5,
            agent_mapping={"player_0": type(None)},
        )
        assert result.game_id == "test_0"
        assert result.placements["player_0"] == 1
        assert result.scores["player_0"] == 100.0
        assert result.duration == 30.5

    def test_game_result_empty_placements(self):
        result = GameResult(
            game_id="game_1",
            placements={},
            scores={},
            duration=0.0,
            agent_mapping={},
        )
        assert result.placements == {}
        assert result.scores == {}
