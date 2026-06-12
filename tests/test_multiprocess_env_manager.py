"""Tests for _MultiProcessEnvManager and the GIL-bypass refactor."""

import asyncio
import sys
import os
import time
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training_orchestrator import (
    _MultiProcessEnvManager,
    GameResult,
    _env_worker_main,
)
from Models.agent_manager import (
    EnhancedAgentManager,
    BatchInferenceServer,
)
from Models.Common_agents import BaseAgent


# ── mock agent ────────────────────────────────────────────────────

class FixedActionAgent(BaseAgent):
    """Agent that always returns the same action for testing."""

    def __init__(self, action=None, **kwargs):
        super().__init__(**kwargs)
        self.action = action or [0, 0, 0]
        self.inference_count = 0
        self.last_observations = []

    def _select_action_impl(self, obs, mask, reward=None, terminated=None, precomputed_results=None):
        self.inference_count += 1
        self.last_observations.append(obs)
        return self.action

    def batch_select_action(self, observations, masks, rewards=None, terminated=None, precomputed_results=None, player_ids=None, **kwargs):
        self.inference_count += 1
        self.last_observations.extend(observations)
        return [self.action for _ in observations]

    def terminate(self, final_value, player_id=None):
        pass


# ── mock env worker ───────────────────────────────────────────────

def _mock_env_worker_3games(env_id, conn):
    """
    Mock env worker that plays 3 short games then exits.
    Uses protocol compatible with _MultiProcessEnvManager.
    """
    for game_idx in range(3):
        num_players = 8
        player_ids = [f"player_{i}" for i in range(num_players)]
        terminated = {p: False for p in player_ids}
        rewards = {p: 0.0 for p in player_ids}

        for step in range(5):
            obs = {
                p: {
                    "tensor": np.zeros((2504,), dtype=np.float32),
                    "action_mask": np.ones(sum(config.ACTION_DIM), dtype=bool),
                }
                for p in player_ids
            }
            float_rewards = {k: float(v) for k, v in rewards.items()}
            conn.send(("infer", obs, float_rewards, terminated))

            msg = conn.recv()
            if msg[0] == "stop":
                return
            actions = msg[1]

            for p in player_ids:
                idx = int(p.split("_")[1])
                if (step >= 3 and (step + idx) % 2 == 0) or step >= 4:
                    terminated[p] = True
                    rewards[p] = float(100 - idx * 10)

        scores = {p: float(100 - int(p.split("_")[1]) * 10) for p in player_ids}
        conn.send(("done", scores))

        msg = conn.recv()
        if msg[0] == "stop":
            return
        elif msg[0] == "pause":
            while True:
                msg2 = conn.recv()
                if msg2[0] == "restart":
                    break
                elif msg2[0] == "stop":
                    return


@pytest.mark.asyncio
async def test_run_fixed_games_returns_results():
    """Verify run_fixed_games returns correct number of GameResult objects."""
    batch_processor = BatchInferenceServer(max_batch_size=16, batch_timeout_ms=1.0)
    agent_manager = EnhancedAgentManager(batch_processor)

    agent = FixedActionAgent()
    agent_manager.setup_agents([(agent, 8)])
    batch_processor.register_agent_instance(FixedActionAgent, agent)

    mgr = _MultiProcessEnvManager(num_workers=2, worker_fn=_mock_env_worker_3games)
    results = await mgr.run_fixed_games(agent_manager, num_games=4)

    assert len(results) == 4
    for r in results:
        assert isinstance(r, GameResult)
        assert len(r.scores) == 8
        assert len(r.placements) == 8


@pytest.mark.asyncio
async def test_single_worker_produces_results():
    """Single worker should produce GameResult objects."""
    batch_processor = BatchInferenceServer(max_batch_size=8, batch_timeout_ms=1.0)
    agent_manager = EnhancedAgentManager(batch_processor)
    agent = FixedActionAgent()
    agent_manager.setup_agents([(agent, 8)])
    batch_processor.register_agent_instance(FixedActionAgent, agent)

    mgr = _MultiProcessEnvManager(num_workers=1, worker_fn=_mock_env_worker_3games)
    results = await mgr.run_fixed_games(agent_manager, num_games=2)

    assert len(results) == 2
    for r in results:
        assert isinstance(r, GameResult)
        assert r.placements["player_0"] == 1  # highest score = best placement


@pytest.mark.asyncio
async def test_inference_routed_through_agent_manager():
    """Verify that env observations reach the agent and actions come back."""
    batch_processor = BatchInferenceServer(max_batch_size=16, batch_timeout_ms=1.0)
    agent_manager = EnhancedAgentManager(batch_processor)

    agent = FixedActionAgent(action=[1, 2, 3])
    agent_manager.setup_agents([(agent, 8)])
    batch_processor.register_agent_instance(FixedActionAgent, agent)

    mgr = _MultiProcessEnvManager(num_workers=1, worker_fn=_mock_env_worker_3games)
    results = await mgr.run_fixed_games(agent_manager, num_games=1)

    assert len(results) == 1
    # The mock env plays 5 steps × 8 players = 40 infer calls (approximately)
    assert agent.inference_count > 0


@pytest.mark.asyncio
async def test_cleanup_after_stop():
    """Process should be terminated cleanly after stop."""
    batch_processor = BatchInferenceServer(max_batch_size=8, batch_timeout_ms=1.0)
    agent_manager = EnhancedAgentManager(batch_processor)
    agent = FixedActionAgent()
    agent_manager.setup_agents([(agent, 8)])
    batch_processor.register_agent_instance(FixedActionAgent, agent)

    mgr = _MultiProcessEnvManager(num_workers=2, worker_fn=_mock_env_worker_3games)
    mgr._start_workers()

    assert len(mgr._processes) == 2
    for pid, (proc, conn) in mgr._processes.items():
        assert proc.is_alive()

    mgr._cleanup()

    assert len(mgr._processes) == 0
    for pid, (proc, conn) in list(mgr._processes.items()):
        assert not proc.is_alive()


# ── test placements are ranked correctly ──────────────────────────

@pytest.mark.asyncio
async def test_placement_ordering():
    """Verify that higher scores get better (lower) placements."""
    batch_processor = BatchInferenceServer(max_batch_size=8, batch_timeout_ms=1.0)
    agent_manager = EnhancedAgentManager(batch_processor)
    agent = FixedActionAgent()
    agent_manager.setup_agents([(agent, 8)])
    batch_processor.register_agent_instance(FixedActionAgent, agent)

    mgr = _MultiProcessEnvManager(num_workers=1, worker_fn=_mock_env_worker_3games)
    results = await mgr.run_fixed_games(agent_manager, num_games=1)

    assert len(results) == 1
    r = results[0]
    # Scores are 100, 90, 80, ... for players 0-7
    # So player_0 should have placement 1, player_7 should have placement 8
    assert r.placements["player_0"] == 1
    assert r.placements["player_7"] == 8


if __name__ == "__main__":
    pytest.main([__file__])
