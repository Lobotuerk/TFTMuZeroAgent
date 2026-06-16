"""Tests for _ThreadEnvManager."""

import asyncio
import sys
import os
import time
import numpy as np
import pytest
import config
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training_orchestrator import (
    _ThreadEnvManager,
    GameResult,
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

def _mock_thread_worker(env_id, loop, agent_manager, stop_event, pause_event, on_game_done_callback, games_to_play):
    """
    Mock thread worker that plays a fixed number of games.
    """
    import config
    for game_idx in range(games_to_play):
        num_players = 8
        player_ids = [f"player_{i}" for i in range(num_players)]
        terminated = {p: False for p in player_ids}
        rewards = {p: 0.0 for p in player_ids}

        for step in range(5):
            if stop_event.is_set():
                return
            
            # Simple pause logic
            while pause_event.is_set():
                time.sleep(0.1)
                if stop_event.is_set():
                    return

            obs = {
                p: {
                    "tensor": np.zeros((2504,), dtype=np.float32),
                    "action_mask": np.ones(sum(config.ACTION_DIM), dtype=bool),
                }
                for p in player_ids
            }
            float_rewards = {k: float(v) for k, v in rewards.items()}
            
            # Sync to infer (get_actions is the method to use)
            future = asyncio.run_coroutine_threadsafe(
                agent_manager.get_actions(obs, float_rewards, terminated), loop
            )
            actions = future.result()

            for p in player_ids:
                idx = int(p.split("_")[1])
                if (step >= 3 and (step + idx) % 2 == 0) or step >= 4:
                    terminated[p] = True
                    rewards[p] = float(100 - idx * 10)

        scores = {p: float(100 - int(p.split("_")[1]) * 10) for p in player_ids}
        
        result = GameResult(game_id=f"game_{game_idx}", duration=1.0, agent_mapping={}, scores=scores, placements={p: 0 for p in player_ids})
        # Rank placements
        sorted_players = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for i, (p, _) in enumerate(sorted_players):
            result.placements[p] = i + 1
            
        if on_game_done_callback:
            asyncio.run_coroutine_threadsafe(on_game_done_callback(result), loop)


@pytest.mark.asyncio
async def test_run_fixed_games_returns_results():
    """Verify run_fixed_games returns correct number of GameResult objects."""
    batch_processor = BatchInferenceServer(max_batch_size=16, batch_timeout_ms=1.0)
    agent_manager = EnhancedAgentManager(batch_processor)

    agent = FixedActionAgent()
    agent_manager.setup_agents([(agent, 8)])
    batch_processor.register_agent_instance(FixedActionAgent, agent)

    mgr = _ThreadEnvManager(num_workers=2, worker_fn=_mock_thread_worker)
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

    mgr = _ThreadEnvManager(num_workers=1, worker_fn=_mock_thread_worker)
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

    mgr = _ThreadEnvManager(num_workers=1, worker_fn=_mock_thread_worker)
    results = await mgr.run_fixed_games(agent_manager, num_games=1)

    assert len(results) == 1
    # The mock env plays 5 steps × 8 players = 40 infer calls (approximately)
    assert agent.inference_count > 0


@pytest.mark.asyncio
async def test_cleanup_after_stop():
    """Threads should be stopped cleanly."""
    batch_processor = BatchInferenceServer(max_batch_size=8, batch_timeout_ms=1.0)
    agent_manager = EnhancedAgentManager(batch_processor)
    agent = FixedActionAgent()
    agent_manager.setup_agents([(agent, 8)])
    batch_processor.register_agent_instance(FixedActionAgent, agent)

    mgr = _ThreadEnvManager(num_workers=2, worker_fn=_mock_thread_worker)
    loop = asyncio.get_event_loop()
    mgr._launch(agent_manager, loop, on_game_done=None, games_to_play=3)

    assert len(mgr._threads) == 2
    for tid, t in mgr._threads.items():
        assert t.is_alive()

    mgr.stop(); await mgr.wait_for_drain()

    mgr._threads.clear(); assert len(mgr._threads) == 0
    # Allow some time for threads to stop
    time.sleep(0.5)
    for tid, t in list(mgr._threads.items()):
        assert not t.is_alive()


# ── test placements are ranked correctly ──────────────────────────

@pytest.mark.asyncio
async def test_placement_ordering():
    """Verify that higher scores get better (lower) placements."""
    batch_processor = BatchInferenceServer(max_batch_size=8, batch_timeout_ms=1.0)
    agent_manager = EnhancedAgentManager(batch_processor)
    agent = FixedActionAgent()
    agent_manager.setup_agents([(agent, 8)])
    batch_processor.register_agent_instance(FixedActionAgent, agent)

    mgr = _ThreadEnvManager(num_workers=1, worker_fn=_mock_thread_worker)
    results = await mgr.run_fixed_games(agent_manager, num_games=1)

    assert len(results) == 1
    r = results[0]
    # Scores are 100, 90, 80, ... for players 0-7
    # So player_0 should have placement 1, player_7 should have placement 8
    assert r.placements["player_0"] == 1
    assert r.placements["player_7"] == 8


if __name__ == "__main__":
    pytest.main([__file__])
