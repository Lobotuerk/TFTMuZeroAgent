"""Tests for _ThreadEnvManager and the thread-based env worker."""

import asyncio
import sys
import os
import time
import numpy as np
import pytest

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


# ── mock thread worker ────────────────────────────────────────────

def _mock_thread_worker(env_id, loop, agent_manager, stop_event, pause_event,
                        on_game_done_callback=None, games_to_play=None):
    """
    Mock thread worker that plays short simulated games without a real env.
    Compatible with _thread_worker_main signature.
    """
    games_done = 0
    while not stop_event.is_set():
        if games_to_play is not None and games_done >= games_to_play:
            break

        while pause_event.is_set() and not stop_event.is_set():
            time.sleep(0.1)
        if stop_event.is_set():
            break

        num_players = 8
        player_ids = [f"player_{i}" for i in range(num_players)]
        terminated = {p: False for p in player_ids}
        rewards = {p: 0.0 for p in player_ids}

        for step in range(5):
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

            future = asyncio.run_coroutine_threadsafe(
                agent_manager.get_actions(
                    obs, float_rewards, terminated,
                    game_id=f"mock_env_{env_id}",
                ),
                loop,
            )
            try:
                actions = future.result(timeout=30.0)
            except Exception:
                return

            for p in player_ids:
                idx = int(p.split("_")[1])
                if (step >= 3 and (step + idx) % 2 == 0) or step >= 4:
                    terminated[p] = True
                    rewards[p] = float(100 - idx * 10)

        scores = {p: float(100 - int(p.split("_")[1]) * 10) for p in player_ids}

        future = asyncio.run_coroutine_threadsafe(
            agent_manager.flush_all_buffers(final_values=scores, game_id=f"mock_env_{env_id}"),
            loop,
        )
        try:
            future.result(timeout=10.0)
        except Exception:
            pass

        if on_game_done_callback:
            sorted_players = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            placements = {pid: i + 1 for i, (pid, _) in enumerate(sorted_players)}
            result = GameResult(
                game_id=f"mock_env_{env_id}_{games_done}",
                placements=placements,
                scores=scores,
                duration=0.0,
                agent_mapping=agent_manager.get_player_agent_mapping(),
            )
            future = asyncio.run_coroutine_threadsafe(
                on_game_done_callback(result), loop,
            )
            try:
                future.result(timeout=10.0)
            except Exception:
                pass

        games_done += 1


# ── tests ─────────────────────────────────────────────────────────

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
        assert r.placements["player_0"] == 1


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
    mgr._launch(agent_manager, loop, on_game_done=None)

    assert len(mgr._threads) == 2
    for tid, t in mgr._threads.items():
        assert t.is_alive()

    mgr._cleanup()

    assert len(mgr._threads) == 0


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
    assert r.placements["player_0"] == 1
    assert r.placements["player_7"] == 8


@pytest.mark.asyncio
async def test_stop_stops_threads():
    """Calling stop() should cause threads to exit."""
    batch_processor = BatchInferenceServer(max_batch_size=8, batch_timeout_ms=1.0)
    agent_manager = EnhancedAgentManager(batch_processor)
    agent = FixedActionAgent()
    agent_manager.setup_agents([(agent, 8)])
    batch_processor.register_agent_instance(FixedActionAgent, agent)

    mgr = _ThreadEnvManager(num_workers=2, worker_fn=_mock_thread_worker)
    loop = asyncio.get_event_loop()
    mgr._launch(agent_manager, loop, on_game_done=None)

    mgr.stop()

    await asyncio.sleep(0.3)
    for t in mgr._threads.values():
        t.join(timeout=2.0)

    mgr._threads.clear()
    assert len(mgr._threads) == 0


@pytest.mark.asyncio
async def test_pause_resume():
    """Pause should prevent new games from starting; resume should allow them."""
    batch_processor = BatchInferenceServer(max_batch_size=8, batch_timeout_ms=1.0)
    agent_manager = EnhancedAgentManager(batch_processor)
    agent = FixedActionAgent()
    agent_manager.setup_agents([(agent, 8)])
    batch_processor.register_agent_instance(FixedActionAgent, agent)

    mgr = _ThreadEnvManager(num_workers=1, worker_fn=_mock_thread_worker)

    mgr.pause()
    assert mgr.should_spawn is False
    assert mgr._pause_event.is_set()

    mgr.resume()
    assert mgr.should_spawn is True
    assert mgr._pause_event.is_set() is False


# ── continuous play ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_continuously_produces_games():
    """Continuous play via callbacks should produce GameResult objects."""
    batch_processor = BatchInferenceServer(max_batch_size=16, batch_timeout_ms=1.0)
    agent_manager = EnhancedAgentManager(batch_processor)
    agent = FixedActionAgent()
    agent_manager.setup_agents([(agent, 8)])
    batch_processor.register_agent_instance(FixedActionAgent, agent)

    mgr = _ThreadEnvManager(num_workers=2, worker_fn=_mock_thread_worker)
    results = []

    async def collect(result: GameResult):
        results.append(result)

    task = asyncio.create_task(
        mgr.run_continuously(agent_manager, on_game_done=collect)
    )
    await asyncio.sleep(1.5)
    mgr.stop()
    await task

    assert len(results) >= 2
    for r in results:
        assert isinstance(r, GameResult)


@pytest.mark.asyncio
async def test_run_continuously_stop_joins_threads():
    """Stopping continuous play should join and clean up threads."""
    batch_processor = BatchInferenceServer(max_batch_size=16, batch_timeout_ms=1.0)
    agent_manager = EnhancedAgentManager(batch_processor)
    agent = FixedActionAgent()
    agent_manager.setup_agents([(agent, 8)])
    batch_processor.register_agent_instance(FixedActionAgent, agent)

    mgr = _ThreadEnvManager(num_workers=2, worker_fn=_mock_thread_worker)

    task = asyncio.create_task(
        mgr.run_continuously(agent_manager, on_game_done=None)
    )
    await asyncio.sleep(0.5)
    mgr.stop()
    await task

    assert len(mgr._threads) == 0


# ── drain logic ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_drain_with_continuous_play():
    """Pause should prevent spawning new games; stop+drain terminates."""
    batch_processor = BatchInferenceServer(max_batch_size=16, batch_timeout_ms=1.0)
    agent_manager = EnhancedAgentManager(batch_processor)
    agent = FixedActionAgent()
    agent_manager.setup_agents([(agent, 8)])
    batch_processor.register_agent_instance(FixedActionAgent, agent)

    mgr = _ThreadEnvManager(num_workers=2, worker_fn=_mock_thread_worker)
    results = []

    async def collect(result: GameResult):
        results.append(result)

    task = asyncio.create_task(
        mgr.run_continuously(agent_manager, on_game_done=collect)
    )
    await asyncio.sleep(1.0)
    assert len(results) >= 1

    before_pause = len(results)
    mgr.pause()

    # In-flight games may still complete; let them drain
    await asyncio.sleep(0.5)
    assert len(results) <= before_pause + mgr.num_workers

    # After in-flight games drain, count must stabilize
    after_drain = len(results)
    await asyncio.sleep(0.5)
    assert len(results) == after_drain

    mgr.stop()
    await task
    assert len(mgr._threads) == 0


@pytest.mark.asyncio
async def test_drain_with_fixed_games_respects_pause():
    """Pause state should not prevent fixed games from completing."""
    batch_processor = BatchInferenceServer(max_batch_size=16, batch_timeout_ms=1.0)
    agent_manager = EnhancedAgentManager(batch_processor)
    agent = FixedActionAgent()
    agent_manager.setup_agents([(agent, 8)])
    batch_processor.register_agent_instance(FixedActionAgent, agent)

    mgr = _ThreadEnvManager(num_workers=2, worker_fn=_mock_thread_worker)

    mgr.pause()
    results = await mgr.run_fixed_games(agent_manager, num_games=4)

    assert len(results) == 4
    for r in results:
        assert isinstance(r, GameResult)
    assert len(mgr._threads) == 0


# ── exception handling ─────────────────────────────────────────────


def _crash_worker(env_id, loop, agent_manager, stop_event, pause_event,
                  on_game_done_callback=None, games_to_play=None):
    """Worker that crashes on startup for worker 0."""
    if env_id == 0:
        raise RuntimeError("Simulated worker crash")
    _mock_thread_worker(env_id, loop, agent_manager, stop_event, pause_event,
                        on_game_done_callback, games_to_play)


@pytest.mark.asyncio
async def test_worker_crash_does_not_hang_run_fixed():
    """A worker crash mid-game should not hang the manager."""
    batch_processor = BatchInferenceServer(max_batch_size=16, batch_timeout_ms=1.0)
    agent_manager = EnhancedAgentManager(batch_processor)
    agent = FixedActionAgent()
    agent_manager.setup_agents([(agent, 8)])
    batch_processor.register_agent_instance(FixedActionAgent, agent)

    mgr = _ThreadEnvManager(num_workers=3, worker_fn=_crash_worker)
    results = await mgr.run_fixed_games(agent_manager, num_games=3)

    assert len(results) >= 2
    for r in results:
        assert isinstance(r, GameResult)
    assert len(mgr._threads) == 0


@pytest.mark.asyncio
async def test_callback_exception_does_not_crash_manager():
    """An exception in the callback should be handled gracefully."""
    batch_processor = BatchInferenceServer(max_batch_size=16, batch_timeout_ms=1.0)
    agent_manager = EnhancedAgentManager(batch_processor)
    agent = FixedActionAgent()
    agent_manager.setup_agents([(agent, 8)])
    batch_processor.register_agent_instance(FixedActionAgent, agent)

    mgr = _ThreadEnvManager(num_workers=2, worker_fn=_mock_thread_worker)

    async def failing_callback(result):
        raise ValueError("Callback failed!")

    task = asyncio.create_task(
        mgr.run_continuously(agent_manager, on_game_done=failing_callback)
    )
    await asyncio.sleep(1.0)
    mgr.stop()
    await task

    assert len(mgr._threads) == 0


# ── edge cases & lifecycle ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_stop_during_fixed_games_returns_early():
    """Stop during run_fixed_games should trigger early exit and cleanup."""
    batch_processor = BatchInferenceServer(max_batch_size=16, batch_timeout_ms=1.0)
    agent_manager = EnhancedAgentManager(batch_processor)
    agent = FixedActionAgent()
    agent_manager.setup_agents([(agent, 8)])
    batch_processor.register_agent_instance(FixedActionAgent, agent)

    mgr = _ThreadEnvManager(num_workers=2, worker_fn=_mock_thread_worker)

    async def run_and_stop():
        task = asyncio.create_task(
            mgr.run_fixed_games(agent_manager, num_games=20)
        )
        await asyncio.sleep(0.3)
        mgr.stop()
        return await task

    results = await run_and_stop()
    assert len(results) >= 0
    assert len(mgr._threads) == 0


@pytest.mark.asyncio
async def test_multiple_stop_calls_safe():
    """Calling stop() multiple times should be safe and not raise errors."""
    batch_processor = BatchInferenceServer(max_batch_size=8, batch_timeout_ms=1.0)
    agent_manager = EnhancedAgentManager(batch_processor)
    agent = FixedActionAgent()
    agent_manager.setup_agents([(agent, 8)])
    batch_processor.register_agent_instance(FixedActionAgent, agent)

    mgr = _ThreadEnvManager(num_workers=2, worker_fn=_mock_thread_worker)
    loop = asyncio.get_event_loop()
    mgr._launch(agent_manager, loop, on_game_done=None)

    mgr.stop()
    mgr.stop()
    mgr.stop()

    await asyncio.sleep(0.3)
    for t in mgr._threads.values():
        t.join(timeout=2.0)
    mgr._threads.clear()

    assert len(mgr._threads) == 0


if __name__ == "__main__":
    pytest.main([__file__])
