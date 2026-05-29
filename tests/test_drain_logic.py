
import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from training_orchestrator import _ParallelEnvManager, _GameWorker, GameResult

@pytest.mark.asyncio
async def test_parallel_env_manager_drain():
    """Verify that pause, resume and wait_for_drain work as expected."""
    num_workers = 2
    mgr = _ParallelEnvManager(num_workers)
    
    # Mock workers to simulate games taking some time
    for worker in mgr.workers:
        async def mock_run_game(agent_manager, return_placements=False):
            await asyncio.sleep(0.5)
            return GameResult("game", {}, {}, 0.1, {})
        worker.run_game = mock_run_game

    agent_manager = MagicMock()
    
    # Start run_continuously in background
    run_task = asyncio.create_task(mgr.run_continuously(agent_manager))
    
    # Wait for initial games to start
    await asyncio.sleep(0.1)
    assert len(mgr.active_tasks) == num_workers
    assert mgr.should_spawn is True
    
    # Pause the manager
    mgr.pause()
    assert mgr.should_spawn is False
    
    # Wait for drain
    # Initially there are 2 tasks running. They take 0.5s.
    # After one finishes, run_continuously should NOT spawn a new one because should_spawn is False.
    await mgr.wait_for_drain()
    
    assert len(mgr.active_tasks) == 0
    
    # Resume
    mgr.resume()
    assert mgr.should_spawn is True
    
    # Give it a moment to spawn new games
    await asyncio.sleep(0.1)
    assert len(mgr.active_tasks) == num_workers
    
    # Stop and clean up
    mgr.stop()
    await run_task

@pytest.mark.asyncio
async def test_orchestrator_integration_drain():
    """Verify that TrainingOrchestrator uses pause/drain during evaluation."""
    from training_orchestrator import TrainingOrchestrator, TrainingConfig
    
    # Mock everything needed for TrainingOrchestrator
    with patch('training_orchestrator.Trainer'), \
         patch('training_orchestrator.GlobalBuffer'), \
         patch('training_orchestrator.MuZeroAgent'), \
         patch('training_orchestrator.create_custom_agent_setup') as MockSetup, \
         patch('training_orchestrator.SummaryWriter'):
        
        MockSetup.return_value = (MagicMock(), MagicMock())
        
        cfg = TrainingConfig(evaluation_interval=2)
        orch = TrainingOrchestrator(cfg)
        orch.setup()
        
        # Mock env_manager
        orch.env_manager = MagicMock()
        orch.env_manager.pause = MagicMock()
        orch.env_manager.resume = MagicMock()
        orch.env_manager.wait_for_drain = AsyncMock()
        
        # Mock evaluate
        orch.evaluate = AsyncMock()
        
        # Mock trainer and buffer for _train_step
        orch.trainer = MagicMock()
        orch.global_buffer = MagicMock()
        orch.global_buffer.read_gameplay_batch.return_value = [None]*5
        
        # First train step (step becomes 1)
        await orch._train_step()
        assert orch.training_step == 1
        orch.env_manager.pause.assert_not_called()
        
        # Second train step (step becomes 2, evaluation_interval is 2)
        await orch._train_step()
        assert orch.training_step == 2
        
        # Should have called pause, wait_for_drain, evaluate, resume
        orch.env_manager.pause.assert_called_once()
        orch.env_manager.wait_for_drain.assert_called_once()
        orch.evaluate.assert_called_once()
        orch.env_manager.resume.assert_called_once()

from unittest.mock import patch

if __name__ == "__main__":
    asyncio.run(test_parallel_env_manager_drain())
    # Note: test_orchestrator_integration_drain needs more mocks to run standalone
