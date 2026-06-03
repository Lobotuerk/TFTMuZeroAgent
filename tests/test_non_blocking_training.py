
import asyncio
import pytest
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from training_orchestrator import TrainingOrchestrator, TrainingConfig, GameResult
import config

@pytest.mark.asyncio
async def test_non_blocking_training_loop():
    """Verify that training occurs in the background and is decoupled from games."""
    
    with patch('training_orchestrator.Trainer') as MockTrainer, \
         patch('training_orchestrator.GlobalBuffer') as MockBuffer, \
         patch('training_orchestrator.MuZeroAgent') as MockAgent, \
         patch('training_orchestrator.create_custom_agent_setup') as MockSetup, \
         patch('training_orchestrator._MultiProcessEnvManager') as MockEnvMgr, \
         patch('training_orchestrator.SummaryWriter') as MockWriter, \
         patch('training_orchestrator.torch.save') as mock_torch_save:
        
        # Setup mocks
        mock_buffer = MockBuffer.return_value
        # Simulate availability. Note: checked in both _training_loop and _train_step
        availability = [True] * 10
        def get_availability(*args, **kwargs):
            if availability:
                return availability.pop(0)
            return False
        mock_buffer.available_gameplay_batch.side_effect = get_availability
        mock_buffer.read_gameplay_batch.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())
        mock_buffer.available_combat_batch.return_value = False
        
        mock_trainer = MockTrainer.return_value
        
        mock_env_mgr = MockEnvMgr.return_value
        # run_continuously will simulate game completions
        async def mock_run_continuously(agent_mgr, on_game_done):
            # Simulate 2 games finishing
            await on_game_done(GameResult("game_1", {}, {}, 0.1, {}))
            await asyncio.sleep(0.1)
            await on_game_done(GameResult("game_2", {}, {}, 0.1, {}))
            await asyncio.sleep(0.2)

        mock_env_mgr.run_continuously = mock_run_continuously
        
        MockSetup.return_value = (MagicMock(), MagicMock())
        
        # Mock agents
        mock_agent = MockAgent.return_value
        mock_agent.get_weights.return_value = {}

        orch = TrainingOrchestrator(TrainingConfig(concurrent_games=1, sync_steps=1))
        orch.setup()
        
        try:
            # Run collect which now starts the background training task
            # We'll use a timeout to ensure it doesn't run forever if something is wrong
            await asyncio.wait_for(orch.collect(), timeout=2.0)
            
        except asyncio.TimeoutError:
            pass
        finally:
            orch.training_active = False

        # Verify games were counted
        assert orch.games_completed == 2
        
        # Verify training steps occurred
        # mock_buffer.available_gameplay_batch was called and returned True twice
        assert orch.training_step >= 2
        assert mock_trainer.train_network.call_count >= 2
        
        print(f"Verified collect: {orch.games_completed} games, {orch.training_step} training steps.")

@pytest.mark.asyncio
async def test_run_non_blocking():
    """Verify the run method also uses the background training loop."""
    with patch('training_orchestrator.Trainer') as MockTrainer, \
         patch('training_orchestrator.GlobalBuffer') as MockBuffer, \
         patch('training_orchestrator.MuZeroAgent') as MockAgent, \
         patch('training_orchestrator.create_custom_agent_setup') as MockSetup, \
         patch('training_orchestrator._MultiProcessEnvManager') as MockEnvMgr, \
         patch('training_orchestrator.SummaryWriter') as MockWriter, \
         patch('training_orchestrator.torch.save') as mock_torch_save:
        
        mock_buffer = MockBuffer.return_value
        mock_buffer.available_gameplay_batch.side_effect = [True] * 20 + [False] * 100
        mock_buffer.read_gameplay_batch.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())
        mock_buffer.available_combat_batch.return_value = False
        
        mock_trainer = MockTrainer.return_value
        mock_env_mgr = MockEnvMgr.return_value
        
        async def mock_run_continuously(agent_mgr, on_game_done):
            # Run until stopped
            while True:
                await on_game_done(GameResult("game", {}, {}, 0, {}))
                await asyncio.sleep(0.01)

        mock_env_mgr.run_continuously = mock_run_continuously
        MockSetup.return_value = (MagicMock(), MagicMock())
        MockAgent.return_value.get_weights.return_value = {}

        orch = TrainingOrchestrator(TrainingConfig(concurrent_games=1, sync_steps=5))
        orch.setup()
        
        # Start run and cancel after some time
        run_task = asyncio.create_task(orch.run(max_steps=10))
        
        await asyncio.sleep(0.5)
        orch.training_active = False # Stop the loop
        
        try:
            await asyncio.wait_for(run_task, timeout=1.0)
        except asyncio.TimeoutError:
            pass
        
        assert orch.training_step > 0
        assert orch.games_completed > 0
        print(f"Verified run: {orch.games_completed} games, {orch.training_step} training steps.")

if __name__ == "__main__":
    asyncio.run(test_non_blocking_training_loop())
