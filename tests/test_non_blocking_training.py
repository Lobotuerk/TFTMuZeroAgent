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
    """Verify that training occurs in a sequential Collect -> Train loop."""
    
    with patch('training_orchestrator.Trainer') as MockTrainer, \
         patch('training_orchestrator.GlobalBuffer') as MockBuffer, \
         patch('training_orchestrator.MuZeroAgent') as MockAgent, \
         patch('training_orchestrator.create_custom_agent_setup') as MockSetup, \
         patch('training_orchestrator._MultiProcessEnvManager') as MockMPEnvMgr, \
         patch('training_orchestrator.SummaryWriter') as MockWriter, \
         patch('training_orchestrator.torch.save') as mock_torch_save:

        mock_env_mgr = MockMPEnvMgr.return_value
        
        first_call = True
        async def mock_run_fixed_games(agent_mgr, concurrent_games):
            nonlocal first_call
            if first_call:
                first_call = False
                return [GameResult("game_1", {}, {}, 0.1, {}), GameResult("game_2", {}, {}, 0.1, {})]
            else:
                orch.training_active = False
                return []

        mock_env_mgr.run_fixed_games = mock_run_fixed_games
        
        MockSetup.return_value = (MagicMock(), MagicMock())
        
        # Mock agents
        mock_agent = MockAgent.return_value
        mock_agent.get_weights.return_value = {}

        orch = TrainingOrchestrator(TrainingConfig(concurrent_games=2, sync_steps=1, collect_games_per_batch=2))
        orch.setup()
        
        # Run collect which executes the sequential Collect -> Train loop
        await orch.collect()

        # Verify games were counted
        assert orch.games_completed == 2
        
        # Verify training steps occurred (availability has 4 Trues before False)
        assert orch.training_step == 2
        assert mock_trainer.train_network.call_count == 2
        
        print(f"Verified collect: {orch.games_completed} games, {orch.training_step} training steps.")

@pytest.mark.asyncio
async def test_run_non_blocking():
    """Verify the run method also uses the sequential Collect -> Train loop."""
    with patch('training_orchestrator.Trainer') as MockTrainer, \
         patch('training_orchestrator.GlobalBuffer') as MockBuffer, \
         patch('training_orchestrator.MuZeroAgent') as MockAgent, \
         patch('training_orchestrator.create_custom_agent_setup') as MockSetup, \
         patch('training_orchestrator._MultiProcessEnvManager') as MockMPEnvMgr, \
         patch('training_orchestrator.SummaryWriter') as MockWriter, \
         patch('training_orchestrator.torch.save') as mock_torch_save:
        
        mock_buffer = MockBuffer.return_value
        availability = [True, True, True, True, False]
        def get_availability(*args, **kwargs):
            if availability:
                return availability.pop(0)
            return False
        mock_buffer.available_gameplay_batch.side_effect = get_availability
        mock_buffer.read_gameplay_batch.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())
        mock_buffer.available_combat_batch.return_value = False
        
        mock_trainer = MockTrainer.return_value
        mock_env_mgr = MockMPEnvMgr.return_value
        
        first_call = True
        async def mock_run_fixed_games(agent_mgr, concurrent_games):
            nonlocal first_call
            if first_call:
                first_call = False
                return [GameResult("game_1", {}, {}, 0.1, {}), GameResult("game_2", {}, {}, 0.1, {})]
            else:
                orch.training_active = False
                return []

        mock_env_mgr.run_fixed_games = mock_run_fixed_games
        MockSetup.return_value = (MagicMock(), MagicMock())
        MockAgent.return_value.get_weights.return_value = {}

        orch = TrainingOrchestrator(TrainingConfig(concurrent_games=2, sync_steps=5, collect_games_per_batch=2))
        orch.setup()
        
        # Start run (which should run one iteration of collect and train, then stop because of mock)
        await orch.run(max_steps=10)
        
        assert orch.training_step == 2
        assert orch.games_completed == 2
        print(f"Verified run: {orch.games_completed} games, {orch.training_step} training steps.")
