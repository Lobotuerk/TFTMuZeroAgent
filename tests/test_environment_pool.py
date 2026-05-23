import pytest
import asyncio
import numpy as np
from Models.enhanced_agent_interface import EnvironmentPool, EnhancedAgentManager, TorchBasedBatchProcessor, AsyncGameEnvironment
from Models.Common_agents import RandomAgent
import config

class MockEnv:
    def __init__(self):
        self.possible_agents = [f"player_{i}" for i in range(8)]
        self.steps = 0
        self.max_steps = 10
    
    def reset(self):
        self.steps = 0
        obs = {p: {"tensor": np.zeros((2504,)), "action_mask": np.ones(54, dtype=bool)} for p in self.possible_agents}
        return obs, {}
    
    def step(self, actions):
        self.steps += 1
        terminated = {p: self.steps >= self.max_steps for p in self.possible_agents}
        rewards = {p: 1.0 if t else 0.0 for p, t in terminated.items()}
        obs = {p: {"tensor": np.zeros((2504,)), "action_mask": np.ones(54, dtype=bool)} for p in self.possible_agents}
        return obs, rewards, terminated, {}, {}

def env_factory():
    return MockEnv()

@pytest.mark.asyncio
async def test_environment_pool_basic():
    # Setup
    batch_processor = TorchBasedBatchProcessor(max_batch_size=16)
    agent_manager = EnhancedAgentManager(batch_processor)
    
    # Register random agents for all players
    random_agent = RandomAgent("TestRandom")
    agent_manager.setup_agents([(random_agent, 8)])
    batch_processor.register_agent_instance(RandomAgent, random_agent)
    
    pool = EnvironmentPool(
        env_factory=env_factory,
        agent_manager=agent_manager,
        num_environments=2,
        max_concurrent_games=2
    )
    
    await pool.start()
    
    # Run games
    results = await pool.run_games(num_games=4)
    
    assert len(results) == 4
    for result in results:
        assert 'game_id' in result
        assert 'scores' in result
        assert 'duration' in result
    
    stats = pool.get_performance_stats()
    assert stats['total_games_completed'] == 4
    
    await pool.stop()

@pytest.mark.asyncio
async def test_environment_pool_continuous():
    # Setup
    batch_processor = TorchBasedBatchProcessor(max_batch_size=16)
    agent_manager = EnhancedAgentManager(batch_processor)
    random_agent = RandomAgent("TestRandom")
    agent_manager.setup_agents([(random_agent, 8)])
    batch_processor.register_agent_instance(RandomAgent, random_agent)
    
    pool = EnvironmentPool(
        env_factory=env_factory,
        agent_manager=agent_manager,
        num_environments=2,
        max_concurrent_games=2
    )
    
    await pool.start()
    
    # Run games continuously
    results = await pool.run_continuous(num_games=4)
    
    assert len(results) == 4
    assert pool.total_games_completed == 4
    
    await pool.stop()

@pytest.mark.asyncio
async def test_environment_pool_experience_collection():
    # Setup
    batch_processor = TorchBasedBatchProcessor(max_batch_size=16)
    agent_manager = EnhancedAgentManager(batch_processor)
    
    # Mock agent with replay buffer
    class MockReplayBuffer:
        def __init__(self):
            self.flushed = False
        async def move_buffer_to_global_async(self):
            self.flushed = True
            
    class MockAgent:
        def __init__(self):
            self.replay_buffer = MockReplayBuffer()
            
    mock_agent = MockAgent()
    agent_manager.agents[MockAgent] = mock_agent
    
    pool = EnvironmentPool(
        env_factory=env_factory,
        agent_manager=agent_manager
    )
    
    # Add a dummy result
    pool.game_results.append({'game_id': 'test', 'scores': {}})
    
    # Collect experiences
    results = await pool.collect_experiences()
    
    assert len(results) == 1
    assert mock_agent.replay_buffer.flushed == True
