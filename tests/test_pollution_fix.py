import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import torch
import config
from Models.MuZero_torch_agent import MuZeroAgent
from Models.global_buffer import GlobalBuffer

def test_muzero_agent_no_buffer_pollution():
    """Verify that MuZeroAgent with global_buffer=None does not store experience."""
    # Create an agent without a buffer
    agent = MuZeroAgent(
        action_size=3,
        action_limits=[8, 37, 10],
        obs_size=config.OBSERVATION_SIZE,
        simulations=1,
        global_buffer=None
    )
    
    assert agent.global_buffer is None
    assert agent.replay_buffers == {}
    assert agent.save_data is False
    
    # Simulate an action selection
    obs = np.zeros(config.OBSERVATION_SIZE)
    mask = np.ones(55, dtype=bool)
    
    # This should not crash and should not store anything
    action = agent.select_action(obs, mask)
    
    assert action is not None
    # Key "default" should not be in replay_buffers because save_data is False
    assert "default" not in agent.replay_buffers

def test_muzero_agent_with_buffer_pollution():
    """Verify that MuZeroAgent with global_buffer DOES store experience."""
    buffer = GlobalBuffer(batch_size=1)
    agent = MuZeroAgent(
        action_size=3,
        action_limits=[8, 37, 10],
        obs_size=config.OBSERVATION_SIZE,
        simulations=1,
        global_buffer=buffer
    )
    
    assert agent.global_buffer is buffer
    assert agent.save_data is True
    
    # Simulate an action selection
    obs = np.zeros(config.OBSERVATION_SIZE)
    mask = np.ones(55, dtype=bool)
    
    # Simulate enough steps to fill the unroll buffer
    for _ in range(config.UNROLL_STEPS + 5):
        action = agent.select_action(obs, mask, reward=1.0, terminated=False, player_id="test_player")
    
    assert "test_player" in agent.replay_buffers
    assert agent.replay_buffers["test_player"].get_len() > config.UNROLL_STEPS
    
    # Move to global buffer
    agent.terminate(final_value=1.0, player_id="test_player")
    
    # Buffer should now have something
    assert len(buffer.gameplay_buffer) > 0 or len(buffer.combat_buffer) > 0
