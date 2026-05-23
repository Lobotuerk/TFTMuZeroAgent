#!/usr/bin/env python3
import sys
sys.path.append('.')

import numpy as np
import asyncio
import pytest
from TFTSet4Gym.tft_set4_gym import parallel_env
from Models.enhanced_agent_interface import create_enhanced_setup, BatchInferenceServer

pytestmark = pytest.mark.asyncio
from Models.Common_agents import RandomAgent

async def test_enhanced_evaluation():
    """Test the enhanced AI interface with actual agent decisions."""
    print("Testing enhanced evaluation...")
    
    # Create environment
    env = parallel_env(rank=0)
    observations = env.reset()[0]
    print(f"Environment created with {env.num_agents} players")
    
    # Create enhanced setup
    agent_manager, batch_processor = create_enhanced_setup(
        max_batch_size=4,
        batch_timeout_ms=10.0,
        gpu_memory_fraction=0.1  # Use very little GPU memory for CPU testing
    )
    
    print(f"Created enhanced setup with batch processor: {type(batch_processor).__name__}")
    
    # Get observations and action masks
    print(f"Observations: {len(observations)} agents")
    
    # Check what's in the observations
    sample_obs = list(observations.values())[0]
    print(f"Sample observation keys: {sample_obs.keys()}")
    
    # Extract action masks from observations
    action_masks = {}
    enhanced_observations = {}
    for agent_id, obs in observations.items():
        # The action mask should be in the observation
        if 'action_mask' in obs:
            action_masks[agent_id] = obs['action_mask']
        else:
            # Create default mask if not found
            action_masks[agent_id] = np.ones(54, dtype=np.bool_)
            print(f"Warning: No action_mask found for {agent_id}, using default")
        
        enhanced_observations[agent_id] = obs
    
    print(f"Action masks: {len(action_masks)} agents")
    
    # Test getting actions through agent manager
    try:
        print(f"Testing agent manager with {len(enhanced_observations)} observations")
        
        actions_dict = await agent_manager.get_actions(
            observations=enhanced_observations,
            rewards={agent_id: 0.0 for agent_id in observations.keys()},
            terminated={agent_id: False for agent_id in observations.keys()}
        )
        
        print(f"Generated actions: {len(actions_dict)} agents")
        for agent_id, action in actions_dict.items():
            print(f"  {agent_id}: {action}")
        
        print("✅ Enhanced evaluation test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during batch processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_enhanced_evaluation())