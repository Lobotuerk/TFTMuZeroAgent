"""
Simple test to understand SuperSuit interface with TFT environment.
"""

import numpy as np
import supersuit as ss
from tft_set4_gym.tft_simulator import parallel_env
from pettingzoo.utils.env import ParallelEnv
from gymnasium import spaces
from typing import Dict


class TFTDictFlattenWrapper(ParallelEnv):
    """
    Simple wrapper to flatten Dict observations for TFT environment.
    """
    
    def __init__(self, env):
        self.env = env
        
        # Set required ParallelEnv attributes
        self.possible_agents = env.possible_agents
        self.agents = []
        self.metadata = getattr(env, 'metadata', {'name': 'tft-wrapped'})
        self.render_mode = getattr(env, 'render_mode', None)
        
        # Get observation/action spaces by resetting once
        temp_obs, _ = env.reset()
        self.agents = env.agents[:]
        sample_agent = self.agents[0]
        sample_obs = temp_obs[sample_agent]
        
        # Calculate flattened observation space size
        tensor_size = np.prod(sample_obs['tensor'].shape)
        action_mask_size = np.prod(sample_obs['action_mask'].shape)
        total_size = tensor_size + action_mask_size
        
        # Store observation/action spaces
        self._obs_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_size,), dtype=np.float32
        )
        self._action_space = env.action_space(sample_agent)
    
    @property
    def unwrapped(self):
        """Return the unwrapped environment."""
        return self.env
    
    def _flatten_observation(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Flatten Dict observation to vector."""
        tensor_flat = obs_dict['tensor'].flatten()
        action_mask_flat = obs_dict['action_mask'].flatten()
        return np.concatenate([tensor_flat, action_mask_flat]).astype(np.float32)
    
    def reset(self, seed=None, options=None):
        """Reset environment and flatten observations."""
        obs, infos = self.env.reset(seed=seed, options=options)
        self.agents = self.env.agents[:]
        
        flattened_obs = {}
        for agent, agent_obs in obs.items():
            flattened_obs[agent] = self._flatten_observation(agent_obs)
        return flattened_obs, infos
    
    def step(self, actions):
        """Step environment and flatten observations."""
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        self.agents = self.env.agents[:]
        
        flattened_obs = {}
        for agent, agent_obs in obs.items():
            flattened_obs[agent] = self._flatten_observation(agent_obs)
        return flattened_obs, rewards, terminations, truncations, infos
    
    def observation_space(self, agent):
        return self._obs_space
    
    def action_space(self, agent):
        return self._action_space
    
    def close(self):
        return self.env.close()


if __name__ == "__main__":
    print("Testing simple SuperSuit integration...")
    
    # Create wrapped environment
    print("\n1. Creating wrapped environment:")
    env = parallel_env()
    wrapped_env = TFTDictFlattenWrapper(env)
    print(f"   Wrapped environment created")
    print(f"   Possible agents: {wrapped_env.possible_agents}")
    
    # Test basic functionality
    print("\n2. Testing basic functionality:")
    obs, infos = wrapped_env.reset()
    print(f"   Reset successful")
    print(f"   Current agents: {wrapped_env.agents}")
    print(f"   Observation keys: {list(obs.keys())}")
    
    if wrapped_env.agents:
        sample_agent = wrapped_env.agents[0]
        print(f"   Sample agent: {sample_agent}")
        print(f"   Obs shape: {obs[sample_agent].shape}")
        print(f"   Action space: {wrapped_env.action_space(sample_agent)}")
    
    # Test SuperSuit conversion
    print("\n3. Testing SuperSuit conversion:")
    try:
        vec_env = ss.pettingzoo_env_to_vec_env_v1(wrapped_env)
        print(f"   SuperSuit conversion successful!")
        print(f"   Vec env type: {type(vec_env)}")
        print(f"   Observation space: {vec_env.observation_space}")
        print(f"   Action space: {vec_env.action_space}")
        
        # Test reset and step
        print("\n4. Testing vectorized environment:")
        obs = vec_env.reset()
        print(f"   Reset obs type: {type(obs)}, shape: {getattr(obs, 'shape', 'N/A')}")
        
        # Create action that matches the number of agents
        if hasattr(vec_env, 'num_envs'):
            num_actions = vec_env.num_envs
        else:
            # Assume it's the number of agents
            num_actions = len(wrapped_env.agents)
        
        print(f"   Number of actions needed: {num_actions}")
        action = vec_env.action_space.sample()
        print(f"   Sample action: {action}")
        
        try:
            step_result = vec_env.step([action] * num_actions)
            print(f"   Step successful! Result length: {len(step_result)}")
        except Exception as e:
            print(f"   Step failed: {e}")
            # Try with different action format
            actions = [vec_env.action_space.sample() for _ in range(num_actions)]
            print(f"   Trying with action list: {len(actions)} actions")
            step_result = vec_env.step(actions)
            print(f"   Step successful with action list!")
        
        vec_env.close()
        
    except Exception as e:
        print(f"   SuperSuit conversion failed: {e}")
        import traceback
        traceback.print_exc()
    
    wrapped_env.close()
    print("\n✅ Test completed!")