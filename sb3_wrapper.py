"""
Custom wrapper for SB3 compatibility with Dict observation spaces.
Handles the TFT environment's Dict observations (tensor + action_mask).
"""

"""
Custom wrapper for SB3 compatibility with Dict observation spaces.
Handles the TFT environment's Dict observations (tensor + action_mask).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pettingzoo.utils import parallel_to_aec, aec_to_parallel
from typing import Dict, Any, Tuple, Optional, Union

# Import parallel_env from the TFTSet4Gym submodule
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'TFTSet4Gym'))
from tft_set4_gym.tft_simulator import parallel_env
# Remove the path after import to keep it clean
sys.path.pop(0)


class TFTSingleAgentWrapper(gym.Env):
    """
    Wrapper to convert multi-agent TFT environment to single-agent for SB3.
    Handles Dict observation spaces by flattening them manually.
    """
    
    def __init__(self, num_dummy_agents: int = 7):
        """
        Initialize wrapper.
        
        Args:
            num_dummy_agents: Number of additional agents to fill the game
        """
        self.num_dummy_agents = num_dummy_agents
        self.env = None
        self.player_agent = None
        
        # Initialize environment to get spaces
        temp_env = parallel_env()
        temp_obs, _ = temp_env.reset()
        
        # Get a sample agent to determine spaces
        sample_agent = list(temp_env.agents)[0]
        sample_obs = temp_obs[sample_agent]
        
        # Calculate flattened observation space
        tensor_size = np.prod(sample_obs['tensor'].shape)
        action_mask_size = np.prod(sample_obs['action_mask'].shape)
        self.total_obs_size = tensor_size + action_mask_size
        
        # Define spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf, 
            shape=(self.total_obs_size,),
            dtype=np.float32
        )
        
        # MultiDiscrete action space from original
        self.action_space = temp_env.action_space(sample_agent)
        
        temp_env.close()
        
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment and return initial observation."""
        super().reset(seed=seed, options=options)
        
        if seed is not None:
            np.random.seed(seed)
            
        # Create fresh environment
        self.env = parallel_env()
        obs, info = self.env.reset()
        
        # Choose our agent (first one)
        self.player_agent = list(self.env.agents)[0]
        
        # Flatten observation
        flattened_obs = self._flatten_observation(obs[self.player_agent])
        
        return flattened_obs, info.get(self.player_agent, {})
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take action and return results."""
        if self.env is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        # Create actions for all agents
        actions = {}
        for agent in self.env.agents:
            if agent == self.player_agent:
                actions[agent] = action
            else:
                # Random actions for other agents
                actions[agent] = self.env.action_space(agent).sample()
        
        # Step environment
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        
        # Check if our agent is still in the game
        if self.player_agent not in obs:
            # Our agent was eliminated
            terminated = True
            truncated = False
            reward = rewards.get(self.player_agent, 0)
            obs_flat = np.zeros(self.total_obs_size, dtype=np.float32)
            info = infos.get(self.player_agent, {})
        else:
            # Normal step
            terminated = terminations.get(self.player_agent, False)
            truncated = truncations.get(self.player_agent, False)
            reward = rewards.get(self.player_agent, 0)
            obs_flat = self._flatten_observation(obs[self.player_agent])
            info = infos.get(self.player_agent, {})
        
        return obs_flat, reward, terminated, truncated, info
    
    def close(self):
        """Close environment."""
        if self.env is not None:
            self.env.close()
            self.env = None
    
    def _flatten_observation(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Flatten Dict observation to vector."""
        tensor_flat = obs_dict['tensor'].flatten()
        action_mask_flat = obs_dict['action_mask'].flatten()
        
        flattened = np.concatenate([tensor_flat, action_mask_flat])
        return flattened.astype(np.float32)


class TFTVectorizedWrapper:
    """
    Wrapper for vectorized environments (multiple parallel single-agent envs).
    """
    
    def __init__(self, num_envs: int = 4):
        """Initialize multiple single-agent environments."""
        self.num_envs = num_envs
        self.envs = [TFTSingleAgentWrapper() for _ in range(num_envs)]
        
        # Copy spaces from first env
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
    
    def reset(self, seed: Optional[int] = None):
        """Reset all environments."""
        obs = []
        infos = []
        
        for i, env in enumerate(self.envs):
            env_seed = None if seed is None else seed + i
            ob, info = env.reset(seed=env_seed)
            obs.append(ob)
            infos.append(info)
        
        return np.array(obs), infos
    
    def step(self, actions: np.ndarray):
        """Step all environments."""
        obs = []
        rewards = []
        terminateds = []
        truncateds = []
        infos = []
        
        for env, action in zip(self.envs, actions):
            ob, reward, terminated, truncated, info = env.step(action)
            obs.append(ob)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)
        
        return (
            np.array(obs),
            np.array(rewards),
            np.array(terminateds),
            np.array(truncateds),
            infos
        )
    
    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()


def create_sb3_env(num_envs: int = 1) -> gym.Env:
    """
    Create SB3-compatible environment.
    
    Args:
        num_envs: Number of parallel environments (1 for single, >1 for vectorized)
    
    Returns:
        Single or vectorized environment compatible with SB3
    """
    if num_envs == 1:
        return TFTSingleAgentWrapper()
    else:
        return TFTVectorizedWrapper(num_envs)


if __name__ == "__main__":
    print("Testing SB3 Wrapper...")
    
    # Test single environment
    print("\n1. Testing single environment:")
    env = TFTSingleAgentWrapper()
    
    obs, info = env.reset()
    print(f"   Observation shape: {obs.shape}")
    print(f"   Action space: {env.action_space}")
    
    # Take a few steps
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"   Step {i+1}: reward={reward:.3f}, terminated={terminated}, obs_shape={obs.shape}")
        
        if terminated or truncated:
            obs, info = env.reset()
            print(f"   Reset after termination")
    
    env.close()
    print("   ✅ Single environment test passed!")
    
    # Test vectorized environment
    print("\n2. Testing vectorized environment:")
    vec_env = TFTVectorizedWrapper(num_envs=2)
    
    obs, infos = vec_env.reset()
    print(f"   Observations shape: {obs.shape}")
    
    # Take a few steps
    for i in range(3):
        actions = np.array([vec_env.action_space.sample() for _ in range(2)])
        obs, rewards, terminateds, truncateds, infos = vec_env.step(actions)
        print(f"   Step {i+1}: rewards={rewards}, any_terminated={any(terminateds)}")
    
    vec_env.close()
    print("   ✅ Vectorized environment test passed!")
    
    print("\n🎉 All SB3 wrapper tests passed!")