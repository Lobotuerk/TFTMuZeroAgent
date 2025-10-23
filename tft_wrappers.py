"""
Environment wrappers for TFT Set 4 Gymnasium Environment.

This module provides wrappers to adapt the multi-agent TFT environment
for use with single-agent RL libraries like Stable Baselines 3.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional


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
        
        # Import here to avoid circular imports
        from .tft_simulator import parallel_env
        
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
            
        # Import here to avoid circular imports
        from .tft_simulator import parallel_env
            
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
        
        return obs_flat, float(reward), terminated, truncated, info
    
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


class RewardShapingWrapper(gym.Wrapper):
    """
    Wrapper that adds intermediate rewards to reduce sparsity.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.episode_steps = 0
        self.survival_bonus = 0.01
        
    def reset(self, **kwargs):
        self.episode_steps = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.episode_steps += 1
        shaped_reward = float(reward)
        
        # Survival bonus (encourages staying alive longer)
        if not terminated and not truncated:
            shaped_reward += self.survival_bonus
        
        # Scale up final rewards and add survival time bonus
        if terminated and float(reward) > 0:
            shaped_reward = float(reward) * 2
            survival_bonus = min(self.episode_steps * 0.1, 50)
            shaped_reward += survival_bonus
        
        return obs, shaped_reward, terminated, truncated, info


# Make wrappers available
__all__ = [
    "TFTSingleAgentWrapper",
    "RewardShapingWrapper",
]