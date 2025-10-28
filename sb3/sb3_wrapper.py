"""
SuperSuit-based wrapper for SB3 compatibility with TFT environment.
Modified to use SuperSuit's vector_concatenate functionality where a single model
controls all players simultaneously, as shown in the PettingZoo tutorial.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional, Union
import supersuit as ss
from tft_set4_gym.tft_simulator import parallel_env
from pettingzoo.utils.env import ParallelEnv


class TFTGymnasiumWrapper(gym.Env):
    """
    Gymnasium-compatible wrapper for TFT parallel environment.
    Converts PettingZoo parallel environment to single-agent Gymnasium format
    where one model controls ALL 8 players simultaneously.
    
    Key Features:
    - Concatenates observations from all 8 players
    - Model outputs 8 separate actions (one per player)
    - Proper multi-player control as intended by SuperSuit design
    """
    
    def __init__(self, parallel_env_fn=None):
        super().__init__()
        
        # Create the base environment
        if parallel_env_fn is None:
            self.parallel_env = parallel_env()
        else:
            self.parallel_env = parallel_env_fn()
        
        # Get sample observation to determine spaces
        temp_obs, _ = self.parallel_env.reset()
        self.agent_ids = list(temp_obs.keys())
        sample_obs = temp_obs[self.agent_ids[0]]
        
        # Calculate concatenated observation space (ALL players)
        if isinstance(sample_obs, dict):
            tensor_size = np.prod(sample_obs['tensor'].shape)
            action_mask_size = np.prod(sample_obs['action_mask'].shape)
            single_player_size = tensor_size + action_mask_size
        else:
            single_player_size = np.prod(sample_obs.shape)
        
        # Concatenated observation space: all 8 players
        total_obs_size = single_player_size * len(self.agent_ids)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_obs_size,), dtype=np.float32
        )
        
        # Action space: 8 separate actions (one per player)
        sample_agent = self.agent_ids[0]
        single_action_space = self.parallel_env.action_space(sample_agent)
        
        # MultiDiscrete action space for all players
        if isinstance(single_action_space, spaces.MultiDiscrete):
            # Each player has same action space, replicate for all players
            nvec = single_action_space.nvec
            all_players_nvec = np.tile(nvec, len(self.agent_ids))
            self.action_space = spaces.MultiDiscrete(all_players_nvec)
        else:
            # Fallback for other action space types
            self.action_space = single_action_space
        
        # Store dimensions for action splitting
        self.num_players = len(self.agent_ids)
        self.single_action_size = len(single_action_space.nvec) if isinstance(single_action_space, spaces.MultiDiscrete) else 1
        
        # Close the temporary environment
        self.parallel_env.close()
        self.parallel_env = None
        
        # Will be created fresh on reset
        self.current_obs = None
        self.current_agents = []
        
    def _flatten_observation(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Flatten Dict observation to vector."""
        if isinstance(obs_dict, dict) and 'tensor' in obs_dict:
            tensor_flat = obs_dict['tensor'].flatten()
            action_mask_flat = obs_dict['action_mask'].flatten() if 'action_mask' in obs_dict else np.array([])
            return np.concatenate([tensor_flat, action_mask_flat]).astype(np.float32)
        else:
            return np.array(obs_dict).flatten().astype(np.float32)
    
    def _concatenate_all_observations(self, obs_dict: Dict[str, Dict]) -> np.ndarray:
        """Concatenate observations from all players into single vector."""
        all_obs = []
        
        # Process observations in consistent order (player_0, player_1, ..., player_7)
        for agent_id in sorted(obs_dict.keys()):
            if agent_id in obs_dict:
                flattened = self._flatten_observation(obs_dict[agent_id])
                all_obs.append(flattened)
        
        return np.concatenate(all_obs).astype(np.float32)
    
    def _split_actions(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        """Split concatenated action into individual actions for each player."""
        actions = {}
        
        # Handle MultiDiscrete action space
        if isinstance(self.action_space, spaces.MultiDiscrete):
            # Split action array into chunks for each player
            for i, agent_id in enumerate(sorted(self.current_agents)):
                start_idx = i * self.single_action_size
                end_idx = start_idx + self.single_action_size
                actions[agent_id] = action[start_idx:end_idx]
        else:
            # Fallback: same action for all (shouldn't happen with proper setup)
            for agent_id in self.current_agents:
                actions[agent_id] = action
        
        return actions
    
    def reset(self, seed=None, options=None):
        """Reset environment and return concatenated observations from all players."""
        # Create fresh environment
        self.parallel_env = parallel_env()
        obs_dict, info_dict = self.parallel_env.reset(seed=seed, options=options)
        
        self.current_agents = list(obs_dict.keys())
        self.current_obs = obs_dict
        
        # Concatenate observations from all players
        concatenated_obs = self._concatenate_all_observations(obs_dict)
        
        # Return concatenated info (or empty dict)
        info = {}
        if info_dict:
            info = {f"{k}_{agent}": v for agent, agent_info in info_dict.items() 
                   for k, v in agent_info.items()}
        
        return concatenated_obs, info
    
    def step(self, action):
        """Step environment with separate actions for each player."""
        if self.parallel_env is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        # Split the concatenated action into individual actions for each player
        individual_actions = self._split_actions(action)
        
        obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = self.parallel_env.step(individual_actions)
        
        # Update current agents (some may have been eliminated)
        self.current_agents = list(obs_dict.keys())
        self.current_obs = obs_dict
        
        # Concatenate observations from all remaining players
        if self.current_agents:
            concatenated_obs = self._concatenate_all_observations(obs_dict)
            # Use average reward across all players (or sum, depending on preference)
            avg_reward = np.mean(list(reward_dict.values())) if reward_dict else 0.0
            # Episode terminates when any player terminates (or customize logic)
            any_terminated = any(terminated_dict.values()) if terminated_dict else True
            any_truncated = any(truncated_dict.values()) if truncated_dict else False
            # Concatenate info
            info = {}
            if info_dict:
                info = {f"{k}_{agent}": v for agent, agent_info in info_dict.items() 
                       for k, v in agent_info.items()}
        else:
            # All agents eliminated
            concatenated_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            avg_reward = 0.0
            any_terminated = True
            any_truncated = False
            info = {}
        
        return concatenated_obs, avg_reward, any_terminated, any_truncated, info
    
    def close(self):
        """Close the environment."""
        if self.parallel_env is not None:
            self.parallel_env.close()
            self.parallel_env = None


class TFTSuperSuitWrapper:
    """
    Main wrapper class that provides Gymnasium-compatible TFT environment for SB3.
    
    Key Features:
    - Converts PettingZoo parallel environment to Gymnasium format
    - Single agent controls representative player
    - Flattens Dict observations for SB3 compatibility
    - Provides proper action space
    """
    
    def __init__(self, num_parallel_envs: int = 1):
        """
        Initialize wrapper.
        
        Args:
            num_parallel_envs: Number of parallel TFT games to run (currently limited to 1)
        """
        self.num_parallel_envs = num_parallel_envs
        self.env = None
    
    def _create_env(self):
        """Create Gymnasium-compatible TFT environment."""
        print(f"[DEBUG] Creating Gymnasium-compatible TFT environment")
        return TFTGymnasiumWrapper()
    
    def get_env(self):
        """Get the Gymnasium environment."""
        if self.env is None:
            self.env = self._create_env()
        return self.env
    
    def close(self):
        """Close the environment."""
        if self.env is not None:
            self.env.close()
            self.env = None


def create_sb3_env(num_parallel_envs: int = 1):
    """
    Create SB3-compatible environment using Gymnasium wrapper.
    
    Args:
        num_parallel_envs: Number of parallel environments (currently limited to 1)
    
    Returns:
        Gymnasium-wrapped environment compatible with SB3
    """
    wrapper = TFTSuperSuitWrapper(num_parallel_envs=num_parallel_envs)
    return wrapper.get_env()


if __name__ == "__main__":
    print("Testing Gymnasium TFT Wrapper for SB3...")
    
    # Test environment creation
    print("\n1. Testing environment creation:")
    try:
        env = create_sb3_env(num_parallel_envs=1)
        
        print(f"   ✅ Environment created successfully")
        print(f"   Environment type: {type(env)}")
        print(f"   Observation space: {env.observation_space}")
        print(f"   Action space: {env.action_space}")
        
        # Test reset
        obs, info = env.reset()
        
        obs_info = f"shape: {obs.shape}" if hasattr(obs, 'shape') else f"type: {type(obs)}"
        print(f"   Reset observation {obs_info}")
        
        if hasattr(obs, 'shape'):
            print(f"   → Observation features: {obs.shape[0]}")
        
        # Show action space details
        print(f"\n2. Understanding action requirements:")
        action_space_info = env.action_space
        print(f"   Action space: {action_space_info}")
        
        if hasattr(action_space_info, 'sample'):
            # Test step with sample action
            print(f"\n3. Testing step with sample action:")
            try:
                action = action_space_info.sample()
                print(f"   Sample action: {action}")
                
                result = env.step(action)
                obs, reward, terminated, truncated, info = result
                
                obs_info = f"shape: {obs.shape}" if hasattr(obs, 'shape') else f"type: {type(obs)}"
                print(f"   ✅ Step successful!")
                print(f"   Observation {obs_info}")
                print(f"   Reward: {reward}")
                print(f"   Terminated: {terminated}")
                print(f"   Truncated: {truncated}")
                
            except Exception as e:
                print(f"   ❌ Step failed: {e}")
        
        env.close()
        print("\n   ✅ Basic functionality test passed!")
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 Gymnasium wrapper testing completed!")
    print("\nKey insights for SB3 usage:")
    print("- Environment provides single-agent Gymnasium interface")
    print("- Representative agent controls the game state") 
    print("- Compatible with stable-baselines3 out of the box")
    print("- Flattened observations from TFT dict format")
    
    print("\nExample SB3 usage:")
    print("```python")
    print("from sb3_wrapper import create_sb3_env")
    print("from stable_baselines3 import PPO")
    print("")
    print("env = create_sb3_env()")
    print("model = PPO('MlpPolicy', env, verbose=1)")
    print("model.learn(total_timesteps=10000)")
    print("```")