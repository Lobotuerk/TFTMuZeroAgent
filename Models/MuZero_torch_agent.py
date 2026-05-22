import torch
import numpy as np
import asyncio
from typing import List, Optional, Any, Union, Dict, Tuple
import config

# MCTS and Model imports
from Models.MCTS_torch import EnhancedMCTS, create_enhanced_mcts
from Models.MuZero_torch_model import MuZeroNetwork

# Enhanced buffer system
from Models.replay_buffer import ReplayBuffer
from Models.global_buffer import GlobalBuffer

# New observation schema system
from Models.Common_agents import extract_field_from_observation, BaseAgent

# Import TFTSet4Gym config for action dimensions
from TFTSet4Gym.tft_set4_gym.config import ACTION_DIM
from TFTSet4Gym.tft_set4_gym.observation_schema import get_observation_schema


class MuZeroAgent(BaseAgent):
    """
    MuZero Agent using new observation schema and Ray-free buffers

    Key improvements:
    - Uses new observation schema for field extraction
    - Ray-free buffer system with async capabilities
    - Better memory management and error handling
    - Cleaner separation of concerns
    - Support for both sync and async operations
    """
    
    def __init__(self,
                 agent_name: str = "MuZeroAgent",
                 global_buffer: Optional[Any] = None):
        super().__init__(agent_name, global_buffer)

        # Read action dimensions from observation schema/config
        self.action_limits = ACTION_DIM.copy()  # [7, 37, 10] from TFTSet4Gym config
        self.action_size = len(self.action_limits)  # 3 action dimensions
        
        # Read observation size from schema if available
        schema = get_observation_schema("current_player")
        self.obs_size = schema.total_size
        
        # Set simulations from config if not provided
        self.simulations = getattr(config, 'NUM_SIMULATIONS', 10)
        
        # Model and MCTS initialization
        self.model = MuZeroNetwork()
        
        # Initialize Enhanced MCTS with action dimensions from schema/config
        # For TFTSet4Gym: ACTION_DIM = [7, 37, 10], so policy_size should accommodate the action space
        policy_size = self.action_limits[1] * self.action_size if len(self.action_limits) > 1 else sum(self.action_limits)
        
        self.mcts = EnhancedMCTS(
            sample_size=80,
            action_size=self.action_size,  # Read from ACTION_DIM
            action_limits=self.action_limits,  # Read from ACTION_DIM
            policy_size=policy_size,  # Calculate policy size based on action dimensions
            network=self.model
        )
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model.to('cuda')
        
        # Performance monitoring
        self.stats = {
            'total_actions': 0,
            'episodes_completed': 0,
            'buffer_stores': 0,
            'combat_encounters': 0
        }

    def load_weights(self, weights_path: str):
        """Load model weights from a specified path"""
        weights = torch.load(weights_path, map_location=self.model.device)
        self.model.load_state_dict(weights)
        print(f"Weights loaded successfully from {weights_path}")
    
    def _select_action_impl(self, 
                     observation, action_mask, reward=None, terminated=None) -> List[int]:
        """
        Select actions using MCTS with enhanced observation processing
        
        Args:
            observation: Raw observation from environment
            mask: Action mask (optional)
            reward: Reward signal (optional)
            terminated: Termination flags (optional)
        
        Returns:
            Selected actions for each player
        """
        
        # Generate actions using MCTS
        env_move, action_vector = self._generate_action_with_mcts(observation, observation)
        self._store_experience(observation=observation, policy=action_vector, reward=reward, terminated=terminated)

        return env_move
    
    def batch_select_action(self, observations: List[np.ndarray], masks: List[np.ndarray],
                            precomputed_results: Optional[List[Dict[str, Any]]] = None) -> List[Any]:
        """Select actions for a batch of observations using MCTS.
        
        Uses pre-computed neural network results when available to avoid
        redundant initial_inference calls on each item.
        
        Args:
            observations: List of observation arrays, one per environment
            masks: List of action mask arrays, one per environment
            precomputed_results: Optional list of pre-computed NN results per item
            
        Returns:
            List of selected environment actions
        """
        actions = []
        for i, (obs, mask) in enumerate(zip(observations, masks)):
            pc = precomputed_results[i] if precomputed_results else None
            env_move, _ = self._generate_action_with_mcts(obs, mask, precomputed=pc)
            actions.append(env_move)
        return actions

    def _generate_action_with_mcts(self, 
                                   observation: np.ndarray, 
                                   mask: np.ndarray,
                                   precomputed: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate actions using Enhanced MCTS
        
        Args:
            observation: Current observation
            mask: Action mask
            precomputed: Pre-computed NN results (hidden_state, policy, value)
                         to avoid redundant initial_inference
        """        
        # Use Enhanced MCTS for action generation
        actions, action_vector = self.mcts.generate_action(
            self.simulations, 
            observation=observation, 
            mask=mask,
            precomputed=precomputed
        )
        
        return actions, action_vector
    
    def get_weights(self) -> Dict[str, Any]:
        """Get model weights for sharing/saving"""
        return self.model.state_dict()
    
    def update_weights(self, weights):
        """Update model weights"""
        self.model.load_state_dict(weights)

    def get_stats(self):
        """Get performance statistics"""
        stats = self.stats.copy()
        stats['active_players'] = 1  # For now
        stats['async_buffers_enabled'] = True
        return stats

    def reset(self):
        """Reset agent state"""
        self.stats = {
            'total_actions': 0,
            'episodes_completed': 0,
            'buffer_stores': 0,
            'combat_encounters': 0
        }


# Aliases and factory functions for testing
EnhancedMuZeroAgent = MuZeroAgent

def create_enhanced_muzero_agent(global_buffer=None):
    return MuZeroAgent(global_buffer=global_buffer)

