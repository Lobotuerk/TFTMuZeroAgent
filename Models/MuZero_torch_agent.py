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
                 global_buffer: Optional[Any] = None,
                 action_size: Optional[int] = None,
                 action_limits: Optional[List[int]] = None,
                 obs_size: Optional[int] = None,
                 simulations: Optional[int] = None,
                 weights: Optional[Dict[str, Any]] = None):
        super().__init__(agent_name, global_buffer)

        # Read action dimensions from observation schema/config
        self.action_limits = action_limits if action_limits is not None else ACTION_DIM.copy()
        self.action_size = action_size if action_size is not None else len(self.action_limits)
        
        # Read observation size from schema if available
        if obs_size is not None:
            self.obs_size = obs_size
        else:
            schema = get_observation_schema("current_player")
            self.obs_size = schema.total_size
        
        # Set simulations from config if not provided
        self.simulations = simulations if simulations is not None else getattr(config, 'NUM_SIMULATIONS', 10)
        
        # Model and MCTS initialization
        self.model = MuZeroNetwork()
        
        # Load weights if provided
        if weights is not None:
            self.load_weights_from_state_dict(weights)
        
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

    def _batch_select_action_impl(self, observations, masks, rewards=None, terminated=None, precomputed_results=None, **kwargs):
        batch_size = len(observations)
        if batch_size == 0:
            return []

        if precomputed_results is not None:
            precomputed_list = precomputed_results
        else:
            obs_tensors = []
            for obs in observations:
                if isinstance(obs, np.ndarray):
                    obs_tensors.append(torch.from_numpy(obs).float())
                else:
                    obs_tensors.append(torch.tensor(obs, dtype=torch.float32))
            batch_tensor = torch.stack(obs_tensors)
            if torch.cuda.is_available():
                batch_tensor = batch_tensor.cuda()

            with torch.no_grad():
                network_outputs = self.model.initial_inference(batch_tensor)
                
            # Handle both dict and (dict, directive, board) returns from model
            if isinstance(network_outputs, tuple):
                network_outputs = network_outputs[0]

            precomputed_list = []
            for i in range(batch_size):
                precomputed_list.append({
                    'hidden_state': network_outputs['hidden_state'][i].detach(),
                    'policy': network_outputs['policy_logits'][i].cpu().numpy(),
                    'value': network_outputs['value'][i].cpu().numpy(),
                })

        results = []
        for i in range(batch_size):
            obs = observations[i]
            mask = masks[i] if i < len(masks) else np.ones(54, dtype=bool)
            pc = precomputed_list[i]

            env_move, action_vector = self._generate_action_with_mcts(obs, mask, precomputed=pc)

            value = 0.0
            if 'value' in pc:
                v = pc['value']
                value = float(v.item() if hasattr(v, 'item') else v)
            results.append((env_move, action_vector, value))

        return results

    def load_weights_from_state_dict(self, weights: Dict[str, Any]):
        """Load model weights from a state dict"""
        device = next(self.model.parameters()).device
        state_dict = {k: torch.as_tensor(v).to(device) for k, v in weights.items()}
        self.model.load_state_dict(state_dict)

    def load_weights(self, weights_path: str):
        """Load model weights from a specified path"""
        weights = torch.load(weights_path, map_location=self.model.device)
        self.model.load_state_dict(weights)
        print(f"Weights loaded successfully from {weights_path}")
    
    def _select_action_impl(self, 
                     observation, action_mask, reward=None, terminated=None, precomputed_results=None) -> Tuple[List[int], np.ndarray, float]:
        """
        Select actions using MCTS with enhanced observation processing
        """
        # Generate actions using MCTS
        env_move, action_vector = self._generate_action_with_mcts(
            observation, action_mask, precomputed=precomputed_results
        )
        
        # Extract value from precomputed if available
        value = 0.0
        if precomputed_results and 'value' in precomputed_results:
            v = precomputed_results['value']
            value = float(v.item() if hasattr(v, 'item') else v)

        return env_move, action_vector, value
    
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
