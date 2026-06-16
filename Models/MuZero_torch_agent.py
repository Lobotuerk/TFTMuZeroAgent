import torch
import numpy as np
import asyncio
import os
from typing import List, Optional, Any, Union, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor
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
                 config_obj: Optional[Any] = None,
                 action_size: Optional[int] = None,
                 action_limits: Optional[List[int]] = None,
                 obs_size: Optional[int] = None,
                 simulations: Optional[int] = None,
                 weights: Optional[Dict[str, Any]] = None,
                 training: bool = True):
        super().__init__(agent_name, global_buffer)

        self.config = config_obj if config_obj is not None else config

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
        self.simulations = simulations if simulations is not None else getattr(self.config, 'NUM_SIMULATIONS', 10)
        
        # Model and MCTS initialization
        self.model = MuZeroNetwork()
        
        # Load weights if provided
        if weights is not None:
            self.load_weights_from_state_dict(weights)
        
        # Initialize Enhanced MCTS with action dimensions from schema/config
        # For TFTSet4Gym: ACTION_DIM = [7, 37, 37], so policy_size = sum(ACTION_DIM) = 81
        policy_size = sum(self.action_limits)
        
        self.mcts = EnhancedMCTS(
            sample_size=80,
            action_size=self.action_size,  # Read from ACTION_DIM
            action_limits=self.action_limits,  # Read from ACTION_DIM
            policy_size=policy_size,  # Calculate policy size based on action dimensions
            network=self.model
        )
        
        # Control exploration noise: added during training, disabled during evaluation
        self.mcts.training = training

        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model.to('cuda')
        
        # Persistent thread pool for batch inference
        self._executor = ThreadPoolExecutor(max_workers=getattr(self.config, 'BATCH_SIZE', 8))

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

        player_ids = kwargs.get('player_ids')

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

            precomputed_list = []
            for i in range(batch_size):
                precomputed_list.append({
                    'hidden_state': network_outputs['hidden_state'][i].detach(),
                    'policy': network_outputs['policy_logits'][i].cpu().numpy(),
                    'value': network_outputs['value'][i].cpu().numpy(),
                })

        def run_mcts_item(i):
            obs = observations[i]
            mask = masks[i] if i < len(masks) else np.ones(sum(config.ACTION_DIM), dtype=bool)
            pc = precomputed_list[i]

            pid = player_ids[i] if player_ids and i < len(player_ids) else "default"
            game_id = "default"
            if "thread_env_" in pid:
                game_id = pid.split("_player")[0]
            elif "env_" in pid:
                game_id = pid.split("_player")[0]

            env_move, action_vector = self._generate_action_with_mcts(obs, mask, precomputed=pc, game_id=game_id)

            value = 0.0
            if 'value' in pc:
                v = pc['value']
                value = float(v.item() if hasattr(v, 'item') else v)
            return env_move, action_vector, value

        results = list(self._executor.map(run_mcts_item, range(batch_size)))

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
                                   precomputed: Optional[Dict[str, Any]] = None,
                                   game_id: str = "default") -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate actions using Enhanced MCTS
        
        Args:
            observation: Current observation
            mask: Action mask
            precomputed: Pre-computed NN results (hidden_state, policy, value)
                         to avoid redundant initial_inference
            game_id: Identifier of the game environment
        """        
        # Use Enhanced MCTS for action generation
        actions, action_vector = self.mcts.generate_action(
            self.simulations, 
            observation=observation, 
            mask=mask,
            precomputed=precomputed,
            game_id=game_id
        )
        
        return actions, action_vector
    
    def get_weights(self) -> Dict[str, Any]:
        """Get model weights for sharing/saving"""
        return self.model.state_dict()
    
    def update_weights(self, weights):
        """Update model weights"""
        self.model.load_state_dict(weights)

    def save_model(self, episode):
        """Save model weights to the configured results path"""
        results_path = getattr(self.config, 'results_path', getattr(self.config, 'RESULTS_PATH', './Checkpoints'))
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        path = os.path.join(results_path, f"checkpoint_{episode}")
        torch.save(self.get_weights(), path)
        print(f"Model saved to {path}")

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



