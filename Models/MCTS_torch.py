import time
import config
import numpy as np
import torch
import torch.nn as nn
import Models.MCTS_Util as util
from typing import Dict, Optional
from Models.tft_mcts import TFTState, TFTMove, create_tft_state_from_env

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MonteCarloTreeSearch'))
import pymcts

"""
EXPLANATION OF ENHANCED MCTS:
1. Use TFT MCTS bridge for proper game state representation
2. Integrate neural network for enhanced rollout evaluation
3. Combine traditional MCTS with neural network guidance
4. Select actions using both tree search and network policy
"""


class EnhancedTFTState(TFTState):
    """Enhanced TFT state that uses neural network for rollout evaluation."""
    
    def __init__(self, observations, current_player, network: nn.Module, **kwargs):
        super().__init__(observations, current_player, **kwargs)
        self.network = network
    
    def rollout(self) -> float:
        """Enhanced rollout using neural network evaluation."""
        # Get current player's observation
        obs = self.observations[self.current_player]
        
        # Convert observation to tensor if needed
        if isinstance(obs, np.ndarray):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        else:
            obs_tensor = obs
        
        # Use network to evaluate state value
        with torch.no_grad():
            value = self.network(obs_tensor)
            
            # Convert to probability between 0 and 1
            if torch.is_tensor(value):
                value = torch.sigmoid(value).item()
            
            return float(np.clip(value, 0.0, 1.0))
    
    def next_state(self, move: TFTMove) -> 'EnhancedTFTState':
        """Return enhanced next state with network propagated."""
        next_tft_state = super().next_state(move)
        return EnhancedTFTState(
            observations=next_tft_state.observations,
            current_player=next_tft_state.current_player,
            network=self.network,
            env_state=next_tft_state.env_state,
            round_num=next_tft_state.round_num
        )


class MCTS:
    def __init__(self, network: nn.Module, sample_size: int = 80, 
                 action_size: int = 3, action_limits = None, 
                 policy_size: int = 1000, max_simulations: Optional[int] = None):
        """
        Initialize MCTS with neural network integration.
        
        Args:
            network: PyTorch neural network for rollout enhancement
            sample_size: Number of actions to sample
            action_size: Size of action space  
            action_limits: Limits for action space
            policy_size: Size of policy vector
            max_simulations: Maximum MCTS simulations per move
        """
        self.network = network
        self.sample_size = sample_size
        self.action_size = action_size
        self.action_limits = action_limits or [6, 37, 28]
        self.policy_size = policy_size
        self.max_simulations = max_simulations or getattr(config, 'NUM_SIMULATIONS', 50)
        
        # Performance tracking
        self.max_depth_search = 0
        self.runs = 0
        self.times = [0] * 6
        self.num_actions = 0
        self.ckpt_time = time.time_ns()
        
        # MCTS state management
        self.current_state = None
        self.mcts_agent = None

    def generate_action(self, observations: Dict, masks: Optional[Dict] = None, 
                       player_id: str = "player_0") -> tuple:
        """
        Generate action using enhanced MCTS with neural network guidance.
        
        Args:
            observations: Dictionary of player observations
            masks: Optional action masks
            player_id: ID of the player to generate action for
            
        Returns:
            tuple: (action, target_policy, state_info)
        """
        # Create enhanced TFT state with neural network
        tft_state = EnhancedTFTState(
            observations=observations,
            current_player=player_id,
            network=self.network
        )
        
        # Create SerializedPythonState for PyMCTS integration
        wrapped_state = pymcts.SerializedPythonState(tft_state)
        
        # Create MCTS agent if needed
        if self.mcts_agent is None:
            self.mcts_agent = pymcts.MCTS_agent(
                wrapped_state, 
                max_iter=self.max_simulations,
                max_seconds=2
            )
        
        # Generate move using MCTS
        move = self.mcts_agent.genmove(None)
        
        # Convert MCTS move to action format
        action_info = self._parse_mcts_move(move)
        
        # Generate target policy
        target_policy = self._create_target_policy(tft_state, action_info)
        
        self.num_actions += 1
        return action_info['action'], target_policy, {
            'move_type': action_info['type'],
            'confidence': action_info.get('confidence', 0.5),
            'search_depth': self.max_depth_search
        }
    

    
    def _parse_mcts_move(self, move) -> Dict:
        """
        Parse PyMCTS move into TFT action format.
        
        Args:
            move: Move object from PyMCTS
            
        Returns:
            Dictionary with action information
        """
        # Convert PyMCTS move to actionable format
        move_str = str(move)
        
        # Default action mapping - extend based on actual move parsing needs
        if 'reroll' in move_str.lower():
            return {
                'action': 'reroll',
                'type': 'shop',
                'confidence': 0.8
            }
        elif 'level' in move_str.lower():
            return {
                'action': 'level',
                'type': 'experience',
                'confidence': 0.7
            }
        elif 'buy' in move_str.lower():
            return {
                'action': 'buy_unit',
                'type': 'shop',
                'confidence': 0.6,
                'shop_index': 0  # Parse from move_str in full implementation
            }
        else:
            return {
                'action': 'pass',
                'type': 'general',
                'confidence': 0.3
            }
    
    def _create_target_policy(self, state: EnhancedTFTState, action_info: Dict) -> np.ndarray:
        """
        Create target policy distribution for training.
        
        Args:
            state: Current TFT state
            action_info: Information about selected action
            
        Returns:
            Policy probability distribution
        """
        # Simple uniform policy for now - could be enhanced with network policy head
        policy = np.zeros(self.policy_size)
        
        # Set higher probability for selected action type
        if action_info['type'] == 'shop':
            policy[0:50] = 0.02  # Shop actions
        elif action_info['type'] == 'experience':
            policy[50:60] = 0.1  # Experience actions
        else:
            policy[60:100] = 0.025  # General actions
        
        # Normalize
        policy_sum = np.sum(policy)
        if policy_sum > 0:
            policy = policy / policy_sum
        
        return policy.reshape(-1, 1)
    


    def update_state(self, observations: Dict, current_player: str):
        """
        Update internal state representation.
        
        Args:
            observations: New observations
            current_player: Current player ID
        """
        self.current_state = EnhancedTFTState(
            observations=observations,
            current_player=current_player,
            network=self.network
        )
    
    def set_network(self, network: nn.Module):
        """
        Update the neural network used for rollout enhancement.
        
        Args:
            network: New PyTorch network
        """
        self.network = network
        if self.current_state is not None:
            self.current_state.network = network
    
    def get_statistics(self) -> Dict:
        """
        Get MCTS performance statistics.
        
        Returns:
            Dictionary of performance metrics
        """
        return {
            'total_actions': self.num_actions,
            'max_search_depth': self.max_depth_search,
            'total_runs': self.runs,
            'avg_time_per_action': np.mean(self.times) if self.times else 0,
            'simulations_per_move': self.max_simulations
        }
    
    def visit_softmax_temperature(self) -> float:
        """
        Temperature for action selection.
        Dynamic temperature that decreases over time.
        
        Returns:
            Temperature value
        """
        # Start high for exploration, decrease for exploitation
        base_temp = 1.5
        decay_rate = 0.95
        min_temp = 0.1
        
        temp = base_temp * (decay_rate ** self.num_actions)
        return max(temp, min_temp)
    
    def select_action(self, distributions: list, temperature: float = 1.0, 
                     deterministic: bool = False):
        """
        Select action from probability distribution.
        
        Args:
            distributions: Action probability distributions
            temperature: Temperature for softmax
            deterministic: Whether to use argmax or sample
            
        Returns:
            Selected action index
        """
        if deterministic:
            return np.argmax(distributions)
        
        # Apply temperature
        if temperature > 0:
            logits = np.log(np.array(distributions) + 1e-8) / temperature
            probs = np.exp(logits) / np.sum(np.exp(logits))
        else:
            probs = distributions
        
        # Sample from distribution
        return np.random.choice(len(probs), p=probs)

