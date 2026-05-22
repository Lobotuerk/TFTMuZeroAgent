import time
import config
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
import sys
import os
from collections import deque

# Add MonteCarloTreeSearch build directory to path
mcts_build_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                               'MonteCarloTreeSearch')
if mcts_build_path not in sys.path:
    sys.path.insert(0, mcts_build_path)


try:
    import pymcts
    PYMCTS_AVAILABLE = True
    MCTS_MoveBase = pymcts.MCTS_move
    MCTS_StateBase = pymcts.MCTS_state
except ImportError:
    raise ImportError("PyMCTS is required for this MCTS implementation. Please install the MonteCarloTreeSearch library and build the Python bindings.")

# Import new observation schema system
from Models.Common_agents import extract_field_from_observation


class TFTMove(MCTS_MoveBase):
    """
    TFT-specific move implementation for PyMCTS integration
    Uses TFTSet4Gym's 3D action format: [action_type, target_1, target_2]
    """
    
    def __init__(self, action_type: int, target_1: int = 0, target_2: int = 0, index: int = 0):
        super().__init__()
        self.action_type = action_type
        self.target_1 = target_1
        self.target_2 = target_2
        self.index = index
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, TFTMove):
            return False
        return (self.action_type == other.action_type and 
                self.target_1 == other.target_1 and
                self.target_2 == other.target_2 and
                self.index == other.index)
    
    def sprint(self) -> str:
        return f"TFT_move({self.action_type}, {self.target_1}, {self.target_2}, {self.index})"
    
    def to_numpy(self) -> List[float]:
        """Convert move to numpy array for neural network processing"""
        result = [0.0] * config.ACTION_CONCAT_SIZE
        result[self.index] = 1.0
        return result
    
    def __str__(self) -> str:
        return self.sprint()
    
    def to_env_action(self) -> List[int]:
        """Convert to environment action format"""
        return [self.action_type, self.target_1, self.target_2]


class TFTState(MCTS_StateBase):
    """
    TFT-specific state implementation for PyMCTS integration
    """
    
    def __init__(self, observation: np.ndarray, mask: Optional[np.ndarray] = None, 
                network=None, is_raw_observation: bool = True):
        super().__init__()
        
        self.mask = mask.copy() if mask is not None else self._create_default_mask()
        self.network = network

        self._is_terminal_cached = None

        if is_raw_observation:        
            if self.network is not None:
                with torch.no_grad():
                    input_obs = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
                    network_output = self.network.initial_inference(input_obs)
                    self.hidden_state = network_output["hidden_state"].squeeze(0).cpu().numpy()
                    self.policy = network_output["policy_logits"].squeeze(0).cpu().numpy()
                    self.value = network_output["value"].squeeze(0).cpu().numpy()
        else:
            with torch.no_grad():
                input_obs = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).cuda()
                self.hidden_state = observation
                self.policy, self.value = self.network.prediction(input_obs)
                self.policy = self.policy.squeeze(0).cpu().numpy()
                self.value = self.value.squeeze(0).cpu().numpy()
    
    def _create_default_mask(self) -> np.ndarray:
        """Create a default mask allowing basic actions"""
        mask = np.zeros((54,), dtype=bool)  # Default allowing most actions for TFTSet4Gym
        return mask
    
    def get_action_probabilities(self, moves: List[TFTMove]) -> List[float]:
        """Return softmax-normalized policy probabilities for each move.
        
        These probabilities are consumed by the PyMCTS PUCT engine for
        PUCT-based exploration (Q + c*P*sqrt(N)/(1+N)).
        
        Args:
            moves: List of TFTMove to evaluate
            
        Returns:
            List of probability values summing to 1.0
        """
        if not hasattr(self, 'policy') or self.policy is None:
            return [1.0 / len(moves)] * len(moves)

        scores = []
        for move in moves:
            idx = move.index
            if 0 <= idx < len(self.policy):
                scores.append(float(self.policy[idx]))
            else:
                scores.append(0.0)

        scores = np.array(scores, dtype=np.float64)
        scores = np.exp(scores - np.max(scores))
        total = scores.sum()
        if total > 0:
            return (scores / total).tolist()
        return [1.0 / len(moves)] * len(moves)

    def actions_to_try(self) -> List[TFTMove]:
        """Get all legal actions from current state.
        
        Sorting is delegated to the C++ PUCT engine.
        """
        actions = []

        # Always allow pass action
        actions.append(TFTMove(0, 0, 0))  # Pass
        
        # Roll action (if enough gold and allowed)
        actions.append(TFTMove(4, 0, 0))  # Roll
        
        # Level up action (if enough exp and allowed)
        actions.append(TFTMove(5, 0, 0))  # Level up

        # Buy actions
        for i in range(58):
            actions.append(TFTMove(2, i, 0))  # Buy champ i
        
        # Sell actions
        for pos in range(37):  # Board + bench positions
            actions.append(TFTMove(3, pos, 0))
        
        # Move actions
        for i in range(28):  # Limit move actions to avoid explosion
            for j in range(37):
                actions.append(TFTMove(1, i, j))  # Move board to bench

        return actions
    
    def next_state(self, move) -> 'TFTState':
        """Get the state resulting from applying the given move"""

        new_mask = self.mask.copy()
        
        # Use muzero model to predict next state
        if self.network is not None:
            with torch.no_grad():
                input_obs = torch.tensor(self.hidden_state, dtype=torch.float32).unsqueeze(0)
                
                network_output = self.network.recurrent_inference(input_obs, move.to_numpy())
                new_observation = network_output["hidden_state"].squeeze(0).cpu().numpy()
        
        return TFTState(new_observation, new_mask, self.network, is_raw_observation=False)
    
    def rollout(self) -> float:
        """Perform rollout based on policy distribution"""
        return self.value
        
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal state"""
        return False
    
    def print(self):
        """Print current state"""
        print(f"TFT State - Health: {self.health}, Round: {self.round_num}, "
              f"Level: {self.level}, TFC: {self.turns_for_combat}")
        print(f"Terminal: {self.is_terminal()}")
        print(f"Available Actions: {len(self.actions_to_try())}")
    
    def clone(self):
        """Create a deep copy of this state - required by MCTS_state base class"""
        return TFTState(
            observation=self.observation.copy(),
            mask=self.mask.copy() if self.mask is not None else None,
            network=self.network
        )
    
    def is_self_side_turn(self) -> bool:
        """Check if it's the self side's turn - required by MCTS_state base class"""
        return True

class EnhancedMCTS:
    """
    Enhanced MCTS implementation using new observation schema and PyMCTS integration
    """
    
    def __init__(self, sample_size: int, action_size: int, action_limits: List[int], 
                 policy_size: int, network, use_pymcts: bool = True, queue_size = 10):
        self.max_depth_search = 0
        self.runs = 0
        self.network = network
        self.times = [0] * 6
        self.NUM_ALIVE = config.NUM_PLAYERS
        self.num_actions = 0
        self.ckpt_time = time.time_ns()
        self.sample_size = sample_size
        self.action_size = action_size
        self.action_limits = action_limits
        self.policy_size = policy_size
        self.num_simulations = 0
        
        # PyMCTS agents for each player
        self.mcts_agent = None
        self.mcts_max_iterations = 100
        self.mcts_max_seconds = 10

        self.obs_queue = deque(maxlen=queue_size)
        # Fill queue with empty observations
        for _ in range(queue_size):
            self.obs_queue.append(np.zeros(config.OBSERVATION_SIZE))
        
        # Performance tracking
        self.stats = {
            'total_generations': 0,
            'pymcts_generations': 0,
            'average_actions_per_state': 0,
            'average_depth': 0
        }
    
    def generate_action(self, n_simulations: int, observation: np.ndarray, 
                       mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate actions using enhanced MCTS with new observation schema
        
        Args:
            n_simulations: Number of MCTS simulations
            observation: Current observations for all players
            mask: Action masks for all players
            
        Returns:
            Tuple of (actions, target_policies)
        """
        self.obs_queue.append(observation)
        self.num_simulations = n_simulations
        return self._generate_action_pymcts(observation, mask)
    
    def _generate_action_pymcts(self, observation: np.ndarray, 
                               mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate actions using PyMCTS integration"""
        self.stats['pymcts_generations'] += 1

        actions = []
        target_policies = []
        
        # Create TFT state
        tft_state = TFTState(np.array(list(self.obs_queue)), mask, self.network)

        self.mcts_agent = pymcts.MCTS_agent(
            pymcts.SerializedPythonState(tft_state), 
            max_iter=self.mcts_max_iterations,
            max_seconds=self.mcts_max_seconds
        )
        
        # Generate move using PyMCTS
        best_move = self.mcts_agent.genmove(None)        
        self.stats['total_generations'] += 1

        return best_move.to_env_action(), best_move.to_numpy()
    
    def add_exploration_noise(self, policy_logits: List, noises: List) -> List:
        """Add exploration noise to policy logits"""
        exploration_fraction = config.ROOT_EXPLORATION_FRACTION
        for i in range(len(noises)):
            for j in range(min(len(noises[i]), len(policy_logits[i]))):
                policy_logits[i][j] = (policy_logits[i][j] * (1 - exploration_fraction) + 
                                     noises[i][j] * exploration_fraction)
        return policy_logits
    
    def fill_metadata(self) -> Dict[str, str]:
        """Fill metadata for tracking"""
        metadata = {
            'network_id': str(self.network.training_steps() if hasattr(self.network, 'training_steps') else 0),
            'pymcts_available': 'True',
            'total_generations': str(self.stats['total_generations']),
            'pymcts_generations': str(self.stats['pymcts_generations'])
        }
        return metadata
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            **self.stats,
            'pymcts_available': True,
            'active_agents': self.mcts_agent is not None,
            'max_depth_search': self.max_depth_search,
            'runs': self.runs,
            'num_actions': self.num_actions
        }
    
    def reset_agents(self):
        """Reset all MCTS agents"""
        self.mcts_agent = None
        self.stats = {
            'total_generations': 0,
            'pymcts_generations': 0,
            'average_actions_per_state': 0,
            'average_depth': 0
        }

# Factory function for easy creation
def create_enhanced_mcts(sample_size: int, action_size: int, action_limits: List[int],
                        policy_size: int, network, use_pymcts: bool = True) -> EnhancedMCTS:
    """
    Create an enhanced MCTS instance
    
    Args:
        sample_size: Number of samples for action selection
        action_size: Size of action space
        action_limits: Limits for each action dimension
        policy_size: Size of policy vector
        network: Neural network for value/policy estimation
        use_pymcts: Ignored (always uses PyMCTS)
        
    Returns:
        Configured EnhancedMCTS instance
    """
    return EnhancedMCTS(
        sample_size=sample_size,
        action_size=action_size,
        action_limits=action_limits,
        policy_size=policy_size,
        network=network,
        use_pymcts=use_pymcts
    )


# Utility functions (avoiding duplicates by removing the obscured functions)

def masked_distribution(x, use_exp, mask=None):
    if mask is None:
        mask = [1] * len(x)
    assert sum(mask) > 0, 'Not all values can be masked.'
    assert len(mask) == len(x), (
        'The dimensions of the mask and x need to be the same.')
    x = np.exp(x) if use_exp else np.array(x, dtype=np.float64)
    mask = np.array(mask, dtype=np.float64)
    x *= mask
    if sum(x) == 0:
        # No unmasked value has any weight. Use uniform distribution over unmasked
        # tokens.
        x = mask
    return x / np.sum(x, keepdims=True)


def masked_softmax(x, mask=None):
    x = np.array(x) - np.max(x, axis=-1)  # to avoid overflow
    return masked_distribution(x, use_exp=True, mask=mask)


def masked_count_distribution(x, mask=None):
    return masked_distribution(x, use_exp=False, mask=mask)
