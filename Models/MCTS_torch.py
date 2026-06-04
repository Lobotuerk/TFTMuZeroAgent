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


from Models.tft_mcts import TFTMove, TFTState
from Models.batched_inference import BlockingBatchInferenceQueue

class EnhancedMCTS:
    """
    Enhanced MCTS implementation using new observation schema and PyMCTS integration
    """
    
    def __init__(self, sample_size: int, action_size: int, action_limits: List[int], 
                 policy_size: int, network, use_pymcts: bool = True, queue_size = None):
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

        # Use config if queue_size is not provided
        if queue_size is None:
            queue_size = getattr(config, 'OBSERVATION_TIME_STEPS', 1)
            
        self.obs_queue = deque(maxlen=queue_size)
        # Fill queue with empty observations
        for _ in range(queue_size):
            self.obs_queue.append(np.zeros(config.OBSERVATION_SIZE))
        
        # Batch inference queue for GPU-efficient batched recurrent_inference
        threshold = getattr(config, 'BATCHED_INFERENCE_THRESHOLD', 64)
        self.batch_queue = BlockingBatchInferenceQueue(
            network=network,
            batch_size=threshold,
            timeout_seconds=0.005,
        )

        # Performance tracking
        self.stats = {
            'total_generations': 0,
            'pymcts_generations': 0,
            'average_actions_per_state': 0,
            'average_depth': 0
        }
    
    def generate_action(self, n_simulations: int, observation: np.ndarray, 
                       mask: np.ndarray,
                       precomputed: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate actions using enhanced MCTS with new observation schema
        
        Args:
            n_simulations: Number of MCTS simulations
            observation: Current observations for all players
            mask: Action masks for all players
            precomputed: Pre-computed neural network results to skip initial_inference
            
        Returns:
            Tuple of (actions, target_policies)
        """
        self.obs_queue.append(observation)
        self.num_simulations = n_simulations
        return self._generate_action_pymcts(observation, mask, precomputed=precomputed)
    
    def _generate_action_pymcts(self, observation: np.ndarray, 
                               mask: np.ndarray,
                               precomputed: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate actions using PyMCTS integration"""
        self.stats['pymcts_generations'] += 1

        actions = []
        target_policies = []
        
        # Create TFT state with batch queue for batched GPU inference
        tft_state = TFTState(np.array(list(self.obs_queue)), mask, self.network,
                            precomputed=precomputed, batch_queue=self.batch_queue)

        self.mcts_agent = pymcts.MCTS_agent(
            pymcts.SerializedPythonState(tft_state), 
            max_iter=self.num_simulations,
            max_seconds=self.mcts_max_seconds
        )
        
        # Generate move using PyMCTS
        best_move = self.mcts_agent.genmove(None)
        # Flush any remaining items in the batch queue
        self.batch_queue.flush()
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
