import time
import config
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
import sys
import os
import threading
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
        
        # Thread safety setup
        self._local = threading.local()
        self._stats_lock = threading.Lock()

        # Exploration noise flag: True during training/data collection, False during evaluation
        self.training = True
        
        # PyMCTS agents configuration
        self.mcts_max_iterations = 100
        self.mcts_max_seconds = 10

        # Batch inference queues mapped by game_id for multi-game concurrency
        self.batch_queues = {}
        self.batch_queues_lock = threading.Lock()

        # For backward-compatibility with tests that expect self.batch_queue
        threshold = getattr(config, 'BATCHED_INFERENCE_THRESHOLD', 64)
        self.batch_queue = BlockingBatchInferenceQueue(
            network=network,
            batch_size=threshold,
            timeout_seconds=0.05,
        )

        # Performance tracking
        self.stats = {
            'total_generations': 0,
            'pymcts_generations': 0,
            'average_actions_per_state': 0,
            'average_depth': 0
        }
    
    def get_batch_queue(self, game_id: str) -> BlockingBatchInferenceQueue:
        """Get or create the dynamic batch queue for a specific game."""
        with self.batch_queues_lock:
            if game_id not in self.batch_queues:
                threshold = getattr(config, 'BATCHED_INFERENCE_THRESHOLD', 64)
                self.batch_queues[game_id] = BlockingBatchInferenceQueue(
                    network=self.network,
                    batch_size=threshold,
                    timeout_seconds=0.05,
                )
            return self.batch_queues[game_id]

    def _get_local_state(self):
        """Ensure thread-local variables are initialized."""
        if not hasattr(self._local, 'obs_queue'):
            queue_size = getattr(config, 'OBSERVATION_TIME_STEPS', 1)
            self._local.obs_queue = deque(maxlen=queue_size)
            for _ in range(queue_size):
                self._local.obs_queue.append(np.zeros(config.OBSERVATION_SIZE))
        if not hasattr(self._local, 'mcts_agent'):
            self._local.mcts_agent = None
        if not hasattr(self._local, 'num_simulations'):
            self._local.num_simulations = 0
        return self._local

    def generate_action(self, n_simulations: int, observation: np.ndarray, 
                       mask: np.ndarray,
                       precomputed: Optional[Dict[str, Any]] = None,
                       game_id: str = "default") -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate actions using enhanced MCTS with new observation schema
        
        Args:
            n_simulations: Number of MCTS simulations
            observation: Current observations for all players
            mask: Action masks for all players
            precomputed: Pre-computed neural network results to skip initial_inference
            game_id: Identifier of the game environment for dynamic queue grouping
            
        Returns:
            Tuple of (actions, target_policies)
        """
        state = self._get_local_state()
        state.obs_queue.append(observation)
        state.num_simulations = n_simulations
        
        batch_queue = self.get_batch_queue(game_id)
        batch_queue.register()
        try:
            return self._generate_action_pymcts(observation, mask, precomputed=precomputed, batch_queue=batch_queue)
        finally:
            batch_queue.deregister()
    
    def _generate_action_pymcts(self, observation: np.ndarray, 
                               mask: np.ndarray,
                               precomputed: Optional[Dict[str, Any]] = None,
                               batch_queue: Optional[BlockingBatchInferenceQueue] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate actions using PyMCTS integration"""
        state = self._get_local_state()
        with self._stats_lock:
            self.stats['pymcts_generations'] += 1

        actions = []
        target_policies = []
        
        if batch_queue is None:
            batch_queue = self.batch_queue
            
        # Create TFT state with batch queue for batched GPU inference
        # Add Dirichlet exploration noise at the root during training
        tft_state = TFTState(np.array(list(state.obs_queue)), mask, self.network,
                            precomputed=precomputed, batch_queue=batch_queue,
                            add_root_noise=self.training)

        state.mcts_agent = pymcts.MCTS_agent(
            pymcts.SerializedPythonState(tft_state), 
            max_iter=state.num_simulations,
            max_seconds=self.mcts_max_seconds
        )
        
        # Generate move using PyMCTS
        best_move = state.mcts_agent.genmove(None)
        # Flush any remaining items in the batch queue
        batch_queue.flush()
        with self._stats_lock:
            self.stats['total_generations'] += 1

        return best_move.to_env_action(), best_move.to_numpy()
    
    def fill_metadata(self) -> Dict[str, str]:
        """Fill metadata for tracking"""
        with self._stats_lock:
            total_g = self.stats['total_generations']
            pymcts_g = self.stats['pymcts_generations']
        metadata = {
            'network_id': str(self.network.training_steps() if hasattr(self.network, 'training_steps') else 0),
            'pymcts_available': 'True',
            'total_generations': str(total_g),
            'pymcts_generations': str(pymcts_g)
        }
        return metadata
    
    def cleanup_game(self, game_id: str):
        with self.batch_queues_lock:
            if game_id in self.batch_queues:
                del self.batch_queues[game_id]

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        state = self._get_local_state()
        with self._stats_lock:
            stats_copy = self.stats.copy()
        return {
            **stats_copy,
            'pymcts_available': True,
            'active_agents': state.mcts_agent is not None,
            'max_depth_search': self.max_depth_search,
            'runs': self.runs,
            'num_actions': self.num_actions
        }
    
    def reset_agents(self):
        """Reset all MCTS agents"""
        state = self._get_local_state()
        state.mcts_agent = None
        with self._stats_lock:
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
