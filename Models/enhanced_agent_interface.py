"""
Enhanced Agent Interface for TFT MuZero Agent

This module provides improved batching, GPU utilization, and agent management
compared to the original AI_interface.py Agregator class.

"""

import torch
import numpy as np
import asyncio
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import time
from collections import defaultdict
import threading
from queue import Queue, Empty
import sys
import os

# Add parent directory to path to access root-level modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import from current Models directory with proper error handling
try:
    from .MuZero_torch_agent import MuZeroAgent
    from .Common_agents import RandomAgent, CultistAgent, DivineAgent
except (ImportError, ValueError):
    # Fallback: add models directory to path and import directly
    models_dir = os.path.dirname(__file__)
    if models_dir not in sys.path:
        sys.path.insert(0, models_dir)
    
    try:
        from MuZero_torch_agent import MuZeroAgent
        from Common_agents import RandomAgent, CultistAgent, DivineAgent
    except ImportError as e:
        print(f"Warning: Could not import agent classes: {e}")
        # Create dummy classes for testing
        class MuZeroAgent:
            def __init__(self, *args, **kwargs): pass
        class RandomAgent:
            def __init__(self, *args, **kwargs): pass
        class CultistAgent:
            def __init__(self, *args, **kwargs): pass
        class DivineAgent:
            def __init__(self, *args, **kwargs): pass

import config


@dataclass
class InferenceRequest:
    """Container for a single inference request"""
    player_id: str
    observation: np.ndarray
    mask: np.ndarray
    reward: float
    terminated: bool
    timestamp: float
    future: Optional[asyncio.Future] = None


@dataclass
class BatchedInferenceRequest:
    """Container for batched inference requests"""
    observations: torch.Tensor
    masks: Union[torch.Tensor, List, np.ndarray]  # Flexible mask type
    rewards: List[float]
    terminated: List[bool]
    request_ids: List[str]
    agent_type: type


class EnhancedBatchProcessor:
    """
    Advanced batching system that efficiently manages GPU inference for multiple agents.
    
    Features:
    - Dynamic batch sizing based on GPU memory
    - Adaptive timeout for batch collection
    - Memory-efficient tensor operations
    - Proper GPU memory management
    """
    
    def __init__(self, 
                 max_batch_size: int = 32,
                 batch_timeout_ms: float = 10.0,
                 gpu_memory_fraction: float = 0.8):
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.gpu_memory_fraction = gpu_memory_fraction
        
        # Request queues per agent type
        self.request_queues: Dict[type, Queue] = defaultdict(Queue)
        self.processing_locks: Dict[type, asyncio.Lock] = defaultdict(asyncio.Lock)  # Separate lock per agent type
        self.processing_tasks: Dict[type, asyncio.Task] = {}  # Track processing tasks to avoid duplicates
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # GPU memory management
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_gpu_memory()
    
    def _setup_gpu_memory(self):
        """Configure GPU memory settings for optimal batch processing"""
        if torch.cuda.is_available():
            # Set memory fraction to avoid OOM
            torch.cuda.set_per_process_memory_fraction(self.gpu_memory_fraction)
            # Enable memory mapping for efficient transfers
            torch.backends.cudnn.benchmark = True
            print(f"GPU setup complete. Device: {self.device}")
    
    async def add_request(self, request: InferenceRequest, agent_type: type) -> Any:
        """Add inference request to appropriate queue and return future result"""
        request.future = asyncio.Future()
        self.request_queues[agent_type].put(request)
        
        # Only start processing task if one isn't already running for this agent type
        if agent_type not in self.processing_tasks or self.processing_tasks[agent_type].done():
            self.processing_tasks[agent_type] = asyncio.create_task(
                self._process_agent_batch(agent_type)
            )
        
        result = await request.future
        return result
    
    async def _process_agent_batch(self, agent_type: type):
        """Process a batch of requests for a specific agent type"""
        async with self.processing_locks[agent_type]:  # Use per-agent-type lock
            requests = self._collect_batch(agent_type)
            if not requests:
                return
            
            # Create batched tensors
            batched_request = self._create_batch(requests, agent_type)
            
            # Perform inference
            results = await self._run_batched_inference(batched_request)
            
            # Distribute results back to futures
            self._distribute_results(requests, results)
    
    def _collect_batch(self, agent_type: type) -> List[InferenceRequest]:
        """Collect requests from queue up to batch size or timeout"""
        requests = []
        queue = self.request_queues[agent_type]
        start_time = time.time()
        
        # Collect available requests more aggressively
        while len(requests) < self.max_batch_size:
            try:
                # Quick collection with very short timeout
                request = queue.get(timeout=0.01)
                requests.append(request)
                
                # If we've been collecting for too long, break
                if (time.time() - start_time) * 1000 > self.batch_timeout_ms:
                    break
                    
            except Empty:
                # If queue is empty and we have requests, process them immediately
                if requests:
                    break
                    
                # If no requests yet, wait a bit more but not too long
                elapsed_ms = (time.time() - start_time) * 1000
                if elapsed_ms > min(10, self.batch_timeout_ms):  # Max 10ms wait for first request
                    break
                    
                time.sleep(0.001)  # Very short sleep to avoid busy waiting
        
        return requests
    
    def _create_batch(self, requests: List[InferenceRequest], agent_type: type) -> BatchedInferenceRequest:
        """Create batched tensors from individual requests"""
        observations = []
        masks = []
        rewards = []
        terminated = []
        request_ids = []
        
        for req in requests:
            # Ensure observation is numpy array before converting to tensor
            obs = req.observation
            if not isinstance(obs, np.ndarray):
                obs = np.array(obs)
            
            # Handle object dtype arrays by converting to float
            if obs.dtype == np.object_:
                try:
                    # Try to convert to float array
                    obs = obs.astype(np.float32)
                except (ValueError, TypeError):
                    # If conversion fails, flatten and convert individually
                    flat_obs = obs.flatten()
                    numeric_obs = []
                    for item in flat_obs:
                        try:
                            numeric_obs.append(float(item))
                        except (ValueError, TypeError):
                            numeric_obs.append(0.0)  # Default value for non-numeric items
                    obs = np.array(numeric_obs, dtype=np.float32).reshape(obs.shape)
            
            # Ensure dtype is compatible with PyTorch
            if obs.dtype not in [np.float32, np.float64, np.int32, np.int64, np.bool_]:
                obs = obs.astype(np.float32)
            
            observations.append(torch.from_numpy(obs))
            masks.append(req.mask)  # Keep as numpy for now
            rewards.append(req.reward)
            terminated.append(req.terminated)
            request_ids.append(req.player_id)
        
        # Stack observations into batch tensor with shape validation
        if observations:
            # Ensure all observations have the same shape
            first_obs_shape = observations[0].shape
            all_same_shape = all(obs.shape == first_obs_shape for obs in observations)
            
            if all_same_shape:
                batch_obs = torch.stack(observations).to(self.device)
            else:
                # Fallback: pad or truncate to match first observation
                target_shape = first_obs_shape
                processed_obs = []
                for obs in observations:
                    if obs.numel() != target_shape.numel():
                        # Flatten and pad/truncate to match target size
                        flat_obs = obs.flatten()
                        target_size = target_shape.numel()
                        if flat_obs.size(0) < target_size:
                            # Pad with zeros
                            padding = torch.zeros(target_size - flat_obs.size(0))
                            flat_obs = torch.cat([flat_obs, padding])
                        elif flat_obs.size(0) > target_size:
                            # Truncate
                            flat_obs = flat_obs[:target_size]
                        obs = flat_obs.reshape(target_shape)
                    processed_obs.append(obs)
                batch_obs = torch.stack(processed_obs).to(self.device)
        else:
            # Empty batch
            batch_obs = torch.empty(0).to(self.device)
        
        return BatchedInferenceRequest(
            observations=batch_obs,
            masks=masks,  # Keep as list for now, will be converted per agent
            rewards=rewards,
            terminated=terminated,
            request_ids=request_ids,
            agent_type=agent_type
        )
    
    async def _run_batched_inference(self, batch: BatchedInferenceRequest) -> List[Any]:
        """Run inference on batched data"""
        # This would be implemented by the specific agent
        # For now, return dummy results
        return [f"action_{i}" for i in range(len(batch.request_ids))]
    
    def _distribute_results(self, requests: List[InferenceRequest], results: List[Any]):
        """Distribute results back to the requesting futures"""
        for req, result in zip(requests, results):
            if req.future and not req.future.done():
                req.future.set_result(result)


class EnhancedAgentManager:
    """
    Advanced agent management system that replaces the simple Agregator.
    
    Features:
    - Efficient batch processing
    - GPU memory optimization
    - Agent lifecycle management
    - Performance monitoring
    """
    
    def __init__(self, batch_processor: Optional[EnhancedBatchProcessor] = None):
        self.agents: Dict[type, Any] = {}
        self.player_to_agent: Dict[str, type] = {}
        self.batch_processor = batch_processor or EnhancedBatchProcessor()
        
        # Performance monitoring
        self.inference_times = defaultdict(list)
        self.batch_sizes = defaultdict(list)
    
    def register_agent(self, agent_instance: Any, player_ids: List[str]):
        """Register an agent instance with specific player IDs"""
        agent_type = type(agent_instance)
        
        if agent_type not in self.agents:
            self.agents[agent_type] = agent_instance
        
        for player_id in player_ids:
            self.player_to_agent[player_id] = agent_type
        
        print(f"Registered {len(player_ids)} players for agent {agent_type.__name__}")
    
    def get_player_agent_mapping(self) -> Dict[str, type]:
        """Get mapping from player IDs to agent types"""
        return self.player_to_agent.copy()
    
    def setup_agents(self, agent_configs: List[Tuple[Any, int]]):
        """Setup agents with specified counts"""
        player_counter = 0
        
        for agent_instance, count in agent_configs:
            if count <= 0:
                continue
                
            player_ids = [f"player_{player_counter + i}" for i in range(count)]
            self.register_agent(agent_instance, player_ids)
            player_counter += count
        
        if player_counter > config.NUM_PLAYERS:
            raise ValueError(f"Total agents ({player_counter}) exceeds max players ({config.NUM_PLAYERS})")
    
    async def get_actions(self, 
                         observations: Dict[str, Dict],
                         rewards: Dict[str, float],
                         terminated: Dict[str, bool]) -> Dict[str, Any]:
        """Get actions for all players using efficient batching"""
        
        # Create inference requests
        requests = []
        for player_id, obs in observations.items():
            if player_id in self.player_to_agent:
                agent_type = self.player_to_agent[player_id]
                
                # Get mask, provide default if not available
                mask = obs.get('action_mask', np.ones(54, dtype=bool))
                
                request = InferenceRequest(
                    player_id=player_id,
                    observation=obs['tensor'],
                    mask=mask,
                    reward=rewards.get(player_id, 0.0),
                    terminated=terminated.get(player_id, False),
                    timestamp=time.time()
                )
                requests.append((request, agent_type))
        
        # Submit all requests concurrently
        tasks = []
        for request, agent_type in requests:
            task = self.batch_processor.add_request(request, agent_type)
            tasks.append((request.player_id, task))
        
        actions = {}
        # Use asyncio.gather to wait for all tasks concurrently
        results = await asyncio.gather(*[task for _, task in tasks])

        for (player_id, _), result in zip(tasks, results):
            actions[player_id] = result
        print(actions)
        return actions
    
    def _get_fallback_action(self) -> Any:
        """Generate fallback action when inference fails"""
        return [0, 0, 0]  # Default action format
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for monitoring"""
        stats = {}
        for agent_type, times in self.inference_times.items():
            if times:
                stats[agent_type.__name__] = {
                    'avg_inference_time': np.mean(times),
                    'total_inferences': len(times),
                    'avg_batch_size': np.mean(self.batch_sizes[agent_type]) if self.batch_sizes[agent_type] else 0
                }
        return stats


class TorchBasedBatchProcessor(EnhancedBatchProcessor):
    """
    PyTorch-optimized batch processor that leverages GPU parallelism effectively.
    
    This replaces Ray for GPU parallel processing with native PyTorch operations.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_instances: Dict[type, Any] = {}
        # Initialize parent class attributes that we'll need
        self.inference_times = defaultdict(list)
        self.batch_sizes = defaultdict(list)
    
    def register_agent_instance(self, agent_type: type, agent_instance: Any):
        """Register agent instance for batched inference"""
        self.agent_instances[agent_type] = agent_instance
    
    async def _run_batched_inference(self, batch: BatchedInferenceRequest) -> List[Any]:
        """Run actual batched inference using the registered agent"""
        agent = self.agent_instances.get(batch.agent_type)
        if not agent:
            raise ValueError(f"No agent registered for type {batch.agent_type}")
        
        start_time = time.time()
        
        # Run inference in thread pool with timeout to avoid blocking event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self.executor, self._run_agent_inference_sync, agent, batch)
        
        inference_time = time.time() - start_time
        
        # Record performance metrics
        self.inference_times[batch.agent_type].append(inference_time)
        self.batch_sizes[batch.agent_type].append(len(batch.request_ids))
            
        return result

    def _run_agent_inference_sync(self, agent: Any, batch: BatchedInferenceRequest) -> List[Any]:
        """Run the actual agent inference synchronously in thread pool.
        
        When the agent exposes both batch_select_action and a model with
        initial_inference, performs a single GPU forward pass on the whole
        batch tensor and passes pre-computed results to the agent, avoiding
        N separate forward passes.
        """
        batch_size = len(batch.request_ids)
        if batch.observations.size == 0:
            batch_size = 0

        # Attempt true batched GPU inference: run model.initial_inference
        # once on the full tensor and thread per-item results through.
        model = getattr(agent, 'model', None)
        has_batch_api = hasattr(agent, 'batch_select_action')
        can_batch_gpu = has_batch_api and model is not None and hasattr(model, 'initial_inference')

        if can_batch_gpu and batch_size > 0:
            try:
                with torch.no_grad():
                    network_output = model.initial_inference(batch.observations)

                precomputed_results = []
                for i in range(batch_size):
                    precomputed_results.append({
                        'hidden_state': network_output['hidden_state'][i].cpu().numpy(),
                        'policy': network_output['policy_logits'][i].cpu().numpy(),
                        'value': network_output['value'][i].cpu().numpy(),
                    })

                # Build numpy observation list and mask list for the agent
                obs_list = []
                mask_list = []
                for i in range(batch_size):
                    obs_np = batch.observations[i].cpu().numpy()
                    if obs_np.ndim > 1:
                        obs_np = obs_np.flatten()
                    obs_list.append(obs_np)
                    mask_list.append(
                        batch.masks[i] if i < len(batch.masks) else np.ones(54, dtype=bool)
                    )

                return agent.batch_select_action(obs_list, mask_list,
                                                  precomputed_results=precomputed_results)
            except Exception as e:
                print(f"Batched GPU inference failed, falling back: {e}")

        # Fallback path: convert batch tensor to numpy list, then use
        # batch_select_action without precomputed results or per-item calls.
        observations = []
        masks = []
        for i in range(batch_size):
            try:
                if batch.observations.size == 0:
                    obs = np.zeros((2504,))
                else:
                    obs = batch.observations[i].cpu().numpy()
                    if obs.ndim > 1:
                        obs = obs.flatten()
                observations.append(obs)
                mask = batch.masks[i] if i < len(batch.masks) else np.ones(54, dtype=bool)
                masks.append(mask)
            except Exception as e:
                print(f"Error extracting observation {i}: {e}")
                observations.append(np.zeros((2504,)))
                masks.append(np.ones(54, dtype=bool))

        if has_batch_api:
            try:
                return agent.batch_select_action(observations, masks)
            except Exception as e:
                print(f"batch_select_action failed, falling back: {e}")

        # Final fallback: per-item select_action calls
        actions = []
        for i, obs in enumerate(observations):
            try:
                if hasattr(agent, 'select_action'):
                    action = agent.select_action(obs, masks[i])
                    actions.append(action)
                else:
                    actions.append([0, 0, 0])
            except Exception as e:
                print(f"Error in agent inference for request {i}: {e}")
                actions.append([0, 0, 0])
        return actions
    
    def _fallback_inference(self, agent: Any, batch: BatchedInferenceRequest) -> List[Any]:
        """Fallback for agents that don't support batching"""
        actions = []
        for i in range(len(batch.request_ids)):
            try:
                # Extract individual observation with proper error handling
                if batch.observations.size == 0:
                    # If observations are empty, create a default observation
                    obs = np.zeros((1, 5152))  # Default observation shape for TFT
                else:
                    obs = batch.observations[i:i+1].cpu().numpy()
                    # Ensure obs has proper shape
                    if obs.size == 0:
                        obs = np.zeros((1, 5152))  # Default observation shape
                
                mask = [batch.masks[i]] if i < len(batch.masks) else [np.ones(54, dtype=bool)]
                reward = [batch.rewards[i]] if i < len(batch.rewards) else [0.0]
                terminated = [batch.terminated[i]] if i < len(batch.terminated) else [False]
                
                if hasattr(agent, 'select_action'):
                    # Check agent signature to call with correct parameters
                    import inspect
                    sig = inspect.signature(agent.select_action)
                    param_count = len(sig.parameters)
                    
                    if param_count >= 4:
                        # For MuZero agents that can handle all parameters
                        action = agent.select_action(obs, np.array(mask), reward, terminated)
                    else:
                        # For simple agents that only expect observation and mask
                        action = agent.select_action(obs, np.array(mask))
                    
                    actions.append(action[0] if isinstance(action, list) else action)
                else:
                    actions.append([0, 0, 0])  # Default action
            except Exception as e:
                print(f"Error in fallback inference for request {i}: {e}")
                actions.append([0, 0, 0])  # Safe fallback action
        
        return actions


# Alternative to Ray for parallel game execution
class AsyncGameEnvironment:
    """
    Async-based game environment that replaces Ray workers for better efficiency.
    
    Benefits over Ray:
    - Lower overhead for communication
    - Better memory management
    - Easier debugging and profiling
    - Native Python async/await patterns
    """
    
    def __init__(self, env_factory, agent_manager: EnhancedAgentManager):
        self.env_factory = env_factory
        self.agent_manager = agent_manager
        self.active_games = {}
    
    async def run_game(self, game_id: str) -> Dict[str, Any]:
        """Run a single game asynchronously"""
        env = self.env_factory()
        observations = env.reset()[0]
        terminated = {player_id: False for player_id in env.possible_agents}
        rewards = {player_id: 0.0 for player_id in env.possible_agents}
        scores = {player_id: 0.0 for player_id in env.possible_agents}
        
        game_start = time.time()
        
        while not all(terminated.values()):
            # Get actions using enhanced batching
            actions = await self.agent_manager.get_actions(observations, rewards, terminated)
            
            # Step environment
            observations, rewards, terminated, _, info = env.step(actions)
            
            # Update scores for terminated players
            for player in terminated.keys():
                if terminated[player]:
                    scores[player] = rewards[player]
        
        game_duration = time.time() - game_start
        
        return {
            'game_id': game_id,
            'scores': scores,
            'duration': game_duration,
            'final_placements': self._calculate_placements(scores)
        }
    
    def _calculate_placements(self, scores: Dict[str, Union[int, float]]) -> Dict[str, int]:
        """Calculate final placements based on scores"""
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        placements = {}
        for i, (player_id, score) in enumerate(sorted_scores):
            placements[player_id] = i + 1
        return placements


# Factory function for creating enhanced agent setups
def create_enhanced_setup(agent_configs: Optional[List[Tuple[Any, int]]] = None,
                         max_batch_size: Optional[int] = None,
                         batch_timeout_ms: float = 5.0,
                         gpu_memory_fraction: float = 0.7):
    """
    Factory function to create the enhanced agent system.
    
    Args:
        agent_configs: List of (agent_instance, count) tuples. If None, creates default setup.
        max_batch_size: Maximum batch size for processing. If None, uses NUM_PLAYERS.
        batch_timeout_ms: Timeout for batch collection in milliseconds.
        gpu_memory_fraction: Fraction of GPU memory to use.
    
    Returns:
        Tuple of (agent_manager, batch_processor)
    """
    # Set default batch size if not provided
    if max_batch_size is None:
        max_batch_size = config.NUM_PLAYERS
    
    # Create batch processor optimized for TFT
    batch_processor = TorchBasedBatchProcessor(
        max_batch_size=max_batch_size,
        batch_timeout_ms=batch_timeout_ms,
        gpu_memory_fraction=gpu_memory_fraction
    )
    
    # Create agent manager
    agent_manager = EnhancedAgentManager(batch_processor)
    
    # Use provided agent configs or create default setup
    if agent_configs is None:
        agent_configs = _create_default_agent_configs()
    
    # Register all agents with batch processor and setup
    for agent_instance, count in agent_configs:
        if count > 0:  # Only register agents that will be used
            batch_processor.register_agent_instance(type(agent_instance), agent_instance)
    
    # Setup agent configurations
    agent_manager.setup_agents(agent_configs)
    
    return agent_manager, batch_processor


def _create_default_agent_configs(global_buffer=None) -> List[Tuple[Any, int]]:
    """
    Create default agent configuration for testing/examples.
    
    Args:
        global_buffer: Global buffer instance for MuZero agents
        
    Returns:
        List of (agent_instance, count) tuples
    """
    # Create agent instances with default parameters
    muzero_agent = MuZeroAgent(agent_name="MuZeroAgent", global_buffer=global_buffer)
    random_agent = RandomAgent("RandomAgent")
    cultist_agent = CultistAgent()
    divine_agent = DivineAgent()
    
    # Default configuration
    return [
        (muzero_agent, 1),
        (random_agent, 5),
        (cultist_agent, 1),
        (divine_agent, 1)
    ]


def create_custom_agent_setup(agents_and_counts: List[Tuple[Any, int]], **kwargs):
    """
    Convenience function to create setup with custom agents.
    
    Args:
        agents_and_counts: List of (agent_instance, count) tuples
        **kwargs: Additional arguments passed to create_enhanced_setup
        
    Returns:
        Tuple of (agent_manager, batch_processor)
    """
    return create_enhanced_setup(agent_configs=agents_and_counts, **kwargs)


# Usage example function
async def example_usage():
    """Example of how to use the enhanced system"""
    try:
        from tft_set4_gym.tft_simulator import parallel_env
        # Try to import GlobalBuffer (may not exist depending on setup)
        try:
            from Models.global_buffer import GlobalBuffer
        except ImportError:
            GlobalBuffer = None
    except ImportError as e:
        print(f"Warning: Could not import required modules for example: {e}")
        print("Make sure tft_set4_gym is available in your environment")
        return
    
    # Example 1: Use default setup
    print("=== Example 1: Default Setup ===")
    agent_manager, batch_processor = create_enhanced_setup()
    
    # Example 2: Custom agent configuration
    print("\n=== Example 2: Custom Setup ===")
    
    # Create your own global buffer instance (you would replace this with your actual buffer)
    global_buffer = None  # Replace with your actual GlobalBuffer instance
    
    # If GlobalBuffer is available, you could create it like this:
    # if GlobalBuffer is not None:
    #     global_buffer = GlobalBuffer(config.BATCH_SIZE)  # Adjust constructor as needed
    
    # Create custom agents
    my_muzero = MuZeroAgent(agent_name="MyMuZero", global_buffer=global_buffer)
    my_random_1 = RandomAgent("FastRandom")
    my_random_2 = RandomAgent("SlowRandom") 
    my_cultist = CultistAgent()
    
    # Define custom configuration
    custom_agents = [
        (my_muzero, 2),      # 2 MuZero agents
        (my_random_1, 3),    # 3 fast random agents
        (my_random_2, 2),    # 2 slow random agents
        (my_cultist, 1)      # 1 cultist agent
    ]
    
    # Create custom setup
    custom_manager, custom_processor = create_custom_agent_setup(
        agents_and_counts=custom_agents,
        max_batch_size=16,
        batch_timeout_ms=10.0,
        gpu_memory_fraction=0.8
    )
    
    # Example 3: Runtime agent configuration
    print("\n=== Example 3: Runtime Configuration ===")
    
    # You can also modify the setup at runtime
    runtime_manager, runtime_processor = create_enhanced_setup()
    
    # Add a new agent type dynamically
    new_divine_agent = DivineAgent()
    runtime_processor.register_agent_instance(type(new_divine_agent), new_divine_agent)
    runtime_manager.register_agent(new_divine_agent, ["player_8"])  # Add as 9th player if allowed
    
    # Use the first setup for actual game running
    async_env = AsyncGameEnvironment(parallel_env, agent_manager)
    
    # Run multiple games concurrently
    print(f"\nRunning {config.CONCURRENT_GAMES} games...")
    game_tasks = []
    for i in range(config.CONCURRENT_GAMES):
        task = async_env.run_game(f"game_{i}")
        game_tasks.append(task)
    
    # Wait for all games to complete
    results = await asyncio.gather(*game_tasks)
    
    # Print results
    print("\n=== Game Results ===")
    for result in results:
        print(f"Game {result['game_id']} completed in {result['duration']:.2f}s")
        print(f"Final placements: {result['final_placements']}")
    
    # Print performance stats
    stats = agent_manager.get_performance_stats()
    print("\n=== Performance Statistics ===")
    for agent_name, agent_stats in stats.items():
        print(f"{agent_name}: {agent_stats}")


# Example helper functions for common configurations
def create_muzero_vs_random_setup(num_muzero: int = 1, num_random: int = 7, global_buffer=None):
    """Create a setup with MuZero agents vs random agents"""
    muzero_agents = [
        (MuZeroAgent(agent_name=f"MuZero_{i}", global_buffer=global_buffer), 1)
        for i in range(num_muzero)
    ]
    random_agents = [(RandomAgent(f"Random_{i}"), 1) for i in range(num_random)]
    
    all_agents = muzero_agents + random_agents
    return create_custom_agent_setup(all_agents)


def create_buying_agents_setup(global_buffer=None):
    """Create a setup with different buying strategy agents"""
    agents = [
        (CultistAgent(), 2),
        (DivineAgent(), 2),
        (RandomAgent("Random"), 4)
    ]
    return create_custom_agent_setup(agents)


def create_tournament_setup(agent_instances: List[Any], global_buffer=None):
    """Create a tournament setup where each agent gets one slot"""
    agents = [(agent, 1) for agent in agent_instances]
    
    # Pad with random agents if we don't have enough
    while sum(count for _, count in agents) < config.NUM_PLAYERS:
        agents.append((RandomAgent(f"Filler_{len(agents)}"), 1))
    
    return create_custom_agent_setup(agents)


if __name__ == "__main__":
    # Run the example
    asyncio.run(example_usage())
