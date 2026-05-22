"""
Enhanced AI Interface for TFT MuZero Agent Training

This module provides a modernized training interface that replaces Ray with
native async/await patterns and uses the enhanced agent management system.

Key improvements over original AI_interface:
1. Removed Ray dependency - uses native Python async/await
2. Enhanced agent management with proper batching
3. Better GPU utilization and memory management
4. Improved parallel environment handling
5. Modern async-based training loop
6. Better error handling and monitoring

The training routine supports:
- Multiple agent types with flexible configuration
- Parallel environment execution without Ray overhead
- Efficient GPU batching for neural network inference
- Proper model checkpointing and evaluation
- TensorBoard logging and monitoring
"""

import asyncio
import time
import config
import datetime
import os
import copy
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from collections import defaultdict
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Core imports
from global_buffer import GlobalBuffer
from TFTSet4Gym.tft_set4_gym.tft_simulator import parallel_env
from Models.replay_buffer_wrapper import BufferWrapper

# Enhanced agent system imports
from Models.enhanced_agent_interface import (
    create_enhanced_setup, 
    create_custom_agent_setup,
    AsyncGameEnvironment,
    TorchBasedBatchProcessor,
    EnhancedAgentManager
)

# Agent and model imports
from Models.MuZero_torch_agent import EnhancedMuZeroAgent as MuZeroAgent
from Models.MuZero_torch_trainer import Trainer
from Models.Common_agents import CultistAgent, DivineAgent, RandomAgent


@dataclass
class TrainingConfig:
    """Configuration for training parameters"""
    starting_train_step: int = 0
    run_name: str = ""
    save_interval: int = config.CHECKPOINT_STEPS
    evaluation_interval: int = config.CHECKPOINT_STEPS
    concurrent_games: int = config.CONCURRENT_GAMES
    evaluation_games: int = config.EVALUATION_GAMES
    evaluation_concurrent: int = config.EVALUATION_CONCURRENT_GAMES
    max_batch_size: int = 16
    batch_timeout_ms: float = 5.0
    gpu_memory_fraction: float = 0.7


@dataclass 
class GameResult:
    """Container for game results"""
    game_id: str
    placements: Dict[str, int]
    scores: Dict[str, float]
    duration: float
    agent_mapping: Dict[str, type]


class EnhancedGameWorker:
    """
    Async game worker that replaces Ray DataWorker.
    Runs games asynchronously without Ray overhead.
    """
    
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.games_completed = 0
        
    async def run_game(self, agent_manager: EnhancedAgentManager, 
                      return_placements: bool = False) -> GameResult:
        """
        Run a single game with the provided agents
        
        Args:
            agent_manager: Enhanced agent manager with configured agents
            return_placements: Whether to return placement information
            
        Returns:
            GameResult with game outcome data
        """
        try:
            print(f"Worker {self.worker_id}: Starting game...")
            start_time = time.time()
            game_id = f"worker_{self.worker_id}_game_{self.games_completed}"
            
            # Create environment
            print(f"Worker {self.worker_id}: Creating environment...")
            env = parallel_env(rank=self.worker_id)
            observations = env.reset()[0]
            print(f"Worker {self.worker_id}: Environment created, {len(observations)} players")
            
            # Initialize game state
            terminated = {player_id: False for player_id in env.possible_agents}
            rewards = {player_id: 0.0 for player_id in env.possible_agents}
            scores = {player_id: 0.0 for player_id in env.possible_agents}
            
            # Game loop
            step_count = 0
            while not all(terminated.values()):
                step_count += 1
                # print(f"Worker {self.worker_id}: Game step {step_count}")
                
                # Debug observations format
                # print(f"Worker {self.worker_id}: Observation type: {type(observations)}")
                # if observations:
                #     sample_key = list(observations.keys())[0]
                #     sample_obs = observations[sample_key]
                #     print(f"Worker {self.worker_id}: Sample obs type: {type(sample_obs)}, keys: {sample_obs.keys() if hasattr(sample_obs, 'keys') else 'N/A'}")
                
                # Get actions using enhanced batch processing with improved error handling
                # print(f"Worker {self.worker_id}: Getting actions from agent manager...")
                
                try:
                    # Use the enhanced batch processing system with short timeout
                    # Convert rewards to proper type
                    float_rewards = {k: float(v) for k, v in rewards.items()}
                    
                    # Add a shorter timeout for batch processing
                    actions_task = agent_manager.get_actions(
                        observations, 
                        float_rewards,
                        terminated
                    )
                    
                    # Wait with a reasonable timeout (2 seconds)
                    actions = await asyncio.wait_for(actions_task, timeout=2.0)
                    # print(f"Worker {self.worker_id}: Got {len(actions)} actions from batch processing")
                    
                    # Ensure actions are in the correct format (3-element lists)
                    processed_actions = {}
                    for player_id, action in actions.items():
                        if not terminated.get(player_id, True):
                            # Ensure action is a 3-element list/array for TFT
                            if isinstance(action, (list, np.ndarray)) and len(action) >= 3:
                                processed_actions[player_id] = action[:3]  # Take first 3 elements
                            elif hasattr(action, 'tolist') and not isinstance(action, list):
                                # Handle torch tensors or numpy arrays (but not lists)
                                action_list = action.tolist()
                                if isinstance(action_list, list) and len(action_list) >= 3:
                                    processed_actions[player_id] = action_list[:3]
                                else:
                                    processed_actions[player_id] = [0, 0, 0]  # Default action
                            else:
                                processed_actions[player_id] = [0, 0, 0]  # Default action
                        else:
                            processed_actions[player_id] = [0, 0, 0]  # Pass for terminated players
                    
                    actions = processed_actions
                    
                except asyncio.TimeoutError:
                    print(f"Worker {self.worker_id}: Batch processing timed out, using random actions")
                    
                    # Fallback to random actions if batch processing times out
                    import random
                    actions = {}
                    for player_id in observations.keys():
                        if not terminated.get(player_id, True):
                            actions[player_id] = [
                                random.randint(0, 5),   # shop_action
                                random.randint(0, 36),  # shop_position  
                                random.randint(0, 27)   # board_position
                            ]
                        else:
                            actions[player_id] = [0, 0, 0]  # Pass for terminated players
                    
                except Exception as e:
                    print(f"Worker {self.worker_id}: Enhanced batch processing failed: {e}")
                    print(f"Worker {self.worker_id}: Falling back to random actions")
                    
                    # Fallback to random actions if batch processing fails
                    import random
                    actions = {}
                    for player_id in observations.keys():
                        if not terminated.get(player_id, True):
                            actions[player_id] = [
                                random.randint(0, 5),   # shop_action
                                random.randint(0, 36),  # shop_position  
                                random.randint(0, 27)   # board_position
                            ]
                        else:
                            actions[player_id] = [0, 0, 0]  # Pass for terminated players
                
                # print(f"Worker {self.worker_id}: Generated {len(actions)} random actions")
                
                # Step environment
                observations, rewards, terminated, _, info = env.step(actions)
                
                # Update scores for terminated players
                for player in terminated.keys():
                    if terminated[player]:
                        scores[player] = rewards[player]
                
                # Prevent infinite loops
                if step_count > 1000:
                    print(f"Worker {self.worker_id}: Game too long, forcing termination")
                    break
            
            print(f"Worker {self.worker_id}: Game completed after {step_count} steps")
            
            # Calculate final placements
            placements = {}
            if return_placements:
                sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                for i, (player_id, score) in enumerate(sorted_scores):
                    placements[player_id] = i + 1
            
            # Get agent mapping if needed
            agent_mapping = {}
            if return_placements:
                agent_mapping = agent_manager.get_player_agent_mapping()
            
            duration = time.time() - start_time
            self.games_completed += 1
            
            print(f"Worker {self.worker_id}: Game {game_id} finished in {duration:.2f}s")
            
            return GameResult(
                game_id=game_id,
                placements=placements,
                scores=scores,
                duration=duration,
                agent_mapping=agent_mapping
            )
        except Exception as e:
            print(f"Worker {self.worker_id}: Error in run_game: {e}")
            import traceback
            traceback.print_exc()
            raise


class ParallelEnvironmentManager:
    """
    Manages multiple parallel environments without Ray.
    Provides async game execution and result collection.
    """
    
    def __init__(self, num_workers: int):
        self.num_workers = num_workers
        self.workers = [EnhancedGameWorker(i) for i in range(num_workers)]
        self.active_games = set()
        self.should_continue = True  # Control flag for continuous execution
        
    def stop_training(self):
        """Signal the environment manager to stop continuous execution"""
        self.should_continue = False
        
    async def run_games_continuously(self, agent_manager: EnhancedAgentManager,
                                   results_callback: Optional[Callable] = None) -> None:
        """
        Run games continuously in parallel, maintaining a constant pool of active games
        
        This ensures that we always have the maximum number of games running,
        immediately starting a new game when one completes to keep the pipeline full.
        
        Args:
            agent_manager: Agent manager for game execution
            results_callback: Optional callback for when games complete
        """
        print(f"Starting continuous parallel execution with {len(self.workers)} concurrent games")
        
        # Start initial pool of games - one per worker
        active_tasks = {}  # task -> worker_id mapping
        
        # Launch initial games
        for i, worker in enumerate(self.workers):
            task = asyncio.create_task(worker.run_game(agent_manager))
            active_tasks[task] = i
            self.active_games.add(task)
            print(f"Started initial game on worker {i}")
        
        games_completed = 0
        
        # Continuously maintain the game pool
        while self.should_continue:
            if not active_tasks:
                print("No active tasks - restarting game pool")
                # Restart the pool if somehow all games ended
                for i, worker in enumerate(self.workers):
                    task = asyncio.create_task(worker.run_game(agent_manager))
                    active_tasks[task] = i
                    self.active_games.add(task)
                continue
                
            # Wait for ANY game to complete
            try:
                done, pending = await asyncio.wait(
                    active_tasks.keys(), 
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=1.0  # Check periodically for training_active flag
                )
            except asyncio.TimeoutError:
                continue  # Check training_active flag
                
            # Process all completed games
            for completed_task in done:
                try:
                    result = await completed_task
                    games_completed += 1
                    
                    # Get the worker that completed
                    worker_id = active_tasks[completed_task]
                    worker = self.workers[worker_id]
                    
                    print(f"Game {games_completed} completed on worker {worker_id} "
                          f"(duration: {result.duration:.2f}s)")
                    
                    # Call the callback with the result
                    if results_callback:
                        try:
                            await results_callback(result)
                        except Exception as e:
                            print(f"Callback error for game {result.game_id}: {e}")
                    
                    # Clean up completed task
                    del active_tasks[completed_task]
                    self.active_games.discard(completed_task)
                    
                    # IMMEDIATELY start a new game on the same worker to maintain parallelism
                    new_task = asyncio.create_task(worker.run_game(agent_manager))
                    active_tasks[new_task] = worker_id
                    self.active_games.add(new_task)
                    
                    print(f"Started new game on worker {worker_id} "
                          f"(total active: {len(active_tasks)})")
                    
                except Exception as e:
                    # Handle errors gracefully
                    worker_id = active_tasks.get(completed_task, -1)
                    print(f"Game error on worker {worker_id}: {e}")
                    
                    # Clean up failed task
                    if completed_task in active_tasks:
                        del active_tasks[completed_task]
                    self.active_games.discard(completed_task)
                    
                    # Restart the game on this worker after a brief delay
                    if worker_id >= 0 and worker_id < len(self.workers):
                        await asyncio.sleep(0.1)  # Brief pause before restart
                        worker = self.workers[worker_id]
                        new_task = asyncio.create_task(worker.run_game(agent_manager))
                        active_tasks[new_task] = worker_id
                        self.active_games.add(new_task)
                        print(f"Restarted game on worker {worker_id} after error")
        
        print(f"Training stopped. Cleaning up {len(active_tasks)} active games...")
        
        # Clean up remaining tasks when training stops
        for task in list(active_tasks.keys()):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                print(f"Error during cleanup: {e}")
        
        print(f"Continuous game execution ended. Total games completed: {games_completed}")
    
    async def run_evaluation_games(self, agent_manager: EnhancedAgentManager,
                                 num_games: int) -> List[GameResult]:
        """
        Run a specific number of evaluation games
        
        Args:
            agent_manager: Agent manager for evaluation
            num_games: Number of games to run
            
        Returns:
            List of game results
        """
        results = []
        games_per_worker = num_games // self.num_workers
        remaining_games = num_games % self.num_workers
        
        # Create evaluation tasks
        eval_tasks = []
        for i, worker in enumerate(self.workers):
            worker_games = games_per_worker + (1 if i < remaining_games else 0)
            for _ in range(worker_games):
                task = asyncio.create_task(worker.run_game(agent_manager, return_placements=True))
                eval_tasks.append(task)
        
        # Wait for all evaluation games to complete
        if eval_tasks:
            completed_results = await asyncio.gather(*eval_tasks, return_exceptions=True)
            
            # Filter out exceptions and collect results
            for result in completed_results:
                if isinstance(result, GameResult):
                    results.append(result)
                elif isinstance(result, Exception):
                    print(f"Evaluation game error: {result}")
                    import traceback
                    print("Full traceback:")
                    traceback.print_exception(type(result), result, result.__traceback__)
        
        return results


class EnhancedAIInterface:
    """
    Enhanced AI Interface that replaces the original Ray-based training system.
    
    Features:
    - Native async/await instead of Ray
    - Enhanced agent management with proper batching
    - Efficient GPU utilization
    - Better error handling and monitoring
    - Modern training loop with proper checkpointing
    """
    
    def __init__(self, training_config: Optional[TrainingConfig] = None):
        self.config = training_config or TrainingConfig()
        self.trainer = Trainer()
        self.training_step = self.config.starting_train_step
        
        # Initialize components
        self.global_buffer: Optional[GlobalBuffer] = None
        self.agent_manager: Optional[EnhancedAgentManager] = None
        self.env_manager: Optional[ParallelEnvironmentManager] = None
        self.summary_writer: Optional[SummaryWriter] = None
        
        # Training state
        self.training_active = False
        self.games_completed = 0
        
        # Model and weights
        self.base_agent: Optional[MuZeroAgent] = None
        self.current_weights: Optional[Dict] = None
        
    def _setup_logging(self, run_name: str) -> SummaryWriter:
        """Setup TensorBoard logging"""
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = f'logs/{run_name}{current_time}'
        return SummaryWriter(log_dir)
    
    def _create_agents(self) -> Tuple[List[Tuple[Any, int]], MuZeroAgent]:
        """
        Create agent configuration for training
        
        Returns:
            Tuple of (agent_configs, base_muzero_agent)
        """
        # Create global buffer
        self.global_buffer = GlobalBuffer(config.BATCH_SIZE)
        
        # Create base MuZero agent
        base_agent = MuZeroAgent(
            action_size=3,
            action_limits=[7, 37, 10],
            obs_size=config.OBSERVATION_SIZE,
            simulations=config.NUM_SIMULATIONS,
            global_buffer=self.global_buffer
        )
        
        # Load existing weights if resuming training
        if self.training_step > 0:
            checkpoint_path = f'./checkpoint/checkpoint_{self.training_step}'
            if os.path.isfile(checkpoint_path):
                print(f"Loading checkpoint from {checkpoint_path}")
                base_agent.model.load_state_dict(torch.load(checkpoint_path))
            else:
                print(f"Checkpoint not found at {checkpoint_path}, starting from scratch")
        
        self.current_weights = base_agent.get_weights()
        
        # Create training agents
        training_muzero = MuZeroAgent(
            action_size=3,
            action_limits=[7, 37, 10], 
            obs_size=config.OBSERVATION_SIZE,
            simulations=config.NUM_SIMULATIONS,
            global_buffer=self.global_buffer,
            weights=copy.deepcopy(self.current_weights)
        )
        
        # Create other agent types
        random_agent = RandomAgent("RandomTraining")
        cultist_agent = CultistAgent()
        divine_agent = DivineAgent()
        
        # Define training configuration
        agent_configs = [
            (training_muzero, 2),  # 2 MuZero agents
            (random_agent, 4),     # 4 random agents
            (cultist_agent, 1),    # 1 cultist agent
            (divine_agent, 1)      # 1 divine agent
        ]
        
        return agent_configs, base_agent
    
    async def _game_completion_callback(self, result: GameResult):
        """Handle completed games and trigger training if buffer is ready"""
        self.games_completed += 1
        
        # Check if we have enough data to train
        if hasattr(self.global_buffer, 'available_gameplay_batch'):
            if self.global_buffer.available_gameplay_batch():
                await self._perform_training_step()
    
    async def _perform_training_step(self):
        """Perform a single training step"""
        if not self.global_buffer.available_gameplay_batch():
            return
            
        # Get training batch
        training_batch = self.global_buffer.read_gameplay_batch()
        
        # Get combat batch if available
        combat_batch = []
        if hasattr(self.global_buffer, 'available_combat_batch'):
            if self.global_buffer.available_combat_batch():
                combat_batch = self.global_buffer.read_combat_batch()
        
        # Perform training step
        self.trainer.train_network(
            batch=training_batch,
            combats=combat_batch,
            agent=self.base_agent.model,
            train_step=self.training_step,
            summary_writer=self.summary_writer
        )
        
        self.training_step += 1
        
        # Check for evaluation and checkpointing
        if self.training_step % self.config.evaluation_interval == 0:
            await self._run_evaluation()
            
        if self.training_step % self.config.save_interval == 0:
            self._save_checkpoint()
    
    async def _run_evaluation(self):
        """Run evaluation games and update model if improved"""
        print(f"\nRunning evaluation at step {self.training_step}")
        
        # Create evaluation agents
        eval_base = MuZeroAgent(
            action_size=3,
            action_limits=[7, 37, 10],
            obs_size=config.OBSERVATION_SIZE, 
            simulations=config.NUM_SIMULATIONS,
            global_buffer=self.global_buffer,
            weights=copy.deepcopy(self.base_agent.get_weights())
        )
        
        eval_old = MuZeroAgent(
            action_size=3,
            action_limits=[7, 37, 10],
            obs_size=config.OBSERVATION_SIZE,
            simulations=config.NUM_SIMULATIONS, 
            global_buffer=self.global_buffer,
            weights=copy.deepcopy(self.current_weights)
        )
        
        random_agent = RandomAgent("EvalRandom")
        cultist_agent = CultistAgent()
        divine_agent = DivineAgent()
        
        # Setup evaluation configuration
        eval_configs = [
            (eval_base, 1),     # New model
            (eval_old, 1),      # Old model
            (random_agent, 4),  # Random agents
            (cultist_agent, 1), # Cultist agent
            (divine_agent, 1)   # Divine agent
        ]
        
        # Create evaluation environment
        eval_manager, _ = create_custom_agent_setup(
            eval_configs,
            max_batch_size=self.config.max_batch_size,
            batch_timeout_ms=self.config.batch_timeout_ms,
            gpu_memory_fraction=self.config.gpu_memory_fraction
        )
        
        # Run evaluation games
        eval_env_manager = ParallelEnvironmentManager(self.config.evaluation_concurrent)
        eval_results = await eval_env_manager.run_evaluation_games(
            eval_manager, 
            self.config.evaluation_games
        )
        
        # Calculate average placements
        base_placements = []
        old_placements = []
        
        for result in eval_results:
            agent_mapping = result.agent_mapping
            for player_id, placement in result.placements.items():
                agent_type = agent_mapping.get(player_id)
                if agent_type == type(eval_base):
                    base_placements.append(placement)
                elif agent_type == type(eval_old):
                    old_placements.append(placement)
        
        # Calculate means
        base_mean = np.mean(base_placements) if base_placements else 8.0
        old_mean = np.mean(old_placements) if old_placements else 8.0
        
        # Log evaluation results
        if self.summary_writer:
            self.summary_writer.add_scalar('evaluation/new_model', base_mean, self.training_step)
            self.summary_writer.add_scalar('evaluation/old_model', old_mean, self.training_step)
        
        print(f"Evaluation results - New: {base_mean:.2f}, Old: {old_mean:.2f}")
        
        # Update model if improved
        if base_mean < old_mean:
            print("Model improved! Updating weights and saving.")
            if self.base_agent is not None:
                self.current_weights = copy.deepcopy(self.base_agent.get_weights())
                self._save_checkpoint()
            
            # Clear buffers
            if self.global_buffer is not None and hasattr(self.global_buffer, 'clear_gameplay_buffer'):
                self.global_buffer.clear_gameplay_buffer()
            if self.global_buffer is not None and hasattr(self.global_buffer, 'clear_combat_buffer'):
                self.global_buffer.clear_combat_buffer()
                
            # Update training agents with new weights
            await self._update_training_agents()
    
    async def _update_training_agents(self):
        """Update training agents with new weights"""
        # This would update the agents in the agent manager
        # Implementation depends on the specific agent manager API
        pass
    
    def _save_checkpoint(self):
        """Save model checkpoint"""
        checkpoint_dir = './checkpoint'
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = f'{checkpoint_dir}/checkpoint_{self.training_step}'
        if self.base_agent is not None and hasattr(self.base_agent, 'model'):
            torch.save(self.base_agent.model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at step {self.training_step}")
        else:
            print(f"Warning: Cannot save checkpoint - base_agent not properly initialized")
    
    async def train_torch_model(self, starting_train_step: int = 0, run_name: str = ""):
        """
        Main training method that replaces the original Ray-based training
        
        Args:
            starting_train_step: Step to resume training from
            run_name: Name prefix for logging
        """
        print("=== Enhanced TFT MuZero Training ===")
        print("Using native async/await instead of Ray")
        print(f"GPU available: {torch.cuda.is_available()}")
        
        # Update config
        self.config.starting_train_step = starting_train_step
        self.config.run_name = run_name
        self.training_step = starting_train_step
        
        # Setup logging
        self.summary_writer = self._setup_logging(run_name)
        
        # Create agents
        agent_configs, self.base_agent = self._create_agents()
        
        # Setup enhanced agent management
        self.agent_manager, batch_processor = create_custom_agent_setup(
            agent_configs,
            max_batch_size=self.config.max_batch_size,
            batch_timeout_ms=self.config.batch_timeout_ms,
            gpu_memory_fraction=self.config.gpu_memory_fraction
        )
        
        # Setup parallel environment manager
        self.env_manager = ParallelEnvironmentManager(self.config.concurrent_games)
        
        print(f"Training setup complete:")
        print(f"  - Concurrent games: {self.config.concurrent_games}")
        print(f"  - Batch size: {self.config.max_batch_size}")
        print(f"  - GPU memory fraction: {self.config.gpu_memory_fraction}")
        print(f"  - Starting from step: {self.training_step}")
        
        # Start training
        self.training_active = True
        
        try:
            # Run continuous training with step limit
            print("Starting continuous game collection and training...")
            
            # Create a task for continuous game execution
            training_task = asyncio.create_task(
                self.env_manager.run_games_continuously(
                    self.agent_manager,
                    self._game_completion_callback
                )
            )
            
            # Monitor training progress and stop when max steps reached
            while self.training_active and self.training_step < 1000000:  # Default max steps
                await asyncio.sleep(1.0)  # Check every second
                
                # Optional: Add periodic logging
                if self.training_step % 100 == 0:
                    print(f"Training step: {self.training_step}, Games completed: {self.games_completed}")
            
            # Stop training if max steps reached
            if self.training_step >= 1000000:
                print(f"Maximum training steps ({1000000}) reached. Stopping training.")
                self.training_active = False
                self.env_manager.stop_training()
            
            # Wait for training task to complete
            await training_task
            
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        except Exception as e:
            print(f"Training error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.training_active = False
            self.env_manager.stop_training()  # Signal environment manager to stop
            if self.summary_writer:
                self.summary_writer.close()
            print("Training stopped")
    
    def collect_dummy_data(self):
        """Test method for simulator performance (no AI)"""
        print("Running dummy data collection test...")
        
        async def run_dummy():
            env = parallel_env()
            while True:
                observations = env.reset()[0]
                terminated = {player_id: False for player_id in env.possible_agents}
                start_time = time.time_ns()
                
                while not all(terminated.values()):
                    actions = {
                        agent: env.action_space(agent).sample()
                        for agent in env.agents
                        if (agent in terminated and not terminated[agent])
                    }
                    observations, rewards, terminated, truncated, info = env.step(actions)
                
                duration = (time.time_ns() - start_time) / 1e9
                print(f"Dummy game completed in {duration:.2f} seconds")
        
        asyncio.run(run_dummy())
    
    async def run_single_evaluation(self, num_games: int = 8) -> List[GameResult]:
        """
        Run a single evaluation session
        
        Args:
            num_games: Number of games to run
            
        Returns:
            List of game results
        """
        try:
            print(f"run_single_evaluation: Starting with {num_games} games")
            
            # Create default agent setup if not already created
            if not self.agent_manager:
                print("Creating agent setup...")
                agent_configs, self.base_agent = self._create_agents()
                self.agent_manager, _ = create_custom_agent_setup(agent_configs)
                print("Agent setup complete")
            
            if not self.env_manager:
                print("Creating environment manager...")
                self.env_manager = ParallelEnvironmentManager(self.config.concurrent_games)
                print("Environment manager created")
            
            # Run evaluation
            print("Starting evaluation games...")
            results = await self.env_manager.run_evaluation_games(self.agent_manager, num_games)
            print(f"Evaluation games completed, got {len(results)} results")
            
            # Print results
            print(f"\nEvaluation Results ({len(results)} games):")
            for result in results:
                print(f"Game {result.game_id}: Duration {result.duration:.2f}s")
                if result.placements:
                    for player, placement in result.placements.items():
                        agent_type = result.agent_mapping.get(player, "Unknown")
                        if isinstance(agent_type, type):
                            agent_name = agent_type.__name__
                        else:
                            agent_name = str(agent_type)
                        print(f"  {player} ({agent_name}): {placement}")
            
            return results
        except Exception as e:
            print(f"Error in run_single_evaluation: {e}")
            import traceback
            traceback.print_exc()
            raise


# Factory function for backward compatibility
def create_ai_interface(training_config: Optional[TrainingConfig] = None) -> EnhancedAIInterface:
    """Create an enhanced AI interface instance"""
    return EnhancedAIInterface(training_config)


# Convenience function for quick training setup
async def quick_training_setup():
    """Quick setup for testing the enhanced training system"""
    try:
        print("Step 1: Creating config...")
        config = TrainingConfig(
            concurrent_games=1,  # Reduce concurrency
            evaluation_games=1,   # Test with just 1 game
            evaluation_concurrent=1,
            max_batch_size=8
        )
        
        print("Step 2: Creating AI interface...")
        ai_interface = EnhancedAIInterface(config)
        
        print("Step 3: Running single evaluation...")
        # Run a quick evaluation to test the system
        print("Running quick evaluation test...")
        results = await ai_interface.run_single_evaluation(1)  # Just 1 game
        print(f"Test completed with {len(results)} games")
        
        return ai_interface
    except Exception as e:
        print(f"Error in quick_training_setup: {e}")
        import traceback
        traceback.print_exc()
        raise



class AIInterface:
    """
    Backward compatibility class that wraps EnhancedAIInterface
    """
    
    def __init__(self):
        self.enhanced = EnhancedAIInterface()
    
    def train_torch_model(self, starting_train_step=0, run_name=""):
        """Train model using the enhanced async interface"""
        return asyncio.run(self.enhanced.train_torch_model(starting_train_step, run_name))
    
    def collect_dummy_data(self):
        """Test method for simulator performance"""
        return self.enhanced.collect_dummy_data()
    
    def testEnv(self):
        """PettingZoo API tests for the simulator"""
        from pettingzoo.test import parallel_api_test, api_test
        from TFTSet4Gym.tft_set4_gym.tft_simulator import env as tft_env
        
        print("Running PettingZoo API tests...")
        
        # Test raw environment
        raw_env = tft_env(rank=0)
        api_test(raw_env, num_cycles=1000)
        
        # Test parallel environment  
        local_env = parallel_env()
        parallel_api_test(local_env, num_cycles=1000)
        
        print("PettingZoo API tests completed successfully!")


# Factory functions for easy usage
def create_training_interface(config: Optional[TrainingConfig] = None) -> EnhancedAIInterface:
    """Create a modern training interface"""
    return EnhancedAIInterface(config)


def create_legacy_interface() -> AIInterface:
    """Create a legacy-compatible interface"""
    return AIInterface()


# Convenience function for quick evaluation
async def run_quick_evaluation(num_games: int = 8, concurrent_games: int = 2):
    """
    Run a quick evaluation with default agents
    
    Args:
        num_games: Total number of games to run
        concurrent_games: Number of games to run concurrently
    """
    config = TrainingConfig(
        concurrent_games=concurrent_games,
        evaluation_games=num_games,
        evaluation_concurrent=concurrent_games,
        max_batch_size=8
    )
    
    interface = EnhancedAIInterface(config)
    results = await interface.run_single_evaluation(num_games)
    
    # Print summary
    print(f"\n=== Evaluation Summary ({len(results)} games) ===")
    
    # Calculate agent performance
    agent_stats = defaultdict(list)
    for result in results:
        for player_id, placement in result.placements.items():
            agent_type = result.agent_mapping.get(player_id)
            if agent_type:
                agent_name = agent_type.__name__ if hasattr(agent_type, '__name__') else str(agent_type)
                agent_stats[agent_name].append(placement)
    
    # Print average placements
    for agent_name, placements in agent_stats.items():
        avg_placement = np.mean(placements)
        print(f"{agent_name}: {avg_placement:.2f} avg placement ({len(placements)} games)")
    
    return results


if __name__ == "__main__":
    # Quick test of the enhanced system
    async def main_with_timeout():
        try:
            # Add timeout to prevent hanging
            await asyncio.wait_for(quick_training_setup(), timeout=60.0)
            print("Script completed successfully!")
        except asyncio.TimeoutError:
            print("Script timed out - likely hanging in async operation")
        except Exception as e:
            print(f"Error in main: {e}")
            import traceback
            traceback.print_exc()
    
    try:
        asyncio.run(main_with_timeout())
    except Exception as e:
        print(f"Critical error: {e}")
        import traceback
        traceback.print_exc()

