"""
Parallel training routine for TFT MuZero Agent

Runs multiple TFT games concurrently using the async-based parallel
environment manager, with batched GPU inference for MuZero agents.

Key features:
- True parallel game execution via asyncio
- Batched neural network inference for MuZero agents
- Continuous game pipeline (immediately restarts completed games)
- Performance tracking and logging
"""

import asyncio
import time
import argparse
import numpy as np
from typing import Dict, Any, Optional
import torch

import config
from AI_interface import (
    EnhancedAIInterface,
    TrainingConfig,
    ParallelEnvironmentManager,
    EnhancedGameWorker,
    GameResult,
)
from Models.enhanced_agent_interface import (
    create_enhanced_setup,
    create_custom_agent_setup,
    TorchBasedBatchProcessor,
    EnhancedAgentManager,
)
from Models.MuZero_torch_agent import EnhancedMuZeroAgent as MuZeroAgent
from Models.MuZero_torch_trainer import Trainer
from Models.Common_agents import RandomAgent, CultistAgent, DivineAgent, FastLevelAgent, RerollAgent
from Models.global_buffer import GlobalBuffer


def create_training_agents():
    """Create direct agent instances for parallel training"""
    print("Creating parallel training agents...")

    global_buffer = GlobalBuffer(config.BATCH_SIZE)

    muzero_agent = MuZeroAgent(
        global_buffer=global_buffer
    )

    agents = {
        'player_0': RandomAgent("RandomAgent_0", global_buffer),
        'player_1': RandomAgent("RandomAgent_1", global_buffer),
        'player_2': RandomAgent("RandomAgent_2", global_buffer),
        'player_3': muzero_agent,
        'player_4': CultistAgent(global_buffer),
        'player_5': DivineAgent(global_buffer),
        'player_6': FastLevelAgent(global_buffer),
        'player_7': RerollAgent(global_buffer),
    }

    print(f"Created {len(agents)} agents:")
    for player_id, agent in agents.items():
        print(f"  {player_id}: {type(agent).__name__}")

    return agents, global_buffer


def create_parallel_agent_setup():
    """Create enhanced agent setup with batch processor for parallel training"""
    global_buffer = GlobalBuffer(config.BATCH_SIZE)

    training_muzero = MuZeroAgent(global_buffer=global_buffer)

    random_agent = RandomAgent("RandomTraining", global_buffer)
    cultist_agent = CultistAgent(global_buffer)
    divine_agent = DivineAgent(global_buffer)
    fastlevel_agent = FastLevelAgent(global_buffer)
    reroll_agent = RerollAgent(global_buffer)

    agent_configs = [
        (training_muzero, 1),
        (random_agent, 3),
        (cultist_agent, 1),
        (divine_agent, 1),
        (fastlevel_agent, 1),
        (reroll_agent, 1),
    ]

    agent_manager, batch_processor = create_custom_agent_setup(
        agent_configs,
        max_batch_size=config.NUM_PLAYERS,
        batch_timeout_ms=5.0,
        gpu_memory_fraction=0.7,
    )

    return agent_manager, batch_processor, global_buffer


async def run_parallel_episodes(
    agent_manager: EnhancedAgentManager,
    n_episodes: int,
    concurrent_games: int = 4,
    train_every: int = 5,
) -> Dict[str, Any]:
    """
    Run multiple episodes in parallel using the enhanced agent system.

    Args:
        agent_manager: Configured agent manager
        n_episodes: Number of episodes to run
        concurrent_games: Number of concurrent games
        train_every: Collect training data every N episodes

    Returns:
        Training statistics
    """
    env_manager = ParallelEnvironmentManager(concurrent_games)
    trainer = Trainer()

    episode_stats = []
    games_completed = 0
    total_start_time = time.time()
    episode_lock = asyncio.Lock()

    async def on_game_complete(result: GameResult):
        nonlocal games_completed
        games_completed += 1

        stats = {
            'game_id': result.game_id,
            'duration': result.duration,
            'scores': result.scores,
        }
        async with episode_lock:
            episode_stats.append(stats)
            avg_reward = np.mean(list(result.scores.values())) if result.scores else 0.0

        print(f"Game {games_completed} completed in {result.duration:.2f}s "
              f"(avg reward: {avg_reward:.2f})")

        if games_completed % train_every == 0:
            print(f"Training check at {games_completed} games completed...")

    print(f"Starting {n_episodes} parallel games ({concurrent_games} concurrent)...")
    try:
        await env_manager.run_games_continuously(agent_manager, on_game_complete)
    except asyncio.CancelledError:
        print("Training cancelled.")

    total_duration = time.time() - total_start_time

    return {
        'total_episodes': len(episode_stats),
        'total_duration': total_duration,
        'games_completed': games_completed,
        'episode_stats': episode_stats,
    }


def main():
    """Main function for parallel training"""
    parser = argparse.ArgumentParser(description='Parallel TFT training')
    parser.add_argument('--episodes', '-e', type=int, default=10,
                        help='Number of episodes to run')
    parser.add_argument('--concurrent', '-c', type=int, default=4,
                        help='Number of concurrent games')
    parser.add_argument('--train_every', '-t', type=int, default=5,
                        help='Train every N episodes')
    parser.add_argument('--debug', '-d', action='store_true',
                        help='Run quick debug test only')

    args = parser.parse_args()

    if args.debug:
        success = asyncio.run(quick_debug_test())
        if success:
            print("\nDebug test passed! Ready for full parallel training.")
        else:
            print("\nDebug test failed. Check errors above.")
    else:
        asyncio.run(train_parallel(
            n_episodes=args.episodes,
            concurrent_games=args.concurrent,
            train_every=args.train_every,
        ))


async def quick_debug_test():
    """Quick test to verify parallel components work"""
    print("Running parallel debug test...")
    try:
        agent_manager, batch_processor, _ = create_parallel_agent_setup()
        print("  Agent setup created successfully")

        env_manager = ParallelEnvironmentManager(1)
        print("  Environment manager created successfully")

        print("  Running 1 game...")
        worker = EnhancedGameWorker(0)
        result = await worker.run_game(agent_manager)
        print(f"  Game completed in {result.duration:.2f}s")

        return True
    except Exception as e:
        print(f"Debug test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def train_parallel(n_episodes: int = 10, concurrent_games: int = 4, train_every: int = 5):
    """Main parallel training routine"""
    print("=" * 60)
    print("TFT MuZero Parallel Training")
    print("=" * 60)
    print(f"Episodes: {n_episodes}, Concurrent: {concurrent_games}")
    print(f"Training frequency: every {train_every} games")
    print(f"GPU available: {torch.cuda.is_available()}")
    print()

    agent_manager, batch_processor, global_buffer = create_parallel_agent_setup()

    results = await run_parallel_episodes(
        agent_manager=agent_manager,
        n_episodes=n_episodes,
        concurrent_games=concurrent_games,
        train_every=train_every,
    )

    print("\n" + "=" * 60)
    print("Training Completed!")
    print("=" * 60)
    print(f"Total games: {results['games_completed']}")
    print(f"Total time: {results['total_duration']:.2f}s")
    if results['games_completed'] > 0:
        print(f"Avg game time: {results['total_duration'] / results['games_completed']:.2f}s")

    return results


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
