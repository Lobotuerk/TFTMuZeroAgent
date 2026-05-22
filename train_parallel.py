#!/usr/bin/env python3
"""
Parallel training entry point for TFT MuZero Agent.

Demonstrates the new parallel training architecture with:
- Batched GPU inference across multiple environments
- Concurrent game execution via async/await
- Enhanced MCTS with PUCT integration
- Efficient agent management and batch processing

Usage:
    PYTHONPATH=MonteCarloTreeSearch python3 train_parallel.py
"""

import asyncio
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import config
from Models.MuZero_torch_agent import MuZeroAgent
from Models.enhanced_agent_interface import (
    TorchBasedBatchProcessor,
    EnhancedAgentManager,
    AsyncGameEnvironment,
    create_enhanced_setup,
)
from Models.Common_agents import RandomAgent


async def train_loop(
    num_episodes: int = 10,
    concurrent_games: int = 4,
    batch_timeout_ms: float = 5.0,
):
    """Main training loop with parallel environments and batched GPU inference."""

    # Create the batch processor with PyTorch-optimized batching
    batch_processor = TorchBasedBatchProcessor(
        max_batch_size=config.NUM_PLAYERS,
        batch_timeout_ms=batch_timeout_ms,
        gpu_memory_fraction=0.7,
    )

    # Create agent manager
    agent_manager = EnhancedAgentManager(batch_processor)

    # Register agents
    muzero_agent = MuZeroAgent(
        agent_name="MuZeroAgent",
        global_buffer=None,
    )
    batch_processor.register_agent_instance(type(muzero_agent), muzero_agent)
    agent_manager.register_agent(muzero_agent, [f"player_{i}" for i in range(1)])

    random_agent = RandomAgent("RandomAgent")
    agent_manager.register_agent(random_agent, [f"player_{i}" for i in range(1, 8)])

    # Run concurrent games
    try:
        from TFTSet4Gym.tft_set4_gym.tft_simulator import parallel_env
    except ImportError:
        print("TFTSet4Gym not available. Install the submodule and rebuild.")
        return

    async_env = AsyncGameEnvironment(parallel_env, agent_manager)

    print(f"Starting {num_episodes} episodes with {concurrent_games} concurrent games...")

    for episode in range(num_episodes):
        start = time.time()
        game_tasks = [
            async_env.run_game(f"ep{episode}_game{g}")
            for g in range(concurrent_games)
        ]
        results = await asyncio.gather(*game_tasks)

        duration = time.time() - start
        avg_score = sum(
            max(r["scores"].values()) if r["scores"] else 0 for r in results
        ) / len(results)

        print(f"Episode {episode}: {duration:.2f}s | avg_top_score={avg_score:.1f}")

        stats = agent_manager.get_performance_stats()
        for name, s in stats.items():
            print(f"  {name}: avg_inference={s['avg_inference_time']*1000:.1f}ms")

    print("Training complete.")


if __name__ == "__main__":
    asyncio.run(
        train_loop(
            num_episodes=int(os.environ.get("NUM_EPISODES", "10")),
            concurrent_games=int(os.environ.get("CONCURRENT_GAMES", str(config.CONCURRENT_GAMES))),
        )
    )
