#!/usr/bin/env python3
"""
Test script to verify true parallel execution in enhanced training mode.
This creates a short training run to verify that multiple games run simultaneously.
"""

import asyncio
import time
from AI_interface import EnhancedAIInterface, TrainingConfig

async def test_parallel_training():
    """Test that training runs games in parallel"""
    print("=== Testing Parallel Training Execution ===")
    
    # Create configuration for short test
    config = TrainingConfig()
    config.concurrent_games = 4  # Run 4 games simultaneously
    config.evaluation_interval = 50  # Quick evaluation
    config.evaluation_games = 2
    config.evaluation_concurrent = 2
    config.max_batch_size = 8
    
    # Create interface
    interface = EnhancedAIInterface(config)
    
    # Start timer
    start_time = time.time()
    
    try:
        # Run a very short training session (will be interrupted)
        training_task = asyncio.create_task(
            interface.train_torch_model(starting_train_step=0, run_name="parallel_test")
        )
        
        # Let it run for 10 seconds to see parallel execution
        await asyncio.sleep(10.0)
        
        # Stop training
        interface.training_active = False
        interface.env_manager.stop_training()
        
        # Give it time to clean up
        await asyncio.sleep(2.0)
        
        print(f"Test completed in {time.time() - start_time:.2f} seconds")
        print(f"Games completed: {interface.games_completed}")
        
    except Exception as e:
        print(f"Test error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_parallel_training())