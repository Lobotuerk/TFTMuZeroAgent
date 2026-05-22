"""
Enhanced TFT MuZero Agent Training Script

This is the main entry point for training and evaluating TFT AI agents using
the modernized async/await training system with enhanced batch processing.

Features:
- Enhanced async training with improved GPU utilization
- Modern batch processing without Ray dependency
- Multiple training modes and agent configurations
- Comprehensive evaluation and testing capabilities
- Improved error handling and monitoring
"""

import asyncio
import argparse
import sys
import time
from typing import Optional

# Core imports
import config
import AI_interface

# Enhanced system imports
from AI_interface import (
    EnhancedAIInterface, 
    TrainingConfig, 
    create_training_interface,
    create_legacy_interface,
    run_quick_evaluation
)

# Model imports for debugging
from Models.MuZero_torch_agent import MuZeroNetwork as TFTNetwork


def create_training_config(args) -> TrainingConfig:
    """Create training configuration from command line arguments"""
    training_config = TrainingConfig()
    
    # Set configuration from args
    training_config.concurrent_games = getattr(args, 'concurrent_games', config.CONCURRENT_GAMES)
    training_config.evaluation_interval = getattr(args, 'eval_interval', config.CHECKPOINT_STEPS)
    training_config.evaluation_games = getattr(args, 'eval_games', config.EVALUATION_GAMES)
    training_config.evaluation_concurrent = getattr(args, 'eval_concurrent', config.EVALUATION_CONCURRENT_GAMES)
    training_config.max_batch_size = getattr(args, 'batch_size', 16)
    training_config.save_interval = getattr(args, 'checkpoint_interval', config.CHECKPOINT_STEPS)
    training_config.starting_train_step = getattr(args, 'starting_episode', 0)
    training_config.run_name = getattr(args, 'run_name', "")
    
    return training_config


async def enhanced_training_mode(args):
    """Run enhanced async training with modern features"""
    print("=== Enhanced Training Mode ===")
    print("Using modern async/await training system with enhanced batch processing")
    
    try:
        # Create training configuration
        training_config = create_training_config(args)
        
        # Create enhanced interface
        interface = EnhancedAIInterface(training_config)
        
        # Run training
        await interface.train_torch_model(
            starting_train_step=args.starting_episode,
            run_name=getattr(args, 'run_name', "")
        )
        
    except Exception as e:
        print(f"Error in enhanced training: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


async def evaluation_mode(args):
    """Run evaluation games with specified agents"""
    print("=== Evaluation Mode ===")
    print(f"Running {args.eval_games} evaluation games")
    
    try:
        # Use quick evaluation for testing
        if args.quick:
            results = await run_quick_evaluation(
                num_games=args.eval_games,
                concurrent_games=args.eval_concurrent
            )
        else:
            # Create interface for more detailed evaluation
            training_config = create_training_config(args)
            interface = EnhancedAIInterface(training_config)
            results = await interface.run_single_evaluation(args.eval_games)
        
        print(f"Evaluation completed with {len(results)} games")
        return True
        
    except Exception as e:
        print(f"Error in evaluation mode: {e}")
        import traceback
        traceback.print_exc()
        return False


def legacy_training_mode(args):
    """Run legacy training mode for backward compatibility"""
    print("=== Legacy Training Mode ===")
    print("Using legacy interface (backward compatibility)")
    
    try:
        interface = AI_interface.AIInterface()
        interface.train_torch_model(starting_train_step=args.starting_episode)
        return True
    except Exception as e:
        print(f"Error in legacy training: {e}")
        import traceback
        traceback.print_exc()
        return False


def simulator_testing_mode(args):
    """Run simulator and environment tests"""
    print("=== Simulator Testing Mode ===")
    
    try:
        interface = AI_interface.AIInterface()
        
        if args.test_env:
            print("Running PettingZoo API tests...")
            interface.testEnv()
        
        if args.test_performance:
            print("Running simulator performance tests...")
            interface.collect_dummy_data()
        
        return True
        
    except Exception as e:
        print(f"Error in simulator testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def debug_mode(args):
    """Run debug and development utilities"""
    print("=== Debug Mode ===")
    
    if args.debug_network:
        print("Debugging neural network architecture...")
        temp = TFTNetwork()
        print(f"Network parameters: {sum(p.numel() for p in temp.parameters())}")
        for name in temp.state_dict():
            print(f"Layer: {name}")
        print(temp)
    
    if args.debug_agents:
        print("Testing agent creation and basic functionality...")
        # Add agent debugging logic here
        pass
    
    return True


async def async_main():
    """Main async function that handles all training modes"""
    # Handle unit tests first
    if config.RUN_UNIT_TEST:
        print("Unit tests are configured to run, but no test suite is available.")
        print("Skipping to main functionality...")

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Enhanced TFT MuZero Agent Training System',
        epilog='For more information, visit https://github.com/silverlight6/TFTMuZeroAgent',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Core training arguments
    parser.add_argument('--starting_episode', '-se', dest='starting_episode', type=int, default=0,
                        help='Episode number to start training (for checkpoint loading)')
    parser.add_argument('--run_name', '-rn', dest='run_name', type=str, default="",
                        help='Name for this training run (for logging)')
    
    # Training mode selection
    parser.add_argument('--mode', '-m', choices=['enhanced', 'legacy', 'eval', 'test', 'debug'], 
                        default='enhanced', help='Training/execution mode')
    
    # Enhanced training configuration
    parser.add_argument('--concurrent_games', '-cg', type=int, default=config.CONCURRENT_GAMES,
                        help='Number of concurrent games during training')
    parser.add_argument('--max_steps', '-ms', type=int, default=1000000,
                        help='Maximum training steps')
    parser.add_argument('--eval_interval', '-ei', type=int, default=config.CHECKPOINT_STEPS,
                        help='Steps between evaluations')
    parser.add_argument('--eval_games', '-eg', type=int, default=config.EVALUATION_GAMES,
                        help='Number of evaluation games')
    parser.add_argument('--eval_concurrent', '-ec', type=int, default=config.EVALUATION_CONCURRENT_GAMES,
                        help='Concurrent evaluation games')
    parser.add_argument('--batch_size', '-bs', type=int, default=8,
                        help='Maximum batch size for neural network inference')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--checkpoint_interval', '-ci', type=int, default=config.CHECKPOINT_STEPS,
                        help='Steps between checkpoints')
    
    # Evaluation mode options
    parser.add_argument('--quick', '-q', action='store_true',
                        help='Use quick evaluation mode')
    
    # Testing mode options
    parser.add_argument('--test_env', action='store_true',
                        help='Run PettingZoo environment tests')
    parser.add_argument('--test_performance', action='store_true',
                        help='Run simulator performance tests')
    
    # Debug mode options
    parser.add_argument('--debug_network', action='store_true',
                        help='Debug neural network architecture')
    parser.add_argument('--debug_agents', action='store_true',
                        help='Debug agent creation and functionality')
    
    args = parser.parse_args()
    
    # Print startup information
    print("=" * 60)
    print("TFT MuZero Agent Training System")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Starting episode: {args.starting_episode}")
    
    # Execute based on mode
    success = False
    
    try:
        if args.mode == 'enhanced':
            success = await enhanced_training_mode(args)
        elif args.mode == 'legacy':
            success = legacy_training_mode(args)
        elif args.mode == 'eval':
            success = await evaluation_mode(args)
        elif args.mode == 'test':
            success = simulator_testing_mode(args)
        elif args.mode == 'debug':
            success = debug_mode(args)
        else:
            print(f"Unknown mode: {args.mode}")
            success = False
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        success = True  # Not an error
    except Exception as e:
        print(f"Unexpected error in main: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    # Print completion status
    if success:
        print("\n" + "=" * 60)
        print("Training/Execution completed successfully!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Training/Execution failed!")
        print("=" * 60)
        sys.exit(1)


def main():
    """Synchronous main function - entry point"""
    try:
        # Run the async main function
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Critical error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
