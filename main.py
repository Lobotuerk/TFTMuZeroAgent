"""
Enhanced TFT MuZero Agent Training Script

Main entry point for training and evaluating TFT AI agents using
the TrainingOrchestrator with the explicit RL lifecycle:

    Collect -> Train -> Sync -> Evaluate
"""

import asyncio
import argparse
import sys
import time
from typing import Optional

import config
from training_orchestrator import (
    TrainingOrchestrator,
    TrainingConfig,
    create_orchestrator,
    quick_evaluation,
)

from Models.MuZero_torch_agent import MuZeroNetwork as TFTNetwork


def _build_config(args) -> TrainingConfig:
    cfg = TrainingConfig()
    cfg.concurrent_games = getattr(args, "concurrent_games", config.CONCURRENT_GAMES)
    cfg.evaluation_interval = getattr(args, "eval_interval", config.CHECKPOINT_STEPS)
    cfg.evaluation_games = getattr(args, "eval_games", config.EVALUATION_GAMES)
    cfg.evaluation_concurrent = getattr(args, "eval_concurrent", config.EVALUATION_CONCURRENT_GAMES)
    cfg.max_batch_size = getattr(args, "batch_size", config.BATCH_SIZE)
    cfg.save_interval = getattr(args, "checkpoint_interval", config.CHECKPOINT_STEPS)
    cfg.starting_train_step = getattr(args, "starting_episode", 0)
    cfg.run_name = getattr(args, "run_name", "")
    return cfg


async def training_mode(args):
    """Full training with the Orchestrator (Collect -> Train -> Sync -> Evaluate)."""
    print("=== Training Mode (TrainingOrchestrator) ===")
    cfg = _build_config(args)
    orch = TrainingOrchestrator(cfg)
    orch.setup()
    await orch.run(max_steps=getattr(args, "max_steps", 1_000_000))
    return True


async def evaluation_mode(args):
    """Run evaluation games."""
    print("=== Evaluation Mode ===")
    try:
        if args.quick:
            await quick_evaluation(
                num_games=args.eval_games,
                concurrent=args.eval_concurrent,
            )
        else:
            cfg = _build_config(args)
            orch = TrainingOrchestrator(cfg)
            orch.setup()
            await orch.run_evaluation(args.eval_games)
        return True
    except Exception as e:
        print(f"Evaluation error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def demo_mode(args):
    """Parallel demo (no training) – replaces train_parallel.py."""
    print("=== Parallel Demo Mode ===")
    cfg = _build_config(args)
    orch = TrainingOrchestrator(cfg)
    orch.setup()
    results = await orch.run_parallel_demo(num_episodes=args.demo_episodes)
    print(f"Demo completed: {len(results)} games")
    return True


async def debug_mode(args):
    """Debug / development utilities."""
    print("=== Debug Mode ===")

    if args.debug_network:
        print("Debugging neural network architecture...")
        net = TFTNetwork()
        total = sum(p.numel() for p in net.parameters())
        print(f"Parameters: {total}")
        for name in net.state_dict():
            print(f"  Layer: {name}")

    if args.debug_single_episode:
        print("Running single episode (replaces train_single.py)...")
        cfg = _build_config(args)
        orch = TrainingOrchestrator(cfg)
        orch.setup()
        result = await orch.run_single_episode()
        print(f"Episode {result.game_id}: {result.duration:.2f}s")
        for pid, placement in result.placements.items():
            print(f"  {pid}: #{placement}")

    return True


def _check_gil():
    if not config.IS_GIL_DISABLED and config.FORCE_THREADING_ENV_MANAGER:
        print("ERROR: GIL is enabled but FORCE_THREADING_ENV_MANAGER is True.", file=sys.stderr)
        print("Run via ./run_tft.sh to use the free-threaded Python build in the TFT conda environment.", file=sys.stderr)
        sys.exit(1)

async def async_main():
    _check_gil()
    """Main async entry point."""
    parser = argparse.ArgumentParser(
        description="TFT MuZero Agent – TrainingOrchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Core args
    parser.add_argument("--starting_episode", "-se", type=int, default=0,
                        help="Episode / step to resume from")
    parser.add_argument("--run_name", "-rn", type=str, default="",
                        help="Run name for logging")

    # Mode
    parser.add_argument("--mode", "-m",
                        choices=["train", "eval", "demo", "debug"],
                        default="train",
                        help="Execution mode")

    # Training
    parser.add_argument("--concurrent_games", "-cg", type=int,
                        default=config.CONCURRENT_GAMES)
    parser.add_argument("--max_steps", "-ms", type=int, default=1_000_000)
    parser.add_argument("--eval_interval", "-ei", type=int,
                        default=config.CHECKPOINT_STEPS)
    parser.add_argument("--eval_games", "-eg", type=int,
                        default=config.EVALUATION_GAMES)
    parser.add_argument("--eval_concurrent", "-ec", type=int,
                        default=config.EVALUATION_CONCURRENT_GAMES)
    parser.add_argument("--batch_size", "-bs", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--checkpoint_interval", "-ci", type=int,
                        default=config.CHECKPOINT_STEPS)

    # Eval
    parser.add_argument("--quick", "-q", action="store_true",
                        help="Quick evaluation")

    # Demo
    parser.add_argument("--demo_episodes", "-de", type=int, default=5,
                        help="Number of demo episodes (demo mode)")

    # Debug
    parser.add_argument("--debug_network", action="store_true",
                        help="Print network architecture")
    parser.add_argument("--debug_single_episode", action="store_true",
                        help="Run one episode for debugging")

    args = parser.parse_args()

    print("=" * 60)
    print("TFT MuZero Agent – TrainingOrchestrator")
    print("=" * 60)
    print(f"Mode: {args.mode}")

    success = False
    try:
        if args.mode == "train":
            success = await training_mode(args)
        elif args.mode == "eval":
            success = await evaluation_mode(args)
        elif args.mode == "demo":
            success = await demo_mode(args)
        elif args.mode == "debug":
            success = await debug_mode(args)
        else:
            print(f"Unknown mode: {args.mode}")
    except KeyboardInterrupt:
        print("\nInterrupted")
        success = True
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    if success:
        print("\n" + "=" * 60)
        print("Completed successfully!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Failed!")
        print("=" * 60)
        sys.exit(1)


def main():
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"Critical error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
