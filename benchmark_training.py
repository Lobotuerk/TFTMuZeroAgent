"""
Standalone benchmark script for the collection process (environment stepping and inference).

Runs the environment collection process for a fixed number of games with
configurable concurrency and outputs a detailed performance breakdown
focusing specifically on how much a round takes and how much a whole game takes.

Usage:
    python benchmark_training.py                          # defaults
    python benchmark_training.py --games 2               # number of games
    python benchmark_training.py --concurrent 8           # custom concurrency
"""

import asyncio
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training_orchestrator import TrainingOrchestrator, TrainingConfig
import config


def build_config(args) -> TrainingConfig:
    return TrainingConfig(
        starting_train_step=0,
        concurrent_games=args.concurrent,
        evaluation_games=0,
        evaluation_concurrent=0,
        max_batch_size=config.BATCH_SIZE,
        sync_steps=999999,
        save_interval=999999,
        evaluation_interval=999999,
    )


async def run_benchmark(args):
    cfg = build_config(args)
    orch = TrainingOrchestrator(cfg)
    orch.setup()

    # Determine number of games to run, allowing fallback from deprecated --steps
    num_games = args.games
    if args.steps is not None and args.games == 2:
        # If --steps was explicitly passed, map it to games for backward compatibility
        num_games = max(1, args.steps // 50)
        print(f"[Warning] --steps parameter is deprecated for collection benchmark. Mapping steps ({args.steps}) to games ({num_games}).")

    print(f"\nStarting Collection Benchmark: {num_games} games, "
          f"{cfg.concurrent_games} concurrent games")
    print(f"Environment manager: MultiProcessEnvManager")
    print("-" * 60)

    try:
        # Run games in parallel without training loop
        await orch.run_parallel_demo(num_episodes=num_games)
    except KeyboardInterrupt:
        print("\nBenchmark interrupted")
    finally:
        orch.cleanup()

    orch.print_profiling_summary()

    summary = orch.profiling.summary()
    total_env = summary["env_step_time"]
    total_inf = summary["inference_wait_time"]
    avg_round = summary.get("avg_round_time", 0.0)
    avg_game = summary.get("avg_game_time", 0.0)
    round_count = summary.get("round_count", 0)
    game_count = summary.get("game_count", 0)

    print("\n" + "=" * 60)
    print("COLLECTION PROCESS BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  Rounds benchmarked : {round_count}")
    print(f"  Average round time : {avg_round:.3f}s ({avg_round*1000:.1f} ms)")
    print(f"  Games benchmarked  : {game_count}")
    print(f"  Average game time  : {avg_game:.3f}s")
    print(f"  ───────────────────────────────────────────")
    print(f"  Environment step   : {total_env:.2f}s")
    print(f"  Inference wait     : {total_inf:.2f}s")
    print("=" * 60)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the TFT MuZero collection process (environment and inference)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--games", type=int, default=2,
                        help="Number of games to run (default: 2)")
    parser.add_argument("--steps", type=int, default=None,
                        help="Deprecated: Number of training steps (fallback compatibility mapped to games)")
    parser.add_argument("--concurrent", type=int, default=config.CONCURRENT_GAMES,
                        help=f"Number of concurrent games (default: {config.CONCURRENT_GAMES})")
    parser.add_argument("--eval-games", type=int, default=0,
                        help="Unused (deprecated)")
    parser.add_argument("--eval-concurrent", type=int, default=0,
                        help="Unused (deprecated)")
    parser.add_argument("--sync-steps", type=int, default=0,
                        help="Unused (deprecated)")

    args = parser.parse_args()
    print("=" * 60)
    print("TFT MuZero Agent – Collection Benchmark")
    print("=" * 60)

    try:
        success = asyncio.run(run_benchmark(args))
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(1)
    except Exception as e:
        print(f"Benchmark error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if success:
        print("\nBenchmark completed successfully!")
    else:
        print("\nBenchmark failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
