"""
Standalone benchmark script for the training pipeline.

Runs the TrainingOrchestrator for a fixed number of training steps with
configurable concurrency and outputs a detailed performance breakdown
comparing environment stepping time, inference wait time, training time,
and idle time.

Usage:
    python benchmark_training.py                          # defaults
    python benchmark_training.py --steps 50               # fewer steps
    python benchmark_training.py --concurrent 8 --eval 5  # custom concurrency
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
        evaluation_games=args.eval_games,
        evaluation_concurrent=args.eval_concurrent,
        max_batch_size=config.BATCH_SIZE,
        sync_steps=args.sync_steps,
        save_interval=999999,
        evaluation_interval=999999,
    )


async def run_benchmark(args):
    cfg = build_config(args)
    orch = TrainingOrchestrator(cfg)
    orch.setup()

    print(f"\nStarting benchmark: {args.steps} training steps, "
          f"{cfg.concurrent_games} concurrent games, "
          f"{cfg.evaluation_games} evaluation games")
    print(f"Environment manager: "
          f"{'ThreadEnvManager' if config.FORCE_THREADING_ENV_MANAGER else 'MultiProcessEnvManager'}")
    print("-" * 60)

    import numpy as np
    # Pre-populate global buffer to bypass the sequential environment data-collection lag
    if orch.global_buffer is not None and type(orch.global_buffer).__name__ != 'MagicMock':
        print("Pre-populating global buffer with synthetic experiences for training benchmark...")
        synthetic_experiences = []
        for _ in range(config.BATCH_SIZE * 4):
            synthetic_experiences.append([
                np.zeros(config.OBSERVATION_SIZE),
                [np.zeros(3, dtype=np.int32) for _ in range(config.UNROLL_STEPS - 1)],
                [0.0] * config.UNROLL_STEPS,
                [0.0] * config.UNROLL_STEPS,
                [np.zeros(config.ACTION_CONCAT_SIZE) for _ in range(config.UNROLL_STEPS)],
            ])
        orch.global_buffer.store_episode(synthetic_experiences)
        print(f"Global buffer size: {orch.global_buffer.get_gameplay_buffer_size()} experiences.")

    try:
        await orch.run(max_steps=args.steps)
    except KeyboardInterrupt:
        print("\nBenchmark interrupted")
    finally:
        orch.cleanup()

    orch.print_profiling_summary()

    total_env = orch.profiling.summary()["env_step_time"]
    total_inf = orch.profiling.summary()["inference_wait_time"]
    total_train = orch.profiling.summary()["train_time"]
    total_idle = orch.profiling.summary()["idle_time"]
    grand = total_env + total_inf + total_train + total_idle

    print(f"\n  Environment stepping : {total_env:.2f}s  ({total_env/grand*100:.1f}%)")
    print(f"  Inference wait       : {total_inf:.2f}s  ({total_inf/grand*100:.1f}%)")
    print(f"  Training             : {total_train:.2f}s  ({total_train/grand*100:.1f}%)")
    print(f"  Idle                 : {total_idle:.2f}s  ({total_idle/grand*100:.1f}%)")
    print(f"  ───────────────────────────────────────────")
    print(f"  Total wall time      : {grand:.2f}s")

    return True


def _check_gil():
    if not config.IS_GIL_DISABLED and config.FORCE_THREADING_ENV_MANAGER:
        print("ERROR: GIL is enabled but FORCE_THREADING_ENV_MANAGER is True.", file=sys.stderr)
        print("Run via ./run_tft.sh to use the free-threaded Python build in the TFT conda environment.", file=sys.stderr)
        sys.exit(1)

def main():
    _check_gil()
    parser = argparse.ArgumentParser(
        description="Benchmark the TFT MuZero training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--steps", type=int, default=100,
                        help="Number of training steps to run (default: 100)")
    parser.add_argument("--concurrent", type=int, default=config.CONCURRENT_GAMES,
                        help=f"Number of concurrent games (default: {config.CONCURRENT_GAMES})")
    parser.add_argument("--eval-games", type=int, default=config.EVALUATION_GAMES,
                        help=f"Number of evaluation games (default: {config.EVALUATION_GAMES})")
    parser.add_argument("--eval-concurrent", type=int, default=config.EVALUATION_CONCURRENT_GAMES,
                        help=f"Concurrent evaluation games (default: {config.EVALUATION_CONCURRENT_GAMES})")
    parser.add_argument("--sync-steps", type=int, default=config.SYNC_STEPS,
                        help=f"Sync interval in steps (default: {config.SYNC_STEPS})")

    args = parser.parse_args()
    print("=" * 60)
    print("TFT MuZero Agent – Training Benchmark")
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
