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
    cfg.collect_games_per_batch = getattr(args, "collect_games", config.COLLECT_GAMES_PER_BATCH)
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


async def train_server_mode(args):
    """GPU-bound Training Server process (Option A)."""
    import os
    import glob
    import pickle
    import torch
    
    print("=== Training Server Mode (Option A) ===")
    cfg = _build_config(args)
    orch = TrainingOrchestrator(cfg)
    
    # We do NOT run full orch.setup() because we don't need game environments!
    # Instead, we manually initialize the minimal training components:
    from Models.global_buffer import create_global_buffer
    from Models.MuZero_torch_agent import MuZeroAgent
    from Models.MuZero_torch_trainer import Trainer as MuZeroTrainer
    
    orch.global_buffer = create_global_buffer(cfg.max_batch_size)
    
    # Initialize active model (current_model)
    orch.current_model = MuZeroAgent(
        action_size=3,
        action_limits=[7, 37, 10],
        obs_size=config.OBSERVATION_SIZE,
        simulations=config.NUM_SIMULATIONS,
        global_buffer=orch.global_buffer,
        config_obj=cfg,
    )
    orch.trainer = MuZeroTrainer()
    
    # Set required executors/logs
    from concurrent.futures import ThreadPoolExecutor
    orch._train_executor = ThreadPoolExecutor(max_workers=1)
    orch.summary_writer = orch._build_logger()
    
    # Create checkpoints, gameplay, and combat directories
    os.makedirs("./checkpoint", exist_ok=True)
    os.makedirs(config.GAMEPLAY_BUFFER_PATH, exist_ok=True)
    os.makedirs(config.COMBAT_BUFFER_PATH, exist_ok=True)
    
    # If starting from a checkpoint, load it
    if args.starting_episode > 0:
        step_path = f"./checkpoint/current_{args.starting_episode}"
        if os.path.isfile(step_path):
            orch.current_model.model.load_state_dict(torch.load(step_path))
            orch.training_step = args.starting_episode
            print(f"Resumed from checkpoint step {orch.training_step}")
    else:
        # Auto-detect latest step if checkpoint files exist and latest_model.pth is present
        import glob
        checkpoints = glob.glob("./checkpoint/current_*")
        if checkpoints and os.path.isfile("./checkpoint/latest_model.pth"):
            steps = []
            for ckpt in checkpoints:
                try:
                    steps.append(int(ckpt.split("current_")[-1]))
                except ValueError:
                    pass
            if steps:
                latest_step = max(steps)
                step_path = f"./checkpoint/current_{latest_step}"
                try:
                    orch.current_model.model.load_state_dict(torch.load(step_path))
                    orch.training_step = latest_step
                    print(f"Auto-detected and resumed from checkpoint step {orch.training_step}")
                except Exception as e:
                    print(f"Failed to load latest checkpoint {step_path}: {e}. Falling back to default latest_model.pth.")
                    try:
                        orch.current_model.model.load_state_dict(torch.load("./checkpoint/latest_model.pth"))
                        print("Found existing latest_model.pth checkpoint, loaded to training server.")
                    except Exception:
                        pass
        elif os.path.isfile("./checkpoint/latest_model.pth"):
            try:
                orch.current_model.model.load_state_dict(torch.load("./checkpoint/latest_model.pth"))
                print("Found existing latest_model.pth checkpoint, loaded to training server.")
            except Exception:
                pass
            
    # Override save_current_checkpoint to ALSO save latest_model.pth for worker sync
    original_save_current = orch.save_current_checkpoint
    def custom_save_current():
        original_save_current()
        try:
            torch.save(orch.current_model.model.state_dict(), "./checkpoint/latest_model.pth")
            print(f"[Server] Synced latest weights to latest_model.pth at step {orch.training_step} (Gameplay Buffer Size: {len(orch.global_buffer.gameplay_buffer)})")
        except Exception as e:
            print(f"Error syncing latest_model.pth: {e}")
    orch.save_current_checkpoint = custom_save_current
    
    # Save first initial weights so workers have something to start with
    torch.save(orch.current_model.model.state_dict(), "./checkpoint/latest_model.pth")
    if not os.path.isfile("./checkpoint/best_model.pth"):
        torch.save(orch.current_model.model.state_dict(), "./checkpoint/best_model.pth")
            
    print("Training server is listening for worker experience files...")
    
    orch.training_active = True
    try:
        while orch.training_active:
            # 1. Sweep gameplay buffer folder for pkl experience files
            pattern = os.path.join(config.GAMEPLAY_BUFFER_PATH, "exp_*.pkl")
            exp_files = glob.glob(pattern)
            
            for f_path in exp_files:
                try:
                    with open(f_path, "rb") as f:
                        replay_set = pickle.load(f)
                    orch.global_buffer.gameplay_buffer.add(replay_set)
                    os.remove(f_path) # Delete consumed file
                except Exception:
                    # File might be partially written, try again next loop
                    pass
                    
            # 2. Sweep combat buffer folder for pkl experience files
            c_pattern = os.path.join(config.COMBAT_BUFFER_PATH, "combat_*.pkl")
            combat_files = glob.glob(c_pattern)
            
            for c_path in combat_files:
                try:
                    with open(c_path, "rb") as f:
                        combat_samples = pickle.load(f)
                    for sample in combat_samples:
                        orch.global_buffer.store_combat(sample)
                    os.remove(c_path) # Delete consumed file
                except Exception:
                    # File might be partially written, try again next loop
                    pass
                    
            # 3. Run training step if data is available
            trained = False
            while orch.training_active and orch.global_buffer.available_gameplay_batch() and orch.training_step < args.max_steps:
                await orch._train_step() # Run 1 training update
                trained = True
                await asyncio.sleep(0.001)
                
            if not trained:
                await asyncio.sleep(1.0) # Rest the CPU/GPU while waiting for data
                
    except KeyboardInterrupt:
        print("\nTraining server stopped.")
    finally:
        orch._train_executor.shutdown(wait=False)
    return True


async def worker_mode(args):
    """CPU-bound Game Collection or Evaluation Worker process (Option A)."""
    import os
    import pickle
    import torch
    import sys
    
    worker_id = getattr(args, "worker_id", 0)
    worker_role = getattr(args, "worker_role", "collector")
    
    # Redirect stdout and stderr to a unique line-buffered file per worker
    log_file_path = f"log_n_{worker_id}.txt"
    log_file = open(log_file_path, "w", buffering=1)
    sys.stdout = log_file
    sys.stderr = log_file
    
    print(f"=== Worker Process {worker_id} Mode: {worker_role} ===")
    
    cfg = _build_config(args)
    
    # We create a local TrainingOrchestrator to run games
    orch = TrainingOrchestrator(cfg)
    
    # Only evaluator and trainer should write to TensorBoard. Disable for collectors
    # before calling setup() to avoid creating empty TensorBoard directories on disk.
    is_collector = (worker_role == "collector")
    if is_collector:
        orch._build_logger = lambda: None
        
    orch.setup(is_collector=is_collector)
            
    os.makedirs(config.GAMEPLAY_BUFFER_PATH, exist_ok=True)
    os.makedirs(config.COMBAT_BUFFER_PATH, exist_ok=True)
    os.makedirs("./checkpoint", exist_ok=True)
    
    try:
        if worker_role == "collector":
            print(f"[Worker {worker_id}] Starting collection loop...")
            while True:
                # 1. Pull best weights if available (MuZero self-play is driven by current best weights)
                best_weights_path = "./checkpoint/best_model.pth"
                if os.path.isfile(best_weights_path):
                    try:
                        weights = torch.load(best_weights_path, map_location="cpu")
                        if orch.current_model is not None:
                            orch.current_model.model.load_state_dict(weights)
                        # Sync weights to training agents
                        for agent in orch._training_agents:
                            agent.update_weights(weights)
                    except Exception:
                        # Might be writing, skip this iteration
                        pass
                
                # 2. Run fixed games to collect experiences
                print(f"[Worker {worker_id}] Starting a batch of {cfg.concurrent_games} game(s)...")
                await orch.env_manager.run_fixed_games(orch.agent_manager, cfg.concurrent_games)
                
                # 3. Pull experiences from local gameplay buffer
                samples = list(orch.global_buffer.gameplay_buffer)
                if samples:
                    # 4. Save to shared gameplay directory
                    timestamp = time.time_ns()
                    f_path = os.path.join(config.GAMEPLAY_BUFFER_PATH, f"exp_worker_{worker_id}_{timestamp}.pkl")
                    with open(f_path, "wb") as f:
                        pickle.dump(samples, f)
                    print(f"[Worker {worker_id}] Saved {len(samples)} steps to {f_path}")
                    
                    # 5. Clear local buffer
                    orch.global_buffer.clear_gameplay_buffer()
                    
                # 6. Pull experiences from local combat buffer
                combat_buffer = orch.global_buffer.combat_buffer
                if combat_buffer._size > 0:
                    combat_samples = combat_buffer._buffer[:combat_buffer._size]
                    timestamp = time.time_ns()
                    c_path = os.path.join(config.COMBAT_BUFFER_PATH, f"combat_worker_{worker_id}_{timestamp}.pkl")
                    with open(c_path, "wb") as f:
                        pickle.dump(combat_samples, f)
                    print(f"[Worker {worker_id}] Saved {len(combat_samples)} combat steps to {c_path}")
                    
                    # 7. Clear local combat buffer
                    combat_buffer._size = 0
                    combat_buffer._pos = 0
                    
                await asyncio.sleep(1.0)
                
        elif worker_role == "evaluator":
            print(f"[Worker {worker_id}] Starting evaluator loop...")
            last_evaluated_mtime = 0
            
            latest_weights_path = "./checkpoint/latest_model.pth"
            best_weights_path = "./checkpoint/best_model.pth"
            
            while True:
                # Check if latest_model.pth was updated
                if os.path.isfile(latest_weights_path):
                    mtime = os.path.getmtime(latest_weights_path)
                    if mtime > last_evaluated_mtime:
                        last_evaluated_mtime = mtime
                        print(f"[Worker {worker_id}] Detected updated latest_model.pth. Running evaluation...")
                        
                        # Load latest model weights
                        try:
                            # Parse step from checkpoints to ensure correct TensorBoard plotting step
                            import glob
                            checkpoints = glob.glob("./checkpoint/current_*")
                            if checkpoints:
                                steps = []
                                for ckpt in checkpoints:
                                    try:
                                        steps.append(int(ckpt.split("current_")[-1]))
                                    except ValueError:
                                        pass
                                if steps:
                                    orch.training_step = max(steps)
                                    
                            latest_weights = torch.load(latest_weights_path, map_location="cpu")
                            orch.current_model.model.load_state_dict(latest_weights)
                        except Exception:
                            # File might be mid-write, skip this iteration
                            last_evaluated_mtime = 0 # Retry next loop
                            await asyncio.sleep(2.0)
                            continue
                            
                        # Load best model weights (if not exists, default to latest)
                        if os.path.isfile(best_weights_path):
                            try:
                                best_weights = torch.load(best_weights_path, map_location="cpu")
                                orch.best_model.model.load_state_dict(best_weights)
                            except Exception:
                                pass
                        else:
                            # If no best_model.pth exists, current latest is the best by default
                            torch.save(latest_weights, best_weights_path)
                            orch.best_model.model.load_state_dict(latest_weights)
                            
                        # Run Standalone Evaluation
                        results = await orch.evaluate()
                        
                        # Save evaluation results
                        current_mean = results["current_placement"]
                        best_mean = results["best_placement"]
                        
                        # If current is strictly better (lower placement score)
                        if current_mean < best_mean:
                            print(f"[Evaluator] \u2713 Model improved! Placement: {current_mean:.2f} vs {best_mean:.2f}")
                            # Save as new best weights
                            torch.save(latest_weights, best_weights_path)
                            print("[Evaluator] Saved new best_model.pth")
                        else:
                            print(f"[Evaluator] \u2717 No improvement. Placement: {current_mean:.2f} vs {best_mean:.2f}")
                            
                await asyncio.sleep(5.0)
                
    except KeyboardInterrupt:
        print(f"\nWorker {worker_id} stopped.")
    finally:
        orch.cleanup()
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
                        choices=["train", "eval", "demo", "debug", "train_server", "worker"],
                        default="train",
                        help="Execution mode")

    # Distributed Option A Core args
    parser.add_argument("--worker_id", type=int, default=0,
                        help="ID of this collection worker (worker mode)")
    parser.add_argument("--worker_role", choices=["collector", "evaluator"], default="collector",
                        help="Role of this worker process (worker mode)")

    # Training
    parser.add_argument("--concurrent_games", "-cg", type=int,
                        default=config.CONCURRENT_GAMES)
    parser.add_argument("--collect_games", type=int, default=config.COLLECT_GAMES_PER_BATCH)
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
    if args.mode == "worker":
        print(f"Worker Role: {args.worker_role}  |  Worker ID: {args.worker_id}")

    success = False
    try:
        if args.mode == "train":
            success = await training_mode(args)
        elif args.mode == "train_server":
            success = await train_server_mode(args)
        elif args.mode == "worker":
            success = await worker_mode(args)
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
