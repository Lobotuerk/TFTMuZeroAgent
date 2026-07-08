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
    """GPU-bound Training Server process with HTTP API."""
    import base64
    import os
    import pickle
    import shutil
    from datetime import datetime as dt
    import torch
    from aiohttp import web
    
    print("=== Training Server Mode (HTTP API) ===")
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
        action_limits=config.ACTION_DIM,
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
    
    # Create checkpoint directory
    os.makedirs("./checkpoint", exist_ok=True)
    
    # If starting from a checkpoint, load it
    if args.starting_episode > 0:
        step_path = f"./checkpoint/current_{args.starting_episode}"
        if os.path.isfile(step_path):
            orch.current_model.model.load_state_dict(torch.load(step_path))
            orch.training_step = args.starting_episode
            print(f"Resumed from checkpoint step {orch.training_step}")
    else:
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
    
    # ── HTTP API handlers ─────────────────────────────────────────
    
    async def handle_experience(request):
        experience_type = request.headers.get("X-Experience-Type", "")
        if experience_type not in ("gameplay", "combat"):
            return web.Response(status=400, text="Invalid or missing X-Experience-Type header")
        body = await request.read()
        try:
            data = pickle.loads(body)
        except Exception:
            return web.Response(status=400, text="Invalid pickle data")
        if experience_type == "gameplay":
            orch.global_buffer.gameplay_buffer.add(data)
        else:
            for sample in data:
                orch.global_buffer.store_combat(sample)
        return web.Response(status=200)
    
    async def handle_weights(request):
        name = request.match_info["name"]
        if name not in ("best", "latest"):
            return web.Response(status=404, text="Invalid weight name")
        path = f"./checkpoint/{name}_model.pth"
        if not os.path.isfile(path):
            return web.Response(status=404, text=f"Weights file not found: {path}")
        if_modified_since = request.headers.get("If-Modified-Since")
        if if_modified_since:
            try:
                since = dt.fromisoformat(if_modified_since)
                mtime = dt.fromtimestamp(os.path.getmtime(path))
                if mtime <= since:
                    return web.Response(status=304)
            except (ValueError, OSError):
                pass
        with open(path, "rb") as f:
            data = f.read()
        last_modified = dt.fromtimestamp(os.path.getmtime(path)).isoformat()
        encoded_weights = base64.b64encode(data).decode('utf-8')
        body = {
            "step": orch.training_step,
            "weights": encoded_weights,
        }
        return web.json_response(body, headers={"Last-Modified": last_modified})
    
    async def handle_promote_best(request):
        try:
            shutil.copy("./checkpoint/latest_model.pth", "./checkpoint/best_model.pth")
            return web.Response(status=200)
        except Exception:
            return web.Response(status=500, text="Failed to promote best model")
    
    # ── Start HTTP server ──────────────────────────────────────────
    app = web.Application(client_max_size=0)
    app.router.add_post("/api/v1/experience", handle_experience)
    app.router.add_get("/api/v1/weights/{name}", handle_weights)
    app.router.add_post("/api/v1/weights/promote_best", handle_promote_best)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, config.SERVER_HOST, config.SERVER_PORT)
    await site.start()
    print(f"HTTP API server listening on {config.SERVER_HOST}:{config.SERVER_PORT}")
    
    # ── Training loop (data arrives via HTTP, no file polling) ─────
    orch.training_active = True
    try:
        while orch.training_active:
            trained = False
            while orch.training_active and orch.global_buffer.available_gameplay_batch() and orch.training_step < args.max_steps:
                await orch._train_step()
                trained = True
                await asyncio.sleep(0.001)
            if not trained:
                await asyncio.sleep(1.0)
    except KeyboardInterrupt:
        print("\nTraining server stopped.")
    finally:
        await runner.cleanup()
        orch._train_executor.shutdown(wait=False)
    return True


async def worker_mode(args):
    """CPU-bound Game Collection or Evaluation Worker process (HTTP client)."""
    import base64
    import os
    import pickle
    import io
    import torch
    import sys
    import aiohttp
    
    worker_id = getattr(args, "worker_id", 0)
    worker_role = getattr(args, "worker_role", "collector")
    server_url = f"http://{config.WORKERS_HOST}:{config.SERVER_PORT}"
    
    # Redirect stdout and stderr to a unique line-buffered file per worker
    log_file_path = f"log_n_{worker_id}.txt"
    log_file = open(log_file_path, "w", buffering=1)
    sys.stdout = log_file
    sys.stderr = log_file
    
    print(f"=== Worker Process {worker_id} Mode: {worker_role} (Server: {server_url}) ===")
    
    cfg = _build_config(args)
    
    # We create a local TrainingOrchestrator to run games
    orch = TrainingOrchestrator(cfg)
    
    # Only evaluator and trainer should write to TensorBoard. Disable for collectors
    # before calling setup() to avoid creating empty TensorBoard directories on disk.
    is_collector = (worker_role == "collector")
    if is_collector:
        orch._build_logger = lambda: None
        
    orch.setup(is_collector=is_collector)
    
    timeout = aiohttp.ClientTimeout(total=30, connect=10)
    
    try:
        async with aiohttp.ClientSession() as session:
            
            async def _request_with_retry(method, url, max_retries=3, **kwargs):
                for attempt in range(max_retries):
                    try:
                        return await session.request(method, url, timeout=timeout, **kwargs)
                    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                        print(f"[Worker {worker_id}] Request failed (attempt {attempt+1}/{max_retries}): {e}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2 ** attempt)
                        else:
                            raise
            if worker_role == "collector":
                print(f"[Worker {worker_id}] Starting collection loop...")
                while True:
                    # 1. Pull best weights from server
                    resp = await _request_with_retry("GET", f"{server_url}/api/v1/weights/best")
                    async with resp:
                        if resp.status == 200:
                            try:
                                resp_json = await resp.json()
                                weights_bytes = base64.b64decode(resp_json["weights"])
                                weights = torch.load(io.BytesIO(weights_bytes), map_location="cpu")
                                if orch.current_model is not None:
                                    orch.current_model.model.load_state_dict(weights)
                                for agent in orch._training_agents:
                                    agent.update_weights(weights)
                            except Exception as e:
                                print(f"[Worker {worker_id}] Error loading weights: {e}")
                        else:
                            body = await resp.text()
                            print(f"[Worker {worker_id}] Failed to fetch weights (status {resp.status}): {body[:200]}")
                    
                    # 2. Run fixed games to collect experiences
                    print(f"[Worker {worker_id}] Starting a batch of {cfg.concurrent_games} game(s)...")
                    await orch.env_manager.run_fixed_games(orch.agent_manager, cfg.concurrent_games)
                    
                    # 3. Send gameplay experiences to server
                    samples = list(orch.global_buffer.gameplay_buffer)
                    if samples:
                        data = pickle.dumps(samples)
                        resp = await _request_with_retry(
                            "POST", f"{server_url}/api/v1/experience",
                            data=data,
                            headers={"Content-Type": "application/octet-stream",
                                     "X-Experience-Type": "gameplay"}
                        )
                        async with resp:
                            if resp.status == 200:
                                print(f"[Worker {worker_id}] Sent {len(samples)} gameplay steps")
                                orch.global_buffer.clear_gameplay_buffer()
                            else:
                                body = await resp.text()
                                print(f"[Worker {worker_id}] Failed to send gameplay steps (status {resp.status}): {body[:200]}")
                    
                    # 4. Send combat experiences to server
                    combat_buffer = orch.global_buffer.combat_buffer
                    if combat_buffer._size > 0:
                        combat_samples = combat_buffer._buffer[:combat_buffer._size]
                        data = pickle.dumps(combat_samples)
                        resp = await _request_with_retry(
                            "POST", f"{server_url}/api/v1/experience",
                            data=data,
                            headers={"Content-Type": "application/octet-stream",
                                     "X-Experience-Type": "combat"}
                        )
                        async with resp:
                            if resp.status == 200:
                                print(f"[Worker {worker_id}] Sent {len(combat_samples)} combat steps")
                                combat_buffer.clear()
                            else:
                                body = await resp.text()
                                print(f"[Worker {worker_id}] Failed to send combat steps (status {resp.status}): {body[:200]}")
                    
                    import gc
                    gc.collect()
                    await asyncio.sleep(1.0)
                    
            elif worker_role == "evaluator":
                print(f"[Worker {worker_id}] Starting evaluator loop...")
                last_modified = ""
                
                while True:
                    # 1. Check if latest weights changed on server
                    headers = {}
                    if last_modified:
                        headers["If-Modified-Since"] = last_modified
                    
                    resp = await _request_with_retry("GET", f"{server_url}/api/v1/weights/latest", headers=headers)
                    async with resp:
                        if resp.status == 304:
                            await asyncio.sleep(5.0)
                            continue
                        if resp.status == 404:
                            body = await resp.text()
                            print(f"[Worker {worker_id}] Latest weights not found (status 404): {body[:200]}")
                            await asyncio.sleep(5.0)
                            continue
                        if resp.status != 200:
                            body = await resp.text()
                            print(f"[Worker {worker_id}] Failed to fetch latest weights (status {resp.status}): {body[:200]}")
                            await asyncio.sleep(5.0)
                            continue
                        
                        last_modified = resp.headers.get("Last-Modified", "")
                        print(f"[Worker {worker_id}] Detected updated weights. Running evaluation...")
                        
                        try:
                            resp_json = await resp.json()
                            step = resp_json.get("step", 0)
                            weights_bytes = base64.b64decode(resp_json["weights"])
                            latest_weights = torch.load(io.BytesIO(weights_bytes), map_location="cpu")
                            orch.current_model.model.load_state_dict(latest_weights)
                        except Exception:
                            print(f"[Worker {worker_id}] Error loading latest weights")
                            await asyncio.sleep(2.0)
                            continue
                    
                    # 2. Load best weights from server
                    best_resp = await _request_with_retry("GET", f"{server_url}/api/v1/weights/best")
                    async with best_resp:
                        if best_resp.status == 200:
                            try:
                                best_json = await best_resp.json()
                                best_bytes = base64.b64decode(best_json["weights"])
                                best_weights = torch.load(io.BytesIO(best_bytes), map_location="cpu")
                                orch.best_model.model.load_state_dict(best_weights)
                            except Exception:
                                pass
                        else:
                            # No best weights yet on server; latest is best by default
                            orch.best_model.model.load_state_dict(latest_weights)
                    
                    # 3. Run standalone evaluation
                    results = await orch.evaluate(step=step)
                    current_mean = results["current_placement"]
                    best_mean = results["best_placement"]
                    
                    # 4. If current is strictly better, promote on server
                    if current_mean < best_mean:
                        print(f"[Evaluator] \u2713 Model improved! Placement: {current_mean:.2f} vs {best_mean:.2f}")
                        prom_resp = await _request_with_retry("POST", f"{server_url}/api/v1/weights/promote_best")
                        async with prom_resp:
                            if prom_resp.status == 200:
                                print("[Evaluator] Promoted latest to best on server")
                            else:
                                body = await prom_resp.text()
                                print(f"[Evaluator] Failed to promote best (status {prom_resp.status}): {body[:200]}")
                    else:
                        print(f"[Evaluator] \u2717 No improvement. Placement: {current_mean:.2f} vs {best_mean:.2f}")
                    
                    import gc
                    gc.collect()
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
