"""
Training Orchestrator for TFT MuZero Agent

Drives the RL lifecycle: Collect -> Train -> Sync -> Evaluate

Explicit lifecycle phases:
   1. COLLECT: Run games in parallel via ParallelEnvironmentManager,
      gather experience into the GlobalBuffer
   2. TRAIN:   Sample from the buffer, update the model via Trainer
   3. SYNC:    Distribute the updated weights to the active collection
      agents so they immediately benefit from the new policy
   4. EVALUATE:Periodically pit the new model against the old one;
      keep the best performing weights
"""

import asyncio
import time
import copy
import datetime
import os
import sys
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import torch
from torch.utils.tensorboard import SummaryWriter

import config
from Models.global_buffer import GlobalBuffer
from Models.action_conversion import action_3d_to_policy
from Models.MuZero_torch_trainer import Trainer
from Models.MuZero_torch_agent import MuZeroAgent
from Models.Common_agents import CultistAgent, DivineAgent, RandomAgent, WarlordAgent
from Models.agent_manager import (
    create_enhanced_setup,
    create_custom_agent_setup,
    AsyncGameEnvironment,
    BatchInferenceServer,
    EnhancedAgentManager,
)
from TFTSet4Gym.tft_set4_gym.tft_simulator import parallel_env

from utils.profiling import EnvironmentBenchmark, MetricsCollector

import multiprocessing as mp
import threading

# Default multiprocessing context (fork on Linux 3.13-).
# Subprocesses only run CPU-bound env logic and do not touch GPU tensors,
# however, fork can cause deadlocks if PyTorch threads are active. Use spawn.
MP_CONTEXT = mp.get_context('spawn')


# ---------------------------------------------------------------------------
# Subprocess target – runs a game in its own process (bypasses GIL)
# ---------------------------------------------------------------------------

def _env_worker_main(env_id: int, conn):
    """
    Target function for an environment subprocess.

    Runs a continuous game loop in this subprocess, sending inference
    requests to the main process via *conn* and receiving actions back.
    Uses :func:`conn.poll` so the loop can be interrupted cleanly.

    Protocol (tuples over ``multiprocessing.Connection``):

    * env → main: ``('infer', observations, rewards, terminated)``
    * main → env: ``('actions', actions_dict)``
    * env → main: ``('done', scores_dict)``
    * main → env: ``('restart', None)`` or ``('stop', None)``
    """
    # Ensure the project root is on sys.path in the subprocess
    _root = os.path.dirname(os.path.abspath(__file__))
    if _root not in sys.path:
        sys.path.insert(0, _root)

    from TFTSet4Gym.tft_set4_gym.tft_simulator import parallel_env

    unique_rank = f"{os.getpid()}_{env_id}"
    env = parallel_env(rank=unique_rank)

    while True:
        observations = env.reset()[0]
        terminated = {pid: False for pid in env.possible_agents}
        rewards = {pid: 0.0 for pid in env.possible_agents}
        scores = {pid: 0.0 for pid in env.possible_agents}
        step_count = 0
        round_start_time = time.time()
        last_round = None
        round_durations = []

        while not all(terminated.values()):
            float_rewards = {k: float(v) for k, v in rewards.items()}

            conn.send(('infer', observations, float_rewards, terminated))

            try:
                msg = conn.recv()
            except (EOFError, BrokenPipeError, OSError):
                return
            if msg[0] == 'stop':
                return

            actions = msg[1]

            # --- process actions (mirrors _GameWorker.run_game) ----------
            processed = {}
            for pid, action in actions.items():
                if terminated.get(pid, True):
                    processed[pid] = [0, 0, 0]
                    continue
                if isinstance(action, (list, np.ndarray)) and len(action) >= 3:
                    processed[pid] = action[:3]
                elif hasattr(action, "tolist"):
                    lst = action.tolist()
                    if isinstance(lst, list) and len(lst) >= 3:
                        processed[pid] = lst[:3]
                    else:
                        processed[pid] = [0, 0, 0]
                else:
                    processed[pid] = [0, 0, 0]

            observations, rewards, terminated, _, infos = env.step(processed)

            # Track round progression
            try:
                game_round_obj = getattr(env, 'game_round', None)
                if game_round_obj is not None:
                    current_round = getattr(game_round_obj, 'current_round', None)
                    if current_round is not None:
                        if last_round is None:
                            last_round = current_round
                            round_start_time = time.time()
                        elif current_round != last_round:
                            round_durations.append(time.time() - round_start_time)
                            last_round = current_round
                            round_start_time = time.time()
            except Exception:
                pass

            for p in terminated:
                if terminated[p]:
                    scores[p] = rewards[p]

            step_count += 1
            if step_count > 1000:
                break

        # Record the last round's duration at the end of the game
        if last_round is not None:
            round_durations.append(time.time() - round_start_time)

        conn.send(('done', scores, round_durations))

        try:
            msg = conn.recv()
        except (EOFError, BrokenPipeError, OSError):
            return
        if msg[0] == 'stop':
            return
        elif msg[0] == 'pause':
            while True:
                try:
                    msg2 = conn.recv()
                except (EOFError, BrokenPipeError, OSError):
                    return
                if msg2[0] == 'restart':
                    break
                elif msg2[0] == 'stop':
                    return
        # 'restart' → fall through to outer loop





# ---------------------------------------------------------------------------
# Data objects
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Configuration for training parameters"""
    starting_train_step: int = 0
    run_name: str = ""
    save_interval: int = config.CHECKPOINT_STEPS
    evaluation_interval: int = config.CHECKPOINT_STEPS
    concurrent_games: int = config.CONCURRENT_GAMES
    collect_games_per_batch: int = config.COLLECT_GAMES_PER_BATCH
    evaluation_games: int = config.EVALUATION_GAMES
    evaluation_concurrent: int = config.EVALUATION_CONCURRENT_GAMES
    max_batch_size: int = config.BATCH_SIZE
    batch_timeout_ms: float = 5.0
    gpu_memory_fraction: float = 0.8
    sync_steps: int = config.SYNC_STEPS
    results_path: str = config.RESULTS_PATH


@dataclass
class GameResult:
    """Container for a single game outcome"""
    game_id: str
    placements: Dict[str, int]
    scores: Dict[str, float]
    duration: float
    agent_mapping: Dict[str, type]


@dataclass
class ProfilingTracker:
    """Thread-safe accumulator for runtime performance metrics."""
    env_step_times: List[float] = field(default_factory=list)
    inference_wait_times: List[float] = field(default_factory=list)
    train_step_times: List[float] = field(default_factory=list)
    idle_times: List[float] = field(default_factory=list)
    round_times: List[float] = field(default_factory=list)
    game_times: List[float] = field(default_factory=list)
    _lock: Any = field(default_factory=threading.Lock)

    def record_inference(self, duration: float):
        with self._lock:
            self.inference_wait_times.append(duration)

    def record_env_step(self, duration: float):
        with self._lock:
            self.env_step_times.append(duration)

    def record_train_step(self, duration: float):
        with self._lock:
            self.train_step_times.append(duration)

    def record_idle(self, duration: float):
        with self._lock:
            self.idle_times.append(duration)

    def record_round(self, duration: float):
        with self._lock:
            self.round_times.append(duration)

    def record_game(self, duration: float):
        with self._lock:
            self.game_times.append(duration)

    def summary(self) -> Dict[str, float]:
        total_inference = sum(self.inference_wait_times)
        total_env_step = sum(self.env_step_times)
        total_train = sum(self.train_step_times)
        total_idle = sum(self.idle_times)
        total = total_inference + total_env_step + total_train + total_idle
        return {
            "total_time": total,
            "env_step_time": total_env_step,
            "inference_wait_time": total_inference,
            "train_time": total_train,
            "idle_time": total_idle,
            "env_step_pct": (total_env_step / total * 100) if total > 0 else 0.0,
            "inference_pct": (total_inference / total * 100) if total > 0 else 0.0,
            "train_pct": (total_train / total * 100) if total > 0 else 0.0,
            "idle_pct": (total_idle / total * 100) if total > 0 else 0.0,
            "env_step_count": len(self.env_step_times),
            "inference_count": len(self.inference_wait_times),
            "train_step_count": len(self.train_step_times),
            "round_count": len(self.round_times),
            "game_count": len(self.game_times),
            "avg_env_step": (total_env_step / len(self.env_step_times)) if self.env_step_times else 0.0,
            "avg_inference_wait": (total_inference / len(self.inference_wait_times)) if self.inference_wait_times else 0.0,
            "avg_train_step": (total_train / len(self.train_step_times)) if self.train_step_times else 0.0,
            "avg_round_time": (sum(self.round_times) / len(self.round_times)) if self.round_times else 0.0,
            "avg_game_time": (sum(self.game_times) / len(self.game_times)) if self.game_times else 0.0,
        }


# ---------------------------------------------------------------------------
# Game worker – runs one async game
# ---------------------------------------------------------------------------

class _GameWorker:
    """
    Async game worker that replaces the legacy Ray DataWorker.
    Runs a single game asynchronously without Ray overhead.
    """

    def __init__(self, worker_id: int, profiling: Optional[ProfilingTracker] = None, metrics_collector: Optional[MetricsCollector] = None):
        self.worker_id = worker_id
        self.games_completed = 0
        self.profiling = profiling
        self.metrics_collector = metrics_collector

    async def run_game(self,
                       agent_manager: EnhancedAgentManager,
                       return_placements: bool = False) -> GameResult:
        """Run a single game end-to-end and return the result."""
        try:
            game_id = f"worker_{self.worker_id}_game_{self.games_completed}"
            start_time = time.time()
            round_start_time = time.time()
            last_round = None

            env = parallel_env(rank=self.worker_id)
            observations = env.reset()[0]
            terminated = {pid: False for pid in env.possible_agents}
            rewards = {pid: 0.0 for pid in env.possible_agents}
            scores = {pid: 0.0 for pid in env.possible_agents}

            step_count = 0
            while not all(terminated.values()):
                step_count += 1
                float_rewards = {k: float(v) for k, v in rewards.items()}

                t0_prof = time.time()
                t0_metrics = time.perf_counter()
                actions_task = agent_manager.get_actions(
                    observations, float_rewards, terminated, game_id=game_id
                )
                actions = await asyncio.wait_for(actions_task, timeout=30.0)
                if self.profiling:
                    self.profiling.record_inference(time.time() - t0_prof)
                if self.metrics_collector:
                    self.metrics_collector.record("worker_get_actions", time.perf_counter() - t0_metrics)

                processed = {}
                for pid, action in actions.items():
                    if terminated.get(pid, True):
                        processed[pid] = [0, 0, 0]
                        continue
                    if isinstance(action, (list, np.ndarray)) and len(action) >= 3:
                        processed[pid] = action[:3]
                    elif hasattr(action, "tolist"):
                        lst = action.tolist()
                        if isinstance(lst, list) and len(lst) >= 3:
                            processed[pid] = lst[:3]
                        else:
                             raise ValueError(f"Invalid action format from agent {pid}: {action}")
                    else:
                        raise ValueError(f"Invalid action format from agent {pid}: {action}")
                actions = processed

                t0_prof = time.time()
                t0_metrics = time.perf_counter()
                observations, rewards, terminated, _, infos = env.step(actions)
                if self.profiling:
                    self.profiling.record_env_step(time.time() - t0_prof)
                if self.metrics_collector:
                    self.metrics_collector.record("worker_env_step", time.perf_counter() - t0_metrics)

                # Track round progression
                try:
                    game_round_obj = getattr(env, 'game_round', None)
                    if game_round_obj is not None:
                        current_round = getattr(game_round_obj, 'current_round', None)
                        if current_round is not None:
                            if last_round is None:
                                last_round = current_round
                                round_start_time = time.time()
                            elif current_round != last_round:
                                round_duration = time.time() - round_start_time
                                if self.profiling:
                                    self.profiling.record_round(round_duration)
                                last_round = current_round
                                round_start_time = time.time()
                except Exception:
                    pass

                for p in terminated:
                    if terminated[p]:
                        scores[p] = rewards[p]

                if step_count > 1000:
                    break

            # Record the last round's duration at the end of the game
            if last_round is not None:
                round_duration = time.time() - round_start_time
                if self.profiling:
                    self.profiling.record_round(round_duration)

            placements = {}
            if return_placements:
                sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                for i, (pid, _) in enumerate(sorted_scores):
                    placements[pid] = i + 1

            agent_mapping = agent_manager.get_player_agent_mapping() if return_placements else {}
            
            # Flush all agent buffers with final scores
            await agent_manager.flush_all_buffers(final_values=scores, game_id=game_id)
            
            duration = time.time() - start_time
            self.games_completed += 1

            # Record total game duration
            if self.profiling:
                self.profiling.record_game(duration)

            return GameResult(
                game_id=game_id,
                placements=placements,
                scores=scores,
                duration=duration,
                agent_mapping=agent_mapping,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise


# ---------------------------------------------------------------------------
# Multi-process environment manager
# ---------------------------------------------------------------------------

class _MultiProcessEnvManager:
    """
    Manages N concurrent game workers running in **separate OS processes**
    to bypass the Python GIL.

    Each environment runs its game loop (``env.step`` / ``env.reset``) in a
    dedicated subprocess.  Inference requests are forwarded to the main
    process where the GPU-resident model lives; batched GPU inference and
    experience storage happen in the main (async) process while the
    subprocesses are free to execute the CPU-bound game logic in parallel.

    API intentionally mirrors :class:`_ParallelEnvManager` so that
    :class:`TrainingOrchestrator` can swap implementations with minimal
    changes.
    """

    def __init__(self, num_workers: int, worker_fn=None,
                 metrics_collector: Optional[MetricsCollector] = None):
        self.num_workers = num_workers
        self._worker_fn = worker_fn or _env_worker_main
        self._processes: Dict[int, Tuple[mp.Process, mp.connection.Connection]] = {}
        self._tasks: Dict[int, asyncio.Task] = {}
        self.should_continue = True
        self.should_spawn = True
        self._started = False
        self.metrics_collector = metrics_collector

        self._game_barrier_counter = mp.Value('i', 0)
        self._game_barrier_event = mp.Event()

    # ── lifecycle ────────────────────────────────────────────────

    def stop(self):
        """Signal all workers to stop and release subprocess resources."""
        self.should_continue = False
        self._game_barrier_event.set()
        self._cleanup()

    def pause(self):
        """Prevent spawning new games; let running ones drain naturally."""
        self.should_spawn = False

    def resume(self):
        """Allow spawning new games again."""
        self.should_spawn = True

    async def wait_for_drain(self):
        """Wait until all active game tasks have finished."""
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)

    # ── start workers ────────────────────────────────────────────

    def _start_workers(self):
        """Launch all subprocesses and create async handlers for each."""
        for i in range(self.num_workers):
            parent_conn, child_conn = MP_CONTEXT.Pipe(duplex=True)
            proc = MP_CONTEXT.Process(
                target=self._worker_fn,
                args=(i, child_conn),
                daemon=True,
            )
            proc.start()
            child_conn.close()  # parent does not need the child's end
            self._processes[i] = (proc, parent_conn)

    # ── continuous execution (collection) ────────────────────────

    async def run_continuously(self,
                               agent_manager: EnhancedAgentManager,
                               on_game_done: Optional[Callable] = None) -> None:
        """Run games back-to-back, spawning a new one as soon as one finishes."""
        self._start_workers()
        self._tasks.clear()

        env_tasks = []
        for env_id, (proc, conn) in self._processes.items():
            t = asyncio.create_task(
                self._handle_env(env_id, conn, agent_manager, on_game_done)
            )
            self._tasks[env_id] = t
            env_tasks.append(t)

        try:
            await asyncio.gather(*env_tasks)
        except Exception:
            raise
        finally:
            self._cleanup()

    # ── fixed-run execution (evaluation) ─────────────────────────

    async def run_fixed_games(self,
                              agent_manager: EnhancedAgentManager,
                              num_games: int) -> List[GameResult]:
        """Run exactly *num_games* evaluation games and return their results.

        Workers are kept alive across calls (started once, paused between
        batches) instead of being killed and re-created every batch.
        Call :meth:`stop` to release subprocess resources.
        """
        if not self._started:
            self._start_workers()
            self._started = True
        else:
            for env_id, (proc, conn) in self._processes.items():
                try:
                    conn.send(('restart', None))
                except Exception:
                    pass

        self._tasks.clear()

        games_per_worker = num_games // self.num_workers
        remainder = num_games % self.num_workers
        results: List[GameResult] = []
        results_lock = asyncio.Lock()

        async def _eval_handler(result: GameResult):
            async with results_lock:
                results.append(result)

        env_tasks = []
        for env_id, (proc, conn) in self._processes.items():
            count = games_per_worker + (1 if env_id < remainder else 0)
            t = asyncio.create_task(
                self._handle_env_fixed(
                    env_id, conn, agent_manager, count, _eval_handler
                )
            )
            self._tasks[env_id] = t
            env_tasks.append(t)

        try:
            await asyncio.gather(*env_tasks)
        except Exception:
            raise

        return results

    # ── per-env async handlers ───────────────────────────────────

    async def _handle_env(self,
                          env_id: int,
                          conn: mp.connection.Connection,
                          agent_manager: EnhancedAgentManager,
                          on_game_done: Optional[Callable] = None) -> None:
        """Continuously handle messages from *env_id* (collection mode)."""
        game_start_time = time.time()

        while self.should_continue:
            try:
                has_data = conn.poll()
                if not has_data:
                    if not self.should_continue:
                        break
                    await asyncio.sleep(0.005)
                    continue
                if not self.should_continue:
                    conn.send(('stop', None))
                    break

                msg = conn.recv()
            except (EOFError, BrokenPipeError, OSError):
                break

            if msg[0] == 'infer':
                _, observations, rewards, terminated = msg
                try:
                    t0 = time.perf_counter()
                    actions = await agent_manager.get_actions(
                        observations, rewards, terminated,
                        game_id=f"env_{env_id}",
                    )
                    if self.metrics_collector:
                        self.metrics_collector.record("mp_inference_wait", time.perf_counter() - t0)
                    conn.send(('actions', actions))
                except Exception as e:
                    print(f"[MPEnv {env_id}] inference error: {e}")
                    conn.send(('stop', None))
                    break

            elif msg[0] == 'done':
                scores = msg[1]
                round_durations = msg[2] if len(msg) > 2 else []
                game_duration = time.time() - game_start_time

                profiling = getattr(self, '_profiling', None)
                if profiling:
                    profiling.record_game(game_duration)
                    for rd in round_durations:
                        profiling.record_round(rd)

                try:
                    t0 = time.perf_counter()
                    await agent_manager.flush_all_buffers(
                        final_values=scores, game_id=f"env_{env_id}",
                    )
                    if self.metrics_collector:
                        self.metrics_collector.record("mp_flush_buffers", time.perf_counter() - t0)
                    if on_game_done:
                        result = GameResult(
                            game_id=f"env_{env_id}_game",
                            placements={},
                            scores=scores,
                            duration=game_duration,
                            agent_mapping=agent_manager.get_player_agent_mapping(),
                        )
                        await on_game_done(result)
                except Exception as e:
                    print(f"[MPEnv {env_id}] flush error: {e}")

                if self.should_continue and self.should_spawn:
                    with self._game_barrier_counter.get_lock():
                        self._game_barrier_counter.value += 1
                        if self._game_barrier_counter.value >= self.num_workers:
                            self._game_barrier_counter.value = 0
                            self._game_barrier_event.set()
                    while not self._game_barrier_event.is_set():
                        if not self.should_continue:
                            conn.send(('stop', None))
                            return
                        await asyncio.sleep(0.005)
                    self._game_barrier_event.clear()
                    game_start_time = time.time()
                    conn.send(('restart', None))
                else:
                    conn.send(('stop', None))
                    break

    async def _handle_env_fixed(self,
                                env_id: int,
                                conn: mp.connection.Connection,
                                agent_manager: EnhancedAgentManager,
                                num_games: int,
                                on_game_done: Callable) -> None:
        """Handle exactly *num_games* games from *env_id* (eval mode)."""
        games_done = 0
        game_start_time = time.time()

        while games_done < num_games:
            try:
                has_data = conn.poll()
                if not has_data:
                    await asyncio.sleep(0.005)
                    continue
                msg = conn.recv()
            except (EOFError, BrokenPipeError, OSError):
                break

            if msg[0] == 'infer':
                _, observations, rewards, terminated = msg
                try:
                    actions = await agent_manager.get_actions(
                        observations, rewards, terminated,
                        game_id=f"eval_env_{env_id}_{games_done}",
                    )
                    conn.send(('actions', actions))
                except Exception:
                    conn.send(('stop', None))
                    break

            elif msg[0] == 'done':
                scores = msg[1]
                round_durations = msg[2] if len(msg) > 2 else []
                game_duration = time.time() - game_start_time

                profiling = getattr(self, '_profiling', None)
                if profiling:
                    profiling.record_game(game_duration)
                    for rd in round_durations:
                        profiling.record_round(rd)

                try:
                    await agent_manager.flush_all_buffers(
                        final_values=scores,
                        game_id=f"eval_env_{env_id}_{games_done}",
                    )
                    sorted_players = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                    placements = {pid: i + 1 for i, (pid, _) in enumerate(sorted_players)}
                    result = GameResult(
                        game_id=f"eval_env_{env_id}_{games_done}",
                        placements=placements,
                        scores=scores,
                        duration=game_duration,
                        agent_mapping=agent_manager.get_player_agent_mapping(),
                    )
                    await on_game_done(result)
                except Exception:
                    pass

                games_done += 1
                if games_done < num_games:
                    game_start_time = time.time()
                    conn.send(('restart', None))
                else:
                    conn.send(('pause', None))
                    break

    # ── cleanup ──────────────────────────────────────────────────

    def _cleanup(self):
        """Terminate all subprocesses and close connections (non-blocking).

        Sends a ``('stop', None)`` signal to each worker (waking paused ones)
        before closing the pipe and killing the process.
        """
        for env_id, (proc, conn) in self._processes.items():
            try:
                conn.send(('stop', None))
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
            try:
                proc.kill()
            except Exception:
                pass
        self._processes.clear()
        self._tasks.clear()
        self._started = False


# ---------------------------------------------------------------------------
# TrainingOrchestrator
# ---------------------------------------------------------------------------

class TrainingOrchestrator:
    """
    Central orchestrator that drives the RL lifecycle:

        Collect -> Train -> Sync -> Evaluate

    Usage::

        orch = TrainingOrchestrator(config)
        orch.setup()
        await orch.run(max_steps=100_000)

    For single-episode debugging::

        result = await orch.run_single_episode()

    For a parallel demo (no training) see :meth:`run_parallel_demo`.
    """

    def __init__(self, training_config: Optional[TrainingConfig] = None):
        self.cfg = training_config or TrainingConfig()

        self.profiling = ProfilingTracker()

        # Components (created in setup)
        self.trainer: Optional[Trainer] = None
        self.global_buffer: Optional[GlobalBuffer] = None
        self.agent_manager: Optional[EnhancedAgentManager] = None
        self.env_manager: Any = None
        self.summary_writer: Optional[SummaryWriter] = None

        # Training state
        self.training_step: int = self.cfg.starting_train_step
        self.games_completed: int = 0
        self.training_active: bool = False

        # Model / weights
        self.best_model: Optional[MuZeroAgent] = None
        self.current_model: Optional[MuZeroAgent] = None
        self._training_agents: List[MuZeroAgent] = []

        # Profiling
        self.metrics_collector = MetricsCollector(window_size=2000)
        self.benchmark: Optional[EnvironmentBenchmark] = None
        self._last_benchmark_step: int = 0
        self._last_metric_log_step: int = 0

        # Dedicated single-thread executor for training steps, isolated from
        # the inference server's executor to avoid state corruption.
        from concurrent.futures import ThreadPoolExecutor
        self._train_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, is_collector: bool = False, is_evaluator: bool = False):
        """Create all components: buffer, agents, batch processor, trainer."""
        self.trainer = Trainer()
        self.summary_writer = self._build_logger()

        # Use lightweight WorkerGlobalBuffer if running as a worker process
        if is_collector or is_evaluator:
            from Models.global_buffer import WorkerGlobalBuffer
            self.global_buffer = WorkerGlobalBuffer(action_to_policy=action_3d_to_policy)
        else:
            self.global_buffer = GlobalBuffer(config.BATCH_SIZE, action_to_policy=action_3d_to_policy)

        # --- agent config -------------------------------------------------
        if not is_collector:
            # best_model: the best performing model — only updated when evaluation beats it
            self.best_model = MuZeroAgent(
                action_size=3,
                action_limits=config.ACTION_DIM,
                obs_size=config.OBSERVATION_SIZE,
                simulations=config.NUM_SIMULATIONS,
                global_buffer=self.global_buffer,
                config_obj=self.cfg,
            )

            # current_model: the model actively being trained
            self.current_model = MuZeroAgent(
                action_size=3,
                action_limits=config.ACTION_DIM,
                obs_size=config.OBSERVATION_SIZE,
                simulations=config.NUM_SIMULATIONS,
                global_buffer=self.global_buffer,
                weights=copy.deepcopy(self.best_model.get_weights()),
                config_obj=self.cfg,
            )

            if self.training_step > 0:
                ckpt = f"./checkpoint/best_{self.training_step}"
                if os.path.isfile(ckpt):
                    state = torch.load(ckpt)
                    self.best_model.model.load_state_dict(state)
                    self.current_model.model.load_state_dict(state)

        # Evaluators do NOT need collection agents, agent manager, env manager, or benchmarking!
        if is_evaluator:
            print("Evaluator worker setup complete (skipping collection agent and environment manager).")
            return

        # MuZero agents for *collection* – start with best model weights
        if is_collector:
            collection_agent = MuZeroAgent(
                action_size=3,
                action_limits=config.ACTION_DIM,
                obs_size=config.OBSERVATION_SIZE,
                simulations=config.NUM_SIMULATIONS,
                global_buffer=self.global_buffer,
                config_obj=self.cfg,
            )
        else:
            collection_agent = MuZeroAgent(
                action_size=3,
                action_limits=config.ACTION_DIM,
                obs_size=config.OBSERVATION_SIZE,
                simulations=config.NUM_SIMULATIONS,
                global_buffer=self.global_buffer,
                weights=copy.deepcopy(self.best_model.get_weights()),
                config_obj=self.cfg,
            )
        self._training_agents = [collection_agent]

        agent_configs: List[Tuple[Any, int]] = [
            (collection_agent, 8)
        ]

        # --- batch processor + agent manager -------------------------------
        self.agent_manager, _ = create_custom_agent_setup(
            agent_configs,
            max_batch_size=self.cfg.max_batch_size,
            batch_timeout_ms=self.cfg.batch_timeout_ms,
            gpu_memory_fraction=self.cfg.gpu_memory_fraction,
            metrics_collector=self.metrics_collector,
        )

        # --- parallel env manager -----------------------------------------
        self.env_manager = self._create_env_manager(
            self.cfg.concurrent_games,
            profiling=self.profiling,
            metrics_collector=self.metrics_collector
        )

        # --- benchmark ----------------------------------------------------
        self.benchmark = EnvironmentBenchmark(parallel_env)
        self._run_benchmark()

        print(f"TrainingOrchestrator setup complete:")
        print(f"  Concurrent games : {self.cfg.concurrent_games}")
        print(f"  Batch size       : {self.cfg.max_batch_size}")
        print(f"  Training step    : {self.training_step}")
        print(f"  GPU available    : {torch.cuda.is_available()}")

    def _run_benchmark(self, num_steps: int = 200):
        if self.benchmark is None:
            return
        try:
            results = self.benchmark.run(num_steps=num_steps)
            s = results.summary()
            print(f"[Benchmark] env step: avg={s['step_time_avg_ms']:.2f}ms "
                  f"median={s['step_time_median_ms']:.2f}ms "
                  f"min={s['step_time_min_ms']:.2f}ms max={s['step_time_max_ms']:.2f}ms "
                  f"reset={s['reset_time_ms']:.2f}ms steps={s['num_steps']}")
            if self.summary_writer:
                for key, val in s.items():
                    self.summary_writer.add_scalar(f"benchmark/{key}", val, self.training_step)
        except Exception as e:
            print(f"[Benchmark] error: {e}")

    def _log_metrics(self):
        stats = self.metrics_collector.all_stats()
        if not stats:
            return
        print("[MetricsCollector] === Performance Metrics ===")
        for name, s in stats.items():
            line = (f"  {name}: mean={s['mean_ms']:.2f}ms median={s['median_ms']:.2f}ms "
                    f"min={s['min_ms']:.2f}ms max={s['max_ms']:.2f}ms count={s['count']}")
            print(line)
            if self.summary_writer and self.training_step > 0:
                for key in ('mean_ms', 'median_ms', 'min_ms', 'max_ms'):
                    self.summary_writer.add_scalar(f"metrics/{name}_{key}", s[key], self.training_step)

    def _build_logger(self) -> SummaryWriter:
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = f"logs/{self.cfg.run_name}{ts}"
        return SummaryWriter(log_dir)

    # ------------------------------------------------------------------
    # 1️⃣  COLLECT phase
    # ------------------------------------------------------------------

    async def collect(self) -> None:
        """
        Start the sequential Collect -> Train loop.
        Runs until :meth:`stop_training` is called.
        """
        self.training_active = True
        print("COLLECT phase started – sequential Collect -> Train loop active.")

        while self.training_active:
            results = []
            remaining = self.cfg.collect_games_per_batch
            while remaining > 0 and self.training_active:
                count = min(remaining, self.cfg.concurrent_games)
                batch = await self.env_manager.run_fixed_games(
                    self.agent_manager, count
                )
                results.extend(batch)
                remaining -= count
            self.games_completed += len(results)

            if not self.training_active:
                break

            trained = False
            while self.training_active and self.global_buffer.available_gameplay_batch():
                t0 = time.time()
                await self._train_step()
                self.profiling.record_train_step(time.time() - t0)
                await asyncio.sleep(0.01)
                trained = True

            if not trained:
                await asyncio.sleep(1.0)

    async def _training_loop(self) -> None:
        """
        Dedicated, non-blocking background task that handles continuous 
        training from the GlobalBuffer.
        """
        print("TRAIN phase started – background training loop active.")
        while self.training_active:
            if self.global_buffer and self.global_buffer.available_gameplay_batch():
                t0 = time.time()
                await self._train_step()
                self.profiling.record_train_step(time.time() - t0)
                await asyncio.sleep(0.01)
            else:
                t0 = time.time()
                await asyncio.sleep(0.5)
                self.profiling.record_idle(time.time() - t0)

    # ------------------------------------------------------------------
    # 2️⃣  TRAIN phase
    # ------------------------------------------------------------------

    async def _train_step(self) -> None:
        """Perform a single training step (called automatically during collect)."""
        try:
            if not self.global_buffer or not self.global_buffer.available_gameplay_batch():
                return
    
            batch = self.global_buffer.read_gameplay_batch()
            combat_batch = []
            if hasattr(self.global_buffer, "available_combat_batch") and self.global_buffer.available_combat_batch():
                cb = self.global_buffer.read_combat_batch()
                if cb is not None:
                    combat_batch = cb
    
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self._train_executor,
                self.trainer.train_network,
                batch,
                combat_batch,
                self.current_model.model,
                self.training_step,
                self.summary_writer,
            )
            self.training_step += 1

            # Periodic saving of current model
            if self.training_step % self.cfg.sync_steps == 0:
                self.save_current_checkpoint()
    
            # Periodic benchmarking (every 200 steps)
            if self.training_step - self._last_benchmark_step >= 200:
                self._last_benchmark_step = self.training_step
                self._run_benchmark(num_steps=100)

            # Periodic metrics logging (every 100 steps)
            if self.training_step - self._last_metric_log_step >= 100:
                self._last_metric_log_step = self.training_step
                self._log_metrics()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error in _train_step: {e}")
            raise

    def train_step(self) -> bool:
        """
        Explicit TRAIN step.  Useful when driving the lifecycle manually.
        Returns True if training actually occurred.
        """
        if not self.global_buffer or not self.global_buffer.available_gameplay_batch():
            return False
        asyncio.create_task(self._train_step())
        return True

    # ------------------------------------------------------------------
    # 3️⃣  SYNC phase
    # ------------------------------------------------------------------

    def sync_weights(self) -> None:
        """
        Distribute the latest trained weights from the best model to the
        active collection agents so they generate data using the best policy.
        """
        if not self.best_model:
            return
        new_weights = self.best_model.get_weights()
        for agent in self._training_agents:
            agent.update_weights(new_weights)
        print(f"SYNC: distributed BEST weights to {len(self._training_agents)} agent(s)")

    # ------------------------------------------------------------------
    # 4️⃣  EVALUATE phase
    # ------------------------------------------------------------------

    async def evaluate(self, step: Optional[int] = None) -> Dict[str, float]:
        """
        Run evaluation games between the current (new) model and the
        best model so far.  Keep the better-performing weights.

        Parameters
        ----------
        step:
            The training step from the server to use for logging.
            Falls back to ``self.training_step`` when not provided.

        Returns a dict with ``current_placement`` and ``best_placement``.
        """
        current_step = step if step is not None else self.training_step
        print(f"\nEVALUATE at step {current_step}")

        eval_current = MuZeroAgent(
            action_size=3,
            action_limits=config.ACTION_DIM,
            obs_size=config.OBSERVATION_SIZE,
            simulations=config.NUM_SIMULATIONS,
            global_buffer=None,
            weights=copy.deepcopy(self.current_model.get_weights()),
            config_obj=self.cfg,
            training=False,
        )
        eval_best = MuZeroAgent(
            action_size=3,
            action_limits=config.ACTION_DIM,
            obs_size=config.OBSERVATION_SIZE,
            simulations=config.NUM_SIMULATIONS,
            global_buffer=None,
            weights=copy.deepcopy(self.best_model.get_weights()),
            config_obj=self.cfg,
            training=False,
        )
        random_agent = RandomAgent("EvalRandom")
        cultist_agent = CultistAgent()
        divine_agent = DivineAgent()
        warlord_agent = WarlordAgent()

        eval_configs = [
            (eval_current, 1),
            (eval_best, 1),
            (random_agent, 3),
            (cultist_agent, 1),
            (divine_agent, 1),
            (warlord_agent, 1)
        ]
        eval_mgr, _ = create_custom_agent_setup(
            eval_configs,
            max_batch_size=self.cfg.max_batch_size,
            batch_timeout_ms=self.cfg.batch_timeout_ms,
            gpu_memory_fraction=self.cfg.gpu_memory_fraction,
        )

        eval_env_mgr = self._create_env_manager(self.cfg.evaluation_concurrent, profiling=self.profiling)
        results = await eval_env_mgr.run_fixed_games(eval_mgr, self.cfg.evaluation_games)

        current_placements, best_placements = [], []
        for r in results:
            mapping = r.agent_mapping
            for pid, placement in r.placements.items():
                at = mapping.get(pid)
                if at is eval_current:
                    current_placements.append(placement)
                elif at is eval_best:
                    best_placements.append(placement)

        current_mean = float(np.mean(current_placements)) if current_placements else 8.0
        best_mean = float(np.mean(best_placements)) if best_placements else 8.0

        if self.summary_writer:
            self.summary_writer.add_scalar("evaluation/current_model", current_mean, current_step)
            self.summary_writer.add_scalar("evaluation/best_model", best_mean, current_step)

        print(f"  Current model placement: {current_mean:.2f}  |  Best model: {best_mean:.2f}")

        if current_mean < best_mean:
            print("  ✓ Model improved – updating best model & clearing buffers.")
            self.best_model.model.load_state_dict(self.current_model.get_weights())
            self.sync_weights()
            self.save_best_checkpoint()
            if self.global_buffer:
                if hasattr(self.global_buffer, "clear_gameplay_buffer"):
                    self.global_buffer.clear_gameplay_buffer()

        # Clean up temporary evaluation agents and environment manager to prevent asyncio task leaks
        eval_mgr.shutdown()
        eval_env_mgr.stop()

        return {"current_placement": current_mean, "best_placement": best_mean}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_current_checkpoint(self) -> None:
        """Save the current (actively-trained) model to disk."""
        os.makedirs("./checkpoint", exist_ok=True)
        path = f"./checkpoint/current_{self.training_step}"
        if self.current_model is not None:
            torch.save(self.current_model.model.state_dict(), path)
            print(f"Current checkpoint saved at step {self.training_step}")

    def save_best_checkpoint(self) -> None:
        """Save the best model to disk (only called when it improves)."""
        os.makedirs("./checkpoint", exist_ok=True)
        path = f"./checkpoint/best_{self.training_step}"
        if self.best_model is not None:
            torch.save(self.best_model.model.state_dict(), path)
            print(f"Best checkpoint saved at step {self.training_step}")

    def load_checkpoint(self, step: int) -> bool:
        best_path = f"./checkpoint/best_{step}"
        if not os.path.isfile(best_path):
            return False
        state = torch.load(best_path)
        if self.best_model is not None:
            self.best_model.model.load_state_dict(state)
        if self.current_model is not None:
            self.current_model.model.load_state_dict(state)
        self.training_step = step
        return True

    # ------------------------------------------------------------------
    # High-level modes
    # ------------------------------------------------------------------

    async def run(self, max_steps: int = 1_000_000) -> None:
        """
        Full training loop orchestrating all lifecycle phases.

        Sequential Collect -> Train loop:
        1. Collect data by running concurrent games.
        2. Train on the collected data.
        3. Repeat.
        """
        if self.env_manager is None:
            self.setup()

        self.training_active = True
        self.num_evaluations = self.training_step // self.cfg.evaluation_interval

        last_logged_step = -1
        try:
            while self.training_active and self.training_step < max_steps:
                results = []
                remaining = self.cfg.collect_games_per_batch
                while remaining > 0 and self.training_active and self.training_step < max_steps:
                    count = min(remaining, self.cfg.concurrent_games)
                    batch = await self.env_manager.run_fixed_games(
                        self.agent_manager, count
                    )
                    results.extend(batch)
                    remaining -= count
                self.games_completed += len(results)
                print(f'Finished {self.games_completed} games.')

                if not self.training_active or self.training_step >= max_steps:
                    break

                trained = False
                while (self.training_active and self.training_step < max_steps
                       and self.global_buffer.available_gameplay_batch()):
                    t0 = time.time()
                    await self._train_step()
                    self.profiling.record_train_step(time.time() - t0)
                    await asyncio.sleep(0.01)
                    trained = True

                if trained and self.training_step - (self.num_evaluations * self.cfg.evaluation_interval) > self.cfg.evaluation_interval:
                    self.env_manager.pause()
                    await self.env_manager.wait_for_drain()
                    await self.evaluate()
                    self.env_manager.resume()
                    self.num_evaluations += 1


                if not trained:
                    await asyncio.sleep(1.0)

                if self.training_step % 100 == 0 and self.training_step > 0 and self.training_step != last_logged_step:
                    print(f"  step={self.training_step}  games={self.games_completed}")
                    last_logged_step = self.training_step
        except asyncio.CancelledError:
            pass
        finally:
            self.training_active = False
            if self.summary_writer:
                self.summary_writer.close()

    def stop_training(self):
        """Gracefully stop the training loop."""
        self.training_active = False
        if hasattr(self, '_train_executor'):
            self._train_executor.shutdown(wait=False)
        if self.env_manager:
            self.env_manager.stop()

    async def run_single_episode(self) -> GameResult:
        """
        Run a single episode for debugging.

        Replaces the old ``train_single.py`` workflow.
        """
        if self.agent_manager is None:
            self.setup()
        worker = _GameWorker(0, profiling=self.profiling, metrics_collector=self.metrics_collector)
        return await worker.run_game(self.agent_manager, return_placements=True)

    async def run_parallel_demo(self, num_episodes: int = 5) -> List[GameResult]:
        """
        Run *num_episodes* games in parallel.

        Replaces the old ``train_parallel.py`` demo workflow.
        """
        if self.agent_manager is None:
            self.setup()
        mgr = self._create_env_manager(
            min(self.cfg.concurrent_games, num_episodes),
            profiling=self.profiling,
            metrics_collector=self.metrics_collector
        )
        return await mgr.run_fixed_games(self.agent_manager, num_episodes)

    async def run_evaluation(self, num_games: int) -> List[GameResult]:
        """Run a standalone evaluation session."""
        if self.agent_manager is None:
            self.setup()
        mgr = self._create_env_manager(
            self.cfg.evaluation_concurrent,
            profiling=self.profiling,
            metrics_collector=self.metrics_collector
        )
        return await mgr.run_fixed_games(self.agent_manager, num_games)

    @staticmethod
    def _create_env_manager(num_workers: int,
                            profiling: Optional[ProfilingTracker] = None,
                            metrics_collector: Optional[MetricsCollector] = None):
        """Factory: returns _MultiProcessEnvManager (process-level isolation
        required by TFTSet4Gym's global state).
        
        When *profiling* is provided the returned manager will record per-step
        timings for environment stepping and inference wait."""
        mgr = _MultiProcessEnvManager(num_workers, metrics_collector=metrics_collector)
        mgr._profiling = profiling
        return mgr

    def cleanup(self):
        """Release resources (writer, etc.)."""
        if self.summary_writer:
            self.summary_writer.close()
        self.training_active = False

    def print_profiling_summary(self) -> Dict[str, float]:
        """Print a detailed performance breakdown and return the summary dict."""
        s = self.profiling.summary()
        print("\n" + "=" * 60)
        print("PERFORMANCE BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"  {'Metric':<35} {'Total (s)':>12} {'Count':>8} {'Avg (ms)':>10}")
        print("  " + "-" * 65)
        if s["inference_count"] > 0:
            print(f"  {'Inference wait time':<35} {s['inference_wait_time']:>12.3f} {s['inference_count']:>8} {s['avg_inference_wait']*1000:>10.2f}")
        if s["env_step_count"] > 0:
            print(f"  {'Environment stepping time':<35} {s['env_step_time']:>12.3f} {s['env_step_count']:>8} {s['avg_env_step']*1000:>10.2f}")
        print(f"  {'Training step time':<35} {s['train_time']:>12.3f} {s['train_step_count']:>8} {s['avg_train_step']*1000:>10.2f}")
        print(f"  {'Idle time':<35} {s['idle_time']:>12.3f} {'':>8} {'':>10}")
        if s.get("round_count", 0) > 0:
            print(f"  {'Round time':<35} {'':>12} {s['round_count']:>8} {s['avg_round_time']*1000:>10.2f}")
        if s.get("game_count", 0) > 0:
            print(f"  {'Game time':<35} {'':>12} {s['game_count']:>8} {s['avg_game_time']*1000:>10.2f}")
        print("  " + "-" * 65)
        print(f"  {'TOTAL':<35} {s['total_time']:>12.3f} {'':>8} {'':>10}")
        print()
        print(f"  {'Category':<35} {'Percentage':>12}")
        print("  " + "-" * 47)
        if s["inference_count"] > 0:
            print(f"  {'Inference wait':<35} {s['inference_pct']:>11.1f}%")
        if s["env_step_count"] > 0:
            print(f"  {'Environment stepping':<35} {s['env_step_pct']:>11.1f}%")
        print(f"  {'Training':<35} {s['train_pct']:>11.1f}%")
        print(f"  {'Idle':<35} {s['idle_pct']:>11.1f}%")
        print("=" * 60)

        # Also include agent-level inference stats from the batch processor if available
        if self.agent_manager is not None:
            agent_stats = self.agent_manager.get_performance_stats()
            if agent_stats:
                print(f"\n  Agent-level inference stats:")
                for name, st in agent_stats.items():
                    print(f"    {name:<30} avg={st.get('avg_inference_time', 0)*1000:.2f}ms  "
                          f"batches={st.get('total_inferences', 0)}  "
                          f"avg_batch={st.get('avg_batch_size', 0):.1f}")

        return s


# ---------------------------------------------------------------------------
# Convenience factory / helpers
# ---------------------------------------------------------------------------

def create_orchestrator(config: Optional[TrainingConfig] = None) -> TrainingOrchestrator:
    return TrainingOrchestrator(config)


async def quick_evaluation(num_games: int = 8, concurrent: int = 2) -> List[GameResult]:
    """Quick evaluation with default agents."""
    cfg = TrainingConfig(
        concurrent_games=concurrent,
        evaluation_games=num_games,
        evaluation_concurrent=concurrent,
        max_batch_size=8,
    )
    orch = TrainingOrchestrator(cfg)
    orch.setup()
    mgr = TrainingOrchestrator._create_env_manager(concurrent)
    results = await mgr.run_fixed_games(orch.agent_manager, num_games)

    agent_stats: Dict[str, List[int]] = defaultdict(list)
    for r in results:
        for pid, placement in r.placements.items():
            at = r.agent_mapping.get(pid)
            name = at.__name__ if at and hasattr(at, "__name__") else str(at)
            agent_stats[name].append(placement)

    print(f"\n=== Evaluation Summary ({len(results)} games) ===")
    for name, placements in agent_stats.items():
        print(f"  {name}: {np.mean(placements):.2f} avg placement ({len(placements)} games)")
    return results
