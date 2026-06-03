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
from dataclasses import dataclass
from collections import defaultdict
import torch
from torch.utils.tensorboard import SummaryWriter

import config
from Models.global_buffer import GlobalBuffer
from Models.action_conversion import action_3d_to_policy
from Models.MuZero_torch_trainer import Trainer
from Models.MuZero_torch_agent import MuZeroAgent
from Models.Common_agents import CultistAgent, DivineAgent, RandomAgent, WarlordAgent
from Models.enhanced_agent_interface import (
    create_enhanced_setup,
    create_custom_agent_setup,
    AsyncGameEnvironment,
    BatchInferenceServer,
    EnhancedAgentManager,
)
from TFTSet4Gym.tft_set4_gym.tft_simulator import parallel_env

import multiprocessing as mp
import threading

# Default multiprocessing context (fork on Linux 3.13-).
# Subprocesses only run CPU-bound env logic and do not touch GPU tensors,
# so fork is safe and avoids re-import overhead.
MP_CONTEXT = mp


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

    env = parallel_env(rank=env_id)

    while True:
        observations = env.reset()[0]
        terminated = {pid: False for pid in env.possible_agents}
        rewards = {pid: 0.0 for pid in env.possible_agents}
        scores = {pid: 0.0 for pid in env.possible_agents}
        step_count = 0

        while not all(terminated.values()):
            float_rewards = {k: float(v) for k, v in rewards.items()}

            conn.send(('infer', observations, float_rewards, terminated))

            msg = conn.recv()
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

            observations, rewards, terminated, _, _ = env.step(processed)
            for p in terminated:
                if terminated[p]:
                    scores[p] = rewards[p]

            step_count += 1
            if step_count > 1000:
                break

        conn.send(('done', scores))

        msg = conn.recv()
        if msg[0] == 'stop':
            return
        # 'restart' → fall through to outer loop


# ---------------------------------------------------------------------------
# Thread worker – runs a game in a thread, bridges to the main event loop
# ---------------------------------------------------------------------------

def _thread_worker_main(env_id: int, loop: asyncio.AbstractEventLoop,
                        agent_manager: 'EnhancedAgentManager',
                        stop_event: threading.Event,
                        pause_event: threading.Event,
                        on_game_done_callback=None,
                        games_to_play: Optional[int] = None):
    """
    Target function for an environment **thread**.

    Runs a continuous game loop in a dedicated thread, bridging to the
    main asyncio event loop via :func:`asyncio.run_coroutine_threadsafe`::

        actions = asyncio.run_coroutine_threadsafe(
            agent_manager.get_actions(...), loop
        ).result()

    This allows synchronous thread-based game workers to use the shared
    async agent manager without pickling or pipe communication.

    Parameters
    ----------
    env_id:
        Rank/environment identifier.
    loop:
        The main asyncio event loop (obtained via ``get_event_loop()``).
    agent_manager:
        Shared async agent manager — all inference is scheduled on *loop*.
    stop_event:
        Set by the manager to request a clean shutdown.
    pause_event:
        Set by the manager to pause spawning new games.
    on_game_done_callback:
        Optional async callback invoked after each completed game.
        Called via ``run_coroutine_threadsafe`` so it runs on the main loop.
    games_to_play:
        If given, the worker exits after this many games (fixed-run /
        evaluation mode).  ``None`` means run forever (collection mode).
    """
    _root = os.path.dirname(os.path.abspath(__file__))
    if _root not in sys.path:
        sys.path.insert(0, _root)

    from TFTSet4Gym.tft_set4_gym.tft_simulator import parallel_env

    env = parallel_env(rank=env_id)
    games_done = 0

    while not stop_event.is_set():
        if games_to_play is not None and games_done >= games_to_play:
            break

        # Respect pause — block here while the manager drains active games
        while pause_event.is_set() and not stop_event.is_set():
            time.sleep(0.1)
        if stop_event.is_set():
            break

        observations = env.reset()[0]
        terminated = {pid: False for pid in env.possible_agents}
        rewards = {pid: 0.0 for pid in env.possible_agents}
        scores = {pid: 0.0 for pid in env.possible_agents}
        step_count = 0

        while not all(terminated.values()):
            if stop_event.is_set():
                return

            step_count += 1
            float_rewards = {k: float(v) for k, v in rewards.items()}

            future = asyncio.run_coroutine_threadsafe(
                agent_manager.get_actions(
                    observations, float_rewards, terminated,
                    game_id=f"thread_env_{env_id}",
                ),
                loop,
            )
            try:
                actions = future.result(timeout=30.0)
            except Exception as e:
                print(f"[ThreadEnv {env_id}] inference error: {e}")
                return

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

            observations, rewards, terminated, _, _ = env.step(processed)
            for p in terminated:
                if terminated[p]:
                    scores[p] = rewards[p]

            if step_count > 1000:
                break

        # ── game finished ─────────────────────────────────────────
        future = asyncio.run_coroutine_threadsafe(
            agent_manager.flush_all_buffers(
                final_values=scores, game_id=f"thread_env_{env_id}",
            ),
            loop,
        )
        try:
            future.result(timeout=10.0)
        except Exception:
            pass

        if on_game_done_callback:
            sorted_players = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            placements = {pid: i + 1 for i, (pid, _) in enumerate(sorted_players)}
            result = GameResult(
                game_id=f"thread_env_{env_id}_{games_done}",
                placements=placements,
                scores=scores,
                duration=0.0,
                agent_mapping=agent_manager.get_player_agent_mapping(),
            )
            future = asyncio.run_coroutine_threadsafe(
                on_game_done_callback(result), loop,
            )
            try:
                future.result(timeout=10.0)
            except Exception:
                pass

        games_done += 1


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


# ---------------------------------------------------------------------------
# Game worker – runs one async game
# ---------------------------------------------------------------------------

class _GameWorker:
    """
    Async game worker that replaces the legacy Ray DataWorker.
    Runs a single game asynchronously without Ray overhead.
    """

    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.games_completed = 0

    async def run_game(self,
                       agent_manager: EnhancedAgentManager,
                       return_placements: bool = False) -> GameResult:
        """Run a single game end-to-end and return the result."""
        try:
            game_id = f"worker_{self.worker_id}_game_{self.games_completed}"
            start_time = time.time()

            env = parallel_env(rank=self.worker_id)
            observations = env.reset()[0]
            terminated = {pid: False for pid in env.possible_agents}
            rewards = {pid: 0.0 for pid in env.possible_agents}
            scores = {pid: 0.0 for pid in env.possible_agents}

            step_count = 0
            while not all(terminated.values()):
                step_count += 1
                float_rewards = {k: float(v) for k, v in rewards.items()}

                actions_task = agent_manager.get_actions(
                    observations, float_rewards, terminated, game_id=game_id
                )
                actions = await asyncio.wait_for(actions_task, timeout=30.0)

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

                observations, rewards, terminated, _, _ = env.step(actions)
                for p in terminated:
                    if terminated[p]:
                        scores[p] = rewards[p]

                if step_count > 1000:
                    break

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
# Parallel environment manager
# ---------------------------------------------------------------------------

class _ParallelEnvManager:
    """
    Manages N concurrent game workers without Ray.
    Supports continuous execution (auto-restart) and fixed-run evaluation.
    """

    def __init__(self, num_workers: int):
        self.num_workers = num_workers
        self.workers = [_GameWorker(i) for i in range(num_workers)]
        self.active_tasks: Dict[asyncio.Task, int] = {}
        self.should_continue = True
        self.should_spawn = True

    def stop(self):
        self.should_continue = False

    def pause(self):
        """Prevent spawning new games; let running ones drain naturally."""
        self.should_spawn = False

    def resume(self):
        """Allow spawning new games again."""
        self.should_spawn = True

    async def wait_for_drain(self):
        """Wait until all active game tasks have finished."""
        while self.active_tasks:
            await asyncio.sleep(0.5)

    async def run_continuously(self,
                               agent_manager: EnhancedAgentManager,
                               on_game_done: Optional[Callable] = None) -> None:
        """Run games back-to-back, starting a new one as soon as one finishes."""
        self.active_tasks.clear()
        for i, worker in enumerate(self.workers):
            task = asyncio.create_task(worker.run_game(agent_manager))
            self.active_tasks[task] = i

        games_completed = 0
        while self.should_continue:
            if not self.active_tasks:
                if not self.should_spawn:
                    await asyncio.sleep(0.1)
                    continue
                for i, w in enumerate(self.workers):
                    task = asyncio.create_task(w.run_game(agent_manager))
                    self.active_tasks[task] = i
                continue

            try:
                done, _ = await asyncio.wait(
                    self.active_tasks.keys(),
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=1.0,
                )
            except asyncio.TimeoutError:
                continue

            for completed in done:
                worker_id = self.active_tasks.pop(completed, -1)
                self._discard(completed)
                try:
                    result = await completed
                    games_completed += 1
                    if on_game_done:
                        await on_game_done(result)
                except Exception as e:
                    print(f"Game worker {worker_id} crashed: {e}")
                    raise e

                if worker_id >= 0 and self.should_spawn:
                    new = asyncio.create_task(self.workers[worker_id].run_game(agent_manager))
                    self.active_tasks[new] = worker_id

        for t in list(self.active_tasks):
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        self.active_tasks.clear()

    async def run_fixed_games(self,
                              agent_manager: EnhancedAgentManager,
                              num_games: int) -> List[GameResult]:
        """Run exactly *num_games* evaluation games and return their results."""
        games_per_worker = num_games // self.num_workers
        remainder = num_games % self.num_workers
        tasks = []
        for i, worker in enumerate(self.workers):
            count = games_per_worker + (1 if i < remainder else 0)
            for _ in range(count):
                tasks.append(asyncio.create_task(worker.run_game(agent_manager, return_placements=True)))

        results = []
        if tasks:
            completed = await asyncio.gather(*tasks, return_exceptions=True)
            for r in completed:
                if isinstance(r, GameResult):
                    results.append(r)
        return results

    def _discard(self, task):
        self.active_tasks.pop(task, None)


# ---------------------------------------------------------------------------
# Multi-process environment manager (bypasses GIL)
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

    def __init__(self, num_workers: int, worker_fn=None):
        self.num_workers = num_workers
        self._worker_fn = worker_fn or _env_worker_main
        self._processes: Dict[int, Tuple[mp.Process, mp.connection.Connection]] = {}
        self._tasks: Dict[int, asyncio.Task] = {}
        self.should_continue = True
        self.should_spawn = True

    # ── lifecycle ────────────────────────────────────────────────

    def stop(self):
        self.should_continue = False

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
        """Run exactly *num_games* evaluation games and return their results."""
        self._start_workers()
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
        finally:
            self._cleanup()

        return results

    # ── per-env async handlers ───────────────────────────────────

    async def _handle_env(self,
                          env_id: int,
                          conn: mp.connection.Connection,
                          agent_manager: EnhancedAgentManager,
                          on_game_done: Optional[Callable] = None) -> None:
        """Continuously handle messages from *env_id* (collection mode)."""
        loop = asyncio.get_event_loop()

        while self.should_continue:
            try:
                # poll with timeout so we can check should_continue regularly
                has_data = await loop.run_in_executor(
                    None, lambda: conn.poll(0.5)
                )
                if not has_data:
                    if not self.should_continue:
                        break
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
                    actions = await agent_manager.get_actions(
                        observations, rewards, terminated,
                        game_id=f"env_{env_id}",
                    )
                    conn.send(('actions', actions))
                except Exception as e:
                    print(f"[MPEnv {env_id}] inference error: {e}")
                    conn.send(('stop', None))
                    break

            elif msg[0] == 'done':
                _, scores = msg
                # flush agent buffers for this finished game
                try:
                    await agent_manager.flush_all_buffers(
                        final_values=scores, game_id=f"env_{env_id}",
                    )
                    if on_game_done:
                        result = GameResult(
                            game_id=f"env_{env_id}_game",
                            placements={},
                            scores=scores,
                            duration=0.0,
                            agent_mapping=agent_manager.get_player_agent_mapping(),
                        )
                        await on_game_done(result)
                except Exception as e:
                    print(f"[MPEnv {env_id}] flush error: {e}")

                if self.should_continue and self.should_spawn:
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
        loop = asyncio.get_event_loop()
        games_done = 0

        while games_done < num_games:
            try:
                has_data = await loop.run_in_executor(
                    None, lambda: conn.poll(0.5)
                )
                if not has_data:
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
                _, scores = msg
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
                        duration=0.0,
                        agent_mapping=agent_manager.get_player_agent_mapping(),
                    )
                    await on_game_done(result)
                except Exception:
                    pass

                games_done += 1
                if games_done < num_games:
                    conn.send(('restart', None))
                else:
                    conn.send(('stop', None))
                    break

    # ── cleanup ──────────────────────────────────────────────────

    def _cleanup(self):
        """Terminate all subprocesses and close connections."""
        for env_id, (proc, conn) in self._processes.items():
            try:
                conn.close()
            except Exception:
                pass
            try:
                proc.terminate()
                proc.join(timeout=2)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=1)
            except Exception:
                pass
        self._processes.clear()
        self._tasks.clear()


# ---------------------------------------------------------------------------
# Thread-based environment manager (alternative to multiprocess)
# ---------------------------------------------------------------------------

class _ThreadEnvManager:
    """
    Manages N concurrent game workers running in **OS threads**, bridging
    to the main asyncio event loop.

    Each environment runs its game loop in a dedicated thread. When the
    thread needs actions from the agent manager it uses ::

        actions = asyncio.run_coroutine_threadsafe(
            agent_manager.get_actions(...), loop
        ).result()

    so that all GPU-bound inference and experience storage stays on the
    main (async) thread while the CPU-bound game logic runs in parallel.

    This is a drop-in alternative to :class:`_MultiProcessEnvManager` for
    environments where subprocess overhead is undesirable.  The public
    interface intentionally mirrors ``_MultiProcessEnvManager``.
    """

    def __init__(self, num_workers: int, worker_fn=None):
        self.num_workers = num_workers
        self._worker_fn = worker_fn or _thread_worker_main
        self._threads: Dict[int, threading.Thread] = {}
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self.should_continue = True
        self.should_spawn = True

    # ── lifecycle ────────────────────────────────────────────────

    def stop(self):
        """Signal all threads to stop after the current step."""
        self.should_continue = False
        self._stop_event.set()

    def pause(self):
        """Prevent spawning new games; let running ones drain naturally."""
        self.should_spawn = False
        self._pause_event.set()

    def resume(self):
        """Allow spawning new games again."""
        self.should_spawn = True
        self._pause_event.clear()

    async def wait_for_drain(self):
        """Wait until all active thread workers have finished."""
        while any(t.is_alive() for t in self._threads.values()):
            await asyncio.sleep(0.5)

    # ── launch helpers ───────────────────────────────────────────

    def _launch(self, agent_manager: 'EnhancedAgentManager',
                loop: asyncio.AbstractEventLoop,
                on_game_done,
                games_to_play: Optional[int] = None):
        """Start *num_workers* threads, each running ``_worker_fn``."""
        self._stop_event.clear()
        self._pause_event.clear()
        self._threads.clear()

        for i in range(self.num_workers):
            t = threading.Thread(
                target=self._worker_fn,
                args=(i, loop, agent_manager,
                      self._stop_event, self._pause_event),
                kwargs={
                    "on_game_done_callback": on_game_done,
                    "games_to_play": games_to_play,
                },
                daemon=True,
            )
            t.start()
            self._threads[i] = t

    # ── continuous execution (collection) ────────────────────────

    async def run_continuously(self,
                                agent_manager: 'EnhancedAgentManager',
                                on_game_done=None) -> None:
        """Run games back-to-back, spawning a new one as soon as one finishes."""
        loop = asyncio.get_event_loop()
        self._launch(agent_manager, loop, on_game_done)

        try:
            while self.should_continue:
                if not any(t.is_alive() for t in self._threads.values()):
                    break
                await asyncio.sleep(0.5)
        finally:
            self._cleanup()

    # ── fixed-run execution (evaluation) ─────────────────────────

    async def run_fixed_games(self,
                               agent_manager: 'EnhancedAgentManager',
                               num_games: int) -> List[GameResult]:
        """Run exactly *num_games* evaluation games and return their results."""
        results: List[GameResult] = []
        results_lock = asyncio.Lock()

        async def _collect(result: GameResult):
            async with results_lock:
                results.append(result)

        loop = asyncio.get_event_loop()
        games_per_worker = num_games // self.num_workers
        remainder = num_games % self.num_workers

        self._stop_event.clear()
        self._pause_event.clear()
        self._threads.clear()

        for i in range(self.num_workers):
            count = games_per_worker + (1 if i < remainder else 0)
            t = threading.Thread(
                target=self._worker_fn,
                args=(i, loop, agent_manager,
                      self._stop_event, self._pause_event),
                kwargs={
                    "on_game_done_callback": _collect,
                    "games_to_play": count,
                },
                daemon=True,
            )
            t.start()
            self._threads[i] = t

        try:
            while any(t.is_alive() for t in self._threads.values()):
                await asyncio.sleep(0.5)
        finally:
            self._cleanup()

        return results

    # ── cleanup ──────────────────────────────────────────────────

    def _cleanup(self):
        """Signal stop and join all threads."""
        self._stop_event.set()
        for t in self._threads.values():
            if t.is_alive():
                t.join(timeout=2.0)
        self._threads.clear()


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

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self):
        """Create all components: buffer, agents, batch processor, trainer."""
        self.trainer = Trainer()
        self.summary_writer = self._build_logger()
        self.global_buffer = GlobalBuffer(config.BATCH_SIZE, action_to_policy=action_3d_to_policy)

        # --- agent config -------------------------------------------------
        # best_model: the best performing model — only updated when evaluation beats it
        self.best_model = MuZeroAgent(
            action_size=3,
            action_limits=[7, 37, 10],
            obs_size=config.OBSERVATION_SIZE,
            simulations=config.NUM_SIMULATIONS,
            global_buffer=self.global_buffer,
            config_obj=self.cfg,
        )

        # current_model: the model actively being trained
        self.current_model = MuZeroAgent(
            action_size=3,
            action_limits=[7, 37, 10],
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

        # MuZero agents for *collection* – start with best model weights
        collection_agent = MuZeroAgent(
            action_size=3,
            action_limits=[7, 37, 10],
            obs_size=config.OBSERVATION_SIZE,
            simulations=config.NUM_SIMULATIONS,
            global_buffer=self.global_buffer,
            weights=copy.deepcopy(self.best_model.get_weights()),
            config_obj=self.cfg,
        )
        self._training_agents = [collection_agent]

        random_agent = RandomAgent("RandomTraining")
        cultist_agent = CultistAgent()
        divine_agent = DivineAgent()

        agent_configs: List[Tuple[Any, int]] = [
            (collection_agent, 8)
        ]

        # --- batch processor + agent manager -------------------------------
        self.agent_manager, _ = create_custom_agent_setup(
            agent_configs,
            max_batch_size=self.cfg.max_batch_size,
            batch_timeout_ms=self.cfg.batch_timeout_ms,
            gpu_memory_fraction=self.cfg.gpu_memory_fraction,
        )

        # --- parallel env manager -----------------------------------------
        self.env_manager = self._create_env_manager(self.cfg.concurrent_games)

        print(f"TrainingOrchestrator setup complete:")
        print(f"  Concurrent games : {self.cfg.concurrent_games}")
        print(f"  Batch size       : {self.cfg.max_batch_size}")
        print(f"  Training step    : {self.training_step}")
        print(f"  GPU available    : {torch.cuda.is_available()}")

    def _build_logger(self) -> SummaryWriter:
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = f"logs/{self.cfg.run_name}{ts}"
        return SummaryWriter(log_dir)

    # ------------------------------------------------------------------
    # 1️⃣  COLLECT phase
    # ------------------------------------------------------------------

    async def collect(self) -> None:
        """
        Start the continuous game-collection loop.
        This runs forever (or until :meth:`stop_training` is called).
        Training runs in a separate background task.
        """
        self.training_active = True
        print("COLLECT phase started – running continuous games...")

        async def _on_game_done(result: GameResult):
            self.games_completed += 1

        train_task = asyncio.create_task(self._training_loop())

        try:
            await self.env_manager.run_continuously(self.agent_manager, _on_game_done)
        finally:
            self.training_active = False
            await train_task

    async def _training_loop(self) -> None:
        """
        Dedicated, non-blocking background task that handles continuous 
        training from the GlobalBuffer.
        """
        print("TRAIN phase started – background training loop active.")
        while self.training_active:
            if self.global_buffer and self.global_buffer.available_gameplay_batch():
                await self._train_step()
            else:
                # Wait for more experience to be collected
                await asyncio.sleep(0.5)

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
    
            self.trainer.train_network(
                batch=batch,
                combats=combat_batch,
                agent=self.current_model.model,
                train_step=self.training_step,
                summary_writer=self.summary_writer,
            )
            self.training_step += 1

            # Periodic saving of current model
            if self.training_step % self.cfg.sync_steps == 0:
                self.save_current_checkpoint()
    
            # Periodic evaluation
            if self.training_step % self.cfg.evaluation_interval == 0:
                self.env_manager.pause()
                await self.env_manager.wait_for_drain()
                await self.evaluate()
                self.env_manager.resume()
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

    async def evaluate(self) -> Dict[str, float]:
        """
        Run evaluation games between the current (new) model and the
        best model so far.  Keep the better-performing weights.

        Returns a dict with ``current_placement`` and ``best_placement``.
        """
        print(f"\nEVALUATE at step {self.training_step}")

        eval_current = MuZeroAgent(
            action_size=3,
            action_limits=[7, 37, 10],
            obs_size=config.OBSERVATION_SIZE,
            simulations=config.NUM_SIMULATIONS,
            global_buffer=None,
            weights=copy.deepcopy(self.current_model.get_weights()),
            config_obj=self.cfg,
        )
        eval_best = MuZeroAgent(
            action_size=3,
            action_limits=[7, 37, 10],
            obs_size=config.OBSERVATION_SIZE,
            simulations=config.NUM_SIMULATIONS,
            global_buffer=None,
            weights=copy.deepcopy(self.best_model.get_weights()),
            config_obj=self.cfg,
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

        eval_env_mgr = self._create_env_manager(self.cfg.evaluation_concurrent)
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
            self.summary_writer.add_scalar("evaluation/current_model", current_mean, self.training_step)
            self.summary_writer.add_scalar("evaluation/best_model", best_mean, self.training_step)

        print(f"  Current model placement: {current_mean:.2f}  |  Best model: {best_mean:.2f}")

        if current_mean < best_mean:
            print("  ✓ Model improved – updating best model & clearing buffers.")
            self.best_model.model.load_state_dict(self.current_model.get_weights())
            self.sync_weights()
            self.save_best_checkpoint()
            if self.global_buffer:
                if hasattr(self.global_buffer, "clear_gameplay_buffer"):
                    self.global_buffer.clear_gameplay_buffer()

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

        COLLECT runs in the background; TRAIN is triggered whenever the
        buffer has data; SYNC follows weight improvement; EVALUATE runs
        periodically.
        """
        if self.env_manager is None:
            self.setup()

        self.training_active = True

        async def _internal_callback(result: GameResult):
            self.games_completed += 1

        collect_task = asyncio.create_task(
            self.env_manager.run_continuously(self.agent_manager, _internal_callback)
        )
        train_task = asyncio.create_task(self._training_loop())

        last_logged_step = -1
        try:
            while self.training_active and self.training_step < max_steps:
                await asyncio.sleep(1.0)
                if self.training_step % 100 == 0 and self.training_step > 0 and self.training_step != last_logged_step:
                    print(f"  step={self.training_step}  games={self.games_completed}")
                    last_logged_step = self.training_step
        except asyncio.CancelledError:
            pass
        finally:
            self.training_active = False
            self.env_manager.stop()
            await asyncio.gather(collect_task, train_task, return_exceptions=True)
            if self.summary_writer:
                self.summary_writer.close()

    def stop_training(self):
        """Gracefully stop the training loop."""
        self.training_active = False
        if self.env_manager:
            self.env_manager.stop()

    async def run_single_episode(self) -> GameResult:
        """
        Run a single episode for debugging.

        Replaces the old ``train_single.py`` workflow.
        """
        if self.agent_manager is None:
            self.setup()
        worker = _GameWorker(0)
        return await worker.run_game(self.agent_manager, return_placements=True)

    async def run_parallel_demo(self, num_episodes: int = 5) -> List[GameResult]:
        """
        Run *num_episodes* games in parallel.

        Replaces the old ``train_parallel.py`` demo workflow.
        """
        if self.agent_manager is None:
            self.setup()
        mgr = self._create_env_manager(min(self.cfg.concurrent_games, num_episodes))
        return await mgr.run_fixed_games(self.agent_manager, num_episodes)

    async def run_evaluation(self, num_games: int) -> List[GameResult]:
        """Run a standalone evaluation session."""
        if self.agent_manager is None:
            self.setup()
        mgr = self._create_env_manager(self.cfg.evaluation_concurrent)
        return await mgr.run_fixed_games(self.agent_manager, num_games)

    @staticmethod
    def _create_env_manager(num_workers: int):
        """Factory: returns _MultiProcessEnvManager by default (process-level isolation
        required by TFTSet4Gym's global state).  Only returns _ThreadEnvManager when
        FORCE_THREADING_ENV_MANAGER is explicitly enabled."""
        if config.FORCE_THREADING_ENV_MANAGER:
            return _ThreadEnvManager(num_workers)
        return _MultiProcessEnvManager(num_workers)

    def cleanup(self):
        """Release resources (writer, etc.)."""
        if self.summary_writer:
            self.summary_writer.close()
        self.training_active = False


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
