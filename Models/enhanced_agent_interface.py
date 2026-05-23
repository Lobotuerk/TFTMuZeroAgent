"""
Batch Inference Server and Agent Manager for TFT MuZero Agent.

Provides a centralized BatchInferenceServer that collects inference
requests from multiple parallel environments, batches them by agent
type, and performs GPU-optimized batched forward passes to maximise
GPU utilisation.

Usage:
    server = BatchInferenceServer(max_batch_size=8)
    server.register_agent(MyAgentType, my_agent_instance)
    action = await server.request_action(MyAgentType, obs, mask)
"""

import torch
import numpy as np
import asyncio
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import time
from collections import defaultdict
from queue import Queue, Empty
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from .MuZero_torch_agent import MuZeroAgent
    from .Common_agents import RandomAgent, CultistAgent, DivineAgent
except (ImportError, ValueError):
    models_dir = os.path.dirname(__file__)
    if models_dir not in sys.path:
        sys.path.insert(0, models_dir)
    try:
        from MuZero_torch_agent import MuZeroAgent
        from Common_agents import RandomAgent, CultistAgent, DivineAgent
    except ImportError as e:
        print(f"Warning: Could not import agent classes: {e}")
        class MuZeroAgent:
            def __init__(self, *args, **kwargs): pass
        class RandomAgent:
            def __init__(self, *args, **kwargs): pass
        class CultistAgent:
            def __init__(self, *args, **kwargs): pass
        class DivineAgent:
            def __init__(self, *args, **kwargs): pass

import config


@dataclass
class InferenceRequest:
    player_id: str
    observation: np.ndarray
    mask: np.ndarray
    reward: float = 0.0
    terminated: bool = False
    timestamp: float = 0.0
    future: Optional[asyncio.Future] = None


@dataclass
class BatchedInferenceRequest:
    observations: torch.Tensor
    masks: Union[torch.Tensor, List, np.ndarray]
    rewards: List[float] = field(default_factory=list)
    terminated: List[bool] = field(default_factory=list)
    request_ids: List[str] = field(default_factory=list)
    agent_type: type = object


class BatchInferenceServer:
    """
    Centralised inference server that batches requests from multiple
    parallel environments by agent type, performs a single GPU forward
    pass per batch, and routes pre-computed hidden states to each
    agent's batch_select_action method.

    Key design decisions:
      - Requests are queued per agent type and collected into batches
        up to *max_batch_size* or until *batch_timeout_ms* elapses.
      - GPU inference runs in a thread-pool executor so the asyncio
        loop is never blocked.
      - When an agent exposes both ``batch_select_action`` and a
        ``model`` with ``initial_inference``, the server runs one
        forward pass on the full batch tensor and passes per-item
        results (hidden_state, policy, value) to the agent, avoiding
        N redundant representation-network calls.
      - Otherwise it falls back to calling
        ``agent.batch_select_action(observations, masks)`` without
        precomputed results.
    """

    def __init__(self,
                 max_batch_size: int = 32,
                 batch_timeout_ms: float = 10.0,
                 gpu_memory_fraction: float = 0.8):
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.gpu_memory_fraction = gpu_memory_fraction

        self.request_queues: Dict[type, Queue] = defaultdict(Queue)
        self._processing_locks: Dict[type, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._processing_tasks: Dict[type, asyncio.Task] = {}

        self.agent_instances: Dict[type, Any] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_gpu()

        self.inference_times = defaultdict(list)
        self.batch_sizes = defaultdict(list)

    # ── public API ──────────────────────────────────────────────

    def register_agent(self, agent_type: type, agent_instance: Any):
        self.agent_instances[agent_type] = agent_instance

    def register_agent_instance(self, agent_type: type, agent_instance: Any):
        self.agent_instances[agent_type] = agent_instance

    async def request_action(self,
                             agent_type: type,
                             observation: np.ndarray,
                             mask: np.ndarray,
                             reward: float = 0.0,
                             terminated: bool = False) -> Any:
        request = InferenceRequest(
            player_id="",
            observation=observation,
            mask=mask,
            reward=reward,
            terminated=terminated,
            timestamp=time.time(),
            future=asyncio.Future(),
        )
        self.request_queues[agent_type].put(request)

        if (agent_type not in self._processing_tasks
                or self._processing_tasks[agent_type].done()):
            self._processing_tasks[agent_type] = asyncio.create_task(
                self._process_batch(agent_type)
            )

        return await request.future

    def get_performance_stats(self) -> Dict[str, Any]:
        stats = {}
        for agent_type, times in self.inference_times.items():
            if times:
                stats[agent_type.__name__] = {
                    'avg_inference_time': np.mean(times),
                    'total_inferences': len(times),
                    'avg_batch_size': (np.mean(self.batch_sizes[agent_type])
                                       if self.batch_sizes[agent_type] else 0),
                }
        return stats

    # ── GPU setup ───────────────────────────────────────────────

    def _setup_gpu(self):
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(self.gpu_memory_fraction)
            torch.backends.cudnn.benchmark = True

    # ── batch lifecycle ─────────────────────────────────────────

    async def _process_batch(self, agent_type: type):
        async with self._processing_locks[agent_type]:
            # FIX: Loop until queue is empty to avoid stuck requests
            while True:
                requests = await self._collect_batch(agent_type)
                if not requests:
                    break

                agent = self.agent_instances.get(agent_type)
                if agent is None:
                    for req in requests:
                        if req.future and not req.future.done():
                            req.future.set_result([0, 0, 0])
                    continue

                results = await self._run_inference(agent, requests)
                self._distribute_results(requests, results)

    async def _collect_batch(self, agent_type: type) -> List[InferenceRequest]:
        requests: List[InferenceRequest] = []
        queue = self.request_queues[agent_type]
        start = time.time()

        while len(requests) < self.max_batch_size:
            try:
                # Use non-blocking get to avoid blocking event loop
                req = queue.get_nowait()
                requests.append(req)
                if (time.time() - start) * 1000 > self.batch_timeout_ms:
                    break
            except Empty:
                if requests:
                    break
                elapsed_ms = (time.time() - start) * 1000
                if elapsed_ms > min(10.0, self.batch_timeout_ms):
                    break
                # FIX: Use async sleep to avoid blocking event loop
                await asyncio.sleep(0.001)

        return requests

    def _distribute_results(self,
                            requests: List[InferenceRequest],
                            results: List[Any]):
        for req, result in zip(requests, results):
            if req.future and not req.future.done():
                req.future.set_result(result)

    # ── inference ───────────────────────────────────────────────

    async def _run_inference(self,
                             agent: Any,
                             requests: List[InferenceRequest]) -> List[Any]:
        start_time = time.time()

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor, self._infer_sync, agent, requests
        )

        self.inference_times[type(agent)].append(time.time() - start_time)
        self.batch_sizes[type(agent)].append(len(requests))
        return result

    def _infer_sync(self,
                    agent: Any,
                    requests: List[InferenceRequest]) -> List[Any]:
        """Blocking inference method run in the thread pool.

        Two execution paths:

        1. **Batched GPU** – when the agent provides both
           ``batch_select_action`` and a ``model`` with
           ``initial_inference``.  A single forward pass is run on the
           stacked observation tensor and per-item pre-computed results
           are threaded through to ``batch_select_action``.

        2. **Numpy fallback** – the observations are converted to CPU
           numpy lists and ``agent.batch_select_action`` is called
           without precomputed results.
        """
        batch_size = len(requests)
        if batch_size == 0:
            return []

        model = getattr(agent, 'model', None)
        has_batch_api = hasattr(agent, 'batch_select_action')
        can_batch_gpu = (has_batch_api
                         and model is not None
                         and hasattr(model, 'initial_inference'))

        batch_tensor, masks = self._stack_observations(requests)

        # ── Path 1: true GPU batched forward pass ──────────────
        if can_batch_gpu and batch_tensor.numel() > 0:
            try:
                with torch.no_grad():
                    net_out = model.initial_inference(batch_tensor)

                precomputed = []
                for i in range(batch_size):
                    precomputed.append({
                        'hidden_state': net_out['hidden_state'][i].cpu().numpy(),
                        'policy': net_out['policy_logits'][i].cpu().numpy(),
                        'value': net_out['value'][i].cpu().numpy(),
                    })

                obs_list = [self._obs_to_flat(requests[i].observation)
                            for i in range(batch_size)]

                return agent.batch_select_action(
                    obs_list, masks, precomputed_results=precomputed
                )
            except Exception as e:
                print(f"Batched GPU inference failed, falling back: {e}")

        # ── Path 2: numpy fallback ──────────────────────────────
        obs_list = []
        for i in range(batch_size):
            try:
                obs = self._obs_to_flat(
                    requests[i].observation
                    if batch_tensor.numel() == 0
                    else batch_tensor[i].cpu().numpy()
                )
            except Exception:
                obs = np.zeros(config.OBSERVATION_SIZE, dtype=np.float32)
            obs_list.append(obs)

        return agent.batch_select_action(obs_list, masks)

    # ── helpers ─────────────────────────────────────────────────

    def _stack_observations(self, requests: List[InferenceRequest]):
        """Stack individual observations into a GPU tensor.

        Returns ``(batch_tensor, mask_list)``.
        """
        obs_list = []
        masks = []
        for req in requests:
            obs = self._as_safe_float32(req.observation)
            obs_list.append(torch.from_numpy(obs))
            masks.append(req.mask
                         if isinstance(req.mask, np.ndarray)
                         else np.ones(54, dtype=bool))

        if not obs_list:
            return torch.empty(0, device=self.device), masks

        first_shape = obs_list[0].shape
        uniform = all(o.shape == first_shape for o in obs_list)

        if uniform:
            batch = torch.stack(obs_list).to(self.device)
        else:
            target = first_shape.numel()
            padded = []
            for o in obs_list:
                flat = o.flatten()
                if flat.numel() < target:
                    flat = torch.cat([flat, torch.zeros(target - flat.numel())])
                elif flat.numel() > target:
                    flat = flat[:target]
                padded.append(flat.reshape(first_shape))
            batch = torch.stack(padded).to(self.device)

        return batch, masks

    @staticmethod
    def _as_safe_float32(arr: Any) -> np.ndarray:
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        return np.asarray(arr, dtype=np.float32)

    @staticmethod
    def _obs_to_flat(obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        return obs.flatten() if obs.ndim > 1 else obs


class EnhancedAgentManager:
    """
    Maps player IDs to agent types and dispatches action requests
    to the BatchInferenceServer.
    """

    def __init__(self, batch_processor: Optional[BatchInferenceServer] = None):
        self.agents: Dict[type, Any] = {}
        self.player_to_agent: Dict[str, type] = {}
        self.batch_processor = batch_processor or BatchInferenceServer()

        self.inference_times = defaultdict(list)
        self.batch_sizes = defaultdict(list)

    def register_agent(self, agent_instance: Any, player_ids: List[str]):
        agent_type = type(agent_instance)
        if agent_type not in self.agents:
            self.agents[agent_type] = agent_instance
            self.batch_processor.register_agent(agent_type, agent_instance)
        for pid in player_ids:
            self.player_to_agent[pid] = agent_type
        print(f"Registered {len(player_ids)} players for agent {agent_type.__name__}")

    def get_player_agent_mapping(self) -> Dict[str, type]:
        return self.player_to_agent.copy()

    def setup_agents(self, agent_configs: List[Tuple[Any, int]]):
        player_counter = 0
        for agent_instance, count in agent_configs:
            if count <= 0:
                continue
            player_ids = [f"player_{player_counter + i}" for i in range(count)]
            self.register_agent(agent_instance, player_ids)
            player_counter += count
        if player_counter > config.NUM_PLAYERS:
            raise ValueError(
                f"Total agents ({player_counter}) exceeds "
                f"max players ({config.NUM_PLAYERS})"
            )

    async def get_actions(self,
                          observations: Dict[str, Dict],
                          rewards: Dict[str, float],
                          terminated: Dict[str, bool]) -> Dict[str, Any]:
        tasks = []
        player_ids = []
        for player_id, obs in observations.items():
            agent_type = self.player_to_agent.get(player_id)
            if agent_type is None:
                continue
            mask = obs.get('action_mask', np.ones(54, dtype=bool))
            task = self.batch_processor.request_action(
                agent_type,
                obs['tensor'],
                mask,
                rewards.get(player_id, 0.0),
                terminated.get(player_id, False),
            )
            tasks.append(task)
            player_ids.append(player_id)

        # Use asyncio.gather to submit all requests concurrently, enabling batching
        results = await asyncio.gather(*tasks)
        
        return {pid: result for pid, result in zip(player_ids, results)}

    def get_performance_stats(self) -> Dict[str, Any]:
        return self.batch_processor.get_performance_stats()

    async def flush_all_buffers(self):
        """Flush all agents' replay buffers to global storage concurrently"""
        tasks = []
        for agent in self.agents.values():
            if hasattr(agent, 'replay_buffer') and agent.replay_buffer is not None:
                if hasattr(agent.replay_buffer, 'move_buffer_to_global_async'):
                    tasks.append(agent.replay_buffer.move_buffer_to_global_async())
                elif hasattr(agent.replay_buffer, 'move_buffer_to_global'):
                    # Call synchronous version in executor if needed, or just call directly
                    # For now, keeping it simple as most are likely async now
                    agent.replay_buffer.move_buffer_to_global(0)
        
        if tasks:
            await asyncio.gather(*tasks)


# ── game runner ──────────────────────────────────────────────────

class AsyncGameEnvironment:
    """
    Runs a single TFT game episode inside an asyncio context,
    delegating action selection to the agent manager.
    """

    def __init__(self, env_factory, agent_manager: EnhancedAgentManager):
        self.env_factory = env_factory
        self.agent_manager = agent_manager

    async def run_game(self, game_id: str) -> Dict[str, Any]:
        env = self.env_factory()
        observations = env.reset()[0]
        terminated = {pid: False for pid in env.possible_agents}
        rewards = {pid: 0.0 for pid in env.possible_agents}
        scores = {pid: 0.0 for pid in env.possible_agents}

        start = time.time()
        while not all(terminated.values()):
            actions = await self.agent_manager.get_actions(
                observations, rewards, terminated
            )
            observations, rewards, terminated, _, info = env.step(actions)
            for player in terminated:
                if terminated[player]:
                    scores[player] = rewards[player]

        return {
            'game_id': game_id,
            'scores': scores,
            'duration': time.time() - start,
            'final_placements': self._placements(scores),
        }

    @staticmethod
    def _placements(scores: Dict[str, Union[int, float]]) -> Dict[str, int]:
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return {pid: i + 1 for i, (pid, _) in enumerate(sorted_scores)}

    _calculate_placements = _placements


class EnvironmentPool:
    """
    Manages N AsyncGameEnvironment instances running concurrently via asyncio.

    Features:
    - Concurrent game execution with configurable parallelism
    - Automatic lifecycle management (start, stop, restart)
    - Experience collection forwarded to GlobalBuffer
    - Performance monitoring and statistics
    - Graceful shutdown
    """

    def __init__(self,
                 env_factory,
                 agent_manager: EnhancedAgentManager,
                 global_buffer: Optional[Any] = None,
                 num_environments: Optional[int] = None,
                 max_concurrent_games: Optional[int] = None):
        """
        Args:
            env_factory: Callable that returns a new environment instance
            agent_manager: Shared EnhancedAgentManager for batched inference
            global_buffer: Optional GlobalBuffer for centralized experience storage
            num_environments: Number of AsyncGameEnvironment instances. Defaults to CONCURRENT_GAMES config.
            max_concurrent_games: Max games running simultaneously. Defaults to num_environments.
        """
        self.env_factory = env_factory
        self.agent_manager = agent_manager
        self.global_buffer = global_buffer
        self.num_environments = num_environments or config.CONCURRENT_GAMES
        self.max_concurrent_games = max_concurrent_games or self.num_environments

        self._environments: List[AsyncGameEnvironment] = []
        self._semaphore = asyncio.Semaphore(self.max_concurrent_games)
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._active_games: int = 0
        self._total_games_completed: int = 0
        self._shutdown_event = asyncio.Event()
        self._shutdown_event.set()
        self._lock = asyncio.Lock()

        self.game_results: List[Dict[str, Any]] = []
        self.game_durations: List[float] = []

    async def start(self, precreate: bool = True):
        """Initialize the pool and optionally pre-create environments."""
        self._shutdown_event.clear()
        self._environments = []
        for i in range(self.num_environments):
            env = AsyncGameEnvironment(self.env_factory, self.agent_manager)
            self._environments.append(env)
        self._active_games = 0
        self._total_games_completed = 0
        self.game_results.clear()
        self.game_durations.clear()

    async def stop(self, cancel_running: bool = False):
        """Gracefully stop the pool and all running games."""
        self._shutdown_event.set()
        if cancel_running:
            for game_id, task in list(self._running_tasks.items()):
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._running_tasks.clear()
        self._active_games = 0
        self._environments.clear()

    async def run_game(self, game_id: str, env_index: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Run a single game using a round-robin selected environment.

        Args:
            game_id: Unique identifier for this game
            env_index: Optional specific environment index (round-robin if None)

        Returns:
            Game result dict, or None if pool is shut down
        """
        if self._shutdown_event.is_set():
            return None

        async with self._semaphore:
            async with self._lock:
                if self._shutdown_event.is_set():
                    return None

                if not self._environments:
                    return None

                if env_index is None:
                    env_index = self._total_games_completed % len(self._environments)

                env = self._environments[env_index]
                self._active_games += 1

            try:
                result = await env.run_game(game_id)
                result['env_index'] = env_index
                return result
            finally:
                async with self._lock:
                    self._active_games -= 1
                    self._total_games_completed += 1

    async def run_games(self,
                        num_games: int,
                        game_id_prefix: str = "game") -> List[Dict[str, Any]]:
        """
        Run multiple games concurrently, collecting results.

        Args:
            num_games: Total number of games to run
            game_id_prefix: Prefix for auto-generated game IDs

        Returns:
            List of completed game result dicts
        """
        game_tasks = []
        for i in range(num_games):
            game_id = f"{game_id_prefix}_{i}"
            task = asyncio.create_task(self.run_game(game_id))
            self._running_tasks[game_id] = task
            game_tasks.append(task)

        results = await asyncio.gather(*game_tasks, return_exceptions=True)
        for task in game_tasks:
            game_id = None
            for gid, t in list(self._running_tasks.items()):
                if t is task:
                    game_id = gid
                    break
            if game_id:
                self._running_tasks.pop(game_id, None)

        completed = []
        for result in results:
            if isinstance(result, Exception):
                continue
            if result is not None:
                completed.append(result)
                self.game_results.append(result)
                self.game_durations.append(result.get('duration', 0.0))

        return completed

    async def run_continuous(self,
                              num_games: Optional[int] = None,
                              game_id_prefix: str = "game") -> List[Dict[str, Any]]:
        """
        Run games continuously, always keeping max_concurrent_games active.

        Spawns a new game as soon as one finishes, until num_games is reached.
        If num_games is None, runs indefinitely until stop() is called.

        Args:
            num_games: Total games to run (None = unlimited)
            game_id_prefix: Prefix for auto-generated game IDs

        Returns:
            List of completed game result dicts
        """
        completed = []
        counter = 0
        pending = set()

        async def _launch_one():
            nonlocal counter
            async with self._lock:
                game_id = f"{game_id_prefix}_{counter}"
                counter += 1
            task = asyncio.create_task(self.run_game(game_id))
            pending.add(task)
            return task

        for _ in range(self.max_concurrent_games):
            if num_games is not None and counter >= num_games:
                break
            await _launch_one()

        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                try:
                    result = task.result()
                    if result is not None:
                        completed.append(result)
                        self.game_results.append(result)
                        self.game_durations.append(result.get('duration', 0.0))
                except asyncio.CancelledError:
                    pass

            if num_games is None or counter < num_games:
                if not self._shutdown_event.is_set():
                    await _launch_one()

            if self._shutdown_event.is_set():
                for t in pending:
                    t.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                pending.clear()

        return completed

    async def collect_experiences(self) -> List[Any]:
        """
        Collect accumulated game results and flush agent experiences to GlobalBuffer.

        Returns:
            List of game result dicts since last collection.
        """
        # Flush all agent replay buffers to global buffer
        await self.agent_manager.flush_all_buffers()

        async with self._lock:
            collected = list(self.game_results)
            self.game_results.clear()
            self.game_durations.clear()
        return collected

    @property
    def active_games(self) -> int:
        return self._active_games

    @property
    def total_games_completed(self) -> int:
        return self._total_games_completed

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the pool."""
        durations = self.game_durations
        avg_duration = float(np.mean(durations)) if durations else 0.0
        return {
            'num_environments': self.num_environments,
            'max_concurrent_games': self.max_concurrent_games,
            'active_games': self._active_games,
            'total_games_completed': self._total_games_completed,
            'avg_game_duration': avg_duration,
            'total_game_durations_logged': len(durations),
            'environments_initialized': len(self._environments),
        }


# ── factory functions ────────────────────────────────────────────

def create_enhanced_setup(agent_configs: Optional[List[Tuple[Any, int]]] = None,
                          max_batch_size: Optional[int] = None,
                          batch_timeout_ms: float = 5.0,
                          gpu_memory_fraction: float = 0.7):
    if max_batch_size is None:
        max_batch_size = config.NUM_PLAYERS

    batch_processor = BatchInferenceServer(
        max_batch_size=max_batch_size,
        batch_timeout_ms=batch_timeout_ms,
        gpu_memory_fraction=gpu_memory_fraction,
    )
    agent_manager = EnhancedAgentManager(batch_processor)

    if agent_configs is None:
        agent_configs = _create_default_agent_configs()

    for agent_instance, count in agent_configs:
        if count > 0:
            batch_processor.register_agent(type(agent_instance), agent_instance)

    agent_manager.setup_agents(agent_configs)
    return agent_manager, batch_processor


def _create_default_agent_configs(global_buffer=None) -> List[Tuple[Any, int]]:
    muzero = MuZeroAgent(agent_name="MuZeroAgent", global_buffer=global_buffer)
    return [
        (muzero, 1),
        (RandomAgent("RandomAgent"), 5),
        (CultistAgent(), 1),
        (DivineAgent(), 1),
    ]


def create_custom_agent_setup(agents_and_counts: List[Tuple[Any, int]], **kwargs):
    return create_enhanced_setup(agent_configs=agents_and_counts, **kwargs)


async def example_usage():
    """Minimal example to verify the server wiring."""
    try:
        from TFTSet4Gym.tft_set4_gym.tft_simulator import parallel_env
    except ImportError:
        print("TFTSet4Gym not available; skipping example.")
        return

    agent_manager, bp = create_enhanced_setup()
    async_env = AsyncGameEnvironment(parallel_env, agent_manager)

    tasks = [async_env.run_game(f"example_{i}") for i in range(2)]
    results = await asyncio.gather(*tasks)

    for r in results:
        print(f"Game {r['game_id']}: {r['duration']:.2f}s  "
              f"placements={r['final_placements']}")

    stats = agent_manager.get_performance_stats()
    for name, s in stats.items():
        print(f"  {name}: {s}")


# ── convenience helpers ──────────────────────────────────────────

def create_muzero_vs_random_setup(num_muzero: int = 1,
                                   num_random: int = 7,
                                   global_buffer=None):
    agents = (
        [(MuZeroAgent(agent_name=f"MuZero_{i}", global_buffer=global_buffer), 1)
         for i in range(num_muzero)]
        + [(RandomAgent(f"Random_{i}"), 1) for i in range(num_random)]
    )
    return create_custom_agent_setup(agents)


def create_buying_agents_setup(global_buffer=None):
    return create_custom_agent_setup([
        (CultistAgent(), 2),
        (DivineAgent(), 2),
        (RandomAgent("Random"), 4),
    ])


def create_tournament_setup(agent_instances: List[Any], global_buffer=None):
    agents = [(a, 1) for a in agent_instances]
    while sum(c for _, c in agents) < config.NUM_PLAYERS:
        agents.append((RandomAgent(f"Filler_{len(agents)}"), 1))
    return create_custom_agent_setup(agents)


if __name__ == "__main__":
    asyncio.run(example_usage())
