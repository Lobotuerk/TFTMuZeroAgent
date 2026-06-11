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

from utils.profiling import MetricsCollector


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
    agent_type: Any = object


class BatchInferenceServer:
    def __init__(self,
                 max_batch_size: int = 32,
                 batch_timeout_ms: float = 10.0,
                 gpu_memory_fraction: float = 0.8,
                 metrics_collector: Optional[MetricsCollector] = None):
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.gpu_memory_fraction = gpu_memory_fraction

        self.request_queues: Dict[Any, asyncio.Queue] = defaultdict(asyncio.Queue)
        self._processing_locks: Dict[Any, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._processing_tasks: Dict[Any, asyncio.Task] = {}

        self.agent_instances: Dict[Any, Any] = {}
        max_workers = 8 if config.IS_GIL_DISABLED else 1
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_gpu()

        self.inference_times = defaultdict(list)
        self.batch_sizes = defaultdict(list)
        self.metrics_collector = metrics_collector

    def register_agent(self, agent_type: Any, agent_instance: Any):
        self.agent_instances[agent_type] = agent_instance

    def register_agent_instance(self, agent_type: Any, agent_instance: Any):
        self.register_agent(agent_type, agent_instance)

    async def request_action(self,
                             agent_type: Any,
                             observation: np.ndarray,
                             mask: np.ndarray,
                             reward: float = 0.0,
                             terminated: bool = False,
                             player_id: str = "") -> Any:
        request = InferenceRequest(
            player_id=player_id,
            observation=observation,
            mask=mask,
            reward=reward,
            terminated=terminated,
            timestamp=time.time(),
            future=asyncio.Future(),
        )
        self.request_queues[agent_type].put_nowait(request)

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
                stats[getattr(agent_type, 'agent_name', getattr(type(agent_type), '__name__', str(agent_type)))] = {
                    'avg_inference_time': np.mean(times),
                    'total_inferences': len(times),
                    'avg_batch_size': (np.mean(self.batch_sizes[agent_type])
                                       if self.batch_sizes[agent_type] else 0),
                }
        return stats

    def _setup_gpu(self):
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(self.gpu_memory_fraction)
            torch.backends.cudnn.benchmark = True

    async def _process_batch(self, agent_type: Any):
        async with self._processing_locks[agent_type]:
            while True:
                requests = await self._collect_batch(agent_type)
                if not requests:
                    break

                agent = self.agent_instances.get(agent_type)
                if agent is None:
                    raise RuntimeError(f"Unregistered agent type requested: {getattr(agent_type, 'agent_name', getattr(type(agent_type), '__name__', str(agent_type)))}")

                asyncio.create_task(self._run_inference_and_distribute(agent, requests))

    async def _run_inference_and_distribute(self, agent: Any, requests: List[InferenceRequest]):
        try:
            results = await self._run_inference(agent, requests)
            self._distribute_results(requests, results)
        except Exception as e:
            print(f"Error in concurrent inference: {e}")

    async def _collect_batch(self, agent_type: Any) -> List[InferenceRequest]:
        requests: List[InferenceRequest] = []
        queue = self.request_queues[agent_type]

        try:
            req = await queue.get()
            requests.append(req)
            first_received = time.perf_counter()
        except asyncio.CancelledError:
            return requests

        while len(requests) < self.max_batch_size:
            try:
                req = queue.get_nowait()
                requests.append(req)
            except asyncio.QueueEmpty:
                break

        if self.metrics_collector and len(requests) > 0:
            wait_time = time.perf_counter() - first_received
            self.metrics_collector.record("inference_queue_wait", wait_time)

        return requests

    def _distribute_results(self,
                            requests: List[InferenceRequest],
                            results: List[Any]):
        for req, result in zip(requests, results):
            if req.future and not req.future.done():
                req.future.set_result(result)

    async def _run_inference(self,
                             agent: Any,
                             requests: List[InferenceRequest]) -> List[Any]:
        start_time = time.perf_counter()

        loop = asyncio.get_event_loop()
        submit_time = time.perf_counter()
        result = await loop.run_in_executor(
            self.executor, self._infer_sync, agent, requests
        )
        thread_wait = time.perf_counter() - submit_time

        self.inference_times[type(agent)].append(time.perf_counter() - start_time)
        self.batch_sizes[type(agent)].append(len(requests))

        if self.metrics_collector:
            self.metrics_collector.record("gpu_inference_total", time.perf_counter() - start_time)
            self.metrics_collector.record("thread_pool_wait", thread_wait)

        n = len(self.inference_times[type(agent)])
        if n % 100 == 0:
            avg_time = np.mean(self.inference_times[type(agent)][-100:])
            avg_batch = np.mean(self.batch_sizes[type(agent)][-100:])
            print(f"[{type(agent).__name__}] avg inference: {avg_time*1000:.2f}ms  "
                  f"avg batch size: {avg_batch:.1f}  (total batches: {n})")

        return result

    def _infer_sync(self,
                    agent: Any,
                    requests: List[InferenceRequest]) -> List[Any]:
        batch_size = len(requests)
        if batch_size == 0:
            return []

        model = getattr(agent, 'model', None)
        has_batch_api = hasattr(agent, 'batch_select_action')
        can_batch_gpu = (has_batch_api
                         and model is not None
                         and hasattr(model, 'initial_inference'))

        batch_tensor, masks = self._stack_observations(requests)

        if can_batch_gpu and batch_tensor.numel() > 0:
            gpu_start = time.perf_counter()
            with torch.no_grad():
                net_out = model.initial_inference(batch_tensor)

            sync_start = time.perf_counter()
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            gpu_time = sync_start - gpu_start
            sync_time = time.perf_counter() - sync_start

            if self.metrics_collector:
                self.metrics_collector.record("gpu_forward_pass", gpu_time)
                self.metrics_collector.record("gpu_sync", sync_time)

            precomputed = []
            for i in range(batch_size):
                precomputed.append({
                    'hidden_state': net_out['hidden_state'][i].cpu().numpy(),
                    'policy': net_out['policy_logits'][i].cpu().numpy(),
                    'value': net_out['value'][i].cpu().numpy(),
                })

            obs_list = [self._obs_to_flat(requests[i].observation)
                        for i in range(batch_size)]

            rewards = [req.reward for req in requests]
            terminated = [req.terminated for req in requests]
            player_ids = [req.player_id for req in requests]

            return agent.batch_select_action(
                obs_list, masks, precomputed_results=precomputed,
                rewards=rewards, terminated=terminated, player_ids=player_ids
            )

        obs_list = []
        for i in range(batch_size):
            obs = self._obs_to_flat(
                requests[i].observation
                if batch_tensor.numel() == 0
                else batch_tensor[i].cpu().numpy()
            )
            obs_list.append(obs)

        rewards = [req.reward for req in requests]
        terminated = [req.terminated for req in requests]
        player_ids = [req.player_id for req in requests]

        return agent.batch_select_action(
            obs_list, masks, rewards=rewards, terminated=terminated, player_ids=player_ids
        )

    def _stack_observations(self, requests: List[InferenceRequest]):
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

    def shutdown(self):
        for agent_type, task in list(self._processing_tasks.items()):
            if not task.done():
                task.cancel()
        self.executor.shutdown(wait=False)


class EnhancedAgentManager:
    def __init__(self, batch_processor: Optional[BatchInferenceServer] = None,
                 metrics_collector: Optional[MetricsCollector] = None):
        self.agents: Dict[Any, Any] = {}
        self.player_to_agent: Dict[str, type] = {}
        self.batch_processor = batch_processor or BatchInferenceServer(metrics_collector=metrics_collector)
        self.metrics_collector = metrics_collector

        self.inference_times = defaultdict(list)
        self.batch_sizes = defaultdict(list)

    def register_agent(self, agent_instance: Any, player_ids: List[str]):
        agent_type = agent_instance
        if agent_type not in self.agents:
            self.agents[agent_type] = agent_instance
            self.batch_processor.register_agent(agent_type, agent_instance)
        for pid in player_ids:
            self.player_to_agent[pid] = agent_type
        print(f"Registered {len(player_ids)} players for agent {getattr(agent_type, 'agent_name', getattr(type(agent_type), '__name__', str(agent_type)))}")

    def get_player_agent_mapping(self) -> Dict[str, Any]:
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
                          terminated: Dict[str, bool],
                          game_id: str = "") -> Dict[str, Any]:
        t0 = time.perf_counter()
        tasks = []
        player_ids = []
        for player_id, obs in observations.items():
            agent_type = self.player_to_agent.get(player_id)
            if agent_type is None:
                continue
            mask = obs.get('action_mask', np.ones(54, dtype=bool))
            unique_pid = f"{game_id}_{player_id}" if game_id else player_id
            task = self.batch_processor.request_action(
                agent_type,
                obs['tensor'],
                mask,
                rewards.get(player_id, 0.0),
                terminated.get(player_id, False),
                player_id=unique_pid,
            )
            tasks.append(task)
            player_ids.append(player_id)

        results = await asyncio.gather(*tasks)

        if self.metrics_collector:
            self.metrics_collector.record("get_actions_total", time.perf_counter() - t0)

        return {pid: result for pid, result in zip(player_ids, results)}

    def get_performance_stats(self) -> Dict[str, Any]:
        return self.batch_processor.get_performance_stats()

    async def flush_all_buffers(self, final_values: Optional[Dict[str, float]] = None, game_id: str = ""):
        final_vals = final_values or {}
        if game_id:
            final_vals = {f"{game_id}_{k}": v for k, v in final_vals.items()}

        for agent_type, agent in self.agents.items():
            if hasattr(agent, 'terminate'):
                agent.terminate(final_vals)
            elif hasattr(agent, 'replay_buffer') and agent.replay_buffer is not None:
                final_value = 0.0
                if final_vals:
                    for pid, p_agent_type in self.player_to_agent.items():
                        if p_agent_type == agent_type and pid in final_vals:
                            final_value = final_vals[pid]
                            break
                if hasattr(agent.replay_buffer, 'move_buffer_to_global_async'):
                    await agent.replay_buffer.move_buffer_to_global_async(final_value)
                else:
                    agent.replay_buffer.move_buffer_to_global(final_value)

    def shutdown(self):
        self.batch_processor.shutdown()


class AsyncGameEnvironment:
    def __init__(self, env_factory, agent_manager: EnhancedAgentManager,
                 metrics_collector: Optional[MetricsCollector] = None):
        self.env_factory = env_factory
        self.agent_manager = agent_manager
        self.metrics_collector = metrics_collector

    async def run_game(self, game_id: str) -> Dict[str, Any]:
        env = self.env_factory()
        observations = env.reset()[0]
        terminated = {pid: False for pid in env.possible_agents}
        rewards = {pid: 0.0 for pid in env.possible_agents}
        scores = {pid: 0.0 for pid in env.possible_agents}

        start = time.time()
        while not all(terminated.values()):
            actions = await self.agent_manager.get_actions(
                observations, rewards, terminated, game_id=game_id
            )
            t0 = time.perf_counter()
            observations, rewards, terminated, _, info = env.step(actions)
            if self.metrics_collector:
                self.metrics_collector.record("env_step", time.perf_counter() - t0)
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
    def __init__(self,
                 env_factory,
                 agent_manager: EnhancedAgentManager,
                 global_buffer: Optional[Any] = None,
                 num_environments: Optional[int] = None,
                 max_concurrent_games: Optional[int] = None,
                 metrics_collector: Optional[MetricsCollector] = None):
        self.env_factory = env_factory
        self.agent_manager = agent_manager
        self.global_buffer = global_buffer
        self.num_environments = num_environments or config.CONCURRENT_GAMES
        self.max_concurrent_games = max_concurrent_games or self.num_environments
        self.metrics_collector = metrics_collector

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
        self._shutdown_event.clear()
        self._environments = []
        for i in range(self.num_environments):
            env = AsyncGameEnvironment(self.env_factory, self.agent_manager,
                                       metrics_collector=self.metrics_collector)
            self._environments.append(env)
        self._active_games = 0
        self._total_games_completed = 0
        self.game_results.clear()
        self.game_durations.clear()

    async def stop(self, cancel_running: bool = False):
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
        final_values = {}
        for result in self.game_results:
            if 'scores' in result:
                final_values.update(result['scores'])

        await self.agent_manager.flush_all_buffers(final_values=final_values)

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


def create_enhanced_setup(agent_configs: Optional[List[Tuple[Any, int]]] = None,
                          max_batch_size: Optional[int] = None,
                          batch_timeout_ms: float = 5.0,
                          gpu_memory_fraction: float = 0.7,
                          metrics_collector: Optional[MetricsCollector] = None):
    if max_batch_size is None:
        max_batch_size = config.NUM_PLAYERS

    batch_processor = BatchInferenceServer(
        max_batch_size=max_batch_size,
        batch_timeout_ms=batch_timeout_ms,
        gpu_memory_fraction=gpu_memory_fraction,
        metrics_collector=metrics_collector,
    )
    agent_manager = EnhancedAgentManager(batch_processor, metrics_collector=metrics_collector)

    if agent_configs is None:
        agent_configs = _create_default_agent_configs()

    for agent_instance, count in agent_configs:
        if count > 0:
            batch_processor.register_agent(agent_instance, agent_instance)

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
