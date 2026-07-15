import time
import sys
import os
import copy
import threading
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


class SystemMetrics:
    @staticmethod
    def get_process_memory_info() -> Dict[str, float]:
        try:
            import psutil
            proc = psutil.Process()
            mem_info = proc.memory_info()
            return {
                'rss_mb': mem_info.rss / (1024 * 1024),
                'vms_mb': mem_info.vms / (1024 * 1024),
            }
        except Exception:
            return {'rss_mb': 0.0, 'vms_mb': 0.0}

    @staticmethod
    def get_system_memory_percent() -> float:
        try:
            import psutil
            return psutil.virtual_memory().percent
        except Exception:
            return 0.0

    @staticmethod
    def get_gpu_memory_info() -> Dict[str, float]:
        try:
            import torch
            if torch.cuda.is_available():
                return {
                    'allocated_mb': torch.cuda.memory_allocated() / (1024 * 1024),
                    'max_allocated_mb': torch.cuda.max_memory_allocated() / (1024 * 1024),
                }
        except Exception:
            pass
        return {'allocated_mb': 0.0, 'max_allocated_mb': 0.0}


class BenchmarkMockEnv:
    def __init__(self, num_players: int = 8, max_steps: int = 100):
        self.possible_agents = [f"player_{i}" for i in range(num_players)]
        self.agents = list(self.possible_agents)
        self.max_steps = max_steps
        self._step_count = 0
        self._obs_shape = (config.OBSERVATION_SIZE,)
        self._mask_shape = sum(config.ACTION_DIM)

    def reset(self) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        self._step_count = 0
        self.agents = list(self.possible_agents)
        observations = {
            pid: {
                'tensor': np.zeros(self._obs_shape, dtype=np.float32),
                'action_mask': np.ones(self._mask_shape, dtype=bool),
            }
            for pid in self.possible_agents
        }
        return observations, {}

    def step(self, actions: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        self._step_count += 1
        terminated = {pid: self._step_count >= self.max_steps for pid in self.possible_agents}
        rewards = {pid: 1.0 if terminated[pid] else 0.0 for pid in self.possible_agents}
        observations = {
            pid: {
                'tensor': np.zeros(self._obs_shape, dtype=np.float32),
                'action_mask': np.ones(self._mask_shape, dtype=bool),
            }
            for pid in self.possible_agents
        }
        for pid in self.possible_agents:
            if terminated[pid]:
                self.agents.remove(pid) if pid in self.agents else None
        return observations, rewards, terminated, {}, {}


class MCTSProfiler:
    def __init__(self):
        self._metrics: Dict[str, List[float]] = defaultdict(list)
        self._original_methods: Dict[str, Any] = {}

    def _patch_generate_action(self, original):
        metrics = self._metrics

        def patched(self_obj, n_simulations, observation, action_mask, reward=None, terminated=None, training=True):
            t0 = time.perf_counter()
            result = original(self_obj, n_simulations, observation, action_mask, reward, terminated, training)
            elapsed = time.perf_counter() - t0
            metrics['generate_action_time_s'].append(elapsed)
            return result

        return patched

    def _patch_run_batch(self, original):
        metrics = self._metrics

        def patched(self_obj, requests, device=None):
            t0 = time.perf_counter()
            result = original(self_obj, requests, device)
            elapsed = time.perf_counter() - t0
            metrics['recurrent_inference_gpu_s'].append(elapsed)
            return result

        return patched

    def __enter__(self):
        from Models.MCTS_torch import EnhancedMCTS
        from Models.batched_inference import BlockingBatchInferenceQueue

        self._original_methods['generate_action'] = EnhancedMCTS.generate_action
        self._original_methods['_run_batch'] = BlockingBatchInferenceQueue._run_batch

        EnhancedMCTS.generate_action = self._patch_generate_action(
            self._original_methods['generate_action']
        )
        BlockingBatchInferenceQueue._run_batch = self._patch_run_batch(
            self._original_methods['_run_batch']
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        from Models.MCTS_torch import EnhancedMCTS
        from Models.batched_inference import BlockingBatchInferenceQueue

        EnhancedMCTS.generate_action = self._original_methods['generate_action']
        BlockingBatchInferenceQueue._run_batch = self._original_methods['_run_batch']

    def get_metrics(self) -> Dict[str, Any]:
        result = {}
        for key, values in self._metrics.items():
            if not values:
                result[key.replace('_s', '_ms_avg')] = 0.0
                continue
            arr = np.array(values)
            display_key = key.replace('_s', '_ms_avg')
            result[display_key] = float(np.mean(arr)) * 1000
        if self._metrics.get('generate_action_time_s'):
            gen_times = np.array(self._metrics['generate_action_time_s'])
            result['generate_action_time_ms_avg'] = float(np.mean(gen_times)) * 1000
            result['generate_action_time_ms_median'] = float(np.median(gen_times)) * 1000
        if self._metrics.get('recurrent_inference_gpu_s'):
            inf_times = np.array(self._metrics['recurrent_inference_gpu_s'])
            result['recurrent_inference_gpu_ms_avg'] = float(np.mean(inf_times)) * 1000
        return result


class MetricsStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._env_steps: List[float] = []
        self._action_times: Dict[str, List[float]] = defaultdict(list)
        self._action_counts: Dict[str, int] = defaultdict(int)
        self._batch_sizes: Dict[str, List[float]] = defaultdict(list)

    def record_env_step(self, elapsed_s: float):
        with self._lock:
            self._env_steps.append(elapsed_s)

    def record_action(self, agent_name: str, elapsed_s: float):
        with self._lock:
            self._action_times[agent_name].append(elapsed_s)
            self._action_counts[agent_name] += 1

    def record_batch_size(self, agent_name: str, batch_size: float):
        with self._lock:
            self._batch_sizes[agent_name].append(batch_size)

    def get_env_step_stats(self) -> Dict[str, float]:
        with self._lock:
            if not self._env_steps:
                return {}
            arr = np.array(self._env_steps)
            return {
                'env_step_time_ms_avg': float(np.mean(arr)) * 1000,
                'env_step_time_ms_median': float(np.median(arr)) * 1000,
                'env_step_time_ms_std': float(np.std(arr)) * 1000,
            }

    def get_agent_stats(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            result = {}
            for agent_name, times in self._action_times.items():
                if not times:
                    continue
                arr = np.array(times)
                entry = {
                    'total_actions': len(times),
                    'time_per_action_ms_avg': float(np.mean(arr)) * 1000,
                    'time_per_action_ms_median': float(np.median(arr)) * 1000,
                }
                if agent_name in self._batch_sizes and self._batch_sizes[agent_name]:
                    bs_arr = np.array(self._batch_sizes[agent_name])
                    entry['average_batch_size'] = float(np.mean(bs_arr))
                result[agent_name] = entry
            return result

    def get_total_action_time_avg(self) -> float:
        all_times = []
        with self._lock:
            for times in self._action_times.values():
                all_times.extend(times)
        if not all_times:
            return 0.0
        return float(np.mean(all_times)) * 1000


class BenchmarkRunner:
    def __init__(
        self,
        use_mock_env: bool = True,
        num_games: int = 1,
        steps_per_game: int = 50,
        agent_setup: str = "muzero_vs_random",
        mcts_simulations: int = 50,
        deep_mcts: bool = False,
        seed: Optional[int] = None,
    ):
        self.use_mock_env = use_mock_env
        self.num_games = num_games
        self.steps_per_game = steps_per_game
        self.agent_setup = agent_setup
        self.mcts_simulations = mcts_simulations
        self.deep_mcts = deep_mcts
        self.seed = seed
        self._metrics_store = MetricsStore()
        self._gpu_memory_samples: List[float] = []

    def run(self) -> Dict[str, Any]:
        import asyncio
        return asyncio.run(self.arun())

    async def arun(self) -> Dict[str, Any]:
        from utils.profiling import MetricsCollector

        if self.seed is not None:
            from utils.seeding import set_seed
            set_seed(self.seed)

        system_start = SystemMetrics.get_process_memory_info()
        mem_percent_start = SystemMetrics.get_system_memory_percent()
        gpu_start = SystemMetrics.get_gpu_memory_info()

        metrics_collector = MetricsCollector()
        mcts_profiler = MCTSProfiler() if self.deep_mcts else None

        env_factory = self._create_env_factory()
        agent_manager, _ = self._create_agent_setup(metrics_collector)

        if self.seed is not None:
            self._disable_mcts_dirichlet_noise(agent_manager)

        if mcts_profiler:
            mcts_profiler.__enter__()

        try:
            total_start = time.perf_counter()
            await self._run_games(env_factory, agent_manager)
            total_duration = time.perf_counter() - total_start
        finally:
            try:
                import torch
                if torch.cuda.is_available():
                    self._gpu_memory_samples.append(torch.cuda.memory_allocated() / (1024 * 1024))
            except ImportError:
                pass
            if mcts_profiler:
                mcts_profiler.__exit__(None, None, None)

        system_end = SystemMetrics.get_process_memory_info()
        mem_percent_avg = (mem_percent_start + SystemMetrics.get_system_memory_percent()) / 2.0
        gpu_peak = SystemMetrics.get_gpu_memory_info()

        env_stats = self._metrics_store.get_env_step_stats()
        agent_stats = self._metrics_store.get_agent_stats()
        deep_mcts_metrics = mcts_profiler.get_metrics() if mcts_profiler else {}

        perf_stats = metrics_collector.get_stats("get_actions_total")
        get_actions_avg = perf_stats.get('mean_ms', 0.0)

        return {
            'metadata': self._build_metadata(),
            'system': {
                'rss_mb_start': system_start.get('rss_mb', 0.0),
                'rss_mb_end': system_end.get('rss_mb', 0.0),
                'vms_mb_start': system_start.get('vms_mb', 0.0),
                'vms_mb_end': system_end.get('vms_mb', 0.0),
                'system_memory_percent_avg': mem_percent_avg,
                'gpu_memory_allocated_mb_peak': max(
                    gpu_start.get('allocated_mb', 0.0),
                    gpu_peak.get('allocated_mb', 0.0),
                ),
                'gpu_memory_max_allocated_mb_peak': max(
                    gpu_start.get('max_allocated_mb', 0.0),
                    gpu_peak.get('max_allocated_mb', 0.0),
                ),
                'gpu_memory_stddev_mb': self._compute_gpu_memory_stddev(),
            },
            'performance': {
                'total_duration_s': total_duration,
                **env_stats,
                'get_actions_time_ms_avg': get_actions_avg,
            },
            'agents': agent_stats,
            'deep_mcts': deep_mcts_metrics,
        }

    def _build_metadata(self) -> Dict[str, Any]:
        import subprocess
        commit = ""
        branch = ""
        try:
            commit = subprocess.check_output(
                ['git', 'rev-parse', '--short', 'HEAD'],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except Exception:
            commit = "unknown"
        try:
            branch = subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except Exception:
            branch = "unknown"
        return {
            'git_commit': commit,
            'git_branch': branch,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'args': {
                'num_games': self.num_games,
                'steps_per_game': self.steps_per_game,
                'agent_setup': self.agent_setup,
                'mcts_simulations': self.mcts_simulations,
                'use_mock_env': self.use_mock_env,
                'deep_mcts': self.deep_mcts,
                'seed': self.seed,
            },
        }

    def _create_env_factory(self):
        if self.use_mock_env:
            max_steps = self.steps_per_game if self.steps_per_game > 0 else 200

            def mock_factory():
                return BenchmarkMockEnv(num_players=8, max_steps=max_steps)

            return mock_factory
        else:
            try:
                from TFTSet4Gym.tft_set4_gym.tft_simulator import parallel_env
                return parallel_env
            except ImportError:
                raise ImportError(
                    "TFTSet4Gym is not available. Use --mock-env (default) to run without the real simulator."
                )

    def _create_agent_setup(self, metrics_collector):
        from Models.agent_manager import create_enhanced_setup, create_muzero_vs_random_setup, create_buying_agents_setup

        if self.agent_setup == "muzero_vs_random":
            try:
                if self.seed is not None:
                    from Models.agent_manager import create_custom_agent_setup
                    from Models.MuZero_torch_agent import MuZeroAgent
                    from Models.Common_agents import SeededRandomAgent
                    agents = (
                        [(MuZeroAgent(agent_name=f"MuZero_{i}"), 1)
                         for i in range(1)]
                        + [(SeededRandomAgent(f"Random_{i}", seed=self.seed), 1)
                           for i in range(7)]
                    )
                    return create_custom_agent_setup(agents)
                return create_muzero_vs_random_setup(
                    num_muzero=1,
                    num_random=7,
                )
            except RuntimeError as e:
                if "PyMCTS" in str(e) or "MonteCarloTreeSearch" in str(e):
                    print("Warning: PyMCTS not available, falling back to all-random agent setup")
                    from Models.agent_manager import create_custom_agent_setup
                    from Models.Common_agents import RandomAgent
                    return create_custom_agent_setup([(RandomAgent(f"Random_{i}"), 1) for i in range(8)])
                raise
        elif self.agent_setup == "buying_agents":
            return create_buying_agents_setup()
        elif self.agent_setup == "tournament":
            from Models.agent_manager import create_tournament_setup
            from Models.MuZero_torch_agent import MuZeroAgent
            from Models.Common_agents import RandomAgent, CultistAgent, DivineAgent

            agents = [
                MuZeroAgent(agent_name="MuZero_Tournament"),
                RandomAgent("Random_Tournament"),
                CultistAgent(),
                DivineAgent(),
            ]
            return create_tournament_setup(agents)
        else:
            return create_enhanced_setup(metrics_collector=metrics_collector)

    async def _run_games(self, env_factory, agent_manager):
        from Models.agent_manager import AsyncGameEnvironment

        for i in range(self.num_games):
            env = AsyncGameEnvironment(env_factory, agent_manager, metrics_collector=None)
            await self._run_single_game(env, f"bench_{i}")

    async def _run_single_game(self, env, game_id: str):
        metrics = self._metrics_store
        actual_env = env.env_factory()
        observations = actual_env.reset()[0]
        terminated = {pid: False for pid in actual_env.possible_agents}
        rewards = {pid: 0.0 for pid in actual_env.possible_agents}

        step_count = 0
        while not all(terminated.values()):
            if self.steps_per_game > 0 and step_count >= self.steps_per_game:
                break

            actions = await env.agent_manager.get_actions(
                observations, rewards, terminated, game_id=game_id
            )

            t0 = time.perf_counter()
            observations, rewards, terminated, _, _ = actual_env.step(actions)
            metrics.record_env_step(time.perf_counter() - t0)

            agent_mapping = env.agent_manager.get_player_agent_mapping()
            for pid, action in actions.items():
                agent_type = agent_mapping.get(pid)
                if agent_type:
                    agent_name = getattr(
                        agent_type, 'agent_name',
                        getattr(type(agent_type), '__name__', str(agent_type))
                    )
                    metrics.record_action(agent_name, env.agent_manager.last_action_times.get(pid, 0.0))

            step_count += 1

        agent_manager = env.agent_manager
        await agent_manager.flush_all_buffers(final_values=rewards, game_id=game_id)

    def _compute_gpu_memory_stddev(self) -> float:
        if len(self._gpu_memory_samples) < 2:
            return 0.0
        try:
            import torch
            if torch.cuda.is_available():
                return float(np.std(self._gpu_memory_samples))
        except ImportError:
            pass
        return 0.0

    def _disable_mcts_dirichlet_noise(self, agent_manager) -> None:
        """Set training=False on all MCTS instances to disable Dirichlet noise."""
        agents = getattr(agent_manager, '_agents', [])
        for agent_entry in agents:
            if isinstance(agent_entry, tuple) and len(agent_entry) >= 2:
                agent = agent_entry[0]
            else:
                agent = agent_entry
            mcts = getattr(agent, 'mcts', None)
            if mcts is not None:
                mcts.training = False
