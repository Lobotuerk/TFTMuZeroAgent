import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class BenchmarkResults:
    step_times: np.ndarray
    reset_time: float
    num_steps: int

    def summary(self) -> Dict[str, float]:
        if len(self.step_times) == 0:
            return {
                'step_time_avg_ms': 0.0,
                'step_time_median_ms': 0.0,
                'step_time_min_ms': 0.0,
                'step_time_max_ms': 0.0,
                'step_time_std_ms': 0.0,
                'reset_time_ms': self.reset_time * 1000,
                'num_steps': 0,
            }
        return {
            'step_time_avg_ms': float(np.mean(self.step_times)) * 1000,
            'step_time_median_ms': float(np.median(self.step_times)) * 1000,
            'step_time_min_ms': float(np.min(self.step_times)) * 1000,
            'step_time_max_ms': float(np.max(self.step_times)) * 1000,
            'step_time_std_ms': float(np.std(self.step_times)) * 1000,
            'reset_time_ms': self.reset_time * 1000,
            'num_steps': self.num_steps,
        }


class EnvironmentBenchmark:
    def __init__(self, env_factory):
        self.env_factory = env_factory

    def run(self, num_steps: int = 500) -> BenchmarkResults:
        env = self.env_factory()
        step_times = []

        t0 = time.perf_counter()
        observations = env.reset()[0]
        reset_time = time.perf_counter() - t0

        terminated = {pid: False for pid in env.possible_agents}
        rewards = {pid: 0.0 for pid in env.possible_agents}

        for _ in range(num_steps):
            actions = {pid: [0, 0, 0] for pid in env.possible_agents}
            t0 = time.perf_counter()
            observations, rewards, terminated, _, _ = env.step(actions)
            step_times.append(time.perf_counter() - t0)

            if all(terminated.values()):
                break

        return BenchmarkResults(
            step_times=np.array(step_times, dtype=np.float64),
            reset_time=reset_time,
            num_steps=len(step_times),
        )

    def run_multiple(self, num_episodes: int = 3, steps_per_episode: int = 500) -> List[BenchmarkResults]:
        return [self.run(steps_per_episode) for _ in range(num_episodes)]


class MetricsCollector:
    def __init__(self, window_size: int = 1000):
        self._lock = threading.Lock()
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self._window_size = window_size

    def record(self, name: str, value: float):
        with self._lock:
            self._metrics[name].append(value)

    def get_stats(self, name: str) -> Dict[str, float]:
        with self._lock:
            values = list(self._metrics.get(name, []))
        if not values:
            return {}
        arr = np.array(values, dtype=np.float64)
        return {
            'count': len(arr),
            'mean_ms': float(np.mean(arr)) * 1000,
            'median_ms': float(np.median(arr)) * 1000,
            'min_ms': float(np.min(arr)) * 1000,
            'max_ms': float(np.max(arr)) * 1000,
            'std_ms': float(np.std(arr)) * 1000,
            'total_s': float(np.sum(arr)),
        }

    def all_stats(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            names = sorted(self._metrics.keys())
        return {name: self.get_stats(name) for name in names}

    def log_summary(self):
        stats = self.all_stats()
        print("[MetricsCollector] === Performance Metrics ===")
        for name, s in stats.items():
            print(f"  {name}: mean={s['mean_ms']:.2f}ms median={s['median_ms']:.2f}ms "
                  f"min={s['min_ms']:.2f}ms max={s['max_ms']:.2f}ms count={s['count']}")
        return stats

    def clear(self):
        with self._lock:
            self._metrics.clear()
