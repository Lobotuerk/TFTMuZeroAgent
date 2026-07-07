import config
import numpy as np
import threading
import random
from collections import deque
from typing import Optional, List, Any, Callable


class CombatBuffer:
    """Fixed-size circular reservoir queue for combat experiences.
    
    Never clears or pops data. Overwrites oldest entries when full.
    Capacity must be a multiple of batch_size.
    Supports uniform random sampling for combat info based training.
    """
    def __init__(self, capacity: int = 64000, batch_size: int = 32):
        assert capacity % batch_size == 0, \
            f"CombatBuffer capacity ({capacity}) must be a multiple of batch_size ({batch_size})"
        self._capacity = capacity
        self._batch_size = batch_size
        self._buffer = [None] * capacity
        self._size = 0
        self._pos = 0
        self._lock = threading.Lock()

    def add(self, sample):
        with self._lock:
            self._buffer[self._pos] = sample
            self._pos = (self._pos + 1) % self._capacity
            if self._size < self._capacity:
                self._size += 1

    def clear(self):
        with self._lock:
            self._buffer = [None] * self._capacity
            self._size = 0
            self._pos = 0

    def sample(self, batch_size):
        with self._lock:
            if self._size < batch_size:
                return None
            indices = random.sample(range(self._size), batch_size)
            samples = [self._buffer[i] for i in indices]
            observation_batch = []
            result_batch = []
            for observation, result in samples:
                observation_batch.append(observation)
                result_batch.append(result)
            return [
                np.array(observation_batch),
                np.array(result_batch)
            ]

    @property
    def size(self):
        return self._size

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        return self._buffer[index]

    def __iter__(self):
        return iter(self._buffer[:self._size])


class GameplayBuffer:
    """Deque-based buffer for gameplay experiences. Retains clear/pop capabilities."""
    def __init__(self, maxlen: int = 10000):
        self._buffer = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._tombstones = 0

    def add(self, sample):
        with self._lock:
            self._buffer.extend(sample)

    def _compact_if_needed(self):
        if self._tombstones > len(self._buffer) // 2:
            self._buffer = deque(
                [item for item in self._buffer if item is not None],
                maxlen=self._buffer.maxlen
            )
            self._tombstones = 0

    def sample(self, batch_size):
        with self._lock:
            if len(self._buffer) - self._tombstones < batch_size:
                return None
            valid_indices = [i for i in range(len(self._buffer)) if self._buffer[i] is not None]
            indices = random.sample(valid_indices, batch_size)
            samples = [self._buffer[i] for i in indices]
            for i in indices:
                self._buffer[i] = None
            self._tombstones += batch_size
            self._compact_if_needed()
            observation_batch = []
            action_batch = []
            value_batch = []
            reward_batch = []
            policy_batch = []
            target_obs_batch = []
            bootstrap_depth_batch = []
            for sample in samples:
                observation_batch.append(sample[0])
                action_batch.append(sample[1])
                value_batch.append(sample[2])
                reward_batch.append(sample[3])
                policy_batch.append(sample[4])
                if len(sample) >= 7:
                    target_obs_batch.append(sample[5])
                    bootstrap_depth_batch.append(sample[6])
                else:
                    target_obs_batch.append(None)
                    bootstrap_depth_batch.append(config.UNROLL_STEPS)
            result = [
                np.array(observation_batch),
                np.array(action_batch),
                np.array(value_batch),
                np.array(reward_batch),
                np.array(policy_batch)
            ]
            result.append(np.array(target_obs_batch))
            result.append(np.array(bootstrap_depth_batch))
            return result

    def clear(self):
        with self._lock:
            self._buffer.clear()
            self._tombstones = 0

    def __len__(self):
        return len(self._buffer) - self._tombstones

    def __getitem__(self, index):
        return self._buffer[index]

    def __iter__(self):
        return iter(item for item in self._buffer if item is not None)

    @property
    def maxlen(self):
        return self._buffer.maxlen


class GlobalBuffer:
    def __init__(self, batch_size: Optional[int] = None, action_to_policy: Optional[Callable] = None):
        self.batch_size = batch_size or config.BATCH_SIZE
        self.action_to_policy = action_to_policy

        default_combat_size = getattr(config, "COMBAT_BUFFER_SIZE", 12800)
        combat_capacity = max(default_combat_size, self.batch_size * 10)
        combat_capacity = (combat_capacity // self.batch_size) * self.batch_size

        self.combat_buffer = CombatBuffer(capacity=combat_capacity, batch_size=self.batch_size)
        self.gameplay_buffer = GameplayBuffer(maxlen=config.REPLAY_BUFFER_SIZE)

    def _convert_sample_if_needed(self, sample):
        if self.action_to_policy is None:
            return sample
        converted = []
        for item in sample:
            obs, action, value, reward, policy = item[:5]
            from Models.action_conversion import action_to_policy_if_needed, is_3d_action
            if is_3d_action(action):
                policy = action_to_policy_if_needed(action, policy, self.action_to_policy)
            extended = list(item)
            if len(extended) >= 7:
                extended[4] = policy
                converted.append(tuple(extended))
            else:
                converted.append((obs, action, value, reward, policy))
        return converted

    def store_episode(self, sample):
        self.gameplay_buffer.add(self._convert_sample_if_needed(sample))

    def store_episode_sync(self, sample):
        self.store_episode(sample)

    async def store_episode_async(self, sample):
        self.store_episode(sample)

    def store_combat(self, sample):
        self.combat_buffer.add(sample)

    def available_gameplay_batch(self):
        return len(self.gameplay_buffer) >= self.batch_size

    def available_combat_batch(self):
        return self.combat_buffer.size >= self.batch_size

    def read_gameplay_batch(self):
        return self.gameplay_buffer.sample(self.batch_size)

    def read_combat_batch(self):
        return self.combat_buffer.sample(self.batch_size)

    def sample_gameplay_batch(self, batch_size):
        return self.gameplay_buffer.sample(batch_size)

    def sample_combat_batch(self, batch_size):
        return self.combat_buffer.sample(batch_size)

    def clear_gameplay_buffer(self):
        self.gameplay_buffer.clear()

    def clear_combat_buffer(self):
        self.combat_buffer.clear()

    def get_gameplay_buffer_size(self):
        return len(self.gameplay_buffer)

    def get_combat_buffer_size(self):
        return self.combat_buffer.size

    @property
    def gameplay_experiences(self):
        return self.gameplay_buffer

    @property
    def combat_experiences(self):
        return self.combat_buffer


def create_global_buffer(batch_size: Optional[int] = None, action_to_policy: Optional[Callable] = None) -> GlobalBuffer:
    return GlobalBuffer(batch_size, action_to_policy=action_to_policy)
