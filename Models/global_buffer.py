import config
import numpy as np
import threading
from collections import deque
from typing import Optional, List, Any


class GlobalBuffer:
    def __init__(self, batch_size: Optional[int] = None):
        self._lock = threading.Lock()
        self.gameplay_experiences = deque(maxlen=500)
        self.combat_experiences = deque(maxlen=500)
        self.batch_size = batch_size or config.BATCH_SIZE

    def sample_gameplay_batch(self, batch_size):
        with self._lock:
            observation_batch = []
            value_batch = []
            policy_batch = []
            for _ in range(min(batch_size, len(self.gameplay_experiences))):
                if not self.gameplay_experiences:
                    break
                observation, value, policy = self.gameplay_experiences.pop()
                observation_batch.append(observation)
                value_batch.append(value)
                policy_batch.append(policy)

            return [
                np.array(observation_batch),
                np.array(value_batch),
                np.array(policy_batch)
            ]
    
    def sample_combat_batch(self, batch_size):
        with self._lock:
            observation_batch = []
            result_batch = []
            for _ in range(min(batch_size, len(self.combat_experiences))):
                if not self.combat_experiences:
                    break
                observation, result = self.combat_experiences.pop()
                observation_batch.append(observation)
                result_batch.append(result)

            return [
                np.array(observation_batch),
                np.array(result_batch)
            ]

    def store_episode(self, sample):
        with self._lock:
            self.gameplay_experiences.extend(sample)

    def store_episode_sync(self, sample):
        self.store_episode(sample)

    async def store_episode_async(self, sample):
        self.store_episode(sample)

    def store_combat(self, sample):
        with self._lock:
            self.combat_experiences.append(sample)

    def available_gameplay_batch(self):
        return len(self.gameplay_experiences) > 0
    
    def available_combat_batch(self):
        return len(self.combat_experiences) > 0
    
    def read_gameplay_batch(self):
        return self.sample_gameplay_batch(self.batch_size)
    
    def read_combat_batch(self):
        return self.sample_combat_batch(self.batch_size)
    
    def clear_gameplay_buffer(self):
        with self._lock:
            self.gameplay_experiences.clear()

    def clear_combat_buffer(self):
        with self._lock:
            self.combat_experiences.clear()

    def get_gameplay_buffer_size(self):
        return len(self.gameplay_experiences)

    def get_combat_buffer_size(self):
        return len(self.combat_experiences)


def create_global_buffer(batch_size: Optional[int] = None) -> GlobalBuffer:
    return GlobalBuffer(batch_size)
