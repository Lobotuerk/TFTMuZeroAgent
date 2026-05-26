import config
import numpy as np
import threading
import random
from collections import deque
from typing import Optional, List, Any, Callable


class GlobalBuffer:
    def __init__(self, batch_size: Optional[int] = None, action_to_policy: Optional[Callable] = None):
        self._lock = threading.Lock()
        self.gameplay_experiences = deque(maxlen=config.REPLAY_BUFFER_SIZE)
        self.combat_experiences = deque(maxlen=config.REPLAY_BUFFER_SIZE)
        self.batch_size = batch_size or config.BATCH_SIZE
        self.action_to_policy = action_to_policy

    def sample_gameplay_batch(self, batch_size):
        with self._lock:
            if len(self.gameplay_experiences) < batch_size:
                return None

            samples = random.sample(self.gameplay_experiences, batch_size)
            
            observation_batch = []
            action_batch = []
            value_batch = []
            reward_batch = []
            policy_batch = []
            
            for observation, action, value, reward, policy in samples:
                observation_batch.append(observation)
                action_batch.append(action)
                value_batch.append(value)
                reward_batch.append(reward)
                policy_batch.append(policy)

            return [
                np.array(observation_batch),
                np.array(action_batch),
                np.array(value_batch),
                np.array(reward_batch),
                np.array(policy_batch)
            ]
    
    def sample_combat_batch(self, batch_size):
        with self._lock:
            if len(self.combat_experiences) < batch_size:
                return None

            samples = random.sample(self.combat_experiences, batch_size)
            
            observation_batch = []
            result_batch = []
            
            for observation, result in samples:
                observation_batch.append(observation)
                result_batch.append(result)

            return [
                np.array(observation_batch),
                np.array(result_batch)
            ]

    def _convert_sample_if_needed(self, sample):
        """Convert 3D actions in a sample to policy format if a converter is available."""
        if self.action_to_policy is None:
            return sample
        converted = []
        for item in sample:
            obs, action, value, reward, policy = item
            from Models.action_conversion import action_to_policy_if_needed, is_3d_action
            if is_3d_action(action):
                policy = action_to_policy_if_needed(action, policy, self.action_to_policy)
            converted.append((obs, action, value, reward, policy))
        return converted

    def store_episode(self, sample):
        with self._lock:
            self.gameplay_experiences.extend(self._convert_sample_if_needed(sample))

    def store_episode_sync(self, sample):
        self.store_episode(sample)

    async def store_episode_async(self, sample):
        self.store_episode(sample)

    def store_combat(self, sample):
        with self._lock:
            self.combat_experiences.append(sample)

    def available_gameplay_batch(self):
        return len(self.gameplay_experiences) >= self.batch_size
    
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


def create_global_buffer(batch_size: Optional[int] = None, action_to_policy: Optional[Callable] = None) -> GlobalBuffer:
    return GlobalBuffer(batch_size, action_to_policy=action_to_policy)
