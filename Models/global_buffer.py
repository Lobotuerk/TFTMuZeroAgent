import asyncio
import config
import numpy as np
import os
import pickle
import random
import time
import uuid
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

    def add(self, sample):
        self._buffer[self._pos] = sample
        self._pos = (self._pos + 1) % self._capacity
        if self._size < self._capacity:
            self._size += 1

    def clear(self):
        self._buffer = [None] * self._capacity
        self._size = 0
        self._pos = 0

    def sample(self, batch_size):
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
        self._tombstones = 0

    def add(self, sample):
        self._buffer.extend(sample)

    def _compact_if_needed(self):
        if self._tombstones > len(self._buffer) // 2:
            self._buffer = deque(
                [item for item in self._buffer if item is not None],
                maxlen=self._buffer.maxlen
            )
            self._tombstones = 0

    def _format_batch(self, samples):
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

    def sample(self, batch_size):
        if len(self._buffer) - self._tombstones < batch_size:
            return None
        valid_indices = [i for i in range(len(self._buffer)) if self._buffer[i] is not None]
        indices = random.sample(valid_indices, batch_size)
        samples = [self._buffer[i] for i in indices]
        for i in indices:
            self._buffer[i] = None
        self._tombstones += batch_size
        self._compact_if_needed()
        return self._format_batch(samples)

    def clear(self):
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

    def add_gameplay_experience(self, samples, skip_memory_buffer: bool = False):
        converted = self._convert_sample_if_needed(samples)
        batch_size = self.batch_size
        num_batches = len(converted) // batch_size
        leftover = len(converted) % batch_size

        if num_batches > 0:
            os.makedirs(config.GAMEPLAY_BUFFER_PATH, exist_ok=True)
            for i in range(num_batches):
                batch_data = converted[i * batch_size : (i + 1) * batch_size]
                filename = f"batch_{time.time_ns()}_{uuid.uuid4().hex}.pkl"
                filepath = os.path.join(config.GAMEPLAY_BUFFER_PATH, filename)
                with open(filepath, "wb") as f:
                    pickle.dump(batch_data, f)

        if leftover > 0 and not skip_memory_buffer:
            leftover_data = converted[num_batches * batch_size:]
            self.gameplay_buffer.add(leftover_data)

    def available_gameplay_batch(self):
        if os.path.exists(config.GAMEPLAY_BUFFER_PATH):
            files = [f for f in os.listdir(config.GAMEPLAY_BUFFER_PATH) if f.endswith(".pkl")]
            if len(files) > 0:
                return True
        return len(self.gameplay_buffer) >= self.batch_size

    def read_gameplay_batch(self):
        if os.path.exists(config.GAMEPLAY_BUFFER_PATH):
            files = sorted([f for f in os.listdir(config.GAMEPLAY_BUFFER_PATH) if f.endswith(".pkl")])
            if len(files) > 0:
                filepath = os.path.join(config.GAMEPLAY_BUFFER_PATH, files[0])
                try:
                    with open(filepath, "rb") as f:
                        batch_samples = pickle.load(f)
                    os.remove(filepath)
                    return self.gameplay_buffer._format_batch(batch_samples)
                except Exception as e:
                    print(f"Error reading/deleting batch file {filepath}: {e}")
                    if os.path.exists(filepath):
                        try:
                            os.remove(filepath)
                        except Exception:
                            pass
        return self.gameplay_buffer.sample(self.batch_size)

    def clear_all_gameplay_data(self):
        self.clear_gameplay_buffer()
        if os.path.exists(config.GAMEPLAY_BUFFER_PATH):
            for f in os.listdir(config.GAMEPLAY_BUFFER_PATH):
                if f.endswith(".pkl"):
                    try:
                        os.remove(os.path.join(config.GAMEPLAY_BUFFER_PATH, f))
                    except Exception as e:
                        print(f"Error deleting file {f}: {e}")

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


class WorkerCombatBuffer:
    def __init__(self, batch_size: int = config.BATCH_SIZE):
        self._buffer: List[Any] = []
        self.batch_size = batch_size

    def add(self, sample: Any) -> bool:
        self._buffer.append(sample)
        return len(self._buffer) >= self.batch_size

    def pop(self) -> List[Any]:
        batch = self._buffer[:self.batch_size]
        self._buffer = self._buffer[self.batch_size:]
        return batch

    def clear(self):
        self._buffer.clear()

    def get_all(self):
        return list(self._buffer)

    def remove_front(self, count: int):
        if count > 0:
            self._buffer = self._buffer[count:]

    @property
    def size(self) -> int:
        return len(self._buffer)


class WorkerGlobalBuffer:
    def __init__(self, action_to_policy: Optional[Callable] = None):
        self.action_to_policy = action_to_policy
        self.batch_size = config.BATCH_SIZE
        self.gameplay_buffer = []
        self.combat_buffer = WorkerCombatBuffer(batch_size=self.batch_size)

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

    async def store_episode_async(self, sample):
        converted = self._convert_sample_if_needed(sample)
        await self._post_to_server(converted, "gameplay")

    def store_episode(self, sample):
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                loop.create_task(self.store_episode_async(sample))
                return
        except RuntimeError:
            pass
        asyncio.run(self.store_episode_async(sample))

    def store_episode_sync(self, sample):
        self.store_episode(sample)

    def store_combat(self, sample):
        if self.combat_buffer.add(sample):
            self._flush_combat()

    def _flush_combat(self):
        batch = self.combat_buffer.pop()
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                loop.create_task(self._post_to_server(batch, "combat"))
                return
        except RuntimeError:
            pass
        asyncio.run(self._post_to_server(batch, "combat"))

    def clear_gameplay_buffer(self):
        pass

    def clear_combat_buffer(self):
        self.combat_buffer.clear()

    async def _post_to_server(self, data, experience_type: str):
        import aiohttp
        import pickle
        import random

        url = f"http://{config.WORKERS_HOST}:{config.SERVER_PORT}/api/v1/experience"
        payload = pickle.dumps(data)
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for attempt in range(5):
                try:
                    async with session.post(url, data=payload, headers={
                        "Content-Type": "application/octet-stream",
                        "X-Experience-Type": experience_type
                    }) as resp:
                        if resp.status == 200:
                            print(f"[WorkerGlobalBuffer] Successfully POSTed {len(data)} {experience_type} steps")
                            return
                        elif resp.status == 503:
                            print(f"[WorkerGlobalBuffer] Server reported 503 on {experience_type} upload. Retrying in 10s...")
                            await asyncio.sleep(10.0)
                        else:
                            body = await resp.text()
                            print(f"[WorkerGlobalBuffer] Failed to upload {experience_type} (status {resp.status}): {body[:200]}")
                            return
                except Exception as e:
                    print(f"[WorkerGlobalBuffer] Connection error on {experience_type} upload (attempt {attempt+1}): {e}")
                    if attempt < 4:
                        await asyncio.sleep(2.0 + random.random() * 2.0)


def create_global_buffer(batch_size: Optional[int] = None, action_to_policy: Optional[Callable] = None) -> GlobalBuffer:
    return GlobalBuffer(batch_size, action_to_policy=action_to_policy)
