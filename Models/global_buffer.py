import config
import datetime
import numpy as np
from collections import deque
import os
import random
import pickle
import asyncio
from typing import List, Optional, Any


class GlobalBuffer:
    """
    Enhanced GlobalBuffer - Ray-free implementation
    
    This class provides the same interface as the original Ray-remote GlobalBuffer
    but works as a regular Python class with optional async capabilities.
    """
    
    def __init__(self, batch_size: Optional[int] = None):
        self.gameplay_experiences = deque(maxlen=500)
        self.combat_experiences = deque(maxlen=500)
        self.batch_size = batch_size or config.BATCH_SIZE
        self.current_gameplay_index: Optional[int] = None
        self.current_combat_index: Optional[int] = None
        self.current_gameplay_order: Optional[List[str]] = None
        self.current_combat_order: Optional[List[str]] = None
        
        # Add async lock for thread safety
        try:
            self._lock = asyncio.Lock()
        except RuntimeError:
            # No event loop, will create lock when needed
            self._lock = None

    def _ensure_lock(self):
        """Ensure async lock exists"""
        if self._lock is None:
            try:
                self._lock = asyncio.Lock()
            except RuntimeError:
                # Still no event loop, operations will be sync only
                pass

    def sample_gameplay_batch(self, batch_size):
        # Returns: a batch of gameplay experiences without regard to which agent.
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

        observation_batch = np.array(observation_batch)
        value_batch = np.array(value_batch)
        policy_batch = np.array(policy_batch)

        return [observation_batch, value_batch, policy_batch]
    
    def sample_combat_batch(self, batch_size):
        # Returns: a batch of combat experiences without regard to which agent.
        observation_batch = []
        result_batch = []
        for _ in range(min(batch_size, len(self.combat_experiences))):
            if not self.combat_experiences:
                break
            observation, result = self.combat_experiences.pop()
            observation_batch.append(observation)
            result_batch.append(result)

        observation_batch = np.array(observation_batch)
        result_batch = np.array(result_batch)

        return [observation_batch, result_batch]

    def store_episode(self, sample):
        """Store episode - can be called directly or async"""
        self.gameplay_experiences.extend(sample)
        while len(self.gameplay_experiences) > self.batch_size:
            data = self.sample_gameplay_batch(self.batch_size)
            # Count batches in the buffer folder
            if not os.path.exists(config.GAMEPLAY_BUFFER_PATH):
                os.makedirs(config.GAMEPLAY_BUFFER_PATH)
            # grab the current time up to the microsecond for uniqueness
            time_stamp = datetime.datetime.now().strftime("%H%M%S%f")
            batch_file = os.path.join(config.GAMEPLAY_BUFFER_PATH, f"batch_{time_stamp}.pickle")
            # Save batch to file
            with open(batch_file, 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def store_episode_sync(self, sample):
        """Synchronous version for compatibility - same as store_episode"""
        self.store_episode(sample)

    async def store_episode_async(self, sample):
        """Async version of store_episode with improved performance"""
        self._ensure_lock()
        
        if self._lock:
            async with self._lock:
                self.gameplay_experiences.extend(sample)
                
                while len(self.gameplay_experiences) > self.batch_size:
                    data = await self._sample_gameplay_batch_async(self.batch_size)
                    await self._save_gameplay_batch_async(data)
        else:
            # No async lock available, fall back to sync
            self.gameplay_experiences.extend(sample)
            
            while len(self.gameplay_experiences) > self.batch_size:
                data = await self._sample_gameplay_batch_async(self.batch_size)
                await self._save_gameplay_batch_async(data)

    async def _sample_gameplay_batch_async(self, batch_size: int) -> List[np.ndarray]:
        """Sample gameplay batch asynchronously"""
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

    async def _save_gameplay_batch_async(self, data: List[np.ndarray]) -> None:
        """Save gameplay batch to file asynchronously"""
        if not os.path.exists(config.GAMEPLAY_BUFFER_PATH):
            os.makedirs(config.GAMEPLAY_BUFFER_PATH)
        
        time_stamp = datetime.datetime.now().strftime("%H%M%S%f")  # Include microseconds
        batch_file = os.path.join(config.GAMEPLAY_BUFFER_PATH, f"batch_{time_stamp}.pickle")
        
        # Use thread executor for I/O
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, self._write_pickle_file, batch_file, data
        )

    def _write_pickle_file(self, filename: str, data: Any) -> None:
        """Write pickle file (for executor)"""
        with open(filename, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def store_combat(self, sample):
        self.combat_experiences.append(sample)
        while len(self.combat_experiences) > self.batch_size:
            data = self.sample_combat_batch(self.batch_size)
            # Count batches in the buffer folder
            if not os.path.exists(config.COMBAT_BUFFER_PATH):
                os.makedirs(config.COMBAT_BUFFER_PATH)
            time_stamp = datetime.datetime.now().strftime("%H%M%S%f")
            batch_file = os.path.join(config.COMBAT_BUFFER_PATH, f"batch_{time_stamp}.pickle")
            with open(batch_file, 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def available_gameplay_batch(self):
        # queue_length = len(self.gameplay_experiences)
        if not os.path.exists(config.GAMEPLAY_BUFFER_PATH):
            return False
        files_length = len(os.listdir(config.GAMEPLAY_BUFFER_PATH))
        # print("QUEUE SIZE: ", queue_length)
        if files_length >= 1:
            return True
        return False
    
    def available_combat_batch(self):
        # queue_length = len(self.combat_experiences)
        if not os.path.exists(config.COMBAT_BUFFER_PATH):
            return False
        files_length = len(os.listdir(config.COMBAT_BUFFER_PATH))
        if files_length >= 1:
            return True
        return False
    
    def read_gameplay_batch(self):
        # Read batch from file
        if not os.path.exists(config.GAMEPLAY_BUFFER_PATH):
            os.makedirs(config.GAMEPLAY_BUFFER_PATH)
        if self.current_gameplay_order is None:
            files = os.listdir(config.GAMEPLAY_BUFFER_PATH)
            random.shuffle(files)
            self.current_gameplay_order = files
            self.current_gameplay_index = 0
        
        if self.current_gameplay_index is None:
            self.current_gameplay_index = 0
            
        if self.current_gameplay_index >= len(self.current_gameplay_order):
            self.current_gameplay_index = 0
            
        if not self.current_gameplay_order:
            return []
            
        batch_file = os.path.join(config.GAMEPLAY_BUFFER_PATH, self.current_gameplay_order[self.current_gameplay_index])
        with open(batch_file, 'rb') as handle:
                data = pickle.load(handle)
                self.current_gameplay_index += 1
        return data
    
    def read_combat_batch(self):
        # Read batch from file
        if not os.path.exists(config.COMBAT_BUFFER_PATH):
            os.makedirs(config.COMBAT_BUFFER_PATH)
        if self.current_combat_order is None:
            files = os.listdir(config.COMBAT_BUFFER_PATH)
            random.shuffle(files)
            self.current_combat_order = files
            self.current_combat_index = 0
            
        if self.current_combat_index is None:
            self.current_combat_index = 0
            
        if self.current_combat_index >= len(self.current_combat_order):
            self.current_combat_index = 0
            
        if not self.current_combat_order:
            return []
            
        batch_file = os.path.join(config.COMBAT_BUFFER_PATH, self.current_combat_order[self.current_combat_index])
        with open(batch_file, 'rb') as handle:
                data = pickle.load(handle)
                self.current_combat_index += 1
        return data
    
    def clear_gameplay_buffer(self):
        # Clear the buffer
        if os.path.exists(config.GAMEPLAY_BUFFER_PATH):
            for file in os.listdir(config.GAMEPLAY_BUFFER_PATH):
                os.remove(os.path.join(config.GAMEPLAY_BUFFER_PATH, file))
        self.current_gameplay_index = None
        self.current_gameplay_order = None

    def clear_combat_buffer(self):
        # Clear the buffer
        if os.path.exists(config.COMBAT_BUFFER_PATH):
            for file in os.listdir(config.COMBAT_BUFFER_PATH):
                os.remove(os.path.join(config.COMBAT_BUFFER_PATH, file))
        self.current_combat_index = None
        self.current_combat_order = None

    def get_gameplay_buffer_size(self):
        # Get the size of the buffer
        self.current_gameplay_index = 0
        if os.path.exists(config.GAMEPLAY_BUFFER_PATH):
            return len(os.listdir(config.GAMEPLAY_BUFFER_PATH))
        return 0

    def get_combat_buffer_size(self):
        # Get the size of the buffer
        self.current_combat_index = 0
        if os.path.exists(config.COMBAT_BUFFER_PATH):
            return len(os.listdir(config.COMBAT_BUFFER_PATH))
        return 0


# Factory functions for easy creation
def create_global_buffer(batch_size: Optional[int] = None) -> GlobalBuffer:
    """
    Create a GlobalBuffer instance
    
    Args:
        batch_size: Batch size for the buffer
    
    Returns:
        GlobalBuffer instance
    """
    return GlobalBuffer(batch_size)
