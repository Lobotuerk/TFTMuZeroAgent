import numpy as np
import config
import asyncio
from typing import List, Any, Optional, Union, Dict


class ReplayBuffer:
    def __init__(self, global_buffer: Optional[Any] = None):
        self.observations = []
        self.values = []
        self.policys = []
        
        # Create or use provided global buffer
        if global_buffer is None:
            from Models.global_buffer import GlobalBuffer
            self.global_buffer = GlobalBuffer()
        else:
            self.global_buffer = global_buffer

    def reset(self):
        self.observations = []
        self.values = []
        self.policys = []

    def store_step(self, observation = [], policy = [], value = 0):
        # Records a single step of gameplay experience
        self.observations.append(observation)
        self.values.append(value)
        self.policys.append(policy)

    def get_value_sequence(self):
        return self.values
    
    def set_value_sequence(self, values):
        self.values = values

    def get_len(self):
        return len(self.observations)
    
    def has_data(self) -> bool:
        """Check if buffer has data"""
        return len(self.observations) > 0
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get statistics for this buffer"""
        return {
            'length': self.get_len(),
            'has_data': self.has_data(),
            'num_values': len(self.get_value_sequence()),
            'num_observations': len(self.observations),
            'num_policies': len(self.policys)
        }

    def move_buffer_to_global(self, final_value):
        """Move buffer to global storage"""
        replay_set = []

        for current_start in range(config.UNROLL_STEPS, len(self.observations) - 1):
            value = float(self.values[-1])
            replay_set.append([self.observations[current_start-config.UNROLL_STEPS],
                               [value] * config.UNROLL_STEPS,
                            #    self.values[current_start-config.UNROLL_STEPS:current_start],
                               self.policys[current_start-config.UNROLL_STEPS:current_start]])
        
        # Handle different global buffer types
        if hasattr(self.global_buffer, 'store_episode_sync'):
            # GlobalBuffer with sync method
            self.global_buffer.store_episode_sync(replay_set)
        elif hasattr(self.global_buffer, 'store_episode'):
            # Standard store_episode method
            self.global_buffer.store_episode(replay_set)
        else:
            raise ValueError(f"Global buffer {type(self.global_buffer)} does not have a supported store method")
            
        self.reset()
    
    async def move_buffer_to_global_async(self):
        """Async version of move_buffer_to_global for better performance"""
        replay_set = []

        for current_start in range(config.UNROLL_STEPS, len(self.observations)):
            value = float(self.values[-1])
            replay_set.append([self.observations[current_start-config.UNROLL_STEPS],
                               [value] * config.UNROLL_STEPS,
                               self.policys[current_start-config.UNROLL_STEPS:current_start]])
        
        # Use async version if available
        if hasattr(self.global_buffer, 'store_episode_async'):
            await self.global_buffer.store_episode_async(replay_set)
        else:
            # Fallback to sync version
            self.move_buffer_to_global()
            return
            
        self.reset()


# Factory function for easy creation
def create_replay_buffer(use_async: bool = True, global_buffer: Optional[Any] = None) -> ReplayBuffer:
    """
    Create a ReplayBuffer with the appropriate global buffer
    
    Args:
        use_async: Whether to use async-capable GlobalBuffer (True) or basic version (False)
        global_buffer: Optional existing global buffer
    
    Returns:
        ReplayBuffer instance
    """
    if global_buffer is not None:
        return ReplayBuffer(global_buffer)
    else:
        # Always use GlobalBuffer now since it supports both sync and async
        from Models.global_buffer import GlobalBuffer
        buffer = GlobalBuffer()
        return ReplayBuffer(buffer)
