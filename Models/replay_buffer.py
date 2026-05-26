import numpy as np
import config
from typing import List, Any, Optional, Union, Dict, Callable


class ReplayBuffer:
    def __init__(self, global_buffer: Optional[Any] = None):
        self.observations = []
        self.actions = []
        self.values = []
        self.rewards = []
        self.policys = []

        # Create or use provided global buffer
        if global_buffer is None:
            from Models.global_buffer import GlobalBuffer
            self.global_buffer = GlobalBuffer()
        else:
            self.global_buffer = global_buffer

    def reset(self):
        self.observations = []
        self.actions = []
        self.values = []
        self.rewards = []
        self.policys = []

    def store_step(self, observation=None, policy=None, value=0, action=None, reward=0):
        if policy is None and self.global_buffer is not None:
            converter = getattr(self.global_buffer, 'action_to_policy', None)
            if converter is not None:
                from Models.action_conversion import is_3d_action
                if is_3d_action(action):
                    policy = converter(action)
        self.observations.append(observation)
        self.actions.append(action)
        self.values.append(value)
        self.rewards.append(reward)
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
            'num_actions': len(self.actions),
            'num_rewards': len(self.rewards),
            'num_policies': len(self.policys)
        }

    def move_buffer_to_global(self, final_value):
        """Move buffer to global storage"""
        replay_set = []
        final_val = float(final_value)

        for current_start in range(config.UNROLL_STEPS, len(self.observations)):
            replay_set.append([
                self.observations[current_start - config.UNROLL_STEPS],
                self.actions[current_start - config.UNROLL_STEPS:current_start - 1],
                [final_val] * config.UNROLL_STEPS,
                self.rewards[current_start - config.UNROLL_STEPS:current_start],
                self.policys[current_start - config.UNROLL_STEPS:current_start],
            ])

        if hasattr(self.global_buffer, 'store_episode_sync'):
            self.global_buffer.store_episode_sync(replay_set)
        elif hasattr(self.global_buffer, 'store_episode'):
            self.global_buffer.store_episode(replay_set)
        else:
            raise ValueError(f"Global buffer {type(self.global_buffer)} does not have a supported store method")

        self.reset()

    async def move_buffer_to_global_async(self, final_value):
        """Async version of move_buffer_to_global for better performance"""
        replay_set = []
        final_val = float(final_value)

        for current_start in range(config.UNROLL_STEPS, len(self.observations)):
            replay_set.append([
                self.observations[current_start - config.UNROLL_STEPS],
                self.actions[current_start - config.UNROLL_STEPS:current_start - 1],
                [final_val] * config.UNROLL_STEPS,
                self.rewards[current_start - config.UNROLL_STEPS:current_start],
                self.policys[current_start - config.UNROLL_STEPS:current_start],
            ])

        if hasattr(self.global_buffer, 'store_episode_async'):
            await self.global_buffer.store_episode_async(replay_set)
        else:
            self.move_buffer_to_global(final_value)
            return

        self.reset()


# Factory function for easy creation
def create_replay_buffer(use_async: bool = True, global_buffer: Optional[Any] = None,
                         action_to_policy: Optional[Callable] = None) -> ReplayBuffer:
    """
    Create a ReplayBuffer with the appropriate global buffer
    
    Args:
        use_async: Whether to use async-capable GlobalBuffer (True) or basic version (False)
        global_buffer: Optional existing global buffer
        action_to_policy: Optional function to convert 3D actions to policy format
    
    Returns:
        ReplayBuffer instance
    """
    if global_buffer is not None:
        return ReplayBuffer(global_buffer)
    else:
        # Always use GlobalBuffer now since it supports both sync and async
        from Models.global_buffer import GlobalBuffer
        buffer = GlobalBuffer(action_to_policy=action_to_policy)
        return ReplayBuffer(buffer)
