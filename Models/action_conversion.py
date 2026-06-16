import numpy as np
from typing import List, Optional, Callable, Union
import config


def action_3d_to_policy(
    action_3d: Union[List[int], np.ndarray],
    action_dims: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Convert a 3D action [action_type, target_1, target_2] to a policy vector.

    The policy is a concatenation of variable-size one-hot blocks,
    one for each dimension of the 3D action. Block sizes come from
    action_dims (defaults to config.ACTION_DIM=[7,37,37]).

    Args:
        action_3d: 3-element list or array [action_type, target_1, target_2]
        action_dims: List of 3 block sizes (default: config.ACTION_DIM)

    Returns:
        numpy float32 array of shape (sum(action_dims),) with one-hot encoding
    """
    if action_dims is None:
        action_dims = config.ACTION_DIM
    block_sizes = list(action_dims)
    action = np.asarray(action_3d, dtype=np.int32).flatten()
    total_size = sum(block_sizes)
    policy = np.zeros(total_size, dtype=np.float32)
    offset = 0
    for i in range(3):
        idx = int(action[i])
        if 0 <= idx < block_sizes[i]:
            policy[offset + idx] = 1.0
        offset += block_sizes[i]
    return policy


def is_3d_action(action) -> bool:
    """Check if an action is in 3D format [type, target1, target2]."""
    if action is None:
        return False
    if isinstance(action, (list, np.ndarray)):
        arr = np.asarray(action)
        return arr.ndim == 1 and arr.shape[0] == 3
    return False


def action_to_policy_if_needed(
    action,
    current_policy: Optional[np.ndarray],
    converter: Optional[Callable] = None,
) -> np.ndarray:
    """
    Return a policy vector for the given action, converting from 3D if needed.

    If current_policy is already a valid full policy vector (matching
    sum(config.ACTION_DIM)), it is returned as-is. Otherwise, if the action
    is in 3D format and a converter is available, convert the action.
    """
    if current_policy is not None:
        arr = np.asarray(current_policy)
        expected_size = sum(config.ACTION_DIM)
        if arr.ndim >= 1 and arr.size == expected_size:
            return arr
    if converter is not None and is_3d_action(action):
        return converter(action)
    if current_policy is not None:
        return np.asarray(current_policy)
    return np.zeros(sum(config.ACTION_DIM), dtype=np.float32)


def make_action_converter(action_dims: Optional[List[int]] = None) -> Callable:
    """Create a converter callable with specific block sizes (defaults to config.ACTION_DIM)."""
    if action_dims is None:
        action_dims = config.ACTION_DIM
    dims = list(action_dims)

    def converter(action_3d):
        return action_3d_to_policy(action_3d, action_dims=dims)
    return converter
