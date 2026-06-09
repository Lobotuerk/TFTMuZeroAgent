import numpy as np
from typing import List, Optional, Callable, Union
import config


def _get_dim_sizes(dim_sizes: Optional[List[int]] = None) -> List[int]:
    if dim_sizes is not None:
        return dim_sizes
    return config.ACTION_DIM


def action_3d_to_policy(action_3d: Union[List[int], np.ndarray],
                        dim_sizes: Optional[List[int]] = None) -> np.ndarray:
    """
    Convert a 3D action [action_type, target_1, target_2] to a policy vector.

    The policy is a concatenation of one-hot blocks, one for each dimension
    of the 3D action. Block sizes are determined by dim_sizes (default
    config.ACTION_DIM = [7, 37, 10]), giving a total of 54 elements.

    Args:
        action_3d: 3-element list or array [action_type, target_1, target_2]
        dim_sizes: Per-dimension block sizes (default config.ACTION_DIM)

    Returns:
        numpy float32 array of shape (sum(dim_sizes),) with one-hot encoding
    """
    dims = _get_dim_sizes(dim_sizes)
    action = np.asarray(action_3d, dtype=np.int32).flatten()
    total_size = sum(dims)
    policy = np.zeros(total_size, dtype=np.float32)
    offset = 0
    for i in range(len(dims)):
        idx = int(action[i])
        bound = dims[i]
        if 0 <= idx < bound:
            policy[offset + idx] = 1.0
        offset += bound
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

    If current_policy is already a valid non-trivial policy vector it is
    returned as-is.  Otherwise, if the action is in 3D format and a converter
    is available, convert the action to a policy vector.
    """
    if current_policy is not None:
        arr = np.asarray(current_policy)
        if arr.ndim >= 1 and arr.size >= 3:
            return arr
    if converter is not None and is_3d_action(action):
        return converter(action)
    if current_policy is not None:
        return np.asarray(current_policy)
    return np.zeros(1, dtype=np.float32)


def make_action_converter(dim_sizes: Optional[List[int]] = None) -> Callable:
    """Create a converter callable with specific dimension sizes."""
    dims = _get_dim_sizes(dim_sizes)
    def converter(action_3d):
        return action_3d_to_policy(action_3d, dim_sizes=dims)
    return converter
