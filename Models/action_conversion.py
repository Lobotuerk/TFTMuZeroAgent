import numpy as np
from typing import List, Optional, Callable, Union


def action_3d_to_policy(action_3d: Union[List[int], np.ndarray], num_slots: int = 37) -> np.ndarray:
    """
    Convert a 3D action [action_type, target_1, target_2] to a policy vector.

    The policy is a concatenation of three num_slots-sized one-hot blocks,
    one for each dimension of the 3D action. With the default num_slots=37,
    the resulting vector has 111 elements (3 * 37).

    This matches the format expected by the MuZero trainer where policy_logits
    has shape (batch, 3, 37) and gets flattened to (batch, 111).

    Args:
        action_3d: 3-element list or array [action_type, target_1, target_2]
        num_slots: Number of slots per dimension (default 37)

    Returns:
        numpy float32 array of shape (num_slots * 3,) with one-hot encoding
    """
    action = np.asarray(action_3d, dtype=np.int32).flatten()
    total_size = num_slots * 3
    policy = np.zeros(total_size, dtype=np.float32)
    for i in range(3):
        idx = int(action[i])
        if 0 <= idx < num_slots:
            policy[i * num_slots + idx] = 1.0
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


def make_action_converter(num_slots: int = 37) -> Callable:
    """Create a converter callable with a specific num_slots value."""
    def converter(action_3d):
        return action_3d_to_policy(action_3d, num_slots=num_slots)
    return converter
