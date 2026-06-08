import sys
import os
import pytest
import numpy as np
import torch

# Add root directory to python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import config
from Models.MuZero_torch_model import MuZeroNetwork

def test_strict_observation_size_validation():
    """Verify that initial_inference fails explicitly with a ValueError when passed mismatched observation sizes."""
    # Ensure model is initialized
    net = MuZeroNetwork()
    
    # Correct size should succeed (using 0-init weights)
    correct_obs = np.zeros((1, config.OBSERVATION_SIZE))
    try:
        net.initial_inference(correct_obs)
    except ValueError as e:
        pytest.fail(f"Correct observation size raised unexpected ValueError: {e}")
    except Exception:
        # Ignore weight/cuda run errors if any (since we only care about shape validation phase at the start)
        pass

    # Mismatched size (smaller) should fail with ValueError
    smaller_obs = np.zeros((1, config.OBSERVATION_SIZE - 10))
    with pytest.raises(ValueError) as excinfo:
        net.initial_inference(smaller_obs)
    assert "does not match config.OBSERVATION_SIZE" in str(excinfo.value)

    # Mismatched size (larger) should fail with ValueError
    larger_obs = np.zeros((1, config.OBSERVATION_SIZE + 10))
    with pytest.raises(ValueError) as excinfo2:
        net.initial_inference(larger_obs)
    assert "does not match config.OBSERVATION_SIZE" in str(excinfo2.value)
