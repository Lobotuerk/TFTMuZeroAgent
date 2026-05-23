"""Tests for TFTSet4Gym submodule integration."""

import sys
import os
import pytest

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(parent_dir, "TFTSet4Gym"))


def test_import_parallel_env():
    from TFTSet4Gym.tft_set4_gym.tft_simulator import parallel_env
    assert parallel_env is not None


def test_create_environment():
    from TFTSet4Gym.tft_set4_gym.tft_simulator import parallel_env
    env = parallel_env()
    assert env is not None
    assert hasattr(env, "reset")
    assert hasattr(env, "step")
    assert hasattr(env, "possible_agents")


def test_environment_reset():
    from TFTSet4Gym.tft_set4_gym.tft_simulator import parallel_env
    env = parallel_env()
    observations, infos = env.reset()
    assert isinstance(observations, dict)
    assert len(observations) > 0
    for pid in observations:
        assert "tensor" in observations[pid]
        assert "action_mask" in observations[pid]
