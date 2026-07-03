import sys
import os
import asyncio
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Models.global_buffer import GlobalBuffer, create_global_buffer


@pytest.fixture
def buffer():
    return GlobalBuffer(batch_size=4)


def test_create_global_buffer_defaults():
    buf = create_global_buffer()
    assert buf is not None
    assert buf.batch_size > 0


def test_create_global_buffer_custom_batch_size():
    buf = create_global_buffer(batch_size=16)
    assert buf.batch_size == 16


def test_initial_buffer_empty(buffer):
    assert buffer.get_gameplay_buffer_size() == 0
    assert buffer.get_combat_buffer_size() == 0
    assert buffer.available_gameplay_batch() is False
    assert buffer.available_combat_batch() is False


def test_store_and_check_gameplay(buffer):
    sample = [(np.array([1.0]), np.array([0]), np.array([0.5]), np.array([0.1]), np.array([0.2]))]
    buffer.store_episode(sample)
    assert buffer.get_gameplay_buffer_size() == 1


def test_store_episode_sync(buffer):
    sample = [(np.array([1.0]), np.array([0]), np.array([0.5]), np.array([0.1]), np.array([0.2]))]
    buffer.store_episode_sync(sample)
    assert buffer.get_gameplay_buffer_size() == 1


def test_store_episode_async(buffer):
    sample = [(np.array([1.0]), np.array([0]), np.array([0.5]), np.array([0.1]), np.array([0.2]))]
    asyncio.run(buffer.store_episode_async(sample))
    assert buffer.get_gameplay_buffer_size() == 1


def test_available_gameplay_batch_true(buffer):
    sample = [(np.array([1.0]), np.array([0]), np.array([0.5]), np.array([0.1]), np.array([0.2]))]
    buffer.store_episode(sample * 4)
    assert buffer.available_gameplay_batch() is True


def test_read_gameplay_batch(buffer):
    for _ in range(4):
        sample = [(np.array([1.0]), np.array([0]), np.array([0.5]), np.array([0.1]), np.array([0.2]))]
        buffer.store_episode(sample)
    batch = buffer.read_gameplay_batch()
    assert batch is not None
    assert len(batch) == 7
    assert buffer.get_gameplay_buffer_size() == 0


def test_clear_gameplay_buffer(buffer):
    sample = [(np.array([1.0]), np.array([0]), np.array([0.5]), np.array([0.1]), np.array([0.2]))]
    buffer.store_episode(sample * 4)
    buffer.clear_gameplay_buffer()
    assert buffer.get_gameplay_buffer_size() == 0


def test_store_and_check_combat(buffer):
    combat_sample = (np.array([1.0]), np.array([0]))
    buffer.store_combat(combat_sample)
    assert buffer.get_combat_buffer_size() == 1


def test_available_combat_batch(buffer):
    for _ in range(4):
        buffer.store_combat((np.array([1.0]), np.array([0])))
    assert buffer.available_combat_batch() is True


def test_read_combat_batch(buffer):
    for _ in range(4):
        buffer.store_combat((np.array([1.0]), np.array([0])))
    batch = buffer.read_combat_batch()
    assert batch is not None
    assert len(batch) == 2
    assert buffer.get_combat_buffer_size() == 4


def test_clear_combat_buffer_is_noop(buffer):
    buffer.store_combat((np.array([1.0]), np.array([0])))
    buffer.clear_combat_buffer()
    assert buffer.get_combat_buffer_size() == 1


def test_sample_gameplay_batch_insufficient(buffer):
    batch = buffer.sample_gameplay_batch(10)
    assert batch is None


def test_sample_combat_batch_insufficient(buffer):
    batch = buffer.sample_combat_batch(10)
    assert batch is None


def test_sample_gameplay_batch_shape(buffer):
    for _ in range(8):
        sample = [(np.array([1.0, 2.0]), np.array([0]), np.array([0.5]), np.array([0.1]), np.array([0.2]))]
        buffer.store_episode(sample)
    batch = buffer.sample_gameplay_batch(4)
    assert batch is not None
    obs, actions, values, rewards, policies, target_obs, bootstrap_depth = batch
    assert len(obs) == 4
    assert len(actions) == 4
    assert buffer.get_gameplay_buffer_size() == 4


def test_store_episode_large(buffer):
    many_samples = [
        (np.array([float(i)]), np.array([i]), np.array([0.5]), np.array([0.1]), np.array([0.2]))
        for i in range(100)
    ]
    buffer.store_episode(many_samples)
    assert buffer.get_gameplay_buffer_size() == 100


def test_combat_buffer_circular_overflow(buffer):
    for i in range(100):
        buffer.store_combat((np.array([float(i)]), np.array([i % 2])))
    assert buffer.get_combat_buffer_size() <= buffer.combat_buffer._capacity


def test_combat_buffer_persists_after_sample(buffer):
    for _ in range(8):
        buffer.store_combat((np.array([1.0]), np.array([0])))
    batch = buffer.read_combat_batch()
    assert batch is not None
    assert buffer.get_combat_buffer_size() == 8


def test_combat_buffer_uniform_sample(buffer):
    for i in range(100):
        buffer.store_combat((np.array([float(i)]), np.array([0])))
    samples_seen = set()
    for _ in range(10):
        batch = buffer.read_combat_batch()
        assert batch is not None
        for obs in batch[0]:
            samples_seen.add(int(obs[0]))
    assert len(samples_seen) > 4
