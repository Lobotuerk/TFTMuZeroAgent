import sys
import os
import asyncio
import pytest
import numpy as np
from unittest.mock import patch, AsyncMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from Models.global_buffer import GlobalBuffer, WorkerGlobalBuffer, WorkerCombatBuffer, create_global_buffer


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


def test_combat_buffer_clear_resets_pointers(buffer):
    for _ in range(4):
        buffer.store_combat((np.array([1.0]), np.array([0])))
    assert buffer.get_combat_buffer_size() > 0
    buffer.combat_buffer.clear()
    assert buffer.get_combat_buffer_size() == 0
    assert buffer.combat_buffer._pos == 0


def test_combat_buffer_clear_releases_references(buffer):
    large_obs = np.ones((100, 100))
    for _ in range(10):
        buffer.store_combat((large_obs, np.array([0])))
    buffer.combat_buffer.clear()
    assert all(elem is None for elem in buffer.combat_buffer._buffer)


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
    assert len(target_obs) == 4
    assert len(bootstrap_depth) == 4
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


def _make_3d_action():
    return np.array([0, 0, 0])


def _fake_action_to_policy(action):
    return np.ones(81, dtype=np.float32)


def test_convert_sample_target_obs_preserved_no_conversion():
    buf = GlobalBuffer(batch_size=4)
    target_obs = np.array([99.0, 99.0])
    start_obs = np.array([1.0, 2.0])
    sample = [(start_obs, np.array([0]), np.array([0.5]), np.array([0.1]), np.array([0.2]), target_obs, np.array([5]))]
    result = buf._convert_sample_if_needed(sample)
    assert len(result) == 1
    assert result[0][5] is target_obs
    assert np.array_equal(result[0][5], np.array([99.0, 99.0]))


def test_convert_sample_target_obs_preserved_with_conversion():
    buf = GlobalBuffer(batch_size=4, action_to_policy=_fake_action_to_policy)
    target_obs = np.array([99.0, 99.0])
    start_obs = np.array([1.0, 2.0])
    sample = [(start_obs, _make_3d_action(), np.array([0.5]), np.array([0.1]), np.array([0.2]), target_obs, np.array([5]))]
    result = buf._convert_sample_if_needed(sample)
    assert len(result) == 1
    assert result[0][5] is target_obs
    assert np.array_equal(result[0][5], np.array([99.0, 99.0]))


def test_convert_sample_policy_converted_in_extended():
    buf = GlobalBuffer(batch_size=4, action_to_policy=_fake_action_to_policy)
    target_obs = np.array([99.0, 99.0])
    start_obs = np.array([1.0, 2.0])
    original_policy = np.array([0.2])
    sample = [(start_obs, _make_3d_action(), np.array([0.5]), np.array([0.1]), original_policy, target_obs, np.array([5]))]
    result = buf._convert_sample_if_needed(sample)
    assert len(result) == 1
    converted_policy = result[0][4]
    assert converted_policy.shape == (81,)
    assert np.all(converted_policy == 1.0)


def test_convert_sample_short_sample_no_conversion():
    buf = GlobalBuffer(batch_size=4)
    sample = [(np.array([1.0]), np.array([0]), np.array([0.5]), np.array([0.1]), np.array([0.2]))]
    result = buf._convert_sample_if_needed(sample)
    assert len(result) == 1
    assert len(result[0]) == 5


def test_convert_sample_short_sample_with_3d_action():
    buf = GlobalBuffer(batch_size=4, action_to_policy=_fake_action_to_policy)
    sample = [(np.array([1.0]), _make_3d_action(), np.array([0.5]), np.array([0.1]), np.array([0.2]))]
    result = buf._convert_sample_if_needed(sample)
    assert len(result) == 1
    assert result[0][4].shape == (81,)


def test_convert_sample_bootstrap_depth_preserved():
    buf = GlobalBuffer(batch_size=4, action_to_policy=_fake_action_to_policy)
    target_obs = np.array([99.0, 99.0])
    start_obs = np.array([1.0, 2.0])
    bootstrap_depth = np.array([3])
    sample = [(start_obs, _make_3d_action(), np.array([0.5]), np.array([0.1]), np.array([0.2]), target_obs, bootstrap_depth)]
    result = buf._convert_sample_if_needed(sample)
    assert len(result) == 1
    assert result[0][6] is bootstrap_depth
    assert np.array_equal(result[0][6], np.array([3]))


@pytest.fixture
def buffer_with_temp_path(tmp_path):
    original_path = config.GAMEPLAY_BUFFER_PATH
    config.GAMEPLAY_BUFFER_PATH = str(tmp_path / "gameplay")
    buf = GlobalBuffer(batch_size=4)
    yield buf
    config.GAMEPLAY_BUFFER_PATH = original_path


def _make_gameplay_sample(value=1.0):
    return [(np.array([float(value)]), np.array([0]), np.array([0.5]), np.array([0.1]), np.array([0.2]))]


def test_add_gameplay_experience_spill_to_disk(buffer_with_temp_path, tmp_path):
    buf = buffer_with_temp_path
    samples = _make_gameplay_sample() * 10
    buf.add_gameplay_experience(samples, skip_memory_buffer=False)
    pkl_files = list(tmp_path.glob("gameplay/*.pkl"))
    assert len(pkl_files) == 2
    assert buf.get_gameplay_buffer_size() == 2


def test_add_gameplay_experience_skip_memory_buffer(buffer_with_temp_path, tmp_path):
    buf = buffer_with_temp_path
    samples = _make_gameplay_sample() * 10
    buf.add_gameplay_experience(samples, skip_memory_buffer=True)
    pkl_files = list(tmp_path.glob("gameplay/*.pkl"))
    assert len(pkl_files) == 2
    assert buf.get_gameplay_buffer_size() == 0


def test_read_gameplay_batch_from_disk(buffer_with_temp_path, tmp_path):
    buf = buffer_with_temp_path
    samples = _make_gameplay_sample() * 10
    buf.add_gameplay_experience(samples, skip_memory_buffer=True)
    assert buf.available_gameplay_batch() is True
    batch1 = buf.read_gameplay_batch()
    assert batch1 is not None
    assert len(batch1) == 7
    assert len(batch1[0]) == 4
    pkl_files = list(tmp_path.glob("gameplay/*.pkl"))
    assert len(pkl_files) == 1
    assert buf.available_gameplay_batch() is True
    batch2 = buf.read_gameplay_batch()
    assert batch2 is not None
    assert len(batch2[0]) == 4
    pkl_files = list(tmp_path.glob("gameplay/*.pkl"))
    assert len(pkl_files) == 0
    assert buf.available_gameplay_batch() is False


def test_worker_combat_buffer_accumulation():
    buf = WorkerCombatBuffer(batch_size=4)
    assert buf.size == 0

    assert buf.add("a") is False
    assert buf.add("b") is False
    assert buf.add("c") is False
    assert buf.size == 3

    assert buf.add("d") is True
    assert buf.size == 4

    batch = buf.pop()
    assert batch == ["a", "b", "c", "d"]
    assert buf.size == 0

    for i in range(5):
        buf.add(f"s{i}")
    assert buf.size == 5

    batch = buf.pop()
    assert batch == ["s0", "s1", "s2", "s3"]
    assert buf.size == 1

    buf.clear()
    assert buf.size == 0


def test_worker_global_buffer_store_combat_batching():
    buf = WorkerGlobalBuffer(action_to_policy=None)
    buf.batch_size = 4
    buf.combat_buffer = WorkerCombatBuffer(batch_size=4)

    with patch.object(buf, '_post_to_server', new_callable=AsyncMock) as mock_post:
        for i in range(3):
            buf.store_combat(f"sample{i}")
        mock_post.assert_not_called()

        buf.store_combat("sample3")
        mock_post.assert_awaited_once_with(["sample0", "sample1", "sample2", "sample3"], "combat")


def test_worker_combat_buffer_get_all():
    buf = WorkerCombatBuffer(batch_size=4)
    assert buf.get_all() == []

    buf.add("a")
    buf.add("b")
    buf.add("c")
    assert buf.get_all() == ["a", "b", "c"]
    assert buf.size == 3


def test_worker_combat_buffer_remove_front():
    buf = WorkerCombatBuffer(batch_size=4)
    for i in range(5):
        buf.add(f"s{i}")
    assert buf.size == 5

    buf.remove_front(2)
    assert buf.size == 3
    assert buf.get_all() == ["s2", "s3", "s4"]

    buf.remove_front(0)
    assert buf.size == 3
    assert buf.get_all() == ["s2", "s3", "s4"]

    buf.remove_front(3)
    assert buf.size == 0
    assert buf.get_all() == []


def test_worker_combat_buffer_concurrency_simulation():
    buf = WorkerCombatBuffer(batch_size=4)
    for i in range(3):
        buf.add(f"initial{i}")
    initial_samples = buf.get_all()
    assert initial_samples == ["initial0", "initial1", "initial2"]

    buf.add("late_arrival")

    buf.remove_front(len(initial_samples))

    assert buf.size == 1
    assert buf.get_all() == ["late_arrival"]
