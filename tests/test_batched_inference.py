import pytest
import torch
import numpy as np
import threading
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Models.batched_inference import BlockingBatchInferenceQueue, InferenceRequest
import config


class MockNetwork(torch.nn.Module):
    def __init__(self, hidden_size=None):
        super().__init__()
        if hidden_size is None:
            hidden_size = config.HIDDEN_STATE_SIZE
        self.hidden_size = hidden_size
        self.fc = torch.nn.Linear(hidden_size, hidden_size)

    def recurrent_inference(self, hidden_state, action):
        batch = hidden_state.size(0)
        h = self.fc(hidden_state)
        return {
            "hidden_state": h,
            "policy_logits": torch.randn(batch, config.POLICY_HEAD_SIZES[0]),
            "value": torch.randn(batch, 1),
        }


def make_hidden(batch=1, size=None):
    if size is None:
        size = config.HIDDEN_STATE_SIZE
    return [torch.randn(size) for _ in range(batch)]


def make_action():
    return np.zeros(config.ACTION_CONCAT_SIZE, dtype=np.float32)


class TestInferenceRequest:
    def test_create_request(self):
        hs = torch.randn(config.HIDDEN_STATE_SIZE)
        act = make_action()
        req = InferenceRequest(hs, act)
        assert torch.equal(req.hidden_state, hs)
        assert np.array_equal(req.action, act)
        assert not req.event.is_set()
        assert req.result == {}


class TestBlockingBatchInferenceQueue:
    @pytest.fixture
    def network(self):
        return MockNetwork()

    @pytest.fixture
    def queue(self, network):
        q = BlockingBatchInferenceQueue(network, batch_size=4, timeout_seconds=0.05)
        yield q
        q.shutdown()

    def test_single_predict(self, queue):
        hs = torch.randn(config.HIDDEN_STATE_SIZE)
        act = make_action()
        result = queue.predict(hs, act)
        assert "hidden_state" in result
        assert "policy_logits" in result
        assert "value" in result
        assert result["hidden_state"].shape == (config.HIDDEN_STATE_SIZE,)
        assert result["policy_logits"].shape == (config.POLICY_HEAD_SIZES[0],)
        assert result["value"].shape == (1,)

    def test_batch_triggered_at_threshold(self, queue):
        hs_list = make_hidden(4)
        act = make_action()
        threads = []
        results = [None] * 4
        def predict_and_store(i):
            results[i] = queue.predict(hs_list[i], act)
        for i in range(4):
            t = threading.Thread(target=predict_and_store, args=(i,))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        for r in results:
            assert r is not None
            assert "hidden_state" in r

    def test_flush_remaining(self, network):
        q = BlockingBatchInferenceQueue(network, batch_size=8, timeout_seconds=0.05)
        hs = torch.randn(config.HIDDEN_STATE_SIZE)
        act = make_action()
        result = q.predict(hs, act)
        assert result is not None
        q.shutdown()

    def test_multiple_sequential_predicts(self, queue):
        for _ in range(6):
            hs = torch.randn(config.HIDDEN_STATE_SIZE)
            act = make_action()
            result = queue.predict(hs, act)
            assert "hidden_state" in result

    def test_concurrent_thread_safety(self, queue):
        hs_list = make_hidden(8)
        act = make_action()
        errors = []
        def thread_predict(i):
            try:
                r = queue.predict(hs_list[i], act)
                assert "hidden_state" in r
                assert "policy_logits" in r
                assert "value" in r
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=thread_predict, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0, f"Errors occurred: {errors}"

    def test_queue_reset_after_full_batch(self, queue):
        hs_list = make_hidden(4)
        act = make_action()
        for hs in hs_list:
            queue.predict(hs, act)
        time.sleep(0.2)
        assert queue._queue.empty()

    def test_predict_with_cpu_fallback(self):
        net = MockNetwork()
        q = BlockingBatchInferenceQueue(net, batch_size=2, timeout_seconds=0.05)
        hs = torch.randn(config.HIDDEN_STATE_SIZE)
        act = make_action()
        result = q.predict(hs, act)
        assert "hidden_state" in result
        q.shutdown()

    def test_no_network_fallback_handling(self):
        class MinimalNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.zeros(1))
            def recurrent_inference(self, hidden_state, action):
                batch = hidden_state.size(0)
                return {
                    "hidden_state": hidden_state + 1,
                    "policy_logits": torch.zeros(batch, config.POLICY_HEAD_SIZES[0]),
                    "value": torch.zeros(batch, 1),
                }
        net = MinimalNet()
        q = BlockingBatchInferenceQueue(net, batch_size=2, timeout_seconds=0.05)
        hs = torch.randn(config.HIDDEN_STATE_SIZE)
        act = make_action()
        result = q.predict(hs, act)
        assert torch.allclose(result["hidden_state"], hs + 1)
        q.shutdown()

    def test_predict_with_numpy_ndarray_hidden_state(self):
        net = MockNetwork()
        q = BlockingBatchInferenceQueue(net, batch_size=2, timeout_seconds=0.05)
        hs = np.random.randn(config.HIDDEN_STATE_SIZE).astype(np.float32)
        act = make_action()
        result = q.predict(hs, act)
        assert "hidden_state" in result
        assert isinstance(result["hidden_state"], torch.Tensor)
        q.shutdown()

    def test_shutdown_does_not_raise(self, queue):
        queue.shutdown()

    def test_parallel_batch_select_action(self):
        from Models.MuZero_torch_agent import MuZeroAgent
        agent = MuZeroAgent()
        agent.simulations = 2
        agent.mcts.mcts_max_seconds = 1
        
        obs_list = [np.zeros(config.OBSERVATION_SIZE) for _ in range(4)]
        masks = [np.ones(54, dtype=bool) for _ in range(4)]
        
        # Call batch action selection with 4 concurrent items
        results = agent.batch_select_action(obs_list, masks)
        
        assert len(results) == 4
        for res in results:
            assert len(res) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
