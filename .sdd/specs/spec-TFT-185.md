# Technical Specification: [TFT-185](mention://issue/e273fe93-30c0-469f-83ad-b6f00a48b7af) - Test Batched Inference

## 1. Overview

This document outlines the technical changes required to fix the batched inference tests in `tests/test_batched_inference.py`. The current implementation uses an incorrect and obsolete configuration value for the policy head size in the mock network, causing tests to be misaligned with the actual model's output. Additionally, a separate test is crashing due to a mismatch in the observation size.

## 2. Background

The issue stems from three main points:
1.  The mock network in `tests/test_batched_inference.py` uses `config.POLICY_HEAD_SIZES[0]` (1743) for the policy logits dimension.
2.  The actual `PredNetwork` in `Models/MuZero_torch_model.py` uses `config.ACTION_CONCAT_SIZE` (55) for the policy logits dimension.
3.  The `test_parallel_batch_select_action` test in `tests/test_batched_inference.py` creates a dummy observation with a size that might not match the model's expected input size, causing a `ValueError`.
4.  The `POLICY_HEAD_SIZES` variable in `config.py` is obsolete and no longer used by the model.

## 3. Proposed Changes

### 3.1. Update Mock Network to Use `ACTION_CONCAT_SIZE`

In `tests/test_batched_inference.py`, the `MockNetwork` will be updated to use `config.ACTION_CONCAT_SIZE` instead of `config.POLICY_HEAD_SIZES[0]`.

**File:** `tests/test_batched_inference.py`

```python
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
            "policy_logits": torch.randn(batch, config.ACTION_CONCAT_SIZE), # Changed from POLICY_HEAD_SIZES[0]
            "value": torch.randn(batch, 1),
        }
```

The assertions in `test_single_predict` will also be updated to reflect this change.

```python
    def test_single_predict(self, queue):
        hs = torch.randn(config.HIDDEN_STATE_SIZE)
        act = make_action()
        result = queue.predict(hs, act)
        assert "hidden_state" in result
        assert "policy_logits" in result
        assert "value" in result
        assert result["hidden_state"].shape == (config.HIDDEN_STATE_SIZE,)
        assert result["policy_logits"].shape == (config.ACTION_CONCAT_SIZE,) # Changed from POLICY_HEAD_SIZES[0]
        assert result["value"].shape == (1,)
```

And in `test_no_network_fallback_handling`:
```python
    def test_no_network_fallback_handling(self):
        class MinimalNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.zeros(1))
            def recurrent_inference(self, hidden_state, action):
                batch = hidden_state.size(0)
                return {
                    "hidden_state": hidden_state + 1,
                    "policy_logits": torch.zeros(batch, config.ACTION_CONCAT_SIZE), # Changed from POLICY_HEAD_SIZES[0]
                    "value": torch.zeros(batch, 1),
                }
```
### 3.2. Remove `POLICY_HEAD_SIZES` from `config.py`

The `POLICY_HEAD_SIZES` variable will be removed from `config.py` as it is obsolete.

**File:** `config.py`

```python
# Remove the following line:
POLICY_HEAD_SIZES = [1624+1+1+58+58+1]  # [All probabble actions without items]
```

### 3.3. Fix Observation Size Mismatch in `test_parallel_batch_select_action`

In `tests/test_batched_inference.py`, the `test_parallel_batch_select_action` will be updated to use the `OBSERVATION_SIZE` from the `MuZeroAgent`'s network, which is the correct size. The `MuZeroAgent` initialization will be updated to use a mock network to avoid loading the real model.

**File:** `tests/test_batched_inference.py`

```python
    def test_parallel_batch_select_action(self):
        from Models.MuZero_torch_agent import MuZeroAgent
        agent = MuZeroAgent()
        # Replace the agent's network with a mock network to avoid loading the real model
        agent.network = MockNetwork().to("cpu")
        agent.simulations = 2
        agent.mcts.mcts_max_seconds = 1
        
        # Use the correct observation size from the agent's network config
        obs_list = [np.zeros(config.OBSERVATION_SIZE) for _ in range(4)]
        masks = [np.ones(54, dtype=bool) for _ in range(4)]
        
        # Call batch action selection with 4 concurrent items
        results = agent.batch_select_action(obs_list, masks)
        
        assert len(results) == 4
        for res in results:
            assert len(res) == 3
```

## 4. Testing and Validation

The implementer will be responsible for running the tests in `tests/test_batched_inference.py` to ensure that all tests pass after the changes are applied. No new tests are required.
