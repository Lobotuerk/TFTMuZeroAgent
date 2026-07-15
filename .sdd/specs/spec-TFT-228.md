# Technical Specification: TFT-228 - Evaluator worker taking too long

## 1. Overview
The evaluation pipeline suffers from extreme latency because non-NN agents (Random, Cultist, etc.) are being queued behind `MuZeroAgent`'s slow MCTS inference in a single-threaded `ThreadPoolExecutor`. Furthermore, MuZero's neural network forward pass launches thousands of sequential CUDA kernels due to Python `for` loops in the state embedding logic. This specification outlines architectural changes to bypass the executor for non-NN agents, fix timing metrics, and vectorize the PyTorch representation network.

## 2. Changes

### 2.1. Bypass Inference Queue for Non-NN Agents
**File:** `Models/agent_manager.py` (Class `EnhancedAgentManager`, Method `get_actions`)
- Currently, `get_actions()` blindly pushes all agent requests to `self.batch_processor.request_action`.
- **Change:** Retrieve the `agent_instance` from `self.agents.get(agent_type)`. If `not hasattr(agent_instance, 'model')`, evaluate the agent's action directly and immediately using `agent_instance.batch_select_action()`.
- Wrap the direct computation in a minimal async task so it can be passed to `asyncio.gather(*tasks)` seamlessly.
- **Why:** Non-NN agents execute in <1ms and do not require GPU batching. This removes them from the single-threaded queue.

### 2.2. Measure and Export Per-Agent Latency
**File:** `Models/agent_manager.py` and `benchmarks/core.py`
- **Change in `agent_manager.py`:** Create an async wrapper inside `get_actions()` to measure the exact await duration for each player's action task:
  ```python
  async def measure_task(task, pid):
      t0 = time.perf_counter()
      res = await task
      return pid, res, time.perf_counter() - t0
  ```
  Store these timings in a new instance attribute `self.last_action_times = {}`.
- **Change in `benchmarks/core.py:419`:** Replace the hardcoded `metrics.record_action(agent_name, 0.0)` with:
  ```python
  metrics.record_action(agent_name, env.agent_manager.last_action_times.get(pid, 0.0))
  ```
- **Why:** Resolves the bug where all agent execution times logged `0.0` in the benchmark.

### 2.3. Vectorize Representation Network Embeddings
**File:** `Models/MuZero_torch_model.py` (Class `RepNetwork`, Method `forward`)
- **Current State:** Python `for` loops iterate over `BOARD_SLOTS` (28), `BENCH_CHAMP_SLOTS` (9), `BENCH_ITEM_SLOTS` (10), `SHOP_CHAMP_SLOTS` (5), and `NUM_OPPONENTS` x `BOARD_SLOTS` (7 x 28 = 196). Inside each loop, `_encode_slot` is called, generating over 10,000 tiny CUDA kernel launches per inference batch.
- **Change:** Flatten the batch and slot dimensions to vectorize `_encode_slot` calls. For example, for the board:
  ```python
  # Original shape: (batch, BOARD_SLOTS, PER_SLOT_DIM)
  flat_board = board_shape.view(-1, PER_SLOT_DIM)
  flat_embed = self._encode_slot(flat_board)
  board_repr = flat_embed.view(-1, BOARD_SLOTS * 122)
  ```
- Apply this exact vectorization pattern to `bench_champ_shape`, `opp_boards_shape`, `bench_item_shape`, and `shop_shape`.
- **Why:** Removes Python-level looping and leverages PyTorch's native C++ batch processing, dramatically reducing MCTS iteration latency.

### 2.4. Reduce MCTS Simulation Count
**File:** `config.py`
- **Change:** Update `NUM_SIMULATIONS` from `50` to `25`.
- **Why:** Safely cuts the MCTS simulation overhead in half per step while retaining reasonable evaluation quality.
