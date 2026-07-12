# Technical Specification: TFT-222 (MCTS Batched Inference Optimization)

## 1. Analysis of `muzero-general` vs. `TFTMuZeroAgent`

The issue asks us to analyze why our MCTS setup does 50+ sequential forward passes per action decision and how `muzero-general` achieves fast experience collection in comparison.

**`muzero-general` Analysis**:
*   `muzero-general` **does not** batch its MCTS forward passes. In fact, "Batch MCTS" is listed as an unchecked TODO in their `README.md`.
*   Inside their `SelfPlay` logic, they perform exactly `num_simulations` sequential `model.recurrent_inference` passes per action decision, with a batch size of 1.
*   They achieve their performance purely through **horizontal scaling using Ray**. They spawn dozens of independent worker processes that run asynchronously. This parallelizes the games across CPU cores or GPUs but does not fundamentally solve the sequential MCTS tree traversal bottleneck.

**`TFTMuZeroAgent` Analysis**:
*   Our architecture is significantly more advanced because we utilize a `BlockingBatchInferenceQueue` and run on Python 3.13 with `python-freethreading` (GIL disabled). Our design is explicitly built to intercept concurrent MCTS forward passes and batch them into a single GPU call.
*   **The Bottleneck**: Currently, our `BatchInferenceServer` processes multiple games concurrently and hands a batch of observations to `agent.batch_select_action()`. However, inside `Models/MuZero_torch_agent.py`, the `_batch_select_action_impl` method processes each game's MCTS search **sequentially** using a list comprehension:
    ```python
    results = [run_mcts_item(i) for i in range(batch_size)]
    ```
*   Because `run_mcts_item(i)` executes synchronously for one game before starting the next, the `active_count` in our `BlockingBatchInferenceQueue` never exceeds 1. Thus, the queue is starved, and we end up executing 50+ sequential inferences of `batch_size=1` for *each* of the 64 games in the batch, completely defeating our batching infrastructure.

## 2. Proposed Changes

To fix this, we will execute `run_mcts_item(i)` concurrently for all games in the batch using a `ThreadPoolExecutor`. Because we run on `python-freethreading`, this will achieve true parallelism without GIL contention. When 64 parallel MCTS searches hit the `BlockingBatchInferenceQueue` simultaneously, they will be efficiently batched into a single GPU `recurrent_inference` call.

### `Models/MuZero_torch_agent.py`

Modify `_batch_select_action_impl` to replace the sequential list comprehension with concurrent execution.

**Before:**
```python
        results = [run_mcts_item(i) for i in range(batch_size)]
```

**After:**
```python
        import concurrent.futures

        # Use ThreadPoolExecutor to run PyMCTS searches concurrently.
        # Since we use python-freethreading (GIL disabled), this allows all 
        # games in the batch to hit the BlockingBatchInferenceQueue simultaneously, 
        # creating massive GPU batch sizes and eliminating the sequential bottleneck.
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            # Map returns an iterator, converting to list forces execution and gathers results in order
            results = list(executor.map(run_mcts_item, range(batch_size)))
```

## 3. Design Philosophy Adherence
*   **High Modular Depth**: We are not modifying the core MCTS algorithms or the complex queue logic. We are simply removing an artificial synchronization bottleneck where the interface interacts with the queue.
*   **Simplicity**: The fix is confined to a standard library `ThreadPoolExecutor` context block.
*   **Thread Safety**: Our underlying `BlockingBatchInferenceQueue` is already thread-safe (`self._lock = threading.Lock()`), so introducing parallel requests is safe and explicitly supported.