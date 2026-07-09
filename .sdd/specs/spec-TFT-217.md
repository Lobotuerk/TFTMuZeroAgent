# Technical Specification: TFT-217 Repo Cleanup & Distributed Optimization

## 1. File Structure Changes

The repository is being cleaned up to support **strictly the distributed training workflow** (GPU server with HTTP API + CPU workers for self-play/evaluation). All legacy, monolithic, benchmark-specific, and standalone demo files are to be deleted, along with their respective test suites.

### 🗑️ Files to Delete
- **Top-Level Scripts:**
  - `benchmark_training.py`
- **Directories:**
  - `demos/` (entire directory, including `demos/mcts_torch_integration_demo.py`)
- **Stale Tests:**
  - `tests/test_benchmark_training.py`
  - `tests/test_single_episode.py`
  - `tests/test_non_blocking_training.py`
- **Legacy SDD Specifications:**
  - `.sdd/specs/spec-TFT-211.md`
  - `.sdd/specs/spec-TFT-213.md`
  - `.sdd/specs/spec-TFT-214.md`
  - `.sdd/specs/spec-TFT-215.md`
  - `.sdd/specs/spec-TFT-216.md`

### 📝 Files to Modify
- `main.py`
- `training_orchestrator.py`
- `readme.md`
- `tests/test_multiprocess_env_manager.py` (bugfix)

---

## 2. Interfaces & Signatures

### 2.1 `main.py`

Simplify the execution flow to strictly support `train_server` and `worker` modes.

* **Delete Functions:**
  - `training_mode(args)`
  - `evaluation_mode(args)`
  - `demo_mode(args)`
  - `debug_mode(args)`
* **Delete Unused Imports:**
  - Remove imports of `create_orchestrator` and `quick_evaluation` from `training_orchestrator`.
* **Modify `async_main()` CLI Argument Parser:**
  - Restrict `--mode` / `-m` choices strictly to `["train_server", "worker"]`, setting the default to `"train_server"`.
  - Remove options that were exclusive to deleted modes:
    - `--demo_episodes` / `-de`
    - `--debug_network`
    - `--debug_single_episode`
    - `--quick` / `-q`
  - Update execution routing block in `async_main()` to only route to `train_server_mode(args)` and `worker_mode(args)`.

### 2.2 `training_orchestrator.py`

Remove dead code and monolithic orchestration logic within `TrainingOrchestrator`.

* **Delete Class Methods:**
  - `collect(self)` (monolithic loop)
  - `_training_loop(self)` (monolithic loop)
  - `train_step(self)` (replaced by `_train_step` on server)
  - `run(self, max_steps)` (monolithic runner)
  - `stop_training(self)` (unused loop controller)
  - `run_single_episode(self)` (legacy debug workflow)
  - `run_parallel_demo(self, num_episodes)` (legacy demo workflow)
  - `run_evaluation(self, num_games)` (legacy evaluation workflow)
* **Delete Global Helper Functions:**
  - `create_orchestrator(config)`
  - `quick_evaluation(num_games, concurrent)`
* **Refactor `setup(self, is_collector: bool = False, is_evaluator: bool = False)`:**
  - Assert that at least one of `is_collector` or `is_evaluator` is `True`. If both are `False` (the old monolithic default), raise a `ValueError`.
  - Simplify the conditional blocks: since either `is_collector` or `is_evaluator` must be `True`, the block `if not is_collector:` is equivalent to `if is_evaluator:`.
  - Structure `setup()` clearly to only initialize either the collector worker or evaluator worker environments, eliminating all references to the monolithic non-worker configurations.

### 2.3 `tests/test_multiprocess_env_manager.py`

Fix a pre-existing import bug in the test setup.
* Add `import config` at the top level of the file so that the spawned `_mock_env_worker_3games` subprocess can properly reference `config.ACTION_DIM` without raising a `NameError`.

### 2.4 `readme.md`

Update the repository documentation to explain the modern, distributed-only training setup.
* **Remove sections** detailing legacy monolithic `--mode training`, `--mode eval`, `--mode demo`, `--mode debug`, and `benchmark_training.py`.
* **Detail the distributed workflow**:
  - Run the HTTP Training Server:
    ```bash
    ./run_server_distributed.sh
    ```
  - Run the cluster workers (evaluator + collectors):
    ```bash
    ./run_workers_distributed.sh
    ```
  - Document relevant options for distributed files, environment setup, and C++ MCTS compilation.

---

## 3. Edge Cases & Exception Paths

- **Illegal Setup Arguments:** If `TrainingOrchestrator.setup()` is called with `is_collector=False` and `is_evaluator=False`, it must raise a `ValueError("setup() requires either is_collector or is_evaluator to be True.")` to fail-fast.
- **Unroll Buffer Safety:** Ensure `_train_step` is only called when there is actually experience data available on the GPU server.
- **Backpressure Handling:** Ensure we preserve the `503 Service Unavailable` retry loops on collector/evaluator workers in `main.py` when the training server is under memory backpressure.
- **Graceful Termination:** Preserve signal trap configurations in `run_server_distributed.sh` and `run_workers_distributed.sh` so child processes are cleanly reaped.

---

## 4. Testing Strategy

### 4.1 Unit Tests
* **Assertion for setup block:** Add a new unit test in `tests/test_orchestrator_units.py` to assert that:
  ```python
  with pytest.raises(ValueError, match="setup\\(\\) requires either is_collector or is_evaluator to be True"):
      orch.setup(is_collector=False, is_evaluator=False)
  ```
* **Verify Remaining Suite:** Run pytest targeting the root `tests/` folder. All remaining tests (217 tests across 26 active files) must pass with zero errors.
  ```bash
  PYTHONPATH=.:MonteCarloTreeSearch:TFTSet4Gym pytest tests/
  ```

### 4.2 Local Integration & Verification
The Implementer must verify that the two distributed launchers can be run without syntax or import errors:
- Execute syntax check on modified scripts:
  ```bash
  python -m py_compile main.py training_orchestrator.py
  ```
