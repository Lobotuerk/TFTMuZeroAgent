# Technical Specification: TFT-217 Repo Cleanup & Distributed Optimization

## 1. File Structure Changes

The repository is being cleaned up to support **strictly the distributed training workflow** (GPU server with HTTP API + CPU workers for self-play/evaluation). All legacy, monolithic, benchmark-specific, and standalone demo files are to be deleted, along with their respective test suites.

### Files to Delete
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

### Files to Modify
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
  - Assert that at least one of `is_collector` or `is_evaluator` is `True`. If both are `False`, raise a `ValueError`.
  - Simplify the conditional blocks: since either `is_collector` or `is_evaluator` must be `True`, the block `if not is_collector:` is equivalent to `if is_evaluator:`.
  - Structure `setup()` clearly to only initialize either the collector worker or evaluator worker environments.

### 2.3 `tests/test_multiprocess_env_manager.py`

Fix a pre-existing import bug in the test setup.
* Add `import config` at the top level so the spawned subprocess can reference `config.ACTION_DIM`.

### 2.4 `readme.md`

Update the repository documentation to explain the modern, distributed-only training setup.
* **Remove sections** detailing legacy monolithic modes.
* **Detail the distributed workflow** with `run_server_distributed.sh` and `run_workers_distributed.sh`.

---

## 3. Edge Cases & Exception Paths

- **Illegal Setup Arguments:** If `TrainingOrchestrator.setup()` is called with `is_collector=False` and `is_evaluator=False`, raise `ValueError`.
- **Graceful Termination:** Preserve signal trap configurations in launcher scripts.

---

## 4. Testing Strategy

### 4.1 Unit Tests
* Add a unit test asserting `ValueError` for invalid setup args.
* Verify remaining suite passes.

### 4.2 Verification
```bash
python -m py_compile main.py training_orchestrator.py
```