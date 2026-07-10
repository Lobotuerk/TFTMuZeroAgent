# Technical Specification - TFT-220: Benchmark Suite and Performance Profiling

## Overview
Following recent updates to agent memory management, performance regressions have been observed where agents—including non-ML ones—experience high latency per action (averaging ~4.7 seconds). 

To systematically diagnose, isolate, and continuously monitor these performance characteristics, this specification designs a standalone **Benchmark and Profiling Suite**. It enables:
1. **System Resource Monitoring**: Process RSS, Virtual Memory (VMS), system memory percentage, and PyTorch CUDA GPU allocated and peak memory.
2. **Execution Timing**: Precise tracking of game loop steps (`env.step()`), agent action selection times (time-per-action), and MCTS deep performance.
3. **Decoupled Architecture**: High modular depth to ensure profiling overhead does not pollute production code paths.
4. **Versioned Artifacts**: Committing structured JSON results to track performance evolution across Git commits.
5. **CI Integration**: Automation via GitHub Actions to block performance regressions in Pull Requests.

---

## 1. File Structure Changes
The following files will be created:
* **`benchmarks/__init__.py`**: Initializer to expose key module functions.
* **`benchmarks/core.py`**: Benchmark orchestrator, custom `MockEnv` for lightweight/CI profiling, system resource collector, and the decourated MCTS deep profiler.
* **`benchmarks/report.py`**: Performance JSON serializer, run comparator (with diff calculating), and Markdown report generator.
* **`benchmarks/run_benchmark.py`**: Command Line Interface (CLI) entry point.
* **`tests/test_benchmark_suite.py`**: Automated unit and integration tests for the benchmark and reporting framework.

The following files will be modified:
* **`.github/workflows/test.yml`**: Integrated with a benchmark run and comment generation on PR submissions.

---

## 2. Interfaces & Signatures

### `benchmarks/core.py`

#### 1. `class SystemMetrics`
Utility class for collecting cross-platform CPU, memory, and GPU stats.
* **`get_process_memory_info() -> Dict[str, float]`**
  * Returns process `rss_mb` and `vms_mb` using `psutil.Process()`.
* **`get_system_memory_percent() -> float`**
  * Returns `psutil.virtual_memory().percent`.
* **`get_gpu_memory_info() -> Dict[str, float]`**
  * Returns `allocated_mb` and `max_allocated_mb` from `torch.cuda.memory_allocated()` and `torch.cuda.max_memory_allocated()`. Gracefully returns `0.0` values if CUDA is not available.

#### 2. `class BenchmarkMockEnv`
A fast PettingZoo-compatible environment designed to run with realistic observation sizes (`config.OBSERVATION_SIZE`) and action masks (`sum(config.ACTION_DIM)`) without simulator compute overhead.
* **`__init__(self, num_players: int = 8, max_steps: int = 100)`**
* **`reset(self) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]`**
  * Returns observation dictionaries mapping player IDs to a structure with `"tensor"` and `"action_mask"`.
* **`step(self, actions: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]`**
  * Standard PettingZoo step signature. Increments step count and sets terminated flags when `max_steps` is reached.

#### 3. `class MCTSProfiler`
A decoupled profiler that temporarily intercepts MCTS operations when deep profiling is enabled.
* **`__enter__(self)`**
  * Monkey-patches `EnhancedMCTS.generate_action` and `BlockingBatchInferenceQueue._run_batch` to record execution durations.
* **`__exit__(self, exc_type, exc_val, exc_tb)`**
  * Restores original methods on exit to prevent any side effects or leaks.
* **`get_metrics() -> Dict[str, Any]`**
  * Exposes MCTS metrics such as MCTS action selection latency, MCTS GPU batch queue wait/execution time, and simulation rate (iters/sec).

#### 4. `class BenchmarkRunner`
Core orchestrator running simulated games and recording results.
* **`__init__(self, use_mock_env: bool = True, num_games: int = 1, steps_per_game: int = 50, agent_setup: str = "muzero_vs_random", mcts_simulations: int = 50, deep_mcts: bool = False)`**
* **`run(self) -> Dict[str, Any]`**
  * Orchestrates the complete benchmark run, measuring system memory, timing, and per-agent metrics, returning a nested JSON-compatible dictionary.

---

### `benchmarks/report.py`

#### `class BenchmarkReport`
Handles metric comparisons, Markdown reporting, and file operations.
* **`save(self, data: Dict[str, Any], filepath: str) -> None`**
  * Serializes and writes JSON results with directory auto-creation.
* **`load(self, filepath: str) -> Dict[str, Any]`**
* **`compare(self, current: Dict[str, Any], reference: Dict[str, Any]) -> Dict[str, Any]`**
  * Compares current vs reference metrics, calculating delta values and percentage changes.
* **`generate_markdown(self, current: Dict[str, Any], reference: Optional[Dict[str, Any]] = None) -> str`**
  * Generates a beautifully formatted Markdown report, containing a system info table, performance comparison matrix, per-agent latency breakdown, and deep MCTS statistics (if available).

---

### `benchmarks/run_benchmark.py`
The CLI entry point.
* **Command Line Arguments**:
  * `--num-games`: Number of games to run (default: `1`).
  * `--steps-per-game`: Maximum steps per game (default: `50` for fast profiling, `0` for infinite/full game).
  * `--agent-setup`: The agent configuration to run (e.g. `"muzero_vs_random"`, `"buying_agents"`, `"tournament"`).
  * `--mcts-simulations`: Number of simulations for MuZero agent MCTS (default: `10`).
  * `--real-env`: If specified, attempts to run with the real simulator environment (`TFTSet4Gym`) rather than `BenchmarkMockEnv`.
  * `--deep-mcts`: Activates the deep MCTS profiling mode.
  * `--compare-with`: Path to a reference JSON file for comparing regression stats.
  * `--output`: Path to write the JSON results file (defaults to `benchmarks/results/benchmark-<commit>-<timestamp>.json`).

---

### Output JSON Schema
```json
{
  "metadata": {
    "git_commit": "abc1234",
    "git_branch": "sdd/feature-TFT-220",
    "timestamp": "2026-07-10T12:00:00Z",
    "args": {
      "num_games": 1,
      "steps_per_game": 50,
      "agent_setup": "muzero_vs_random",
      "mcts_simulations": 10,
      "use_mock_env": true,
      "deep_mcts": false
    }
  },
  "system": {
    "rss_mb_start": 128.5,
    "rss_mb_end": 185.2,
    "vms_mb_start": 512.1,
    "vms_mb_end": 560.4,
    "system_memory_percent_avg": 42.1,
    "gpu_memory_allocated_mb_peak": 450.8,
    "gpu_memory_max_allocated_mb_peak": 890.1
  },
  "performance": {
    "total_duration_s": 15.4,
    "env_step_time_ms_avg": 5.2,
    "env_step_time_ms_median": 4.8,
    "env_step_time_ms_std": 1.1,
    "get_actions_time_ms_avg": 12.1
  },
  "agents": {
    "MuZeroAgent": {
      "total_actions": 400,
      "time_per_action_ms_avg": 8.5,
      "time_per_action_ms_median": 7.9,
      "average_batch_size": 1.0,
      "avg_inference_time_ms": 8.5
    },
    "RandomAgent": {
      "total_actions": 2000,
      "time_per_action_ms_avg": 0.12,
      "time_per_action_ms_median": 0.10,
      "average_batch_size": 5.0,
      "avg_inference_time_ms": 0.60
    }
  },
  "deep_mcts": {
    "generate_action_time_ms_avg": 12.4,
    "recurrent_inference_gpu_ms_avg": 2.1,
    "batch_queue_wait_ms_avg": 1.2,
    "total_mcts_simulations": 4000
  }
}
```

---

## 3. Edge Cases & Concurrency

### 1. No CUDA / CPU-only Environments
* **Edge Case**: Running on a machine or in a CI environment (like GitHub Actions free runners) where GPUs/CUDA are unavailable.
* **Handling**: All CUDA and PyTorch GPU memory queries are wrapped in try-except blocks and guarded with `torch.cuda.is_available()`. If unavailable, peak GPU allocations report as `0.0` or are cleanly omitted from visual reports without raising exceptions.

### 2. Missing External Libraries (`psutil` or `TFTSet4Gym`)
* **Edge Case**: Running without administrative privileges or required packages.
* **Handling**: Imports are checked. If `psutil` is missing, RSS and memory stats fall back gracefully to dummy/zero values and write a clean warning. If `TFTSet4Gym` is missing and `--real-env` is requested, the program raises a descriptive validation error suggesting the use of `--mock-env` (default behavior).

### 3. Concurrency Thread-Safety
* **Edge Case**: `BatchInferenceServer` processes concurrent requests from different environment steps or games on thread executors.
* **Handling**: The benchmark timing metrics are stored in a thread-safe sliding-window class (`MetricsCollector` or a thread-safe local metrics dict using `threading.Lock`) to prevent race conditions during parallel game loops.

---

## 4. Testing Strategy

The correctness of the profiling suite will be verified with the following automated test cases in `tests/test_benchmark_suite.py`:

1. **`test_system_metrics_retrieval`**
   * **Verification**: Call `SystemMetrics.get_process_memory_info()`, verify it return a dict containing float values for `rss_mb` and `vms_mb`.
2. **`test_benchmark_mock_env_pettingzoo_compliance`**
   * **Verification**: Run standard reset/step iterations over `BenchmarkMockEnv`. Verify the observation shapes align with `config.OBSERVATION_SIZE` and the action mask shapes align with `sum(config.ACTION_DIM)`.
3. **`test_benchmark_runner_mock_mode`**
   * **Verification**: Execute `BenchmarkRunner` in mock mode with 1 game and 10 steps. Assert that the output dictionary has all required schema fields (`system`, `performance`, `agents`) and contains non-null performance numbers.
4. **`test_mcts_profiler_activation_and_cleanup`**
   * **Verification**: Enter/exit the `MCTSProfiler` context, verify that monkey-patched methods are cleanly overwritten and safely restored to original class methods, and ensure that metrics are recorded.
5. **`test_benchmark_report_markdown_and_diff`**
   * **Verification**: Load two mock JSON result payloads. Call `BenchmarkReport.compare()`, assert that diff percentage calculations (e.g. current memory is +10% vs baseline) are mathematically correct. Verify markdown output is a valid non-empty string.
