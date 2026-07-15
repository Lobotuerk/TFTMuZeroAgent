# Technical Specification: Deterministic CI Benchmarks & Regression Gates
**Issue:** TFT-230

## 1. Overview
The current CI benchmark (`benchmarks.run_benchmark`) on self-hosted GPU runners fluctuates due to lack of seeding, concurrent inference timing, and MCTS exploration noise. This spec details how to introduce bit-level determinism for benchmarks and introduce regression gates for 5 specific metrics (Determinism, Embedding Fidelity, Latency, GPU Memory, Config Freeze) that fit within the current ~30s mock-env budget.

## 2. Seed Infrastructure
1. **Global Config:** Add `SEED = 42` to `config.py`.
2. **Seeding Utility:** Create `utils/seeding.py` with `set_seed(seed: int)` that sets seeds for:
   - `torch.manual_seed()`
   - `torch.cuda.manual_seed_all()`
   - `numpy.random.seed()`
   - `random.seed()`
   - `torch.backends.cudnn.deterministic = True`
   - `torch.backends.cudnn.benchmark = False`
3. **Propagation:**
   - Initialize the seed in `TrainingConfig`.
   - Propagate the seed through `TrainingOrchestrator` and `_MultiProcessEnvManager`.
   - Each subprocess must call `set_seed(seed)` upon spawn.

## 3. Deterministic Benchmark Mode
To achieve bit-level reproducibility (same seed -> identical JSON output) in benchmarks:
1. **Runner Parameter:** Add a `--seed` argument to `benchmarks/run_benchmark.py` and pass it to `BenchmarkRunner.__init__`, calling `set_seed(seed)` before starting.
2. **Disable Dirichlet Noise:** In benchmark mode, conditionally disable MCTS Dirichlet noise (e.g. check a benchmark flag inside `tft_mcts.py` or strictly enforce `self.mcts.training = False`).
3. **Seeded Random Agent:** Create a `SeededRandomAgent` (or pass seed to `RandomAgent`) in `Models.Common_agents` that initializes a seeded random number generator for action sampling, replacing the standard unseeded random calls.
4. **CI Workflow:** Update `.github/workflows/test.yml` to pass `--seed 42` to `benchmarks.run_benchmark`.

## 4. CI Metrics & Regression Gates
Add the following metrics to the benchmark pipeline and gate on them to catch regressions from recent PRs:

1. **Determinism Score**
   - *Implementation:* Run the benchmark twice with `--seed 42`. Compare the generated JSON artifacts.
   - *Gate:* Assert 100% identical outputs. (Catches thread-ordering/batching non-determinism from PR #41).
2. **Embedding Fidelity**
   - *Implementation:* Run one forward pass of `RepNetwork` with a fixed mock input tensor, and compare it against a hardcoded golden reference tensor using cosine similarity.
   - *Gate:* Cosine similarity > 0.999. (Catches numerical drift from vectorization in PR #43).
3. **Inference Latency P50/P95**
   - *Implementation:* The `MetricsStore` already collects this. Add assertion checks against a static baseline JSON.
   - *Gate:* P95 Latency must be < 1.5x of the baseline. (Catches UDS overhead, serialization bloat from PR #44/45).
4. **GPU Memory Stability**
   - *Implementation:* The `MetricsStore` already measures allocated MB. Calculate the standard deviation across benchmark steps.
   - *Gate:* Stddev must be < Threshold. (Catches memory thrashing from concurrent inference from PR #41).
5. **Config Freeze Check**
   - *Implementation:* At the start of the benchmark, read all keys from `config.py`.
   - *Gate:* Assert that a predefined set of crucial config keys are present and match expected types/values. (Catches accidental config deletions from PR #42).

## 5. Security & Side-Effects
- Setting `cudnn.deterministic = True` may have slight performance impacts on the forward pass, but it is necessary for reproducibility.
- Deterministic behavior applies to the benchmark pipeline and is opted-in via the `--seed` flag. Training mode continues to benefit from unseeded exploration if no seed is provided.

## 6. Development Checklist
- [ ] Create `utils/seeding.py` and add `SEED` to `config.py`.
- [ ] Implement `set_seed` propagation in orchestrator and multiprocessing environments.
- [ ] Implement `--seed` in `run_benchmark.py` and `BenchmarkRunner`.
- [ ] Modify `RandomAgent` or create `SeededRandomAgent`.
- [ ] Turn off Dirichlet noise in MCTS when in benchmark mode.
- [ ] Add the 5 regression checks (Determinism, Embedding Fidelity, Latency, GPU Memory, Config Freeze) to `benchmarks/run_benchmark.py` or a wrapper script.
- [ ] Update `.github/workflows/test.yml` to enforce these gates.
