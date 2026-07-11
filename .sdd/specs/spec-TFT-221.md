# Technical Specification - TFT-221: Fix CI

## Overview
The current CI workflow in `TFTMuZeroAgent` contains a `benchmark` job that runs on a GitHub-hosted CPU runner (`ubuntu-latest`). This setup has several environment limitations:
1. It downloads ~2GB of PyTorch with CUDA support on every push, which consumes substantial time and often causes installation or timeout failures on a CPU-only runner.
2. Without a GPU, the benchmark fallback to all-random agents runs on CPU MCTS, making the benchmark results non-representative of the actual MuZero GPU throughput.
3. The GPU-specific batched inference benchmark (`tests/test_batch_benchmark.py`) is completely excluded from the CI run because it cannot assert speedups without a GPU.

To address these limitations, we will refactor the CI benchmark suite to run on the local machine (`Lobo-NEW`), which contains an RTX 5070 Ti with CUDA 13.3 and a pre-configured `TFT` conda environment.

This specification outlines the design to:
1. Transition the `benchmark` job to target `runs-on: [self-hosted, gpu]`.
2. Configure the workflow to run commands using the pre-installed local `TFT` conda environment via `conda run -n TFT`, avoiding 2GB PyTorch downloads and speeding up the CI run.
3. Integrate submodule dependency linking in a non-destructive manner for the self-hosted environment.
4. Run `test_batch_benchmark.py` within the `benchmark` job on the self-hosted GPU runner, verifying both correctness and actual GPU speedup performance.
5. Generate and publish benchmark metrics to GitHub Actions artifacts as `ci-benchmark.json`.

---

## 1. File Structure Changes
The following files will be modified:
* **`.github/workflows/test.yml`**: Refactor the `benchmark` job to run on the self-hosted GPU runner and execute the benchmark and speedup tests using the local conda environment.

No other files are created, modified, or deleted.

---

## 2. Interfaces & Signatures

### GitHub Actions Workflow Refactoring (`.github/workflows/test.yml`)
The `test` job will remain on `ubuntu-latest` for fast CPU-only unit test validation.

The `benchmark` job will be modified as follows:

1. **Runner Target**:
   * Change `runs-on: ubuntu-latest` to `runs-on: [self-hosted, gpu]`.

2. **Conda Environment Execution**:
   * Use `conda run -n TFT <command>` to run commands within the pre-installed conda environment on `Lobo-NEW`.
   * Eliminate standard python setup (`actions/setup-python@v5`) and general `pip install -r requirements.txt` from the GHA steps for the self-hosted job, as the host's `TFT` conda environment is already fully pre-installed and optimized with PyTorch+CUDA and other packages.

3. **Submodule Link Step**:
   * For checked-out submodules, link `TFTSet4Gym` in editable mode inside the `TFT` environment if present in the workspace:
     `conda run -n TFT pip install -e TFTSet4Gym`
   * Check and build the C++ extension `pymcts` if it has local modifications or needs linking:
     ```bash
     if [ -d "MonteCarloTreeSearch" ] && [ -f "MonteCarloTreeSearch/setup.py" ]; then
       conda run -n TFT pip install -e MonteCarloTreeSearch || (cd MonteCarloTreeSearch && conda run -n TFT python setup.py build_ext --inplace)
     fi
     ```

4. **Speedup Benchmark Test Step**:
   * Run the GPU speedup benchmark test using pytest:
     `conda run -n TFT pytest tests/test_batch_benchmark.py -v`

5. **Benchmark Suite Run Step**:
   * Execute the main benchmark command through the conda environment:
     ```bash
     conda run -n TFT python -m benchmarks.run_benchmark \
       --num-games 1 \
       --steps-per-game 20 \
       --agent-setup muzero_vs_random \
       --mcts-simulations 5 \
       --output benchmarks/results/ci-benchmark.json
     ```

---

## 3. Edge Cases & Environment Handling

### CUDA Check Failover & Device Isolation
* The self-hosted runner will execute on a machine with a physical GPU. `test_batch_benchmark.py` expects `torch.cuda.is_available()` to be `True`. If CUDA is somehow unavailable or blocked by another process, `test_batch_benchmark.py` will fail to assert the `ratio >= 1.5` speedup.
* **Mitigation**: The specification ensures the Coder agent verifies CUDA is available inside the GHA workflow job step before launching tests. If `nvidia-smi` or PyTorch CUDA is not responsive, the step will fail explicitly with a clear log message rather than a silent failure or hanging.

### Submodule Workspace Cleaning
* Since self-hosted runners persist their workspace directories (`_work`), consecutive runs might leave untracked files or compiled `.so` objects.
* **Mitigation**: Ensure that the `actions/checkout@v4` step with `submodules: recursive` is configured correctly, and if needed, clean up compiled cache files before compiling the extension.

### Concurrency and GPU VRAM Contention
* The host machine `Lobo-NEW` runs agent workloads alongside GHA. While the benchmark suite is lightweight (~1-2 GB VRAM with 1 game and MCTS-simulations=5), standard GHA job executions might overlap.
* **Mitigation**: The benchmark run is kept extremely small (`--num-games 1 --steps-per-game 20`) so that execution is sub-minute and resource usage is negligible, minimizing any potential contention.

---

## 4. Testing Strategy

The correctness of the design and setup will be verified via the following steps:

1. **Pre-Execution Validation**:
   * Verify that the GHA workflow file parses correctly and contains no syntax errors.

2. **Integration Verification on self-hosted runner**:
   * Push the workflow changes to the remote feature branch `sdd/feature-TFT-221`.
   * Monitor the triggered GitHub Actions workflow run for `TFT-221`.
   * Verify that the `benchmark` job successfully executes on `[self-hosted, gpu]`.
   * Confirm that `tests/test_batch_benchmark.py` passes successfully with actual GPU-batched speedup reported in the test logs.
   * Confirm that the benchmark suite `ci-benchmark.json` file is generated, contains valid metrics (no fallback to all-random unless specified), and is successfully uploaded to GitHub Actions artifacts.
