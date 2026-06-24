# Technical Specification for TFT-196: Test CI

## Problem
PR #12 introduced significant model changes including n-step bootstrap, zero-reward synthesis, constructor-based network refactoring, and new metrics. However, there are no tests covering these changes, and there is no CI pipeline configured to automatically run tests on PRs and main branches. This led to uncaught regressions.

## Proposed Changes

1. **GitHub Actions CI Pipeline (`.github/workflows/test.yml`)**
   - **Trigger:** Push and pull requests to `main`, `sdd/**`, `feature/**`, and `fix/**` branches.
   - **Environment:** Ubuntu-latest with Python 3.10. Ensure recursive submodule checkouts are enabled.
   - **Dependencies:** Install requirements from `requirements.txt`.
   - **Tests:** Execute `pytest` over the `tests/` directory.

2. **Test Infrastructure**
   - Create `tests/__init__.py` to correctly establish the tests package for module discovery by `pytest`.

3. **New Pytest Files for PR #12 Features**
   - `tests/test_n_step_bootstrap.py`: 
     - Verify n-step bootstrap computation.
     - Check $\gamma^n$ discount applications.
     - Validate bootstrap depth limits.
   - `tests/test_model_refactor.py`:
     - Test `PredNetwork`, `DynNetwork`, and `RepNetwork` instantiations.
     - Validate layer constructors and ensure parameter initialization matches configuration.
     - Test forward pass output shapes and embedding distributions.
   - `tests/test_metrics.py`:
     - Ensure policy entropy correctly penalizes deterministic outputs and validates non-negativity.
     - Test value Mean Absolute Error (MAE) computation.
   - `tests/test_dynamics_zero_reward.py`:
     - Test dynamic network assumptions for the zero-reward synthesis (terminal-only environments).
     - Test reward device and batch consistency.

## Design Philosophy Rules
- **High Modular Depth:** Keep the pipeline definitions strictly isolated. Ensure tests reuse any shared setup via clear fixtures rather than deep inheritance.
- **Clean Exception Paths:** The CI should cleanly fail with descriptive test names if assertions do not pass.

## Review Constraints
The implementation must ensure all 4 new test files are valid under `pytest` and that `.github/workflows/test.yml` strictly adheres to standard GitHub Actions schema.