# TFT-227: Remove dead checkpoint config, unify on `SYNC_STEPS`

## Summary

`CHECKPOINT_STEPS` and its two consumer fields (`save_interval`, `evaluation_interval`) in `TrainingConfig` are dead code — never read at runtime. The actual checkpoint cadence is driven solely by `SYNC_STEPS` via `cfg.sync_steps`. This spec removes the dead constant, dead fields, dead CLI arguments, the dead shell-script flag, and updates documentation so the single source of truth is `SYNC_STEPS`.

## Motivation

Two config constants with identical values (`200`) create confusion about which one controls checkpointing. The Orchestrator confirmed that:
- `SYNC_STEPS` → `cfg.sync_steps` → `training_orchestrator.py:859` is the **only live path**.
- `CHECKPOINT_STEPS` → `cfg.save_interval` / `cfg.evaluation_interval` are **never consumed** at runtime.

Owner decision (comment `a31bb7ff`): remove the dead fields, keep `sync_steps` config-only, and stop passing `--checkpoint_interval` from the shell script.

---

## File Structure Changes

| File | Action | What changes |
|---|---|---|
| `config.py` | **Modify** | Delete `CHECKPOINT_STEPS = 200` (line 56). |
| `training_orchestrator.py` | **Modify** | Remove `save_interval` and `evaluation_interval` fields from `TrainingConfig` dataclass (lines 155–156). |
| `main.py` | **Modify** | Remove `cfg.evaluation_interval = ...` (line 30), `cfg.save_interval = ...` (line 34), `--eval_interval` arg (lines 463–464), `--checkpoint_interval` arg (lines 470–471). |
| `run_server_distributed.sh` | **Modify** | Remove `--checkpoint_interval 200` from the server launch command (line 58). |
| `readme.md` | **Modify** | Replace `CHECKPOINT_STEPS` reference with `SYNC_STEPS` and update its description (line 75). |
| `tests/test_orchestrator_units.py` | **Modify** | No changes needed — existing tests only reference `sync_steps` and `results_path`, not the dead fields. The `test_all_fields_have_defaults` and `test_custom_values` tests will pass because the deleted fields won't appear in `fields(TrainingConfig)` anymore. |
| `tests/test_profiling.py` | **No change** | Only uses `sync_steps` (line 327). |

---

## Interfaces & Signatures

### `config.py` — Constants

**Before:**
```python
SYNC_STEPS = 200
# ...
CHECKPOINT_STEPS = 200
```

**After:**
```python
SYNC_STEPS = 200
# CHECKPOINT_STEPS removed entirely
```

### `TrainingConfig` dataclass (`training_orchestrator.py`)

**Before:**
```python
@dataclass
class TrainingConfig:
    starting_train_step: int = 0
    run_name: str = ""
    save_interval: int = config.CHECKPOINT_STEPS
    evaluation_interval: int = config.CHECKPOINT_STEPS
    concurrent_games: int = config.CONCURRENT_GAMES
    ...
    sync_steps: int = config.SYNC_STEPS
    results_path: str = config.RESULTS_PATH
```

**After:**
```python
@dataclass
class TrainingConfig:
    starting_train_step: int = 0
    run_name: str = ""
    concurrent_games: int = config.CONCURRENT_GAMES
    ...
    sync_steps: int = config.SYNC_STEPS
    results_path: str = config.RESULTS_PATH
```

Fields removed: `save_interval`, `evaluation_interval`.

### `_build_config()` in `main.py`

**Before:**
```python
def _build_config(args) -> TrainingConfig:
    cfg = TrainingConfig()
    cfg.concurrent_games = getattr(args, "concurrent_games", config.CONCURRENT_GAMES)
    cfg.collect_games_per_batch = getattr(args, "collect_games", config.COLLECT_GAMES_PER_BATCH)
    cfg.evaluation_interval = getattr(args, "eval_interval", config.CHECKPOINT_STEPS)
    cfg.evaluation_games = getattr(args, "eval_games", config.EVALUATION_GAMES)
    cfg.evaluation_concurrent = getattr(args, "eval_concurrent", config.EVALUATION_CONCURRENT_GAMES)
    cfg.max_batch_size = getattr(args, "batch_size", config.BATCH_SIZE)
    cfg.save_interval = getattr(args, "checkpoint_interval", config.CHECKPOINT_STEPS)
    cfg.starting_train_step = getattr(args, "starting_episode", 0)
    cfg.run_name = getattr(args, "run_name", "")
    return cfg
```

**After:**
```python
def _build_config(args) -> TrainingConfig:
    cfg = TrainingConfig()
    cfg.concurrent_games = getattr(args, "concurrent_games", config.CONCURRENT_GAMES)
    cfg.collect_games_per_batch = getattr(args, "collect_games", config.COLLECT_GAMES_PER_BATCH)
    cfg.evaluation_games = getattr(args, "eval_games", config.EVALUATION_GAMES)
    cfg.evaluation_concurrent = getattr(args, "eval_concurrent", config.EVALUATION_CONCURRENT_GAMES)
    cfg.max_batch_size = getattr(args, "batch_size", config.BATCH_SIZE)
    cfg.starting_train_step = getattr(args, "starting_episode", 0)
    cfg.run_name = getattr(args, "run_name", "")
    return cfg
```

Lines removed: `cfg.evaluation_interval = ...`, `cfg.save_interval = ...`.

### CLI arguments in `main.py`

**Removed:**
```python
parser.add_argument("--eval_interval", "-ei", type=int, default=config.CHECKPOINT_STEPS)
parser.add_argument("--checkpoint_interval", "-ci", type=int, default=config.CHECKPOINT_STEPS)
```

### `run_server_distributed.sh`

**Before:**
```bash
PYTHON_GIL=0 ./run_tft.sh "$PYTHON_EXEC" main.py --mode train_server --checkpoint_interval 200 $EXTRA_ARGS &
```

**After:**
```bash
PYTHON_GIL=0 ./run_tft.sh "$PYTHON_EXEC" main.py --mode train_server $EXTRA_ARGS &
```

### `readme.md`

**Before:**
```markdown
- `CHECKPOINT_STEPS`: Interval for saving models and running evaluations.
```

**After:**
```markdown
- `SYNC_STEPS`: Interval (in training steps) for saving model checkpoints.
```

---

## Edge Cases

1. **External scripts passing `--checkpoint_interval` or `--eval_interval`:** After this change, `argparse` will reject these flags with an unrecognized-argument error. This is the desired behavior — any external caller still passing them should be updated. The only known caller is `run_server_distributed.sh`, which is updated in this spec.

2. **Code that accesses `cfg.save_interval` or `cfg.evaluation_interval`:** A full codebase search (`grep -rn`) confirms no code reads these fields outside of the dead assignment in `_build_config`. Removing them is safe.

3. **Serialized `TrainingConfig` objects:** `TrainingConfig` is a plain dataclass used only for in-process configuration — it is never serialized to disk or sent over the wire. No deserialization backward-compatibility concern.

4. **`config.CHECKPOINT_STEPS` import elsewhere:** A full search confirms the only imports/references to `CHECKPOINT_STEPS` are in `main.py` (lines 30, 34, 464, 471), `training_orchestrator.py` (lines 155–156), and `readme.md` (line 75) — all addressed in this spec.

---

## Testing Strategy

| Assertion | Rationale |
|---|---|
| `TrainingConfig()` has no `save_interval` attribute | Field removed |
| `TrainingConfig()` has no `evaluation_interval` attribute | Field removed |
| `TrainingConfig().sync_steps == config.SYNC_STEPS` | Sole checkpoint config still works |
| `config` module has no `CHECKPOINT_STEPS` attribute | Constant removed |
| Existing `test_default_creation` passes | No regressions in remaining fields |
| Existing `test_custom_values` passes | Tests will need `save_interval` and `evaluation_interval` kwargs removed if present (grep confirms they are not present in the custom-values test constructor call — safe) |
| Existing `test_all_fields_have_defaults` passes | Iterates `fields()` dynamically; removing fields shrinks the loop |
| `main.py` argparse does not accept `--checkpoint_interval` | CLI flag removed |
| `main.py` argparse does not accept `--eval_interval` | CLI flag removed |
| `run_server_distributed.sh` does not contain `--checkpoint_interval` | Shell flag removed |
