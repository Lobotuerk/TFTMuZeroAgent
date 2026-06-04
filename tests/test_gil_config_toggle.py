import sys
import os
from unittest.mock import patch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from training_orchestrator import (
    TrainingOrchestrator,
    _ThreadEnvManager,
    _MultiProcessEnvManager,
)

def test_gil_disabled_detection():
    # Verify IS_GIL_DISABLED is defined and of boolean type
    assert hasattr(config, "IS_GIL_DISABLED")
    assert isinstance(config.IS_GIL_DISABLED, bool)

def test_force_threading_env_manager_config():
    # Verify FORCE_THREADING_ENV_MANAGER is defined and is of boolean type
    assert hasattr(config, "FORCE_THREADING_ENV_MANAGER")
    assert isinstance(config.FORCE_THREADING_ENV_MANAGER, bool)

def test_create_env_manager_factory():
    # Scenario 1: Both GIL active (IS_GIL_DISABLED=False) and FORCE_THREADING=False -> MultiProcess
    with patch("config.IS_GIL_DISABLED", False), \
         patch("config.FORCE_THREADING_ENV_MANAGER", False):
        mgr = TrainingOrchestrator._create_env_manager(2)
        assert isinstance(mgr, _MultiProcessEnvManager)
        assert not isinstance(mgr, _ThreadEnvManager)

    # Scenario 2: GIL disabled (IS_GIL_DISABLED=True) and FORCE_THREADING=False -> MultiProcess
    with patch("config.IS_GIL_DISABLED", True), \
         patch("config.FORCE_THREADING_ENV_MANAGER", False):
        mgr = TrainingOrchestrator._create_env_manager(2)
        assert isinstance(mgr, _MultiProcessEnvManager)
        assert not isinstance(mgr, _ThreadEnvManager)

    # Scenario 3: GIL active (IS_GIL_DISABLED=False) but FORCE_THREADING=True -> Thread
    with patch("config.IS_GIL_DISABLED", False), \
         patch("config.FORCE_THREADING_ENV_MANAGER", True):
        mgr = TrainingOrchestrator._create_env_manager(2)
        assert isinstance(mgr, _ThreadEnvManager)
        assert not isinstance(mgr, _MultiProcessEnvManager)

    # Scenario 4: Both True -> Thread
    with patch("config.IS_GIL_DISABLED", True), \
         patch("config.FORCE_THREADING_ENV_MANAGER", True):
        mgr = TrainingOrchestrator._create_env_manager(2)
        assert isinstance(mgr, _ThreadEnvManager)
        assert not isinstance(mgr, _MultiProcessEnvManager)


def test_main_check_gil():
    from main import _check_gil as main_check_gil

    # Scenario 1: GIL active (IS_GIL_DISABLED=False) and FORCE_THREADING=True -> Should raise SystemExit(1)
    with patch("config.IS_GIL_DISABLED", False), \
         patch("config.FORCE_THREADING_ENV_MANAGER", True):
        with pytest.raises(SystemExit) as excinfo:
            main_check_gil()
        assert excinfo.value.code == 1

    # Scenario 2: GIL disabled (IS_GIL_DISABLED=True) and FORCE_THREADING=True -> Should pass
    with patch("config.IS_GIL_DISABLED", True), \
         patch("config.FORCE_THREADING_ENV_MANAGER", True):
        main_check_gil()  # No exception should be raised

    # Scenario 3: GIL active (IS_GIL_DISABLED=False) and FORCE_THREADING=False -> Should pass
    with patch("config.IS_GIL_DISABLED", False), \
         patch("config.FORCE_THREADING_ENV_MANAGER", False):
        main_check_gil()  # No exception should be raised


def test_benchmark_check_gil():
    from benchmark_training import _check_gil as bench_check_gil

    # Scenario 1: GIL active (IS_GIL_DISABLED=False) and FORCE_THREADING=True -> Should raise SystemExit(1)
    with patch("config.IS_GIL_DISABLED", False), \
         patch("config.FORCE_THREADING_ENV_MANAGER", True):
        with pytest.raises(SystemExit) as excinfo:
            bench_check_gil()
        assert excinfo.value.code == 1

    # Scenario 2: GIL disabled (IS_GIL_DISABLED=True) and FORCE_THREADING=True -> Should pass
    with patch("config.IS_GIL_DISABLED", True), \
         patch("config.FORCE_THREADING_ENV_MANAGER", True):
        bench_check_gil()  # No exception should be raised

    # Scenario 3: GIL active (IS_GIL_DISABLED=False) and FORCE_THREADING=False -> Should pass
    with patch("config.IS_GIL_DISABLED", False), \
         patch("config.FORCE_THREADING_ENV_MANAGER", False):
        bench_check_gil()  # No exception should be raised

