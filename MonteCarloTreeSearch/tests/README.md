# 🧪 PyMCTS Test Suite

This folder contains a consolidated pytest-based test suite for the PyMCTS library. All redundant tests have been removed and functionality has been organized into focused test modules.

## 📁 Test Files

### Core Test Modules
- **`test_core.py`** - Core functionality tests (module import, TicTacToe, MCTS agent)
- **`test_parallel.py`** - Parallel rollout and threading tests  
- **`test_python_inheritance.py`** - Python inheritance from C++ base classes
- **`conftest.py`** - Pytest configuration and shared fixtures

### Configuration
- **`pytest.ini`** - Pytest configuration (in project root)
- **`run_tests.py`** - Test runner script (in project root)

## 🚀 How to Run Tests

### Quick Start
```bash
# From project root directory
python run_tests.py
```

### Using pytest directly
```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/test_core.py
pytest tests/test_parallel.py  
pytest tests/test_python_inheritance.py

# Run with verbose output
pytest tests/ -v

# Run only fast tests (skip slow performance tests)
pytest tests/ -m "not slow"

# Run only slow tests
pytest tests/ -m "slow"
```

## 🎯 Test Categories

### ✅ Core Functionality (`test_core.py`)
- **Module Import Tests**: Verify PyMCTS imports correctly
- **Hardware Info Tests**: Check thread configuration functions
- **TicTacToe Tests**: Built-in C++ TicTacToe functionality
- **MCTS Agent Tests**: Agent creation and move generation
- **Performance Tests**: Basic timing and iteration tests

### ⚡ Parallel Rollouts (`test_parallel.py`)
- **Configuration Tests**: Thread setting and hardware limits
- **Performance Tests**: Single vs multi-threaded comparison
- **Thread Safety Tests**: Concurrent agents and thread changes
- **Stress Tests**: High-iteration and multiple game tests (marked as `slow`)

### � Python Inheritance (`test_python_inheritance.py`)
- **Inheritance Tests**: Python classes inherit from C++ bases
- **Integration Tests**: Python games work with MCTS agent
- **Connect Four Tests**: Complex Python game example
- **Trampoline Tests**: Edge cases and robustness

## 📊 Test Organization Benefits

### ✅ Consolidated Structure
- **Removed 8 redundant test files** with overlapping functionality
- **3 focused test modules** covering all functionality
- **Pytest framework** for better test organization and reporting

### ✅ Improved Maintainability  
- **Shared fixtures** in `conftest.py` eliminate code duplication
- **Clear test categories** make it easy to run specific test types
- **Consistent naming** and structure across all tests

### ✅ Better Performance
- **Fast vs slow test separation** using pytest markers
- **Parallel test execution** possible with pytest-xdist
- **Selective test running** for development workflow

## 🔧 Development Workflow

### For Bug Fixes:
```bash
pytest tests/test_core.py              # Verify core functionality
```

### For Performance Changes:
```bash
pytest tests/test_parallel.py          # Test parallel rollouts
```

### For Python Integration:
```bash
pytest tests/test_python_inheritance.py # Test Python games
```

### For Full Validation:
```bash
python run_tests.py                    # Complete test suite
```

## 🐛 Debugging

### Test Failures
```bash
# Run with more verbose output
pytest tests/ -v -s

# Run single test for debugging
pytest tests/test_core.py::TestModuleImport::test_module_imports -v -s

# Show local variables on failure
pytest tests/ --tb=long
```

### Performance Issues
```bash
# Run only performance tests
pytest tests/test_parallel.py::TestParallelPerformance -v

# Skip slow tests during development
pytest tests/ -m "not slow"
```

## � Test Coverage

### What's Tested:
- ✅ **Module Import & Configuration** - All core functions
- ✅ **C++ TicTacToe Integration** - Reference implementation
- ✅ **MCTS Agent Functionality** - Move generation, state tracking
- ✅ **Parallel Rollout Performance** - 1-N thread configurations
- ✅ **Thread Safety** - Concurrent operations, thread changes
- ✅ **Python Inheritance** - Trampoline classes, method overrides
- ✅ **Complex Python Games** - Connect Four example
- ✅ **Edge Cases** - Error handling, robustness

### Performance Benchmarks:
- Single vs multi-threaded rollout comparison
- Thread scaling characteristics (1, 2, 4, 8+ threads)
- High-iteration stress tests
- Multiple concurrent game simulations

## 🎉 Summary

The test suite has been **completely reorganized** from 9 redundant files to 3 focused modules:

**Before**: 9 overlapping test files with duplicated code
**After**: 3 organized test modules with pytest framework

**Benefits**:
- � **Easier maintenance** - No more duplicate code
- ⚡ **Faster execution** - Skip slow tests during development  
- 📊 **Better reporting** - Pytest's clear output format
- 🎯 **Focused testing** - Run only what you need
- 🚀 **Professional structure** - Industry-standard pytest framework

All PyMCTS functionality is thoroughly tested with improved organization and maintainability!