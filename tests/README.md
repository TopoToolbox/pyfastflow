# PyFastFlow Test Suite

This directory contains the test suite for PyFastFlow, built using pytest.

## Structure

```
tests/
├── __init__.py                    # Test package initialization
├── conftest.py                    # Pytest configuration and fixtures
├── test_imports.py                # Import tests for all modules
├── unit/                          # Unit tests
│   ├── __init__.py
│   ├── test_cli.py               # CLI functionality tests
│   └── test_general_algorithms.py # General algorithm tests
├── integration/                   # Integration tests
│   ├── __init__.py
│   └── test_basic_workflows.py   # Basic workflow tests
└── README.md                     # This file
```

## Running Tests

### Basic Usage

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run only import tests
pytest -m import

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run specific test file
pytest tests/test_imports.py
```

### Test Markers

The following markers are available:

- `import`: Import tests for modules
- `unit`: Unit tests
- `integration`: Integration tests
- `slow`: Tests that might take longer to run
- `gpu`: Tests requiring GPU/Taichi functionality

### Running Specific Test Categories

```bash
# Run only import tests
pytest -m import

# Run unit tests excluding slow ones
pytest -m "unit and not slow"

# Run all tests except GPU tests
pytest -m "not gpu"

# Run only fast tests
pytest -m "not slow"
```

## Test Dependencies

### Required
- `pytest>=6.0`
- `numpy`
- `click` (for CLI tests)

### Optional
- `taichi` (for GPU/compute tests)
- `topotoolbox` (for I/O tests)
- `matplotlib` (for visualization tests)

## Adding New Tests

### Import Tests
Add new import tests to `test_imports.py` when adding new modules.

### Unit Tests
Create new test files in `tests/unit/` for testing individual functions and classes:

```python
# tests/unit/test_new_module.py
import pytest

class TestNewModule:
    @pytest.mark.unit
    def test_basic_functionality(self):
        from pyfastflow.new_module import new_function
        assert new_function() is not None
```

### Integration Tests
Add integration tests to `tests/integration/` for testing complete workflows:

```python
# tests/integration/test_new_workflow.py
import pytest

class TestNewWorkflow:
    @pytest.mark.integration
    @pytest.mark.slow
    def test_complete_workflow(self, test_data_manager):
        # Test complete workflow here
        pass
```

## Fixtures

Common fixtures available in all tests:

- `sample_elevation_data`: Large sample elevation data for testing
- `small_elevation_data`: Small elevation data for quick tests  
- `test_data_manager`: Utility class for creating test data
- `skip_if_no_taichi`: Skip test if Taichi not available
- `skip_if_no_gpu`: Skip test if GPU not available
- `skip_if_no_topotoolbox`: Skip test if TopoToolbox not available

## Continuous Integration

Tests are designed to work in CI environments:

- Import tests run without external dependencies
- GPU tests are skipped when GPU not available
- Optional dependency tests are skipped when dependencies missing
- Fast tests can be run separately from slow tests

## Test Data

Test data is generated programmatically using utilities in `conftest.py`:

- `TestDataManager.create_simple_dem()`: Simple synthetic DEM
- `TestDataManager.create_drainage_pattern()`: DEM with drainage patterns
- Fixtures provide consistent test data across tests