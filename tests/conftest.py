"""
Pytest configuration and fixtures for PyFastFlow test suite.

This file contains shared fixtures, test configuration, and utilities
used across the test suite.
"""
import os
import sys
import pytest
import numpy as np


def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Add the package root to Python path for testing
    package_root = os.path.dirname(os.path.dirname(__file__))
    if package_root not in sys.path:
        sys.path.insert(0, package_root)


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers and ordering."""
    # Add slow marker to tests that might take time
    for item in items:
        # Mark GPU tests
        if "gpu" in item.keywords or "taichi" in str(item.function).lower():
            item.add_marker("slow")
            
        # Mark import tests for easy selection
        if "import" in item.name.lower() or "test_imports.py" in str(item.fspath):
            item.add_marker("importtest")


@pytest.fixture(scope="session")
def sample_elevation_data():
    """Provide sample elevation data for tests."""
    np.random.seed(42)  # For reproducible tests
    nx, ny = 50, 40
    elevation = np.random.rand(ny, nx) * 100.0
    return elevation, nx, ny


@pytest.fixture(scope="session")
def small_elevation_data():
    """Provide small elevation data for quick tests."""
    nx, ny = 10, 8
    elevation = np.random.rand(ny, nx) * 50.0
    return elevation, nx, ny


@pytest.fixture
def skip_if_no_taichi():
    """Skip test if Taichi is not available or fails to initialize."""
    try:
        import taichi as ti
        ti.init(arch=ti.cpu, offline_cache=False)
        return True
    except Exception:
        pytest.skip("Taichi not available or initialization failed")


@pytest.fixture
def skip_if_no_gpu():
    """Skip test if GPU/CUDA is not available."""
    try:
        import taichi as ti
        ti.init(arch=ti.gpu, offline_cache=False)
        return True
    except Exception:
        pytest.skip("GPU/CUDA not available")


@pytest.fixture
def skip_if_no_topotoolbox():
    """Skip test if TopoToolbox is not available."""
    try:
        import topotoolbox
        return True
    except ImportError:
        pytest.skip("TopoToolbox not available")


class TestDataManager:
    """Helper class for managing test data."""
    
    @staticmethod
    def create_simple_dem(nx=20, ny=15, dx=1.0):
        """Create a simple synthetic DEM for testing."""
        x = np.linspace(0, (nx-1)*dx, nx)
        y = np.linspace(0, (ny-1)*dx, ny)
        X, Y = np.meshgrid(x, y)
        
        # Create a simple landscape with a central peak
        cx, cy = nx//2 * dx, ny//2 * dx
        elevation = 100 * np.exp(-((X-cx)**2 + (Y-cy)**2) / (50*dx)**2)
        
        # Add some noise
        elevation += np.random.rand(ny, nx) * 5
        
        return elevation
    
    @staticmethod
    def create_drainage_pattern(nx=30, ny=25, dx=1.0):
        """Create a DEM with clear drainage patterns."""
        x = np.linspace(0, (nx-1)*dx, nx)
        y = np.linspace(0, (ny-1)*dx, ny)
        X, Y = np.meshgrid(x, y)
        
        # Create an inclined plane with a central valley
        elevation = 100 - 0.1*Y  # General slope
        valley = -20 * np.exp(-((X - nx//2*dx)**2) / (10*dx)**2)  # Central valley
        elevation += valley
        
        # Add small random perturbations
        elevation += np.random.rand(ny, nx) * 0.5
        
        return elevation


@pytest.fixture
def test_data_manager():
    """Provide access to test data creation utilities."""
    return TestDataManager()