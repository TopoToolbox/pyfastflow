"""
Simple tests for rastermanip module that avoid Taichi re-initialization issues.

Tests basic functionality without complex setup/teardown.

Author: B.G.
"""

import pytest
import numpy as np
import taichi as ti
from pyfastflow.rastermanip import double_resolution, halve_resolution


# Initialize Taichi once for the entire test module
ti.init(arch=ti.cpu)


def test_import_functions():
    """Test that functions can be imported."""
    assert callable(double_resolution)
    assert callable(halve_resolution)


def test_basic_double_resolution():
    """Test basic doubling of resolution."""
    # Simple 2x2 test grid
    original = np.array([[1, 2], [3, 4]], dtype=np.float32)
    
    # Double resolution
    result = double_resolution(original, noise_amplitude=0.0)
    
    # Check output dimensions
    assert result.shape == (4, 4)
    assert np.all(np.isfinite(result))


def test_basic_halve_resolution():
    """Test basic halving of resolution."""
    # Simple 4x4 test grid where each 2x2 block has known values
    original = np.array([
        [1, 2, 5, 6],
        [3, 4, 7, 8],
        [9, 10, 13, 14],
        [11, 12, 15, 16]
    ], dtype=np.float32)
    
    result = halve_resolution(original, method='mean')
    
    # Check dimensions
    assert result.shape == (2, 2)
    
    # Check mean calculations
    expected = np.array([[2.5, 6.5], [10.5, 14.5]], dtype=np.float32)
    assert np.allclose(result, expected)


def test_halve_methods():
    """Test different halving methods."""
    original = np.array([
        [1, 2, 5, 6],
        [3, 4, 7, 8],
        [9, 10, 13, 14], 
        [11, 12, 15, 16]
    ], dtype=np.float32)
    
    # Test different methods
    result_max = halve_resolution(original, method='max')
    result_min = halve_resolution(original, method='min')
    result_mean = halve_resolution(original, method='mean')
    
    # All should have same dimensions
    assert result_max.shape == (2, 2)
    assert result_min.shape == (2, 2)
    assert result_mean.shape == (2, 2)
    
    # Max should be >= mean >= min
    assert np.all(result_max >= result_mean)
    assert np.all(result_mean >= result_min)


def test_invalid_method():
    """Test error handling for invalid method."""
    original = np.ones((4, 4), dtype=np.float32)
    
    with pytest.raises(ValueError):
        halve_resolution(original, method='invalid')


def test_odd_dimensions_error():
    """Test error handling for odd dimensions."""
    with pytest.raises(ValueError):
        halve_resolution(np.ones((3, 4), dtype=np.float32))