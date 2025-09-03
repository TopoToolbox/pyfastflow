"""
Simple tests for rastermanip module that avoid Taichi re-initialization issues.

Tests basic functionality without complex setup/teardown.

Author: B.G.
"""

import numpy as np
import pytest
import taichi as ti

from pyfastflow.rastermanip import (
    double_resolution,
    halve_resolution,
    resize_raster,
)

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
    original = np.array(
        [[1, 2, 5, 6], [3, 4, 7, 8], [9, 10, 13, 14], [11, 12, 15, 16]],
        dtype=np.float32,
    )

    result = halve_resolution(original, method="mean")

    # Check dimensions
    assert result.shape == (2, 2)

    # Check mean calculations
    expected = np.array([[2.5, 6.5], [10.5, 14.5]], dtype=np.float32)
    assert np.allclose(result, expected)


def test_halve_methods():
    """Test different halving methods."""
    original = np.array(
        [[1, 2, 5, 6], [3, 4, 7, 8], [9, 10, 13, 14], [11, 12, 15, 16]],
        dtype=np.float32,
    )

    # Test different methods
    result_max = halve_resolution(original, method="max")
    result_min = halve_resolution(original, method="min")
    result_mean = halve_resolution(original, method="mean")

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
        halve_resolution(original, method="invalid")


def test_odd_dimensions_error():
    """Test error handling for odd dimensions."""
    with pytest.raises(ValueError):
        halve_resolution(np.ones((3, 4), dtype=np.float32))


def test_double_resolution_taichi_field_non_square():
    """Double resolution should handle non-square Taichi fields."""
    field = ti.field(dtype=ti.f32, shape=(3, 5))
    for j in range(3):
        for i in range(5):
            field[j, i] = j * 5 + i

    result = double_resolution(field, noise_amplitude=0.0)
    assert result.shape == (6, 10)


def test_halve_resolution_taichi_field_non_square():
    """Halving should handle non-square Taichi fields."""
    field = ti.field(dtype=ti.f32, shape=(4, 6))
    for j in range(4):
        for i in range(6):
            field[j, i] = j * 6 + i

    result = halve_resolution(field, method="mean")
    assert result.shape == (2, 3)


def test_halve_cubic_linear_plane():
    """Cubic halving should reproduce values of a linear plane."""
    nx, ny = 8, 8
    grid = np.zeros((ny, nx), dtype=np.float32)
    for j in range(ny):
        for i in range(nx):
            grid[j, i] = i + j

    result = halve_resolution(grid, method="cubic")

    def cubic_interp(v0, v1, v2, v3, t):
        a = -0.5 * v0 + 1.5 * v1 - 1.5 * v2 + 0.5 * v3
        b = v0 - 2.5 * v1 + 2.0 * v2 - 0.5 * v3
        c = -0.5 * v0 + 0.5 * v2
        d = v1
        return ((a * t + b) * t + c) * t + d

    expected = np.zeros((ny // 2, nx // 2), dtype=np.float32)
    for tj in range(ny // 2):
        for ti_out in range(nx // 2):
            center_j = tj * 2.0 + 0.5
            center_i = ti_out * 2.0 + 0.5
            base_j = int(np.floor(center_j))
            base_i = int(np.floor(center_i))
            frac_j = center_j - base_j
            frac_i = center_i - base_i
            row_vals = []
            for dj in range(4):
                jj = np.clip(base_j + dj - 1, 0, ny - 1)
                samples = []
                for di in range(4):
                    ii = np.clip(base_i + di - 1, 0, nx - 1)
                    samples.append(grid[jj, ii])
                row_vals.append(cubic_interp(*samples, frac_i))
            expected[tj, ti_out] = cubic_interp(*row_vals, frac_j)

    assert np.allclose(result, expected)


def test_double_resolution_seed_reproducible():
    original = np.arange(16, dtype=np.float32).reshape(4, 4)
    r1 = double_resolution(original, noise_amplitude=0.5, seed=42)
    r2 = double_resolution(original, noise_amplitude=0.5, seed=42)
    r3 = double_resolution(original, noise_amplitude=0.5, seed=43)
    assert np.allclose(r1, r2)
    assert not np.allclose(r1, r3)


def test_halve_custom_kernel():
    original = np.zeros((4, 4), dtype=np.float32)

    @ti.kernel
    def fill_one(src: ti.template(), tgt: ti.template(), nx: ti.i32, ny: ti.i32):
        for i in tgt:
            tgt[i] = 1.0

    result = halve_resolution(original, kernel=fill_one)
    assert np.all(result == 1.0)


def test_boundary_option_wrap_changes_result():
    grid = np.array([[1, 2], [3, 4]], dtype=np.float32)
    clamp_res = double_resolution(grid, noise_amplitude=0.0, boundary="clamp")
    wrap_res = double_resolution(grid, noise_amplitude=0.0, boundary="wrap")
    assert not np.allclose(clamp_res, wrap_res)


def test_resize_raster_downscale_linear_plane():
    nx, ny = 4, 4
    grid = np.zeros((ny, nx), dtype=np.float32)
    for j in range(ny):
        for i in range(nx):
            grid[j, i] = i + j

    result = resize_raster(grid, 0.5)
    expected = np.array([[1, 3], [3, 5]], dtype=np.float32)
    assert np.allclose(result, expected)
