"""
Raster manipulation tests for the context-driven API.

Author: B.G (02/2026)
"""

import numpy as np
import taichi as ti

from pyfastflow.rastermanip import RasManContext


ti.init(arch=ti.cpu)
rasmanctx = RasManContext()


def test_upscale_methods():
    grid = np.array([[1, 2], [3, 4]], dtype=np.float32)
    for method in ["nearest", "bilinear", "bicubic", "lanczos"]:
        out = rasmanctx.upscale_grid(
            grid,
            scale_factor=2.0,
            method=method,
            as_numpy=True,
            output_layout="2d",
        )
        assert out.shape == (4, 4)
        assert np.all(np.isfinite(out))


def test_downscale_mean_min_max():
    grid = np.array(
        [[1, 2, 5, 6], [3, 4, 7, 8], [9, 10, 13, 14], [11, 12, 15, 16]],
        dtype=np.float32,
    )
    mean_out = rasmanctx.downscale_grid(
        grid, scale_factor=0.5, method="mean", as_numpy=True, output_layout="2d"
    )
    min_out = rasmanctx.downscale_grid(
        grid, scale_factor=0.5, method="min", as_numpy=True, output_layout="2d"
    )
    max_out = rasmanctx.downscale_grid(
        grid, scale_factor=0.5, method="max", as_numpy=True, output_layout="2d"
    )

    expected_mean = np.array([[2.5, 6.5], [10.5, 14.5]], dtype=np.float32)
    assert np.allclose(mean_out, expected_mean)
    assert np.all(max_out >= mean_out)
    assert np.all(mean_out >= min_out)


def test_downscale_median_and_percentile():
    grid = np.array(
        [[1, 2, 5, 6], [3, 4, 7, 8], [9, 10, 13, 14], [11, 12, 15, 16]],
        dtype=np.float32,
    )
    med = rasmanctx.downscale_grid(
        grid, scale_factor=0.5, method="median", as_numpy=True, output_layout="2d"
    )
    p90 = rasmanctx.downscale_grid(
        grid,
        scale_factor=0.5,
        method="percentile",
        percentile=90.0,
        as_numpy=True,
        output_layout="2d",
    )
    assert med.shape == (2, 2)
    assert p90.shape == (2, 2)
    assert np.all(p90 >= med)


def test_resize_wrappers():
    grid = np.arange(36, dtype=np.float32).reshape(6, 6)

    out1 = rasmanctx.resize_raster(
        grid, scale_factor=1.5, upscale_method="bilinear", as_numpy=True, output_layout="2d"
    )
    assert out1.shape == (9, 9)

    out2 = rasmanctx.resize_to_dims(
        grid, target_nx=4, target_ny=3, downscale_method="mean", as_numpy=True, output_layout="2d"
    )
    assert out2.shape == (3, 4)

    out3 = rasmanctx.resize_to_max_dim(
        grid, max_dim=4, downscale_method="mean", as_numpy=True, output_layout="2d"
    )
    assert max(out3.shape) == 4
