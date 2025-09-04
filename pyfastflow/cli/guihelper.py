"""Reusable GUI helpers for Taichi ggui image-based tools.

This module centralizes common functionality used by CLI visualization tools:
- Terrain colormap + hillshade compositing
- Padded canvas preparation and viewport math
- Simple lasso drawing and application on raster grids

Author: B.G. + refactor
"""

from __future__ import annotations

import numpy as np
import matplotlib.cm as cm
import taichi as ti
from matplotlib.path import Path

# Optional imports from package
try:
    from .. import constants as cte
    from ..visu.hillshading import hillshade_numpy
except Exception:  # pragma: no cover - during minimal loads
    cte = None
    hillshade_numpy = None


def nodata_neighbor_mask(nodata: np.ndarray) -> np.ndarray:
    """Return mask of cells that have at least one NoData neighbor.

    Works on 2D boolean array where True marks NoData cells.
    """
    nb = np.zeros_like(nodata, dtype=bool)
    nb[1:, :] |= nodata[:-1, :]
    nb[:-1, :] |= nodata[1:, :]
    nb[:, 1:] |= nodata[:, :-1]
    nb[:, :-1] |= nodata[:, 1:]
    return nb


def compute_visual_layers(dem: np.ndarray, sea_min: float, sea_max: float) -> tuple[np.ndarray, np.ndarray]:
    """Precompute terrain colormap RGB and hillshade grayscale RGB.

    Returns (terrain_rgb, hill_rgb), both float32 arrays in [0,1] with shape (ny,nx,3).
    Requires Taichi to be initialized before call (for hillshade kernel).
    """
    ny, nx = dem.shape
    valid_mask = ~np.isnan(dem)
    dem_norm = np.zeros_like(dem, dtype=np.float32)
    if np.any(valid_mask):
        dem_norm[valid_mask] = (dem[valid_mask] - sea_min) / (sea_max - sea_min + 1e-6)
    terrain_cmap = cm.get_cmap("terrain")
    terrain_rgb = terrain_cmap(np.clip(dem_norm, 0.0, 1.0))[..., :3].astype(np.float32)

    hs = np.zeros_like(dem, dtype=np.float32)
    if hillshade_numpy is not None:
        if cte is not None and hasattr(cte, "DX"):
            hs = hillshade_numpy(dem.astype(np.float32), dx=cte.DX)
        else:
            hs = hillshade_numpy(dem.astype(np.float32))
    hs = np.nan_to_num(hs, nan=0.0).astype(np.float32)
    hill_rgb = np.stack([hs, hs, hs], axis=-1)
    return terrain_rgb, hill_rgb


def array_to_canvas(rgb_raw: np.ndarray) -> np.ndarray:
    """Convert raster RGB (row-from-top, col-from-left) to Taichi canvas order.

    Transpose to (x, y) and flip Y so top row appears at top on screen.
    Output shape: (nx, ny, 3) float32.
    """
    return np.transpose(rgb_raw, (1, 0, 2))[:, ::-1, :].astype(np.float32)


def place_on_padded(rgb_disp: np.ndarray, nx: int, ny: int, pad_left: int, pad_right: int, pad_top: int, pad_bottom: int) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Embed (nx,ny,3) display RGB into a padded buffer and return viewport bounds.

    Returns (padded, (vx0,vx1,vy0,vy1)) where v* are normalized in [0,1] wrt padded size.
    """
    disp_w = nx + pad_left + pad_right
    disp_h = ny + pad_top + pad_bottom
    padded = np.zeros((disp_w, disp_h, 3), dtype=np.float32)
    padded[pad_left:pad_left + nx, pad_bottom:pad_bottom + ny, :] = rgb_disp
    vx0 = pad_left / disp_w
    vx1 = (pad_left + nx) / disp_w
    vy0 = pad_bottom / disp_h
    vy1 = (pad_bottom + ny) / disp_h
    return padded, (vx0, vx1, vy0, vy1)


def draw_lasso_points(rgb: np.ndarray, lasso_path: list[tuple[float, float]], nx: int, ny: int, color=(1.0, 0.0, 1.0)) -> None:
    for p in lasso_path:
        x = int(p[0] * nx)
        y = int(p[1] * ny)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                px, py = x + dx, y + dy
                if 0 <= px < nx and 0 <= py < ny:
                    rgb[py, px] = color


def _draw_line(rgb: np.ndarray, p1: tuple[int, int], p2: tuple[int, int], color=(0.0, 1.0, 0.0), nx: int | None = None, ny: int | None = None) -> None:
    if nx is None or ny is None:
        ny, nx, _ = rgb.shape
    x1, y1 = p1
    x2, y2 = p2
    x1 = max(0, min(nx - 1, x1))
    y1 = max(0, min(ny - 1, y1))
    x2 = max(0, min(nx - 1, x2))
    y2 = max(0, min(ny - 1, y2))
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    x, y = x1, y1
    while True:
        if 0 <= x < nx and 0 <= y < ny:
            rgb[y, x] = color
        if x == x2 and y == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def draw_lasso_polygon(rgb: np.ndarray, lasso_path: list[tuple[float, float]], nx: int, ny: int, color=(0.0, 1.0, 0.0)) -> None:
    if len(lasso_path) < 2:
        return
    pts: list[tuple[int, int]] = []
    for u, v in lasso_path:
        pts.append((int(u * nx), int(v * ny)))
    for i in range(len(pts) - 1):
        _draw_line(rgb, pts[i], pts[i + 1], color=color, nx=nx, ny=ny)
    if len(pts) >= 3:
        _draw_line(rgb, pts[-1], pts[0], color=color, nx=nx, ny=ny)


def apply_lasso_to_boundaries(boundaries: np.ndarray, nodata: np.ndarray, lasso_path: list[tuple[float, float]]) -> None:
    """Apply lasso selection: mark outlets (3) for selected edge/NoData-neighboring cells.

    Modifies boundaries in place. Expects boundaries shape matches nodata.
    """
    if len(lasso_path) < 3:
        return
    ny, nx = boundaries.shape
    poly = np.array([[u * nx, v * ny] for (u, v) in lasso_path], dtype=np.float32)
    gx, gy = np.meshgrid(np.arange(nx), np.arange(ny))
    pts = np.stack([gx.ravel(), gy.ravel()], axis=-1)
    mask = Path(poly).contains_points(pts).reshape(ny, nx)
    valid = mask & ~nodata
    nb = nodata_neighbor_mask(nodata)
    edges = np.zeros_like(valid, dtype=bool)
    edges[0, :] = True
    edges[-1, :] = True
    edges[:, 0] = True
    edges[:, -1] = True
    boundaries[valid & (edges | nb)] = 3

