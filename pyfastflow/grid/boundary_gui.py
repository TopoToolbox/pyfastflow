"""Interactive GUI to generate boundary arrays from a DEM.

This module provides a small Taichi ``ggui`` based application that helps
building custom boundary condition arrays compatible with the :mod:`grid`
submodule.  The tool loads a DEM stored as ``.npy`` file and lets the user
interactively decide which cells are considered "NoData" through a sea level
slider.  Two editing actions are available:

* **Auto boundary** – mark all valid cells that touch ``NoData`` cells or lie
  on the outer grid edge as code ``3`` (``can out``).
* **Lasso** – draw a polygon selection.  Only the selected cells that are on the
  grid edge or adjacent to ``NoData`` are turned into code ``3``.

Once satisfied, the user can save the resulting boundary array to another
``.npy`` file that can be loaded in a :class:`pyfastflow.grid.Grid` instance
using ``boundary_mode="custom"``.

Example
-------

.. code-block:: bash

    # Using .npy file with explicit output
    pff-boundary-gui dem.npy boundaries.npy
    
    # Using any raster format with auto-generated output
    pff-boundary-gui elevation.tif
    # This will create elevation_bc.npy in the same directory

Author
------
B.G., extended by OpenAI's ChatGPT.
"""

from __future__ import annotations

import numpy as np
import click
import taichi as ti
from matplotlib.path import Path
import os
from ..io import raster_to_numpy, TOPOTOOLBOX_AVAILABLE


def _nodata_neighbor_mask(nodata: np.ndarray) -> np.ndarray:
    """Return mask of cells that have a ``NoData`` neighbour.

    Parameters
    ----------
    nodata : np.ndarray
        Boolean mask where ``True`` marks NoData cells.
    """

    nb = np.zeros_like(nodata, dtype=bool)
    nb[1:, :] |= nodata[:-1, :]
    nb[:-1, :] |= nodata[1:, :]
    nb[:, 1:] |= nodata[:, :-1]
    nb[:, :-1] |= nodata[:, 1:]
    return nb


@click.command()
@click.argument("dem_file", type=click.Path(exists=True))
@click.argument("output_npy", type=click.Path(), required=False)
def boundary_gui(dem_file: str, output_npy: str = None) -> None:
    """Launch the boundary condition editor.

    Parameters
    ----------
    dem_file : str
        Path to a raster DEM file (supports .npy, .tif, .asc, and other formats via TopoToolbox).
    output_npy : str, optional
        Destination where the boundary array will be stored. If not provided, 
        saves to same directory as input with '_bc.npy' suffix.
    """

    # Determine output filename if not provided
    if output_npy is None:
        base_path, ext = os.path.splitext(dem_file)
        output_npy = f"{base_path}_bc.npy"

    # Load DEM data - try .npy first, then use TopoToolbox
    if dem_file.lower().endswith('.npy'):
        dem = np.load(dem_file).astype(np.float32)
    else:
        if not TOPOTOOLBOX_AVAILABLE:
            raise ImportError(
                "TopoToolbox is required to read non-.npy raster files. "
                "Install with: pip install topotoolbox"
            )
        dem = raster_to_numpy(dem_file).astype(np.float32)
    ny, nx = dem.shape

    # Handle NaN values in DEM
    valid_mask = ~np.isnan(dem)
    if not np.any(valid_mask):
        raise ValueError("DEM contains only NaN values")
    
    valid_dem = dem[valid_mask]
    sea_min = float(valid_dem.min())
    sea_max = float(valid_dem.max())
    sea_level = sea_min

    boundaries = np.ones((ny, nx), dtype=np.uint8)
    nodata = np.isnan(dem) | (dem < sea_level)
    boundaries[nodata] = 0

    # Initialise Taichi (prefer GPU but fall back to CPU).
    try:
        ti.init(arch=ti.gpu)
    except Exception:
        ti.init(arch=ti.cpu)

    # Check if we're in a headless environment
    headless = not bool(os.environ.get('DISPLAY', ''))
    
    window = ti.ui.Window("Boundary Editor", (nx, ny), show_window=not headless)
    canvas = window.get_canvas()
    gui = window.get_gui()

    if not headless:
        slider = gui.slider_float("Sea level", sea_level, sea_min, sea_max)

    img_field = ti.Vector.field(3, dtype=ti.f32, shape=(ny, nx))

    lasso_mode = False
    lasso_path: list[tuple[float, float]] = []

    def update_display() -> None:
        """Update image field used for visualization."""

        # Only normalize valid (non-NaN) values
        valid_mask = ~np.isnan(dem)
        norm = np.zeros_like(dem)
        if np.any(valid_mask):
            norm[valid_mask] = (dem[valid_mask] - sea_min) / (sea_max - sea_min + 1e-6)
        
        rgb = np.stack([norm, norm, norm], axis=-1)
        rgb[nodata] = 0.0  # mask NoData as black
        rgb[boundaries == 3] = np.array([1.0, 0.0, 0.0])  # outlets in red
        img_field.from_numpy(rgb.astype(np.float32))

    def apply_auto_boundary() -> None:
        """Mark all interfaces between valid cells and NoData as outlets."""

        boundaries[:, :] = np.where(nodata, 0, 1)
        valid = ~nodata

        edges = np.zeros_like(valid, dtype=bool)
        edges[0, :] = valid[0, :]
        edges[-1, :] = valid[-1, :]
        edges[:, 0] = valid[:, 0]
        edges[:, -1] = valid[:, -1]

        nb = _nodata_neighbor_mask(nodata)
        boundaries[valid & (edges | nb)] = 3

    def apply_lasso(path: list[tuple[float, float]]) -> None:
        """Convert a polygon path to boundary outlets."""

        if len(path) < 3:
            return

        poly = np.array([[p[0] * nx, (1.0 - p[1]) * ny] for p in path])
        gx, gy = np.meshgrid(np.arange(nx), np.arange(ny))
        pts = np.stack([gx.ravel(), gy.ravel()], axis=-1)
        mask = Path(poly).contains_points(pts).reshape(ny, nx)

        valid = mask & ~nodata
        nb = _nodata_neighbor_mask(nodata)

        edges = np.zeros_like(valid, dtype=bool)
        edges[0, :] = True
        edges[-1, :] = True
        edges[:, 0] = True
        edges[:, -1] = True

        boundaries[valid & (edges | nb)] = 3

    update_display()

    if headless:
        # Headless mode: just create auto boundary and save
        apply_auto_boundary()
        np.save(output_npy, boundaries)
        print(f"Boundary conditions saved to {output_npy}")
    else:
        # GUI mode: interactive editing
        while window.running:
            sea_level = gui.slider_float("Sea level", sea_level, sea_min, sea_max)
            nodata[:, :] = np.isnan(dem) | (dem < sea_level)
            boundaries[nodata] = 0
            boundaries[~nodata & (boundaries == 0)] = 1

            if gui.button("Auto boundary"):
                apply_auto_boundary()

            if gui.button("Lasso"):
                lasso_mode = True
                lasso_path.clear()

            if gui.button("Save and Quit"):
                np.save(output_npy, boundaries)
                break

            if lasso_mode:
                if window.is_pressed(ti.ui.LMB):
                    lasso_path.append(window.get_cursor_pos())
                elif lasso_path:
                    apply_lasso(lasso_path)
                    lasso_path.clear()
                    lasso_mode = False

            update_display()
            canvas.set_image(img_field)
            window.show()

    window.destroy()


if __name__ == "__main__":
    boundary_gui()
