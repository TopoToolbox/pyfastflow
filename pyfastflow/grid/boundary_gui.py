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

    python -m pyfastflow.grid.boundary_gui dem.npy boundaries.npy

Author
------
B.G., extended by OpenAI's ChatGPT.
"""

from __future__ import annotations

import numpy as np
import click
import taichi as ti
from matplotlib.path import Path


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
@click.argument("dem_npy", type=click.Path(exists=True))
@click.argument("output_npy", type=click.Path())
def boundary_gui(dem_npy: str, output_npy: str) -> None:
    """Launch the boundary condition editor.

    Parameters
    ----------
    dem_npy : str
        Path to a ``.npy`` DEM file (2-D array of float32/float64).
    output_npy : str
        Destination where the boundary array will be stored.
    """

    dem = np.load(dem_npy).astype(np.float32)
    ny, nx = dem.shape

    sea_min = float(dem.min())
    sea_max = float(dem.max())
    sea_level = sea_min

    boundaries = np.ones((ny, nx), dtype=np.uint8)
    nodata = dem < sea_level
    boundaries[nodata] = 0

    # Initialise Taichi (prefer GPU but fall back to CPU).
    ti.init(arch=ti.gpu if ti.core.is_arch_supported(ti.gpu) else ti.cpu)

    window = ti.ui.Window("Boundary Editor", (nx, ny))
    canvas = window.get_canvas()
    gui = window.get_gui()

    slider = gui.slider_float("Sea level", sea_min, sea_max)
    slider.value = sea_level

    img_field = ti.Vector.field(3, dtype=ti.f32, shape=(ny, nx))

    lasso_mode = False
    lasso_path: list[tuple[float, float]] = []

    def update_display() -> None:
        """Update image field used for visualization."""

        norm = (dem - sea_min) / (sea_max - sea_min + 1e-6)
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

    while window.running:
        sea_level = slider.value
        nodata[:, :] = dem < sea_level
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
