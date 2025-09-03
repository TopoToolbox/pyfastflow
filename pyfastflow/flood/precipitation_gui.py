"""Interactive GUI to paint precipitation or discharge values on a DEM.

This module offers a Taichi ``ggui`` based application similar to
:mod:`pyfastflow.grid.boundary_gui`.  It loads a DEM stored as ``.npy`` file
and lets the user interactively paint precipitation rates on top of it.  The
value provided by the user can be interpreted either as precipitation rate
(``m/s``) or discharge (``m^3/s``).  Internally the resulting array always
stores precipitation rates; values entered as discharge are converted using
the cell area (``cte.DX**2``).

Two painting modes are available:

* **Lasso** – draw a polygon selection and fill all cells inside.
* **Gaussian brush** – paint with a circular Gaussian kernel.

For both tools the user can choose between **additive** mode (each affected
cell receives the full value scaled by the kernel) or **distributed** mode
(the total added over the painted area equals the provided value).

The GUI displays a hillshade background derived from the DEM.  If a boundary
array is supplied, ``NoData`` (code ``0``) cells are shown in black and outlet
nodes (code ``3``) in red to ease navigation.

Example
-------

.. code-block:: bash

    python -m pyfastflow.flood.precipitation_gui dem.npy precip.npy \
        --boundary boundaries.npy

Author
------
OpenAI's ChatGPT.
"""

from __future__ import annotations

import numpy as np
import click
import taichi as ti
from matplotlib.path import Path

from ..visu.hillshading import hillshade_numpy
from .. import constants as cte


def _gaussian_kernel(radius: int) -> np.ndarray:
    """Return a 2D Gaussian kernel with the given ``radius``."""

    if radius <= 0:
        return np.ones((1, 1), dtype=np.float32)
    x = np.arange(-radius, radius + 1)
    y = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(x, y)
    sigma = radius / 2.0 if radius > 0 else 1.0
    k = np.exp(-((xx**2 + yy**2) / (2.0 * sigma**2)))
    return k.astype(np.float32)


@click.command()
@click.argument("dem_npy", type=click.Path(exists=True))
@click.argument("output_npy", type=click.Path())
@click.option(
    "--boundary",
    "boundary_npy",
    type=click.Path(exists=True),
    default=None,
    help="Optional boundary array (.npy) for overlay.",
)
def precipitation_gui(
    dem_npy: str, output_npy: str, boundary_npy: str | None
) -> None:
    """Launch the precipitation/discharge editor."""

    dem = np.load(dem_npy).astype(np.float32)
    ny, nx = dem.shape
    precip = np.zeros((ny, nx), dtype=np.float32)

    nodata = np.zeros_like(dem, dtype=bool)
    outlets = np.zeros_like(dem, dtype=bool)
    if boundary_npy is not None:
        boundaries = np.load(boundary_npy).astype(np.uint8)
        if boundaries.shape != dem.shape:
            raise ValueError("Boundary array must match DEM dimensions")
        nodata = boundaries == 0
        outlets = boundaries == 3

    hill = hillshade_numpy(dem, dx=cte.DX)
    base_rgb = np.stack([hill, hill, hill], axis=-1)

    ti.init(arch=ti.gpu if ti.core.is_arch_supported(ti.gpu) else ti.cpu)
    window = ti.ui.Window("Precipitation Editor", (nx, ny))
    canvas = window.get_canvas()
    gui = window.get_gui()

    value_slider = gui.slider_float("Value", 0.0, 1.0)
    value_slider.value = 1.0
    radius_slider = gui.slider_float("Radius", 1.0, 50.0)
    radius_slider.value = 5.0

    additive = False
    discharge_mode = False
    lasso_mode = False
    lasso_path: list[tuple[float, float]] = []

    img_field = ti.Vector.field(3, dtype=ti.f32, shape=(ny, nx))

    def update_display() -> None:
        rgb = base_rgb.copy()
        max_p = float(precip.max())
        if max_p > 0.0:
            rgb[:, :, 2] = np.clip(rgb[:, :, 2] + precip / max_p, 0.0, 1.0)
        rgb[nodata] = np.array([0.0, 0.0, 0.0])
        rgb[outlets] = np.array([1.0, 0.0, 0.0])
        img_field.from_numpy(rgb.astype(np.float32))

    def apply_lasso(path: list[tuple[float, float]]) -> None:
        if len(path) < 3:
            return
        poly = np.array([[p[0] * nx, (1.0 - p[1]) * ny] for p in path])
        gx, gy = np.meshgrid(np.arange(nx), np.arange(ny))
        pts = np.stack([gx.ravel(), gy.ravel()], axis=-1)
        mask = Path(poly).contains_points(pts).reshape(ny, nx)
        if not mask.any():
            return
        n = mask.sum()
        val = value_slider.value
        if discharge_mode:
            val /= cte.DX * cte.DX
        if additive:
            precip[mask] += val
        else:
            precip[mask] += val / n

    def apply_brush(pos: tuple[float, float]) -> None:
        cx = int(pos[0] * nx)
        cy = int((1.0 - pos[1]) * ny)
        radius = int(radius_slider.value)
        k = _gaussian_kernel(radius)
        h, w = k.shape
        x0 = max(cx - radius, 0)
        y0 = max(cy - radius, 0)
        x1 = min(cx + radius + 1, nx)
        y1 = min(cy + radius + 1, ny)
        sub_k = k[radius - (cy - y0) : radius + (y1 - cy),
                  radius - (cx - x0) : radius + (x1 - cx)]
        if additive:
            weights = sub_k
        else:
            weights = sub_k / sub_k.sum()
        val = value_slider.value
        if discharge_mode:
            val /= cte.DX * cte.DX
        precip[y0:y1, x0:x1] += val * weights

    update_display()
    while window.running:
        _ = value_slider.value  # keep widgets responsive
        _ = radius_slider.value
        additive = gui.checkbox("Additive", additive)
        discharge_mode = gui.checkbox("Discharge input", discharge_mode)

        if gui.button("Lasso"):
            lasso_mode = True
            lasso_path.clear()

        if gui.button("Save and Quit"):
            np.save(output_npy, precip)
            break

        if lasso_mode:
            if window.is_pressed(ti.ui.LMB):
                lasso_path.append(window.get_cursor_pos())
            elif lasso_path:
                apply_lasso(lasso_path)
                lasso_path.clear()
                lasso_mode = False
        else:
            if window.is_pressed(ti.ui.LMB):
                apply_brush(window.get_cursor_pos())

        update_display()
        canvas.set_image(img_field)
        window.show()

    window.destroy()


if __name__ == "__main__":
    precipitation_gui()
