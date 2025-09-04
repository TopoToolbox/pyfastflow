"""Interactive GUI to paint precipitation or discharge values on a DEM.

This module offers a Taichi ``ggui`` based application similar to
:mod:`pyfastflow.cli.boundary_gui`.  It loads a DEM stored as ``.npy`` file
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

from .. import constants as cte
from ..cli.guihelper import (
    prepare_display_textures,
    compute_viewport,
)


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
@click.argument("dem_file", type=click.Path(exists=True))
@click.argument("output_npy", type=click.Path(), required=False)
@click.option("--boundary", "boundary_npy", type=click.Path(exists=True), default=None, help="Optional BC array (.npy) for overlay/masking.")
@click.option("--disp-max", type=int, default=2048, show_default=True, help="Max display texture dimension (visuals only).")
def precipitation_gui(dem_file: str, output_npy: str | None, boundary_npy: str | None, disp_max: int) -> None:
    """Launch the precipitation/discharge editor."""

    # Load DEM: .npy direct, other rasters via TopoToolbox wrapper
    try:
        from ..io import raster_to_numpy, TOPOTOOLBOX_AVAILABLE
    except Exception:
        TOPOTOOLBOX_AVAILABLE = False
        raster_to_numpy = None
    if dem_file.lower().endswith(".npy"):
        dem = np.load(dem_file).astype(np.float32)
    else:
        if not TOPOTOOLBOX_AVAILABLE:
            raise ImportError("TopoToolbox is required to read non-.npy raster files. Install with: pip install topotoolbox")
        dem = raster_to_numpy(dem_file).astype(np.float32)
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

    # Init Taichi before any resizing kernels (used by helpers)
    try:
        ti.init(arch=ti.gpu)
    except Exception:
        ti.init(arch=ti.cpu)

    # Prepare display textures
    sea_min = float(np.nanmin(dem))
    sea_max = float(np.nanmax(dem))
    dem_disp, terrain_u8, hill_u8, nx_disp, ny_disp, sx, sy = prepare_display_textures(dem, sea_min, sea_max, max(256, int(disp_max)))

    # Window
    WIN_W, WIN_H = 1280, 840
    PANEL_W, MARGIN = 280, 16
    window = ti.ui.Window("Precipitation Editor", (WIN_W, WIN_H))
    canvas = window.get_canvas()
    gui = window.get_gui()

    # Viewport
    vp_x0, vp_y0, vp_w, vp_h, vx0, vx1, vy0, vy1 = compute_viewport(WIN_W, WIN_H, PANEL_W, MARGIN, nx, ny)

    # GPU fields
    frame = ti.Vector.field(3, dtype=ti.f32, shape=(WIN_W, WIN_H))
    terrain_tex = ti.Vector.field(3, dtype=ti.u8, shape=(ny_disp, nx_disp))
    hill_tex = ti.field(dtype=ti.u8, shape=(ny_disp, nx_disp))
    nodata_tex = ti.field(dtype=ti.u8, shape=(ny_disp, nx_disp))
    outlets_tex = ti.field(dtype=ti.u8, shape=(ny_disp, nx_disp))
    precip_disp_tex = ti.field(dtype=ti.f32, shape=(ny_disp, nx_disp))
    dem_disp_tex = ti.field(dtype=ti.f32, shape=(ny_disp, nx_disp))

    # Upload textures
    terrain_tex.from_numpy(terrain_u8)
    hill_tex.from_numpy(hill_u8)
    dem_disp_tex.from_numpy(dem_disp.astype(np.float32))
    if boundary_npy is not None:
        from ..rastermanip import resize_to_dims
        nod_ds = (resize_to_dims(nodata.astype(np.float32), nx_disp, ny_disp) >= 0.5).astype(np.uint8)
        out_ds = (resize_to_dims(outlets.astype(np.float32), nx_disp, ny_disp) >= 0.5).astype(np.uint8)
        nodata_tex.from_numpy(nod_ds)
        outlets_tex.from_numpy(out_ds)
    else:
        nodata_tex.from_numpy(np.zeros((ny_disp, nx_disp), dtype=np.uint8))
        outlets_tex.from_numpy(np.zeros((ny_disp, nx_disp), dtype=np.uint8))

    # Painting parameters
    value = 1.0
    radius = 5.0

    additive = False
    discharge_mode = False
    lasso_mode = False
    lasso_path: list[tuple[float, float]] = []

    # Uniforms
    vp_x0_k = ti.field(dtype=ti.i32, shape=())
    vp_y0_k = ti.field(dtype=ti.i32, shape=())
    vp_w_k = ti.field(dtype=ti.i32, shape=())
    vp_h_k = ti.field(dtype=ti.i32, shape=())
    win_w_k = ti.field(dtype=ti.i32, shape=())
    win_h_k = ti.field(dtype=ti.i32, shape=())
    # Zoom/pan uniforms for display-texture sampling window and inner rect
    view_x_min_ds_k = ti.field(dtype=ti.f32, shape=())
    view_y_min_ds_k = ti.field(dtype=ti.f32, shape=())
    view_w_ds_k = ti.field(dtype=ti.f32, shape=())
    view_h_ds_k = ti.field(dtype=ti.f32, shape=())
    inner_x0_k = ti.field(dtype=ti.i32, shape=())
    inner_y0_k = ti.field(dtype=ti.i32, shape=())
    inner_w_k = ti.field(dtype=ti.i32, shape=())
    inner_h_k = ti.field(dtype=ti.i32, shape=())

    @ti.kernel
    def render():
        for x, y in ti.ndrange(win_w_k[None], win_h_k[None]):
            c = ti.Vector([0.0, 0.0, 0.0])
            # Draw only within inner rect (supports zoom<1 letterboxing)
            if x >= inner_x0_k[None] and x < inner_x0_k[None] + inner_w_k[None] and y >= inner_y0_k[None] and y < inner_y0_k[None] + inner_h_k[None]:
                u = (ti.cast(x - inner_x0_k[None], ti.f32) + 0.5) / ti.max(1, inner_w_k[None])
                v_b = (ti.cast(y - inner_y0_k[None], ti.f32) + 0.5) / ti.max(1, inner_h_k[None])
                # Map to display texture window (pan/zoom)
                ax = view_x_min_ds_k[None] + u * view_w_ds_k[None]
                ay = view_y_min_ds_k[None] + (1.0 - v_b) * view_h_ds_k[None]
                ix_ds = ti.max(0, ti.min(nx_disp - 1, ti.i32(ax)))
                iy_ds = ti.max(0, ti.min(ny_disp - 1, ti.i32(ay)))
                t_u8 = terrain_tex[iy_ds, ix_ds]
                tcol = ti.Vector([ti.cast(t_u8[0], ti.f32), ti.cast(t_u8[1], ti.f32), ti.cast(t_u8[2], ti.f32)]) / 255.0
                h_u8 = hill_tex[iy_ds, ix_ds]
                hval = ti.cast(h_u8, ti.f32) / 255.0
                hcol = ti.Vector([hval, hval, hval])
                base = (tcol + hcol) * 0.5
                # Precip overlay in blue
                pval = precip_disp_tex[iy_ds, ix_ds]
                cblue = ti.min(1.0, base[2] + pval)
                col = ti.Vector([base[0], base[1], cblue])
                # BC overlays from display masks
                if outlets_tex[iy_ds, ix_ds] != ti.u8(0):
                    col = ti.Vector([1.0, 0.0, 0.0])
                elif nodata_tex[iy_ds, ix_ds] != ti.u8(0):
                    col = ti.Vector([0.0, 0.0, 0.0])
                c = col
            frame[x, y] = c

    # Zoom/pan state in display texture space
    zoom = 1.0
    zoom_min, zoom_max = 0.5, 8.0
    view_x_min_ds = 0.0
    view_y_min_ds = 0.0
    disp_off_x_px = 0.0
    disp_off_y_px = 0.0

    def update_display() -> None:
        # Rebuild precip display texture from precip (full-res) when changed
        p = precip.copy()
        mp = float(p.max())
        if mp > 0.0:
            p = (p / mp).astype(np.float32)
        else:
            p = np.zeros_like(p, dtype=np.float32)
        from ..rastermanip import resize_to_dims
        p_ds = resize_to_dims(p, nx_disp, ny_disp)
        precip_disp_tex.from_numpy(p_ds.astype(np.float32))
        # Draw
        # Compute inner rect + view window per zoom mode
        if zoom >= 1.0:
            inner_x0, inner_y0, inner_w, inner_h = vp_x0, vp_y0, vp_w, vp_h
            view_w_ds = nx_disp / max(zoom, 1e-6)
            view_h_ds = ny_disp / max(zoom, 1e-6)
            vx_ds = view_x_min_ds
            vy_ds = view_y_min_ds
        else:
            out_w = max(1, int(vp_w * zoom))
            out_h = max(1, int(vp_h * zoom))
            base_x = vp_x0 + (vp_w - out_w) // 2
            base_y = vp_y0 + (vp_h - out_h) // 2
            max_off_x = (vp_w - out_w) // 2
            max_off_y = (vp_h - out_h) // 2
            ox = int(np.clip(disp_off_x_px, -max_off_x, max_off_x))
            oy = int(np.clip(disp_off_y_px, -max_off_y, max_off_y))
            inner_x0, inner_y0, inner_w, inner_h = base_x + ox, base_y + oy, out_w, out_h
            view_w_ds, view_h_ds = float(nx_disp), float(ny_disp)
            vx_ds, vy_ds = 0.0, 0.0

        vp_x0_k[None], vp_y0_k[None], vp_w_k[None], vp_h_k[None] = vp_x0, vp_y0, vp_w, vp_h
        win_w_k[None], win_h_k[None] = WIN_W, WIN_H
        inner_x0_k[None], inner_y0_k[None], inner_w_k[None], inner_h_k[None] = inner_x0, inner_y0, inner_w, inner_h
        view_x_min_ds_k[None], view_y_min_ds_k[None] = float(vx_ds), float(vy_ds)
        view_w_ds_k[None], view_h_ds_k[None] = float(view_w_ds), float(view_h_ds)
        render()
        canvas.set_image(frame)

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
        val = value
        if discharge_mode:
            val /= cte.DX * cte.DX
        if additive:
            precip[mask] += val
        else:
            precip[mask] += val / n

    update_display()
    while window.running:
        # (sliders and toggles handled later in the loop)

        if gui.button("Lasso"):
            lasso_mode = True
            lasso_path.clear()

        if gui.button("Save and Quit"):
            # Default output name if not provided: <input>_P.npy
            if output_npy is None:
                import os
                base, _ = os.path.splitext(dem_file)
                output_npy = f"{base}_P.npy"
            np.save(output_npy, precip.astype(np.float32))
            break

        if lasso_mode:
            if window.get_event(ti.ui.PRESS):
                ev = window.event
                if ev.key == ti.ui.LMB:
                    pos = window.get_cursor_pos()
                    # Only accept clicks inside image viewport
                    if pos[0] >= vx0:
                        px = pos[0] * WIN_W
                        py_top = (1.0 - pos[1]) * WIN_H
                        u = (px - vp_x0) / max(vp_w, 1)
                        v_top = (py_top - vp_y0) / max(vp_h, 1)
                        u = float(np.clip(u, 0.0, 1.0))
                        v_top = float(np.clip(v_top, 0.0, 1.0))
                        # Store normalized DEM coords (top-left origin)
                        lasso_path.append((u, v_top))
                elif ev.key == ti.ui.RMB and lasso_path:
                    apply_lasso(lasso_path)
                    lasso_path.clear()
                    lasso_mode = False
        else:
            if window.is_pressed(ti.ui.LMB):
                pos = window.get_cursor_pos()
                px = pos[0] * WIN_W
                py_top = (1.0 - pos[1]) * WIN_H
                if pos[0] >= vx0:
                    u = (px - vp_x0) / max(vp_w, 1)
                    v_top = (py_top - vp_y0) / max(vp_h, 1)
                    cx = int(np.clip(u * nx, 0, nx - 1))
                    cy = int(np.clip(v_top * ny, 0, ny - 1))
                    # Brush at (cx, cy)
                    r_int = int(radius)
                    k = _gaussian_kernel(r_int)
                    h, w = k.shape
                    x0 = max(cx - r_int, 0)
                    y0 = max(cy - r_int, 0)
                    x1 = min(cx + r_int + 1, nx)
                    y1 = min(cy + r_int + 1, ny)
                    sub_k = k[r_int - (cy - y0) : r_int + (y1 - cy), r_int - (cx - x0) : r_int + (x1 - cx)]
                    weights = sub_k if additive else (sub_k / sub_k.sum())
                    val = value
                    if discharge_mode:
                        val /= cte.DX * cte.DX
                    precip[y0:y1, x0:x1] += val * weights

        # Sliders and toggles
        value = gui.slider_float("Value", value, 0.0, 1.0)
        radius = gui.slider_float("Radius", radius, 1.0, 50.0)
        additive = gui.checkbox("Additive", additive)
        discharge_mode = gui.checkbox("Discharge input", discharge_mode)

        # Keyboard pan
        pan_frac = 0.05
        if zoom >= 1.0:
            vw_ds = nx_disp / max(zoom, 1e-6)
            vh_ds = ny_disp / max(zoom, 1e-6)
            left_key = getattr(ti.ui, 'LEFT', None)
            right_key = getattr(ti.ui, 'RIGHT', None)
            up_key = getattr(ti.ui, 'UP', None)
            down_key = getattr(ti.ui, 'DOWN', None)
            if left_key and window.is_pressed(left_key):
                view_x_min_ds = float(np.clip(view_x_min_ds - pan_frac * vw_ds, 0.0, max(0.0, nx_disp - vw_ds)))
            if right_key and window.is_pressed(right_key):
                view_x_min_ds = float(np.clip(view_x_min_ds + pan_frac * vw_ds, 0.0, max(0.0, nx_disp - vw_ds)))
            if up_key and window.is_pressed(up_key):
                view_y_min_ds = float(np.clip(view_y_min_ds - pan_frac * vh_ds, 0.0, max(0.0, ny_disp - vh_ds)))
            if down_key and window.is_pressed(down_key):
                view_y_min_ds = float(np.clip(view_y_min_ds + pan_frac * vh_ds, 0.0, max(0.0, ny_disp - vh_ds)))
        else:
            step_x = max(1, int(pan_frac * vp_w))
            step_y = max(1, int(pan_frac * vp_h))
            out_w = max(1, int(vp_w * zoom))
            out_h = max(1, int(vp_h * zoom))
            max_off_x = (vp_w - out_w) // 2
            max_off_y = (vp_h - out_h) // 2
            left_key = getattr(ti.ui, 'LEFT', None)
            right_key = getattr(ti.ui, 'RIGHT', None)
            up_key = getattr(ti.ui, 'UP', None)
            down_key = getattr(ti.ui, 'DOWN', None)
            if left_key and window.is_pressed(left_key):
                disp_off_x_px = float(np.clip(disp_off_x_px - step_x, -max_off_x, max_off_x))
            if right_key and window.is_pressed(right_key):
                disp_off_x_px = float(np.clip(disp_off_x_px + step_x, -max_off_x, max_off_x))
            if up_key and window.is_pressed(up_key):
                disp_off_y_px = float(np.clip(disp_off_y_px - step_y, -max_off_y, max_off_y))
            if down_key and window.is_pressed(down_key):
                disp_off_y_px = float(np.clip(disp_off_y_px + step_y, -max_off_y, max_off_y))

        # Zoom controls (buttons)
        if gui.button("Zoom +"):
            if zoom >= 1.0:
                cx = view_x_min_ds + (nx_disp / max(zoom, 1e-6)) * 0.5
                cy = view_y_min_ds + (ny_disp / max(zoom, 1e-6)) * 0.5
                zoom = min(zoom_max, zoom * 1.25)
                vw = nx_disp / zoom
                vh = ny_disp / zoom
                view_x_min_ds = float(np.clip(cx - vw * 0.5, 0.0, max(0.0, nx_disp - vw)))
                view_y_min_ds = float(np.clip(cy - vh * 0.5, 0.0, max(0.0, ny_disp - vh)))
            else:
                zoom = min(zoom_max, zoom * 1.25)
        if gui.button("Zoom -"):
            if zoom >= 1.0:
                cx = view_x_min_ds + (nx_disp / max(zoom, 1e-6)) * 0.5
                cy = view_y_min_ds + (ny_disp / max(zoom, 1e-6)) * 0.5
                zoom = max(zoom_min, zoom / 1.25)
                if zoom >= 1.0:
                    vw = nx_disp / zoom
                    vh = ny_disp / zoom
                    view_x_min_ds = float(np.clip(cx - vw * 0.5, 0.0, max(0.0, nx_disp - vw)))
                    view_y_min_ds = float(np.clip(cy - vh * 0.5, 0.0, max(0.0, ny_disp - vh)))
                else:
                    disp_off_x_px = 0.0
                    disp_off_y_px = 0.0
            else:
                zoom = max(zoom_min, zoom / 1.25)

        update_display()
        window.show()

    window.destroy()


if __name__ == "__main__":
    precipitation_gui()
