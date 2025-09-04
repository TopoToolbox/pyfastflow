"""Boundary GUI CLI (refactored under cli/ using guihelper).

Reuses generic helpers for:
- Terrain colormap + hillshade compositing
- Correct raster-to-canvas transform
- Lasso draw + apply

Usage (entry point): pff-boundary-gui dem.npy [output.npy]
"""

from __future__ import annotations

import os
import numpy as np
import taichi as ti
import click

from .guihelper import (
    compute_visual_layers,
    array_to_canvas,
    place_on_padded,
    draw_lasso_points,
    draw_lasso_polygon,
    apply_lasso_to_boundaries,
)
from ..io import raster_to_numpy, TOPOTOOLBOX_AVAILABLE


@click.command()
@click.argument("dem_file", type=click.Path(exists=True))
@click.argument("output_npy", type=click.Path(), required=False)
def boundary_gui(dem_file: str, output_npy: str | None = None) -> None:
    if output_npy is None:
        base, _ = os.path.splitext(dem_file)
        output_npy = f"{base}_bc.npy"

    # Load DEM
    if dem_file.lower().endswith(".npy"):
        dem = np.load(dem_file).astype(np.float32)
    else:
        if not TOPOTOOLBOX_AVAILABLE:
            raise ImportError("TopoToolbox required for non-.npy rasters (pip install topotoolbox)")
        dem = raster_to_numpy(dem_file).astype(np.float32)
    ny, nx = dem.shape

    # Sea level range from valid pixels
    valid_mask = ~np.isnan(dem)
    if not np.any(valid_mask):
        raise ValueError("DEM contains only NaN values")
    sea_min = float(dem[valid_mask].min())
    sea_max = float(dem[valid_mask].max())
    sea_level = sea_min

    # Boundary base
    boundaries = np.ones((ny, nx), dtype=np.uint8)
    nodata = np.isnan(dem) | (dem < sea_level)
    boundaries[nodata] = 0

    # Init Taichi BEFORE hillshade
    try:
        ti.init(arch=ti.gpu)
    except Exception:
        ti.init(arch=ti.cpu)

    # Precompute visuals
    terrain_rgb, hill_rgb = compute_visual_layers(dem, sea_min, sea_max)
    hs_alpha = 0.5

    # Window and GUI
    headless = not bool(os.environ.get("DISPLAY", ""))
    base_pad = max(20, int(0.06 * min(nx, ny)))
    panel_px = max(base_pad, int(0.22 * nx))
    pad_left, pad_right, pad_top, pad_bottom = panel_px, base_pad, base_pad, base_pad
    disp_w = nx + pad_left + pad_right
    disp_h = ny + pad_top + pad_bottom

    window = ti.ui.Window("Boundary Editor", (disp_w, disp_h), show_window=not headless)
    canvas = window.get_canvas()
    gui = window.get_gui()
    img_field = ti.Vector.field(3, dtype=ti.f32, shape=(disp_w, disp_h))

    # Viewport bounds in normalized window coords
    vx0 = pad_left / disp_w
    vx1 = (pad_left + nx) / disp_w
    vy0 = pad_bottom / disp_h
    vy1 = (pad_bottom + ny) / disp_h

    lasso_mode = False
    lasso_wait_release = False
    lasso_path: list[tuple[float, float]] = []

    def update_display() -> None:
        # Compose terrain + hillshade once per frame
        rgb_raw = (1.0 - hs_alpha) * terrain_rgb + hs_alpha * hill_rgb
        rgb_raw[np.isnan(dem) | (dem < sea_level)] = 0.0
        rgb_raw[boundaries == 3] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if lasso_mode and lasso_path:
            draw_lasso_points(rgb_raw, lasso_path, nx, ny)
            if len(lasso_path) >= 2:
                draw_lasso_polygon(rgb_raw, lasso_path, nx, ny)
        # To canvas and padded
        rgb_disp = array_to_canvas(rgb_raw)
        padded, _ = place_on_padded(rgb_disp, nx, ny, pad_left, pad_right, pad_top, pad_bottom)
        img_field.from_numpy(padded)

    update_display()

    if headless:
        np.save(output_npy, boundaries)
        print(f"Boundary conditions saved to {output_npy}")
        window.destroy()
        return

    while window.running:
        sea_level = gui.slider_float("Sea level", sea_level, sea_min, sea_max)
        nodata[:, :] = np.isnan(dem) | (dem < sea_level)
        boundaries[nodata] = 0
        boundaries[(~nodata) & (boundaries == 0)] = 1

        if gui.button("Auto boundary"):
            # Reset then mark edges and NoData neighbors as outlets
            boundaries[:, :] = np.where(nodata, 0, 1)
            from .guihelper import nodata_neighbor_mask
            valid = ~nodata
            edges = np.zeros_like(valid, dtype=bool)
            edges[0, :] = valid[0, :]
            edges[-1, :] = valid[-1, :]
            edges[:, 0] = valid[:, 0]
            edges[:, -1] = valid[:, -1]
            nb = nodata_neighbor_mask(nodata)
            boundaries[valid & (edges | nb)] = 3

        if gui.button("Reset all"):
            boundaries[:, :] = np.where(nodata, 0, 1)
            lasso_path.clear()
            lasso_mode = False

        gui.text("Boundary GUI")
        gui.text("- Auto: edges & NoData neighbors -> outlets (3)")
        gui.text("- Lasso: left-click add, right-click apply")
        gui.text("- Save: write boundary array and exit")

        if not lasso_mode:
            if gui.button("Lasso (OFF)"):
                lasso_mode = True
                lasso_wait_release = True
                lasso_path.clear()
        else:
            gui.text("[LASSO ACTIVE]")
            if gui.button("Undo last point") and lasso_path:
                lasso_path.pop()
            if gui.button("Cancel lasso"):
                lasso_path.clear()
                lasso_mode = False

        if gui.button("Save and Quit"):
            np.save(output_npy, boundaries)
            break

        if lasso_mode:
            if lasso_wait_release:
                if not (window.is_pressed(ti.ui.LMB) or window.is_pressed(ti.ui.RMB)):
                    lasso_wait_release = False
                update_display()
                canvas.set_image(img_field)
                window.show()
                continue
            if window.get_event(ti.ui.PRESS):
                ev = window.event
                if ev.key == ti.ui.LMB:
                    pos = window.get_cursor_pos()
                    if pos[0] <= vx0:
                        pass  # ignore panel clicks
                    else:
                        if vx0 <= pos[0] <= vx1 and vy0 <= pos[1] <= vy1:
                            u = (pos[0] - vx0) / (vx1 - vx0)
                            v_top = 1.0 - (pos[1] - vy0) / (vy1 - vy0 + 1e-12)
                            u = float(np.clip(u, 0.0, 1.0))
                            v_top = float(np.clip(v_top, 0.0, 1.0))
                            lasso_path.append((u, v_top))
                elif ev.key == ti.ui.RMB:
                    if len(lasso_path) > 2:
                        boundaries[:, :] = np.where(nodata, 0, 1)
                        apply_lasso_to_boundaries(boundaries, nodata, lasso_path)
                    lasso_path.clear()
                    lasso_mode = False

        update_display()
        canvas.set_image(img_field)
        window.show()

    window.destroy()


if __name__ == "__main__":
    boundary_gui()

