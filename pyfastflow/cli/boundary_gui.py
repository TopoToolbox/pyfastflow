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
    sample_view_raw,
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

    # Simple pan/zoom state (array coordinates)
    zoom = 1.0
    zoom_min, zoom_max = 0.5, 8.0  # allow unzoom below 1.0
    view_x_min = 0.0
    view_y_min = 0.0  # row from top
    # When zoom < 1.0, we render a scaled image inside the viewport; allow display offsets (in pixels)
    disp_off_x_px = 0.0
    disp_off_y_px = 0.0

    def update_display() -> None:
        # Compose terrain + hillshade once per frame (full res, array space)
        rgb_raw = (1.0 - hs_alpha) * terrain_rgb + hs_alpha * hill_rgb
        rgb_raw[np.isnan(dem) | (dem < sea_level)] = 0.0
        rgb_raw[boundaries == 3] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        # Lasso overlay drawn in array space (full image)
        if lasso_mode and lasso_path:
            draw_lasso_points(rgb_raw, lasso_path, nx, ny)
            if len(lasso_path) >= 2:
                draw_lasso_polygon(rgb_raw, lasso_path, nx, ny)
        if zoom >= 1.0:
            # Sample current view window to viewport size (ny,nx)
            view_w = nx / max(zoom, 1e-6)
            view_h = ny / max(zoom, 1e-6)
            view_rgb_raw = sample_view_raw(rgb_raw, view_x_min, view_y_min, view_w, view_h, nx, ny)
            rgb_disp = array_to_canvas(view_rgb_raw)
            padded, _ = place_on_padded(rgb_disp, nx, ny, pad_left, pad_right, pad_top, pad_bottom)
        else:
            # Render smaller image inside the viewport and allow shifting with offsets
            out_w = max(1, int(nx * zoom))
            out_h = max(1, int(ny * zoom))
            view_rgb_raw = sample_view_raw(rgb_raw, 0.0, 0.0, nx, ny, out_w, out_h)
            rgb_disp_small = array_to_canvas(view_rgb_raw)  # (out_w, out_h, 3)
            # Compose padded buffer and place the small image with offsets inside viewport
            disp_w = nx + pad_left + pad_right
            disp_h = ny + pad_top + pad_bottom
            padded = np.zeros((disp_w, disp_h, 3), dtype=np.float32)
            # Base placement centered in viewport
            base_x = pad_left + (nx - out_w) // 2
            base_y = pad_bottom + (ny - out_h) // 2
            # Allowed offset range keeps image within viewport bounds
            max_off_x = (nx - out_w) // 2
            max_off_y = (ny - out_h) // 2
            ox = int(np.clip(disp_off_x_px, -max_off_x, max_off_x))
            oy = int(np.clip(disp_off_y_px, -max_off_y, max_off_y))
            x0 = base_x + ox
            y0 = base_y + oy
            padded[x0:x0 + out_w, y0:y0 + out_h, :] = rgb_disp_small
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

        # Zoom buttons
        if gui.button("Zoom +"):
            # Keep center fixed
            if zoom >= 1.0:
                cx = view_x_min + (nx / max(zoom, 1e-6)) * 0.5
                cy = view_y_min + (ny / max(zoom, 1e-6)) * 0.5
                zoom = min(zoom_max, zoom * 1.25)
                vw = nx / zoom
                vh = ny / zoom
                view_x_min = float(np.clip(cx - vw * 0.5, 0.0, max(0.0, nx - vw)))
                view_y_min = float(np.clip(cy - vh * 0.5, 0.0, max(0.0, ny - vh)))
            else:
                # From small image, zoom in keeping display offsets
                zoom = min(zoom_max, zoom * 1.25)
                if zoom >= 1.0:
                    # Transition to content sampling; center view
                    view_x_min = 0.0
                    view_y_min = 0.0
                # Keep disp_off as-is
        if gui.button("Zoom -"):
            if zoom >= 1.0:
                cx = view_x_min + (nx / max(zoom, 1e-6)) * 0.5
                cy = view_y_min + (ny / max(zoom, 1e-6)) * 0.5
                zoom = max(zoom_min, zoom / 1.25)
                if zoom >= 1.0:
                    vw = nx / zoom
                    vh = ny / zoom
                    view_x_min = float(np.clip(cx - vw * 0.5, 0.0, max(0.0, nx - vw)))
                    view_y_min = float(np.clip(cy - vh * 0.5, 0.0, max(0.0, ny - vh)))
                else:
                    # Transition to small-image mode; center image and reset content view
                    view_x_min = 0.0
                    view_y_min = 0.0
                    disp_off_x_px = 0.0
                    disp_off_y_px = 0.0
            else:
                zoom = max(zoom_min, zoom / 1.25)

        if gui.button("Save and Quit"):
            np.save(output_npy, boundaries)
            break

        # Keyboard panning: Arrow keys (preferred) with WASD fallback
        pan_frac = 0.05
        vw = nx / max(zoom, 1e-6)
        vh = ny / max(zoom, 1e-6)
        left_key = getattr(ti.ui, 'LEFT', None)
        right_key = getattr(ti.ui, 'RIGHT', None)
        up_key = getattr(ti.ui, 'UP', None)
        down_key = getattr(ti.ui, 'DOWN', None)
        A = getattr(ti.ui, 'A', None)
        D = getattr(ti.ui, 'D', None)
        W = getattr(ti.ui, 'W', None)
        S = getattr(ti.ui, 'S', None)
        if zoom >= 1.0:
            if (left_key and window.is_pressed(left_key)) or (A and window.is_pressed(A)):
                view_x_min = float(np.clip(view_x_min - pan_frac * vw, 0.0, max(0.0, nx - vw)))
            if (right_key and window.is_pressed(right_key)) or (D and window.is_pressed(D)):
                view_x_min = float(np.clip(view_x_min + pan_frac * vw, 0.0, max(0.0, nx - vw)))
            if (up_key and window.is_pressed(up_key)) or (W and window.is_pressed(W)):
                view_y_min = float(np.clip(view_y_min - pan_frac * vh, 0.0, max(0.0, ny - vh)))
            if (down_key and window.is_pressed(down_key)) or (S and window.is_pressed(S)):
                view_y_min = float(np.clip(view_y_min + pan_frac * vh, 0.0, max(0.0, ny - vh)))
        else:
            # Adjust display offsets in pixels within viewport
            step_x = max(1, int(pan_frac * nx))
            step_y = max(1, int(pan_frac * ny))
            out_w = max(1, int(nx * zoom))
            out_h = max(1, int(ny * zoom))
            max_off_x = (nx - out_w) // 2
            max_off_y = (ny - out_h) // 2
            if (left_key and window.is_pressed(left_key)) or (A and window.is_pressed(A)):
                disp_off_x_px = float(np.clip(disp_off_x_px - step_x, -max_off_x, max_off_x))
            if (right_key and window.is_pressed(right_key)) or (D and window.is_pressed(D)):
                disp_off_x_px = float(np.clip(disp_off_x_px + step_x, -max_off_x, max_off_x))
            if (up_key and window.is_pressed(up_key)) or (W and window.is_pressed(W)):
                disp_off_y_px = float(np.clip(disp_off_y_px - step_y, -max_off_y, max_off_y))
            if (down_key and window.is_pressed(down_key)) or (S and window.is_pressed(S)):
                disp_off_y_px = float(np.clip(disp_off_y_px + step_y, -max_off_y, max_off_y))

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
                        # Allow clicks outside viewport; clamp to edges
                        u_img = (pos[0] - vx0) / (vx1 - vx0)
                        v_img_top = 1.0 - (pos[1] - vy0) / (vy1 - vy0 + 1e-12)
                        # Map to array coordinates through current view
                        vw = nx / max(zoom, 1e-6)
                        vh = ny / max(zoom, 1e-6)
                        x = view_x_min + u_img * vw
                        y = view_y_min + v_img_top * vh
                        u_full = float(np.clip(x / nx, 0.0, 1.0))
                        v_full = float(np.clip(y / ny, 0.0, 1.0))
                        lasso_path.append((u_full, v_full))
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
