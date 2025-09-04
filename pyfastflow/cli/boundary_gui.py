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
    sample_view_raw,
    sample_view_mask,
    sample_view_uint8,
    draw_polyline,
    draw_points_px,
    nodata_neighbor_mask,
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

    # Window and GUI (fixed size for performance)
    headless = not bool(os.environ.get("DISPLAY", ""))
    WIN_W, WIN_H = 1280, 840
    PANEL_W, MARGIN = 280, 16

    window = ti.ui.Window("Boundary Editor", (WIN_W, WIN_H), show_window=not headless)
    canvas = window.get_canvas()
    gui = window.get_gui()
    img_field = ti.Vector.field(3, dtype=ti.f32, shape=(WIN_W, WIN_H))

    # Viewport rect that preserves DEM aspect ratio inside available area
    avail_x0 = PANEL_W + MARGIN
    avail_y0 = MARGIN
    avail_w = WIN_W - PANEL_W - 2 * MARGIN
    avail_h = WIN_H - 2 * MARGIN
    dem_aspect = nx / ny
    vp_w = min(avail_w, int(avail_h * dem_aspect))
    vp_h = min(avail_h, int(avail_w / dem_aspect))
    vp_x0 = avail_x0 + (avail_w - vp_w) // 2
    vp_y0 = avail_y0 + (avail_h - vp_h) // 2
    # Normalized bounds for quick tests (cursor coords)
    vx0 = vp_x0 / WIN_W
    vx1 = (vp_x0 + vp_w) / WIN_W
    vy0 = vp_y0 / WIN_H
    vy1 = (vp_y0 + vp_h) / WIN_H

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
        buffer = np.zeros((WIN_W, WIN_H, 3), dtype=np.float32)
        if zoom >= 1.0:
            # Sample current view window to viewport size
            view_w = nx / max(zoom, 1e-6)
            view_h = ny / max(zoom, 1e-6)
            t_small = sample_view_raw(terrain_rgb, view_x_min, view_y_min, view_w, view_h, vp_w, vp_h)
            h_small = sample_view_raw(hill_rgb, view_x_min, view_y_min, view_w, view_h, vp_w, vp_h)
            nod_small = sample_view_mask(np.isnan(dem) | (dem < sea_level), view_x_min, view_y_min, view_w, view_h, vp_w, vp_h)
            b_small = sample_view_uint8(boundaries, view_x_min, view_y_min, view_w, view_h, vp_w, vp_h)
            view_rgb = (1.0 - hs_alpha) * t_small + hs_alpha * h_small
            view_rgb[nod_small] = 0.0
            view_rgb[b_small == 3] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            # Lasso overlay: map full DEM normalized coords to viewport pixels
            if lasso_mode and lasso_path:
                pts = []
                for u, v in lasso_path:
                    x = u * nx
                    y = v * ny
                    px = int((x - view_x_min) / view_w * vp_w)
                    py = int((y - view_y_min) / view_h * vp_h)
                    pts.append((px, py))
                draw_points_px(view_rgb, pts)
                draw_polyline(view_rgb, pts, close=len(pts) >= 3)
            rgb_disp = array_to_canvas(view_rgb)
            buffer[vp_x0:vp_x0 + vp_w, vp_y0:vp_y0 + vp_h, :] = rgb_disp
        else:
            # Render full DEM scaled down within viewport
            out_w = max(1, int(vp_w * zoom))
            out_h = max(1, int(vp_h * zoom))
            t_small = sample_view_raw(terrain_rgb, 0.0, 0.0, nx, ny, out_w, out_h)
            h_small = sample_view_raw(hill_rgb, 0.0, 0.0, nx, ny, out_w, out_h)
            nod_small = sample_view_mask(np.isnan(dem) | (dem < sea_level), 0.0, 0.0, nx, ny, out_w, out_h)
            b_small = sample_view_uint8(boundaries, 0.0, 0.0, nx, ny, out_w, out_h)
            view_rgb = (1.0 - hs_alpha) * t_small + hs_alpha * h_small
            view_rgb[nod_small] = 0.0
            view_rgb[b_small == 3] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            # Lasso overlay: small image covers full DEM
            if lasso_mode and lasso_path:
                pts = [(int(u * out_w), int(v * out_h)) for (u, v) in lasso_path]
                draw_points_px(view_rgb, pts)
                draw_polyline(view_rgb, pts, close=len(pts) >= 3)
            rgb_disp = array_to_canvas(view_rgb)  # (out_w, out_h, 3)
            # Place centered within viewport with offsets
            base_x = vp_x0 + (vp_w - out_w) // 2
            base_y = vp_y0 + (vp_h - out_h) // 2
            max_off_x = (vp_w - out_w) // 2
            max_off_y = (vp_h - out_h) // 2
            nonlocal disp_off_x_px, disp_off_y_px
            ox = int(np.clip(disp_off_x_px, -max_off_x, max_off_x))
            oy = int(np.clip(disp_off_y_px, -max_off_y, max_off_y))
            x0 = base_x + ox
            y0 = base_y + oy
            buffer[x0:x0 + out_w, y0:y0 + out_h, :] = rgb_disp

        img_field.from_numpy(buffer)

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
            step_x = max(1, int(pan_frac * vp_w))
            step_y = max(1, int(pan_frac * vp_h))
            out_w = max(1, int(vp_w * zoom))
            out_h = max(1, int(vp_h * zoom))
            max_off_x = (vp_w - out_w) // 2
            max_off_y = (vp_h - out_h) // 2
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
                        # Map click to full DEM normalized coords; accept outside viewport
                        px = pos[0] * WIN_W
                        py_top = (1.0 - pos[1]) * WIN_H
                        if zoom >= 1.0:
                            u_rel = (px - vp_x0) / max(vp_w, 1)
                            v_rel = (py_top - vp_y0) / max(vp_h, 1)
                            x = view_x_min + u_rel * (nx / max(zoom, 1e-6))
                            y = view_y_min + v_rel * (ny / max(zoom, 1e-6))
                            u_full = float(np.clip(x / nx, 0.0, 1.0))
                            v_full = float(np.clip(y / ny, 0.0, 1.0))
                        else:
                            out_w = max(1, int(vp_w * zoom))
                            out_h = max(1, int(vp_h * zoom))
                            base_x = vp_x0 + (vp_w - out_w) // 2
                            base_y = vp_y0 + (vp_h - out_h) // 2
                            max_off_x = (vp_w - out_w) // 2
                            max_off_y = (vp_h - out_h) // 2
                            ox = int(np.clip(disp_off_x_px, -max_off_x, max_off_x))
                            oy = int(np.clip(disp_off_y_px, -max_off_y, max_off_y))
                            img_x0 = base_x + ox
                            img_y0 = base_y + oy
                            u_rel = (px - img_x0) / max(out_w, 1)
                            v_rel = (py_top - img_y0) / max(out_h, 1)
                            u_full = float(np.clip(u_rel, 0.0, 1.0))
                            v_full = float(np.clip(v_rel, 0.0, 1.0))
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
