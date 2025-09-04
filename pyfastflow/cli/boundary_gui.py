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
    nodata_neighbor_mask,
    apply_lasso_to_boundaries,
    prepare_display_textures,
    compute_viewport,
    lasso_pixels_for_view,
    pad_to_length,
    burn_sea_level,
    live_nodata_mask,
)
from ..io import raster_to_numpy, TOPOTOOLBOX_AVAILABLE

@click.command()
@click.argument("dem_file", type=click.Path(exists=True))
@click.argument("output_npy", type=click.Path(), required=False)
@click.option("--disp-max", type=int, default=2048, show_default=True,
              help="Max display texture dimension (downscales visuals only).")
def boundary_gui(dem_file: str, output_npy: str | None = None, disp_max: int = 2048) -> None:
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

    # Optional display downscale to cap texture size for performance
    MAX_TEX = max(256, int(disp_max))
    dem_disp, terrain_u8, hill_u8, nx_disp, ny_disp, sx, sy = prepare_display_textures(dem, sea_min, sea_max, MAX_TEX)
    hs_alpha = 0.5

    # Window and GUI (fixed size for performance)
    headless = not bool(os.environ.get("DISPLAY", ""))
    WIN_W, WIN_H = 1280, 840
    PANEL_W, MARGIN = 280, 16

    window = ti.ui.Window("Boundary Editor", (WIN_W, WIN_H), show_window=not headless)
    canvas = window.get_canvas()
    gui = window.get_gui()
    # GPU fields (declared after ti.init)
    frame = ti.Vector.field(3, dtype=ti.f32, shape=(WIN_W, WIN_H))
    # Use 8-bit display textures to cut bandwidth (4x smaller)
    terrain_tex = ti.Vector.field(3, dtype=ti.u8, shape=(ny_disp, nx_disp))
    hill_tex = ti.field(dtype=ti.u8, shape=(ny_disp, nx_disp))
    dem_disp_tex = ti.field(dtype=ti.f32, shape=(ny_disp, nx_disp))
    outlet_disp_tex = ti.field(dtype=ti.u8, shape=(ny_disp, nx_disp))
    burned_disp_tex = ti.field(dtype=ti.u8, shape=(ny_disp, nx_disp))
    nan_disp_tex = ti.field(dtype=ti.u8, shape=(ny_disp, nx_disp))
    dem_field = ti.field(dtype=ti.f32, shape=(ny, nx))
    boundaries_field = ti.field(dtype=ti.u8, shape=(ny, nx))
    # Uniforms as 0D fields
    sea_level_k = ti.field(dtype=ti.f32, shape=())
    vp_x0_k = ti.field(dtype=ti.i32, shape=())
    vp_y0_k = ti.field(dtype=ti.i32, shape=())
    vp_w_k = ti.field(dtype=ti.i32, shape=())
    vp_h_k = ti.field(dtype=ti.i32, shape=())
    nx_k = ti.field(dtype=ti.i32, shape=())
    ny_k = ti.field(dtype=ti.i32, shape=())
    nx_disp_k = ti.field(dtype=ti.i32, shape=())
    ny_disp_k = ti.field(dtype=ti.i32, shape=())
    sx_k = ti.field(dtype=ti.f32, shape=())  # display texture x scale = nx_disp/nx
    sy_k = ti.field(dtype=ti.f32, shape=())  # display texture y scale = ny_disp/ny
    view_x_min_k = ti.field(dtype=ti.f32, shape=())
    view_y_min_top_k = ti.field(dtype=ti.f32, shape=())
    view_w_k = ti.field(dtype=ti.f32, shape=())
    view_h_k = ti.field(dtype=ti.f32, shape=())
    hs_alpha_k = ti.field(dtype=ti.f32, shape=())
    win_w_k = ti.field(dtype=ti.i32, shape=())
    win_h_k = ti.field(dtype=ti.i32, shape=())
    # Upload static textures and data
    terrain_tex.from_numpy(terrain_u8)
    hill_tex.from_numpy(hill_u8)
    dem_disp_tex.from_numpy(dem_disp.astype(np.float32))

    # Build display-resolution masks (resize boolean masks with bilinear then threshold)
    from ..rastermanip import resize_raster
    def rebuild_display_masks():
        scale = sx  # same as sy
        outlet_mask = (boundaries == 3).astype(np.float32)
        burned_mask = (boundaries == 0).astype(np.float32)
        nan_mask = np.isnan(dem).astype(np.float32)
        outlet_ds = (resize_raster(outlet_mask, scale) >= 0.5).astype(np.uint8)
        burned_ds = (resize_raster(burned_mask, scale) >= 0.5).astype(np.uint8)
        nan_ds = (resize_raster(nan_mask, scale) >= 0.5).astype(np.uint8)
        outlet_disp_tex.from_numpy(outlet_ds)
        burned_disp_tex.from_numpy(burned_ds)
        nan_disp_tex.from_numpy(nan_ds)

    rebuild_display_masks()
    dem_field.from_numpy(dem)
    boundaries_field.from_numpy(boundaries)

    # Define Taichi kernels after fields exist so they can close over them
    @ti.kernel
    def render_kernel():
        for x, y in ti.ndrange(win_w_k[None], win_h_k[None]):
            c = ti.Vector([0.0, 0.0, 0.0])
            if x >= vp_x0_k[None] and x < vp_x0_k[None] + vp_w_k[None] and y >= vp_y0_k[None] and y < vp_y0_k[None] + vp_h_k[None]:
                u = (ti.cast(x - vp_x0_k[None], ti.f32) + 0.5) / ti.max(1, vp_w_k[None])
                v_bottom = (ti.cast(y - vp_y0_k[None], ti.f32) + 0.5) / ti.max(1, vp_h_k[None])
                v_top = 1.0 - v_bottom
                ax = view_x_min_k[None] + u * view_w_k[None]
                ay = view_y_min_top_k[None] + v_top * view_h_k[None]
                ix = ti.max(0, ti.min(nx_k[None] - 1, ti.i32(ax)))
                iy = ti.max(0, ti.min(ny_k[None] - 1, ti.i32(ay)))
                # Map to display textures (downsampled) using scale factors
                ix_ds = ti.max(0, ti.min(nx_disp_k[None] - 1, ti.i32(ti.cast(ix, ti.f32) * sx_k[None])))
                iy_ds = ti.max(0, ti.min(ny_disp_k[None] - 1, ti.i32(ti.cast(iy, ti.f32) * sy_k[None])))
                t_u8 = terrain_tex[iy_ds, ix_ds]
                tcol = ti.Vector([ti.cast(t_u8[0], ti.f32), ti.cast(t_u8[1], ti.f32), ti.cast(t_u8[2], ti.f32)]) / 255.0
                h_u8 = hill_tex[iy_ds, ix_ds]
                hval = ti.cast(h_u8, ti.f32) / 255.0
                hcol = ti.Vector([hval, hval, hval])
                base_col = (1.0 - hs_alpha_k[None]) * tcol + hs_alpha_k[None] * hcol
                # Display-resolution overlays
                is_outlet = outlet_disp_tex[iy_ds, ix_ds] != ti.u8(0)
                is_burned = burned_disp_tex[iy_ds, ix_ds] != ti.u8(0)
                is_nan = nan_disp_tex[iy_ds, ix_ds] != ti.u8(0)
                is_sea = dem_disp_tex[iy_ds, ix_ds] < sea_level_k[None]
                col = base_col
                if is_outlet:
                    col = ti.Vector([1.0, 0.0, 0.0])
                elif is_burned or is_nan:
                    col = ti.Vector([0.0, 0.0, 0.0])
                elif is_sea:
                    col = ti.Vector([0.5, 0.5, 0.5])
                c = col
            frame[x, y] = c

    # Lasso GPU buffers (fixed max points)
    NMAX_LASSO = 4096
    lasso_px = ti.field(dtype=ti.i32, shape=NMAX_LASSO)
    lasso_py = ti.field(dtype=ti.i32, shape=NMAX_LASSO)
    lasso_n = ti.field(dtype=ti.i32, shape=())

    @ti.kernel
    def draw_segments_kernel():
        n = lasso_n[None]
        for i in range(n - 1):
            x0 = lasso_px[i]
            y0 = lasso_py[i]
            x1 = lasso_px[i + 1]
            y1 = lasso_py[i + 1]
            dx = x1 - x0
            dy = y1 - y0
            steps = ti.max(ti.abs(dx), ti.abs(dy))
            if steps <= 0:
                if 0 <= x0 and x0 < win_w_k[None] and 0 <= y0 and y0 < win_h_k[None]:
                    frame[x0, y0] = ti.Vector([0.0, 1.0, 0.0])
            else:
                sx = ti.cast(dx, ti.f32) / ti.cast(steps, ti.f32)
                sy = ti.cast(dy, ti.f32) / ti.cast(steps, ti.f32)
                fx = ti.cast(x0, ti.f32)
                fy = ti.cast(y0, ti.f32)
                for s in range(steps + 1):
                    xi = ti.i32(fx + 0.5)
                    yi = ti.i32(fy + 0.5)
                    if 0 <= xi and xi < win_w_k[None] and 0 <= yi and yi < win_h_k[None]:
                        frame[xi, yi] = ti.Vector([0.0, 1.0, 0.0])
                    fx += sx
                    fy += sy

    # Viewport rect that preserves DEM aspect ratio inside available area
    vp_x0, vp_y0, vp_w, vp_h, vx0, vx1, vy0, vy1 = compute_viewport(WIN_W, WIN_H, PANEL_W, MARGIN, nx, ny)

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
        # Determine view window and effective viewport rect
        if zoom >= 1.0:
            view_w = nx / max(zoom, 1e-6)
            view_h = ny / max(zoom, 1e-6)
            vx0_i = vp_x0
            vy0_i = vp_y0
            vw_i = vp_w
            vh_i = vp_h
            vxmin = view_x_min
            vymin_top = view_y_min
        else:
            view_w = float(nx)
            view_h = float(ny)
            vxmin = 0.0
            vymin_top = 0.0
            out_w = max(1, int(vp_w * zoom))
            out_h = max(1, int(vp_h * zoom))
            base_x = vp_x0 + (vp_w - out_w) // 2
            base_y = vp_y0 + (vp_h - out_h) // 2
            max_off_x = (vp_w - out_w) // 2
            max_off_y = (vp_h - out_h) // 2
            ox = int(np.clip(disp_off_x_px, -max_off_x, max_off_x))
            oy = int(np.clip(disp_off_y_px, -max_off_y, max_off_y))
            vx0_i = base_x + ox
            vy0_i = base_y + oy
            vw_i = out_w
            vh_i = out_h

        # Render full frame on GPU
        # Update uniforms
        sea_level_k[None] = float(sea_level)
        vp_x0_k[None] = int(vx0_i)
        vp_y0_k[None] = int(vy0_i)
        vp_w_k[None] = int(vw_i)
        vp_h_k[None] = int(vh_i)
        nx_k[None] = int(nx)
        ny_k[None] = int(ny)
        nx_disp_k[None] = int(nx_disp)
        ny_disp_k[None] = int(ny_disp)
        sx_k[None] = float(nx_disp) / float(nx)
        sy_k[None] = float(ny_disp) / float(ny)
        view_x_min_k[None] = float(vxmin)
        view_y_min_top_k[None] = float(vymin_top)
        view_w_k[None] = float(view_w)
        view_h_k[None] = float(view_h)
        hs_alpha_k[None] = float(hs_alpha)
        win_w_k[None] = int(WIN_W)
        win_h_k[None] = int(WIN_H)
        render_kernel()
        # Draw lasso overlay as polyline
        if lasso_mode and lasso_path:
            import numpy as _np
            pts = lasso_pixels_for_view(lasso_path, nx, ny, vx0_i, vy0_i, vw_i, vh_i, vxmin, vymin_top, view_w, view_h, zoom)
            if pts:
                if len(pts) >= 3:
                    pts = pts + [pts[0]]
                px_np = _np.array([p[0] for p in pts], dtype=_np.int32)
                py_np = _np.array([p[1] for p in pts], dtype=_np.int32)
                lasso_n[None] = int(min(px_np.shape[0], NMAX_LASSO))
                lpx = pad_to_length(px_np, NMAX_LASSO, 0)
                lpy = pad_to_length(py_np, NMAX_LASSO, 0)
                lasso_px.from_numpy(lpx)
                lasso_py.from_numpy(lpy)
                draw_segments_kernel()

    update_display()

    if headless:
        np.save(output_npy, boundaries)
        print(f"Boundary conditions saved to {output_npy}")
        window.destroy()
        return

    while window.running:
        gui.text("Control")
        gui.text("Sea Level (grey overlay until burned)")
        sea_level = gui.slider_float("Sea level", sea_level, sea_min, sea_max)
        # Fine controls for sea level: --, -, +, ++ (10x and 1x steps)
        if gui.button("--"):
            sea_level = max(sea_min, sea_level - 10.0)
        if gui.button("-"):
            sea_level = max(sea_min, sea_level - 1.0)
        if gui.button("+"):
            sea_level = min(sea_max, sea_level + 1.0)
        if gui.button("++"):
            sea_level = min(sea_max, sea_level + 10.0)

        if gui.button("Burn sea level to BC (0)"):
            sea_mask = dem < sea_level
            boundaries[sea_mask] = 0
            boundaries_field.from_numpy(boundaries)
            rebuild_display_masks()

        gui.text("Boundary Tools")
        if gui.button("Auto boundary"):
            # Reset then mark edges and NoData neighbors as outlets
            nodata = np.isnan(dem) | (dem < sea_level)
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
            boundaries_field.from_numpy(boundaries)
            rebuild_display_masks()

        if gui.button("Reset all"):
            nodata = np.isnan(dem) | (dem < sea_level)
            boundaries[:, :] = np.where(nodata, 0, 1)
            lasso_path.clear()
            lasso_mode = False
            boundaries_field.from_numpy(boundaries)
            rebuild_display_masks()

        gui.text("Lasso Selection (left-click add, right-click apply)")

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

        # Save/Zoom controls
        gui.text("Session")
        if gui.button("Save and Quit"):
            np.save(output_npy, boundaries)
            break
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

        # end Session group

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
                canvas.set_image(frame)
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
                        # Build live nodata mask from burned or current sea level
                        nodata = live_nodata_mask(boundaries, dem, sea_level)
                        boundaries[:, :] = np.where(nodata, 0, 1)
                        apply_lasso_to_boundaries(boundaries, nodata, lasso_path)
                        boundaries_field.from_numpy(boundaries)
                    lasso_path.clear()
                    lasso_mode = False

        update_display()
        canvas.set_image(frame)
        window.show()

    window.destroy()


if __name__ == "__main__":
    boundary_gui()
