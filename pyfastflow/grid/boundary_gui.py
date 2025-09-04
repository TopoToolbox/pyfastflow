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
import matplotlib.cm as cm
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

    # Initialise Taichi (prefer GPU but fall back to CPU) BEFORE Taichi kernels
    try:
        ti.init(arch=ti.gpu)
    except Exception:
        ti.init(arch=ti.cpu)

    # Precompute visualization layers (efficient: computed once)
    from .. import constants as cte
    from ..visu.hillshading import hillshade_numpy

    valid_mask = ~np.isnan(dem)
    dem_norm = np.zeros_like(dem, dtype=np.float32)
    if np.any(valid_mask):
        dem_norm[valid_mask] = (dem[valid_mask] - sea_min) / (sea_max - sea_min + 1e-6)

    terrain_cmap = cm.get_cmap('terrain')
    terrain_rgb = terrain_cmap(np.clip(dem_norm, 0.0, 1.0))[..., :3].astype(np.float32)

    try:
        hill = hillshade_numpy(dem.astype(np.float32), dx=cte.DX)
    except Exception:
        hill = hillshade_numpy(dem.astype(np.float32))
    hill = np.nan_to_num(hill, nan=0.0).astype(np.float32)
    hill_rgb = np.stack([hill, hill, hill], axis=-1)
    hs_alpha = 0.5

    # Check if we're in a headless environment
    headless = not bool(os.environ.get('DISPLAY', ''))
    
    # Define visible image viewport with explicit padding and a left margin for GUI panel
    base_pad = max(20, int(0.06 * min(nx, ny)))  # pixels relative to DEM size
    panel_px = max(base_pad, int(0.22 * nx))  # left GUI panel width in pixels
    pad_left = panel_px
    pad_right = base_pad
    pad_top = base_pad
    pad_bottom = base_pad
    disp_w = nx + pad_left + pad_right
    disp_h = ny + pad_top + pad_bottom

    # Create window sized exactly like the padded display to avoid scaling mismatch
    window = ti.ui.Window("Boundary Editor", (disp_w, disp_h), show_window=not headless)
    canvas = window.get_canvas()
    gui = window.get_gui()

    # Normalized viewport bounds inside the window/canvas for the DEM image
    vx0 = pad_left / disp_w
    vx1 = (pad_left + nx) / disp_w
    vy0 = pad_bottom / disp_h
    vy1 = (pad_bottom + ny) / disp_h

    # Taichi canvas expects image shaped (width, height). Use padded display size.
    img_field = ti.Vector.field(3, dtype=ti.f32, shape=(disp_w, disp_h))

    lasso_mode = False
    lasso_wait_release = False  # wait for mouse release after enabling lasso
    lasso_path: list[tuple[float, float]] = []

    def update_display() -> None:
        """Update image field used for visualization."""

        # Only normalize valid (non-NaN) values
        valid_mask = ~np.isnan(dem)
        norm = np.zeros_like(dem)
        if np.any(valid_mask):
            norm[valid_mask] = (dem[valid_mask] - sea_min) / (sea_max - sea_min + 1e-6)
        
        # Compose color: terrain colormap blended with hillshade
        rgb_raw = (1.0 - hs_alpha) * terrain_rgb + hs_alpha * hill_rgb
        # Mask NoData as black based on current sea level
        rgb_raw[nodata] = 0.0
        # Overlay outlets in red
        rgb_raw[boundaries == 3] = np.array([1.0, 0.0, 0.0])
        
        # Draw lasso polygon if in lasso mode and have points
        if lasso_mode and len(lasso_path) >= 1:
            draw_lasso_points(rgb_raw)
            if len(lasso_path) >= 2:
                draw_lasso_polygon(rgb_raw)
        # Convert to canvas coordinates: (x, y) = (col, row-from-bottom)
        # Start from raster-like (row-from-top, col-from-left): transpose then flip Y
        rgb_disp = np.transpose(rgb_raw, (1, 0, 2))[:, ::-1, :].astype(np.float32)

        # Place into padded display buffer to create visible margins around the image
        padded = np.zeros((disp_w, disp_h, 3), dtype=np.float32)
        padded[pad_left:pad_left + nx, pad_bottom:pad_bottom + ny, :] = rgb_disp
        img_field.from_numpy(padded)
        
    def draw_lasso_polygon(rgb: np.ndarray) -> None:
        """Draw lasso polygon on the display."""
        if len(lasso_path) < 2:
            return
            
        # Convert normalized coordinates to pixel coordinates
        points = []
        for p in lasso_path:
            x = int(p[0] * nx)  # Convert from normalized to pixel coordinates
            y = int(p[1] * ny)  # Convert from normalized to pixel coordinates
            points.append((x, y))
        
        # Draw lines between consecutive points
        for i in range(len(points) - 1):
            draw_line(rgb, points[i], points[i + 1], color=[0.0, 1.0, 0.0])  # Green
            
        # If we have 3+ points, draw closing line
        if len(points) >= 3:
            draw_line(rgb, points[-1], points[0], color=[0.0, 1.0, 0.0])  # Green
    
    def draw_line(rgb: np.ndarray, p1: tuple, p2: tuple, color: list) -> None:
        """Draw a line between two points using Bresenham's algorithm."""
        x1, y1 = p1
        x2, y2 = p2
        
        # Clamp coordinates to image bounds
        x1 = max(0, min(nx - 1, x1))
        y1 = max(0, min(ny - 1, y1))
        x2 = max(0, min(nx - 1, x2))
        y2 = max(0, min(ny - 1, y2))
        
        # Simple line drawing (Bresenham's algorithm simplified)
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
                
    def draw_lasso_points(rgb: np.ndarray) -> None:
        """Draw red dots at lasso click positions for debugging."""
        for i, p in enumerate(lasso_path):
            x = int(p[0] * nx)
            y = int(p[1] * ny)
            # Draw a 3x3 red square at click position
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    px, py = x + dx, y + dy
                    if 0 <= px < nx and 0 <= py < ny:
                        rgb[py, px] = [1.0, 0.0, 1.0]  # Magenta for visibility

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

        poly = np.array([[p[0] * nx, p[1] * ny] for p in path])
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
            # Controls overlay (top-left). We reserve left viewport in canvas for it.
            sea_level = gui.slider_float("Sea level", sea_level, sea_min, sea_max)
            nodata[:, :] = np.isnan(dem) | (dem < sea_level)
            boundaries[nodata] = 0
            boundaries[~nodata & (boundaries == 0)] = 1

            if gui.button("Auto boundary"):
                apply_auto_boundary()
            if gui.button("Reset all"):
                # Reset to base: 0 for NoData, 1 for valid; remove outlets
                boundaries[:, :] = np.where(nodata, 0, 1)
                lasso_path.clear()
                lasso_mode = False

            # Help text
            gui.text("Boundary GUI")
            gui.text("- Auto: edges & NoData neighbors -> outlets (3)")
            gui.text("- Lasso: left-click add, right-click apply")
            # gui.text for zoom/pan intentionally omitted to keep UI simple and stable
            gui.text("- Save: write boundary array and exit")

            # Lasso controls
            if not lasso_mode:
                if gui.button("Lasso (OFF)"):
                    lasso_mode = True
                    lasso_wait_release = True  # ignore pending mouse presses from the button click
                    lasso_path.clear()
                    print("Lasso activated. Left-click to add points; Right-click to apply.")
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
                # Arm lasso only after all mouse buttons are released post toggle
                if lasso_wait_release:
                    if not (window.is_pressed(ti.ui.LMB) or window.is_pressed(ti.ui.RMB)):
                        lasso_wait_release = False
                    # Skip handling events until release detected
                    update_display()
                    canvas.set_image(img_field)
                    window.show()
                    continue
                # Check for mouse press events to add points
                if window.get_event(ti.ui.PRESS):
                    event = window.event
                    if event.key == ti.ui.LMB:
                        pos = window.get_cursor_pos()
                        # Ignore clicks inside GUI panel region only
                        if pos[0] <= vx0:
                            print(f"Ignored click on GUI panel: {pos}")
                        else:
                            # Map window coords to normalized image coords (origin top-left)
                            if vx0 <= pos[0] <= vx1 and vy0 <= pos[1] <= vy1:
                                u = (pos[0] - vx0) / (vx1 - vx0)
                                v_top = 1.0 - (pos[1] - vy0) / (vy1 - vy0 + 1e-12)
                            else:
                                u = v_top = None
                            if u is not None:
                                u = float(np.clip(u, 0.0, 1.0))
                                v_top = float(np.clip(v_top, 0.0, 1.0))
                                lasso_path.append((u, v_top))
                                px = int(u * nx)
                                py = int(v_top * ny)
                                print(f"Added lasso point {len(lasso_path)}: raw{pos} -> uv({u:.3f}, {v_top:.3f}) -> pixel({px}, {py})")
                            else:
                                print(f"Ignored click outside image viewport: {pos}")
                    elif event.key == ti.ui.RMB:
                        if len(lasso_path) > 2:
                            # Reset boundaries before applying lasso
                            boundaries[:, :] = np.where(nodata, 0, 1)
                            apply_lasso(lasso_path)
                            print(f"Applied lasso selection with {len(lasso_path)} points")
                            # Force display update to show new boundaries
                            update_display()
                        else:
                            print(f"Need at least 3 points for lasso, got {len(lasso_path)}")
                        lasso_path.clear()
                        lasso_mode = False

            update_display()
            canvas.set_image(img_field)
            window.show()

    window.destroy()


if __name__ == "__main__":
    boundary_gui()
