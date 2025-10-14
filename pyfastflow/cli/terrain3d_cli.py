from __future__ import annotations

import os
from typing import Optional
import math

import click
import numpy as np

import taichi as ti


def _init_taichi_safe():
    for arch in (ti.cuda, ti.vulkan, ti.metal, ti.opengl, ti.cpu):
        try:
            ti.init(arch=arch)
            return
        except Exception:
            pass
    ti.init(arch=ti.cpu)

def _read_array(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
    else:
        try:
            import rasterio as rio  # type: ignore
        except Exception as e:
            raise SystemExit("Reading GeoTIFF requires rasterio. Install optional extra 'pyfastflow[3dterrain]'.") from e
        with rio.open(path) as ds:
            arr = ds.read(1)
    # Return raw float32 array (no normalization here)
    arr = np.asarray(arr, dtype=np.float32)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return arr


def _make_height_from_raw(raw: np.ndarray, threshold: float, water: Optional[np.ndarray] = None) -> np.ndarray:
    """Build a normalized height map from raw values, masking values < threshold.

    - If water is provided, it's added to terrain BEFORE normalization
    - Values < threshold are tagged as masked and mapped to a small negative sentinel
      so they can be discarded in the fragment shader (keeps mesh intact).
    - Normalization [0,1] is computed from finite, unmasked values only.
    """
    a = np.asarray(raw, dtype=np.float32)

    # Add water to terrain if provided (BEFORE normalization)
    if water is not None:
        water_arr = np.asarray(water, dtype=np.float32)
        if water_arr.shape == a.shape:
            a = a + water_arr

    finite = np.isfinite(a)
    mask_valid = finite & (a >= float(threshold))
    if not np.any(mask_valid):
        # Fallback: nothing above threshold; just return zeros (no crash)
        out = np.full_like(a, -1.0, dtype=np.float32)
        if not out.flags.c_contiguous:
            out = np.ascontiguousarray(out)
        return out
    vmin = float(np.min(a[mask_valid]))
    vmax = float(np.max(a[mask_valid]))
    denom = max(1e-6, (vmax - vmin))
    out = np.empty_like(a, dtype=np.float32)
    # valid → normalized [0,1]
    out[mask_valid] = (a[mask_valid] - vmin) / denom
    # masked or non-finite → negative sentinel so shader can discard
    out[~mask_valid] = -1.0
    if not out.flags.c_contiguous:
        out = np.ascontiguousarray(out)
    return out


def _file_dialog() -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        # Put "All files" first so everything is visible by default.
        # Use space-separated patterns for multi-extensions (Tk expects spaces, not semicolons).
        path = filedialog.askopenfilename(
            title="Select input (NumPy .npy or GeoTIFF)",
            filetypes=[
                ("All files", "*.*"),
                ("NumPy", "*.npy"),
                ("GeoTIFF", "*.tif *.tiff"),
            ],
        )
        root.destroy()
        return path or None
    except Exception:
        return None


@click.command(name="pff-terrain3d", context_settings={"ignore_unknown_options": True})
@click.argument("path", required=False, type=click.Path(exists=True, dir_okay=False))
@click.option("--perlin", is_flag=True, help="Start with Perlin noise demo")
@click.option("--mesh", default=2048, show_default=True, type=int, help="Mesh max dim (longest side)")
def cli(path: Optional[str], perlin: bool, mesh: int) -> None:
    _init_taichi_safe()

    # Resolve data (keep raw array)
    arr_raw: Optional[np.ndarray] = None
    water_raw: Optional[np.ndarray] = None  # Water height data
    if perlin:
        import pyfastflow as pff
        arr = pff.noise.perlin_noise(512, 512, frequency=1.0, octaves=6, persistence=0.6, seed=42).astype(np.float32, copy=False)
        # normalize explicitly to [0,1] for demo raw
        vmin = float(np.min(arr))
        vmax = float(np.max(arr))
        arr_raw = (arr - vmin) / max(1e-6, (vmax - vmin))
    else:
        if not path:
            path = _file_dialog()
        if not path:
            raise SystemExit("No input provided.")
        arr_raw = _read_array(path)

    # Build app using visuGL
    from pyfastflow import visuGL

    app = visuGL.create_3D_app(title="PyFastFlow Terrain 3D")
    layer = visuGL.adapters.Heightfield3D(mesh_max_dim=int(mesh))
    app.scene.add(layer)

    # Prepare initial masked/normalized heightmap with safe slider bounds
    def _finite_range(a: Optional[np.ndarray]) -> tuple[float, float]:
        if a is None:
            return 0.0, 1.0
        arr = np.asarray(a)
        finite = np.isfinite(arr)
        if not np.any(finite):
            return 0.0, 1.0
        lo = float(np.min(arr[finite]))
        hi = float(np.max(arr[finite]))
        if not math.isfinite(lo) or not math.isfinite(hi):
            return 0.0, 1.0
        if hi <= lo:
            hi = lo + 1.0
        return lo, hi

    r_min, r_max = _finite_range(arr_raw)
    # Default masking: hide values < 0.1 (raw units)
    default_thresh = float(np.clip(0.1, min(r_min, r_max), max(r_min, r_max)))
    height_img = _make_height_from_raw(arr_raw, default_thresh) if arr_raw is not None else None

    def init(app_inst):
        app_inst.data.tex2d("height", height_img if height_img is not None else np.zeros((2, 2), dtype=np.float32), fmt="R32F")
        layer.use_hub("height")
        # Initialize empty water texture (will be populated when user loads water data)
        app_inst.data.tex2d("water", np.zeros_like(height_img if height_img is not None else np.zeros((2, 2), dtype=np.float32)), fmt="R32F")
        layer.use_water_hub("water")

    app.on_init(init)

    # UI
    p = app.ui.add_panel("Display", dock="right")
    z = p.slider("Z exaggeration", 0.5, 0.01, 5.0, input_field=True)
    z.subscribe(layer.set_height_scale)
    p.checkbox("Sphere mode", False).subscribe(layer.set_sphere_mode)
    # Reset camera to default
    def _reset_camera():
        cam = app.camera
        cam.yaw = 0.0
        cam.pitch = 30.0
        cam.distance = 5.0
        cam.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    p.button("Reset camera", _reset_camera)

    # Mesh resolution control
    mesh_ref = p.int_slider("Mesh max dim", int(mesh), 16, 8192, input_field=True)
    def _rebuild():
        layer.set_mesh_max_dim(int(mesh_ref.value))
    p.button("Rebuild mesh", _rebuild)

    # Masking controls (passive; requires clicking update to apply)
    # Slider goes from safe finite raw min to max
    tmin = float(r_min)
    tmax = float(r_max if r_max > r_min else (r_min + 1.0))
    thr_ref = p.slider("Mask below (raw)", float(default_thresh), tmin, tmax, input_field=True)

    def _apply_mask():
        if arr_raw is None:
            return
        # Recompute height with current water if present
        new_h = _make_height_from_raw(arr_raw, float(thr_ref.value), water=water_raw)
        app.data.update_tex("height", new_h)

    p.button("Update mask", _apply_mask)

    if perlin:
        # Perlin noise parameters
        fx = p.slider("Frequency", 1.0, 0.05, 8.0, input_field=True)
        oc = p.int_slider("Octaves", 6, 1, 12, input_field=True)
        pe = p.slider("Persistence", 0.6, 0.1, 0.99, input_field=True)
        sd = p.int_slider("Seed", 42, 0, 99999, input_field=True)
        # Dimensions controls (texture size)
        w_ref = p.int_slider("Noise width", int(arr_raw.shape[1]), 32, 8192, input_field=True)
        h_ref = p.int_slider("Noise height", int(arr_raw.shape[0]), 32, 8192, input_field=True)

        import pyfastflow as pff

        def _regen():
            nonlocal arr_raw
            # Regenerate raw demo noise, normalize to [0,1], then reapply current mask
            width = max(2, int(w_ref.value))
            height = max(2, int(h_ref.value))
            base_freq = float(fx.value)
            # Keep isotropy in world units: scale Y frequency by height/width
            fy = base_freq * (float(height) / float(width))
            new = pff.noise.perlin_noise(width, height, frequency=base_freq, frequency_x=base_freq, frequency_y=fy, octaves=int(oc.value), persistence=float(pe.value), seed=int(sd.value)).astype(np.float32, copy=False)
            vmin = float(np.min(new))
            vmax = float(np.max(new))
            new_raw = (new - vmin) / max(1e-6, (vmax - vmin))
            # Update arr_raw reference and reapply mask with water if present
            arr_raw = new_raw
            masked = _make_height_from_raw(arr_raw, float(thr_ref.value), water=water_raw)
            app.data.update_tex("height", masked)

        p.button("Regenerate Perlin", _regen)

        # Save to .npy dialog
        def _save_npy():
            if arr_raw is None:
                return
            try:
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk()
                root.withdraw()
                path = filedialog.asksaveasfilename(
                    title="Save Perlin array as .npy",
                    defaultextension=".npy",
                    filetypes=[("NumPy", "*.npy"), ("All files", "*.*")],
                )
                root.destroy()
                if not path:
                    return
                # Ensure .npy extension
                if not str(path).lower().endswith(".npy"):
                    path = str(path) + ".npy"
                np.save(path, arr_raw)
            except Exception:
                pass

        p.button("Save .npy", _save_npy)

    # ===== Water Visualization Section (collapsible within Display panel) =====
    p.begin_collapsing_header("Water", default_open=False)

    # Water enable/disable toggle (disabled by default)
    water_enabled_ref = p.checkbox("Enable water", False)

    def _on_water_enabled_changed(enabled):
        layer.set_water_enabled(enabled)
        # Recompute height texture with or without water
        if arr_raw is not None:
            if enabled and water_raw is not None:
                # Combine terrain + water
                combined_height = _make_height_from_raw(arr_raw, float(thr_ref.value), water=water_raw)
            else:
                # Terrain only
                combined_height = _make_height_from_raw(arr_raw, float(thr_ref.value), water=None)
            app.data.update_tex("height", combined_height)

    water_enabled_ref.subscribe(_on_water_enabled_changed)

    # Load water data button
    def _load_water():
        nonlocal water_raw
        water_path = _file_dialog()
        if not water_path:
            return
        try:
            # Load water data
            water_candidate = _read_array(water_path)

            # Validate dimensions match terrain
            if arr_raw is None:
                return
            if water_candidate.shape != arr_raw.shape:
                print(f"Error: Water dimensions {water_candidate.shape} don't match terrain {arr_raw.shape}")
                return

            # Store water data
            water_raw = water_candidate

            # Recompute height texture with terrain + water combined BEFORE normalization
            combined_height = _make_height_from_raw(arr_raw, float(thr_ref.value), water=water_raw)
            app.data.update_tex("height", combined_height)

            # Upload raw water separately for coloring
            app.data.update_tex("water", water_raw)

            # Enable water visualization by default after loading
            water_enabled_ref.value = True
            layer.set_water_enabled(True)

            # Update vmin/vmax sliders with reasonable defaults based on water data
            finite = np.isfinite(water_raw)
            if np.any(finite):
                w_min = float(np.min(water_raw[finite]))
                w_max = float(np.max(water_raw[finite]))
                if w_max > w_min:
                    water_vmin_ref.value = w_min
                    water_vmax_ref.value = w_max
                    layer.set_water_vmin(w_min)
                    layer.set_water_vmax(w_max)
                    # Set threshold to something small but visible
                    water_thresh_ref.value = max(w_min, 0.01)
                    layer.set_water_threshold(water_thresh_ref.value)

            print(f"Water data loaded: {water_raw.shape}, range [{w_min:.3f}, {w_max:.3f}]")
        except Exception as e:
            print(f"Failed to load water data: {e}")

    p.button("Load water file", _load_water)

    # Water threshold control
    water_thresh_ref = p.slider("Water threshold", 0.01, 0.0, 1.0, input_field=True)
    water_thresh_ref.subscribe(layer.set_water_threshold)

    # Color mapping range controls
    water_vmin_ref = p.slider("Water vmin", 0.0, 0.0, 10.0, input_field=True)
    water_vmin_ref.subscribe(layer.set_water_vmin)

    water_vmax_ref = p.slider("Water vmax", 1.0, 0.0, 10.0, input_field=True)
    water_vmax_ref.subscribe(layer.set_water_vmax)

    p.end_collapsing_header()

    app.run()


def main(argv: Optional[list[str]] = None) -> None:
    # Allow direct invocation by console_scripts
    try:
        cli(standalone_mode=False)  # click will parse sys.argv by default
    except SystemExit as e:
        # Propagate normal exit codes
        if e.code not in (0, None):
            raise


if __name__ == "__main__":
    main()
