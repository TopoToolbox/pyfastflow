from __future__ import annotations

import os
from typing import Optional

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
    arr = np.asarray(arr)
    # Normalize to [0,1] float32 for now
    arr = arr.astype(np.float32, copy=False)
    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
        arr = (arr - vmin) / (vmax - vmin)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return arr


def _file_dialog() -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Select .npy or GeoTIFF",
            filetypes=[("NumPy", "*.npy"), ("GeoTIFF", "*.tif;*.tiff"), ("All", "*.*")],
        )
        root.destroy()
        return path or None
    except Exception:
        return None


@click.command(name="pff-terrain3d", context_settings={"ignore_unknown_options": True})
@click.argument("path", required=False, type=click.Path(exists=True, dir_okay=False))
@click.option("--demo", is_flag=True, help="Start with Perlin noise demo")
@click.option("--mesh", default=2048, show_default=True, type=int, help="Mesh max dim (longest side)")
def cli(path: Optional[str], demo: bool, mesh: int) -> None:
    _init_taichi_safe()

    # Resolve data
    arr: Optional[np.ndarray] = None
    if demo:
        import pyfastflow as pff
        arr = pff.noise.perlin_noise(512, 512, frequency=1.0, octaves=6, persistence=0.6, seed=42).astype(np.float32, copy=False)
        # normalize explicitly to [0,1]
        vmin = float(np.min(arr))
        vmax = float(np.max(arr))
        arr = (arr - vmin) / max(1e-6, (vmax - vmin))
    else:
        if not path:
            path = _file_dialog()
        if not path:
            raise SystemExit("No input provided.")
        arr = _read_array(path)

    # Build app using visuGL
    from pyfastflow import visuGL

    app = visuGL.create_3D_app(title="PyFastFlow Terrain 3D")
    layer = visuGL.adapters.Heightfield3D(mesh_max_dim=int(mesh))
    app.scene.add(layer)

    def init(app_inst):
        app_inst.data.tex2d("height", arr, fmt="R32F")
        layer.use_hub("height")

    app.on_init(init)

    # UI
    p = app.ui.add_panel("Display", dock="right")
    z = p.slider("Z exaggeration", 0.5, 0.01, 5.0)
    z.subscribe(layer.set_height_scale)
    p.checkbox("Sphere mode", False).subscribe(layer.set_sphere_mode)

    # Mesh resolution control
    mesh_ref = p.int_slider("Mesh max dim", int(mesh), 16, 8192)
    def _rebuild():
        layer.set_mesh_max_dim(int(mesh_ref.value))
    p.button("Rebuild mesh", _rebuild)

    if demo:
        # Perlin noise parameters
        fx = p.slider("Frequency", 1.0, 0.05, 8.0)
        oc = p.int_slider("Octaves", 6, 1, 12)
        pe = p.slider("Persistence", 0.6, 0.1, 0.99)
        sd = p.int_slider("Seed", 42, 0, 99999)

        import pyfastflow as pff

        def _regen():
            new = pff.noise.perlin_noise(arr.shape[1], arr.shape[0], frequency=float(fx.value), octaves=int(oc.value), persistence=float(pe.value), seed=int(sd.value)).astype(np.float32, copy=False)
            vmin = float(np.min(new))
            vmax = float(np.max(new))
            new = (new - vmin) / max(1e-6, (vmax - vmin))
            app.data.update_tex("height", new)

        p.button("Regenerate Perlin", _regen)

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
