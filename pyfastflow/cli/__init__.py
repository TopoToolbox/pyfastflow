"""
Command Line Interface for PyFastFlow

This module provides command line utilities for PyFastFlow, enabling
easy access to common operations from the terminal without writing Python scripts.

Available Commands:
- raster2npy: Convert raster files to numpy arrays
- raster-upscale: Double raster resolution using rastermanip utilities
- raster-downscale: Halve raster resolution using rastermanip utilities
- dem2png: Convert DEM files to PNG format
- precip-gui: Interactive precipitation map editor
- terrain3d: 3D terrain visualization using visuGL

Author: B.G.
"""

_CLI_SUBMODULES = {
    "raster2npy": (".raster_commands", "raster2npy"),
    "raster_upscale": (".rastermanip_commands", "raster_upscale"),
    "raster_downscale": (".rastermanip_commands", "raster_downscale"),
    "dem2png": (".dem2png_commands", "dem2png"),
    "boundary_gui": (".grid_commands", "boundary_gui"),
    "precipitation_gui": (".precip_commands", "precipitation_gui"),
    "terrain3d": (".terrain3d_cli", "main"),
}

__all__ = list(_CLI_SUBMODULES.keys())


def __getattr__(name):
    info = _CLI_SUBMODULES.get(name)
    if info is None:
        raise AttributeError(name)
    pkg, attr = info
    import importlib
    mod = importlib.import_module(pkg, __package__)
    obj = getattr(mod, attr)
    globals()[name] = obj
    return obj
