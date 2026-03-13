"""Example showing how to paint a precipitation map and use it."""

import numpy as np
import taichi as ti

import pyfastflow as pf
from pyfastflow.cli.precip_commands import precipitation_gui

# Initialize Taichi (CPU backend for portability)
ti.init(ti.cpu)

# Paths to input/output files
DEM_PATH = "dem.npy"
BOUND_PATH = "boundaries.npy"
PRECIP_PATH = "precip.npy"

# Launch interactive editor (blocks until window is closed)
precipitation_gui.main([DEM_PATH, PRECIP_PATH, "--boundary", BOUND_PATH])

# Load results and plug into Flooder
dem = np.load(DEM_PATH).astype(np.float32)
precip = np.load(PRECIP_PATH).astype(np.float32)
ny, nx = dem.shape

grid = pf.flow.GridField(nx, ny, pf.constants.DX)
grid.set_z(dem)

flooder = pf.flood.Flooder(grid, precipitation_rates=precip)
# Flooder can now be used for hydrodynamic simulations
