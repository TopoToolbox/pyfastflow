"""
Diagnostic: does repeated solve_lm_zh() ever drive the z+h local-minima count
to zero on an unprocessed DEM (no CPU fill/breach)?

Loads greenriver as-is, calls flowctx.solve_lm_zh() in a loop, monitors
flowctx.monitor_lm_zh() every iteration, stops when it hits 0 or a hard cap.

Author: B.G.
"""

import numpy as np
import taichi as ti
import topotoolbox as ttb

import pyfastflow.constants as cte
from pyfastflow import tp
from pyfastflow.flow import FlowContext
from pyfastflow.grid import GridContext
import matplotlib.pyplot as plt

MODE = "zh"  # "z" -> solve_lm_z/monitor_lm_z, "zh" -> solve_lm_zh/monitor_lm_zh
MAX_ITER = 100_000
CHECK_EVERY = 1000

ti.init(arch=ti.gpu, offline_cache=False, debug = False)

dem = ttb.load_dem("greenriver")
NX, NY, DX = dem.columns, dem.rows, dem.cellsize
N = NX * NY

gridctx = GridContext(NX, NY, DX, boundary_mode="normal", topology="D4")
flowctx = FlowContext(
    gridctx,
    weight_mode="const",
    weight=1.0,
    min_slope_mode="const",
    min_slope=1e-2,
    diagonal_partition_correction=True,
)

z = tp.get_tpfield(cte.FLOAT_TYPE_TI, N)
h = tp.get_tpfield(cte.FLOAT_TYPE_TI, N)

z.field.from_numpy(dem.z.ravel().astype(np.float32))
np.random.seed(42)
h.field.from_numpy(np.random.rand(dem.z.ravel().shape[0]).astype(np.float32) * 10)
# h.field.fill(0.0)

if MODE == "z":
    solve = lambda: flowctx.solve_lm_z(z.field)
    monitor = lambda: flowctx.monitor_lm_z(z.field)
elif MODE == "zh":
    solve = lambda: flowctx.solve_lm_zh(z.field, h.field)
    monitor = lambda: flowctx.monitor_lm_zh(z.field, h.field)
else:
    raise ValueError("MODE must be 'z' or 'zh'")

n_lm0 = monitor()
print(f"mode={MODE}  initial local minima count: {n_lm0}")

it = 0
n_lm = n_lm0
while it < MAX_ITER:
    solve()
    it += 1

    if it % CHECK_EVERY == 0 or it == MAX_ITER:
        n_lm = monitor()
        print(f"iter {it}: lm count = {n_lm}", end="          \r")
        if n_lm == 0:
            break

print()
if n_lm == 0:
    print(f"all local minima resolved after {it} solve_lm_{MODE}() calls")
else:
    print(f"stopped at {it} calls, {n_lm} local minima still remaining")


plt.imshow((z.to_numpy() + h.to_numpy()).reshape(NY,NX))
plt.colorbar()
plt.show()

z.release()
h.release()
flowctx.destroy()
gridctx.destroy()
