"""
Tier 3: white-noise DEM -> reconstruction fill + flow correction -> MFD
drainage area via the persistent kernel, over every main grid combination.

cupy only: `persistent_mfd` has no closure-backend equivalent by design, and
the MFD topology / reconstruct-epsilon blocks it needs are cupy-only.

Per config (boundary normal/periodic_EW/periodic_NS x nodata off/on x outlet
edge/mask):
  1. white-noise DEM
  2. make_fill_reconstruct + solver -> filled / parent
  3. reconstruct-epsilon flow correction (hop-distance perturbation) ->
     dist
  4. build_mfd_topology(filled, dist) -> dirs / mfd_w / indegree
  5. make_accumulation(method="persistent_mfd"), source = 1.0

Checks:
  - accum finite and non-negative;
  - no stuck cell (every indegree drained to <= 0);
  - no interior MFD sink: every live cell with no outgoing direction is a
    can_out node;
  - mass balance across the outlets: accum summed over the can_out nodes ==
    number of live (non-nodata) cells, to a small relative tolerance (MFD
    weights are fractional f32).

Note: `q_init` seeds every cell including nodata (no nodata gate in it), so
this test zeros accum at nodata cells after q_init before accumulating.

Author: B.G (08/2026)
"""

import math

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("no cupy device", allow_module_level=True)
except Exception:
    pytest.skip("no cupy device", allow_module_level=True)

from pyfastflow.core.context.backends import backend_classes
from pyfastflow.flow._verify_depressions import make_noisy_terrain

DX = 1.0
SEED = 2024
SIDE = 128
BLOCK = 256
NN = 8

_CONFIGS = [
    (boundary, nodata, custom_outlet)
    for boundary in ("normal", "periodic_EW", "periodic_NS")
    for nodata in (False, True)
    for custom_outlet in (False, True)
]
_IDS = [f"{b}-{'nd' if nd else 'x'}-{'mask' if mo else 'edge'}" for b, nd, mo in _CONFIGS]


def _edge_can_out(boundary: str, nx: int, ny: int) -> np.ndarray:
    m = np.zeros((ny, nx), dtype=bool)
    if boundary in ("normal", "periodic_EW"):
        m[0, :] = True
        m[-1, :] = True
    if boundary in ("normal", "periodic_NS"):
        m[:, 0] = True
        m[:, -1] = True
    return m.ravel()


@pytest.mark.parametrize("boundary,nodata,custom_outlet", _CONFIGS, ids=_IDS)
def test_accum_mfd(boundary, nodata, custom_outlet):
    from pyfastflow.core.pool.cupy_pool import CupyPool
    from pyfastflow.flow import make_accumulation, make_fill_reconstruct, make_fill_reconstruct_solver
    from pyfastflow.flow._cupy_mfd_accum import init_frontier_mfd, persistent_grid_block
    from pyfastflow.graphflood import _cupy_mfd_topology, _cupy_reconstruct_epsilon
    from pyfastflow.grid import make_grid_group, make_grid_parameters

    bk = backend_classes("cupy")
    Param, dt = bk.ParameterCls, bk.dtypes
    i32, f32 = dt["i32"], dt["f32"]
    u8, u32 = np.dtype(np.uint8), np.dtype(np.uint32)
    nx = ny = SIDE
    n = nx * ny
    outlet_cfg = "mask" if custom_outlet else "edge"
    launch = {"grid": ((n + BLOCK - 1) // BLOCK,), "block": (BLOCK,)}

    pool = CupyPool()
    grid = make_grid_group("cupy", topology="D8", boundary=boundary, nodata=nodata, outlet=outlet_cfg)
    gp = make_grid_parameters("cupy", pool, nx, ny, DX, topology="D8", nodata=nodata, outlet=outlet_cfg)

    z_np = make_noisy_terrain(nx, ny, SEED).copy()
    nodata_np = np.zeros(n, dtype=np.uint8)
    if nodata:
        blob = np.zeros((ny, nx), dtype=np.uint8)
        blob[ny // 4 : ny // 2, nx // 4 : nx // 2] = 1
        nodata_np = blob.ravel()
        z_np[nodata_np == 1] = 9999.0
        gp["NODATA_MASK"].set(nodata_np)
    if custom_outlet:
        om = np.zeros((ny, nx), dtype=np.uint8)
        om[0, :] = 1
        outlet_np = om.ravel()
        gp["OUTLET_MASK"].set(outlet_np)
        can_out = outlet_np.astype(bool)
    else:
        can_out = _edge_can_out(boundary, nx, ny)

    z = pool.get_data(f32, (n,))
    z.from_numpy(z_np)
    filled = pool.get_data(f32, (n,))
    parent = pool.get_data(i32, (n,))
    frontier = pool.get_data(i32, (2 * n,))
    max_passes = 4 * max(nx, ny)
    counters = pool.get_data(i32, (max_passes + 2,))
    queued_gen = pool.get_data(i32, (n,))
    counters.from_numpy(np.zeros(max_passes + 2, dtype=np.int32))
    queued_gen.from_numpy(np.full(n, -1, dtype=np.int32))
    pass_p = Param("PASS", dtype=i32, mode="scalar", value=0, pool=pool)
    active_p = Param("ACTIVE", dtype=i32, mode="scalar", value=0, pool=pool)

    recon = make_fill_reconstruct("cupy", grid, nx=nx, ny=ny)
    solver = make_fill_reconstruct_solver(
        "cupy", recon, gp,
        z=z.data, filled=filled.data, parent=parent.data, frontier=frontier.data,
        counters=counters.data, queued_gen=queued_gen.data, pass_p=pass_p, active_p=active_p,
        n_flat=n, nx=nx, ny=ny, block_size=BLOCK, max_passes=max_passes,
    )
    solver()

    # --- reconstruct-epsilon flow correction: dist ------------------------
    dist = pool.get_data(f32, (n,))
    dist2 = pool.get_data(f32, (n,))
    anc = pool.get_data(i32, (n,))
    anc2 = pool.get_data(i32, (n,))

    hi = _cupy_reconstruct_epsilon.build_hops_init(n_flat=n).build()
    hi.bind("parent", parent.data)
    hi.bind("filled", filled.data)
    hi.bind("dist", dist.data)
    hi.bind("anc", anc.data)
    hops_init = hi.compile("cupy", **launch)

    hj_frozen = _cupy_reconstruct_epsilon.build_hops_jump(n_flat=n)
    hj_fwd = hj_frozen.build()
    hj_fwd.bind("dist_in", dist.data)
    hj_fwd.bind("anc_in", anc.data)
    hj_fwd.bind("dist_out", dist2.data)
    hj_fwd.bind("anc_out", anc2.data)
    hops_fwd = hj_fwd.compile("cupy", **launch)
    hj_bwd = hj_frozen.build()
    hj_bwd.bind("dist_in", dist2.data)
    hj_bwd.bind("anc_in", anc2.data)
    hj_bwd.bind("dist_out", dist.data)
    hj_bwd.bind("anc_out", anc.data)
    hops_bwd = hj_bwd.compile("cupy", **launch)

    hops_rounds = math.ceil(math.log2(max(2, n))) + 1
    if hops_rounds % 2:
        hops_rounds += 1

    hops_init()
    for _ in range(hops_rounds // 2):
        hops_fwd()
        hops_bwd()

    # --- MFD topology on filled, with dist as the flat tie-break ---------
    dirs = pool.get_data(u8, (n,))
    mfd_w = pool.get_data(f32, (n * NN,))
    indegree = pool.get_data(i32, (n,))

    topo = _cupy_mfd_topology.build_mfd_topology(
        grid=grid, n_flat=n, topology="D8", diagonal_partition_correction=True,
    )
    dw = topo["dirs_weights"].build()
    dw.bind("filled", filled.data)
    dw.bind("dist", dist.data)
    dw.bind("dirs", dirs.data)
    dw.bind("mfd_w", mfd_w.data)
    dw.bind_leaf(gp)
    dirs_weights = dw.compile("cupy", **launch)

    ir = topo["indegree_reset"].build()
    ir.bind("indegree", indegree.data)
    indegree_reset = ir.compile("cupy", **launch)

    ic = topo["indegree_count"].build()
    ic.bind("dirs", dirs.data)
    ic.bind("indegree", indegree.data)
    ic.bind_leaf(gp)
    indegree_count = ic.compile("cupy", **launch)

    indegree_reset()
    dirs_weights()
    indegree_count()

    # --- persistent MFD accumulation, source = 1.0 ----------------------
    accum_h = pool.get_data(f32, (n,))
    frontier0 = pool.get_data(i32, (n,))
    frontier1 = pool.get_data(i32, (n,))
    count = pool.get_data(i32, (2,))
    barrier = pool.get_data(u32, (1,))

    source_p = Param("SRC", dtype=f32, mode="const", value=1.0, pool=pool)
    accum = make_accumulation("cupy", grid, method="persistent_mfd", n_flat=n, n_neighbours=NN)

    qib = accum["q_init"].build()
    qib.bind("SOURCE", source_p)
    qib.bind("accum", accum_h.data)
    qib.compile("cupy", **launch)()

    # q_init has no nodata gate - zero the spurious source at nodata cells
    if nodata:
        a = accum_h.to_numpy()
        a[nodata_np == 1] = 0.0
        accum_h.from_numpy(a)

    n0 = init_frontier_mfd(indegree.data, frontier0.data)
    count.data[0] = n0
    count.data[1] = 0
    barrier.data[0] = 0

    ab = accum["accum"].build()
    ab.bind_leaf(gp, prefix=("grid",))
    ab.bind("frontier0", frontier0.data)
    ab.bind("frontier1", frontier1.data)
    ab.bind("count", count.data)
    ab.bind("barrier", barrier.data)
    ab.bind("dirs", dirs.data)
    ab.bind("mfd_w", mfd_w.data)
    ab.bind("accum", accum_h.data)
    ab.bind("indegree", indegree.data)
    pgrid, pblock = persistent_grid_block()
    ab.compile("cupy", grid=pgrid, block=pblock)()

    got = accum_h.to_numpy().astype(np.float64)
    dirs_np = dirs.to_numpy()
    live = nodata_np == 0

    assert np.all(np.isfinite(got)), "non-finite accum"
    assert got.min() >= -1e-3, f"negative accum (min {got.min():.3e})"

    n_stuck = int((indegree.to_numpy() > 0).sum())
    assert n_stuck == 0, f"{n_stuck} stuck cell(s) - frontier stalled"

    interior_sinks = int(np.count_nonzero((dirs_np == 0) & live & ~can_out))
    assert interior_sinks == 0, f"{interior_sinks} live sink(s) that are not outlets"

    n_live = int(np.count_nonzero(live))
    outlet_mass = float(got[can_out].sum())
    assert outlet_mass == pytest.approx(n_live, rel=1e-4), (
        f"mass balance: {outlet_mass:.3f} at outlets vs {n_live} live cells"
    )

    pool.clear_all(force=True)
