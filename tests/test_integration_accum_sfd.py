"""
Tier 3: white-noise DEM -> carve depressions -> SFD accumulation, over every
main grid combination.

Per config (boundary normal/periodic_EW/periodic_NS x nodata off/on x outlet
edge/mask), per backend:
  1. white-noise DEM
  2. make_receivers -> make_depressions(method="optimized", reroute="carve")
     + make_depression_solver
  3. make_accumulation both "rake_compress" and "pointer_jump_push", source=1.0

Checks (source=1.0 -> every partial sum is an exact integer < 2**24, so f32
accumulation is exact regardless of summation order):
  - each method's q equals the numpy topological reference exactly;
  - the two methods equal each other exactly;
  - mass balance: q summed over the can_out roots == number of live
    (non-nodata) cells.

Author: B.G (08/2026)
"""

import importlib

import numpy as np
import pytest

from pyfastflow.core.context.backends import backend_classes
from pyfastflow.flow._verify_accum import numpy_topological_accum
from pyfastflow.flow._verify_depressions import make_noisy_terrain

DX = 1.0
SEED = 2024
SIDE = 128
BLOCK = 256

_BACKENDS = ("taichi", "quadrants", "cupy")

_CONFIGS = [
    (boundary, nodata, custom_outlet)
    for boundary in ("normal", "periodic_EW", "periodic_NS")
    for nodata in (False, True)
    for custom_outlet in (False, True)
]
_IDS = [f"{b}-{'nd' if nd else 'x'}-{'mask' if mo else 'edge'}" for b, nd, mo in _CONFIGS]


def _available(name: str) -> bool:
    try:
        mod = importlib.import_module(name)
    except Exception:
        return False
    if name == "cupy":
        try:
            return mod.cuda.runtime.getDeviceCount() > 0
        except Exception:
            return False
    return True


@pytest.fixture(params=_BACKENDS)
def backend(request):
    name = request.param
    if not _available(name):
        pytest.skip(f"{name} not available")
    if name == "taichi":
        import taichi as ti

        ti.init(arch=ti.gpu)
    elif name == "quadrants":
        import quadrants as qd

        qd.init(arch=qd.gpu)
    return name


def _pool_cls(name):
    if name == "taichi":
        from pyfastflow.core.pool.taichi_pool import TaichiPool as P
    elif name == "quadrants":
        from pyfastflow.core.pool.quadrants_pool import QuadrantsPool as P
    else:
        from pyfastflow.core.pool.cupy_pool import CupyPool as P
    return P


def _edge_can_out(boundary: str, nx: int, ny: int) -> np.ndarray:
    m = np.zeros((ny, nx), dtype=bool)
    if boundary in ("normal", "periodic_EW"):
        m[0, :] = True
        m[-1, :] = True
    if boundary in ("normal", "periodic_NS"):
        m[:, 0] = True
        m[:, -1] = True
    return m.ravel()


def _bind_pjp(bound, closure, *, source_p, q, work, work2, q_work, rec):
    bound.bind(("q_init", "SOURCE"), source_p)
    bound.bind(("q_init", "q"), q.data)
    bound.bind(("copy_rec_to_work", "rec"), rec.data)
    bound.bind(("copy_rec_to_work", "work"), work.data)
    if closure:
        bound.bind_leaf(
            {"rec_curr": work.data, "rec_next": work2.data, "q_curr": q.data, "q_next": q_work.data},
            prefix=("step_a",), strict=True,
        )
        bound.bind_leaf(
            {"rec_curr": work2.data, "rec_next": work.data, "q_curr": q_work.data, "q_next": q.data},
            prefix=("step_b",), strict=True,
        )
    else:
        bound.bind_leaf({"q_curr": q.data, "q_next": q_work.data}, prefix=("step_a_copy",), strict=True)
        bound.bind_leaf(
            {"rec_curr": work.data, "rec_next": work2.data, "q_curr": q.data, "q_next": q_work.data},
            prefix=("step_a_core",), strict=True,
        )
        bound.bind_leaf({"q_curr": q_work.data, "q_next": q.data}, prefix=("step_b_copy",), strict=True)
        bound.bind_leaf(
            {"rec_curr": work2.data, "rec_next": work.data, "q_curr": q_work.data, "q_next": q.data},
            prefix=("step_b_core",), strict=True,
        )


@pytest.mark.parametrize("boundary,nodata,custom_outlet", _CONFIGS, ids=_IDS)
def test_accum_sfd(backend, boundary, nodata, custom_outlet):
    from pyfastflow.flow import make_accumulation, make_depression_solver, make_depressions, make_receivers
    from pyfastflow.grid import make_grid_group, make_grid_parameters

    bk = backend_classes(backend)
    Param, dt = bk.ParameterCls, bk.dtypes
    i32, i64, f32, u8 = dt["i32"], dt["i64"], dt["f32"], dt["u8"]
    closure = backend in ("taichi", "quadrants")
    nx = ny = SIDE
    n = nx * ny
    nn = 8
    outlet_cfg = "mask" if custom_outlet else "edge"
    launch = {} if closure else {"grid": ((n + BLOCK - 1) // BLOCK,), "block": (BLOCK,)}

    pool = _pool_cls(backend)()
    grid = make_grid_group(backend, topology="D8", boundary=boundary, nodata=nodata, outlet=outlet_cfg)
    gp = make_grid_parameters(backend, pool, nx, ny, DX, topology="D8", nodata=nodata, outlet=outlet_cfg)

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
    rec = pool.get_data(i32, (n,))

    recv = make_receivers(backend, grid, topology="D8", mode="steepest")
    rb = recv["receivers"].build()
    rb.bind_leaf(gp)
    rb.bind("z", z.data)
    rb.bind("rec", rec.data)
    rb.compile(backend)(**launch)
    rec0 = rec.to_numpy().astype(np.int32)

    # carve, optimized
    ndep_p = Param("NDEP", dtype=i32, mode="scalar", value=0, pool=pool)
    carve_bufs = dict(
        rec=rec.data, z=z.data,
        bid=pool.get_data(i32, (n,)).data,
        rec_jump=pool.get_data(i32, (n,)).data,
        z_prime=pool.get_data(f32, (n,)).data,
        is_border=pool.get_data(u8, (n,)).data,
        basin_saddle=pool.get_data(i64, (n,)).data,
        basin_saddlenode=pool.get_data(i32, (n,)).data,
        outlet=pool.get_data(i64, (n,)).data,
        rerouted=pool.get_data(u8, (n,)).data,
        tag=pool.get_data(u8, (n,)).data,
        tag_alt=pool.get_data(u8, (n,)).data,
        rec_scratch=pool.get_data(i32, (n,)).data,
        basin_route=pool.get_data(i32, (n,)).data,
        b_rcv=pool.get_data(i32, (n,)).data,
    )
    rec.from_numpy(rec0)
    deps = make_depressions(backend, grid, ndep_p, method="optimized", reroute="carve", n_flat=n)
    solver = make_depression_solver(
        backend, deps, gp, method="optimized", reroute="carve",
        n_flat=n, block_size=BLOCK, **carve_bufs,
    )
    solver()
    assert int(ndep_p.read()) == 0, "carve/optimized left unresolved pits"
    rec_np = rec.to_numpy().astype(np.int64)

    # numpy reference over the exact resolved graph, source = 1.0
    src_ones = np.ones(n, dtype=np.float32)
    ref = numpy_topological_accum(rec_np, src_ones)

    source_p = Param("SOURCE", dtype=f32, mode="const", value=1.0, pool=pool)

    # rake_compress
    iter_p = Param("ITER", dtype=i32, mode="scalar", value=0, pool=pool)
    acc_rc = make_accumulation(backend, grid, method="rake_compress", n_flat=n, n_neighbours=nn)
    b_rc = acc_rc.sequence.freeze().build()
    q_rc = pool.get_data(f32, (n,))
    donors = pool.get_data(i32, (n * nn,))
    ndonors = pool.get_data(i32, (n,))
    donors_alt = pool.get_data(i32, (n * nn,))
    ndonors_alt = pool.get_data(i32, (n,))
    q_alt = pool.get_data(f32, (n,))
    src = pool.get_data(i32, (n,))
    b_rc.bind_leaf({
        "rec": rec.data, "q": q_rc.data, "donors": donors.data, "ndonors": ndonors.data,
        "donors_alt": donors_alt.data, "ndonors_alt": ndonors_alt.data,
        "q_alt": q_alt.data, "src": src.data,
    })
    b_rc.bind_leaf({"SOURCE": source_p, "ITER": iter_p})
    b_rc.compile(backend, **launch)()
    q_rake = q_rc.to_numpy().astype(np.float64)

    # pointer_jump_push
    acc_pjp = make_accumulation(backend, grid, method="pointer_jump_push", n_flat=n)
    b_pjp = acc_pjp.sequence.freeze().build()
    q_pjp = pool.get_data(f32, (n,))
    work = pool.get_data(i32, (n,))
    work2 = pool.get_data(i32, (n,))
    q_work = pool.get_data(f32, (n,))
    _bind_pjp(b_pjp, closure, source_p=source_p, q=q_pjp, work=work, work2=work2, q_work=q_work, rec=rec)
    b_pjp.compile(backend, **launch)()
    q_jump = q_pjp.to_numpy().astype(np.float64)

    assert np.array_equal(q_rake, ref), f"rake_compress != reference (max |d| {np.abs(q_rake - ref).max()})"
    assert np.array_equal(q_jump, ref), f"pointer_jump_push != reference (max |d| {np.abs(q_jump - ref).max()})"
    assert np.array_equal(q_rake, q_jump), "rake_compress != pointer_jump_push"

    n_live = int(np.count_nonzero(nodata_np == 0))
    roots = rec_np == np.arange(n)
    outlet_mass = float(q_rake[roots & can_out].sum())
    assert outlet_mass == float(n_live), f"mass balance: {outlet_mass} at outlets vs {n_live} live cells"

    pool.clear_all(force=True)
