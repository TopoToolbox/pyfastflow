"""Cordonnier carve -> rank-gated topology -> persistent MFD, CuPy only.

The full D8 grid matrix mirrors the established depression suite: three
boundary modes, nodata off/on, and edge/masked outlets. D4 topology building
is supported, but is not included here because the existing Cordonnier D4
solver does not yet resolve every masked-outlet corner; that is upstream of
the rank gate.
"""

from collections import deque

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("no cupy device", allow_module_level=True)
except Exception:
    pytest.skip("no cupy device", allow_module_level=True)

from pyfastflow.core import Backend
from pyfastflow.flow._verify_depressions import make_noisy_terrain


SIDE = 64
SEED = 2024
DX = 1.0

_CONFIGS = [
    ("D8", boundary, nodata, custom_outlet)
    for boundary in ("normal", "periodic_EW", "periodic_NS")
    for nodata in (False, True)
    for custom_outlet in (False, True)
]
_IDS = [
    f"{t}-{b}-{'nd' if nd else 'x'}-{'mask' if mask else 'edge'}"
    for t, b, nd, mask in _CONFIGS
]


def _edge_can_out(boundary, nx, ny):
    out = np.zeros((ny, nx), dtype=bool)
    if boundary in ("normal", "periodic_EW"):
        out[0, :] = out[-1, :] = True
    if boundary in ("normal", "periodic_NS"):
        out[:, 0] = out[:, -1] = True
    return out.ravel()


def _neighbour(i, k, *, topology, boundary, nx, ny):
    moves = (
        ((-1, 0), (0, -1), (0, 1), (1, 0))
        if topology == "D4"
        else ((-1, -1), (-1, 0), (-1, 1), (0, -1),
              (0, 1), (1, -1), (1, 0), (1, 1))
    )
    row, col = divmod(i, nx)
    row += moves[k][0]
    col += moves[k][1]
    if boundary == "periodic_EW":
        col %= nx
    if boundary == "periodic_NS":
        row %= ny
    if row < 0 or row >= ny or col < 0 or col >= nx:
        return -1
    return row * nx + col


def _reference_accum(dirs, weights, live, *, topology, boundary, nx, ny):
    n = nx * ny
    nn = 4 if topology == "D4" else 8
    indegree = np.zeros(n, dtype=np.int32)
    outgoing = [[] for _ in range(n)]
    for i, mask in enumerate(dirs):
        for k in range(nn):
            if int(mask) & (1 << k):
                j = _neighbour(i, k, topology=topology, boundary=boundary, nx=nx, ny=ny)
                assert j >= 0
                outgoing[i].append((j, float(weights[i, k])))
                indegree[j] += 1
    queue = deque(np.flatnonzero(indegree == 0).tolist())
    accum = live.astype(np.float64)
    seen = 0
    while queue:
        i = queue.popleft()
        seen += 1
        for j, weight in outgoing[i]:
            accum[j] += accum[i] * weight
            indegree[j] -= 1
            if indegree[j] == 0:
                queue.append(j)
    assert seen == n, f"CPU reference found a cycle ({seen}/{n} nodes processed)"
    return accum, outgoing


def _compile(frozen, backend, bindings, grid_params=None, grid_prefix=()):
    bound = frozen.build()
    bound.bind_leaf(bindings)
    if grid_params is not None:
        bound.bind_leaf(grid_params, prefix=grid_prefix)
    compiled = bound.compile(backend)
    bound.close()
    return compiled


@pytest.mark.parametrize("topology,boundary,nodata,custom_outlet", _CONFIGS, ids=_IDS)
def test_rank_gated_cordonnier_mfd(topology, boundary, nodata, custom_outlet):
    from pyfastflow.flow import (
        bind_depression_solver,
        bind_mfd_receiver_rank,
        make_accumulation,
        make_depression_solver,
        make_depressions,
        make_mfd_topology,
        make_receivers,
    )
    from pyfastflow.flow._cupy_mfd_accum import init_frontier_mfd
    from pyfastflow.grid import make_grid_group, make_grid_parameters

    backend = Backend.from_name("cupy")
    pool = backend.pool()
    dt = backend.dtypes
    nx = ny = SIDE
    n = nx * ny
    nn = 4 if topology == "D4" else 8
    outlet_kind = "mask" if custom_outlet else "edge"
    grid = make_grid_group(
        backend, topology=topology, boundary=boundary,
        nodata=nodata, outlet=outlet_kind,
    )
    gp = make_grid_parameters(
        backend, pool, nx, ny, DX, topology=topology,
        nodata=nodata, outlet=outlet_kind,
    )

    z_np = make_noisy_terrain(nx, ny, SEED).copy()
    nodata_np = np.zeros(n, dtype=np.uint8)
    if nodata:
        blob = np.zeros((ny, nx), dtype=np.uint8)
        blob[ny // 4:ny // 2, nx // 4:nx // 2] = 1
        nodata_np = blob.ravel()
        z_np[nodata_np != 0] = 9999.0
        gp["NODATA_MASK"].set(nodata_np)
    if custom_outlet:
        outlet_np = np.zeros((ny, nx), dtype=np.uint8)
        outlet_np[0, :] = 1
        outlet_np = outlet_np.ravel()
        gp["OUTLET_MASK"].set(outlet_np)
        can_out = outlet_np.astype(bool)
    else:
        can_out = _edge_can_out(boundary, nx, ny)
    live = nodata_np == 0

    def data(dtype, shape):
        return pool.get_data(dt[dtype], shape)

    z, rec, rec_initial = data("f32", (n,)), data("i32", (n,)), data("i32", (n,))
    z.from_numpy(z_np)
    compiled = []

    receivers = make_receivers(backend, grid, topology=topology)["receivers"]
    route = _compile(receivers, backend, {"z": z, "rec": rec}, gp)
    compiled.append(route)

    topology_parts = make_mfd_topology(
        backend, grid, method="cordonnier_rank", n_flat=n,
        topology=topology, diagonal_partition_correction=True,
    )
    snapshot = _compile(
        topology_parts["snapshot_receivers"], backend,
        {"rec": rec, "rec_initial": rec_initial},
    )
    compiled.append(snapshot)

    ndep = backend.ParameterCls("NDEP", dtype="i32", mode="scalar", value=0, pool=pool)
    carve_buffers = dict(
        rec=rec, z=z,
        bid=data("i32", (n,)), rec_jump=data("i32", (n,)),
        z_prime=data("f32", (n,)), is_border=data("u8", (n,)),
        basin_saddle=data("i64", (n,)), basin_saddlenode=data("i32", (n,)),
        outlet=data("i64", (n,)), basin_route=data("i32", (n,)),
        b_rcv=data("i32", (n,)),
    )
    deps = make_depressions(
        backend, grid, ndep, method="optimized", reroute="carve", n_flat=n,
    )
    carve_frozen, _ = make_depression_solver(
        backend, deps, gp, method="optimized", reroute="carve", n_flat=n,
    )
    carve_bound = bind_depression_solver(
        carve_frozen, gp, ndep_p=ndep, method="optimized", reroute="carve",
        **carve_buffers,
    )
    carve = carve_bound.compile(backend)
    carve_bound.close()
    compiled.append(carve)

    ancestor, ancestor_alt = data("i32", (n,)), data("i32", (n,))
    rank, rank_alt = data("i32", (n,)), data("i32", (n,))
    rank_bound = bind_mfd_receiver_rank(
        topology_parts["receiver_rank"], rec=rec, ancestor=ancestor,
        ancestor_alt=ancestor_alt, rank=rank, rank_alt=rank_alt,
    )
    rank_sequence = rank_bound.compile(backend)
    rank_bound.close()
    compiled.append(rank_sequence)

    dirs, weights = data("u8", (n,)), data("f32", (n * nn,))
    dirs_weights = _compile(
        topology_parts["dirs_weights"], backend,
        {"z": z, "rec_initial": rec_initial, "rec": rec,
         "rank": rank, "dirs": dirs, "mfd_w": weights},
        gp,
    )
    compiled.append(dirs_weights)
    indegree = data("i32", (n,))
    indegree_reset = _compile(
        topology_parts["indegree_reset"], backend, {"indegree": indegree},
    )
    indegree_count = _compile(
        topology_parts["indegree_count"], backend,
        {"dirs": dirs, "indegree": indegree}, gp,
    )
    compiled.extend((indegree_reset, indegree_count))

    accum_parts = make_accumulation(
        backend, grid, method="persistent_mfd", n_flat=n, n_neighbours=nn,
    )
    accumulation = data("f32", (n,))
    source = backend.ParameterCls("SOURCE", dtype="f32", mode="const", value=1.0, pool=pool)
    q_init = _compile(
        accum_parts["q_init"], backend,
        {"SOURCE": source, "accum": accumulation}, gp, ("grid",),
    )
    frontier0, frontier1 = data("i32", (n,)), data("i32", (n,))
    count, barrier = data("i32", (3,)), data("u32", (1,))
    persistent = _compile(
        accum_parts["accum"], backend,
        {"frontier0": frontier0, "frontier1": frontier1, "count": count,
         "barrier": barrier, "dirs": dirs, "mfd_w": weights,
         "accum": accumulation, "indegree": indegree},
        gp, ("grid",),
    )
    compiled.extend((q_init, persistent))

    route()
    snapshot()
    carve()
    rank_sequence()
    dirs_weights()
    indegree_reset()
    indegree_count()

    rec0_np = rec_initial.to_numpy().astype(np.int64)
    rec_np = rec.to_numpy().astype(np.int64)
    rank_np = rank.to_numpy().astype(np.int64)
    dirs_np = dirs.to_numpy()
    weights_np = weights.to_numpy().reshape(n, nn)

    linked = rec_np != np.arange(n)
    assert np.all(rank_np[rec_np[linked]] == rank_np[linked] - 1)
    reference, outgoing = _reference_accum(
        dirs_np, weights_np, live,
        topology=topology, boundary=boundary, nx=nx, ny=ny,
    )
    for i, edges in enumerate(outgoing):
        assert all(rank_np[j] < rank_np[i] for j, _ in edges)
        if live[i] and not can_out[i]:
            assert edges, f"live interior sink at {i}"
            assert sum(weight for _, weight in edges) == pytest.approx(1.0, abs=2e-6)
        if rec0_np[i] != rec_np[i] and live[i] and not can_out[i]:
            assert len(edges) == 1
            assert edges[0][0] == rec_np[i]
            assert edges[0][1] == pytest.approx(1.0)

    q_init()
    n_ready = init_frontier_mfd(indegree.array, frontier0.array)
    count.array[0], count.array[1], barrier.array[0] = n_ready, 0, 0
    persistent()
    got = accumulation.to_numpy().astype(np.float64)
    assert not np.any(indegree.to_numpy() > 0)
    np.testing.assert_allclose(got, reference, rtol=3e-4, atol=3e-3)
    assert got[can_out].sum() == pytest.approx(int(live.sum()), rel=2e-4)

    for item in compiled:
        item.close()
    pool.clear_all(force=True)
