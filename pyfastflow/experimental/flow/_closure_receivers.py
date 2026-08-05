"""
Taichi/Quadrants (closure) block templates behind make_receivers:
distance/slope helpers, the stochastic rand_unit hash, and the receivers
kernel itself (mode x h_aware variants).

Split out of a single _closure_blocks.py that used to hold every flow
algorithm (receivers, accumulation, depressions, reconstruction fill) - see
_closure_accum.py/_closure_depressions.py/_closure_reconstruct.py for the
others. Same split as ../grid/_closure_blocks.py and
../noise/_closure_blocks.py: every private block is a plain python def,
PICKED - never branched on inside one function body - by build_receivers()
according to the caller's `mode` ("steepest"|"stochastic"), `h_aware` and
`diagonal_partition_correction` flags. The one runtime branch that exists
(which diagonal k values get the sqrt(2) correction) is inside the
*corrected* distance helper only - k is genuine per-call device data, so it
cannot be resolved by picking a function ahead of time; the *uncorrected*
variant, used whenever the correction is off, is a different helper with no
branch at all (grid.dist_from_k / grid.dist_between_nodes themselves, reused
as-is).

The receiver kernel body is one of four variants (mode x h_aware), each a
nested def inside build_receivers so it can close over the per-backend data
argument annotation (`ti.template()` vs `qd.Tensor`) the way
../ops/_closure_blocks.py's build_elementwise does - a kernel's data
arguments need a real type annotation chosen at build time, unlike a helper.

Author: B.G (07/2026)
"""

import functools
import math

from ..core.context.backends import bag_need, helper_need, make_helper, make_kernel, param_need
from ..core.context.need import Kind, Need
from ._closure_shared import _tensor_annotation


# ---------------------------------------------------------------------------
# distance/slope helpers
# ---------------------------------------------------------------------------


def _dist_from_k_corrected_tmpl(k):
    d = _GRID.dist_from_k(k)
    if k == 0 or k == 2 or k == 5 or k == 7:
        d = d / SQRT2
    return d


def _dist_between_nodes_corrected_tmpl(i, j):
    d = _GRID.dist_between_nodes(i, j)
    if d > _GRID.dx.get(0) * 1.1:
        d = d / SQRT2
    return d


def _slope_from_values_k_tmpl(zi, hi, zj, hj, k):
    # (zi-zj)+(hi-hj) rather than (zi+hi)-(zj+hj) - avoids float cancellation
    # when z dominates h in magnitude.
    return ((zi - zj) + (hi - hj)) / _DISTFROMK(k)


def _slope_between_nodes_tmpl(vi, vj, i, j):
    return (vi - vj) / _DISTBETWEEN(i, j)


# ---------------------------------------------------------------------------
# rand_unit(i, k): hash_u32 mixing node, neighbour direction and seed
# ---------------------------------------------------------------------------


def _rand_unit_tmpl(i, k):
    # node index and neighbour direction mixed separately, mirroring
    # noise's _white_unit_tmpl (col/row -> i/k) so every (node, k) candidate
    # draws its own value, the same way legacy calls ti.random() once per
    # candidate inside the k loop rather than once per node - a node-keyed
    # hash would scale every candidate by the same factor and weaken the
    # randomisation.
    key = _BK.u32(SEED.get(0))
    key ^= _BK.u32(i) * _BK.u32(374761393)
    key ^= _BK.u32(k) * _BK.u32(668265263)
    hashed = _HASH(key)
    return _BK.cast(hashed, _BK.f32) / 4294967296.0


def build_distance_slope_helpers(HelperCls, *, grid, diagonal_partition_correction):
    """
    dist_from_k_corrected/dist_between_nodes_corrected/slope_from_values_k/
    slope_between_nodes for a closure backend (Taichi or Quadrants).

    When `diagonal_partition_correction` is off, or the grid is not D8, the
    "corrected" distance helpers are simply the grid's own dist_from_k /
    dist_between_nodes HelperBuilders - no branch, no separate template.

    Every bind goes through a Need (param_need/helper_need/bag_need, see
    backends.py) and every HelperBuilder is constructed strict_needs=True -
    see grid/_closure_blocks.py's build_helpers for the reference conversion.
    `_GRID=grid` declares only the members each corrected-distance template
    actually reads (`dist_from_k_corrected` needs `dist_from_k`/`dx`;
    `dist_between_nodes_corrected` needs `dist_between_nodes`/`dx`), mirroring
    ops/_closure_blocks.py's build_slope.

    Returns {name: HelperBuilder}.

    Author: B.G (07/2026)
    """
    mk = functools.partial(make_helper, HelperCls, strict_needs=True)
    sqrt2 = math.sqrt(2.0)

    d8 = grid.n_neighbours.get() == 8
    if diagonal_partition_correction and d8:
        dist_from_k_contains = [
            Need("dist_from_k", kind=Kind.HELPER),
            Need("dx", kind=Kind.PARAM, dtype=grid.dx.dtype, modes={grid.dx.mode}),
        ]
        dist_between_contains = [
            Need("dist_between_nodes", kind=Kind.HELPER),
            Need("dx", kind=Kind.PARAM, dtype=grid.dx.dtype, modes={grid.dx.mode}),
        ]
        dist_from_k_corrected = mk(
            _dist_from_k_corrected_tmpl, _GRID=bag_need("_GRID", grid, contains=dist_from_k_contains), SQRT2=sqrt2
        )
        dist_between_nodes_corrected = mk(
            _dist_between_nodes_corrected_tmpl, _GRID=bag_need("_GRID", grid, contains=dist_between_contains), SQRT2=sqrt2
        )
    else:
        dist_from_k_corrected = grid.dist_from_k
        dist_between_nodes_corrected = grid.dist_between_nodes

    slope_from_values_k = mk(_slope_from_values_k_tmpl, _DISTFROMK=helper_need("_DISTFROMK", dist_from_k_corrected))
    slope_between_nodes = mk(_slope_between_nodes_tmpl, _DISTBETWEEN=helper_need("_DISTBETWEEN", dist_between_nodes_corrected))

    return {
        "dist_from_k_corrected": dist_from_k_corrected,
        "dist_between_nodes_corrected": dist_between_nodes_corrected,
        "slope_from_values_k": slope_from_values_k,
        "slope_between_nodes": slope_between_nodes,
    }


def build_rand_unit(HelperCls, *, seed_need: Need, hash_u32):
    """
    rand_unit(i, k) HelperBuilder, binding the caller-supplied `hash_u32`
    (noise's public hash helper - see ../noise/_closure_blocks.py) rather
    than a private copy.

    `seed_need` is the caller's already-bound `Need("seed_p", kind=Kind.PARAM)`
    (see make_receivers) - a fresh, internally-named `Need("SEED", ...)`,
    matching this template's own `SEED.get(0)` reference, is bound here to
    the same underlying Parameter and declared on the helper via `.need()`.
    `_HASH=hash_u32` goes through helper_need; `strict_needs=True` - see
    grid/_closure_blocks.py's build_helpers for the reference conversion.
    `_BK` needs no bind at all - auto-injected (see
    core/context/_closure_backend.py's module docstring).

    Author: B.G (07/2026)
    """
    seed_n = Need("SEED", kind=Kind.PARAM, dtype=seed_need.dtype, modes=seed_need.modes)
    seed_n.bind(seed_need.value)
    return (
        HelperCls(strict_needs=True)
        .need(seed_n)
        .need(helper_need("_HASH", hash_u32))
        .bind("_HASH", hash_u32)
        .ingest(_rand_unit_tmpl)
    )


def build_receivers(
    KernelCls,
    HelperCls,
    *,
    backend: str,
    backend_mod,
    grid,
    hash_u32,
    mode: str,
    seed_need: Need,
    diagonal_partition_correction: bool,
    h_aware: bool,
):
    """
    Build one closure-backend `receivers` KernelBuilder plus the distance/
    slope (and, for mode="stochastic", rand_unit) HelperBuilders it is made
    of, picking one of four kernel body variants (mode x h_aware) - never
    branching on either inside a single kernel body.

    `hash_u32` is the noise module's public hash_u32 HelperBuilder, reused
    here rather than re-implemented, so rand_unit and noise's own white_unit
    share the exact same integer hash. `seed_need` (see build_rand_unit) and
    `hash_u32` are only required when mode="stochastic".

    Returns {name: HelperBuilder/KernelBuilder} - the distance/slope helpers
    plus "receivers", plus "rand_unit" when mode="stochastic".

    Every bind on the `receivers` KernelBuilder goes through a Need
    (param_need/helper_need/bag_need, see backends.py), `strict_needs=True` -
    see grid/_closure_blocks.py's build_helpers for the reference conversion.
    `_GRID=grid` declares only `can_out`/`n_neighbours`/`neighbour`, the
    members the kernel body actually reads. `_BK` needs no bind - auto-
    injected (see core/context/_closure_backend.py's module docstring).

    Author: B.G (07/2026)
    """
    out = build_distance_slope_helpers(HelperCls, grid=grid, diagonal_partition_correction=diagonal_partition_correction)
    slope = out["slope_from_values_k"]
    T = _tensor_annotation(backend_mod, backend)

    if mode == "stochastic":
        rand_unit = build_rand_unit(HelperCls, seed_need=seed_need, hash_u32=hash_u32)
        out["rand_unit"] = rand_unit

    if mode == "steepest" and not h_aware:

        def receivers_template(z: T, rec: T):
            for i in z:
                if _GRID.can_out(i):
                    rec[i] = i
                    continue
                r = i
                sr = 0.0
                for k in range(_GRID.n_neighbours.get(0)):
                    j = _GRID.neighbour(i, k)
                    valid = j != -1
                    tsr = -1.0
                    if valid:
                        tsr = _SLOPE(z[i], 0.0, z[j], 0.0, k)
                    better = valid and tsr > sr
                    sr = tsr if better else sr
                    r = j if better else r
                rec[i] = r

    elif mode == "steepest" and h_aware:

        def receivers_template(z: T, h: T, rec: T):
            for i in z:
                if _GRID.can_out(i):
                    rec[i] = i
                    continue
                r = i
                sr = 0.0
                for k in range(_GRID.n_neighbours.get(0)):
                    j = _GRID.neighbour(i, k)
                    valid = j != -1
                    tsr = -1.0
                    if valid:
                        tsr = _SLOPE(z[i], h[i], z[j], h[j], k)
                    better = valid and tsr > sr
                    sr = tsr if better else sr
                    r = j if better else r
                rec[i] = r

    elif mode == "stochastic" and not h_aware:

        def receivers_template(z: T, rec: T):
            for i in z:
                if _GRID.can_out(i):
                    rec[i] = i
                    continue
                r = i
                sr = 0.0
                for k in range(_GRID.n_neighbours.get(0)):
                    j = _GRID.neighbour(i, k)
                    valid = j != -1
                    tsr = -1.0
                    if valid:
                        tsr = _SLOPE(z[i], 0.0, z[j], 0.0, k)
                        if tsr > 0.0:
                            tsr = _RAND(i, k) * _BK.sqrt(tsr)
                    better = valid and tsr > sr
                    sr = tsr if better else sr
                    r = j if better else r
                rec[i] = r

    else:  # mode == "stochastic" and h_aware

        def receivers_template(z: T, h: T, rec: T):
            for i in z:
                if _GRID.can_out(i):
                    rec[i] = i
                    continue
                r = i
                sr = 0.0
                for k in range(_GRID.n_neighbours.get(0)):
                    j = _GRID.neighbour(i, k)
                    valid = j != -1
                    tsr = -1.0
                    if valid:
                        tsr = _SLOPE(z[i], h[i], z[j], h[j], k)
                        if tsr > 0.0:
                            tsr = _RAND(i, k) * _BK.sqrt(tsr)
                    better = valid and tsr > sr
                    sr = tsr if better else sr
                    r = j if better else r
                rec[i] = r

    grid_contains = [
        Need("can_out", kind=Kind.HELPER),
        Need("n_neighbours", kind=Kind.PARAM, dtype=grid.n_neighbours.dtype, modes={grid.n_neighbours.mode}),
        Need("neighbour", kind=Kind.HELPER),
    ]
    kernel_binds = {
        "_GRID": bag_need("_GRID", grid, contains=grid_contains),
        "_SLOPE": helper_need("_SLOPE", slope),
    }
    if mode == "stochastic":
        kernel_binds["_RAND"] = helper_need("_RAND", out["rand_unit"])
    receivers_builder = make_kernel(KernelCls, receivers_template, strict_needs=True, **kernel_binds)

    out["receivers"] = receivers_builder
    return out


