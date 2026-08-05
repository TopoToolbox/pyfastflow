"""
cupy (CUDA source) block templates behind make_receivers: distance/slope
helpers, the stochastic rand_unit hash, and the receivers kernel itself
(mode x h_aware variants).

Split out of a single _cupy_blocks.py that used to hold every flow algorithm
(receivers, accumulation, depressions, reconstruction fill) - see
_cupy_accum.py/_cupy_depressions.py/_cupy_reconstruct.py for the others.
Written as CUDA text through the `$...$` span mechanism (see
../core/context/cupy_backend.py's module docstring). Every
`__device__`/`__global__` symbol is prefixed with this build's own tag (a
fresh new_uid()) so two make_receivers() calls in one process never collide
inside a single compiled cupy module. Mirrors _closure_receivers.py block for
block: same private/public split, same `mode`/`h_aware`/
`diagonal_partition_correction` selectors picking which CUDA text gets built.

Author: B.G (07/2026)
"""

import functools

from ..core.context.backends import bag_need, helper_need, make_helper, make_kernel
from ..core.context.need import Kind, Need
from ..core.pool.base import new_uid


def build_distance_slope_helpers(HelperCls, *, grid, diagonal_partition_correction):
    """
    dist_from_k_corrected/dist_between_nodes_corrected/slope_from_values_k/
    slope_between_nodes for the cupy backend.

    When `diagonal_partition_correction` is off, or the grid is not D8, the
    "corrected" distance helpers are simply the grid's own dist_from_k /
    dist_between_nodes HelperBuilders - no branch, no separate template.

    Every bind goes through a Need (helper_need/bag_need, see backends.py)
    and every HelperBuilder is constructed strict_needs=True - see
    _closure_receivers.py's build_distance_slope_helpers for the reference
    conversion. `grid=grid` declares only the members each template actually
    reads.

    Returns {name: HelperBuilder}.

    Author: B.G (07/2026)
    """
    t = f"pr{new_uid()}"
    mk = functools.partial(make_helper, HelperCls, strict_needs=True)

    d8 = grid.n_neighbours.get() == 8
    if diagonal_partition_correction and d8:
        dist_from_k_contains = [
            Need("dist_from_k", kind=Kind.HELPER),
        ]
        dist_between_contains = [
            Need("dist_between_nodes", kind=Kind.HELPER),
            Need("dx", kind=Kind.PARAM, dtype=grid.dx.dtype, modes={grid.dx.mode}),
        ]
        dist_from_k_corrected = mk(
            f"""
__device__ float {t}_dist_from_k_corrected(int k) {{
    float d = $grid.dist_from_k(k)$;
    if (k == 0 || k == 2 || k == 5 || k == 7) {{
        d = d / 1.4142135623730951f;
    }}
    return d;
}}
""",
            grid=bag_need("grid", grid, contains=dist_from_k_contains),
        )
        dist_between_nodes_corrected = mk(
            f"""
__device__ float {t}_dist_between_nodes_corrected(int i, int j) {{
    float d = $grid.dist_between_nodes(i, j)$;
    if (d > $grid.dx.get(0)$ * 1.1f) {{
        d = d / 1.4142135623730951f;
    }}
    return d;
}}
""",
            grid=bag_need("grid", grid, contains=dist_between_contains),
        )
    else:
        dist_from_k_corrected = grid.dist_from_k
        dist_between_nodes_corrected = grid.dist_between_nodes

    slope_from_values_k = mk(
        f"""
__device__ float {t}_slope_from_values_k(float zi, float hi, float zj, float hj, int k) {{
    return ((zi - zj) + (hi - hj)) / $dist_from_k_corrected(k)$;
}}
""",
        dist_from_k_corrected=helper_need("dist_from_k_corrected", dist_from_k_corrected),
    )
    slope_between_nodes = mk(
        f"""
__device__ float {t}_slope_between_nodes(float vi, float vj, int i, int j) {{
    return (vi - vj) / $dist_between_nodes_corrected(i, j)$;
}}
""",
        dist_between_nodes_corrected=helper_need("dist_between_nodes_corrected", dist_between_nodes_corrected),
    )

    return {
        "dist_from_k_corrected": dist_from_k_corrected,
        "dist_between_nodes_corrected": dist_between_nodes_corrected,
        "slope_from_values_k": slope_from_values_k,
        "slope_between_nodes": slope_between_nodes,
    }


def build_rand_unit(HelperCls, *, seed_need: Need, hash_u32):
    """
    rand_unit(i, k) HelperBuilder, binding the caller-supplied `hash_u32`
    (noise's public hash_u32 HelperBuilder - see ../noise/_cupy_blocks.py)
    rather than a private copy. Node index and neighbour direction are mixed
    separately (mirroring noise's white_unit col/row mixing), so every
    (node, k) candidate draws its own value.

    `seed_need` is the caller's already-bound `Need("seed_p", kind=Kind.PARAM)`
    (see make_receivers) - a fresh, internally-named `Need("SEED", ...)`,
    matching this template's own `$SEED.get(0)$` span, is bound here to the
    same underlying Parameter and declared on the helper via `.need()`.
    `hash_u32` goes through helper_need; `strict_needs=True` - see
    _closure_receivers.py's build_rand_unit for the reference conversion.

    Author: B.G (07/2026)
    """
    seed_n = Need("SEED", kind=Kind.PARAM, dtype=seed_need.dtype, modes=seed_need.modes)
    seed_n.bind(seed_need.value)
    t = f"pr{new_uid()}"
    return (
        HelperCls(strict_needs=True)
        .need(seed_n)
        .need(helper_need("hash_u32", hash_u32))
        .bind("hash_u32", hash_u32)
        .ingest(
            f"""
__device__ float {t}_rand_unit(int i, int k) {{
    unsigned int key = (unsigned int)$SEED.get(0)$;
    key ^= (unsigned int)i * 374761393u;
    key ^= (unsigned int)k * 668265263u;
    unsigned int hashed = $hash_u32(key)$;
    return (float)hashed / 4294967296.0f;
}}
"""
        )
    )


def build_receivers(
    KernelCls,
    HelperCls,
    *,
    grid,
    hash_u32,
    mode: str,
    seed_need: Need,
    diagonal_partition_correction: bool,
    h_aware: bool,
):
    """
    Build one cupy `receivers` KernelBuilder plus the distance/slope (and,
    for mode="stochastic", rand_unit) HelperBuilders it is made of, picking
    one of four kernel body text variants (mode x h_aware) - never a runtime
    branch on either inside the generated kernel.

    `hash_u32` is the noise module's public hash_u32 HelperBuilder, reused
    here rather than re-implemented. `seed_need` (see build_rand_unit) and
    `hash_u32` are only required when mode="stochastic".

    Returns {name: HelperBuilder/KernelBuilder} - the distance/slope helpers
    plus "receivers", plus "rand_unit" when mode="stochastic".

    Every bind on the `receivers` KernelBuilder goes through a Need
    (helper_need/bag_need, see backends.py), strict_needs=True - see
    _closure_receivers.py's build_receivers for the reference conversion.
    `grid=grid` declares only nx/ny/can_out/n_neighbours/neighbour, the
    members the kernel body actually reads.

    Author: B.G (07/2026)
    """
    out = build_distance_slope_helpers(HelperCls, grid=grid, diagonal_partition_correction=diagonal_partition_correction)
    slope = out["slope_from_values_k"]

    receivers_contains = [
        Need("nx", kind=Kind.PARAM, dtype=grid.nx.dtype, modes={grid.nx.mode}),
        Need("ny", kind=Kind.PARAM, dtype=grid.ny.dtype, modes={grid.ny.mode}),
        Need("can_out", kind=Kind.HELPER),
        Need("n_neighbours", kind=Kind.PARAM, dtype=grid.n_neighbours.dtype, modes={grid.n_neighbours.mode}),
        Need("neighbour", kind=Kind.HELPER),
    ]
    binds = {
        "grid": bag_need("grid", grid, contains=receivers_contains),
        "slope_from_values_k": helper_need("slope_from_values_k", slope),
    }
    if mode == "stochastic":
        rand_unit = build_rand_unit(HelperCls, seed_need=seed_need, hash_u32=hash_u32)
        out["rand_unit"] = rand_unit
        binds["rand_unit"] = helper_need("rand_unit", rand_unit)
        stochastic_insert = """
                        if (tsr > 0.0f) {
                            tsr = $rand_unit(i, k)$ * sqrtf(tsr);
                        }"""
    else:
        stochastic_insert = ""

    t = f"pr{new_uid()}"
    if h_aware:
        args = "const float* z, const float* h, int* rec"
        slope_call = "$slope_from_values_k(z[i], h[i], z[j], h[j], k)$"
    else:
        args = "const float* z, int* rec"
        slope_call = "$slope_from_values_k(z[i], 0.0f, z[j], 0.0f, k)$"

    body = f"""
__global__ void {t}_receivers({args}) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = $grid.nx.get(0)$ * $grid.ny.get(0)$;
    if (i >= n) return;

    if ($grid.can_out(i)$) {{
        rec[i] = i;
        return;
    }}

    int r = i;
    float sr = 0.0f;
    int nk = $grid.n_neighbours.get(0)$;
    for (int k = 0; k < nk; k++) {{
        int j = $grid.neighbour(i, k)$;
        int valid = j != -1;
        float tsr = -1.0f;
        if (valid) {{
            tsr = {slope_call};{stochastic_insert}
        }}
        int better = valid && (tsr > sr);
        sr = better ? tsr : sr;
        r = better ? j : r;
    }}
    rec[i] = r;
}}
"""
    receivers_builder = make_kernel(KernelCls, body, strict_needs=True, **binds)

    out["receivers"] = receivers_builder
    return out


