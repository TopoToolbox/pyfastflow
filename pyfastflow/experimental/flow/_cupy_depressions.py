"""
cupy (CUDA source) block templates behind make_depressions/
make_depression_solver: the i64 atomic_min helper, copy_field, both basin
labelling variants, saddlesort, both carve variants, jump reroute, and the
depression counter - on the new builder/frozen/bound/routine stack (../core/
context/builder.py, frozen.py, bound.py, routine.py). Mirrors
_closure_depressions.py step for step - same routine composition, same
grid/bitpack occurrence-per-site shape - CUDA text instead of python defs.

See _cupy_receivers.py/_cupy_accum.py/_cupy_reconstruct.py for the other
flow algorithms. Based on ../../flow/flow_reroute_kernels.py; `bitpack`'s pack/
unpack_value/unpack_index (ops.make_bitpack_group, a FrozenGroup) replace
legacy's f32_i32_struct module. Every array here (rec, bid, tag,
basin_saddle, outlet, ...) is n_flat-sized, basin id = pit index + 1, so a
per-basin array is safely indexed by any node index too - the same double
duty the legacy kernels rely on.

Unlike the closure backends, cupy has no grid-wide barrier a single
`__global__` can rely on - every ordering dependency a pass needs is a real,
separate kernel launch (matching _cupy_accum.py's own closure/cupy split for
the same reason): `label_basins_walk` (one closure kernel) becomes three
launches here ("walk_copy"/"walk_halving"/"walk_finalize"),
`iteration_reroute_carve` (one closure kernel) becomes two
("iter_build_work"/"iter_jump"), `reroute_jump` (one closure kernel with two
top-level loops) becomes two ("reset_rerouted"/"jump").

`n_flat` is a required, explicit build-time python int (baked into every
launch-bounds check, `{n_flat}`, the same idiom _cupy_accum.py's build_atomic
uses) - this factory takes no pool and reads no Parameter for it. `grid`'s
own `N_NEIGHBOURS` is read on-device (`$ctx.grid.N_NEIGHBOURS.get(0)$`,
exactly as _cupy_receivers.py's build_receivers already does), never a
host-side python int - no build-time n_neighbours argument needed at all.

Every `__global__`/`__device__` symbol is prefixed with this build's own tag
(a fresh new_uid()) so two make_depressions() calls in one process never
collide inside a single compiled cupy module - matching _cupy_receivers.py/
_cupy_accum.py.

A fixed, build-time-constant repeat (propagate_basin_iter's/
iteration_reroute_carve's `logn+1` rounds) is unrolled as `logn+1` distinct
routine compose() names for the SAME FrozenKernel - see
_closure_depressions.py's module docstring for why (no per-round host
readback, so nothing a SequenceBuilder loop would buy over a flat unroll).

Author: B.G (08/2026)
"""

from ..core.context.builder import HelperBuilder, KernelBuilder
from ..core.context.routine import RoutineBuilder
from ..core.pool.base import new_uid


def build_atomic_min_ll():
    """
    atomicMin over a signed 64-bit cell via a CAS loop - CUDA has no native
    atomicMin for signed long long (only int and unsigned long long), and
    the bitpacked saddle/outlet values need signed comparison to match
    Taichi/Quadrants' `atomic_min` over an i64 field.

    Returns
    -------
    HelperBuilder

    Author: B.G (08/2026)
    """
    t = f"pd{new_uid()}"
    return HelperBuilder().ingest(
        f"""
__device__ long long {t}_atomic_min_ll(long long* addr, long long val) {{
    long long old = *addr, assumed;
    do {{
        assumed = old;
        if (assumed <= val) break;
        old = (long long)atomicCAS((unsigned long long*)addr, (unsigned long long)assumed, (unsigned long long)val);
    }} while (assumed != old);
    return old;
}}
"""
    )


def build_copy_field(*, n_flat: int):
    """
    dst[i] = src[i] over a whole n_flat int32 buffer - see
    _closure_depressions.py's build_copy_field.

    Parameters
    ----------
    n_flat : int

    Returns
    -------
    KernelBuilder

    Author: B.G (08/2026)
    """
    t = f"pd{new_uid()}"
    return (
        KernelBuilder().wire_data("src").wire_data("dst").ingest(
            f"""
__global__ void {t}_copy_field(const int* src, int* dst) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    dst[i] = src[i];
}}
"""
        )
    )


def build_basin_id_init(*, grid, n_flat: int):
    """
    bid[i] = 0 on a can_out node, i+1 otherwise. Data arg (bid,). Composes
    its own `grid` occurrence.

    Parameters
    ----------
    grid : FrozenGroup
    n_flat : int

    Returns
    -------
    KernelBuilder

    Author: B.G (08/2026)
    """
    t = f"pbi{new_uid()}"
    return (
        KernelBuilder().compose("grid", grid).wire_data("bid").ingest(
            f"""
__global__ void {t}_basin_id_init(int* bid) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    bid[i] = $ctx.grid.can_out(i)$ ? 0 : (i + 1);
}}
"""
        )
    )


def build_propagate_basin_iter(*, n_flat: int):
    """One pointer-jump step over `rec_jump`. Data arg (rec_jump,).

    Parameters
    ----------
    n_flat : int

    Returns
    -------
    KernelBuilder
    """
    t = f"pbi{new_uid()}"
    return (
        KernelBuilder().wire_data("rec_jump").ingest(
            f"""
__global__ void {t}_propagate_basin_iter(int* rec_jump) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    if (rec_jump[i] != rec_jump[rec_jump[i]]) {{
        rec_jump[i] = rec_jump[rec_jump[i]];
    }}
}}
"""
        )
    )


def build_propagate_basin_final(*, n_flat: int):
    """bid[i] = bid[root(i)]. Data args (bid, rec_jump).

    Parameters
    ----------
    n_flat : int

    Returns
    -------
    KernelBuilder
    """
    t = f"pbf{new_uid()}"
    return (
        KernelBuilder().wire_data("bid").wire_data("rec_jump").ingest(
            f"""
__global__ void {t}_propagate_basin_final(int* bid, const int* rec_jump) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    bid[i] = bid[rec_jump[i]];
}}
"""
        )
    )


def build_basin_labelling_vanilla(*, grid, copy_field, n_flat: int, logn: int):
    """
    RoutineBuilder (routine) for vanilla basin labelling - see
    _closure_depressions.py's own (identical step sequence and unroll
    choice). Every step here is already one launch (no cross-loop splitting
    needed for this variant - it has no per-round cross-thread ordering
    dependency other kernels here need split for).

    Parameters
    ----------
    grid : FrozenGroup
    copy_field : KernelBuilder
    n_flat : int
    logn : int

    Returns
    -------
    tuple[RoutineBuilder, dict]

    Author: B.G (08/2026)
    """
    basin_id_init = build_basin_id_init(grid=grid, n_flat=n_flat)
    propagate_basin_iter = build_propagate_basin_iter(n_flat=n_flat)
    propagate_basin_final = build_propagate_basin_final(n_flat=n_flat)

    kernels = {
        "basin_id_init": basin_id_init,
        "propagate_basin_iter": propagate_basin_iter,
        "propagate_basin_final": propagate_basin_final,
    }

    rb = RoutineBuilder()
    rb.compose("basin_id_init", basin_id_init)
    rb.compose("copy_rec_to_recjump", copy_field)
    for k in range(logn + 1):
        rb.compose(f"propagate_iter_{k}", propagate_basin_iter)
    rb.compose("propagate_basin_final", propagate_basin_final)

    return rb, kernels


def build_basin_labelling_optimized(*, grid, n_flat: int):
    """
    RoutineBuilder (routine) for optimized basin labelling - the closure
    backends' single label_basins_walk launch split into three real
    launches (copy, path-halving, bid finalize), since the path-halving
    phase needs every thread's copy to have landed first, and the finalize
    phase needs every thread's path-halving to have converged first.

    Parameters
    ----------
    grid : FrozenGroup
    n_flat : int

    Returns
    -------
    tuple[RoutineBuilder, dict]

    Author: B.G (08/2026)
    """
    t = f"pbo{new_uid()}"

    walk_copy = (
        KernelBuilder().wire_data("rec").wire_data("rec_jump").ingest(
            f"""
__global__ void {t}_walk_copy(const int* rec, int* rec_jump) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    rec_jump[i] = rec[i];
}}
"""
        )
    )
    walk_halving = (
        KernelBuilder().wire_data("rec_jump").ingest(
            f"""
__global__ void {t}_walk_halving(int* rec_jump) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    int guard = 0;
    while (rec_jump[i] != rec_jump[rec_jump[i]] && guard < {n_flat}) {{
        rec_jump[i] = rec_jump[rec_jump[i]];
        guard++;
    }}
}}
"""
        )
    )
    walk_finalize = (
        KernelBuilder().compose("grid", grid).wire_data("rec_jump").wire_data("bid").ingest(
            f"""
__global__ void {t}_walk_finalize(const int* rec_jump, int* bid) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    int root = rec_jump[i];
    bid[i] = $ctx.grid.can_out(root)$ ? 0 : root + 1;
}}
"""
        )
    )

    kernels = {"walk_copy": walk_copy, "walk_halving": walk_halving, "walk_finalize": walk_finalize}

    rb = RoutineBuilder()
    rb.compose("walk_copy", walk_copy)
    rb.compose("walk_halving", walk_halving)
    rb.compose("walk_finalize", walk_finalize)

    return rb, kernels


def build_saddlesort(*, grid, bitpack, n_flat: int):
    """
    RoutineBuilder (routine) for the six saddlesort passes - see
    _closure_depressions.py's build_saddlesort for the step sequence.
    `bitpack` is the FrozenGroup ops.make_bitpack_group returns
    (`$ctx.bitpack.pack(...)$`/`.unpack_value`/`.unpack_index`); each
    KernelBuilder composes its own occurrence, only where it actually calls
    one of the three. `atomic_min_ll` (build_atomic_min_ll) is composed onto
    the two sites that need it.

    Parameters
    ----------
    grid : FrozenGroup
    bitpack : FrozenGroup
    n_flat : int

    Returns
    -------
    tuple[RoutineBuilder, dict]

    Author: B.G (08/2026)
    """
    atomic_min_ll = build_atomic_min_ll()
    t = f"pss{new_uid()}"

    border_zprime = (
        KernelBuilder().compose("grid", grid)
        .wire_data("bid").wire_data("z").wire_data("z_prime").wire_data("is_border")
        .ingest(
            f"""
__global__ void {t}_border_zprime(const int* bid, const float* z, float* z_prime, unsigned char* is_border) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    if ($ctx.grid.can_out(i)$) {{
        z_prime[i] = z[i];
        return;
    }}
    is_border[i] = 0;
    z_prime[i] = 1e9f;
    float zn = 1e9f;
    int nk = $ctx.grid.N_NEIGHBOURS.get(0)$;
    for (int k = 0; k < nk; k++) {{
        int j = $ctx.grid.neighbour(i, k)$;
        if (j != -1 && bid[j] != bid[i]) {{
            is_border[i] = 1;
            zn = fminf(zn, z[j]);
        }}
    }}
    if (is_border[i]) {{
        z_prime[i] = fmaxf(z[i], zn);
    }}
}}
"""
        )
    )
    init_saddle_outlet = (
        KernelBuilder().compose("bitpack", bitpack)
        .wire_data("basin_saddle").wire_data("outlet").wire_data("basin_saddlenode")
        .ingest(
            f"""
__global__ void {t}_init_saddle_outlet(long long* basin_saddle, long long* outlet, int* basin_saddlenode) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    long long invalid = $ctx.bitpack.pack(1e8, 42)$;
    basin_saddle[i] = invalid;
    outlet[i] = invalid;
    basin_saddlenode[i] = -1;
}}
"""
        )
    )
    atomic_min_saddle = (
        KernelBuilder().compose("grid", grid).compose("bitpack", bitpack).compose("atomic_min_ll", atomic_min_ll)
        .wire_data("bid").wire_data("is_border").wire_data("z_prime").wire_data("basin_saddle")
        .ingest(
            f"""
__global__ void {t}_atomic_min_saddle(const int* bid, const unsigned char* is_border, const float* z_prime, long long* basin_saddle) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    if (!is_border[i]) return;
    long long invalid = $ctx.bitpack.pack(1e8, 42)$;
    int tbid = bid[i];
    long long res = invalid;
    int nk = $ctx.grid.N_NEIGHBOURS.get(0)$;
    for (int k = 0; k < nk; k++) {{
        int j = $ctx.grid.neighbour(i, k)$;
        if (j != -1 && bid[j] != tbid) {{
            long long candidate = $ctx.bitpack.pack(z_prime[i], bid[j])$;
            res = (candidate < res) ? candidate : res;
        }}
    }}
    if (res != invalid) {{
        $ctx.atomic_min_ll(&basin_saddle[tbid], res)$;
    }}
}}
"""
        )
    )
    find_saddlenode = (
        KernelBuilder().compose("grid", grid).compose("bitpack", bitpack)
        .wire_data("bid").wire_data("is_border").wire_data("z_prime")
        .wire_data("basin_saddle").wire_data("basin_saddlenode")
        .ingest(
            f"""
__global__ void {t}_find_saddlenode(const int* bid, const unsigned char* is_border, const float* z_prime,
                                     const long long* basin_saddle, int* basin_saddlenode) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    if (!is_border[i] || bid[i] == 0) return;
    long long packed = basin_saddle[bid[i]];
    float target_z = $ctx.bitpack.unpack_value(packed)$;
    int target_b = $ctx.bitpack.unpack_index(packed)$;
    int is_here = 0;
    int nk = $ctx.grid.N_NEIGHBOURS.get(0)$;
    for (int k = 0; k < nk; k++) {{
        int j = $ctx.grid.neighbour(i, k)$;
        if (j != -1 && bid[j] == target_b && z_prime[i] == target_z) {{
            is_here = 1;
        }}
    }}
    if (is_here) {{
        basin_saddlenode[bid[i]] = i;
    }}
}}
"""
        )
    )
    atomic_min_outlet = (
        KernelBuilder().compose("grid", grid).compose("bitpack", bitpack).compose("atomic_min_ll", atomic_min_ll)
        .wire_data("bid").wire_data("basin_saddle").wire_data("basin_saddlenode")
        .wire_data("z").wire_data("outlet")
        .ingest(
            f"""
__global__ void {t}_atomic_min_outlet(const int* bid, const long long* basin_saddle, const int* basin_saddlenode,
                                       const float* z, long long* outlet) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    long long invalid = $ctx.bitpack.pack(1e8, 42)$;
    if (i == 0 || basin_saddle[i] == invalid) return;
    int node = basin_saddlenode[i];
    float tz = 1e9f;
    int rec_out = -1;
    int nk = $ctx.grid.N_NEIGHBOURS.get(0)$;
    for (int k = 0; k < nk; k++) {{
        int j = $ctx.grid.neighbour(node, k)$;
        if (j != -1 && bid[j] != i && tz > z[j]) {{
            tz = z[j];
            rec_out = j;
        }}
    }}
    if (rec_out > -1) {{
        long long candidate = $ctx.bitpack.pack(tz, rec_out)$;
        $ctx.atomic_min_ll(&outlet[i], candidate)$;
    }}
}}
"""
        )
    )
    break_cycle = (
        KernelBuilder().compose("bitpack", bitpack)
        .wire_data("bid").wire_data("outlet").wire_data("basin_saddle").wire_data("basin_saddlenode")
        .ingest(
            f"""
__global__ void {t}_break_cycle(const int* bid, long long* outlet, long long* basin_saddle, int* basin_saddlenode) {{
    int bid_d = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid_d >= {n_flat}) return;
    long long invalid = $ctx.bitpack.pack(1e8, 42)$;
    if (bid_d == 0 || outlet[bid_d] == invalid) return;
    int rec_out = $ctx.bitpack.unpack_index(outlet[bid_d])$;
    int bid_d_prime = bid[rec_out];
    if (bid_d_prime == 0) return;
    int rec_out_prime = $ctx.bitpack.unpack_index(outlet[bid_d_prime])$;
    int bid_d_prime_prime = bid[rec_out_prime];
    if (bid_d_prime_prime == bid_d && bid_d_prime < bid_d) {{
        outlet[bid_d] = invalid;
        basin_saddle[bid_d] = invalid;
        basin_saddlenode[bid_d] = -1;
    }}
}}
"""
        )
    )

    kernels = {
        "border_zprime": border_zprime,
        "init_saddle_outlet": init_saddle_outlet,
        "atomic_min_saddle": atomic_min_saddle,
        "find_saddlenode": find_saddlenode,
        "atomic_min_outlet": atomic_min_outlet,
        "break_cycle": break_cycle,
    }

    rb = RoutineBuilder()
    rb.compose("border_zprime", border_zprime)
    rb.compose("init_saddle_outlet", init_saddle_outlet)
    rb.compose("atomic_min_saddle", atomic_min_saddle)
    rb.compose("find_saddlenode", find_saddlenode)
    rb.compose("atomic_min_outlet", atomic_min_outlet)
    rb.compose("break_cycle", break_cycle)

    return rb, kernels


def build_reroute_carve_vanilla(*, bitpack, copy_field, n_flat: int, logn: int):
    """
    RoutineBuilder (routine) for carve+vanilla reroute - see
    _closure_depressions.py's build_reroute_carve_vanilla for the buffer
    roles (`rec_jump` here is finalise's original, unjumped snapshot, not
    the pointer-jumped result - same note applies). init_reroute_carve,
    iteration_reroute_carve and finalise_reroute_carve are each further
    split into several real launches here (no grid-wide barrier inside one
    `__global__`); the closure backends keep each as one kernel.

    Parameters
    ----------
    bitpack : FrozenGroup
    copy_field : KernelBuilder
    n_flat : int
    logn : int

    Returns
    -------
    tuple[RoutineBuilder, dict]

    Author: B.G (08/2026)
    """
    t = f"prc{new_uid()}"

    init_reset_tag = (
        KernelBuilder().wire_data("tag").ingest(
            f"""
__global__ void {t}_init_reset_tag(unsigned char* tag) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    tag[i] = 0;
}}
"""
        )
    )
    init_scatter_tag = (
        KernelBuilder().wire_data("tag").wire_data("saddlenode").ingest(
            f"""
__global__ void {t}_init_scatter_tag(unsigned char* tag, const int* saddlenode) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    if (saddlenode[i] != -1) {{
        tag[saddlenode[i]] = 1;
    }}
}}
"""
        )
    )
    init_copy_tag_alt = (
        KernelBuilder().wire_data("tag").wire_data("tag_alt").ingest(
            f"""
__global__ void {t}_init_copy_tag_alt(const unsigned char* tag, unsigned char* tag_alt) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    tag_alt[i] = tag[i];
}}
"""
        )
    )
    iter_build_work = (
        KernelBuilder()
        .wire_data("tag").wire_data("tag_alt").wire_data("rec").wire_data("rec_work").wire_data("bid")
        .ingest(
            f"""
__global__ void {t}_iter_build_work(const unsigned char* tag, unsigned char* tag_alt, const int* rec,
                                     int* rec_work, const int* bid) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    if (bid[i] == 0) return;
    if (tag[i] && rec[i] != i) {{
        tag_alt[rec[i]] = 1;
    }}
    rec_work[i] = rec[i];
}}
"""
        )
    )
    iter_jump = (
        KernelBuilder()
        .wire_data("tag").wire_data("tag_alt").wire_data("rec").wire_data("rec_work").wire_data("bid")
        .ingest(
            f"""
__global__ void {t}_iter_jump(unsigned char* tag, const unsigned char* tag_alt, int* rec,
                               const int* rec_work, const int* bid) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    if (bid[i] == 0) return;
    if (rec_work[i] != i) {{
        rec[i] = rec_work[rec_work[i]];
    }}
    tag[i] = tag_alt[i];
}}
"""
        )
    )
    finalise_reset_rec = (
        KernelBuilder().wire_data("rec").wire_data("rec_orig").ingest(
            f"""
__global__ void {t}_finalise_reset_rec(int* rec, const int* rec_orig) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    rec[i] = rec_orig[i];
}}
"""
        )
    )
    finalise_reverse = (
        KernelBuilder().wire_data("rec").wire_data("rec_orig").wire_data("tag").wire_data("rerouted").ingest(
            f"""
__global__ void {t}_finalise_reverse(int* rec, const int* rec_orig, const unsigned char* tag, unsigned char* rerouted) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    int ro = rec_orig[i];
    if (tag[ro] && tag[i] && i != ro) {{
        rec[ro] = i;
        rerouted[ro] = 1;
    }}
}}
"""
        )
    )
    finalise_outlet = (
        KernelBuilder().compose("bitpack", bitpack)
        .wire_data("rec").wire_data("outlet").wire_data("saddlenode").wire_data("rerouted")
        .ingest(
            f"""
__global__ void {t}_finalise_outlet(int* rec, const long long* outlet, const int* saddlenode, unsigned char* rerouted) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    long long invalid = $ctx.bitpack.pack(1e8, 42)$;
    if (outlet[i] != invalid) {{
        int node = $ctx.bitpack.unpack_index(outlet[i])$;
        rec[saddlenode[i]] = node;
        rerouted[saddlenode[i]] = 1;
    }}
}}
"""
        )
    )

    kernels = {
        "init_reset_tag": init_reset_tag,
        "init_scatter_tag": init_scatter_tag,
        "init_copy_tag_alt": init_copy_tag_alt,
        "iter_build_work": iter_build_work,
        "iter_jump": iter_jump,
        "finalise_reset_rec": finalise_reset_rec,
        "finalise_reverse": finalise_reverse,
        "finalise_outlet": finalise_outlet,
    }

    rb = RoutineBuilder()
    rb.compose("init_reset_tag", init_reset_tag)
    rb.compose("init_scatter_tag", init_scatter_tag)
    rb.compose("init_copy_tag_alt", init_copy_tag_alt)
    rb.compose("copy_recwork_to_rec", copy_field)
    rb.compose("copy_recwork_to_recjump", copy_field)
    for k in range(logn + 1):
        rb.compose(f"iter_build_work_{k}", iter_build_work)
        rb.compose(f"iter_jump_{k}", iter_jump)
    rb.compose("finalise_reset_rec", finalise_reset_rec)
    rb.compose("finalise_reverse", finalise_reverse)
    rb.compose("finalise_outlet", finalise_outlet)
    rb.compose("copy_rec_to_recwork", copy_field)

    return rb, kernels


def build_reroute_carve_optimized(*, bitpack, n_flat: int):
    """
    carve_basins_serial KernelBuilder - one launch, one serial thread per
    basin; see _closure_depressions.py's build_reroute_carve_optimized.
    Node-disjoint chains across basins mean no cross-thread dependency at
    all, so this needs no splitting the way the vanilla carve routine does.

    Parameters
    ----------
    bitpack : FrozenGroup
    n_flat : int

    Returns
    -------
    KernelBuilder

    Author: B.G (08/2026)
    """
    t = f"pco{new_uid()}"
    return (
        KernelBuilder().compose("bitpack", bitpack)
        .wire_data("rec").wire_data("basin_saddlenode").wire_data("outlet")
        .ingest(
            f"""
__global__ void {t}_carve_basins_serial(int* rec, const int* basin_saddlenode, const long long* outlet) {{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= {n_flat}) return;
    long long invalid = $ctx.bitpack.pack(1e8, 42)$;
    int s = basin_saddlenode[b];
    if (s == -1 || outlet[b] == invalid) return;
    int out_node = $ctx.bitpack.unpack_index(outlet[b])$;
    int node = s;
    int nxt = rec[node];
    rec[node] = out_node;
    while (nxt != node) {{
        int nnxt = rec[nxt];
        rec[nxt] = node;
        node = nxt;
        nxt = nnxt;
    }}
}}
"""
        )
    )


def build_reroute_jump(*, bitpack, n_flat: int):
    """
    RoutineBuilder (routine) for reroute_jump - split into a reset launch
    and the jump launch itself, since the jump phase writes
    `rerouted[i - 1]` from thread `i`, a cell a *different* thread's reset
    zeroed. The closure backends keep this as one two-loop kernel.

    The write is deliberately `rec[i - 1]`, not `rec[i]` - see
    _closure_depressions.py's build_reroute_jump docstring for why; ported
    exactly.

    Parameters
    ----------
    bitpack : FrozenGroup
    n_flat : int

    Returns
    -------
    tuple[RoutineBuilder, dict]

    Author: B.G (08/2026)
    """
    t = f"prj{new_uid()}"

    reset_rerouted = (
        KernelBuilder().wire_data("rerouted").ingest(
            f"""
__global__ void {t}_reset_rerouted(unsigned char* rerouted) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    rerouted[i] = 0;
}}
"""
        )
    )
    jump = (
        KernelBuilder().compose("bitpack", bitpack)
        .wire_data("rec").wire_data("outlet").wire_data("rerouted")
        .ingest(
            f"""
__global__ void {t}_jump(int* rec, const long long* outlet, unsigned char* rerouted) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    long long invalid = $ctx.bitpack.pack(1e8, 42)$;
    if (outlet[i] != invalid) {{
        int rrec = $ctx.bitpack.unpack_index(outlet[i])$;
        rec[i - 1] = rrec;
        rerouted[i - 1] = 1;
    }}
}}
"""
        )
    )

    kernels = {"reset_rerouted": reset_rerouted, "jump": jump}

    rb = RoutineBuilder()
    rb.compose("reset_rerouted", reset_rerouted)
    rb.compose("jump", jump)

    return rb, kernels


def build_depression_counter(*, grid, n_flat: int):
    """
    depression_counter KernelBuilder, data args (rec, ndep) - `ndep` is
    `ndep_p.get().data`, passed positionally same as `rec` (a Parameter
    reached only through `$...$` get() spans is registered read-only in the
    constant block, so atomicAdd into it needs the raw pointer as an
    ordinary DATA argument instead). The caller must reset `ndep_p` to 0
    (`.set(0)`) before each launch. Composes its own `grid` occurrence.

    Parameters
    ----------
    grid : FrozenGroup
    n_flat : int

    Returns
    -------
    KernelBuilder

    Author: B.G (08/2026)
    """
    t = f"pdc{new_uid()}"
    return (
        KernelBuilder().compose("grid", grid)
        .wire_data("rec").wire_data("ndep")
        .ingest(
            f"""
__global__ void {t}_depression_counter(const int* rec, int* ndep) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    if (rec[i] == i && !($ctx.grid.can_out(i)$)) {{
        atomicAdd(ndep, 1);
    }}
}}
"""
        )
    )
