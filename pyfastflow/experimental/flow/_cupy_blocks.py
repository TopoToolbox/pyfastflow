"""
cupy (CUDA source) block templates behind make_receivers and
make_accumulation.

Mirrors _closure_blocks.py block for block: same private/public split, same
`mode`/`h_aware`/`diagonal_partition_correction` selectors picking which CUDA
text gets built - written as CUDA text instead of python defs, through the
`$...$` span mechanism (see ../core/context/cupy_backend.py's module
docstring). Every `__device__`/`__global__` symbol is prefixed with this
build's own tag (a fresh new_uid()) so two make_receivers() calls in one
process never collide inside a single compiled cupy module.

The accumulation kernels (build_rake_compress/build_pointer_jump_push/
build_atomic, further down) build a CupyRoutineBuilder. cupy has no source
fusion (see cupy_backend.py) - captured=True (the default) launches every
step once, for real, then replays the same launch sequence off a CUDA graph,
so the barrier every round needs between steps is just the ordinary stream
ordering CUDA already gives consecutive launches; there is nothing here
equivalent to the closure backend's fused-vs-split() choice.

Author: B.G (07/2026)
"""

import functools

from ..core.context.backends import make_helper
from ..core.context.bag import Bag
from ..core.pool.base import new_uid


def build_distance_slope_helpers(HelperCls, *, grid, diagonal_partition_correction):
    """
    dist_from_k_corrected/dist_between_nodes_corrected/slope_from_values_k/
    slope_between_nodes for the cupy backend.

    When `diagonal_partition_correction` is off, or the grid is not D8, the
    "corrected" distance helpers are simply the grid's own dist_from_k /
    dist_between_nodes HelperBuilders - no branch, no separate template.

    Returns {name: HelperBuilder}.

    Author: B.G (07/2026)
    """
    t = f"pr{new_uid()}"
    mk = functools.partial(make_helper, HelperCls)

    d8 = grid.n_neighbours.get() == 8
    if diagonal_partition_correction and d8:
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
            grid=grid,
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
            grid=grid,
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
        dist_from_k_corrected=dist_from_k_corrected,
    )
    slope_between_nodes = mk(
        f"""
__device__ float {t}_slope_between_nodes(float vi, float vj, int i, int j) {{
    return (vi - vj) / $dist_between_nodes_corrected(i, j)$;
}}
""",
        dist_between_nodes_corrected=dist_between_nodes_corrected,
    )

    return {
        "dist_from_k_corrected": dist_from_k_corrected,
        "dist_between_nodes_corrected": dist_between_nodes_corrected,
        "slope_from_values_k": slope_from_values_k,
        "slope_between_nodes": slope_between_nodes,
    }


def build_rand_unit(HelperCls, *, seed_p, hash_u32):
    """
    rand_unit(i, k) HelperBuilder, binding the caller-supplied `hash_u32`
    (noise's public hash_u32 HelperBuilder - see ../noise/_cupy_blocks.py)
    rather than a private copy. Node index and neighbour direction are mixed
    separately (mirroring noise's white_unit col/row mixing), so every
    (node, k) candidate draws its own value.

    Author: B.G (07/2026)
    """
    t = f"pr{new_uid()}"
    mk = functools.partial(make_helper, HelperCls)
    return mk(
        f"""
__device__ float {t}_rand_unit(int i, int k) {{
    unsigned int key = (unsigned int)$SEED.get(0)$;
    key ^= (unsigned int)i * 374761393u;
    key ^= (unsigned int)k * 668265263u;
    unsigned int hashed = $hash_u32(key)$;
    return (float)hashed / 4294967296.0f;
}}
""",
        SEED=seed_p,
        hash_u32=hash_u32,
    )


def build_receivers(
    KernelCls,
    HelperCls,
    *,
    grid,
    hash_u32,
    mode: str,
    seed_p,
    diagonal_partition_correction: bool,
    h_aware: bool,
):
    """
    Build one cupy `receivers` KernelBuilder plus the distance/slope (and,
    for mode="stochastic", rand_unit) HelperBuilders it is made of, picking
    one of four kernel body text variants (mode x h_aware) - never a runtime
    branch on either inside the generated kernel.

    `hash_u32` is the noise module's public hash_u32 HelperBuilder, reused
    here rather than re-implemented. Only required when mode="stochastic".

    Returns {name: HelperBuilder/KernelBuilder} - the distance/slope helpers
    plus "receivers", plus "rand_unit" when mode="stochastic".

    Author: B.G (07/2026)
    """
    out = build_distance_slope_helpers(HelperCls, grid=grid, diagonal_partition_correction=diagonal_partition_correction)
    slope = out["slope_from_values_k"]

    binds = {"grid": grid, "slope_from_values_k": slope}
    if mode == "stochastic":
        rand_unit = build_rand_unit(HelperCls, seed_p=seed_p, hash_u32=hash_u32)
        out["rand_unit"] = rand_unit
        binds["rand_unit"] = rand_unit
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
    receivers_builder = KernelCls().ingest(body)
    for name, obj in binds.items():
        receivers_builder = receivers_builder.bind(name, obj)

    out["receivers"] = receivers_builder
    return out


# ---------------------------------------------------------------------------
# accumulation: ping-pong src encoding (get_src/update_src, reading the
# iteration scalar Parameter instead of taking iteration as a call argument)
# ---------------------------------------------------------------------------


def build_ping_pong_helpers(HelperCls, *, iteration_p):
    """
    get_src(src, tid)/update_src(src, tid, flip) HelperBuilders - same
    sign/magnitude encoding as pyfastflow/general_algorithms/pingpong.py's
    getSrc/updateSrc, reading `iteration_p` internally instead of taking
    iteration as a call argument.

    Author: B.G (07/2026)
    """
    t = f"pp{new_uid()}"
    mk = functools.partial(make_helper, HelperCls)
    get_src = mk(
        f"""
__device__ int {t}_get_src(const int* src, int tid) {{
    int entry = src[tid];
    int it = $ITER.get(0)$;
    int flip = entry < 0;
    if (abs(entry) == (it + 1)) flip = !flip;
    return flip;
}}
""",
        ITER=iteration_p,
    )
    update_src = mk(
        f"""
__device__ void {t}_update_src(int* src, int tid, int flip) {{
    int it = $ITER.get(0)$;
    src[tid] = (flip ? 1 : -1) * (it + 1);
}}
""",
        ITER=iteration_p,
    )
    return get_src, update_src


def build_atomic(KernelCls, *, source, n_flat: int):
    """
    Two KernelBuilders, data args (q) and (rec, q): "q_init" (q[i] =
    source.get(i)) must be launched before "accum" (the atomic descent). Two
    real launches rather than one, unlike the closure backends' single
    KernelBuilder (see _closure_blocks.py's build_atomic) - a single CUDA
    __global__ has no portable grid-wide barrier the way two consecutive
    top-level Taichi/Quadrants for-loops do, so without a second launch a
    thread could atomic_add into q[j] before node j's own thread has run its
    q[j] = source.get(j) initialization.

    Author: B.G (07/2026)
    """
    t = f"pa{new_uid()}"
    q_init = KernelCls().bind("source", source).ingest(
        f"""
__global__ void {t}_q_init(float* q) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    q[i] = $source.get(i)$;
}}
"""
    )
    # wi is re-read from `source`, not from q[i]: q[i] is a live accumulation
    # target every other thread may already be atomic-adding into by the
    # time this thread runs (thread order across blocks is not guaranteed),
    # so reading q[i] here would race against those writes and silently
    # inflate wi with contributions that arrived early - exactly the bug an
    # earlier version of this kernel had (q[i] as a shortcut for wi, to avoid
    # re-binding `source`), caught by a max_abs deviation of ~1e23 at
    # n_flat=1e6 in _verify_accum.py. Matches legacy
    # accum_downstream_atomic_kernel and the closure-backend port, which
    # both re-read the weight function/Parameter directly for this reason.
    accum = KernelCls().bind("source", source).ingest(
        f"""
__global__ void {t}_accum_downstream_atomic(const int* rec, float* q) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    if (rec[i] == i) return;
    float wi = $source.get(i)$;
    int j = rec[i];
    int guard = 0;
    while (j != rec[j] && guard < {n_flat}) {{
        atomicAdd(&q[j], wi);
        j = rec[j];
        guard++;
    }}
    atomicAdd(&q[j], wi);
}}
"""
    )
    return {"q_init": q_init, "accum": accum}


def build_rake_compress(
    RoutineBuilderCls,
    KernelCls,
    HelperCls,
    *,
    grid,
    source,
    iteration_p,
    logn: int,
    n_flat: int,
):
    """
    CupyRoutineBuilder for the rake-and-compress accumulation, plus the
    KernelBuilders it is made of - see _closure_blocks.py's
    build_rake_compress for the step sequence and the iteration off-by-one.

    The routine's default launch grid/block is sized for `n_flat` threads -
    every step here reads/writes an n_flat-sized index space except the
    single-thread iteration bookkeeping steps, which override grid=(1,)/
    block=(1,) at add_kernel time.

    Author: B.G (07/2026)
    """
    NN = grid.n_neighbours.get()
    block_size = 256
    default_grid, default_block = ((n_flat + block_size - 1) // block_size,), (block_size,)
    get_src, update_src = build_ping_pong_helpers(HelperCls, iteration_p=iteration_p)
    t = f"pr{new_uid()}"

    zero_init = KernelCls().ingest(
        f"""
__global__ void {t}_zero_init(int* ndonors, int* ndonors_alt, int* src) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    ndonors[i] = 0;
    ndonors_alt[i] = 0;
    src[i] = 0;
}}
"""
    )
    reset_iteration = KernelCls().bind("ITER", iteration_p).ingest(
        f"""
__global__ void {t}_reset_iteration() {{
    $ITER.set_node(0, 0)$;
}}
"""
    )
    bump_iteration = KernelCls().bind("ITER", iteration_p).ingest(
        f"""
__global__ void {t}_bump_iteration() {{
    int cur = $ITER.get(0)$;
    $ITER.set_node(0, cur + 1)$;
}}
"""
    )
    decrement_iteration = KernelCls().bind("ITER", iteration_p).ingest(
        f"""
__global__ void {t}_decrement_iteration() {{
    int cur = $ITER.get(0)$;
    $ITER.set_node(0, cur - 1)$;
}}
"""
    )
    q_init = KernelCls().bind("source", source).ingest(
        f"""
__global__ void {t}_q_init(float* q) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    q[i] = $source.get(i)$;
}}
"""
    )
    receivers_to_donors = KernelCls().ingest(
        f"""
__global__ void {t}_receivers_to_donors(const int* rec, int* donors, int* ndonors) {{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= {n_flat}) return;
    int rcv = rec[tid];
    if (rcv != tid) {{
        int old_val = atomicAdd(&ndonors[rcv], 1);
        donors[rcv * {NN} + old_val] = tid;
    }}
}}
"""
    )
    rake_compress_accum = (
        KernelCls()
        .bind("_GETSRC", get_src)
        .bind("_UPDATESRC", update_src)
        .ingest(
            f"""
__global__ void {t}_rake_compress_accum(int* donors, int* ndonors, float* q, int* src,
                                         int* donors_alt, int* ndonors_alt, float* q_alt) {{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= {n_flat}) return;
    int flip = $_GETSRC(src, tid)$;

    int worked = 0;
    int todo = flip ? ndonors_alt[tid] : ndonors[tid];
    int base = tid * {NN};
    int donors_local[{NN}];
    for (int t2 = 0; t2 < {NN}; t2++) donors_local[t2] = -1;
    float q_added = 0.0f;

    int i = 0;
    while (i < todo && i < {NN}) {{
        if (donors_local[i] == -1) {{
            donors_local[i] = flip ? donors_alt[base + i] : donors[base + i];
        }}
        int did = donors_local[i];

        int flip_donor = $_GETSRC(src, did)$;
        int ndnr_val = flip_donor ? ndonors_alt[did] : ndonors[did];

        if (ndnr_val <= 1) {{
            if (!worked) {{
                q_added = flip ? q_alt[tid] : q[tid];
            }}
            worked = 1;

            float q_val = flip_donor ? q_alt[did] : q[did];
            q_added += q_val;

            if (ndnr_val == 0) {{
                todo -= 1;
                if (todo > i) {{
                    donors_local[i] = flip ? donors_alt[base + todo] : donors[base + todo];
                }}
                i -= 1;
            }} else {{
                int donor_base = did * {NN};
                donors_local[i] = flip_donor ? donors_alt[donor_base] : donors[donor_base];
            }}
        }}
        i += 1;
    }}

    if (worked) {{
        if (flip) {{
            ndonors[tid] = todo;
            q[tid] = q_added;
            for (int j = 0; j < {NN}; j++) {{
                if (j < todo) donors[base + j] = donors_local[j];
            }}
        }} else {{
            ndonors_alt[tid] = todo;
            q_alt[tid] = q_added;
            for (int j = 0; j < {NN}; j++) {{
                if (j < todo) donors_alt[base + j] = donors_local[j];
            }}
        }}
        $_UPDATESRC(src, tid, flip)$;
    }}
}}
"""
        )
    )
    fuse_accum_buffers = KernelCls().bind("_GETSRC", get_src).ingest(
        f"""
__global__ void {t}_fuse_accum_buffers(float* q, int* src, float* q_alt) {{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= {n_flat}) return;
    if ($_GETSRC(src, tid)$) {{
        q[tid] = q_alt[tid];
    }}
}}
"""
    )

    kernels = {
        "zero_init": zero_init,
        "reset_iteration": reset_iteration,
        "bump_iteration": bump_iteration,
        "decrement_iteration": decrement_iteration,
        "q_init": q_init,
        "receivers_to_donors": receivers_to_donors,
        "rake_compress_accum": rake_compress_accum,
        "fuse_accum_buffers": fuse_accum_buffers,
    }

    rb = RoutineBuilderCls(grid=default_grid, block=default_block)
    rb.bind_bag(Bag({"ITER": iteration_p, "source": source, "_GETSRC": get_src, "_UPDATESRC": update_src}))
    for name in ("rec", "q", "donors", "ndonors", "donors_alt", "ndonors_alt", "q_alt", "src"):
        rb.add_data(name, None)

    single_grid, single_block = (1,), (1,)

    rb.add_kernel(zero_init, data_handle_ref=("ndonors", "ndonors_alt", "src"))
    rb.add_kernel(reset_iteration, data_handle_ref=(), grid=single_grid, block=single_block)
    rb.add_kernel(q_init, data_handle_ref=("q",))
    rb.add_kernel(receivers_to_donors, data_handle_ref=("rec", "donors", "ndonors"))
    rb.begin_repeat(times=logn + 1)
    rb.add_kernel(rake_compress_accum, data_handle_ref=("donors", "ndonors", "q", "src", "donors_alt", "ndonors_alt", "q_alt"))
    rb.add_kernel(bump_iteration, data_handle_ref=(), grid=single_grid, block=single_block)
    rb.end_repeat()
    rb.add_kernel(decrement_iteration, data_handle_ref=(), grid=single_grid, block=single_block)
    rb.add_kernel(fuse_accum_buffers, data_handle_ref=("q", "src", "q_alt"))

    return rb, kernels


def build_pointer_jump_push(
    RoutineBuilderCls,
    KernelCls,
    *,
    source,
    rounds: int,
    n_flat: int,
):
    """
    CupyRoutineBuilder for the pointer-jump-push accumulation, plus the
    KernelBuilders it is made of - see _closure_blocks.py's
    build_pointer_jump_push for the step sequence and retirement rule.

    The routine's default launch grid/block is sized for `n_flat` threads -
    every step here reads/writes an n_flat-sized index space.

    Author: B.G (07/2026)
    """
    t = f"pj{new_uid()}"
    block_size = 256
    default_grid, default_block = ((n_flat + block_size - 1) // block_size,), (block_size,)

    q_init = KernelCls().bind("source", source).ingest(
        f"""
__global__ void {t}_q_init(float* q) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    q[i] = $source.get(i)$;
}}
"""
    )
    copy_rec_to_work = KernelCls().ingest(
        f"""
__global__ void {t}_copy_rec_to_work(const int* rec, int* work) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    work[i] = rec[i];
}}
"""
    )
    # Two separate __global__ launches, not one kernel with a __syncthreads():
    # __syncthreads() only orders threads within one block, but a thread's
    # atomic_add into q_next[parent] may target an index owned by a thread
    # in a different block - only a real launch boundary (implicit grid-wide
    # barrier between two kernel launches on the same stream) guarantees the
    # copy into q_next has landed everywhere before any push into it, the
    # same guarantee two consecutive top-level for-loops in one Taichi/
    # Quadrants kernel give for free (see _closure_blocks.py's equivalent).
    step_copy = KernelCls().ingest(
        f"""
__global__ void {t}_step_copy(const float* q_curr, float* q_next) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    q_next[i] = q_curr[i];
}}
"""
    )
    step_core = KernelCls().ingest(
        f"""
__global__ void {t}_step_core(const int* rec_curr, int* rec_next, const float* q_curr, float* q_next) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    int parent = rec_curr[i];
    rec_next[i] = parent;
    if (parent != i) {{
        float wi = q_curr[i];
        if (wi != 0.0f) {{
            atomicAdd(&q_next[parent], wi);
        }}
        int grandparent = rec_curr[parent];
        rec_next[i] = (grandparent == parent) ? i : grandparent;
    }}
}}
"""
    )

    kernels = {
        "q_init": q_init,
        "copy_rec_to_work": copy_rec_to_work,
        "accum_pointer_jump_push_step_copy": step_copy,
        "accum_pointer_jump_push_step_core": step_core,
    }

    rb = RoutineBuilderCls(grid=default_grid, block=default_block)
    rb.bind_bag(Bag({"source": source}))
    for name in ("rec", "work", "work2", "q", "q_work"):
        rb.add_data(name, None)

    rb.add_kernel(q_init, data_handle_ref=("q",))
    rb.add_kernel(copy_rec_to_work, data_handle_ref=("rec", "work"))
    rb.begin_repeat(times=rounds)
    rb.add_kernel(step_copy, data_handle_ref=("q", "q_work"))
    rb.add_kernel(step_core, data_handle_ref=("work", "work2", "q", "q_work"))
    rb.add_swap("work", "work2")
    rb.add_swap("q", "q_work")
    rb.end_repeat()

    return rb, kernels


# ---------------------------------------------------------------------------
# depression handling: basin labelling, saddlesort, reroute - cupy mirror of
# _closure_blocks.py's equivalent section (see its module-level comment
# there for the port source and the n_flat-sized-per-basin-array note).
#
# Every closure-backend kernel with more than one top-level `for` loop where
# a later loop reads a cell a *different* thread wrote in an earlier loop
# needs a real launch boundary here, since cupy has no grid-wide barrier
# inside one __global__ (see this module's own docstring and
# build_atomic/build_pointer_jump_push above for the established pattern).
# Audited per kernel: label_basins_walk (copy -> path-halving -> bid
# finalize), init_reroute_carve (reset -> scatter -> copy), the two-loop
# iteration_reroute_carve and reroute_jump, and three-loop
# finalise_reroute_carve all have such a cross-thread dependency and are
# therefore split into that many separate launches below, chained in a
# RoutineBuilder. carve_basins_serial and every saddlesort kernel have no
# such dependency *within* their own single loop (their cross-basin/cross-
# node reads are of state a strictly earlier kernel already finished
# writing) and stay one launch each, exactly as on the closure backends.
# ---------------------------------------------------------------------------


def build_atomic_min_ll(HelperCls):
    """
    atomicMin over a signed 64-bit cell via a CAS loop - CUDA has no native
    atomicMin for signed long long (only int and unsigned long long), and
    the bitpacked saddle/outlet values need signed comparison to match
    Taichi/Quadrants' `atomic_min` over an i64 field.

    Author: B.G (07/2026)
    """
    t = f"pd{new_uid()}"
    return make_helper(
        HelperCls,
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
""",
    )


def _launch_dims(n_flat: int, block_size: int = 256):
    return ((n_flat + block_size - 1) // block_size,), (block_size,)


def build_copy_field(KernelCls, *, n_flat: int):
    """
    dst[i] = src[i] over a whole n_flat int32 buffer - see
    _closure_blocks.build_copy_field.

    Author: B.G (07/2026)
    """
    t = f"pd{new_uid()}"
    return KernelCls().ingest(
        f"""
__global__ void {t}_copy_field(const int* src, int* dst) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    dst[i] = src[i];
}}
"""
    )


def build_basin_labelling_vanilla(RoutineBuilderCls, KernelCls, *, grid, copy_field, n_flat: int, logn: int):
    """
    RoutineBuilder for vanilla basin labelling - see
    _closure_blocks.build_basin_labelling_vanilla for the step sequence.
    Every step here is one launch already (no cross-loop split needed - see
    this module's own section docstring).

    Data names: "rec", "bid", "rec_jump".

    Author: B.G (07/2026)
    """
    default_grid, default_block = _launch_dims(n_flat)
    t = f"pbl{new_uid()}"

    basin_id_init = KernelCls().bind("grid", grid).ingest(
        f"""
__global__ void {t}_basin_id_init(int* bid) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    bid[i] = $grid.can_out(i)$ ? 0 : (i + 1);
}}
"""
    )
    propagate_basin_iter = KernelCls().ingest(
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
    propagate_basin_final = KernelCls().ingest(
        f"""
__global__ void {t}_propagate_basin_final(int* bid, const int* rec_jump) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    bid[i] = bid[rec_jump[i]];
}}
"""
    )

    kernels = {
        "basin_id_init": basin_id_init,
        "propagate_basin_iter": propagate_basin_iter,
        "propagate_basin_final": propagate_basin_final,
    }

    rb = RoutineBuilderCls(grid=default_grid, block=default_block)
    rb.bind_bag(Bag({"grid": grid}))
    for name in ("rec", "bid", "rec_jump"):
        rb.add_data(name, None)

    rb.add_kernel(basin_id_init, data_handle_ref=("bid",))
    rb.add_kernel(copy_field, data_handle_ref=("rec", "rec_jump"))
    rb.begin_repeat(times=logn + 1)
    rb.add_kernel(propagate_basin_iter, data_handle_ref=("rec_jump",))
    rb.end_repeat()
    rb.add_kernel(propagate_basin_final, data_handle_ref=("bid", "rec_jump"))

    return rb, kernels


def build_basin_labelling_optimized(RoutineBuilderCls, KernelCls, *, grid, n_flat: int):
    """
    RoutineBuilder for optimized basin labelling - the closure backends'
    single label_basins_walk launch split into three real launches (copy,
    path-halving, bid finalize), since the path-halving phase needs every
    thread's copy to have landed first, and the finalize phase needs every
    thread's path-halving to have converged first - see this module's own
    section docstring.

    Data names: "rec", "rec_jump", "bid".

    Author: B.G (07/2026)
    """
    default_grid, default_block = _launch_dims(n_flat)
    t = f"pbo{new_uid()}"

    walk_copy = KernelCls().ingest(
        f"""
__global__ void {t}_walk_copy(const int* rec, int* rec_jump) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    rec_jump[i] = rec[i];
}}
"""
    )
    walk_halving = KernelCls().ingest(
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
    walk_finalize = KernelCls().bind("grid", grid).ingest(
        f"""
__global__ void {t}_walk_finalize(const int* rec_jump, int* bid) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    int root = rec_jump[i];
    bid[i] = $grid.can_out(root)$ ? 0 : root + 1;
}}
"""
    )

    kernels = {"walk_copy": walk_copy, "walk_halving": walk_halving, "walk_finalize": walk_finalize}

    rb = RoutineBuilderCls(grid=default_grid, block=default_block)
    rb.bind_bag(Bag({"grid": grid}))
    for name in ("rec", "rec_jump", "bid"):
        rb.add_data(name, None)

    rb.add_kernel(walk_copy, data_handle_ref=("rec", "rec_jump"))
    rb.add_kernel(walk_halving, data_handle_ref=("rec_jump",))
    rb.add_kernel(walk_finalize, data_handle_ref=("rec_jump", "bid"))

    return rb, kernels


def build_saddlesort(RoutineBuilderCls, KernelCls, HelperCls, *, grid, bitpack, n_flat: int):
    """
    RoutineBuilder for the six saddlesort passes - see
    _closure_blocks.build_saddlesort for the step sequence, and this
    module's own bitpack-mirroring notes. `bitpack` is the
    {"pack","unpack_value","unpack_index"} dict from ops.make_bitpack built
    for "cupy".

    Data names: "bid", "z", "z_prime", "is_border", "basin_saddle",
    "basin_saddlenode", "outlet".

    Author: B.G (07/2026)
    """
    default_grid, default_block = _launch_dims(n_flat)
    NN = grid.n_neighbours.get()
    pack = bitpack["pack"]
    unpack_value = bitpack["unpack_value"]
    unpack_index = bitpack["unpack_index"]
    atomic_min_ll = build_atomic_min_ll(HelperCls)
    t = f"pss{new_uid()}"

    border_zprime = KernelCls().bind("grid", grid).ingest(
        f"""
__global__ void {t}_border_zprime(const int* bid, const float* z, float* z_prime, unsigned char* is_border) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    if ($grid.can_out(i)$) {{
        z_prime[i] = z[i];
        return;
    }}
    is_border[i] = 0;
    z_prime[i] = 1e9f;
    float zn = 1e9f;
    int nk = {NN};
    for (int k = 0; k < nk; k++) {{
        int j = $grid.neighbour(i, k)$;
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
    init_saddle_outlet = KernelCls().bind("pack", pack).ingest(
        f"""
__global__ void {t}_init_saddle_outlet(long long* basin_saddle, long long* outlet, int* basin_saddlenode) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    long long invalid = $pack(1e8, 42)$;
    basin_saddle[i] = invalid;
    outlet[i] = invalid;
    basin_saddlenode[i] = -1;
}}
"""
    )
    atomic_min_saddle = (
        KernelCls()
        .bind("grid", grid)
        .bind("pack", pack)
        .bind("atomic_min_ll", atomic_min_ll)
        .ingest(
            f"""
__global__ void {t}_atomic_min_saddle(const int* bid, const unsigned char* is_border, const float* z_prime, long long* basin_saddle) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    if (!is_border[i]) return;
    long long invalid = $pack(1e8, 42)$;
    int tbid = bid[i];
    long long res = invalid;
    int nk = {NN};
    for (int k = 0; k < nk; k++) {{
        int j = $grid.neighbour(i, k)$;
        if (j != -1 && bid[j] != tbid) {{
            long long candidate = $pack(z_prime[i], bid[j])$;
            res = (candidate < res) ? candidate : res;
        }}
    }}
    if (res != invalid) {{
        $atomic_min_ll(&basin_saddle[tbid], res)$;
    }}
}}
"""
        )
    )
    find_saddlenode = (
        KernelCls()
        .bind("grid", grid)
        .bind("unpack_value", unpack_value)
        .bind("unpack_index", unpack_index)
        .ingest(
            f"""
__global__ void {t}_find_saddlenode(const int* bid, const unsigned char* is_border, const float* z_prime,
                                     const long long* basin_saddle, int* basin_saddlenode) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    if (!is_border[i] || bid[i] == 0) return;
    long long packed = basin_saddle[bid[i]];
    float target_z = $unpack_value(packed)$;
    int target_b = $unpack_index(packed)$;
    int is_here = 0;
    int nk = {NN};
    for (int k = 0; k < nk; k++) {{
        int j = $grid.neighbour(i, k)$;
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
        KernelCls()
        .bind("grid", grid)
        .bind("pack", pack)
        .bind("atomic_min_ll", atomic_min_ll)
        .ingest(
            f"""
__global__ void {t}_atomic_min_outlet(const int* bid, const long long* basin_saddle, const int* basin_saddlenode,
                                       const float* z, long long* outlet) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    long long invalid = $pack(1e8, 42)$;
    if (i == 0 || basin_saddle[i] == invalid) return;
    int node = basin_saddlenode[i];
    float tz = 1e9f;
    int rec_out = -1;
    int nk = {NN};
    for (int k = 0; k < nk; k++) {{
        int j = $grid.neighbour(node, k)$;
        if (j != -1 && bid[j] != i && tz > z[j]) {{
            tz = z[j];
            rec_out = j;
        }}
    }}
    if (rec_out > -1) {{
        long long candidate = $pack(tz, rec_out)$;
        $atomic_min_ll(&outlet[i], candidate)$;
    }}
}}
"""
        )
    )
    break_cycle = KernelCls().bind("pack", pack).bind("unpack_index", unpack_index).ingest(
        f"""
__global__ void {t}_break_cycle(const int* bid, long long* outlet, long long* basin_saddle, int* basin_saddlenode) {{
    int bid_d = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid_d >= {n_flat}) return;
    long long invalid = $pack(1e8, 42)$;
    if (bid_d == 0 || outlet[bid_d] == invalid) return;
    int rec_out = $unpack_index(outlet[bid_d])$;
    int bid_d_prime = bid[rec_out];
    if (bid_d_prime == 0) return;
    int rec_out_prime = $unpack_index(outlet[bid_d_prime])$;
    int bid_d_prime_prime = bid[rec_out_prime];
    if (bid_d_prime_prime == bid_d && bid_d_prime < bid_d) {{
        outlet[bid_d] = invalid;
        basin_saddle[bid_d] = invalid;
        basin_saddlenode[bid_d] = -1;
    }}
}}
"""
    )

    kernels = {
        "border_zprime": border_zprime,
        "init_saddle_outlet": init_saddle_outlet,
        "atomic_min_saddle": atomic_min_saddle,
        "find_saddlenode": find_saddlenode,
        "atomic_min_outlet": atomic_min_outlet,
        "break_cycle": break_cycle,
    }

    rb = RoutineBuilderCls(grid=default_grid, block=default_block)
    rb.bind_bag(Bag({"grid": grid, "pack": pack, "unpack_value": unpack_value, "unpack_index": unpack_index, "atomic_min_ll": atomic_min_ll}))
    for name in ("bid", "z", "z_prime", "is_border", "basin_saddle", "basin_saddlenode", "outlet"):
        rb.add_data(name, None)

    rb.add_kernel(border_zprime, data_handle_ref=("bid", "z", "z_prime", "is_border"))
    rb.add_kernel(init_saddle_outlet, data_handle_ref=("basin_saddle", "outlet", "basin_saddlenode"))
    rb.add_kernel(atomic_min_saddle, data_handle_ref=("bid", "is_border", "z_prime", "basin_saddle"))
    rb.add_kernel(find_saddlenode, data_handle_ref=("bid", "is_border", "z_prime", "basin_saddle", "basin_saddlenode"))
    rb.add_kernel(atomic_min_outlet, data_handle_ref=("bid", "basin_saddle", "basin_saddlenode", "z", "outlet"))
    rb.add_kernel(break_cycle, data_handle_ref=("bid", "outlet", "basin_saddle", "basin_saddlenode"))

    return rb, kernels


def build_reroute_carve_vanilla(RoutineBuilderCls, KernelCls, *, bitpack, copy_field, n_flat: int, logn: int):
    """
    RoutineBuilder for carve+vanilla reroute - see
    _closure_blocks.build_reroute_carve_vanilla for the buffer roles
    (`rec_jump` here is finalise's original, unjumped snapshot, not the
    pointer-jumped result - same note applies). init_reroute_carve,
    iteration_reroute_carve and finalise_reroute_carve are each further
    split into several real launches here (this module's own section
    docstring explains why); the closure backends keep each as one kernel.

    Data names: "rec", "rec_work", "rec_jump", "tag", "tag_alt", "bid",
    "basin_saddlenode", "outlet", "rerouted".

    Author: B.G (07/2026)
    """
    default_grid, default_block = _launch_dims(n_flat)
    pack = bitpack["pack"]
    unpack_index = bitpack["unpack_index"]
    t = f"prc{new_uid()}"

    init_reset_tag = KernelCls().ingest(
        f"""
__global__ void {t}_init_reset_tag(unsigned char* tag) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    tag[i] = 0;
}}
"""
    )
    init_scatter_tag = KernelCls().ingest(
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
    init_copy_tag_alt = KernelCls().ingest(
        f"""
__global__ void {t}_init_copy_tag_alt(const unsigned char* tag, unsigned char* tag_alt) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    tag_alt[i] = tag[i];
}}
"""
    )
    iter_build_work = KernelCls().ingest(
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
    iter_jump = KernelCls().ingest(
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
    finalise_reset_rec = KernelCls().ingest(
        f"""
__global__ void {t}_finalise_reset_rec(int* rec, const int* rec_orig) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    rec[i] = rec_orig[i];
}}
"""
    )
    finalise_reverse = KernelCls().ingest(
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
    finalise_outlet = KernelCls().bind("pack", pack).bind("unpack_index", unpack_index).ingest(
        f"""
__global__ void {t}_finalise_outlet(int* rec, const long long* outlet, const int* saddlenode, unsigned char* rerouted) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    long long invalid = $pack(1e8, 42)$;
    if (outlet[i] != invalid) {{
        int node = $unpack_index(outlet[i])$;
        rec[saddlenode[i]] = node;
        rerouted[saddlenode[i]] = 1;
    }}
}}
"""
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

    rb = RoutineBuilderCls(grid=default_grid, block=default_block)
    rb.bind_bag(Bag({"pack": pack, "unpack_index": unpack_index}))
    for name in ("rec", "rec_work", "rec_jump", "tag", "tag_alt", "bid", "basin_saddlenode", "outlet", "rerouted"):
        rb.add_data(name, None)

    rb.add_kernel(init_reset_tag, data_handle_ref=("tag",))
    rb.add_kernel(init_scatter_tag, data_handle_ref=("tag", "basin_saddlenode"))
    rb.add_kernel(init_copy_tag_alt, data_handle_ref=("tag", "tag_alt"))
    rb.add_kernel(copy_field, data_handle_ref=("rec_work", "rec"))
    rb.add_kernel(copy_field, data_handle_ref=("rec_work", "rec_jump"))
    rb.begin_repeat(times=logn + 1)
    rb.add_kernel(iter_build_work, data_handle_ref=("tag", "tag_alt", "rec", "rec_work", "bid"))
    rb.add_kernel(iter_jump, data_handle_ref=("tag", "tag_alt", "rec", "rec_work", "bid"))
    rb.end_repeat()
    rb.add_kernel(finalise_reset_rec, data_handle_ref=("rec", "rec_jump"))
    rb.add_kernel(finalise_reverse, data_handle_ref=("rec", "rec_jump", "tag", "rerouted"))
    rb.add_kernel(finalise_outlet, data_handle_ref=("rec", "outlet", "basin_saddlenode", "rerouted"))
    rb.add_kernel(copy_field, data_handle_ref=("rec", "rec_work"))

    return rb, kernels


def build_reroute_carve_optimized(KernelCls, *, bitpack, n_flat: int):
    """
    carve_basins_serial KernelBuilder - one launch, one serial thread per
    basin; see _closure_blocks.build_reroute_carve_optimized. Node-disjoint
    chains across basins mean no cross-thread dependency at all, so this
    needs no splitting the way the vanilla carve routine does.

    Data args (rec, basin_saddlenode, outlet).

    Author: B.G (07/2026)
    """
    pack = bitpack["pack"]
    unpack_index = bitpack["unpack_index"]
    t = f"pco{new_uid()}"
    return (
        KernelCls()
        .bind("pack", pack)
        .bind("unpack_index", unpack_index)
        .ingest(
            f"""
__global__ void {t}_carve_basins_serial(int* rec, const int* basin_saddlenode, const long long* outlet) {{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= {n_flat}) return;
    long long invalid = $pack(1e8, 42)$;
    int s = basin_saddlenode[b];
    if (s == -1 || outlet[b] == invalid) return;
    int out_node = $unpack_index(outlet[b])$;
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


def build_reroute_jump(RoutineBuilderCls, KernelCls, *, bitpack, n_flat: int):
    """
    RoutineBuilder for reroute_jump - split into a reset launch and the
    jump launch itself, since the jump phase writes `rerouted[i - 1]` from
    thread `i`, a cell a *different* thread's reset zeroed - see this
    module's own section docstring. The closure backends keep this as one
    two-loop kernel.

    The write is deliberately `rec[i - 1]`, not `rec[i]` - see
    _closure_blocks.build_reroute_jump's docstring for why; ported exactly.

    Data names: "rec", "outlet", "rerouted".

    Author: B.G (07/2026)
    """
    default_grid, default_block = _launch_dims(n_flat)
    pack = bitpack["pack"]
    unpack_index = bitpack["unpack_index"]
    t = f"prj{new_uid()}"

    reset_rerouted = KernelCls().ingest(
        f"""
__global__ void {t}_reset_rerouted(unsigned char* rerouted) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    rerouted[i] = 0;
}}
"""
    )
    jump = KernelCls().bind("pack", pack).bind("unpack_index", unpack_index).ingest(
        f"""
__global__ void {t}_jump(int* rec, const long long* outlet, unsigned char* rerouted) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    long long invalid = $pack(1e8, 42)$;
    if (outlet[i] != invalid) {{
        int rrec = $unpack_index(outlet[i])$;
        rec[i - 1] = rrec;
        rerouted[i - 1] = 1;
    }}
}}
"""
    )

    kernels = {"reset_rerouted": reset_rerouted, "jump": jump}

    rb = RoutineBuilderCls(grid=default_grid, block=default_block)
    rb.bind_bag(Bag({"pack": pack, "unpack_index": unpack_index}))
    for name in ("rec", "outlet", "rerouted"):
        rb.add_data(name, None)

    rb.add_kernel(reset_rerouted, data_handle_ref=("rerouted",))
    rb.add_kernel(jump, data_handle_ref=("rec", "outlet", "rerouted"))

    return rb, kernels


def build_depression_counter(KernelCls, *, grid, n_flat: int):
    """
    depression_counter KernelBuilder, data args (rec, ndep) - unlike the
    closure backends' single (rec,) arg (see
    _closure_blocks.build_depression_counter): a Parameter reached only
    through $...$ get() spans is registered read-only (`const T*`) in the
    constant block (see cupy_backend.py's _SpanParser._register_ptr, which
    only flips a Parameter to writable on a set_node span), so atomicAdd
    into it needs the raw pointer as an ordinary kernel argument instead.
    `ndep` is `ndep_p.get().data` - the caller passes it positionally, same
    as `rec`. The caller must reset `ndep_p` to 0 (`.set(0)`) before each
    launch.

    Author: B.G (07/2026)
    """
    t = f"pdc{new_uid()}"
    return KernelCls().bind("grid", grid).ingest(
        f"""
__global__ void {t}_depression_counter(const int* rec, int* ndep) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    if (rec[i] == i && !($grid.can_out(i)$)) {{
        atomicAdd(ndep, 1);
    }}
}}
"""
    )
