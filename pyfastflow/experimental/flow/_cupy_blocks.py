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
