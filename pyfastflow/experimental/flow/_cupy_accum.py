"""
cupy (CUDA source) block templates behind make_accumulation: the ping-pong
src helpers, and the three accumulation methods ("atomic", "rake_compress",
"pointer_jump_push").

Split out of a single _cupy_blocks.py that used to hold every flow algorithm
- see _cupy_receivers.py/_cupy_depressions.py/_cupy_reconstruct.py for the
others. build_rake_compress/build_pointer_jump_push build a
CupyRoutineBuilder; cupy has no source fusion (see cupy_backend.py) -
captured=True (the default) launches every step once, for real, then replays
the same launch sequence off a CUDA graph, so the barrier every round needs
between steps is just the ordinary stream ordering CUDA already gives
consecutive launches; there is nothing here equivalent to the closure
backend's fused-vs-split() choice.

Author: B.G (07/2026)
"""

from ..core.context.backends import helper_need
from ..core.context.builder import KernelBuilder
from ..core.context.need import Kind, Need
from ..core.pool.base import new_uid


def build_ping_pong_helpers(HelperCls, *, iteration_need: Need):
    """
    get_src(src, tid)/update_src(src, tid, flip) HelperBuilders - same
    sign/magnitude encoding as pyfastflow/general_algorithms/pingpong.py's
    getSrc/updateSrc, reading `iteration_need` internally instead of taking
    iteration as a call argument.

    `iteration_need` is the caller's already-bound `Need("iteration_p",
    kind=Kind.PARAM)` (see make_accumulation) - a fresh, internally-named
    `Need("ITER", ...)`, matching what these templates' bodies actually
    reference, is bound here to the same underlying Parameter and declared
    on both helpers via `.need()`.

    Author: B.G (07/2026)
    """
    t = f"pp{new_uid()}"
    iter_need = Need("ITER", kind=Kind.PARAM, dtype=iteration_need.dtype, modes=iteration_need.modes)
    iter_need.bind(iteration_need.value)
    get_src = HelperCls(strict_needs=True).need(iter_need).ingest(
        f"""
__device__ int {t}_get_src(const int* src, int tid) {{
    int entry = src[tid];
    int it = $ITER.get(0)$;
    int flip = entry < 0;
    if (abs(entry) == (it + 1)) flip = !flip;
    return flip;
}}
"""
    )
    update_src = HelperCls(strict_needs=True).need(iter_need).ingest(
        f"""
__device__ void {t}_update_src(int* src, int tid, int flip) {{
    int it = $ITER.get(0)$;
    src[tid] = (flip ? 1 : -1) * (it + 1);
}}
"""
    )
    return get_src, update_src


def build_atomic(*, n_flat: int):
    """
    Two FrozenKernels (new builder/frozen/bound stack - ../core/context/
    builder.py, frozen.py, bound.py), data args (q,) and (rec, q): "q_init"
    (q[i] = SOURCE.get(i)) must be run, and finish, before "accum" (the
    atomic descent). Two real launches rather than one, unlike the closure
    backends' single KernelBuilder (see _closure_accum.py's build_atomic) -
    a single CUDA __global__ has no portable grid-wide barrier the way two
    consecutive top-level Taichi/Quadrants for-loops do, so without a second
    launch a thread could atomicAdd into q[j] before node j's own thread has
    run its q[j] = SOURCE.get(j) initialization.

    `SOURCE` is each kernel's own wired PARAM slot (any mode) - a caller
    binds a Parameter there, on each, after `.build()`; there is no Need
    indirection in this stack. `q` stays plain DATA (native CUDA
    `atomicAdd`, no `ctx.bk` involved - cupy keeps native C, see bk.py's own
    module docstring).

    Author: B.G (08/2026)
    """
    t = f"pa{new_uid()}"
    q_init = (
        KernelBuilder()
        .wire_param("SOURCE")
        .wire_data("q")
        .ingest(
            f"""
extern "C" __global__ void {t}_q_init(float* q) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    q[i] = $ctx.SOURCE.get(i)$;
}}
"""
        )
    )
    # wi is re-read from `SOURCE`, not from q[i]: q[i] is a live accumulation
    # target every other thread may already be atomic-adding into by the
    # time this thread runs (thread order across blocks is not guaranteed),
    # so reading q[i] here would race against those writes and silently
    # inflate wi with contributions that arrived early - exactly the bug an
    # earlier version of this kernel had (q[i] as a shortcut for wi, to avoid
    # re-binding `source`), caught by a max_abs deviation of ~1e23 at
    # n_flat=1e6 in _verify_accum.py. Matches legacy
    # accum_downstream_atomic_kernel and the closure-backend port, which
    # both re-read the weight function/Parameter directly for this reason.
    accum = (
        KernelBuilder()
        .wire_param("SOURCE")
        .wire_data("rec")
        .wire_data("q")
        .ingest(
            f"""
extern "C" __global__ void {t}_accum_downstream_atomic(const int* rec, float* q) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    if (rec[i] == i) return;
    float wi = $ctx.SOURCE.get(i)$;
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
    )
    return {"q_init": q_init, "accum": accum}


def build_rake_compress(
    RoutineBuilderCls,
    KernelCls,
    HelperCls,
    *,
    grid,
    source: Need,
    iteration_p: Need,
    logn: int,
    n_flat: int,
):
    """
    CupyRoutineBuilder for the rake-and-compress accumulation, plus the
    KernelBuilders it is made of - see _closure_accum.py's
    build_rake_compress for the step sequence and the iteration off-by-one.

    The routine's default launch grid/block is sized for `n_flat` threads -
    every step here reads/writes an n_flat-sized index space except the
    single-thread iteration bookkeeping steps, which override grid=(1,)/
    block=(1,) at add_kernel time.

    `source`/`iteration_p` are the caller's already-bound `Need("source",
    kind=Kind.PARAM)`/`Need("iteration_p", kind=Kind.PARAM)` (see
    make_accumulation) - fresh, internally-named `Need("source", ...)`/
    `Need("ITER", ...)`, matching what these templates' bodies actually
    reference, are bound here to the same underlying Parameters and declared
    on every KernelBuilder/HelperBuilder that needs them via `.need()`.
    Every KernelBuilder/HelperBuilder is constructed strict_needs=True;
    `_GETSRC`/`_UPDATESRC` go through helper_need. The returned RoutineBuilder
    is never given a bind_bag() bag: every step's own dependencies are
    already fully resolved via Need by the time it is built, so there is
    nothing left for a routine-level bag to supply (see routine.py,
    RoutineBuilder._validate).

    Author: B.G (07/2026)
    """
    NN = grid.n_neighbours.get()
    block_size = 256
    default_grid, default_block = ((n_flat + block_size - 1) // block_size,), (block_size,)

    source_need = Need("source", kind=Kind.PARAM, dtype=source.dtype, modes=source.modes)
    source_need.bind(source.value)
    iter_need = Need("ITER", kind=Kind.PARAM, dtype=iteration_p.dtype, modes=iteration_p.modes)
    iter_need.bind(iteration_p.value)

    get_src, update_src = build_ping_pong_helpers(HelperCls, iteration_need=iteration_p)
    t = f"pr{new_uid()}"

    zero_init = KernelCls(strict_needs=True).ingest(
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
    reset_iteration = KernelCls(strict_needs=True).need(iter_need).ingest(
        f"""
__global__ void {t}_reset_iteration() {{
    $ITER.set_node(0, 0)$;
}}
"""
    )
    bump_iteration = KernelCls(strict_needs=True).need(iter_need).ingest(
        f"""
__global__ void {t}_bump_iteration() {{
    int cur = $ITER.get(0)$;
    $ITER.set_node(0, cur + 1)$;
}}
"""
    )
    decrement_iteration = KernelCls(strict_needs=True).need(iter_need).ingest(
        f"""
__global__ void {t}_decrement_iteration() {{
    int cur = $ITER.get(0)$;
    $ITER.set_node(0, cur - 1)$;
}}
"""
    )
    q_init = KernelCls(strict_needs=True).need(source_need).ingest(
        f"""
__global__ void {t}_q_init(float* q) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    q[i] = $source.get(i)$;
}}
"""
    )
    receivers_to_donors = KernelCls(strict_needs=True).ingest(
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
    getsrc_need = helper_need("_GETSRC", get_src)
    updatesrc_need = helper_need("_UPDATESRC", update_src)
    rake_compress_accum = (
        KernelCls(strict_needs=True)
        .need(getsrc_need)
        .bind("_GETSRC", getsrc_need.value)
        .need(updatesrc_need)
        .bind("_UPDATESRC", updatesrc_need.value)
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
    fuse_accum_buffers = (
        KernelCls(strict_needs=True).need(helper_need("_GETSRC", get_src)).bind("_GETSRC", get_src).ingest(
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
    # No bind_bag() call: "_GETSRC"/"_UPDATESRC" are bound to
    # rake_compress_accum/fuse_accum_buffers' own already-bound helper_need
    # at construction time; "ITER"/"source" arrive via .need() alone,
    # resolved directly by each step's own compile() - nothing left for a
    # routine-level bag to supply (see routine.py, RoutineBuilder._validate).
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
    source: Need,
    rounds: int,
    n_flat: int,
):
    """
    CupyRoutineBuilder for the pointer-jump-push accumulation, plus the
    KernelBuilders it is made of - see _closure_accum.py's
    build_pointer_jump_push for the step sequence and retirement rule.

    The routine's default launch grid/block is sized for `n_flat` threads -
    every step here reads/writes an n_flat-sized index space.

    `source` is the caller's already-bound `Need("source", kind=Kind.PARAM)`
    (see make_accumulation) - a fresh, internally-named `Need("source", ...)`
    is bound here to the same underlying Parameter and declared on q_init via
    `.need()`. Every KernelBuilder is constructed strict_needs=True.

    Author: B.G (07/2026)
    """
    t = f"pj{new_uid()}"
    block_size = 256
    default_grid, default_block = ((n_flat + block_size - 1) // block_size,), (block_size,)

    source_need = Need("source", kind=Kind.PARAM, dtype=source.dtype, modes=source.modes)
    source_need.bind(source.value)

    q_init = KernelCls(strict_needs=True).need(source_need).ingest(
        f"""
__global__ void {t}_q_init(float* q) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    q[i] = $source.get(i)$;
}}
"""
    )
    copy_rec_to_work = KernelCls(strict_needs=True).ingest(
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
    # Quadrants kernel give for free (see _closure_accum.py's equivalent).
    step_copy = KernelCls(strict_needs=True).ingest(
        f"""
__global__ void {t}_step_copy(const float* q_curr, float* q_next) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    q_next[i] = q_curr[i];
}}
"""
    )
    step_core = KernelCls(strict_needs=True).ingest(
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
    # No bind_bag() call: "source" arrives via .need() alone, resolved
    # directly by q_init's own compile(); no step here binds anything else -
    # see build_rake_compress's own equivalent note above.
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


