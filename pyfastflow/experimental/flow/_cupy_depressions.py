"""
cupy (CUDA source) block templates behind make_depressions/
make_depression_solver: the i64 atomic_min helper, copy_field, both basin
labelling variants, saddlesort, both carve variants, jump reroute, and the
depression counter.

Split out of a single _cupy_blocks.py that used to hold every flow algorithm
- see _cupy_receivers.py/_cupy_accum.py/_cupy_reconstruct.py for the others.
Ported from ../../flow/flow_reroute_kernels.py; `bitpack`'s pack/
unpack_value/unpack_index (ops.make_bitpack) replace legacy's
f32_i32_struct module for the lexicographic (elevation, target-basin) and
(elevation, node) argmins saddlesort's atomic_min passes need. Every array
here (rec, bid, tag, basin_saddle, outlet, ...) is n_flat-sized, basin id =
pit index + 1, so a per-basin array is safely indexed by any node index too -
the same double duty the legacy kernels rely on.

Author: B.G (07/2026)
"""

from ..core.context.backends import make_helper
from ..core.context.bag import Bag
from ..core.pool.base import new_uid


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


