"""
Taichi/Quadrants (closure) block templates behind make_receivers and
make_accumulation.

Same split as ../grid/_closure_blocks.py and ../noise/_closure_blocks.py:
every private block is a plain python def, PICKED - never branched on inside
one function body - by build_receivers() according to the caller's `mode`
("steepest"|"stochastic"), `h_aware` and `diagonal_partition_correction`
flags. The one runtime branch that exists (which diagonal k values get the
sqrt(2) correction) is inside the *corrected* distance helper only - k is
genuine per-call device data, so it cannot be resolved by picking a function
ahead of time; the *uncorrected* variant, used whenever the correction is
off, is a different helper with no branch at all (grid.dist_from_k /
grid.dist_between_nodes themselves, reused as-is).

The receiver kernel body is one of four variants (mode x h_aware), each a
nested def inside build_receivers so it can close over the per-backend data
argument annotation (`ti.template()` vs `qd.Tensor`) the way
../ops/_closure_blocks.py's build_elementwise does - a kernel's data
arguments need a real type annotation chosen at build time, unlike a helper.

The accumulation kernels (build_rake_compress/build_pointer_jump_push/
build_atomic, further down) are nested defs closing over the per-backend
Tensor annotation and over small build-time-constant python ints
(n_neighbours, n_flat) - the same idiom ../ops/_closure_blocks.py's
build_elementwise/build_scan_routine use. capture_template_meta dedents a
nested def's source before parsing it and _fuse_group synthesizes each data
argument's annotation from the bound backend module rather than reading one
out of the AST, so these templates fuse; make_accumulation's caller still
decides fused=True/False when it compiles the returned RoutineBuilder. A
round's cross-buffer dependency on the previous round's writes needs no
barrier fusion could not give it either way: consecutive top-level `for`
loops inside one compiled Taichi/Quadrants kernel are already separate
offloaded tasks launched in order (confirmed empirically - see
make_accumulation's docstring in __init__.py), the same guarantee legacy
pyfastflow/flow/lakeflow.py's saddlesort relies on for its own six
mutually-dependent passes in one hand-written kernel.

Author: B.G (07/2026)
"""

import functools
import math

from ..core.context.backends import make_helper
from ..core.context.bag import Bag

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

    Returns {name: HelperBuilder}.

    Author: B.G (07/2026)
    """
    mk = functools.partial(make_helper, HelperCls)
    sqrt2 = math.sqrt(2.0)

    d8 = grid.n_neighbours.get() == 8
    if diagonal_partition_correction and d8:
        dist_from_k_corrected = mk(_dist_from_k_corrected_tmpl, _GRID=grid, SQRT2=sqrt2)
        dist_between_nodes_corrected = mk(_dist_between_nodes_corrected_tmpl, _GRID=grid, SQRT2=sqrt2)
    else:
        dist_from_k_corrected = grid.dist_from_k
        dist_between_nodes_corrected = grid.dist_between_nodes

    slope_from_values_k = mk(_slope_from_values_k_tmpl, _DISTFROMK=dist_from_k_corrected)
    slope_between_nodes = mk(_slope_between_nodes_tmpl, _DISTBETWEEN=dist_between_nodes_corrected)

    return {
        "dist_from_k_corrected": dist_from_k_corrected,
        "dist_between_nodes_corrected": dist_between_nodes_corrected,
        "slope_from_values_k": slope_from_values_k,
        "slope_between_nodes": slope_between_nodes,
    }


def build_rand_unit(HelperCls, *, seed_p, hash_u32, backend_mod):
    """
    rand_unit(i, k) HelperBuilder, binding the caller-supplied `hash_u32`
    (noise's public hash helper - see ../noise/_closure_blocks.py) rather
    than a private copy.

    Author: B.G (07/2026)
    """
    mk = functools.partial(make_helper, HelperCls)
    return mk(_rand_unit_tmpl, SEED=seed_p, _HASH=hash_u32, _BK=backend_mod)


def _tensor_annotation(backend_mod, backend: str):
    """
    The data-argument annotation a kernel template needs on this closure
    backend: `ti.template()` for Taichi, `qd.Tensor` for Quadrants - mirrors
    ../ops/_closure_blocks.py's _tensor_annotation.

    Author: B.G (07/2026)
    """
    return backend_mod.template() if backend == "taichi" else backend_mod.Tensor


def build_receivers(
    KernelCls,
    HelperCls,
    *,
    backend: str,
    backend_mod,
    grid,
    hash_u32,
    mode: str,
    seed_p,
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
    share the exact same integer hash. Only required when mode="stochastic".

    Returns {name: HelperBuilder/KernelBuilder} - the distance/slope helpers
    plus "receivers", plus "rand_unit" when mode="stochastic".

    Author: B.G (07/2026)
    """
    out = build_distance_slope_helpers(HelperCls, grid=grid, diagonal_partition_correction=diagonal_partition_correction)
    slope = out["slope_from_values_k"]
    T = _tensor_annotation(backend_mod, backend)

    if mode == "stochastic":
        rand_unit = build_rand_unit(HelperCls, seed_p=seed_p, hash_u32=hash_u32, backend_mod=backend_mod)
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

    receivers_builder = KernelCls().bind("_GRID", grid).bind("_SLOPE", slope)
    if mode == "stochastic":
        receivers_builder = receivers_builder.bind("_RAND", out["rand_unit"]).bind("_BK", backend_mod)
    receivers_builder = receivers_builder.ingest(receivers_template)

    out["receivers"] = receivers_builder
    return out


# ---------------------------------------------------------------------------
# accumulation: ping-pong src encoding (getSrc/updateSrc, reading the
# iteration scalar Parameter instead of taking iteration as a call argument)
# ---------------------------------------------------------------------------


def _get_src_tmpl(src, tid):
    entry = src[tid]
    it = _ITER.get(0)
    flip = entry < 0
    flip = (not flip) if (abs(entry) == (it + 1)) else flip
    return flip


def _update_src_tmpl(src, tid, flip):
    it = _ITER.get(0)
    src[tid] = (1 if flip else -1) * (it + 1)


def build_ping_pong_helpers(HelperCls, *, iteration_p):
    """
    get_src(src, tid)/update_src(src, tid, flip) HelperBuilders - same
    sign/magnitude encoding as pyfastflow/general_algorithms/pingpong.py's
    getSrc/updateSrc, reading `iteration_p` internally instead of taking
    iteration as a call argument.

    Author: B.G (07/2026)
    """
    mk = functools.partial(make_helper, HelperCls)
    get_src = mk(_get_src_tmpl, _ITER=iteration_p)
    update_src = mk(_update_src_tmpl, _ITER=iteration_p)
    return get_src, update_src


def build_atomic(KernelCls, *, backend, backend_mod, source, n_flat: int):
    """
    accum_downstream_atomic KernelBuilder, data args (rec, q): q[i] is
    initialized from `source.get(i)`, then every node walks its receiver
    chain to the root atomic-adding its own weight into each downstream
    node. Requires an acyclic receiver graph (run after depression
    handling); the `guard < n_flat` bound makes a cycle degrade the result
    instead of hanging, rather than guaranteeing correctness on one.

    Author: B.G (07/2026)
    """
    T = _tensor_annotation(backend_mod, backend)
    NFLAT = n_flat

    def accum_downstream_atomic_template(rec: T, q: T):
        for i in q:
            q[i] = _SOURCE.get(i)
        for i in rec:
            if rec[i] == i:
                continue
            wi = _SOURCE.get(i)
            j = rec[i]
            guard = 0
            while j != rec[j] and guard < NFLAT:
                _BK.atomic_add(q[j], wi)
                j = rec[j]
                guard += 1
            _BK.atomic_add(q[j], wi)

    return KernelCls().bind("_SOURCE", source).bind("_BK", backend_mod).ingest(accum_downstream_atomic_template)


def build_rake_compress(
    RoutineBuilderCls,
    KernelCls,
    HelperCls,
    *,
    backend: str,
    backend_mod,
    grid,
    source,
    iteration_p,
    logn: int,
):
    """
    RoutineBuilder for the rake-and-compress accumulation, plus the
    KernelBuilders it is made of.

    Steps: zero-init (ndonors, ndonors_alt, src) + reset iteration to 0;
    q[i] = source.get(i); receivers_to_donors (atomic donor-list build);
    begin_repeat(times=logn+1): rake_compress_accum (its own second
    top-level `for` loop bumps the iteration counter by 1 after the rake
    pass, as a separate offloaded task ordered after it - see
    rake_compress_accum_template); end_repeat(); decrement iteration by 1
    (undoes the repeat's last bump, so fuse_accum_buffers reads the same
    iteration value the last rake round used - see make_accumulation's
    docstring on the off-by-one); fuse_accum_buffers.

    Returns (routine_builder, kernel_builders_dict). The routine is
    uncompiled and its data names are placeholders (add_data(name, None)) -
    every real call must pass all of them positionally; see
    make_accumulation's docstring for the exact order.

    Author: B.G (07/2026)
    """
    T = _tensor_annotation(backend_mod, backend)
    NN = grid.n_neighbours.get()
    get_src, update_src = build_ping_pong_helpers(HelperCls, iteration_p=iteration_p)

    def zero_init_template(ndonors: T, ndonors_alt: T, src: T):
        for i in ndonors:
            ndonors[i] = 0
            ndonors_alt[i] = 0
            src[i] = 0

    def reset_iteration_template():
        _ITER.set_node(0, 0)

    def bump_iteration_template():
        _ITER.set_node(0, _ITER.get(0) + 1)

    def decrement_iteration_template():
        _ITER.set_node(0, _ITER.get(0) - 1)

    def q_init_template(q: T):
        for i in q:
            q[i] = _SOURCE.get(i)

    def receivers_to_donors_template(rec: T, donors: T, ndonors: T):
        for tid in rec:
            rcv = rec[tid]
            if rcv != tid:
                old_val = _BK.atomic_add(ndonors[rcv], 1)
                donors[rcv * NN + old_val] = tid

    def rake_compress_accum_template(donors: T, ndonors: T, q: T, src: T, donors_alt: T, ndonors_alt: T, q_alt: T):
        for tid in q:
            flip = _GETSRC(src, tid)

            worked = False
            todo = ndonors[tid] if not flip else ndonors_alt[tid]
            base = tid * NN
            donors_local = _BK.Vector([-1] * NN)
            q_added = 0.0

            i = 0
            while i < todo and i < NN:
                if donors_local[i] == -1:
                    donors_local[i] = donors[base + i] if not flip else donors_alt[base + i]
                did = donors_local[i]

                flip_donor = _GETSRC(src, did)
                ndnr_val = ndonors[did] if not flip_donor else ndonors_alt[did]

                if ndnr_val <= 1:
                    if not worked:
                        q_added = q[tid] if not flip else q_alt[tid]
                    worked = True

                    q_val = q[did] if not flip_donor else q_alt[did]
                    q_added += q_val

                    if ndnr_val == 0:
                        todo -= 1
                        if todo > i:
                            donors_local[i] = donors[base + todo] if not flip else donors_alt[base + todo]
                        i -= 1
                    else:
                        donor_base = did * NN
                        donors_local[i] = donors[donor_base] if not flip_donor else donors_alt[donor_base]
                i += 1

            if worked:
                if flip:
                    ndonors[tid] = todo
                    q[tid] = q_added
                    for j in range(NN):
                        if j < todo:
                            donors[base + j] = donors_local[j]
                else:
                    ndonors_alt[tid] = todo
                    q_alt[tid] = q_added
                    for j in range(NN):
                        if j < todo:
                            donors_alt[base + j] = donors_local[j]
                _UPDATESRC(src, tid, flip)
        for _ in range(1):
            _ITER.set_node(0, _ITER.get(0) + 1)

    def fuse_accum_buffers_template(q: T, src: T, q_alt: T):
        for tid in q:
            if _GETSRC(src, tid):
                q[tid] = q_alt[tid]

    zero_init = KernelCls().ingest(zero_init_template)
    reset_iteration = KernelCls().bind("_ITER", iteration_p).ingest(reset_iteration_template)
    bump_iteration = KernelCls().bind("_ITER", iteration_p).ingest(bump_iteration_template)
    decrement_iteration = KernelCls().bind("_ITER", iteration_p).ingest(decrement_iteration_template)
    q_init = KernelCls().bind("_SOURCE", source).ingest(q_init_template)
    receivers_to_donors = KernelCls().bind("_BK", backend_mod).ingest(receivers_to_donors_template)
    rake_compress_accum = (
        KernelCls()
        .bind("_BK", backend_mod)
        .bind("_GETSRC", get_src)
        .bind("_UPDATESRC", update_src)
        .bind("_ITER", iteration_p)
        .ingest(rake_compress_accum_template)
    )
    fuse_accum_buffers = KernelCls().bind("_GETSRC", get_src).ingest(fuse_accum_buffers_template)

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

    rb = RoutineBuilderCls()
    # rebind() (routine.py, RoutineBuilder._validate) re-resolves every name
    # each step's own KernelBuilder currently binds against this one shared
    # bag - so every name any step bound via .bind() above must be a member
    # here too, under the same name.
    rb.bind_bag(
        Bag(
            {
                "_ITER": iteration_p,
                "_SOURCE": source,
                "_BK": backend_mod,
                "_GETSRC": get_src,
                "_UPDATESRC": update_src,
            }
        )
    )
    for name in ("rec", "q", "donors", "ndonors", "donors_alt", "ndonors_alt", "q_alt", "src"):
        rb.add_data(name, None)

    rb.add_kernel(zero_init, data_handle_ref=("ndonors", "ndonors_alt", "src"))
    rb.add_kernel(reset_iteration, data_handle_ref=())
    rb.add_kernel(q_init, data_handle_ref=("q",))
    rb.add_kernel(receivers_to_donors, data_handle_ref=("rec", "donors", "ndonors"))
    rb.begin_repeat(times=logn + 1)
    rb.add_kernel(rake_compress_accum, data_handle_ref=("donors", "ndonors", "q", "src", "donors_alt", "ndonors_alt", "q_alt"))
    rb.end_repeat()
    rb.add_kernel(decrement_iteration, data_handle_ref=())
    rb.add_kernel(fuse_accum_buffers, data_handle_ref=("q", "src", "q_alt"))

    return rb, kernels


def build_pointer_jump_push(
    RoutineBuilderCls,
    KernelCls,
    *,
    backend: str,
    backend_mod,
    source,
    rounds: int,
):
    """
    RoutineBuilder for the pointer-jump-push accumulation, plus the
    KernelBuilders it is made of.

    Steps: q[i] = source.get(i); copy rec -> work (so round 0 is not a
    special case and rec itself is never written); begin_repeat(times=
    `rounds`, already rounded to even by the caller):
    accum_pointer_jump_push_step(work, work2, q, q_work), swap("work",
    "work2"), swap("q", "q_work"); end_repeat(). An even round count makes
    the net swap permutation the identity, so the result always lands back
    in "work"/"q" without a host-side conditional copy-back.

    Retirement rule (kept exactly, not restructured): when a node's parent
    is a sink in the current jumped graph (grandparent == parent), the node
    pushes once more and then points at itself, so it never re-pushes a
    growing sum into the sink.

    Author: B.G (07/2026)
    """
    T = _tensor_annotation(backend_mod, backend)

    def q_init_template(q: T):
        for i in q:
            q[i] = _SOURCE.get(i)

    def copy_rec_to_work_template(rec: T, work: T):
        for i in rec:
            work[i] = rec[i]

    def accum_pointer_jump_push_step_template(rec_curr: T, rec_next: T, q_curr: T, q_next: T):
        for i in q_next:
            q_next[i] = q_curr[i]
        for i in rec_curr:
            parent = rec_curr[i]
            rec_next[i] = parent
            if parent != i:
                wi = q_curr[i]
                if wi != 0.0:
                    _BK.atomic_add(q_next[parent], wi)
                grandparent = rec_curr[parent]
                rec_next[i] = i if grandparent == parent else grandparent

    q_init = KernelCls().bind("_SOURCE", source).ingest(q_init_template)
    copy_rec_to_work = KernelCls().ingest(copy_rec_to_work_template)
    step = KernelCls().bind("_BK", backend_mod).ingest(accum_pointer_jump_push_step_template)

    kernels = {"q_init": q_init, "copy_rec_to_work": copy_rec_to_work, "accum_pointer_jump_push_step": step}

    rb = RoutineBuilderCls()
    rb.bind_bag(Bag({"_SOURCE": source, "_BK": backend_mod}))
    for name in ("rec", "work", "work2", "q", "q_work"):
        rb.add_data(name, None)

    rb.add_kernel(q_init, data_handle_ref=("q",))
    rb.add_kernel(copy_rec_to_work, data_handle_ref=("rec", "work"))
    rb.begin_repeat(times=rounds)
    rb.add_kernel(step, data_handle_ref=("work", "work2", "q", "q_work"))
    rb.add_swap("work", "work2")
    rb.add_swap("q", "q_work")
    rb.end_repeat()

    return rb, kernels


# ---------------------------------------------------------------------------
# depression handling: basin labelling, saddlesort, reroute - ported from
# ../../flow/flow_reroute_kernels.py, `_PACK`/`_UNPACK_VALUE`/`_UNPACK_INDEX`
# bound from ops.make_bitpack in place of legacy's f32_i32_struct module.
# Every array here (rec, bid, tag, basin_saddle, outlet, ...) is n_flat-sized,
# basin id = pit index + 1, so a per-basin array is safely indexed by any
# node index too - the same double duty the legacy kernels rely on.
# ---------------------------------------------------------------------------


def build_copy_field(KernelCls, *, backend, backend_mod):
    """
    dst[i] = src[i] over a whole flat buffer - the generic copy used
    everywhere a depression pass needs one buffer snapshotted into another
    (rec -> rec_jump, rec_work -> rec, ...), reused across every routine that
    needs it rather than rebuilt per call site.

    Author: B.G (07/2026)
    """
    T = _tensor_annotation(backend_mod, backend)

    def copy_field_template(src: T, dst: T):
        for i in src:
            dst[i] = src[i]

    return KernelCls().ingest(copy_field_template)


def build_basin_id_init(KernelCls, *, backend, backend_mod, grid):
    """
    bid[i] = 0 on a can_out node, i+1 otherwise - the seed for vanilla basin
    labelling's pointer-jump propagation.

    Author: B.G (07/2026)
    """
    T = _tensor_annotation(backend_mod, backend)

    def basin_id_init_template(bid: T):
        for i in bid:
            bid[i] = 0 if _GRID.can_out(i) else (i + 1)

    return KernelCls().bind("_GRID", grid).ingest(basin_id_init_template)


def build_propagate_basin_iter(KernelCls, *, backend, backend_mod):
    """
    One pointer-jump step over `rec_jump`, halving path length to the root
    each call.

    Author: B.G (07/2026)
    """
    T = _tensor_annotation(backend_mod, backend)

    def propagate_basin_iter_template(rec_jump: T):
        for i in rec_jump:
            if rec_jump[i] != rec_jump[rec_jump[i]]:
                rec_jump[i] = rec_jump[rec_jump[i]]

    return KernelCls().ingest(propagate_basin_iter_template)


def build_propagate_basin_final(KernelCls, *, backend, backend_mod):
    """
    bid[i] = bid[root(i)] - finalizes vanilla basin labelling once
    `rec_jump` has been pointer-jumped down to (near-)roots.

    Author: B.G (07/2026)
    """
    T = _tensor_annotation(backend_mod, backend)

    def propagate_basin_final_template(bid: T, rec_jump: T):
        for i in bid:
            bid[i] = bid[rec_jump[i]]

    return KernelCls().ingest(propagate_basin_final_template)


def build_basin_labelling_vanilla(RoutineBuilderCls, KernelCls, *, backend, backend_mod, grid, copy_field, logn: int):
    """
    RoutineBuilder for vanilla basin labelling: basin_id_init(bid);
    copy_field(rec -> rec_jump); begin_repeat(logn+1):
    propagate_basin_iter(rec_jump); end_repeat();
    propagate_basin_final(bid, rec_jump).

    `copy_field` is the shared KernelBuilder from build_copy_field, reused
    here rather than rebuilt. Returns (routine_builder, kernels_dict);
    kernels_dict holds "basin_id_init", "propagate_basin_iter",
    "propagate_basin_final" (not "copy_field" - that one is the caller's own
    to keep track of, shared across every routine that needs a copy).

    Data names (add_data placeholders, positional order of a real call):
    "rec", "bid", "rec_jump".

    Author: B.G (07/2026)
    """
    basin_id_init = build_basin_id_init(KernelCls, backend=backend, backend_mod=backend_mod, grid=grid)
    propagate_basin_iter = build_propagate_basin_iter(KernelCls, backend=backend, backend_mod=backend_mod)
    propagate_basin_final = build_propagate_basin_final(KernelCls, backend=backend, backend_mod=backend_mod)

    kernels = {
        "basin_id_init": basin_id_init,
        "propagate_basin_iter": propagate_basin_iter,
        "propagate_basin_final": propagate_basin_final,
    }

    rb = RoutineBuilderCls()
    rb.bind_bag(Bag({"_GRID": grid}))
    for name in ("rec", "bid", "rec_jump"):
        rb.add_data(name, None)

    rb.add_kernel(basin_id_init, data_handle_ref=("bid",))
    rb.add_kernel(copy_field, data_handle_ref=("rec", "rec_jump"))
    rb.begin_repeat(times=logn + 1)
    rb.add_kernel(propagate_basin_iter, data_handle_ref=("rec_jump",))
    rb.end_repeat()
    rb.add_kernel(propagate_basin_final, data_handle_ref=("bid", "rec_jump"))

    return rb, kernels


def build_basin_labelling_optimized(KernelCls, *, backend, backend_mod, grid, n_flat: int):
    """
    label_basins_walk KernelBuilder - one launch: copy rec -> rec_jump,
    per-thread path-halving to the root under a `guard < n_flat` bound
    (races are benign - entries only ever move rootward), then
    bid[i] = 0 if can_out(root) else root + 1.

    Data args (rec, rec_jump, bid).

    Author: B.G (07/2026)
    """
    T = _tensor_annotation(backend_mod, backend)
    NFLAT = n_flat

    def label_basins_walk_template(rec: T, rec_jump: T, bid: T):
        for i in rec_jump:
            rec_jump[i] = rec[i]
        for i in rec_jump:
            guard = 0
            while rec_jump[i] != rec_jump[rec_jump[i]] and guard < NFLAT:
                rec_jump[i] = rec_jump[rec_jump[i]]
                guard += 1
        for i in bid:
            root = rec_jump[i]
            bid[i] = 0 if _GRID.can_out(root) else root + 1

    return KernelCls().bind("_GRID", grid).ingest(label_basins_walk_template)


def build_saddlesort(RoutineBuilderCls, KernelCls, *, backend, backend_mod, grid, bitpack):
    """
    RoutineBuilder for the six saddlesort passes (see the module docstring
    and todo.md's ordered list): border/z_prime detection, saddle/outlet/
    saddlenode init, bitpacked atomic-min saddle search, saddlenode
    identification, bitpacked atomic-min outlet search, basin-graph
    2-cycle break. Shared unchanged by both `method`s.

    `bitpack` is the {"pack", "unpack_value", "unpack_index"} dict from
    ops.make_bitpack, replacing legacy's f32_i32_struct helpers.

    Returns (routine_builder, kernels_dict) - kernels_dict keys
    "border_zprime", "init_saddle_outlet", "atomic_min_saddle",
    "find_saddlenode", "atomic_min_outlet", "break_cycle".

    Data names: "bid", "z", "z_prime", "is_border", "basin_saddle",
    "basin_saddlenode", "outlet".

    Author: B.G (07/2026)
    """
    T = _tensor_annotation(backend_mod, backend)
    NN = grid.n_neighbours.get()
    pack = bitpack["pack"]
    unpack_value = bitpack["unpack_value"]
    unpack_index = bitpack["unpack_index"]

    def border_zprime_template(bid: T, z: T, z_prime: T, is_border: T):
        for i in z:
            if _GRID.can_out(i):
                z_prime[i] = z[i]
                continue
            is_border[i] = 0
            z_prime[i] = 1e9
            zn = 1e9
            for k in range(NN):
                j = _GRID.neighbour(i, k)
                if j != -1 and bid[j] != bid[i]:
                    is_border[i] = 1
                    zn = _BK.min(zn, z[j])
            if is_border[i]:
                z_prime[i] = _BK.max(z[i], zn)

    def init_saddle_outlet_template(basin_saddle: T, outlet: T, basin_saddlenode: T):
        invalid_i = _PACK(1e8, 42)
        for i in basin_saddle:
            basin_saddle[i] = invalid_i
            outlet[i] = invalid_i
            basin_saddlenode[i] = -1

    def atomic_min_saddle_template(bid: T, is_border: T, z_prime: T, basin_saddle: T):
        invalid_a = _PACK(1e8, 42)
        for i in bid:
            if not is_border[i]:
                continue
            tbid = bid[i]
            res = invalid_a
            for k in range(NN):
                j = _GRID.neighbour(i, k)
                if j != -1 and bid[j] != tbid:
                    candidate = _PACK(z_prime[i], bid[j])
                    res = _BK.min(res, candidate)
            if res != invalid_a:
                _BK.atomic_min(basin_saddle[tbid], res)

    def find_saddlenode_template(bid: T, is_border: T, z_prime: T, basin_saddle: T, basin_saddlenode: T):
        for i in bid:
            if not is_border[i] or bid[i] == 0:
                continue
            target_z = _UNPACKVALUE(basin_saddle[bid[i]])
            target_b = _UNPACKINDEX(basin_saddle[bid[i]])
            is_here = False
            for k in range(NN):
                j = _GRID.neighbour(i, k)
                if j != -1 and bid[j] == target_b and z_prime[i] == target_z:
                    is_here = True
            if is_here:
                basin_saddlenode[bid[i]] = i

    def atomic_min_outlet_template(bid: T, basin_saddle: T, basin_saddlenode: T, z: T, outlet: T):
        invalid_o = _PACK(1e8, 42)
        for i in bid:
            if i == 0 or basin_saddle[i] == invalid_o:
                continue
            node = basin_saddlenode[i]
            tz = 1e9
            rec_out = -1
            for k in range(NN):
                j = _GRID.neighbour(node, k)
                if j != -1 and bid[j] != i and tz > z[j]:
                    tz = z[j]
                    rec_out = j
            if rec_out > -1:
                candidate = _PACK(tz, rec_out)
                _BK.atomic_min(outlet[i], candidate)

    def break_cycle_template(bid: T, outlet: T, basin_saddle: T, basin_saddlenode: T):
        invalid_c = _PACK(1e8, 42)
        for i in bid:
            bid_d = i
            if bid_d == 0 or outlet[bid_d] == invalid_c:
                continue
            rec_out = _UNPACKINDEX(outlet[bid_d])
            bid_d_prime = bid[rec_out]
            if bid_d_prime == 0:
                continue
            rec_out_prime = _UNPACKINDEX(outlet[bid_d_prime])
            bid_d_prime_prime = bid[rec_out_prime]
            if bid_d_prime_prime == bid_d:
                if bid_d_prime < bid_d:
                    outlet[bid_d] = invalid_c
                    basin_saddle[bid_d] = invalid_c
                    basin_saddlenode[bid_d] = -1

    border_zprime = KernelCls().bind("_GRID", grid).bind("_BK", backend_mod).ingest(border_zprime_template)
    init_saddle_outlet = KernelCls().bind("_PACK", pack).ingest(init_saddle_outlet_template)
    atomic_min_saddle = (
        KernelCls()
        .bind("_GRID", grid)
        .bind("_BK", backend_mod)
        .bind("_PACK", pack)
        .ingest(atomic_min_saddle_template)
    )
    find_saddlenode = (
        KernelCls()
        .bind("_GRID", grid)
        .bind("_UNPACKVALUE", unpack_value)
        .bind("_UNPACKINDEX", unpack_index)
        .ingest(find_saddlenode_template)
    )
    atomic_min_outlet = (
        KernelCls()
        .bind("_GRID", grid)
        .bind("_BK", backend_mod)
        .bind("_PACK", pack)
        .ingest(atomic_min_outlet_template)
    )
    break_cycle = KernelCls().bind("_PACK", pack).bind("_UNPACKINDEX", unpack_index).ingest(break_cycle_template)

    kernels = {
        "border_zprime": border_zprime,
        "init_saddle_outlet": init_saddle_outlet,
        "atomic_min_saddle": atomic_min_saddle,
        "find_saddlenode": find_saddlenode,
        "atomic_min_outlet": atomic_min_outlet,
        "break_cycle": break_cycle,
    }

    rb = RoutineBuilderCls()
    rb.bind_bag(
        Bag(
            {
                "_GRID": grid,
                "_BK": backend_mod,
                "_PACK": pack,
                "_UNPACKVALUE": unpack_value,
                "_UNPACKINDEX": unpack_index,
            }
        )
    )
    for name in ("bid", "z", "z_prime", "is_border", "basin_saddle", "basin_saddlenode", "outlet"):
        rb.add_data(name, None)

    rb.add_kernel(border_zprime, data_handle_ref=("bid", "z", "z_prime", "is_border"))
    rb.add_kernel(init_saddle_outlet, data_handle_ref=("basin_saddle", "outlet", "basin_saddlenode"))
    rb.add_kernel(atomic_min_saddle, data_handle_ref=("bid", "is_border", "z_prime", "basin_saddle"))
    rb.add_kernel(find_saddlenode, data_handle_ref=("bid", "is_border", "z_prime", "basin_saddle", "basin_saddlenode"))
    rb.add_kernel(atomic_min_outlet, data_handle_ref=("bid", "basin_saddle", "basin_saddlenode", "z", "outlet"))
    rb.add_kernel(break_cycle, data_handle_ref=("bid", "outlet", "basin_saddle", "basin_saddlenode"))

    return rb, kernels


def build_reroute_carve_vanilla(RoutineBuilderCls, KernelCls, *, backend, backend_mod, bitpack, copy_field, logn: int):
    """
    RoutineBuilder for carve+vanilla reroute: init_reroute_carve(tag,
    tag_alt, basin_saddlenode); copy_field(rec_work -> rec);
    copy_field(rec_work -> rec_jump); begin_repeat(logn+1):
    iteration_reroute_carve(tag, tag_alt, rec, rec_work, bid); end_repeat();
    finalise_reroute_carve(rec, rec_jump, tag, basin_saddlenode, outlet,
    rerouted); copy_field(rec -> rec_work).

    finalise's second data arg is bound to `rec_jump` - the snapshot taken
    of `rec_work` *before* the repeat block ran, i.e. the original,
    unjumped receiver chain - not the pointer-jumped `rec` the repeat block
    produced. The repeat block's pointer jumping is only used internally to
    propagate `tag` quickly; the actual edge reversal in finalise operates
    on the original chain, which is why finalise's own first statement
    resets `rec` from that original snapshot before reversing anything -
    ported exactly as legacy's flow_reroute_kernels.py has it.

    Data names: "rec", "rec_work", "rec_jump", "tag", "tag_alt", "bid",
    "basin_saddlenode", "outlet", "rerouted".

    Author: B.G (07/2026)
    """
    T = _tensor_annotation(backend_mod, backend)
    pack = bitpack["pack"]
    unpack_index = bitpack["unpack_index"]

    def init_reroute_carve_template(tag: T, tag_alt: T, saddlenode: T):
        for i in tag:
            tag[i] = 0
        for i in tag:
            if saddlenode[i] != -1:
                tag[saddlenode[i]] = 1
        for i in tag:
            tag_alt[i] = tag[i]

    def iteration_reroute_carve_template(tag: T, tag_alt: T, rec: T, rec_work: T, bid: T):
        for i in tag:
            if bid[i] == 0:
                continue
            if tag[i] and rec[i] != i:
                tag_alt[rec[i]] = 1
            rec_work[i] = rec[i]
        for i in tag:
            if bid[i] == 0:
                continue
            if rec_work[i] != i:
                rec[i] = rec_work[rec_work[i]]
            tag[i] = tag_alt[i]

    def finalise_reroute_carve_template(rec: T, rec_orig: T, tag: T, saddlenode: T, outlet: T, rerouted: T):
        invalid = _PACK(1e8, 42)
        for i in rec:
            rec[i] = rec_orig[i]
        for i in rec:
            if tag[rec_orig[i]] and tag[i] and i != rec_orig[i]:
                rec[rec_orig[i]] = i
                rerouted[rec_orig[i]] = 1
        for i in rec:
            if outlet[i] != invalid:
                node = _UNPACKINDEX(outlet[i])
                rec[saddlenode[i]] = node
                rerouted[saddlenode[i]] = 1

    init_reroute_carve = KernelCls().ingest(init_reroute_carve_template)
    iteration_reroute_carve = KernelCls().ingest(iteration_reroute_carve_template)
    finalise_reroute_carve = (
        KernelCls().bind("_PACK", pack).bind("_UNPACKINDEX", unpack_index).ingest(finalise_reroute_carve_template)
    )

    kernels = {
        "init_reroute_carve": init_reroute_carve,
        "iteration_reroute_carve": iteration_reroute_carve,
        "finalise_reroute_carve": finalise_reroute_carve,
    }

    rb = RoutineBuilderCls()
    rb.bind_bag(Bag({"_PACK": pack, "_UNPACKINDEX": unpack_index}))
    for name in ("rec", "rec_work", "rec_jump", "tag", "tag_alt", "bid", "basin_saddlenode", "outlet", "rerouted"):
        rb.add_data(name, None)

    rb.add_kernel(init_reroute_carve, data_handle_ref=("tag", "tag_alt", "basin_saddlenode"))
    rb.add_kernel(copy_field, data_handle_ref=("rec_work", "rec"))
    rb.add_kernel(copy_field, data_handle_ref=("rec_work", "rec_jump"))
    rb.begin_repeat(times=logn + 1)
    rb.add_kernel(iteration_reroute_carve, data_handle_ref=("tag", "tag_alt", "rec", "rec_work", "bid"))
    rb.end_repeat()
    rb.add_kernel(finalise_reroute_carve, data_handle_ref=("rec", "rec_jump", "tag", "basin_saddlenode", "outlet", "rerouted"))
    rb.add_kernel(copy_field, data_handle_ref=("rec", "rec_work"))

    return rb, kernels


def build_reroute_carve_optimized(KernelCls, *, backend, backend_mod, bitpack):
    """
    carve_basins_serial KernelBuilder - one launch, one serial thread per
    basin walking `rec` from the saddle node to the pit reversing links,
    then saddle -> outlet. Distinct basins' chains are node-disjoint so the
    writes never race.

    Data args (rec, basin_saddlenode, outlet).

    Author: B.G (07/2026)
    """
    T = _tensor_annotation(backend_mod, backend)
    unpack_index = bitpack["unpack_index"]

    def carve_basins_serial_template(rec: T, basin_saddlenode: T, outlet: T):
        invalid = _PACK(1e8, 42)
        for b in basin_saddlenode:
            s = basin_saddlenode[b]
            if s == -1 or outlet[b] == invalid:
                continue
            out_node = _UNPACKINDEX(outlet[b])
            node = s
            nxt = rec[node]
            rec[node] = out_node
            while nxt != node:
                nnxt = rec[nxt]
                rec[nxt] = node
                node = nxt
                nxt = nnxt

    return (
        KernelCls()
        .bind("_PACK", bitpack["pack"])
        .bind("_UNPACKINDEX", unpack_index)
        .ingest(carve_basins_serial_template)
    )


def build_reroute_jump(KernelCls, *, backend, backend_mod, bitpack):
    """
    reroute_jump KernelBuilder - one launch, pit points straight at the
    outlet. Shared unchanged by both `method`s: called with the currently
    resolved receiver buffer bound to its `rec` argument, whichever buffer
    that is for a given method.

    The write is deliberately `rec[i - 1]`, not `rec[i]`: the loop is over
    basin ids (`i` ranges over `outlet`'s own index space) and basin id =
    pit index + 1, so `i - 1` is the pit node. Ported exactly as legacy has
    it - see build_reroute_jump's cupy counterpart and make_depressions'
    docstring for the same note.

    Data args (rec, outlet, rerouted).

    Author: B.G (07/2026)
    """
    T = _tensor_annotation(backend_mod, backend)

    def reroute_jump_template(rec: T, outlet: T, rerouted: T):
        invalid = _PACK(1e8, 42)
        for i in rerouted:
            rerouted[i] = 0
        for i in rec:
            if outlet[i] != invalid:
                rrec = _UNPACKINDEX(outlet[i])
                rec[i - 1] = rrec
                rerouted[i - 1] = 1

    return (
        KernelCls()
        .bind("_PACK", bitpack["pack"])
        .bind("_UNPACKINDEX", bitpack["unpack_index"])
        .ingest(reroute_jump_template)
    )


def build_depression_counter(KernelCls, *, backend, backend_mod, grid, ndep_raw):
    """
    depression_counter KernelBuilder, data arg (rec,): atomic-adds 1 into
    the raw device cell `ndep_raw` (bound directly, not through a
    Parameter's device_view - same idiom as ops.make_reduce's own atomic
    kernels) for every self-receiving node that cannot drain. The caller
    must reset the backing scalar Parameter to 0 (`.set(0)`) before each
    launch - this kernel only accumulates, mirroring ops.Reduce.run_sum's
    own reset-then-launch pattern.

    Author: B.G (07/2026)
    """
    T = _tensor_annotation(backend_mod, backend)

    def depression_counter_template(rec: T):
        for i in rec:
            if rec[i] == i and not _GRID.can_out(i):
                _BK.atomic_add(_NDEP[None], 1)

    return (
        KernelCls()
        .bind("_GRID", grid)
        .bind("_BK", backend_mod)
        .bind("_NDEP", ndep_raw)
        .ingest(depression_counter_template)
    )


# ---------------------------------------------------------------------------
# reconstruction fill (make_fill_reconstruct) - see _cupy_blocks.py's own
# section note for the algorithm (ported from
# experimental/LM/fill_reconstruct_optimised.py) and for why frontier_a/
# frontier_b become one combined (2*n_flat,) buffer here, addressed by a
# `p % 2` parity computed from the bound `P` Parameter - identical reasoning
# on this backend, since a Sequence's kernel_step binds data once at compile
# time on every backend, not just cupy.
#
# Two closure-specific substitutions from the cupy version, both verified
# directly against this Taichi/Quadrants install before use here:
#
# - No 3-argument `range()` (reverse step) inside a kernel - confirmed
#   Taichi rejects it ("Range should have 1 or 2 arguments"). The two
#   right-to-left/bottom-to-top sweeps below instead drive a forward-
#   counting loop variable and compute the descending index from it
#   (`c = NX - 2 - cc`); still a single serial nested loop per thread, same
#   execution order as a real reverse range.
# - No `atomicExch` on Taichi (only Quadrants has `atomic_exchange`) - both
#   backends use `_BK.atomic_max(queued_gen[j], p)` instead, whose returned
#   old value gives the identical "first writer this pass wins" dedup
#   `atomicExch` does, because `p` only ever increases across passes: the
#   first thread to touch queued_gen[j] this pass raises it from some
#   earlier (smaller) value to p and gets that smaller value back; every
#   later thread doing the same atomic_max this pass finds it already at p,
#   contributes no change, and gets p back - confirmed empirically (a fresh
#   -1-filled field, one atomic_max(..., p) per candidate, only the winner's
#   returned old value differs from p) before relying on it here.
# ---------------------------------------------------------------------------

_POS_SENTINEL = 1.0e9


def build_fill_reconstruct_init(KernelCls, *, backend, backend_mod, grid):
    """
    init_filled KernelBuilder, data args (z, filled, parent) - see
    _cupy_blocks.build_fill_reconstruct_init.

    Author: B.G (07/2026)
    """
    T = _tensor_annotation(backend_mod, backend)

    def init_filled_template(z: T, filled: T, parent: T):
        for i in z:
            if _GRID.can_out(i):
                filled[i] = z[i]
                parent[i] = i
            else:
                filled[i] = _POS_SENTINEL
                parent[i] = -1

    return KernelCls().bind("_GRID", grid).ingest(init_filled_template)


def build_fill_reconstruct_sweeps(KernelCls, *, backend, backend_mod, nx: int, ny: int):
    """
    Four KernelBuilders, each data args (z, filled, parent) - see
    _cupy_blocks.build_fill_reconstruct_sweeps. Keyed "row_lr", "row_rl",
    "col_tb", "col_bt".

    Author: B.G (07/2026)
    """
    T = _tensor_annotation(backend_mod, backend)
    NX = nx
    NY = ny

    def sweep_row_lr_template(z: T, filled: T, parent: T):
        for r in range(NY):
            base = r * NX
            for c in range(1, NX):
                i = base + c
                left = i - 1
                cand = z[i] if z[i] > filled[left] else filled[left]
                if cand < filled[i]:
                    filled[i] = cand
                    parent[i] = left

    def sweep_row_rl_template(z: T, filled: T, parent: T):
        for r in range(NY):
            base = r * NX
            for cc in range(NX - 1):
                c = NX - 2 - cc
                i = base + c
                right = i + 1
                cand = z[i] if z[i] > filled[right] else filled[right]
                if cand < filled[i]:
                    filled[i] = cand
                    parent[i] = right

    def sweep_col_tb_template(z: T, filled: T, parent: T):
        for c in range(NX):
            for r in range(1, NY):
                i = r * NX + c
                up = i - NX
                cand = z[i] if z[i] > filled[up] else filled[up]
                if cand < filled[i]:
                    filled[i] = cand
                    parent[i] = up

    def sweep_col_bt_template(z: T, filled: T, parent: T):
        for c in range(NX):
            for rr in range(NY - 1):
                r = NY - 2 - rr
                i = r * NX + c
                down = i + NX
                cand = z[i] if z[i] > filled[down] else filled[down]
                if cand < filled[i]:
                    filled[i] = cand
                    parent[i] = down

    return {
        "row_lr": KernelCls().ingest(sweep_row_lr_template),
        "row_rl": KernelCls().ingest(sweep_row_rl_template),
        "col_tb": KernelCls().ingest(sweep_col_tb_template),
        "col_bt": KernelCls().ingest(sweep_col_bt_template),
    }


def build_fill_reconstruct_frontier_init(KernelCls, *, backend, backend_mod):
    """
    frontier_init KernelBuilder, data args (z, filled, frontier, counters) -
    see _cupy_blocks.build_fill_reconstruct_frontier_init.

    Author: B.G (07/2026)
    """
    T = _tensor_annotation(backend_mod, backend)

    def frontier_init_template(z: T, filled: T, frontier: T, counters: T):
        for i in z:
            if filled[i] > z[i]:
                pos = _BK.atomic_add(counters[0], 1)
                frontier[pos] = i

    return KernelCls().bind("_BK", backend_mod).ingest(frontier_init_template)


def build_fill_reconstruct_relax(KernelCls, *, backend, backend_mod, grid, pass_p, n_flat: int):
    """
    relax KernelBuilder, data args (z, filled, parent, frontier, counters,
    queued_gen) - see _cupy_blocks.build_fill_reconstruct_relax; the same
    push-gate and combined-buffer-parity logic, without the cupy version's
    neighbour-value caching (see that function's docstring for why dropping
    it does not change correctness) - a top-level `for idx in range(count)`
    with `count` read from `counters[p]` at kernel entry, confirmed to
    compile and execute correctly with a runtime (not compile-time) bound on
    this Taichi/Quadrants install before relying on it here.

    `pass_p` is read only here (`_P.get(0)`); bumping it between passes is
    the caller's job, the same division of labour `iteration_p` has for
    rake_compress.

    Author: B.G (07/2026)
    """
    T = _tensor_annotation(backend_mod, backend)
    NFLAT = n_flat

    def relax_template(z: T, filled: T, parent: T, frontier: T, counters: T, queued_gen: T):
        p = _P.get(0)
        par = p % 2
        in_base = par * NFLAT
        out_base = (1 - par) * NFLAT
        count = counters[p]
        for idx in range(count):
            i = frontier[in_base + idx]
            nk = _GRID.n_neighbours.get(0)

            best = _POS_SENTINEL
            best_j = -1
            for k in range(nk):
                j = _GRID.neighbour(i, k)
                if j != -1:
                    v = filled[j]
                    if v < best:
                        best = v
                        best_j = j
            candidate = z[i] if z[i] > best else best

            if candidate < filled[i]:
                filled[i] = candidate
                parent[i] = best_j
                for k in range(nk):
                    j = _GRID.neighbour(i, k)
                    if j != -1:
                        cand_j = z[j] if z[j] > candidate else candidate
                        if cand_j < filled[j]:
                            old = _BK.atomic_max(queued_gen[j], p)
                            if old != p:
                                pos = _BK.atomic_add(counters[p + 1], 1)
                                frontier[out_base + pos] = j

    return (
        KernelCls()
        .bind("_GRID", grid)
        .bind("_BK", backend_mod)
        .bind("_P", pass_p)
        .ingest(relax_template)
    )
