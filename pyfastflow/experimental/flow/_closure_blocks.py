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
