"""
Taichi/Quadrants (closure) block templates behind make_accumulation: the
ping-pong src helpers, and the three accumulation methods ("atomic",
"rake_compress", "pointer_jump_push").

Split out of a single _closure_blocks.py that used to hold every flow
algorithm - see _closure_receivers.py/_closure_depressions.py/
_closure_reconstruct.py for the others. build_rake_compress/
build_pointer_jump_push are nested defs closing over the per-backend Tensor
annotation and over small build-time-constant python ints (n_neighbours,
n_flat) - the same idiom ../ops/_closure_blocks.py's build_elementwise/
build_scan_routine use. capture_template_meta dedents a nested def's source
before parsing it and _fuse_group synthesizes each data argument's
annotation from the bound backend module rather than reading one out of the
AST, so these templates fuse; make_accumulation's caller still decides
fused=True/False when it compiles the returned RoutineBuilder. A round's
cross-buffer dependency on the previous round's writes needs no barrier -
fusion could not give it either way: consecutive top-level `for` loops
inside one compiled Taichi/Quadrants kernel are already separate offloaded
tasks launched in order (confirmed empirically - see make_accumulation's
docstring in __init__.py), the same guarantee legacy
pyfastflow/flow/lakeflow.py's saddlesort relies on for its own six
mutually-dependent passes in one hand-written kernel.

Author: B.G (07/2026)
"""

from ..core.context.bag import Bag
from ..core.context.need import Kind, Need
from ._closure_shared import _tensor_annotation


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


def build_ping_pong_helpers(HelperCls, *, iteration_need: Need):
    """
    get_src(src, tid)/update_src(src, tid, flip) HelperBuilders - same
    sign/magnitude encoding as pyfastflow/general_algorithms/pingpong.py's
    getSrc/updateSrc, reading `iteration_need` internally instead of taking
    iteration as a call argument.

    `iteration_need` is the caller's already-bound `Need("iteration_p",
    kind=Kind.PARAM)` (see make_accumulation) - a fresh, internally-named
    `Need("_ITER", ...)`, matching what these templates' bodies actually
    reference, is bound here to the same underlying Parameter and declared
    on both helpers via `.need()`.

    Author: B.G (07/2026)
    """
    iter_need = Need("_ITER", kind=Kind.PARAM, dtype=iteration_need.dtype, modes=iteration_need.modes)
    iter_need.bind(iteration_need.value)
    get_src = HelperCls().need(iter_need).ingest(_get_src_tmpl)
    update_src = HelperCls().need(iter_need).ingest(_update_src_tmpl)
    return get_src, update_src


def build_atomic(KernelCls, *, backend, backend_mod, source: Need, n_flat: int):
    """
    accum_downstream_atomic KernelBuilder, data args (rec, q): q[i] is
    initialized from `source.get(i)`, then every node walks its receiver
    chain to the root atomic-adding its own weight into each downstream
    node. Requires an acyclic receiver graph (run after depression
    handling); the `guard < n_flat` bound makes a cycle degrade the result
    instead of hanging, rather than guaranteeing correctness on one.

    `source` is the caller's already-bound `Need("source", kind=Kind.PARAM)`
    (see make_accumulation) - a fresh, internally-named `Need("_SOURCE", ...)`,
    matching this template's own `_SOURCE.get(i)` references, is bound here
    to the same underlying Parameter and declared on the KernelBuilder via
    `.need()`.

    Author: B.G (07/2026)
    """
    T = _tensor_annotation(backend_mod, backend)
    NFLAT = n_flat

    source_need = Need("_SOURCE", kind=Kind.PARAM, dtype=source.dtype, modes=source.modes)
    source_need.bind(source.value)

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

    return KernelCls().need(source_need).bind("_BK", backend_mod).ingest(accum_downstream_atomic_template)


def build_rake_compress(
    RoutineBuilderCls,
    KernelCls,
    HelperCls,
    *,
    backend: str,
    backend_mod,
    grid,
    source: Need,
    iteration_p: Need,
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

    `source`/`iteration_p` are the caller's already-bound `Need("source",
    kind=Kind.PARAM)`/`Need("iteration_p", kind=Kind.PARAM)` (see
    make_accumulation) - fresh, internally-named `Need("_SOURCE", ...)`/
    `Need("_ITER", ...)`, matching what these templates' bodies actually
    reference, are bound here to the same underlying Parameters and declared
    on every KernelBuilder/HelperBuilder that needs them via `.need()`.

    Returns (routine_builder, kernel_builders_dict). The routine is
    uncompiled and its data names are placeholders (add_data(name, None)) -
    every real call must pass all of them positionally; see
    make_accumulation's docstring for the exact order.

    Author: B.G (07/2026)
    """
    T = _tensor_annotation(backend_mod, backend)
    NN = grid.n_neighbours.get()

    source_need = Need("_SOURCE", kind=Kind.PARAM, dtype=source.dtype, modes=source.modes)
    source_need.bind(source.value)
    iter_need = Need("_ITER", kind=Kind.PARAM, dtype=iteration_p.dtype, modes=iteration_p.modes)
    iter_need.bind(iteration_p.value)

    get_src, update_src = build_ping_pong_helpers(HelperCls, iteration_need=iteration_p)

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
    reset_iteration = KernelCls().need(iter_need).ingest(reset_iteration_template)
    bump_iteration = KernelCls().need(iter_need).ingest(bump_iteration_template)
    decrement_iteration = KernelCls().need(iter_need).ingest(decrement_iteration_template)
    q_init = KernelCls().need(source_need).ingest(q_init_template)
    receivers_to_donors = KernelCls().bind("_BK", backend_mod).ingest(receivers_to_donors_template)
    rake_compress_accum = (
        KernelCls()
        .bind("_BK", backend_mod)
        .bind("_GETSRC", get_src)
        .bind("_UPDATESRC", update_src)
        .need(iter_need)
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
    # here too, under the same name. "_ITER"/"_SOURCE" are absent here on
    # purpose: they arrive via .need(), resolved directly by each step's own
    # compile() rather than through this bag - see need.py/compile.py's
    # CompileBuilder._resolve_needs.
    rb.bind_bag(
        Bag(
            {
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
    source: Need,
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

    `source` is the caller's already-bound `Need("source", kind=Kind.PARAM)`
    (see make_accumulation) - a fresh, internally-named `Need("_SOURCE", ...)`
    is bound here to the same underlying Parameter and declared on q_init via
    `.need()`.

    Retirement rule (kept exactly, not restructured): when a node's parent
    is a sink in the current jumped graph (grandparent == parent), the node
    pushes once more and then points at itself, so it never re-pushes a
    growing sum into the sink.

    Author: B.G (07/2026)
    """
    T = _tensor_annotation(backend_mod, backend)

    source_need = Need("_SOURCE", kind=Kind.PARAM, dtype=source.dtype, modes=source.modes)
    source_need.bind(source.value)

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

    q_init = KernelCls().need(source_need).ingest(q_init_template)
    copy_rec_to_work = KernelCls().ingest(copy_rec_to_work_template)
    step = KernelCls().bind("_BK", backend_mod).ingest(accum_pointer_jump_push_step_template)

    kernels = {"q_init": q_init, "copy_rec_to_work": copy_rec_to_work, "accum_pointer_jump_push_step": step}

    rb = RoutineBuilderCls()
    rb.bind_bag(Bag({"_BK": backend_mod}))
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


