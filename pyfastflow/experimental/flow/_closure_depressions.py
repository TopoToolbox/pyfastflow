"""
Taichi/Quadrants (closure) block templates behind make_depressions/
make_depression_solver: copy_field, both basin labelling variants,
saddlesort, both carve variants, jump reroute, and the depression counter.

Split out of a single _closure_blocks.py that used to hold every flow
algorithm - see _closure_receivers.py/_closure_accum.py/
_closure_reconstruct.py for the others. Ported from
../../flow/flow_reroute_kernels.py, `_PACK`/`_UNPACK_VALUE`/`_UNPACK_INDEX`
bound from ops.make_bitpack in place of legacy's f32_i32_struct module.
Every array here (rec, bid, tag, basin_saddle, outlet, ...) is n_flat-sized,
basin id = pit index + 1, so a per-basin array is safely indexed by any node
index too - the same double duty the legacy kernels rely on.

Every KernelBuilder built here is constructed strict_needs=True, every bind
going through a Need (param_need/helper_need/bag_need, see backends.py) -
mirrors _closure_receivers.py/_closure_accum.py's own conversions. `grid`
binds go through bag_need, one `contains` list per site declaring only the
members that template actually reads (`can_out`, `neighbour`, or both - never
one blanket "grid" contract). `bitpack` (make_depressions' own argument) is
the real `Bag` ops.make_bitpack returns (`pack`/`unpack_value`/
`unpack_index`), not a re-wrapped dict - each site here reaches only the
member(s) it calls via helper_need, since no single site uses all three.
`_BK` needs no bind anywhere - auto-injected (see
core/context/_closure_backend.py's module docstring). No RoutineBuilder built
here is ever given a bind_bag() bag: every step's own dependencies are
already fully resolved via Need by the time it is built, so there is nothing
left for a routine-level bag to supply (see routine.py,
RoutineBuilder._validate) - a Sequence wrapping one of these routines still
propagates its own outer bag into it via RoutineBuilder.bind_bag, which is
harmless since rebind() against it only ever re-resolves names to the exact
objects they already hold.

Author: B.G (07/2026)
"""

from ..core.context.backends import bag_need, helper_need, make_kernel, param_need
from ..core.context.need import Kind, Need
from ._closure_shared import _tensor_annotation


def build_copy_field(KernelCls, *, backend, backend_mod):
    """
    dst[i] = src[i] over a whole flat buffer - the generic copy used
    everywhere a depression pass needs one buffer snapshotted into another
    (rec -> rec_jump, rec_work -> rec, ...), reused across every routine that
    needs it rather than rebuilt per call site.

    `strict_needs=True`, though nothing here binds anything - see
    _closure_receivers.py's build_receivers for the reference conversion.

    Author: B.G (07/2026)
    """
    T = _tensor_annotation(backend_mod, backend)

    def copy_field_template(src: T, dst: T):
        for i in src:
            dst[i] = src[i]

    return KernelCls(strict_needs=True).ingest(copy_field_template)


def build_basin_id_init(KernelCls, *, backend, backend_mod, grid):
    """
    bid[i] = 0 on a can_out node, i+1 otherwise - the seed for vanilla basin
    labelling's pointer-jump propagation.

    `_GRID=grid` goes through bag_need, declaring only `can_out` - the
    member this template actually reads. `strict_needs=True`.

    Author: B.G (07/2026)
    """
    T = _tensor_annotation(backend_mod, backend)

    def basin_id_init_template(bid: T):
        for i in bid:
            bid[i] = 0 if _GRID.can_out(i) else (i + 1)

    grid_contains = [Need("can_out", kind=Kind.HELPER)]
    return make_kernel(KernelCls, basin_id_init_template, strict_needs=True, _GRID=bag_need("_GRID", grid, contains=grid_contains))


def build_propagate_basin_iter(KernelCls, *, backend, backend_mod):
    """
    One pointer-jump step over `rec_jump`, halving path length to the root
    each call.

    `strict_needs=True`, though nothing here binds anything.

    Author: B.G (07/2026)
    """
    T = _tensor_annotation(backend_mod, backend)

    def propagate_basin_iter_template(rec_jump: T):
        for i in rec_jump:
            if rec_jump[i] != rec_jump[rec_jump[i]]:
                rec_jump[i] = rec_jump[rec_jump[i]]

    return KernelCls(strict_needs=True).ingest(propagate_basin_iter_template)


def build_propagate_basin_final(KernelCls, *, backend, backend_mod):
    """
    bid[i] = bid[root(i)] - finalizes vanilla basin labelling once
    `rec_jump` has been pointer-jumped down to (near-)roots.

    `strict_needs=True`, though nothing here binds anything.

    Author: B.G (07/2026)
    """
    T = _tensor_annotation(backend_mod, backend)

    def propagate_basin_final_template(bid: T, rec_jump: T):
        for i in bid:
            bid[i] = bid[rec_jump[i]]

    return KernelCls(strict_needs=True).ingest(propagate_basin_final_template)


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

    The returned RoutineBuilder is never given a bind_bag() bag - see the
    module docstring.

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
    # No bind_bag() call: "_GRID" is bound to basin_id_init's own already-
    # bound bag_need at construction time - nothing left for a routine-level
    # bag to supply (see routine.py, RoutineBuilder._validate).
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

    `_GRID=grid` goes through bag_need, declaring only `can_out`.
    `strict_needs=True`.

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

    grid_contains = [Need("can_out", kind=Kind.HELPER)]
    return make_kernel(
        KernelCls, label_basins_walk_template, strict_needs=True, _GRID=bag_need("_GRID", grid, contains=grid_contains)
    )


def build_saddlesort(RoutineBuilderCls, KernelCls, *, backend, backend_mod, grid, bitpack):
    """
    RoutineBuilder for the six saddlesort passes (see the module docstring
    and todo.md's ordered list): border/z_prime detection, saddle/outlet/
    saddlenode init, bitpacked atomic-min saddle search, saddlenode
    identification, bitpacked atomic-min outlet search, basin-graph
    2-cycle break. Shared unchanged by both `method`s.

    `bitpack` is the Bag {"pack", "unpack_value", "unpack_index"} from
    ops.make_bitpack, replacing legacy's f32_i32_struct helpers - each
    KernelBuilder below reaches only the members it actually calls, via
    helper_need (not one blanket bag_need contract, since no single site here
    uses all three). `_GRID=grid` likewise declares, per site, only
    `can_out`/`neighbour` - whichever this template actually reads.
    `strict_needs=True` throughout; `_BK` needs no bind at all - auto-
    injected (see core/context/_closure_backend.py's module docstring).

    Returns (routine_builder, kernels_dict) - kernels_dict keys
    "border_zprime", "init_saddle_outlet", "atomic_min_saddle",
    "find_saddlenode", "atomic_min_outlet", "break_cycle".

    Data names: "bid", "z", "z_prime", "is_border", "basin_saddle",
    "basin_saddlenode", "outlet".

    The returned RoutineBuilder is never given a bind_bag() bag - see the
    module docstring.

    Author: B.G (07/2026)
    """
    T = _tensor_annotation(backend_mod, backend)
    NN = grid.n_neighbours.get()
    pack = bitpack.pack
    unpack_value = bitpack.unpack_value
    unpack_index = bitpack.unpack_index

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

    can_out_need = Need("can_out", kind=Kind.HELPER)
    neighbour_need = Need("neighbour", kind=Kind.HELPER)
    border_zprime_grid = [can_out_need, neighbour_need]
    neighbour_only_grid = [neighbour_need]

    border_zprime = make_kernel(
        KernelCls, border_zprime_template, strict_needs=True,
        _GRID=bag_need("_GRID", grid, contains=border_zprime_grid),
    )
    init_saddle_outlet = make_kernel(
        KernelCls, init_saddle_outlet_template, strict_needs=True, _PACK=helper_need("_PACK", pack)
    )
    atomic_min_saddle = make_kernel(
        KernelCls, atomic_min_saddle_template, strict_needs=True,
        _GRID=bag_need("_GRID", grid, contains=neighbour_only_grid),
        _PACK=helper_need("_PACK", pack),
    )
    find_saddlenode = make_kernel(
        KernelCls, find_saddlenode_template, strict_needs=True,
        _GRID=bag_need("_GRID", grid, contains=neighbour_only_grid),
        _UNPACKVALUE=helper_need("_UNPACKVALUE", unpack_value),
        _UNPACKINDEX=helper_need("_UNPACKINDEX", unpack_index),
    )
    atomic_min_outlet = make_kernel(
        KernelCls, atomic_min_outlet_template, strict_needs=True,
        _GRID=bag_need("_GRID", grid, contains=neighbour_only_grid),
        _PACK=helper_need("_PACK", pack),
    )
    break_cycle = make_kernel(
        KernelCls, break_cycle_template, strict_needs=True,
        _PACK=helper_need("_PACK", pack),
        _UNPACKINDEX=helper_need("_UNPACKINDEX", unpack_index),
    )

    kernels = {
        "border_zprime": border_zprime,
        "init_saddle_outlet": init_saddle_outlet,
        "atomic_min_saddle": atomic_min_saddle,
        "find_saddlenode": find_saddlenode,
        "atomic_min_outlet": atomic_min_outlet,
        "break_cycle": break_cycle,
    }

    rb = RoutineBuilderCls()
    # No bind_bag() call: "_GRID"/"_PACK"/"_UNPACKVALUE"/"_UNPACKINDEX" are
    # each bound to their own step's already-bound bag_need/helper_need at
    # construction time, and "_BK" is auto-injected - nothing left for a
    # routine-level bag to supply (see routine.py, RoutineBuilder._validate).
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

    `bitpack` is the Bag from ops.make_bitpack; `finalise_reroute_carve`
    reaches its `pack`/`unpack_index` members via helper_need, the only site
    here that needs either. `strict_needs=True` throughout.
    The returned RoutineBuilder is never given a bind_bag() bag - see the
    module docstring.

    Author: B.G (07/2026)
    """
    T = _tensor_annotation(backend_mod, backend)
    pack = bitpack.pack
    unpack_index = bitpack.unpack_index

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

    init_reroute_carve = KernelCls(strict_needs=True).ingest(init_reroute_carve_template)
    iteration_reroute_carve = KernelCls(strict_needs=True).ingest(iteration_reroute_carve_template)
    finalise_reroute_carve = make_kernel(
        KernelCls, finalise_reroute_carve_template, strict_needs=True,
        _PACK=helper_need("_PACK", pack), _UNPACKINDEX=helper_need("_UNPACKINDEX", unpack_index),
    )

    kernels = {
        "init_reroute_carve": init_reroute_carve,
        "iteration_reroute_carve": iteration_reroute_carve,
        "finalise_reroute_carve": finalise_reroute_carve,
    }

    rb = RoutineBuilderCls()
    # No bind_bag() call: "_PACK"/"_UNPACKINDEX" are bound to
    # finalise_reroute_carve's own already-bound helper_need at construction
    # time - nothing left for a routine-level bag to supply (see routine.py,
    # RoutineBuilder._validate).
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

    `bitpack` is the Bag from ops.make_bitpack; `pack`/`unpack_index` go
    through helper_need. `strict_needs=True`.

    Author: B.G (07/2026)
    """
    T = _tensor_annotation(backend_mod, backend)
    unpack_index = bitpack.unpack_index

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

    return make_kernel(
        KernelCls, carve_basins_serial_template, strict_needs=True,
        _PACK=helper_need("_PACK", bitpack.pack), _UNPACKINDEX=helper_need("_UNPACKINDEX", unpack_index),
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

    `bitpack` is the Bag from ops.make_bitpack; `pack`/`unpack_index` go
    through helper_need. `strict_needs=True`.

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

    return make_kernel(
        KernelCls, reroute_jump_template, strict_needs=True,
        _PACK=helper_need("_PACK", bitpack.pack), _UNPACKINDEX=helper_need("_UNPACKINDEX", bitpack.unpack_index),
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

    `_GRID=grid` goes through bag_need, declaring only `can_out`.
    `_NDEP=ndep_raw` is a raw backend field (bound directly, not through a
    Parameter, per this function's own docstring above) - a plain value with
    no dtype/mode for a Need to check, same category as ops's own raw-field
    binds (`_SUM`/`_MIN`/... in ops/_closure_blocks.py's build_reduce_kernels);
    strict_needs=True binds it unchanged. `_BK` needs no bind at all - auto-
    injected.

    Author: B.G (07/2026)
    """
    T = _tensor_annotation(backend_mod, backend)

    def depression_counter_template(rec: T):
        for i in rec:
            if rec[i] == i and not _GRID.can_out(i):
                _BK.atomic_add(_NDEP[None], 1)

    grid_contains = [Need("can_out", kind=Kind.HELPER)]
    return make_kernel(
        KernelCls, depression_counter_template, strict_needs=True,
        _GRID=bag_need("_GRID", grid, contains=grid_contains),
        _NDEP=ndep_raw,
    )


