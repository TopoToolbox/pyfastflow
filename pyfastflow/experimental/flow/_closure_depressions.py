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

Author: B.G (07/2026)
"""

from ..core.context.bag import Bag
from ._closure_shared import _tensor_annotation


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


