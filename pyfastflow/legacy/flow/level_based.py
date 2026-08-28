"""
Level-based downstream propagation (MFD) with multiple scheduling strategies.

This module provides kernels and helpers to perform level-based downstream
propagation for multiple-flow-direction (MFD) accumulation using three
strategies for selecting processable nodes each iteration:

1) Brute force: process all nodes but skip if not processable.
2) Atomic compaction: compact processable node indices with atomics, then process list.
3) Parallel-scan compaction: use inclusive scan to compact indices, then process list.

State semantics per node (u8/i32):
    0 = NOT_PROCESSED
    1 = PROCESSABLE (ndons == 0 and not processed)
    2 = PROCESSED (already handled in a prior iteration)

The donor counts array `ndons` stores, for each node i, how many donor neighbors
remain to be processed (initially the count of upslope neighbors z[j] > z[i]).
Once a donor node i is processed, each of its downstream neighbors j (z[j] < z[i])
has ndons[j] decremented by 1.

Author: B.G. (implementation outline), additions by assistant
"""

import taichi as ti

import pyfastflow as pf
from ..general_algorithms import inclusive_scan

@ti.func
def S_NOT_PROCESSED() -> ti.u8:
    return ti.u8(0)


@ti.func
def S_PROCESSABLE() -> ti.u8:
    return ti.u8(1)


@ti.func
def S_PROCESSED() -> ti.u8:
    return ti.u8(2)


@ti.kernel
def precompute_downstream(
    z: ti.template(), dn_idx: ti.template(), w: ti.template(), wsum: ti.template()
):
    """
    Precompute downstream neighbors and MFD weights per node.

    dn_idx[base+k] = neighbor index or -1 for each cardinal direction.
    w[base+k] = positive slope to neighbor (0 if not downhill or invalid).
    wsum[i] = sum_k w[base+k].
    """
    for i in z:
        base = i * 4
        total = 0.0
        zi = z[i]
        for k in ti.static(range(4)):
            j = pf.flow.neighbourer_flat.neighbour(i, k)
            if j > -1 and z[j] < zi:
                dn_idx[base + k] = j
                ww = (zi - z[j]) / pf.constants.DX
                w[base + k] = ww
                total += ww
            else:
                dn_idx[base + k] = -1
                w[base + k] = 0.0
        wsum[i] = total


@ti.kernel
def reset_states(state: ti.template()):
    """Set all states to NOT_PROCESSED."""
    for i in state:
        state[i] = S_NOT_PROCESSED()


@ti.func
def donors_processed(z: ti.template(), state: ti.template(), i: ti.i32) -> ti.u1:
    ready: ti.u1 = True
    zi = z[i]
    for k in ti.static(range(4)):
        j = pf.flow.neighbourer_flat.neighbour(i, k)
        if j > -1 and z[j] > zi:
            if state[j] != S_PROCESSED():
                ready = False
    return ready


@ti.kernel
def flag_processable(z: ti.template(), state: ti.template()) -> bool:
    """
    Flag processable nodes for current iteration based on donor counts.

    Returns True if at least one node became processable.
    """
    changed: ti.u1 = False
    for i in state:
        if state[i] != S_PROCESSED() and donors_processed(z, state, i):
            # Mark as processable only if not already processed
            if state[i] != S_PROCESSABLE():
                state[i] = S_PROCESSABLE()
                changed = True
    return changed


@ti.kernel
def build_flags_from_state(z: ti.template(), state: ti.template(), flags: ti.template()):
    """flags[i] = 1 if processable else 0 (int field)."""
    for i in state:
        flags[i] = 1 if (state[i] == S_PROCESSABLE() and donors_processed(z, state, i)) else 0


@ti.kernel
def process_node_list_precomp(
    ids: ti.template(),
    count: ti.template(),
    Q: ti.template(),
    state: ti.template(),
    dn_idx: ti.template(),
    w: ti.template(),
    wsum: ti.template(),
    z: ti.template(),
):
    """
    Process only nodes listed in `ids[0:count]` for MFD accumulation.

    - Distribute Q[i] to all lower neighbors proportional to slope
    - Mark node as processed and set ndons[i] to -1 sentinel
    - Decrement ndons of each downstream neighbor
    """
    for idx in range(count[None]):
        i = ids[idx]
        if state[i] != S_PROCESSABLE():
            continue

        base = i * 4
        ws = wsum[i]
        if ws > 0.0:
            Qi = Q[i]
            for k in ti.static(range(4)):
                j = dn_idx[base + k]
                if j != -1:
                    ww = w[base + k]
                    if ww > 0.0:
                        ti.atomic_add(Q[j], Qi * ww / ws)

        state[i] = S_PROCESSED()


@ti.kernel
def process_all_bruteforce_precomp(
    Q: ti.template(), state: ti.template(), dn_idx: ti.template(), w: ti.template(), wsum: ti.template(), z: ti.template()
) -> bool:
    """
    Brute-force process: loop all nodes and handle only processable ones.

    Returns True if at least one node was processed.
    """
    changed: ti.u1 = False
    for i in Q:
        if state[i] != S_PROCESSED() and donors_processed(z, state, i):
            changed = True
            base = i * 4
            ws = wsum[i]
            if ws > 0.0:
                Qi = Q[i]
                for k in ti.static(range(4)):
                    j = dn_idx[base + k]
                    if j != -1:
                        ww = w[base + k]
                        if ww > 0.0:
                            ti.atomic_add(Q[j], Qi * ww / ws)

            state[i] = S_PROCESSED()
    return changed


@ti.kernel
def atomic_compact_ids_precomp(
    up_idx: ti.template(), state: ti.template(), ids: ti.template(), count: ti.template()
):
    """
    Build active list of processable nodes using an atomic counter.

    `ids` must be sized at least N; `count` is a 0D i32 field.
    """
    count[None] = 0
    for i in state:
        if state[i] == S_PROCESSABLE() and donors_processed_up(up_idx, state, i):
            pos = ti.atomic_add(count[None], 1)
            ids[pos] = i


@ti.kernel
def scatter_compacted_from_scan(
    flags: ti.template(), scan_out: ti.template(), ids: ti.template()
):
    """
    Scatter indices where flags[i]==1 into `ids` using inclusive scan results.

    For flags[i]==1, position is scan_out[i]-1.
    """
    for i in flags:
        if flags[i] == 1:
            pos = scan_out[i] - 1
            ids[pos] = i


@ti.kernel
def read_count_from_scan(scan_out: ti.template(), n: int, out: ti.template()):
    out[None] = scan_out[n - 1]


def parallel_scan_compact(
    flags: ti.template(), scan_out: ti.template(), work: ti.template(), ids: ti.template(), n: int, count_scalar: ti.template()
) -> int:
    """
    Inclusive-scan based compaction that avoids host copies.

    Fills count_scalar[None] with number of active elements and fills `ids[0:count]`.
    Returns the count as Python int.
    """
    inclusive_scan(flags, scan_out, work, n)
    read_count_from_scan(scan_out, n, count_scalar)
    count = int(count_scalar[None])
    if count <= 0:
        return 0
    scatter_compacted_from_scan(flags, scan_out, ids)
    return count


def mfd_iteration_bruteforce(Q, state, dn_idx, w, wsum, up_idx) -> bool:
    """One iteration: single pass processes nodes whose donors are all processed (uses precomputed up_idx)."""
    return bool(process_all_bruteforce_precomp(Q, state, dn_idx, w, wsum, up_idx))


def mfd_iteration_atomic(Q, z, state, ids, count_scalar, dn_idx, w, wsum, up_idx=None) -> int:
    """One iteration: flag, atomic compact, process list; returns processed count."""
    # Mark processable
    if up_idx is None:
        flag_processable(z, state)  # fallback
    else:
        flag_processable_up(up_idx, state)
    # Build list
    if up_idx is None:
        atomic_compact_ids_precomp(z, state, ids, count_scalar)  # type: ignore
    else:
        atomic_compact_ids_precomp(up_idx, state, ids, count_scalar)
    count = int(count_scalar[None])
    if count == 0:
        return 0
    process_node_list_precomp(ids, count_scalar, Q, state, dn_idx, w, wsum, z)
    return count


def mfd_iteration_scan(Q, z, state, flags, scan_out, work, ids, n: int, count_scalar=None, dn_idx=None, w=None, wsum=None, up_idx=None) -> int:
    """One iteration: flag, scan-compact, process list; returns processed count."""
    # First flag nodes that became processable in this iteration
    if up_idx is None:
        flag_processable(z, state)
    else:
        flag_processable_up(up_idx, state)
    # Build flags
    if up_idx is None:
        build_flags_from_state(z, state, flags)
    else:
        build_flags_from_state_up(up_idx, state, flags)
    # Prepare a 0D count scalar if not supplied
    created = False
    if count_scalar is None:
        count_scalar = ti.field(dtype=ti.i32, shape=())
        created = True
    count = parallel_scan_compact(flags, scan_out, work, ids, n, count_scalar)
    if count == 0:
        return 0
    process_node_list_precomp(ids, count_scalar, Q, state, dn_idx, w, wsum, z)
    return count
