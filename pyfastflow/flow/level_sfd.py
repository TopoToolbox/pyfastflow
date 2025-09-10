"""
Level-based steepest-descent (SFD) flow accumulation with multiple scheduling strategies.

Implements three GPU strategies to accumulate single-direction flow (all discharge
goes to the unique receiver) in topological order from sources to sinks:
- Brute force: fused per-iteration kernel scans all nodes and processes those ready.
- Atomic compaction: compact ready nodes to a list using an atomic counter, then process.
- Scan compaction: build flags, inclusive-scan to compact IDs, then process.

Readiness is tracked via donors_remaining[i], initialized to the in-degree (number
of donors) from the receiver graph. When a node i is processed, donors_remaining[rcv[i]]
is decremented atomically.
"""

import taichi as ti

import pyfastflow as pf
from ..general_algorithms import inclusive_scan
from . import downstream_propag as dpr


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
def reset_states(state: ti.template()):
    for i in state:
        state[i] = S_NOT_PROCESSED()


@ti.kernel
def clear_i32(arr: ti.template()):
    for i in arr:
        arr[i] = 0


@ti.kernel
def flag_processable(donors_remaining: ti.template(), state: ti.template()) -> bool:
    changed: ti.u1 = False
    for i in state:
        if state[i] != S_PROCESSED() and donors_remaining[i] == 0:
            if state[i] != S_PROCESSABLE():
                state[i] = S_PROCESSABLE()
                changed = True
    return changed


@ti.kernel
def build_flags(donors_remaining: ti.template(), state: ti.template(), flags: ti.template()):
    for i in state:
        flags[i] = 1 if (state[i] == S_PROCESSABLE() and donors_remaining[i] == 0) else 0


@ti.kernel
def process_node_list_sfd(
    ids: ti.template(), count: ti.template(), Q: ti.template(), receivers: ti.template(), donors_remaining: ti.template(), state: ti.template()
):
    for idx in range(count[None]):
        i = ids[idx]
        if state[i] != S_PROCESSABLE() or donors_remaining[i] != 0:
            continue
        r = receivers[i]
        if r != i and r >= 0:
            ti.atomic_add(Q[r], Q[i])
            ti.atomic_add(donors_remaining[r], -1)
        state[i] = S_PROCESSED()


@ti.kernel
def process_all_bruteforce_sfd(
    Q: ti.template(), receivers: ti.template(), donors_remaining: ti.template(), state: ti.template()
) -> bool:
    changed: ti.u1 = False
    for i in Q:
        if state[i] == S_PROCESSABLE() and donors_remaining[i] == 0:
            changed = True
            r = receivers[i]
            if r != i and r >= 0:
                ti.atomic_add(Q[r], Q[i])
                ti.atomic_add(donors_remaining[r], -1)
            state[i] = S_PROCESSED()
    return changed


@ti.kernel
def atomic_compact_ids(state: ti.template(), donors_remaining: ti.template(), ids: ti.template(), count: ti.template()):
    count[None] = 0
    for i in state:
        if state[i] == S_PROCESSABLE() and donors_remaining[i] == 0:
            pos = ti.atomic_add(count[None], 1)
            ids[pos] = i


@ti.kernel
def read_count_from_scan(scan_out: ti.template(), n: int, out: ti.template()):
    out[None] = scan_out[n - 1]


def parallel_scan_compact(flags: ti.template(), scan_out: ti.template(), work: ti.template(), ids: ti.template(), n: int, count_scalar: ti.template()) -> int:
    inclusive_scan(flags, scan_out, work, n)
    read_count_from_scan(scan_out, n, count_scalar)
    count = int(count_scalar[None])
    if count <= 0:
        return 0
    scatter_compacted_from_scan(flags, scan_out, ids)
    return count


@ti.kernel
def scatter_compacted_from_scan(flags: ti.template(), scan_out: ti.template(), ids: ti.template()):
    for i in flags:
        if flags[i] == 1:
            pos = scan_out[i] - 1
            ids[pos] = i


def init_donors_from_receivers(receivers: ti.template(), donors: ti.template(), donors_remaining: ti.template()):
    clear_i32(donors_remaining)
    dpr.rcv2donor(receivers, donors, donors_remaining)


def iteration_bruteforce(Q, receivers, donors_remaining, state) -> bool:
    flagged = flag_processable(donors_remaining, state)
    if not flagged:
        return False
    return bool(process_all_bruteforce_sfd(Q, receivers, donors_remaining, state))


def iteration_atomic(Q, receivers, donors_remaining, state, ids, count_scalar) -> int:
    flag_processable(donors_remaining, state)
    atomic_compact_ids(state, donors_remaining, ids, count_scalar)
    count = int(count_scalar[None])
    if count == 0:
        return 0
    process_node_list_sfd(ids, count_scalar, Q, receivers, donors_remaining, state)
    return count


def iteration_scan(Q, receivers, donors_remaining, state, flags, scan_out, work, ids, n: int, count_scalar=None) -> int:
    flag_processable(donors_remaining, state)
    build_flags(donors_remaining, state, flags)
    created = False
    if count_scalar is None:
        count_scalar = ti.field(dtype=ti.i32, shape=())
        created = True
    count = parallel_scan_compact(flags, scan_out, work, ids, n, count_scalar)
    if count == 0:
        return 0
    process_node_list_sfd(ids, count_scalar, Q, receivers, donors_remaining, state)
    return count

