"""
Parallel downstream propagation algorithms for flow accumulation.

Implements the rake-and-compress algorithm from Jain et al. 2024 for efficient
parallel computation of flow accumulation on GPU. Uses ping-pong buffering
to enable data-parallel processing of donor-receiver relationships.

Key algorithms:
- Rake: Process leaf nodes (≤1 donor) to accumulate values
- Compress: Reduce tree depth by pointer jumping
- Ping-pong: Alternate between buffer sets for parallelization

Author: B.G.
"""

import taichi as ti

from .. import constants as cte
from .. import general_algorithms as gena


@ti.kernel
def rcv2donor(rcv: ti.template(), dnr: ti.template(), ndnr: ti.template()):
    """
    Build donor list from receiver relationships.

    Args:
        rcv: Receiver array (each node's downstream receiver)
        dnr: Donor array (lists of upstream donors per node)
        ndnr: Number of donors per node

    Author: B.G.
    """
    for tid in rcv:
        if rcv[tid] != tid:  # If this node has a receiver (not itself)
            # Atomically increment receiver's donor count and get previous value
            old_val = ti.atomic_add(ndnr[rcv[tid]], 1)
            donid = rcv[tid] * 4 + old_val  # Calculate donor array index
            if donid < cte.NX * cte.NY * 4:  # Bounds check
                # Store this node as a donor to its receiver
                dnr[rcv[tid] * 4 + old_val] = tid


@ti.kernel
def init_affine_weights(
    dnr: ti.template(), ndnr: ti.template(), w: ti.template(), a: ti.f32
):
    """
    Initialize per-edge weights for affine downstream accumulation.

    Sets the initial weight for each donor edge of a node to `a`, which represents
    the local transmission coefficient applied at the receiver node. During path
    compression, these weights are multiplied by upstream coefficients to compose
    along the path.

    Args:
        dnr: Donor array (lists of upstream donors per node)
        ndnr: Number of donors per node
        w: Per-edge weights array (same shape as dnr)
        a: Constant transmission coefficient (0<=a<=1 typical)

    Author: B.G.
    """
    for tid in ndnr:
        base = tid * 4
        todo = ndnr[tid]
        for i in range(4):
            if i < todo:
                w[base + i] = a
            else:
                w[base + i] = 0.0


@ti.kernel
def rake_compress_accum(
    dnr: ti.template(),
    ndnr: ti.template(),
    p: ti.template(),
    src: ti.template(),
    dnr_: ti.template(),
    ndnr_: ti.template(),
    p_: ti.template(),
    iteration: int,
):
    """
    Main rake and compress accumulation kernel from Jain et al. 2024.

    Args:
        dnr: Primary donor array (lists of upstream donors per node)
        ndnr: Primary number of donors per node
        p: Primary property values to accumulate
        src: Ping-pong state array
        dnr_: Alternate donor array
        ndnr_: Alternate number of donors per node
        p_: Alternate property values
        iteration: Current iteration number

    Author: B.G.
    """

    for tid in p:
        # Determine which buffer set to read from based on ping-pong state
        flip = gena.getSrc(src, tid, iteration)

        # Initialize processing state
        worked = False  # Track if any work was done
        donors = ti.Vector([-1, -1, -1, -1])  # Local donor cache (max 4 per node)
        todo = ndnr[tid] if not flip else ndnr_[tid]  # Number of donors to process
        base = tid * 4  # Base index for this node's donors in global array
        p_added = 0.0  # Accumulated value for this node

        # Process each donor using rake and compress
        i = 0
        while i < todo and i < 4:  # Max 4 donors per node
            # Load donor ID if not already cached
            if donors[i] == -1:
                donors[i] = dnr[base + i] if not flip else dnr_[base + i]
            did = donors[i]  # Current donor ID

            # Check donor's ping-pong state and get its donor count
            flip_donor = gena.getSrc(src, did, iteration)
            ndnr_val = ndnr[did] if not flip_donor else ndnr_[did]

            # RAKE: Process donors with ≤1 remaining donors (leaves or near-leaves)
            if ndnr_val <= 1:
                # Initialize accumulator with current node's value on first work
                if not worked:
                    p_added = p[tid] if not flip else p_[tid]
                worked = True

                # Add donor's accumulated value
                p_val = p[did] if not flip_donor else p_[did]
                p_added += p_val

                # COMPRESS: Handle donor based on its remaining donor count
                if ndnr_val == 0:
                    # Donor is fully processed - remove from list by swapping with last
                    todo -= 1
                    if todo > i and base + todo < cte.NX * cte.NY * 4:  # Bounds check
                        donors[i] = dnr[base + todo] if not flip else dnr_[base + todo]
                    i -= 1  # Reprocess this slot with swapped donor
                else:
                    # Donor has 1 remaining - replace with its single donor
                    donors[i] = dnr[did * 4] if not flip_donor else dnr_[did * 4]
            i += 1

        # Write results to opposite buffer set (ping-pong)
        if worked:
            if flip:
                # Write to primary buffers
                ndnr[tid] = todo
                p[tid] = p_added
                for j in range(min(todo, 4)):  # Store compressed donor list
                    dnr[base + j] = donors[j]
            else:
                # Write to alternate buffers
                ndnr_[tid] = todo
                p_[tid] = p_added
                for j in range(min(todo, 4)):  # Store compressed donor list
                    dnr_[base + j] = donors[j]
            # Update ping-pong state to indicate this node was processed
            gena.updateSrc(src, tid, iteration, flip)


@ti.kernel
def rake_compress_accum_affine(
    dnr: ti.template(),
    ndnr: ti.template(),
    p: ti.template(),
    src: ti.template(),
    dnr_: ti.template(),
    ndnr_: ti.template(),
    p_: ti.template(),
    w: ti.template(),
    w_: ti.template(),
    iteration: int,
    a: ti.f32,
    b: ti.template(),
):
    """
    Rake-and-compress accumulation with affine transform q_out = a*q_in + b.

    Semantics at node i:
        q[i] = (local source) b[i]  +  sum_{d in donors(i)} (A_path_to_i_from_d * q[d])
    where A_path_to_i_from_d is the product of coefficients along the path from d to i.

    We implement this by attaching a per-edge weight to each pending donor entry, stored
    in arrays w/w_. Initially for a direct donor edge (d -> i), weight = a (at i). When
    compressing a donor with a single donor, we multiply the edge weight by `a` again to
    compose along the path (since `a` is constant here).

    Args:
        dnr, ndnr, p, src, dnr_, ndnr_, p_: ping-pong arrays as in additive version
        w, w_: Per-edge weights arrays (same shape as dnr/dnr_), ping-ponged
        iteration: Current iteration index
        a: Constant transmission coefficient
        b: Local source field (not used internally; p must be initialized with b)

    Note:
        - Ensure p is initialized to b before iterations.
        - For spatially varying `a`, extend signature to take a field and multiply by a[did].

    Author: B.G.
    """
    for tid in p:
        flip = gena.getSrc(src, tid, iteration)

        worked = False
        donors = ti.Vector([-1, -1, -1, -1])
        weights = ti.Vector([0.0, 0.0, 0.0, 0.0])
        todo = ndnr[tid] if not flip else ndnr_[tid]
        base = tid * 4
        p_added = 0.0

        i = 0
        while i < todo and i < 4:
            if donors[i] == -1:
                if not flip:
                    donors[i] = dnr[base + i]
                    weights[i] = w[base + i]
                else:
                    donors[i] = dnr_[base + i]
                    weights[i] = w_[base + i]

            did = donors[i]

            flip_donor = gena.getSrc(src, did, iteration)
            ndnr_val = ndnr[did] if not flip_donor else ndnr_[did]

            if ndnr_val <= 1:
                if not worked:
                    # Start from current node's accumulated value (should contain b already)
                    p_added = p[tid] if not flip else p_[tid]
                worked = True

                # Add affine-transformed donor contribution: weight * q_donor
                p_val = p[did] if not flip_donor else p_[did]
                p_added += weights[i] * p_val

                if ndnr_val == 0:
                    # Remove donor: swap with last, propagate weight
                    todo -= 1
                    if todo > i and base + todo < cte.NX * cte.NY * 4:
                        if not flip:
                            donors[i] = dnr[base + todo]
                            weights[i] = w[base + todo]
                        else:
                            donors[i] = dnr_[base + todo]
                            weights[i] = w_[base + todo]
                    i -= 1
                else:
                    # Compress: replace donor by its single donor and compose weights
                    if not flip_donor:
                        donors[i] = dnr[did * 4]
                    else:
                        donors[i] = dnr_[did * 4]
                    weights[i] = weights[i] * a
            i += 1

        if worked:
            if flip:
                ndnr[tid] = todo
                p[tid] = p_added
                for j in range(min(todo, 4)):
                    dnr[base + j] = donors[j]
                    w[base + j] = weights[j]
            else:
                ndnr_[tid] = todo
                p_[tid] = p_added
                for j in range(min(todo, 4)):
                    dnr_[base + j] = donors[j]
                    w_[base + j] = weights[j]
            gena.updateSrc(src, tid, iteration, flip)


@ti.kernel
def init_affine_weights_var(ndnr: ti.template(), w: ti.template(), a: ti.template()):
    """
    Initialize per-edge weights for affine downstream accumulation with spatially varying a.

    For each donor edge (d -> i), set weight to a[i]. During compression, multiply by a[did]
    for the skipped intermediate node `did`.

    Args:
        ndnr: Number of donors per node
        w: Per-edge weights array (same shape as donor array)
        a: Field of local transmission coefficients per node

    Author: B.G.
    """
    for tid in ndnr:
        base = tid * 4
        todo = ndnr[tid]
        for i in range(4):
            if i < todo:
                w[base + i] = a[tid]
            else:
                w[base + i] = 0.0


@ti.kernel
def init_unit_weights(ndnr: ti.template(), w: ti.template(), val: ti.f32):
    """
    Initialize per-edge weights to a constant value (typically 1.0).

    Useful to accumulate pre-attenuation incoming flux at each node when combined
    with compression-time multiplication by per-node a.

    Args:
        ndnr: Number of donors per node
        w: Per-edge weights array (same shape as donor array)
        val: Constant to assign to existing donor edges (e.g., 1.0)

    Author: B.G.
    """
    for tid in ndnr:
        base = tid * 4
        todo = ndnr[tid]
        for i in range(4):
            if i < todo:
                w[base + i] = val
            else:
                w[base + i] = 0.0


@ti.kernel
def rake_compress_accum_affine_var_from_external(
    dnr: ti.template(),
    ndnr: ti.template(),
    accum: ti.template(),
    src: ti.template(),
    dnr_: ti.template(),
    ndnr_: ti.template(),
    accum_: ti.template(),
    w: ti.template(),
    w_: ti.template(),
    iteration: int,
    a: ti.template(),
    qdonor: ti.template(),
):
    """
    Rake-and-compress accumulation reading donor values from external field qdonor.

    Computes for each node the sum of transformed donor contributions where each edge's
    weight composes a along skipped nodes. Initialize weights as needed before calling.

    Typical use: compute pre-attenuation incoming flux by setting initial weights=1 and
    composing with a[did] during compress; accum is initialized to 0.

    Author: B.G.
    """
    for tid in accum:
        flip = gena.getSrc(src, tid, iteration)

        worked = False
        donors = ti.Vector([-1, -1, -1, -1])
        weights = ti.Vector([0.0, 0.0, 0.0, 0.0])
        todo = ndnr[tid] if not flip else ndnr_[tid]
        base = tid * 4
        acc_added = 0.0

        i = 0
        while i < todo and i < 4:
            if donors[i] == -1:
                if not flip:
                    donors[i] = dnr[base + i]
                    weights[i] = w[base + i]
                else:
                    donors[i] = dnr_[base + i]
                    weights[i] = w_[base + i]

            did = donors[i]

            flip_donor = gena.getSrc(src, did, iteration)
            ndnr_val = ndnr[did] if not flip_donor else ndnr_[did]

            if ndnr_val <= 1:
                if not worked:
                    acc_added = accum[tid] if not flip else accum_[tid]
                worked = True

                # Use external donor value
                acc_added += weights[i] * qdonor[did]

                if ndnr_val == 0:
                    todo -= 1
                    if todo > i and base + todo < cte.NX * cte.NY * 4:
                        if not flip:
                            donors[i] = dnr[base + todo]
                            weights[i] = w[base + todo]
                        else:
                            donors[i] = dnr_[base + todo]
                            weights[i] = w_[base + todo]
                    i -= 1
                else:
                    if not flip_donor:
                        donors[i] = dnr[did * 4]
                    else:
                        donors[i] = dnr_[did * 4]
                    # Compose with a at skipped node
                    weights[i] = weights[i] * a[did]
            i += 1

        if worked:
            if flip:
                ndnr[tid] = todo
                accum[tid] = acc_added
                for j in range(min(todo, 4)):
                    dnr[base + j] = donors[j]
                    w[base + j] = weights[j]
            else:
                ndnr_[tid] = todo
                accum_[tid] = acc_added
                for j in range(min(todo, 4)):
                    dnr_[base + j] = donors[j]
                    w_[base + j] = weights[j]
            gena.updateSrc(src, tid, iteration, flip)


@ti.kernel
def compute_loss_from_incoming(incoming: ti.template(), a: ti.template(), loss: ti.template()):
    """
    Compute per-node loss as (1 - a[i]) * incoming[i].

    Args:
        incoming: Pre-attenuation incoming flux at node i
        a: Per-node transmission coefficient
        loss: Output per-node loss/deposition

    Author: B.G.
    """
    for i in incoming:
        loss[i] = (1.0 - a[i]) * incoming[i]


@ti.kernel
def rake_compress_accum_affine_var(
    dnr: ti.template(),
    ndnr: ti.template(),
    p: ti.template(),
    src: ti.template(),
    dnr_: ti.template(),
    ndnr_: ti.template(),
    p_: ti.template(),
    w: ti.template(),
    w_: ti.template(),
    iteration: int,
    a: ti.template(),
    b: ti.template(),
):
    """
    Rake-and-compress accumulation with spatially varying a: q_out = a[i]*q_in + b[i].

    We store a cumulative per-edge weight in w/w_. For direct edges (d -> i), w = a[i].
    When compressing through an intermediate node `did`, we multiply w by a[did].

    Args mirror the constant-a variant, except `a` is a field.

    Author: B.G.
    """
    for tid in p:
        flip = gena.getSrc(src, tid, iteration)

        worked = False
        donors = ti.Vector([-1, -1, -1, -1])
        weights = ti.Vector([0.0, 0.0, 0.0, 0.0])
        todo = ndnr[tid] if not flip else ndnr_[tid]
        base = tid * 4
        p_added = 0.0

        i = 0
        while i < todo and i < 4:
            if donors[i] == -1:
                if not flip:
                    donors[i] = dnr[base + i]
                    weights[i] = w[base + i]
                else:
                    donors[i] = dnr_[base + i]
                    weights[i] = w_[base + i]

            did = donors[i]

            flip_donor = gena.getSrc(src, did, iteration)
            ndnr_val = ndnr[did] if not flip_donor else ndnr_[did]

            if ndnr_val <= 1:
                if not worked:
                    p_added = p[tid] if not flip else p_[tid]
                worked = True

                p_val = p[did] if not flip_donor else p_[did]
                p_added += weights[i] * p_val

                if ndnr_val == 0:
                    todo -= 1
                    if todo > i and base + todo < cte.NX * cte.NY * 4:
                        if not flip:
                            donors[i] = dnr[base + todo]
                            weights[i] = w[base + todo]
                        else:
                            donors[i] = dnr_[base + todo]
                            weights[i] = w_[base + todo]
                    i -= 1
                else:
                    if not flip_donor:
                        donors[i] = dnr[did * 4]
                    else:
                        donors[i] = dnr_[did * 4]
                    # Compose weight with a at the skipped node
                    weights[i] = weights[i] * a[did]
            i += 1

        if worked:
            if flip:
                ndnr[tid] = todo
                p[tid] = p_added
                for j in range(min(todo, 4)):
                    dnr[base + j] = donors[j]
                    w[base + j] = weights[j]
            else:
                ndnr_[tid] = todo
                p_[tid] = p_added
                for j in range(min(todo, 4)):
                    dnr_[base + j] = donors[j]
                    w_[base + j] = weights[j]
            gena.updateSrc(src, tid, iteration, flip)


@ti.kernel
def rake_compress_accum_affine_with_incoming(
    dnr: ti.template(),
    ndnr: ti.template(),
    p: ti.template(),
    incoming: ti.template(),
    src: ti.template(),
    dnr_: ti.template(),
    ndnr_: ti.template(),
    p_: ti.template(),
    incoming_: ti.template(),
    w: ti.template(),
    w_: ti.template(),
    iteration: int,
    a: ti.f32,
    b: ti.template(),
):
    """
    Constant-a affine rake+compress that also accumulates incoming pre-attenuation.

    incoming[i] accumulates sum of weights * q_donor (without adding b[i]).
    """
    for tid in p:
        flip = gena.getSrc(src, tid, iteration)

        worked = False
        donors = ti.Vector([-1, -1, -1, -1])
        weights = ti.Vector([0.0, 0.0, 0.0, 0.0])
        todo = ndnr[tid] if not flip else ndnr_[tid]
        base = tid * 4
        p_added = 0.0
        inc_added = 0.0

        i = 0
        while i < todo and i < 4:
            if donors[i] == -1:
                if not flip:
                    donors[i] = dnr[base + i]
                    weights[i] = w[base + i]
                else:
                    donors[i] = dnr_[base + i]
                    weights[i] = w_[base + i]

            did = donors[i]
            flip_donor = gena.getSrc(src, did, iteration)
            ndnr_val = ndnr[did] if not flip_donor else ndnr_[did]

            if ndnr_val <= 1:
                if not worked:
                    p_added = p[tid] if not flip else p_[tid]
                    inc_added = 0.0
                worked = True

                p_val = p[did] if not flip_donor else p_[did]
                contrib = weights[i] * p_val
                p_added += contrib
                inc_added += contrib

                if ndnr_val == 0:
                    todo -= 1
                    if todo > i and base + todo < cte.NX * cte.NY * 4:
                        if not flip:
                            donors[i] = dnr[base + todo]
                            weights[i] = w[base + todo]
                        else:
                            donors[i] = dnr_[base + todo]
                            weights[i] = w_[base + todo]
                    i -= 1
                else:
                    if not flip_donor:
                        donors[i] = dnr[did * 4]
                    else:
                        donors[i] = dnr_[did * 4]
                    weights[i] = weights[i] * a
            i += 1

        if worked:
            if flip:
                ndnr[tid] = todo
                p[tid] = p_added
                incoming[tid] = inc_added if (iteration == 0) else incoming[tid] + inc_added
                for j in range(min(todo, 4)):
                    dnr[base + j] = donors[j]
                    w[base + j] = weights[j]
            else:
                ndnr_[tid] = todo
                p_[tid] = p_added
                incoming_[tid] = (
                    inc_added if (iteration == 0) else incoming_[tid] + inc_added
                )
                for j in range(min(todo, 4)):
                    dnr_[base + j] = donors[j]
                    w_[base + j] = weights[j]
            gena.updateSrc(src, tid, iteration, flip)


@ti.kernel
def rake_compress_accum_affine_var_with_incoming(
    dnr: ti.template(),
    ndnr: ti.template(),
    p: ti.template(),
    incoming: ti.template(),
    src: ti.template(),
    dnr_: ti.template(),
    ndnr_: ti.template(),
    p_: ti.template(),
    incoming_: ti.template(),
    w: ti.template(),
    w_: ti.template(),
    iteration: int,
    a: ti.template(),
    b: ti.template(),
):
    """
    Variable-a affine rake+compress with incoming pre-attenuation accumulation.
    """
    for tid in p:
        flip = gena.getSrc(src, tid, iteration)

        worked = False
        donors = ti.Vector([-1, -1, -1, -1])
        weights = ti.Vector([0.0, 0.0, 0.0, 0.0])
        todo = ndnr[tid] if not flip else ndnr_[tid]
        base = tid * 4
        p_added = 0.0
        inc_added = 0.0

        i = 0
        while i < todo and i < 4:
            if donors[i] == -1:
                if not flip:
                    donors[i] = dnr[base + i]
                    weights[i] = w[base + i]
                else:
                    donors[i] = dnr_[base + i]
                    weights[i] = w_[base + i]

            did = donors[i]
            flip_donor = gena.getSrc(src, did, iteration)
            ndnr_val = ndnr[did] if not flip_donor else ndnr_[did]

            if ndnr_val <= 1:
                if not worked:
                    p_added = p[tid] if not flip else p_[tid]
                    inc_added = 0.0
                worked = True

                p_val = p[did] if not flip_donor else p_[did]
                contrib = weights[i] * p_val
                p_added += contrib
                inc_added += contrib

                if ndnr_val == 0:
                    todo -= 1
                    if todo > i and base + todo < cte.NX * cte.NY * 4:
                        if not flip:
                            donors[i] = dnr[base + todo]
                            weights[i] = w[base + todo]
                        else:
                            donors[i] = dnr_[base + todo]
                            weights[i] = w_[base + todo]
                    i -= 1
                else:
                    if not flip_donor:
                        donors[i] = dnr[did * 4]
                    else:
                        donors[i] = dnr_[did * 4]
                    weights[i] = weights[i] * a[did]
            i += 1

        if worked:
            if flip:
                ndnr[tid] = todo
                p[tid] = p_added
                incoming[tid] = inc_added if (iteration == 0) else incoming[tid] + inc_added
                for j in range(min(todo, 4)):
                    dnr[base + j] = donors[j]
                    w[base + j] = weights[j]
            else:
                ndnr_[tid] = todo
                p_[tid] = p_added
                incoming_[tid] = (
                    inc_added if (iteration == 0) else incoming_[tid] + inc_added
                )
                for j in range(min(todo, 4)):
                    dnr_[base + j] = donors[j]
                    w_[base + j] = weights[j]
            gena.updateSrc(src, tid, iteration, flip)
