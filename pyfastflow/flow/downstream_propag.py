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
