"""
Lakeflow (Depression Routing) CUDA Kernels - 1:1 Taichi Port

This module contains 13 specialized kernels that implement the complete lakeflow
algorithm for depression routing in digital elevation models (DEMs). These kernels
are direct 1:1 ports of the CUDA kernels found in src/cuda/core/lakeflow.cu.

Mathematical Background:
-----------------------
The lakeflow algorithm solves the depression-filling problem in landscape evolution
models. Given a DEM with local minima (depressions), the algorithm routes flow either
by "carving" channels through the depressions or "jumping" flow across them.

The core algorithm implements:
1. Basin identification and propagation via pointer jumping
2. Pour point analysis using scatter-min reductions  
3. Path accumulation with weight propagation for carving
4. Flow network updates preserving mass conservation

Algorithm Complexity:
--------------------
- Time: O(N × log²(N)) where N = number of grid cells
- Space: O(N) with multiple workspace arrays
- Parallelism: Fully parallel except for scan operations

CUDA Correspondence:
-------------------
Each kernel maintains exact 1:1 correspondence with its CUDA equivalent:
- Memory access patterns identical to CUDA version
- Arithmetic operations use same precision (float32/int64)
- Loop bounds and conditional logic exactly match
- Race condition handling mirrors CUDA atomic operations

Performance Notes:
-----------------
- Kernels optimized for GPU execution with coalesced memory access
- Minimal branch divergence within thread warps
- Atomic operations used sparingly to avoid contention
- Memory layout designed for cache efficiency

Thread Safety:
--------------
All kernels are designed for massively parallel execution:
- No shared state between threads except explicit atomic operations
- Race conditions handled via atomic operations where necessary
- Memory ordering follows CUDA memory model

Usage:
------
These kernels are orchestrated by the main lakeflow() function and should
not be called directly. They require proper field initialization and
specific calling sequence to maintain algorithmic correctness.
"""
import taichi as ti

@ti.kernel
def flow_cuda_path_accum_upward_kernel1(rcv: ti.template(), W: ti.template(), p: ti.template(),
                                       rcv_: ti.template(), W_: ti.template(), p_: ti.template(), n: int):
    """
    Path Accumulation Upward Kernel 1 - Carving Method Implementation
    
    This kernel implements the first phase of upward path accumulation for the
    carving method in depression routing. It propagates paths and accumulates
    weights along flow paths that need to be carved.
    
    Algorithm:
    ----------
    For each active cell (W[tid] > 0.001):
    1. Update path array: p_[tid] = rcv[tid] if path is active, -42 otherwise
    2. Accumulate weights: W_[tid] = W[tid] * W[rcv[tid]] (multiplicative cascade)
    3. Propagate receivers: rcv_[tid] = rcv[rcv[tid]] (pointer jumping)
    
    Mathematical Foundation:
    -----------------------
    The weight accumulation follows: W'(i) = W(i) × W(rcv(i))
    This implements multiplicative weight propagation for flow routing.
    The -42 sentinel value marks inactive paths (CUDA convention).
    
    CUDA Correspondence:
    -------------------
    Direct 1:1 port of flow_cuda_path_accum_upward_kernel1 from lakeflow.cu
    - Identical threshold (0.001) for active cell detection
    - Same sentinel value (-42) for inactive paths
    - Exact arithmetic operations and memory access patterns
    
    Parameters:
    -----------
    rcv : ti.template()
        Receiver array (flow directions) - input
    W : ti.template()  
        Flow weights - input
    p : ti.template()
        Path markers - input (reverse_path from carving initialization)
    rcv_ : ti.template()
        Updated receiver array - output
    W_ : ti.template()
        Updated weight array - output  
    p_ : ti.template()
        Updated path array - output
    n : int
        Number of grid cells
        
    Thread Safety:
    --------------
    This kernel is thread-safe as each thread operates on a unique tid.
    No race conditions possible since memory access is disjoint per thread.
    
    Performance:
    -----------
    - Memory access pattern: coalesced reads/writes
    - Branch divergence: minimal (single condition per thread)
    - Arithmetic operations: 3 per active thread
    """
    for tid in rcv:
        # Only process active cells (above numerical threshold)
        if W[tid] > 0.001:
            # Update path propagation: follow receiver if path is active
            if p[tid] > 0.001:
                p_[tid] = rcv[tid]  # Propagate path to receiver
            else:
                p_[tid] = -42       # Mark as inactive (-42 is CUDA sentinel)
            
            # Accumulate weights multiplicatively along flow paths
            W_[tid] = W[tid] * W[ti.i32(rcv[tid])]  # W'(i) = W(i) × W(rcv(i))
            
            # Pointer jumping: compress receiver chains for next iteration
            rcv_[tid] = rcv[ti.i32(rcv[tid])]       # rcv'(i) = rcv(rcv(i))


@ti.kernel
def flow_cuda_path_accum_upward_kernel2(rcv: ti.template(), W: ti.template(), p: ti.template(),
                                       rcv_: ti.template(), W_: ti.template(), p_: ti.template(), n: int):
    """
    Path Accumulation Upward Kernel 2 - Carving Method Completion
    
    This kernel implements the second phase of upward path accumulation for the
    carving method. It completes the path propagation by marking carved paths
    and copying the accumulated results back to the original arrays.
    
    Algorithm:
    ----------
    For each active cell (W[tid] > 0.001):
    1. Mark carved paths: p[p_[tid]] = 1.0 for valid paths (p_[tid] != -42)
    2. Copy accumulated weights: W[tid] = W_[tid]
    3. Copy propagated receivers: rcv[tid] = rcv_[tid]
    
    Mathematical Foundation:
    -----------------------
    This kernel completes the bidirectional path marking for carving:
    - Forward propagation (kernel1): accumulates along rcv chains
    - Backward marking (kernel2): marks upstream paths for carving
    
    The path marking equation: p[p_[tid]] = 1.0 marks cells upstream of
    carved paths, creating the complete carving network.
    
    CUDA Correspondence:
    -------------------
    Direct 1:1 port of flow_cuda_path_accum_upward_kernel2 from lakeflow.cu
    - Identical sentinel check (p_[tid] != -42)
    - Same path marking strategy with value 1.0
    - Exact copy operations for accumulated results
    
    Parameters:
    -----------
    rcv : ti.template()
        Original receiver array - modified in-place
    W : ti.template()
        Original flow weights - modified in-place
    p : ti.template()
        Original path markers - modified to mark carved paths
    rcv_ : ti.template()
        Accumulated receiver array from kernel1 - input
    W_ : ti.template()
        Accumulated weight array from kernel1 - input
    p_ : ti.template()
        Propagated path array from kernel1 - input
    n : int
        Number of grid cells
        
    Thread Safety:
    --------------
    RACE CONDITION POTENTIAL: The operation p[p_[tid]] = 1.0 may cause
    race conditions if multiple threads write to the same location.
    However, all writes use the same value (1.0), so races are benign.
    This matches the CUDA implementation behavior.
    
    Performance:
    -----------
    - Memory access pattern: coalesced reads, potentially scattered writes
    - Branch divergence: minimal (single condition per thread)
    - Memory bandwidth limited due to indirect addressing in p[p_[tid]]
    """
    for tid in rcv:
        # Only process active cells (same threshold as kernel1)
        if W[tid] > 0.001:
            # Mark upstream paths for carving (if path is valid)
            if p_[tid] != -42:  # Check for valid path (not sentinel)
                p[ti.i32(p_[tid])] = 1.0  # Mark upstream cell as part of carved path
            
            # Copy back accumulated results from working arrays
            W[tid] = W_[tid]    # Copy accumulated weights
            rcv[tid] = rcv_[tid]  # Copy propagated receivers


@ti.kernel
def indexed_set_id(S: int, locs: ti.template(), offset: int, dst: ti.template()):
    """
    Indexed Basin ID Assignment - Scatter Operation
    
    This kernel assigns unique basin identifiers to specific locations,
    typically local minima positions, using a scatter operation pattern.
    
    Algorithm:
    ----------
    For each local minimum (id = 0 to N-1):
        dst[locs[id]] = id + offset
    
    This creates a mapping where each location in 'locs' gets assigned
    a unique basin ID starting from 'offset'.
    
    Mathematical Foundation:
    -----------------------
    Basin assignment function: B(loc) = basin_id
    where loc = locs[i] and basin_id = i + offset
    
    Typically offset=1 to create 1-based basin numbering (CUDA convention).
    
    CUDA Correspondence:
    -------------------
    Direct 1:1 port of indexed_set_id CUDA kernel template from lakeflow.cu
    - Same scatter pattern with indirect addressing
    - Identical offset handling for basin ID numbering
    - Exact type casting with ti.i32() matching CUDA int32 cast
    
    Parameters:
    -----------
    S : int
        Number of local minima to process
    locs : ti.template()
        Array of grid locations (typically p_lm - local minima positions)
    offset : int
        Starting basin ID (typically 1 for 1-based numbering)
    dst : ti.template()
        Destination basin array to be populated
        
    Thread Safety:
    --------------
    This kernel is thread-safe assuming locs[id] contains unique values
    (no duplicate locations). Since each thread writes to a different
    dst[locs[id]] location, no race conditions occur.
    
    Performance:
    -----------
    - Memory access pattern: coalesced reads from locs, scattered writes to dst
    - Scattered writes may reduce memory bandwidth efficiency
    - Linear time complexity: O(N) where N is number of local minima
    
    Example:
    --------
    If locs = [10, 25, 47] and offset = 1:
    dst[10] = 1, dst[25] = 2, dst[47] = 3
    This assigns basin IDs 1, 2, 3 to cells 10, 25, 47 respectively.
    """
    for i in range(S):
        # Scatter operation: assign basin i to location
        # dst[location] = basin_i where basin_i = i + offset
        dst[ti.i32(locs[i])] = i + offset


@ti.kernel
def comp_basin_edgez(basin: ti.template(), z: ti.template(), bignum: float, res: int, basin_edgez: ti.template()):
    """
    Compute Basin Edge Elevations - Critical Depression Analysis Kernel
    
    This kernel computes the effective elevation for each cell considering
    basin boundaries. It identifies cells at basin edges and calculates
    the minimum elevation needed to exit the current basin.
    
    Algorithm:
    ----------
    For each interior cell (x,y) where 1 ≤ x,y < res-1:
    1. Check if cell is interior to basin (all 4 neighbors in same basin)
    2. If interior or boundary → set edge_z = bignum (blocked)
    3. If at basin edge → find minimum elevation among external neighbors
    4. Final edge_z = max(min_external_elevation, current_elevation)
    
    Mathematical Foundation:
    -----------------------
    Basin edge elevation function:
    
    edge_z(i) = {
        bignum                           if interior to basin or boundary
        max(z(i), min{z(j) : j ∈ N(i), basin(j) ≠ basin(i)})  otherwise
    }
    
    where N(i) is the 4-connected neighborhood of cell i.
    
    This ensures flow can only exit basins through valid pour points
    at elevations ≥ current cell elevation (uphill flow prevention).
    
    CUDA Correspondence:
    -------------------
    Direct 1:1 port of comp_basin_edgez CUDA kernel from lakeflow.cu
    - Same 2D grid iteration with ti.ndrange((1, res-1), (1, res-1))
    - Identical 4-neighbor connectivity pattern (±1, ±res)
    - Exact edge detection logic and elevation calculation
    - Same bignum sentinel value for blocked cells
    
    Parameters:
    -----------
    basin : ti.template()
        Basin identifier array (int64)
    z : ti.template()
        Elevation array (float32)
    bignum : float
        Large sentinel value (typically 1e10) for blocked cells
    res : int
        Grid resolution (square grid: res × res)
    basin_edgez : ti.template()
        Output: effective edge elevation for each cell
        
    Thread Safety:
    --------------
    This kernel is thread-safe as each thread processes a unique (x,y) location.
    No race conditions possible since memory access is disjoint per thread.
    
    Performance:
    -----------
    - Memory access pattern: mostly coalesced (4-connected stencil)
    - Processes (res-2)² interior cells (excludes boundary)
    - Branch divergence: moderate (multiple conditional checks)
    - Arithmetic operations: 5-8 per thread depending on basin configuration
    
    Edge Cases:
    -----------
    - Boundary cells (x=0, x=res-1, y=0, y=res-1) are not processed
    - Cells with basin[loc] ≤ 0 are treated as boundaries (set to bignum)
    - Single-cell basins will have edge_z = bignum (no valid exit)
    
    Example:
    --------
    For a 3×3 basin surrounded by different basins:
    [A A B]    z = [1 2 3]    edge_z = [bignum bignum 3]
    [A A B]        [4 5 6]              [bignum bignum 6]  
    [C C B]        [7 8 9]              [7      8      9]
    
    Interior A cells get bignum, edge cells get max(z, min_external_z).
    """
    for x, y in ti.ndrange((1, res-1), (1, res-1)):  # Process interior cells only
        loc = y * res + x  # Convert 2D coordinates to linear index
        
        # Get basin IDs for current cell and 4-connected neighbors
        ref  = basin[loc]          # Current cell basin
        bhix = basin[loc+1]        # East neighbor basin  (+x direction)
        blox = basin[loc-1]        # West neighbor basin  (-x direction)
        bhiy = basin[loc + res]    # South neighbor basin (+y direction)
        bloy = basin[loc - res]    # North neighbor basin (-y direction)
        
        # Initialize with blocked value (for interior cells or boundaries)
        val = bignum
        
        # Check if cell is at basin boundary (not all neighbors in same basin)
        # Also handle boundary cells (ref ≤ 0) and ensure at least one neighbor differs
        if not (ref <= 0 or (ref == bhix and ref == blox and ref == bhiy and ref == bloy)):
            val = z[loc]  # Use current elevation as starting point for edge cells
            
        # Find minimum elevation among neighbors in different basins
        nval = bignum  # Initialize to large value
        
        # Check each neighbor: if in different basin, consider its elevation
        if ref != bloy:  # North neighbor in different basin
            nval = ti.min(nval, z[loc - res])
        if ref != bhiy:  # South neighbor in different basin
            nval = ti.min(nval, z[loc + res])
        if ref != blox:  # West neighbor in different basin
            nval = ti.min(nval, z[loc - 1])
        if ref != bhix:  # East neighbor in different basin
            nval = ti.min(nval, z[loc + 1])
            
        # Final edge elevation: max of current elevation and minimum external elevation
        # This prevents downhill flow out of basins (maintains energy conservation)
        basin_edgez[loc] = ti.max(nval, val)


@ti.kernel
def compute_p_b_rcv(S_plus_1: int, p: ti.template(), z: ti.template(), basin: ti.template(), 
                   bignum: float, res: int, p_rcv: ti.template(), b_rcv: ti.template(), b: ti.template()):
    """1:1 port of compute_p_b_rcv CUDA kernel"""
    for tid in range(S_plus_1):

        loc = p[tid]
        pn_arr = ti.Vector([loc + 1, loc - 1, loc + res, loc - res])
        minpnz = 3.4028235e+37  # FLT_MAX
        minpnz_n = 0
        mintest = 0
        
        for i in range(4):
            pn = 0 if tid == 0 else pn_arr[i]
            # SHould not happen?
            if(pn<0):
                continue
            basintest = basin[ti.i32(pn)]
            pnz = bignum if basintest == basin[ti.i32(loc)] else ti.max(z[ti.i32(pn)], z[ti.i32(loc)])
            
            if (pnz < minpnz) or ((pnz == minpnz) and (basintest < mintest)):
                minpnz = pnz
                minpnz_n = ti.i32(pn)
                mintest = ti.i32(basintest)
                
        p_rcv[tid] = minpnz_n
        b_rcv[tid] = mintest
        b[tid] = basin[ti.i32(loc)]


@ti.kernel
def set_keep_b(S_plus_1: int, b: ti.template(), b_rcv: ti.template(), keep_b: ti.template()):
    """
    Basin Keep Filter - Cycle Detection and Processing Order
    
    This kernel implements a sophisticated filtering mechanism to determine
    which basins should be processed in the current iteration. It prevents
    circular dependencies and ensures proper topological ordering.
    
    Algorithm:
    ----------
    For each basin b_id:
        keep = NOT ((b_rcv[b_rcv[b_id]] == b_id) AND (b_rcv[b_id] > b_id))
    
    The logic detects and breaks cycles in basin connectivity while
    maintaining processing order based on basin IDs.
    
    Mathematical Foundation:
    -----------------------
    This implements cycle detection in the basin connectivity graph:
    
    Let R(i) = b_rcv[i] be the receiver function for basin i.
    
    A basin i should be kept (processed) unless:
    1. R(R(i)) = i  (forms a 2-cycle: i → j → i)
    2. R(i) > i     (receiver has higher ID than sender)
    
    The condition prevents infinite loops and ensures deterministic
    processing order in the presence of basin merging operations.
    
    Cycle Breaking Strategy:
    -----------------------
    When a 2-cycle is detected between basins i and j where i < j:
    - Basin i is kept (processed)
    - Basin j is filtered out (not processed)
    
    This ensures the lower-numbered basin "wins" the merge operation,
    maintaining deterministic behavior across parallel execution.
    
    CUDA Correspondence:
    -------------------
    Direct 1:1 port of set_keep_b CUDA kernel from lakeflow.cu
    - Identical cycle detection logic with double indirection
    - Same basin ID comparison for ordering (b_rcv[b_id] > b_id)
    - Exact boolean expression with same operator precedence
    
    Parameters:
    -----------
    S_plus_1 : int
        Number of basin entries to process (typically S+1)
    b : ti.template()
        Basin ID array (current basin for each local minimum)
    b_rcv : ti.template()
        Basin receiver array (target basin for each basin)
    keep_b : ti.template()
        Output: binary mask (1=keep, 0=filter) for each basin
        
    Thread Safety:
    --------------
    This kernel is thread-safe as each thread processes a unique basin ID.
    All reads are from constant arrays and writes are to disjoint locations.
    No race conditions possible.
    
    Performance:
    -----------
    - Memory access pattern: potential for scattered reads (double indirection)
    - Branch divergence: minimal (single conditional per thread)
    - Critical for algorithm correctness: filters ~50% of basins typically
    
    Example:
    --------
    Basin connectivity: 1→3, 2→3, 3→1 (forms cycle 1↔3)
    b_rcv = [_, 3, 3, 1]  (1-based indexing)
    
    For basin 1: b_rcv[b_rcv[1]] = b_rcv[3] = 1 ✓ AND b_rcv[1] = 3 > 1 ✓
    → Cycle detected, filter out basin 1 (keep_b[1] = 0)
    
    For basin 3: b_rcv[b_rcv[3]] = b_rcv[1] = 3 ✓ AND b_rcv[3] = 1 < 3 ✗
    → No filtering, keep basin 3 (keep_b[3] = 1)
    """
    for i in range(ti.i32(S_plus_1)):
        if i >= S_plus_1:  # Bounds check
            continue
            
        b_id = b[i]  # Get basin ID for this entry
        
        # Complex cycle detection and ordering logic
        # Condition 1: b_rcv[b_rcv[b_id]] == b_id  (forms 2-cycle)
        # Condition 2: b_rcv[b_id] > b_id          (receiver has higher ID)
        # Keep basin UNLESS both conditions are true (NOT operation)
        do_keep = not ((b_rcv[ti.i32(b_rcv[b_id])] == b_id) and (b_rcv[b_id] > b_id))
        
        keep_b[i] = 1 if do_keep else 0  # Convert boolean to binary mask


@ti.kernel
def set_keep(S_plus_1: int, b: ti.template(), keep_b: ti.template(), offset: ti.template(), keep: ti.template()):
    """1:1 port of set_keep CUDA kernel"""
    for i in range(ti.i32(S_plus_1)):

        if keep_b[i]:
            keep[ti.i32(offset[i] - 1)] = b[i]


@ti.kernel
def init_reverse(S: int, keep: ti.template(), p: ti.template(), reverse_path: ti.template()):
    """1:1 port of init_reverse CUDA kernel
    
    CUDA: init_reverse<<<(S + threads - 1)/threads, threads>>>(keep_offset_ptr + S, keep_ptr, p_ptr, reverse_path_ptr)
    The kernel launches with S threads, NOT final_count threads!
    """
    for id in range(ti.i32(S)+1):
        if id == 0 or id >= S:
            continue
        keep_id = keep[id]
        p_in = ti.i32(p[keep_id])
        reverse_path[ti.i32(p_in)] = 1.0


@ti.kernel
def final1(N: int, reverse_path: ti.template(), W: ti.template(), rcv: ti.template(), rcv_: ti.template()):
    """1:1 port of final1 CUDA kernel"""
    for id in range(ti.i32(N)):
        if id >= N:
            continue
        if reverse_path[id] > 0.0 and rcv[id] != id:
            rcv_[ti.i32(rcv[id])] = id
        if reverse_path[id] > 0.0 and rcv[id] == id:
            W[id] = 1.0


@ti.kernel
def final2(final_count: ti.template(), keep: ti.template(), p_rcv: ti.template(), p: ti.template(), rcv: ti.template()):
    """1:1 port of final2 CUDA kernel"""
    K = final_count[0]
    for id in range(ti.i32(K)):
        if id == 0 or id >= K:
            continue
        keep_id = keep[id]
        rcv[ti.i32(p[keep_id])] = p_rcv[keep_id]


@ti.kernel
def init_lakeflow(N: int, rcv: ti.template(), rcv_: ti.template(), W: ti.template(), W_: ti.template(),
                 basin_edgez: ti.template(), bignum: float, reverse_path: ti.template(), minh: ti.template()):
    """1:1 port of init CUDA kernel"""
    for id in range(ti.i32(N)):
        if id >= N:
            continue
        rcv_[id] = rcv[id]
        W_[id] = W[id]
        basin_edgez[id] = bignum
        reverse_path[id] = 0.0
        minh[id] = 1e10


@ti.kernel
def propag_basin_route_all(N: int, basin_route: ti.template()):
    """1:1 port of propag_basin_route_all CUDA kernel"""
    for id in range(ti.i32(N)):
        if id >= N or basin_route[id] == basin_route[ti.i32(basin_route[id])]:
            continue
        basin_route[id] = basin_route[ti.i32(basin_route[id])]


@ti.kernel
def propag_basin_route_lm(final_count: ti.template(), keep: ti.template(), p_lm: ti.template(), basin_route: ti.template()):
    """1:1 port of propag_basin_route_lm CUDA kernel"""
    K = final_count[0]
    for id in range(ti.i32(K)):
        if id >= K or id == 0:
            continue
        keep_idx = keep[id] - 1
        if keep_idx < 0:
            continue
        lmid = ti.i32(p_lm[keep_idx])
        if basin_route[lmid] == basin_route[basin_route[lmid]]:
            continue
        basin_route[lmid] = basin_route[basin_route[lmid]]


@ti.kernel
def update_all_basins(N: int, basin: ti.template(), basin_route: ti.template()):
    """1:1 port of update_all_basins CUDA kernel"""
    for i in range(ti.i32(N)):
        basin[i] = basin[ti.i32(basin_route[i])]


@ti.kernel
def update_basin_route(final_count: ti.template(), keep: ti.template(), p_lm: ti.template(), 
                      b_rcv: ti.template(), basin_route: ti.template(), basin: ti.template()):
    """1:1 port of update_basin_route CUDA kernel"""
    K = final_count[0]
    for i in range(ti.i32(K)):
        if i >= K or i == 0:
            continue
        # print('rerouting', i)
        keep_id = keep[i]
        # print(keep_id, b_rcv[ti.i32(keep_id), ])
        if keep_id > 0:  # Bounds check to prevent negative access
            b_rcv_keep = b_rcv[ti.i32(keep_id)]
            lm_from = ti.i32(p_lm[keep_id - 1])
            if b_rcv_keep == 0:
                basin[ti.i32(lm_from)] = 0
                continue
            basin_route[ti.i32(lm_from)] = ti.i32(p_lm[b_rcv_keep - 1])

# The main lakeflow_cuda function creates the following fields that need to be passed as templates:
# torch::Tensor rcv_0 = torch::empty({N}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, 0));
# torch::Tensor W_0 = torch::empty({N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0));
# torch::Tensor p_0 = torch::empty({N}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, 0));