"""
Flow accumulation interface - Taichi port of CUDA drainage calculation pipeline

This module implements the complete flow accumulation pipeline using the rake-compress
tree accumulation algorithm. It combines receiver computation with upward tree traversal
to calculate drainage areas efficiently on the GPU.

CUDA Correspondence:
- Direct port of the main.py drainage calculation pipeline
- 1:1 mapping with CUDA tree accumulation kernels
- Same rake-compress algorithm for O(log N) depth complexity
- Identical deterministic and randomized receiver options

Mathematical Background:
Flow accumulation computes the total upslope area draining to each cell using:
1. Receiver computation: determine flow directions (steepest descent)
2. Donor construction: invert receiver relationships to find contributors  
3. Tree accumulation: propagate drainage values up the flow tree

The rake-compress algorithm achieves optimal parallelism by:
- Compressing long paths in the flow tree
- Raking (accumulating) values at tree nodes
- Alternating between arrays to avoid data dependencies
- Converging in O(log N) iterations

Drainage Accumulation Formula:
drain[i] = sum(drain[j] for all j where receiver[j] leads to i)

This creates a tree where leaves are ridge cells and internal nodes aggregate 
upslope contributions.

Algorithm Complexity:
- Time: O(N log N) parallel, O(N²) sequential
- Space: O(N) for main arrays + O(N) working space
- Depth: O(log N) with optimal parallelization

Memory Layout:
- All arrays are 1D flattened from 2D grids
- Donor arrays use 4×N storage (max 4 donors per cell)
- Working arrays alternate between iterations
- Final result is drainage accumulation per cell

Performance Considerations:
- GPU-optimized with parallel tree operations
- Minimizes synchronization points between kernels
- Efficient memory access patterns for coalesced reads
- Handles irregular flow trees with load balancing
"""
import numpy as np
from .kernels.rcv import make_rcv, make_rcv_rand
from .kernels.tree_accum_up import rcv2donor, rake_compress_accum, fuse
from .kernels.parallel_scan import inclusive_scan, upsweep_step, downsweep_step, copy_input_to_work, set_zero, make_inclusive_and_copy
import taichi as ti
import math

def compute_drainage_accumulation(flow_fields, drain, method='deterministic', seed=42, compute_rcv = True):
    """
    Compute drainage accumulation using rake-compress tree algorithm.
    
    This is the main entry point for flow accumulation calculations, implementing
    the complete pipeline from receiver computation to final drainage values.
    It directly mirrors the CUDA main.py drainage calculation workflow.
    
    Algorithm Overview:
    1. Receiver Computation: Determine flow directions using steepest descent
    2. Donor Construction: Invert receiver graph to find upslope contributors
    3. Tree Accumulation: Use rake-compress to sum drainage areas efficiently
    4. Result Extraction: Copy final values back to output array
    
    CUDA Correspondence:
    - Exact port of main.py drainage calculation pipeline
    - Same kernel sequence and parameter passing
    - Identical numerical results for both deterministic/randomized modes
    - Preserves all boundary conditions and edge cases
    
    Rake-Compress Algorithm:
    The tree accumulation uses a work-efficient parallel algorithm:
    1. Initialize all cells with unit drainage (1.0)
    2. Build donor relationships (inverse of receiver graph)
    3. Iteratively compress flow paths and accumulate values
    4. Alternate between working arrays to avoid race conditions
    5. Fuse final results after log₂(N) iterations
    
    Mathematical Foundation:
    For each cell i, the final drainage is:
    drain[i] = 1 + sum(drain[j] for all j in upslope(i))
    
    Where upslope(i) is the set of all cells that eventually drain to i.
    
    Method Options:
    - 'deterministic': Uses steepest descent with tie-breaking
    - 'randomized': Uses weighted random selection based on gradients
    
    Performance Characteristics:
    - Parallel complexity: O(log N) depth, O(N log N) work
    - Memory usage: ~6N floats + 8N integers for working arrays
    - GPU utilization: High parallelism in all phases
    - Scalability: Efficient for large grids (tested up to 4096²)
    
    Race Condition Handling:
    - Donor construction uses atomic operations for thread safety
    - Tree accumulation alternates arrays to prevent read-write conflicts
    - No synchronization required within individual kernel launches
    
    Boundary Conditions:
    - Edge cells are treated as flow sinks (drain to themselves)
    - Interior drainage accumulates normally
    - No special handling for flat areas (uses numerical gradients)
    
    Parameters:
    -----------
    flow_fields : FlowComputeFields
        Pre-initialized field container with terrain data loaded.
        Must have z (elevation) and boundary fields set up.
        
    drain : numpy.ndarray
        External 2D output array (res × res) to store drainage results.
        Will be modified in-place with computed drainage values.
        
    method : str, default='deterministic'
        Flow routing method:
        - 'deterministic': Steepest descent routing
        - 'randomized': Weighted random routing based on gradients
        
    seed : int, default=42
        Random seed for 'randomized' method. Ensures reproducible results.
    
    Modifies:
    ---------
    drain : Output array filled with drainage accumulation values
    flow_fields.rcv : Receiver indices computed during process
    flow_fields.W : Flow weights computed during process
    flow_fields.p : Final drainage values (internal storage)
    
    Notes:
    ------
    - Assumes square grid with uniform cell spacing
    - Uses 4-connected flow (N, S, E, W neighbors only)
    - Numerical precision matches CUDA implementation
    - All intermediate arrays are reused for memory efficiency
    
    Example Usage:
    --------------
    # Initialize and load terrain
    flow_fields = FlowComputeFields(N)
    flow_fields.load_terrain(elevation_2d)
    flow_fields.set_boundary_edges()
    
    # Compute drainage accumulation
    drainage_output = np.zeros_like(elevation_2d)
    compute_drainage_accumulation(flow_fields, drainage_output)
    
    # drainage_output now contains drainage areas for each cell
    """

    if(compute_rcv):
        # Phase 1: Compute receivers using specified method
        # This determines the flow direction from each cell to its steepest neighbor
        if method == 'deterministic':
            # Deterministic steepest descent - consistent with CUDA make_rcv
            make_rcv(flow_fields.z, flow_fields.res, flow_fields.N, 
                    flow_fields.boundary, flow_fields.rcv, flow_fields.W)
        elif method == 'randomized':
            # Randomized routing based on gradient-weighted probabilities
            # Generates random values and passes to make_rcv_rand kernel
            np.random.seed(seed)
            rand_vals = np.random.random(flow_fields.N).astype(np.float32)
            flow_fields.rand_array.from_numpy(rand_vals)
            
            make_rcv_rand(flow_fields.z, flow_fields.res, flow_fields.N,
                         flow_fields.boundary, flow_fields.rcv, flow_fields.W, 
                         flow_fields.rand_array)
        else:
            raise ValueError("Method must be 'deterministic' or 'randomized'")
    
    # Phase 2: Initialize accumulation values to unity
    # Each cell starts with drainage value of 1.0 (representing itself)
    flow_fields.reset_accumulation()
    
    # Phase 3: Tree accumulation using rake-compress algorithm
    # Calculate the number of iterations needed: ceil(log₂(N))
    logn = int(math.ceil(math.log2(flow_fields.N)))
    
    # Initialize working arrays for donor construction
    flow_fields.ndnr.fill(0)  # Number of donors per cell
    flow_fields.src.fill(0)   # Source tracking for rake-compress
    
    # Phase 4: Build donor relationships (inverse of receiver graph)
    # For each cell, record which cells drain into it
    rcv2donor(flow_fields.rcv, flow_fields.dnr, flow_fields.ndnr, 
              flow_fields.N, flow_fields.res)
    
    # Phase 5: Rake-compress iterations for tree accumulation
    # Each iteration doubles the effective path length being compressed
    for i in range(logn):
        rake_compress_accum(flow_fields.dnr, flow_fields.ndnr, flow_fields.p, flow_fields.src,
                           flow_fields.dnr_, flow_fields.ndnr_, flow_fields.p_, 
                           flow_fields.N, i)
    
    # Phase 6: Final fuse step to consolidate results
    # Merge accumulated values from working arrays
    fuse(flow_fields.p, flow_fields.src, flow_fields.p_, flow_fields.N, logn + 1)
    
    # Phase 7: Copy result to external output array
    # Convert from internal 1D storage to external 2D format
    drainage_result = flow_fields.get_drainage_2d()
    drain[:] = drainage_result