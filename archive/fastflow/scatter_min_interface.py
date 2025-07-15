"""
Interface for scatter_min operations - Taichi port of CUDA scatter-min kernels

This module provides the scatter-min operations essential for depression routing (lakeflow).
It implements parallel reduction patterns to find minimum elevations and corresponding 
indices across different basins in the drainage network.

CUDA Correspondence:
- Direct 1:1 port of flow_cuda_scatter_min_atomic function
- Exact kernel sequence and parameter passing
- Same atomic operation patterns for thread-safe reductions
- Identical basin boundary detection logic

Mathematical Background:
Scatter-min operations solve the parallel reduction problem:
For each basin b, find: min(elevation[i] for all i where basin[i] == b)

This is a fundamental building block for depression routing where we need to:
1. Find the minimum elevation on each basin's boundary
2. Identify which neighboring basin offers the lowest escape route
3. Route flow from local minima to these optimal outlets

Algorithm Steps:
1. scatter_min_atomic: Find minimum boundary elevation per basin
2. scatter_argbasin_atomic: Find neighboring basin with minimum connection
3. scatter_argmin_atomic: Find exact cell index of the minimum location

Race Condition Handling:
All operations use atomic primitives to ensure thread safety:
- ti.atomic_min for finding minimum values
- Careful ordering of atomic operations to prevent inconsistent states
- Basin boundary checks to avoid invalid neighbor access

Memory Access Patterns:
- Each thread processes one grid cell
- Reads: elevation, basin assignment, neighbor values
- Writes: basin-indexed arrays (minh, argminh, argbasin)
- Optimal for GPU parallelization with high memory bandwidth

Performance Considerations:
- Atomic operations may serialize at high contention
- Memory coalescing optimized for row-major grid traversal
- Boundary checking prevents out-of-bounds access
- Workload balanced across all grid cells
"""
import taichi as ti
import math
from .kernels.scatter_min import (
    fillArray, scatter_min_atomic, scatter_argbasin_atomic, 
    scatter_argmin_atomic
)

def flow_cuda_scatter_min_atomic(zz, z, basin, rcv, minh, argminh, depression_fields, N, S_plus_1):
    """
    Execute parallel scatter-min operations for depression routing.
    
    This function orchestrates the three-stage scatter-min reduction process
    that identifies optimal drainage paths between basins. It's a critical
    component of the lakeflow algorithm that finds the lowest connections
    between depressions and their outlets.
    
    CUDA Correspondence:
    - Exact 1:1 port of flow_cuda_scatter_min_atomic function
    - Same kernel launch sequence with identical parameters
    - Preserves all atomic operation semantics
    - Maintains same numerical precision and edge case handling
    
    Algorithm Stages:
    
    Stage 1 - scatter_min_atomic:
    For each basin, find the minimum elevation among all boundary cells.
    Uses atomic_min operations to handle parallel writes safely.
    
    Stage 2 - scatter_argbasin_atomic:
    For each basin boundary cell at minimum elevation, find the neighboring
    basin that offers the lowest connection. This determines optimal flow
    routing between basins.
    
    Stage 3 - scatter_argmin_atomic:
    For each basin, identify the exact cell index that achieves the minimum
    boundary elevation. This provides the specific outlet location.
    
    Mathematical Foundation:
    The scatter-min operations solve these parallel reductions:
    
    minh[b] = min(z[i] for i where basin[i] == b and is_boundary(i))
    argbasin[b] = argmin(neighbor_basin[i] for i where basin[i] == b and z[i] == minh[b])
    argminh[b] = argmin(i for i where basin[i] == b and z[i] == minh[b])
    
    Where is_boundary(i) means cell i has at least one neighbor in a different basin.
    
    Race Condition Prevention:
    - atomic_min ensures thread-safe minimum finding
    - Deterministic tie-breaking using basin IDs
    - Careful memory ordering to prevent inconsistent states
    - Boundary checking prevents invalid memory access
    
    Memory Layout:
    - Input arrays: 1D flattened grid data (N elements)
    - Output arrays: basin-indexed (up to S+1 elements for S basins)
    - Working arrays: allocated in depression_fields for reuse
    
    Performance Optimization:
    - All kernels launch with full parallelism (one thread per cell)
    - Memory access patterns optimized for GPU coalescing
    - Atomic contention minimized by sparse basin boundaries
    - Working arrays reused to reduce allocation overhead
    
    Parameters:
    -----------
    zz : ti.Field
        Original elevation values (N elements, 1D flattened)
        Used for finding actual minimum elevations at boundaries
        
    z : ti.Field  
        Basin edge elevations (N elements, 1D flattened)
        Pre-computed boundary elevations for each basin
        
    basin : ti.Field
        Basin assignments (N elements, 1D flattened)
        Each cell assigned to basin ID (0 for non-basin cells)
        
    rcv : ti.Field
        Receiver indices (N elements, not used in kernels)
        Passed for compatibility with CUDA interface
        
    minh : ti.Field
        Output minimum heights per basin (S+1 elements)
        Will be filled with minimum boundary elevations
        
    argminh : ti.Field
        Output argmin indices per basin (S+1 elements)
        Will be filled with cell indices of minimum locations
        
    depression_fields : DepressionRoutingFields
        Container for working arrays (argbasin, nbasin)
        Pre-allocated to avoid memory allocation overhead
    
    Returns:
    --------
    argminh : ti.Field
        Basin-indexed array of minimum boundary cell indices
        Same as input argminh parameter (modified in-place)
    
    Side Effects:
    -------------
    - minh array filled with minimum elevations per basin
    - argminh array filled with corresponding cell indices
    - Working arrays in depression_fields modified
    
    Notes:
    ------
    - Assumes 4-connected grid topology (N, S, E, W neighbors)
    - Basin IDs must be positive integers (0 reserved for non-basin)
    - Grid resolution inferred as sqrt(N) for boundary checking
    - All arrays must be pre-allocated with correct sizes
    
    Example Usage:
    --------------
    # After basin assignment and edge elevation computation
    flow_cuda_scatter_min_atomic(
        original_elevation, basin_edge_elevation, basin_assignment,
        receivers, min_heights, argmin_indices, depression_fields
    )
    
    # Results available in min_heights and argmin_indices arrays
    """
    n = N  # Total number of grid cells
    m = S_plus_1  # Number of basins + 1 (for 0-indexing)
    res = int(math.sqrt(n))  # Grid resolution (assuming square grid)
    
    # Access pre-allocated working arrays from depression_fields
    # These arrays are reused across iterations to minimize memory allocation
    argbasin = depression_fields.argbasin  # Minimum neighboring basin per basin
    nbasin = depression_fields.nbasin      # Working array for basin neighbors
    
    # Stage 1: Initialize working arrays
    # Fill argbasin with large sentinel value to ensure proper min operations
    # Value 1000000 chosen to fit in i32 while being larger than any valid basin ID
    fillArray(argbasin, 1000000, m)
    
    # Stage 2: Execute the three scatter-min kernels in sequence
    
    # Kernel 1: Find minimum boundary elevation for each basin
    # For each cell in a basin, atomically update the basin's minimum elevation
    scatter_min_atomic(z, basin, minh, n)
    
    # Kernel 2: Find neighboring basin with minimum connection elevation
    # For boundary cells at minimum elevation, find the best neighboring basin
    scatter_argbasin_atomic(zz, z, basin, minh, argbasin, nbasin, res, n)
    
    # Kernel 3: Find exact cell index that achieves the minimum
    # For each basin, identify the specific cell at the minimum boundary elevation
    scatter_argmin_atomic(z, basin, minh, argminh, argbasin, nbasin, n)
    
    # Return the argmin array (also modified in-place)
    return argminh