"""
Simple receiver computation interface - Taichi port of CUDA flow routing

This module provides a high-level interface for computing receiver indices and flow weights
from elevation data. It serves as the entry point for flow routing calculations without
depression handling (no lakeflow).

CUDA Correspondence:
- This corresponds to calling the make_rcv CUDA kernel directly
- 1:1 functional mapping with src/cuda/core/rcv.cu
- Uses the same steepest-descent algorithm for receiver selection

Mathematical Background:
The receiver calculation implements a steepest-descent flow routing model where:
1. For each cell, examine 4-connected neighbors (N, S, E, W)
2. Calculate elevation differences (gradients) to each downhill neighbor
3. Compute normalized flow weights based on gradient magnitudes
4. Select the receiver as the neighbor with maximum flow weight

Flow Weight Calculation:
W_i = max(0, z_center - z_neighbor_i) / sum(max(0, z_center - z_neighbor_j))

Where the sum is over all downhill neighbors. The steepest neighbor becomes the receiver.

Memory Layout:
- Input: z (elevation), boundary (edge flags)  
- Output: rcv (receiver indices), W (flow weights)
- All arrays are 1D flattened from 2D grid in row-major order

Performance Considerations:
- Fully parallel kernel with no dependencies between cells
- Memory access pattern: each thread reads 5 values (center + 4 neighbors)
- Boundary checking prevents out-of-bounds access
- Computational complexity: O(N) where N = number of grid cells
"""
from .kernels.rcv import make_rcv

def compute_receivers(flow_fields):
    """
    Compute receivers in place for FlowComputeFields using steepest-descent routing.
    
    This function implements the core flow routing algorithm that determines where
    water flows from each cell based on topographic gradients. It directly maps
    to the CUDA make_rcv kernel with identical mathematical behavior.
    
    Algorithm Steps:
    1. For each grid cell, examine its 4-connected neighbors (N, S, E, W)
    2. Calculate elevation differences to determine downhill neighbors
    3. Compute flow weights based on gradient magnitudes
    4. Select the steepest neighbor as the receiver
    5. Handle boundary conditions (edge cells drain to themselves)
    
    CUDA Mapping:
    - Directly calls make_rcv kernel (rcv.py) 
    - Same thread-per-cell parallelization strategy
    - Identical boundary handling and weight calculation
    - Preserves numerical precision and edge case behavior
    
    Memory Access Pattern:
    - Each cell reads: elevation[self] + elevation[4 neighbors]
    - Each cell writes: receiver[self] + weight[self]
    - No race conditions (each thread writes to unique locations)
    - Optimal memory coalescing for row-major traversal
    
    Boundary Conditions:
    - Edge cells (x=0, x=res-1, y=0, y=res-1) are marked as boundaries
    - Boundary cells drain to themselves (rcv[i] = i)
    - This prevents flow from leaving the model domain
    
    Parameters:
    -----------
    flow_fields : FlowComputeFields
        Pre-initialized field container with:
        - z: elevation data (must be loaded)
        - boundary: boundary flags (must be set)
        - rcv: receiver array (will be filled)
        - W: flow weight array (will be filled)
        - res: grid resolution 
        - N: total number of cells (res²)
    
    Modifies:
    ---------
    flow_fields.rcv : Modified in-place with receiver indices
    flow_fields.W : Modified in-place with flow weights
    
    Notes:
    ------
    - Assumes square grid with 4-connectivity
    - Uses deterministic tie-breaking (consistent with CUDA version)
    - No depression handling - use lakeflow for filled routing
    - Optimized for GPU execution with parallel threads
    
    Example Usage:
    --------------
    # Load terrain data
    flow_fields.load_terrain(elevation_2d)
    flow_fields.set_boundary_edges()
    
    # Compute flow routing
    compute_receivers(flow_fields)
    
    # Access results
    receivers_2d = flow_fields.get_receivers_2d()
    weights_2d = flow_fields.get_weights_2d()
    """
    res = flow_fields.res
    N = flow_fields.N
    
    # Call the core receiver computation kernel
    # This is a 1:1 mapping to the CUDA make_rcv function
    make_rcv(flow_fields.z, res, N, flow_fields.boundary, flow_fields.rcv, flow_fields.W)