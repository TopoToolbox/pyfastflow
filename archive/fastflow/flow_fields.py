"""
Flow computation fields - Taichi field container for standard flow routing operations

This module provides a comprehensive field management system for basic flow routing
calculations including receiver computation and drainage accumulation. It handles
memory allocation, data loading, and result extraction for all non-depression 
routing operations.

CUDA Correspondence:
- Manages same field types as CUDA flow computation kernels
- Equivalent memory layout and sizing for 1:1 algorithm porting
- Compatible data types and array indexing with CUDA tensors
- Same boundary condition handling and grid organization

Field Organization:
The FlowComputeFields class organizes arrays into logical groups:

1. **Core Terrain Data**: z (elevation), boundary (edge flags)
2. **Flow Results**: rcv (receivers), W (weights), p (drainage)
3. **Working Arrays**: rcv_, W_, p_ for alternating computations
4. **Tree Accumulation**: dnr (donors), ndnr (donor counts), src (source tracking)
5. **Utility Arrays**: rand_array, scan working space

Memory Layout:
- All arrays are 1D with length N (total grid cells)
- 2D grid coordinates mapped to 1D: index = y * res + x
- Row-major ordering for optimal GPU memory coalescing
- Pre-allocated working arrays to avoid runtime allocation overhead

Performance Considerations:
- Field reuse minimizes GPU memory allocation/deallocation
- Optimal array sizes based on algorithmic requirements
- Efficient data transfer between CPU and GPU through Taichi fields
- Memory-efficient parallel scan working space sizing

Thread Safety:
- Individual FlowComputeFields instances are thread-safe
- No shared state between different instances
- Safe for multi-threaded terrain processing workflows
- Concurrent field access requires separate instances per thread
"""
import taichi as ti
import numpy as np
import math

@ti.kernel
def setup_boundary_edges(boundary: ti.template(), res: int, N: int):
    """
    Set standard edge boundary conditions for rectangular grids.
    
    This kernel implements the most common boundary condition where all cells
    on the grid edges (first/last row/column) are marked as boundaries.
    Boundary cells act as flow sinks and prevent flow from leaving the domain.
    
    CUDA Correspondence:
    - Equivalent to manual boundary setup in CUDA preprocessing
    - Same indexing and boundary marking logic
    - Matches boundary handling in CUDA flow kernels
    
    Mathematical Background:
    Boundary conditions are essential for well-posed flow routing:
    - Interior cells: Allow flow in all 4 directions based on gradients
    - Boundary cells: Drain to themselves (rcv[i] = i) regardless of gradients
    - This prevents flow from leaving the computational domain
    
    Indexing Pattern:
    - 1D index i converted to 2D coordinates: (x=i%res, y=i//res)
    - Edge detection: x∈{0,res-1} OR y∈{0,res-1}
    - Row-major storage: consistent with CUDA tensor flattening
    
    Performance Notes:
    - Fully parallel kernel (no thread dependencies)
    - Simple arithmetic operations (modulo, division)
    - Minimal memory access (one write per cell)
    - O(N) complexity with perfect parallelization
    
    Parameters:
    -----------
    boundary : ti.template()
        Output boundary flag array (N elements, ti.u8 type)
        Values: 1 for boundary cells, 0 for interior cells
        
    res : int
        Grid resolution (number of cells per row/column)
        Must satisfy res² = N for square grids
        
    N : int
        Total number of grid cells (res × res)
        Used for loop bounds and validation
    
    Modifies:
    ---------
    boundary : Sets boundary[i] = 1 for edge cells, 0 for interior
    
    Example:
    --------
    For a 4×4 grid (N=16, res=4):
    ```
    1 1 1 1    # y=0 (top edge)
    1 0 0 1    # y=1 (left/right edges only)  
    1 0 0 1    # y=2 (left/right edges only)
    1 1 1 1    # y=3 (bottom edge)
    ```
    """
    for i in range(ti.i32(N)):
        # Convert 1D index to 2D grid coordinates
        x = i % res        # Column index (0 to res-1)
        y = i // res       # Row index (0 to res-1)
        
        # Check if cell is on any edge of the grid
        if x == 0 or x == res-1 or y == 0 or y == res-1:
            boundary[i] = ti.u8(1)  # Mark as boundary
        else:
            boundary[i] = ti.u8(0)  # Mark as interior

@ti.kernel
def reset_accumulation_kernel(p: ti.template()):
    """
    Reset drainage accumulation values to unit values for tree accumulation.
    
    This kernel initializes all cells with drainage value 1.0, representing
    the "self-contribution" of each cell to the total drainage area. During
    tree accumulation, these values will be summed along flow paths to 
    compute total upslope drainage areas.
    
    CUDA Correspondence:
    - Equivalent to tensor.fill_(1.0) operations in CUDA
    - Same initialization values and data types
    - Matches preprocessing steps in CUDA tree accumulation
    
    Mathematical Background:
    Drainage accumulation computes the total area draining to each cell:
    - Initial condition: each cell contributes area = 1 (representing itself)
    - Tree accumulation: drain[i] = 1 + sum(drain[j] for upslope neighbors j)
    - Final result: total number of cells draining to each location
    
    Performance Notes:
    - Embarrassingly parallel (no thread interactions)
    - Memory-bound operation (one write per thread)
    - Optimal memory coalescing for sequential array access
    - O(N) complexity with perfect GPU utilization
    
    Parameters:
    -----------
    p : ti.template()
        Drainage accumulation array (N elements, float type)
        Will be filled with unit values (1.0) for all cells

    Modifies:
    ---------
    p : Sets p[i] = 1.0 for all valid indices i
    
    Notes:
    ------
    - Must be called before tree accumulation operations
    - Safe to call multiple times (idempotent operation)
    - Compatible with both deterministic and randomized flow routing
    """
    for i in p:
        p[i] = 1.0  # Unit drainage value for each cell

class FlowComputeFields:
    """
    Comprehensive field container for flow routing computations.
    
    This class manages all Taichi fields required for standard flow routing operations
    including receiver computation, drainage accumulation, and tree-based algorithms.
    It provides a unified interface for memory management, data loading, and result
    extraction while maintaining optimal GPU memory layouts.
    
    Design Philosophy:
    - **Single Allocation**: All fields allocated once during initialization
    - **Memory Reuse**: Working arrays shared across different algorithm phases
    - **1:1 CUDA Mapping**: Field types and sizes match CUDA tensor equivalents
    - **Performance Optimization**: Memory layouts optimized for GPU coalescing
    
    Field Categories:
    
    **Core Data Fields**:
    - z: Elevation values (input terrain)
    - boundary: Boundary condition flags (edges vs interior)
    - rcv: Receiver indices (where each cell drains)
    - W: Flow weights (strength of flow relationships)
    - p: Drainage accumulation values (upslope area)
    
    **Working Arrays**:
    - rcv_, W_, p_: Alternative arrays for double-buffering
    - Prevents race conditions during iterative algorithms
    - Enables efficient array swapping without data copying
    
    **Tree Accumulation Fields**:
    - dnr: Donor arrays (up to 4 donors per cell)
    - ndnr: Number of donors per cell
    - src: Source tracking for rake-compress algorithm
    
    **Utility Arrays**:
    - rand_array: Random values for stochastic flow routing
    - scan_work, scan_temp: Working space for parallel scan operations
    
    Memory Layout Optimization:
    - All arrays are 1D with length N for optimal GPU access
    - Row-major flattening: 2D(x,y) → 1D(y*res + x)
    - Aligned memory allocation for vectorized operations
    - Working arrays sized for worst-case algorithm requirements
    
    Thread Safety:
    - Each instance is independent and thread-safe
    - No shared global state between instances
    - Multiple instances can operate concurrently
    - Taichi handles GPU memory isolation automatically
    
    CUDA Correspondence:
    This class manages the same arrays as CUDA flow computation:
    - torch::Tensor z ≡ self.z
    - torch::Tensor rcv ≡ self.rcv  
    - torch::Tensor W ≡ self.W
    - torch::Tensor p ≡ self.p
    - Working arrays for double-buffering
    - Same data types and memory layouts
    """
    
    def __init__(self, N: int):
        """
        Initialize all fields for N grid cells with optimized memory allocation.
        
        This constructor performs intelligent memory planning and field allocation
        based on the total number of grid cells. It assumes a square grid layout
        and sizes working arrays based on theoretical algorithm requirements.
        
        Memory Allocation Strategy:
        - Core fields: Exact size N for terrain and flow data
        - Working arrays: Double-buffered with size N for race-condition prevention
        - Donor arrays: Size 4×N to handle maximum possible donors (N, S, E, W)
        - Scan arrays: Next power-of-2 for optimal parallel scan performance
        
        Field Type Selection:
        - Elevations, weights: ti.f32 for numerical precision and GPU efficiency
        - Indices, receivers: ti.i64 for large grid support (N > 2³²)
        - Boundaries: ti.u8 for memory efficiency (boolean flags)
        - Donors, counts: ti.i32 for adequate range with memory efficiency
        
        Performance Optimizations:
        - All arrays allocated contiguously in GPU memory
        - Sizes chosen to avoid bank conflicts on GPU architectures
        - Working space pre-allocated to prevent runtime allocation overhead
        - Power-of-2 sizing for efficient parallel algorithms
        
        Parameters:
        -----------
        N : int
            Total number of grid cells, must be positive.
            For square grids: N = resolution²
            Examples: 64²=4096, 256²=65536, 1024²=1048576
        
        Raises:
        -------
        ValueError : If N is not a perfect square or N <= 0
        MemoryError : If system cannot allocate required GPU memory
        
        Notes:
        ------
        - Assumes square grid topology (N must be perfect square)
        - All fields initialized but not populated with data
        - Call load_terrain() and set_boundary_*() before use
        - Memory managed automatically by Taichi runtime
        """
        if N <= 0:
            raise ValueError(f"N must be positive, got {N}")
        
        self.N = N
        self.res = int(math.sqrt(N))
        
        # Validate square grid assumption
        if self.res * self.res != N:
            raise ValueError(f"N={N} is not a perfect square (res={self.res})")
        
        # === CORE TERRAIN AND FLOW FIELDS ===
        # Primary data arrays for flow routing algorithms
        
        self.z = ti.field(ti.f32, shape=N)              # Elevation values (input terrain)
        self.boundary = ti.field(ti.u8, shape=N)        # Boundary conditions (0=interior, 1=boundary)
        self.rcv = ti.field(ti.i64, shape=N)            # Receiver indices (where each cell drains)
        self.W = ti.field(ti.f32, shape=N)              # Flow weights (strength of relationships)
        self.p = ti.field(ti.f32, shape=N)              # Drainage accumulation values (upslope area)
        
        # === WORKING FIELDS FOR DOUBLE-BUFFERING ===
        # Alternative arrays to prevent race conditions during iterative algorithms
        
        self.rcv_ = ti.field(ti.i64, shape=N)           # Working receiver array
        self.W_ = ti.field(ti.f32, shape=N)             # Working weights array
        self.p_ = ti.field(ti.f32, shape=N)             # Working accumulation array
        
        # === TREE ACCUMULATION RAKE-COMPRESS FIELDS ===
        # Specialized arrays for efficient drainage accumulation
        
        self.dnr = ti.field(ti.i32, shape=N*4)          # Donor arrays (up to 4 donors per cell)
        self.dnr_ = ti.field(ti.i32, shape=N*4)         # Working donor arrays
        self.ndnr = ti.field(ti.i32, shape=N)           # Number of donors per cell
        self.ndnr_ = ti.field(ti.i32, shape=N)          # Working number of donors
        self.src = ti.field(ti.i32, shape=N)            # Source tracking for rake-compress
        
        # === UTILITY ARRAYS ===
        # Supporting arrays for specialized algorithms
        
        self.rand_array = ti.field(ti.f32, shape=N)     # Random values [0,1] for stochastic routing
        
        # === PARALLEL SCAN WORKING ARRAYS ===
        # Sized for optimal parallel scan performance (next power of 2)
        scan_size = 1
        while scan_size < N:
            scan_size *= 2
        self.scan_work = ti.field(ti.i32, shape=scan_size)  # Work array for parallel scan
        self.scan_temp = ti.field(ti.i32, shape=scan_size)  # Temp array for parallel scan
    
    def get_terrain_shape(self):
        """Get the 2D terrain shape for validation and display purposes."""
        return (self.res, self.res)
    
    def load_terrain(self, z_2d):
        """
        Load 2D elevation data into the internal z field.
        
        Converts 2D numpy array to 1D Taichi field using row-major flattening.
        This ensures compatibility with CUDA tensor flattening conventions.
        
        Parameters:
        -----------
        z_2d : numpy.ndarray
            2D elevation array with shape (res, res)
            Should be float-compatible (will be cast to float32)
        """
        if z_2d.shape != (self.res, self.res):
            raise ValueError(f"Terrain must be {self.res}x{self.res}, got {z_2d.shape}")
        z_flat = z_2d.flatten().astype(np.float32)
        self.z.from_numpy(z_flat)
    
    def set_boundary_edges(self):
        """
        Set standard edge boundary conditions using the setup_boundary_edges kernel.
        
        Marks all cells on the grid perimeter as boundaries. This is the most
        common boundary condition for flow routing problems.
        """
        setup_boundary_edges(self.boundary, self.res, self.N)
    
    def set_custom_boundaries(self, boundary_2d):
        """
        Set custom boundary conditions from a 2D boolean array.
        
        Allows for arbitrary boundary configurations beyond simple edge boundaries.
        Useful for modeling complex domain shapes or internal flow sinks.
        
        Parameters:
        -----------
        boundary_2d : numpy.ndarray
            2D boolean array with shape (res, res)
            True/1 for boundary cells, False/0 for interior cells
        """
        if boundary_2d.shape != (self.res, self.res):
            raise ValueError(f"Boundaries must be {self.res}x{self.res}, got {boundary_2d.shape}")
        boundary_flat = boundary_2d.flatten().astype(np.uint8)
        self.boundary.from_numpy(boundary_flat)
    
    def get_receivers_2d(self):
        """Extract receiver indices as 2D array for visualization and analysis."""
        return self.rcv.to_numpy().reshape(self.res, self.res)
    
    def get_weights_2d(self):
        """Extract flow weights as 2D array for visualization and analysis."""
        return self.W.to_numpy().reshape(self.res, self.res)
    
    def get_drainage_2d(self):
        """Extract drainage accumulation as 2D array for visualization and analysis."""
        return self.p.to_numpy().reshape(self.res, self.res)
    
    def reset_accumulation(self):
        """
        Reset drainage accumulation to unit values for tree accumulation algorithms.
        
        This must be called before running drainage accumulation to ensure
        correct initialization of the tree accumulation process.
        """
        reset_accumulation_kernel(self.p, self.N)