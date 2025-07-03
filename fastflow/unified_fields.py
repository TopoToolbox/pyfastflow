"""
Unified Flow Field Management - Single Container for Flow and Depression Routing
===============================================================================

This module provides a unified field management system that combines both standard
flow routing and depression routing (lakeflow) operations in a single container.
This design matches the CUDA implementation where the same workspace is shared
between flow computation and depression routing phases.

Design Philosophy:
-----------------
- **Shared Workspace**: Same fields used for both flow and depression routing
- **On-demand Allocation**: Depression-specific fields only allocated when needed
- **CUDA Correspondence**: 1:1 mapping with CUDA tensor organization
- **Memory Efficiency**: Eliminates field duplication between containers

Field Organization:
------------------
**Core Fields** (always allocated):
- z, boundary: Terrain and boundary conditions
- rcv, W, p: Primary flow routing results
- rcv_, W_, p_: Working arrays for double-buffering

**Depression-Only Fields** (allocated on-demand):
- basin, basin_route, basin_edgez: Basin management
- p_lm, argminh, minh: Local minima and scatter-min operations
- keep arrays, reverse_path: Path tracking and filtering
- All workspace arrays for lakeflow algorithm

CUDA Parameter Mapping:
----------------------
Based on order.lakeflow_cuda() call with 24 parameters:

| CUDA Parameter    | Type           | Unified Field Name | Purpose                    |
|-------------------|----------------|-------------------|----------------------------|
| N, S, res, B      | int           | dimensions        | Grid and count parameters  |
| p_lm              | int64[]       | p_lm              | Local minima positions     |
| rcv, rcv_         | int64[]       | rcv, rcv_         | Receivers (current/working)|
| W, W_             | float32[]     | W, W_             | Weights (current/working)  |
| basin             | int64[]       | basin             | Basin ID assignments       |
| basin_route       | int64[]       | basin_route       | Basin routing connections  |
| basin_edgez       | float32[]     | basin_edgez       | Basin edge elevations     |
| bound_ind         | int64[]       | bound_ind         | Boundary indices          |
| z                 | float32[]     | z                 | Elevation data            |
| argminh_space     | int64[]       | argminh           | Min height arguments      |
| minh_space        | float32[]     | minh              | Min height values         |
| p_rcv_space       | int64[]       | p_rcv             | Path receivers            |
| b_rcv_space       | int64[]       | b_rcv             | Basin receivers           |
| b_space           | int64[]       | b                 | Basin workspace           |
| keep_space        | int32[]       | keep_space        | Keep flags (3*(N+1) size) |
| reverse_path      | float32[]     | reverse_path      | Reverse path tracking     |

Memory Layout:
-------------
Core fields:     ~64 bytes per cell (8 arrays × 8 bytes)
Depression:      ~240 bytes per cell (30 arrays × 8 bytes) 
Total (when):    ~304 bytes per cell (competitive with CUDA)

Thread Safety:
-------------
- Each instance is thread-safe and independent
- No shared state between instances
- Safe for concurrent terrain processing
"""

import taichi as ti
import numpy as np
import math

@ti.kernel
def setup_boundary_edges(boundary: ti.template(), res: int, N: int):
    """Set standard edge boundary conditions for rectangular grids."""
    for i in range(ti.i32(N)):
        x = i % res
        y = i // res
        if x == 0 or x == res-1 or y == 0 or y == res-1:
            boundary[i] = ti.u8(1)
        else:
            boundary[i] = ti.u8(0)

@ti.kernel
def reset_accumulation_kernel(p: ti.template()):
    """Reset drainage accumulation to unit values."""
    for i in p:
        p[i] = 1.0

@ti.kernel
def extract_local_minima_filter(rcv: ti.template(), bound: ti.template(), 
                                mask: ti.template(), N: int):
    """Filter operation: mark all local minima that are not boundaries."""
    for i in range(ti.i32(N)):
        if rcv[i] == i and bound[i] == 0:
            mask[i] = 1
        else:
            mask[i] = 0

def compact_indices_cpu(mask_field, p_lm_field, N):
    """
    CPU-based compact operation equivalent to torch.where().
    
    Fixed to match CUDA torch.where() behavior by only filling valid positions
    without -1 padding that could cause out-of-bounds access in indexed_set_id.
    
    CUDA Correspondence:
    - Equivalent to torch.where((~bound) & (rcv==idt))[0] 
    - Only fills positions 0 to len(indices)-1 with valid grid indices
    - No sentinel value padding that could cause basin[-1] access
    - Ensures indexed_set_id sees only valid indices in first S positions
    """
    mask_cpu = mask_field.to_numpy()
    indices = np.where(mask_cpu == 1)[0].astype(np.int64)
    
    # Get existing p_lm data (preserve unmodified areas)
    p_lm_cpu = p_lm_field.to_numpy()
    
    # Only fill valid positions (no -1 padding like CUDA torch.where)
    if len(indices) > 0:
        copy_len = min(len(indices), p_lm_field.shape[0])
        p_lm_cpu[:copy_len] = indices[:copy_len]
        # Leave remaining positions unchanged (don't overwrite with -1)
    
    p_lm_field.from_numpy(p_lm_cpu)
    return len(indices)

@ti.kernel  
def create_basin2_from_basin_edgez(basin_edgez: ti.template(), basin: ti.template(), 
                                  bignum: float, basin2: ti.template(), N: int):
    """Create basin2 equivalent to torch.where(basin_edgez == bignum, 0, basin)."""
    for i in range(ti.i32(N)):
        if basin_edgez[i] == bignum:
            basin2[i] = 0
        else:
            basin2[i] = basin[i]

class UnifiedFlowFields:
    """
    Unified field container for both flow routing and depression routing.
    
    This class provides a single container that manages all fields needed for
    both standard flow routing and depression routing (lakeflow) operations.
    Depression-specific fields are allocated on-demand to optimize memory usage
    for cases where only flow accumulation is needed.
    
    Design matches CUDA implementation where the same workspace is shared between
    flow computation and depression routing phases.
    """
    
    def __init__(self, N: int):
        """
        Initialize unified flow fields with core flow routing fields.
        Depression routing fields are allocated on-demand.
        
        Parameters:
        -----------
        N : int
            Total number of grid cells (must be perfect square)
        """
        if N <= 0:
            raise ValueError(f"N must be positive, got {N}")
        
        self.N = N
        self.res = int(math.sqrt(N))
        
        if self.res * self.res != N:
            raise ValueError(f"N={N} is not a perfect square (res={self.res})")
        
        # === CORE FIELDS (always allocated) ===
        # These correspond to the primary CUDA tensors used in both flow and lakeflow
        
        # Primary terrain and flow data
        self.z = ti.field(ti.f32, shape=N)              # elevation data (z parameter)
        self.boundary = ti.field(ti.u8, shape=N)        # boundary conditions
        self.rcv = ti.field(ti.i64, shape=N)            # receivers (rcv parameter)
        self.W = ti.field(ti.f32, shape=N)              # flow weights (W parameter)
        self.p = ti.field(ti.f32, shape=N)              # drainage accumulation
        
        # Working arrays for double-buffering (prevent race conditions)
        self.rcv_ = ti.field(ti.i64, shape=N)           # working receivers (rcv_ parameter)
        self.W_ = ti.field(ti.f32, shape=N)             # working weights (W_ parameter)
        self.p_ = ti.field(ti.f32, shape=N)             # working accumulation
        
        # Tree accumulation fields (for standard flow routing)
        self.dnr = ti.field(ti.i32, shape=N*4)          # donor arrays
        self.dnr_ = ti.field(ti.i32, shape=N*4)         # working donor arrays
        self.ndnr = ti.field(ti.i32, shape=N)           # number of donors
        self.ndnr_ = ti.field(ti.i32, shape=N)          # working number of donors
        self.src = ti.field(ti.i32, shape=N)            # source tracking
        
        # Utility arrays
        self.rand_array = ti.field(ti.f32, shape=N)     # random values for stochastic routing
        
        # Parallel scan working arrays
        scan_size = 1
        while scan_size < N:
            scan_size *= 2
        self.scan_work = ti.field(ti.i32, shape=scan_size)
        self.scan_temp = ti.field(ti.i32, shape=scan_size)
        
        # === DEPRESSION ROUTING FIELDS (allocated on-demand) ===
        self._depression_allocated = False
        
        # These will be created when enable_depression_routing() is called
        # Corresponds to CUDA lakeflow parameters:
        self.basin = None                               # basin parameter
        self.basin_route = None                         # basin_route parameter
        self.basin_edgez = None                         # basin_edgez parameter
        self.basin2 = None                              # derived from basin_edgez
        self.bound_ind = None                           # bound_ind parameter
        self.p_lm = None                                # p_lm parameter
        self.lm_mask = None                             # for local minima filtering
        self.lm_counter = None                          # atomic counter
        
        # Scatter-min workspace (corresponds to *_space parameters)
        self.argminh = None                             # argminh_space parameter
        self.minh = None                                # minh_space parameter
        self.p_rcv = None                               # p_rcv_space parameter
        self.b_rcv = None                               # b_rcv_space parameter
        self.b = None                                   # b_space parameter
        
        # Keep arrays (part of keep_space parameter - 3*(N+1) total)
        self.keep_ptr = None                            # keep array portion
        self.keep_offset_ptr = None                     # offset array portion
        self.keep_b_ptr = None                          # keep_b array portion
        
        # Path tracking
        self.reverse_path = None                        # reverse_path parameter
        
        # Additional working arrays for path accumulation
        self.rcv_0 = None                               # temporary receivers
        self.W_0 = None                                 # temporary weights
        self.p_0 = None                                 # temporary paths
        self.W2_carve = None                            # carving workspace
        self.rcv2_carve = None                          # carving workspace
        
        # Scatter-min internal arrays
        self.argbasin = None                            # scatter-min results
        self.nbasin = None                              # basin counts
        
        # Counters and metadata
        self.S_actual = None                            # actual local minima count
        self.final_count = None                         # final keep count
    
    def enable_depression_routing(self, max_S: int = None):
        """
        Allocate depression routing fields on-demand.
        This matches the CUDA approach where workspace is allocated for lakeflow.
        
        Parameters:
        -----------
        max_S : int, optional
            Maximum expected number of local minima (defaults to N//10)
        """
        if self._depression_allocated:
            return  # Already allocated
        
        if max_S is None:
            max_S = max(self.N // 10, 100)  # Reasonable default
        
        # === BASIN MANAGEMENT ===
        # Direct correspondence to CUDA parameters
        self.basin = ti.field(ti.i64, shape=self.N)                    # basin parameter
        self.basin_route = ti.field(ti.i64, shape=self.N)              # basin_route parameter
        self.basin_edgez = ti.field(ti.f32, shape=self.N)              # basin_edgez parameter
        self.basin2 = ti.field(ti.i64, shape=self.N)                   # derived tensor equivalent
        self.bound_ind = ti.field(ti.i64, shape=self.N)                # bound_ind parameter
        
        # === LOCAL MINIMA MANAGEMENT ===
        self.p_lm = ti.field(ti.i64, shape=self.N)                     # p_lm parameter
        self.lm_mask = ti.field(ti.i64, shape=self.N)                  # for filtering
        self.lm_counter = ti.field(ti.i64, shape=1)                    # atomic counter
        
        # === SCATTER-MIN WORKSPACE ===
        # These correspond to the *_space parameters in CUDA call
        self.argminh = ti.field(ti.i64, shape=self.N+1)                # argminh_space parameter
        self.minh = ti.field(ti.f32, shape=self.N+1)                   # minh_space parameter
        self.p_rcv = ti.field(ti.i64, shape=self.N+1)                  # p_rcv_space parameter
        self.b_rcv = ti.field(ti.i64, shape=self.N+1)                  # b_rcv_space parameter
        self.b = ti.field(ti.i64, shape=self.N+1)                      # b_space parameter
        
        # === KEEP ARRAYS ===
        # These represent the 3*(N+1) keep_space parameter split into components
        self.keep_ptr = ti.field(ti.i64, shape=self.N+1)               # keep portion
        self.keep_offset_ptr = ti.field(ti.i64, shape=self.N+1)        # offset portion  
        self.keep_b_ptr = ti.field(ti.i64, shape=self.N+1)             # keep_b portion
        
        # === PATH TRACKING ===
        self.reverse_path = ti.field(ti.f32, shape=self.N)             # reverse_path parameter
        
        # === ADDITIONAL WORKING ARRAYS ===
        # These are created inside the CUDA kernel as temporary arrays
        self.rcv_0 = ti.field(ti.i64, shape=self.N)                    # temp rcv
        self.W_0 = ti.field(ti.f32, shape=self.N)                      # temp W
        self.p_0 = ti.field(ti.i64, shape=self.N)                      # temp p
        self.W2_carve = ti.field(ti.f32, shape=self.N)                 # carve workspace
        self.rcv2_carve = ti.field(ti.i64, shape=self.N)               # carve workspace
        
        # === SCATTER-MIN INTERNAL ARRAYS ===
        self.argbasin = ti.field(ti.i64, shape=self.N+1)               # scatter-min results
        self.nbasin = ti.field(ti.i64, shape=self.N)                   # basin counts
        
        # === COUNTERS AND METADATA ===
        self.S_actual = ti.field(ti.i64, shape=1)                      # S parameter equivalent
        self.final_count = ti.field(ti.i64, shape=1)                   # final count from keep
        
        self._depression_allocated = True
    
    def is_depression_enabled(self):
        """Check if depression routing fields are allocated."""
        return self._depression_allocated
    
    # === TERRAIN AND BOUNDARY MANAGEMENT ===
    
    def get_terrain_shape(self):
        """Get 2D terrain shape."""
        return (self.res, self.res)
    
    def load_terrain(self, z_2d):
        """Load 2D elevation data into z field."""
        if z_2d.shape != (self.res, self.res):
            raise ValueError(f"Terrain must be {self.res}x{self.res}, got {z_2d.shape}")
        z_flat = z_2d.flatten().astype(np.float32)
        self.z.from_numpy(z_flat)
    
    def set_boundary_edges(self):
        """Set standard edge boundary conditions."""
        setup_boundary_edges(self.boundary, self.res, self.N)
    
    def set_custom_boundaries(self, boundary_2d):
        """Set custom boundary conditions from 2D array."""
        if boundary_2d.shape != (self.res, self.res):
            raise ValueError(f"Boundaries must be {self.res}x{self.res}, got {boundary_2d.shape}")
        boundary_flat = boundary_2d.flatten().astype(np.uint8)
        self.boundary.from_numpy(boundary_flat)
    
    # === RESULT EXTRACTION ===
    
    def get_receivers_2d(self):
        """Extract receiver indices as 2D array."""
        return self.rcv.to_numpy().reshape(self.res, self.res)

    def get_basins_2d(self):
        """Extract receiver indices as 2D array."""
        return self.basin.to_numpy().reshape(self.res, self.res)
    
    def get_weights_2d(self):
        """Extract flow weights as 2D array."""
        return self.W.to_numpy().reshape(self.res, self.res)
    
    def get_drainage_2d(self):
        """Extract drainage accumulation as 2D array."""
        return self.p.to_numpy().reshape(self.res, self.res)
    
    def reset_accumulation(self):
        """Reset drainage accumulation to unit values."""
        reset_accumulation_kernel(self.p)
    
    # === DEPRESSION ROUTING UTILITIES ===
    
    def reset_for_iteration(self, bignum: float):
        """Reset depression fields for new lakeflow iteration."""
        if not self._depression_allocated:
            raise RuntimeError("Depression routing not enabled. Call enable_depression_routing() first.")
        
        # Clear basin arrays
        self.basin.fill(0)
        self.basin_edgez.fill(bignum) 
        self.reverse_path.fill(0.0)
        self.minh.fill(1e10)
        
        # Clear working arrays
        self.argbasin.fill(1000000)
        self.nbasin.fill(0)
        
        # Clear keep arrays
        self.keep_ptr.fill(0)
        self.keep_offset_ptr.fill(0)
        self.keep_b_ptr.fill(0)
        
        # Clear local minima tracking
        self.lm_mask.fill(0)
        self.lm_counter.fill(0)
    
    def extract_local_minima_count(self, rcv_field, bound_field):
        """Extract local minima and return count (torch.where equivalent)."""
        if not self._depression_allocated:
            raise RuntimeError("Depression routing not enabled. Call enable_depression_routing() first.")
        
        # Filter: mark local minima
        extract_local_minima_filter(rcv_field, bound_field, self.lm_mask, self.N)
        
        # Compact: gather indices (CPU-based for correctness)
        S = compact_indices_cpu(self.lm_mask, self.p_lm, self.N)
        self.S_actual[0] = S
        return S
    
    def create_basin2_field(self, bignum: float):
        """Create basin2 field from basin_edgez and basin."""
        if not self._depression_allocated:
            raise RuntimeError("Depression routing not enabled. Call enable_depression_routing() first.")
        
        create_basin2_from_basin_edgez(self.basin_edgez, self.basin, bignum, self.basin2, self.N)
    
    def load_boundary_indices(self, boundary_indices):
        """Load boundary indices (equivalent to bound.nonzero() in CUDA)."""
        if not self._depression_allocated:
            raise RuntimeError("Depression routing not enabled. Call enable_depression_routing() first.")
        
        if len(boundary_indices) > self.bound_ind.shape[0]:
            raise ValueError(f"Too many boundary indices: {len(boundary_indices)} > {self.bound_ind.shape[0]}")
        
        boundary_flat = boundary_indices.astype(np.int64)
        full_array = np.full(self.bound_ind.shape[0], -1, dtype=np.int64)
        full_array[:len(boundary_flat)] = boundary_flat
        self.bound_ind.from_numpy(full_array)
    
    def get_keep_arrays(self):
        """Get the three keep arrays."""
        if not self._depression_allocated:
            raise RuntimeError("Depression routing not enabled. Call enable_depression_routing() first.")
        
        return (self.keep_ptr, self.keep_offset_ptr, self.keep_b_ptr)



def create_unified_flow_fields(N: int) -> UnifiedFlowFields:
    """Factory function to create unified flow fields."""
    return UnifiedFlowFields(N)