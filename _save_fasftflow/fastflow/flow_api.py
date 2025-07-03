"""
Main Flow API - Clean interface using data bag classes
1:1 CUDA port philosophy with proper Taichi memory management
"""
import taichi as ti
import numpy as np
from .data_structures import BasicFlowFields, LakeflowFields, create_basic_flow_fields, create_lakeflow_fields
from .rcv_kernels import run_rcv_deterministic, run_rcv_randomized, setup_default_bounds
from .lakeflow_kernels import run_lakeflow_algorithm
from .order_kernels import extract_local_minima
from .tree_kernels import run_tree_accum_upward_rake_compress

@ti.kernel
def swap_rcv_arrays_2(rcv:ti.template(), rcv_:ti.template()):
    for i in rcv:
        temp = rcv[i]
        rcv[i] = rcv_[i]
        rcv_[i] = temp


class FastFlowProcessor:
    """
    Main FastFlow processor using proper data bag architecture
    Eliminates all field declarations in loops and provides clean 1:1 CUDA interface
    """
    
    def __init__(self, res: int):
        """Initialize processor with grid resolution"""
        self.res = res
        self.N = res * res
        
        # Create data bags (allocated once, reused throughout)
        self.flow_fields = create_basic_flow_fields(self.N)
        self.lakeflow_fields = None  # Created when needed
        
        # Configuration
        self.bignum = 1e10
    
    def setup_terrain(self, z_2d: np.ndarray, custom_bounds: np.ndarray = None):
        """
        Set up terrain and boundary conditions
        z_2d: 2D elevation array (res x res)
        custom_bounds: Optional custom boundary array (res x res), None for default edge boundaries
        """
        if z_2d.shape != (self.res, self.res):
            raise ValueError(f"Terrain must be {self.res}x{self.res}, got {z_2d.shape}")
        
        # Flatten and copy terrain
        z_flat = z_2d.flatten().astype(np.float32)
        self.flow_fields.z.from_numpy(z_flat)
        
        # Set up boundaries
        if custom_bounds is not None:
            if custom_bounds.shape != (self.res, self.res):
                raise ValueError(f"Boundaries must be {self.res}x{self.res}, got {custom_bounds.shape}")
            bound_flat = custom_bounds.flatten().astype(np.int32)
            self.flow_fields.bound.from_numpy(bound_flat)
        else:
            # Use default edge boundaries
            setup_default_bounds(self.flow_fields)
    
    def compute_receivers_deterministic(self):
        """
        Compute receivers using deterministic steepest descent
        1:1 port of CUDA rcv_matrix_cuda
        """
        run_rcv_deterministic(self.flow_fields)
        return self.flow_fields.rcv.to_numpy().reshape(self.res, self.res)
    
    def compute_receivers_randomized(self, seed: int = 42):
        """
        Compute receivers using randomized proportional flow
        1:1 port of CUDA rcv_matrix_rand_cuda
        """
        # Generate random array
        np.random.seed(seed)
        rand_vals = np.random.random(self.N).astype(np.float32)
        self.flow_fields.rand_array.from_numpy(rand_vals)
        
        run_rcv_randomized(self.flow_fields)
        return self.flow_fields.rcv.to_numpy().reshape(self.res, self.res)
    
    def route_depressions(self, method: str = 'carve', num_iter: int = 0):
        """
        Route depressions using lakeflow algorithm
        1:1 port of CUDA lakeflow with carve or jump method
        
        method: 'carve' or 'jump'
        num_iter: 0 for automatic, >0 for fixed iterations
        """
        if method not in ['carve', 'jump']:
            raise ValueError("Method must be 'carve' or 'jump'")
        
        # Extract local minima first
        if self.lakeflow_fields is None:
            # Estimate S (number of local minima) - use conservative upper bound
            S = max(self.N // 10, 1000000)  # At least 1000000000 for safety
            self.lakeflow_fields = create_lakeflow_fields(self.N, S)
        
        # Extract actual local minima
        S_actual = extract_local_minima(self.flow_fields, self.lakeflow_fields)
        if(S_actual > S):
            raise ValueError(f'S_actual {S_actual} << S {S} ')
        # print(f"  Found {S_actual} local minima")
        
        
        if S_actual == 0:
            # No depressions to route
            return self.flow_fields.rcv.to_numpy().reshape(self.res, self.res)
        
        # EXACT CUDA: Process ALL local minima (no artificial limits)
        if S_actual > self.lakeflow_fields.S:
            # Recreate lakeflow fields with exact size needed
            S_usable = S_actual + 10  # Small buffer for safety
            print(f"  Recreating lakeflow fields for {S_usable} local minima")
            self.lakeflow_fields = create_lakeflow_fields(self.N, S_usable)
            # Re-extract with full size 
            S_actual = extract_local_minima(self.flow_fields, self.lakeflow_fields)
        
        # Update S in lakeflow fields
        self.lakeflow_fields.S = S_actual
        
        # CONSERVATIVE: Try 2 iterations max to test multi-level without breaking single-level
        import math
        logN = int(math.ceil(math.log2(self.N)))
        carve = (method == 'carve')
        
        # try:
        # First iteration - this was working
        # print(f"  === ITERATION 0 ===")
        for it in range(120):
            run_lakeflow_algorithm(self.flow_fields, self.lakeflow_fields, 
                                 self.bignum, carve, it)  # num_iter=0 for first iteration
            
            # # EXACT CUDA: rcv_, rcv = rcv, rcv_ after each iteration
            # if carve:
            #     print(f"  DEBUG: Performing swap after iteration 0")
            #     cop = self.flow_fields.rcv.to_numpy()
            #     swap_rcv_arrays_2(self.flow_fields.rcv, self.flow_fields.rcv_)
            #     cop2 = self.flow_fields.rcv.to_numpy()
            #     print(f"  DEBUG: Swap completed for iteration 0")
            
            # Check if there are still local minima for iteration 1
            # S_iter1 = extract_local_minima(self.flow_fields, self.lakeflow_fields)
            # print(f"  After iteration {it}: Found {S_iter1} remaining local minima")
        
        # if S_iter1 > 0 and S_iter1 < S_actual:  # Only proceed if we made progress
        #     print(f"  === ITERATION 1 ===")
        #     # Recreate lakeflow fields if needed
        #     if S_iter1 > self.lakeflow_fields.S:
        #         print(f"  WARNING: More minima ({S_iter1}) than expected, limiting to {self.lakeflow_fields.S}")
        #         S_iter1 = self.lakeflow_fields.S
            
        #     self.lakeflow_fields.S = S_iter1
            
        #     # Second iteration
        #     run_lakeflow_algorithm(self.flow_fields, self.lakeflow_fields, 
        #                          self.bignum, carve, 1)  # num_iter=1 for second iteration
            
        #     # EXACT CUDA: Final swap
        #     if carve:
        #         print(f"  DEBUG: Performing swap after iteration 1")
        #         cop = self.flow_fields.rcv.to_numpy()
        #         swap_rcv_arrays_2(self.flow_fields.rcv, self.flow_fields.rcv_)
        #         cop2 = self.flow_fields.rcv.to_numpy()
        #         print(np.unique(cop-cop2))
        #         raise ValueError
        #         print(f"  DEBUG: Swap completed for iteration 1")
        
        return self.flow_fields.rcv.to_numpy().reshape(self.res, self.res)
        # except Exception as e:
        #     print(f"  GPU depression routing failed, falling back to no carving: {e}")
        #     raise ValueError('YOLO')
        #     # Return original receivers (no depression routing)
        #     return self.flow_fields.rcv.to_numpy().reshape(self.res, self.res)
    
    def accumulate_flow_upward(self):
        """
        Accumulate flow using upward tree traversal
        1:1 port of CUDA tree_accum_upward_rake_compress
        """
        # Initialize p with unit values for accumulation
        @ti.kernel
        def init_unit_flow():
            for i in range(self.N):
                self.flow_fields.p[i] = 1.0
        
        init_unit_flow()
        run_tree_accum_upward_rake_compress(self.flow_fields)
        return self.flow_fields.p.to_numpy().reshape(self.res, self.res)
    
    def accumulate_flow_downward(self):
        """
        Accumulate flow using downward tree traversal
        1:1 port of CUDA tree_accum_downward
        """
        run_tree_accum_downward(self.flow_fields)
        return self.flow_fields.p.to_numpy().reshape(self.res, self.res)
    
    def propagate_max_downward(self, values: np.ndarray = None):
        """
        Propagate maximum values downward through tree
        1:1 port of CUDA tree_max_downward
        
        values: Optional 2D array to propagate, uses current p field if None
        """
        if values is not None:
            if values.shape != (self.res, self.res):
                raise ValueError(f"Values must be {self.res}x{self.res}, got {values.shape}")
            vals_flat = values.flatten().astype(np.float32)
            self.flow_fields.p.from_numpy(vals_flat)
        
        run_tree_max_downward(self.flow_fields)
        return self.flow_fields.p.to_numpy().reshape(self.res, self.res)
    
    def get_drainage_area(self):
        """Get current drainage area (accumulated flow)"""
        return self.flow_fields.p.to_numpy().reshape(self.res, self.res)
    
    def get_weights(self):
        """Get current flow weights"""
        return self.flow_fields.W.to_numpy().reshape(self.res, self.res)
    
    def get_receivers_1d(self):
        """Get receivers as 1D array (for debugging)"""
        return self.flow_fields.rcv.to_numpy()
    
    def get_local_minima_count(self):
        """Get number of local minima"""
        if self.lakeflow_fields is None:
            return 0
        return self.lakeflow_fields.nlm[0]
    
    def get_local_minima_indices(self):
        """Get indices of local minima"""
        if self.lakeflow_fields is None:
            return np.array([])
        count = self.lakeflow_fields.nlm[0]
        if count == 0:
            return np.array([])
        return self.lakeflow_fields.p_lm.to_numpy()[:count]


# ============================================================================
# CONVENIENCE FUNCTIONS (COMPATIBLE WITH OLD API)
# ============================================================================

def compute_flow_routing_1to1(z_2d: np.ndarray, method: str = 'deterministic', 
                             depression_method: str = None) -> tuple:
    """
    Convenience function for complete flow routing
    
    z_2d: 2D elevation array
    method: 'deterministic' or 'randomized' for receiver calculation
    depression_method: None, 'carve', or 'jump' for depression routing
    
    Returns: (receivers_2d, drainage_area_2d, weights_2d)
    """
    res = z_2d.shape[0]
    if z_2d.shape[1] != res:
        raise ValueError("Elevation array must be square")
    
    # Create processor
    processor = FastFlowProcessor(res)
    processor.setup_terrain(z_2d)
    
    # Compute receivers
    if method == 'deterministic':
        rcv_2d = processor.compute_receivers_deterministic()
    elif method == 'randomized':
        rcv_2d = processor.compute_receivers_randomized()
    else:
        raise ValueError("Method must be 'deterministic' or 'randomized'")
    
    # Route depressions if requested
    if depression_method is not None:
        rcv_2d = processor.route_depressions(depression_method)
    
    # Compute drainage area
    drainage_2d = processor.accumulate_flow_upward()
    weights_2d = processor.get_weights()
    
    return rcv_2d, drainage_2d, weights_2d


# Initialize Taichi when module is imported
def initialize_taichi(arch=ti.gpu, debug=False):
    """Initialize Taichi with appropriate settings"""
    ti.init(arch=arch, debug=debug, advanced_optimization=True)


# Auto-initialize with CPU backend by default
try:
    initialize_taichi()
except Exception:
    # Taichi might already be initialized
    pass