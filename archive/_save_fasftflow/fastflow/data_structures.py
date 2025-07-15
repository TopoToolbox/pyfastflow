"""
Data bag classes for Taichi fields - replaces all declare_XXX functions
Follows 1:1 CUDA structure but with proper Taichi memory management
"""
import taichi as ti
import math


class BasicFlowFields:
    """Data bag for basic flow receiver calculation and tree accumulation"""
    
    def __init__(self, N: int):
        self.N = N
        self.res = int(math.sqrt(N))
        
        # Core flow fields (1:1 CUDA equivalent)
        self.z = ti.field(ti.f32, shape=N)
        self.bound = ti.field(ti.i64, shape=N)
        self.rcv = ti.field(ti.i64, shape=N)  # CUDA: int64_t*
        self.W = ti.field(ti.f32, shape=N)
        
        # Tree accumulation working fields
        self.rcv_ = ti.field(ti.i64, shape=N)  # CUDA: int64_t*
        self.W_ = ti.field(ti.f32, shape=N)
        self.p = ti.field(ti.f32, shape=N)  # Drainage accumulation values
        
        # Tree traversal temporary fields (for upward/downward kernels)
        # EXACT CUDA: path accumulation uses int32 arrays (converted from int64)
        self.rcv_temp = ti.field(ti.i64, shape=N)  # Fix: i64 for consistency
        self.W_temp = ti.field(ti.f32, shape=N)
        self.p_temp = ti.field(ti.i64, shape=N)  # Fix: i64 for consistency
        
        # CUDA clones for path accumulation: auto rcv2 = rcv.to(torch::kInt32);
        self.rcv_i32 = ti.field(ti.i64, shape=N)  # Fix: i64 for consistency
        self.W_i32 = ti.field(ti.f32, shape=N)    # Working copy for path accumulation
        
        # Random flow fields (for randomized receiver selection)
        self.rand_array = ti.field(ti.f32, shape=N)
        
        # Drainage accumulation
        self.drain = ti.field(ti.f32, shape=N)
        
        # Tree max downward fields
        self.tree_max_temp = ti.field(ti.f32, shape=N)
        
        # Tree accumulation rake-compress fields (pre-allocated to avoid recompilation)
        self.dnr0 = ti.field(ti.i64, shape=N*4)
        self.dnr_0 = ti.field(ti.i64, shape=N*4)  
        self.ndnr0 = ti.field(ti.i64, shape=N)
        self.ndnr_0 = ti.field(ti.i64, shape=N)
        self.p_0_tree = ti.field(ti.f32, shape=N)
        self.src_tree = ti.field(ti.i64, shape=N)
        
        # Local minima extraction working fields
        self.is_minima = ti.field(ti.i64, shape=N)
        self.minima_offsets = ti.field(ti.i64, shape=N+1)


class LakeflowFields:
    """Data bag for depression routing (lakeflow) algorithm"""
    
    def __init__(self, N: int, S: int):
        self.N = N
        self.S = S
        self.res = int(math.sqrt(N))
        
        # Local minima management
        self.p_lm = ti.field(ti.i64, shape=S)  # CUDA: int64_t* p_lm
        self.nlm = ti.field(ti.i64, shape=1)   # Fix: i64 for consistency
        self.keep_p_lm = ti.field(ti.i64, shape=S)  # Temporary storage for reordering
        
        # Basin management (1:1 CUDA equivalent)
        self.basin = ti.field(ti.i64, shape=N)      # CUDA: int64_t* basin
        self.basin_route = ti.field(ti.i64, shape=N)  # CUDA: int64_t* basin_route
        self.basin_edgez = ti.field(ti.f32, shape=N)
        self.basin2 = ti.field(ti.i64, shape=N)  # CRITICAL FIX: must match basin field type
        
        # Scatter-min atomic operation fields
        self.argminh = ti.field(ti.i64, shape=S+10)  # CUDA: int64_t* argminh
        self.minh = ti.field(ti.f32, shape=S+10)
        self.argbasin_ = ti.field(ti.i64, shape=N)   # CUDA: int64_t* argbasin
        self.nbasin_ = ti.field(ti.i64, shape=N)  # CUDA: int64_t* nbasin (sized for all cells)
        
        # Path calculation fields (exact CUDA structure)
        self.p_rcv = ti.field(ti.i64, shape=S+10)    # CUDA: int64_t* p_rcv
        self.b_rcv = ti.field(ti.i64, shape=S+10)    # CUDA: int64_t* b_rcv
        self.b = ti.field(ti.i64, shape=S+10)        # CUDA: int64_t* b
        
        # Keep array generation (for carving path selection)
        self.keep_b = ti.field(ti.i64, shape=S+10)      # Fix: i64 for consistency
        self.keep_offset = ti.field(ti.i64, shape=N+1)  # Fix: i64 for indexing
        self.keep = ti.field(ti.i64, shape=N+1)         # Fix: i64 for basin values
        self.final_count = ti.field(ti.i64, shape=1)
        
        # Carving path accumulation (working fields for upward kernels)
        self.rcv_work = ti.field(ti.i64, shape=N)  # Fix: i64 for consistency
        self.W_work = ti.field(ti.f32, shape=N)
        self.p_work = ti.field(ti.i64, shape=N)  # Fix: i64 for indexing
        
        # Reverse path initialization
        self.reverse_path = ti.field(ti.f32, shape=N)
        
        # Boundary indicators
        self.bound_ind = ti.field(ti.i64, shape=N)  # Fix: i64 for consistency


class ErodeDepositFields:
    """Data bag for erosion/deposition calculations"""
    
    def __init__(self, N: int):
        self.N = N
        
        # Sediment flux fields
        self.Qs = ti.field(ti.f32, shape=N)
        self.Qs_temp = ti.field(ti.f32, shape=N)
        
        # Erosion calculation working fields
        self.erosion_rate = ti.field(ti.f32, shape=N)
        self.deposition_rate = ti.field(ti.f32, shape=N)
        
        # Tree accumulation for erosion (working fields)
        self.tree_down_temp = ti.field(ti.f32, shape=N)
        self.tree_down_result = ti.field(ti.f32, shape=N)


# Factory functions to create data bags with proper initialization
def create_basic_flow_fields(N: int) -> BasicFlowFields:
    """Create and initialize basic flow computation fields"""
    return BasicFlowFields(N)


def create_lakeflow_fields(N: int, S: int) -> LakeflowFields:
    """Create and initialize lakeflow (depression routing) fields"""
    return LakeflowFields(N, S)


def create_erode_deposit_fields(N: int) -> ErodeDepositFields:
    """Create and initialize erosion/deposition fields"""
    return ErodeDepositFields(N)