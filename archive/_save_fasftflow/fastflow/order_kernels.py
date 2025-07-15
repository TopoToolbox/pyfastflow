"""
Order (Local Minima) extraction kernels - 1:1 CUDA port with proper Taichi structure
All kernels declared globally with template() arguments
"""
import taichi as ti
from .data_structures import BasicFlowFields, LakeflowFields
from .parallel_scan import inclusive_scan


# ============================================================================
# LOCAL MINIMA EXTRACTION KERNELS (1:1 CUDA PORT)
# ============================================================================

@ti.kernel
def mark_local_minima(rcv: ti.template(), bound: ti.template(), is_minima: ti.template(), N: int):
    """1:1 port of CUDA local minima marking kernel"""
    for id in range(N):
        if id >= N:
            continue
        # A cell is a local minimum if it flows to itself and is not a boundary
        is_minima[id] = 1 if (rcv[id] == id and bound[id] == 0) else 0


@ti.kernel
def extract_minima_indices(is_minima: ti.template(), minima_offsets: ti.template(), 
                          p_lm: ti.template(), N: int):
    """1:1 port of CUDA local minima extraction - sequential order for now"""
    for id in range(N):
        if id >= N:
            continue
        if is_minima[id] == 1:
            # Get the index from the prefix sum (offset by 1)
            minima_idx = minima_offsets[id] - 1
            if minima_idx >= 0 and minima_idx < p_lm.shape[0]:
                p_lm[ti.cast(minima_idx, ti.i64)] = id


@ti.kernel
def get_total_minima_count(minima_offsets: ti.template(), N: int, total_count: ti.template()):
    """1:1 port of CUDA total count extraction kernel"""
    total_count[0] = minima_offsets[N-1]


# ============================================================================
# CUDA-COMPATIBLE ORDERING FUNCTIONS
# ============================================================================

@ti.kernel
def reorder_p_lm_cuda_style(p_lm: ti.template(), temp_p_lm: ti.template(), N: int, S: int):
    """Reorder p_lm array to match CUDA's row-wise reverse ordering"""
    res = int(ti.sqrt(float(N)))
    
    # Group local minima by row and reorder
    write_idx = 0
    for y in range(res):
        # Collect minima in this row
        row_minima = ti.Vector([0] * 32, dt=ti.i64)  # Assuming max 32 minima per row
        row_count = 0
        
        for i in range(S):
            pos = p_lm[i]
            pos_y = pos // res
            if pos_y == y and row_count < 32:
                row_minima[row_count] = pos
                row_count += 1
        
        # Sort within row by x-coordinate (left to right)
        for i in range(row_count):
            for j in range(i + 1, row_count):
                pos_i = row_minima[i]
                pos_j = row_minima[j]
                x_i = pos_i % res
                x_j = pos_j % res
                if x_i > x_j:
                    # Swap
                    temp_val = row_minima[i]
                    row_minima[i] = row_minima[j]
                    row_minima[j] = temp_val
        
        # Write in reverse order (right to left)
        for i in range(row_count):
            rev_idx = row_count - 1 - i
            if write_idx < S:
                temp_p_lm[write_idx] = row_minima[rev_idx]
                write_idx += 1

@ti.kernel  
def copy_back_p_lm(p_lm: ti.template(), temp_p_lm: ti.template(), S: int):
    """Copy reordered p_lm back"""
    for i in range(S):
        p_lm[i] = temp_p_lm[i]


# ============================================================================
# LOCAL MINIMA INTERFACE FUNCTION
# ============================================================================

@ti.kernel
def clear_array(array: ti.template(), n: int):
    """Clear array to zero"""
    for i in range(n):
        array[i] = 0


def extract_local_minima(flow_fields: BasicFlowFields, lakeflow_fields: LakeflowFields):
    """
    Extract local minima from RCV field with CUDA-compatible ordering
    1:1 port of CUDA local minima extraction with proper Taichi memory management
    """
    N = flow_fields.N
    
    # Use properly sized fields from flow_fields for local minima extraction
    is_minima = flow_fields.is_minima  # Correct size N
    minima_offsets = flow_fields.minima_offsets  # Correct size N+1
    
    # CRITICAL: Clear arrays to ensure clean state
    clear_array(is_minima, N)
    clear_array(minima_offsets, N+1)
    clear_array(lakeflow_fields.nlm, 1)
    
    # Mark local minima
    mark_local_minima(flow_fields.rcv, flow_fields.bound, is_minima, N)
    
    # Compute prefix sum to get indices
    inclusive_scan(is_minima, minima_offsets, N)
    
    # Extract actual minima indices
    extract_minima_indices(is_minima, minima_offsets, lakeflow_fields.p_lm, N)
    
    # Get total count
    get_total_minima_count(minima_offsets, N, lakeflow_fields.nlm)
    
    S = lakeflow_fields.nlm[0]
    
    # SIMPLE TEST: Just reverse the entire p_lm array to see if this affects basin assignment
    # reorder_p_lm_cuda_style(lakeflow_fields.p_lm, lakeflow_fields.keep_p_lm, N, S)
    # copy_back_p_lm(lakeflow_fields.p_lm, lakeflow_fields.keep_p_lm, S)
    
    return S