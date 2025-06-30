"""
Tree accumulation kernels - 1:1 CUDA rake-compress port with proper Taichi structure
All kernels declared globally with template() arguments
"""
import taichi as ti
import math
from .data_structures import BasicFlowFields


# ============================================================================
# RAKE-COMPRESS UTILITY FUNCTIONS (1:1 CUDA PORT)
# ============================================================================

@ti.func
def get_src_flag(src: ti.template(), id: int, iter: int):
    """1:1 port of CUDA getSrc function"""
    entry = src[id]
    flip = entry < 0
    flip = (not flip) if abs(entry) == (iter + 1) else flip
    return flip


@ti.func
def update_src_flag(src: ti.template(), tid: int, iter: int, flip: int):
    """1:1 port of CUDA updateSrc function"""
    src[tid] = (1 if flip else -1) * (iter + 1)


# ============================================================================
# RAKE-COMPRESS KERNELS (1:1 CUDA PORT)
# ============================================================================

@ti.kernel
def rcv_to_donor_mapping(rcv: ti.template(), dnr: ti.template(), ndnr: ti.template(), n: int, res: int):
    """1:1 port of CUDA rcv2donor kernel"""
    for tid in range(n):
        if tid < n and rcv[tid] != tid:
            rcv_idx = rcv[tid]
            old_val = ti.atomic_add(ndnr[rcv_idx], 1)
            dnr[rcv_idx * 4 + old_val] = tid


@ti.kernel
def rake_compress_accumulation(dnr: ti.template(), ndnr: ti.template(), p: ti.template(), src: ti.template(), 
                              dnr_: ti.template(), ndnr_: ti.template(), p_: ti.template(), n: int, iter: int):
    """1:1 port of CUDA rake_compress_accum kernel"""
    for tid in range(n):
        if tid >= n:
            continue

        flip = get_src_flag(src, tid, iter)
        
        worked = 0
        donors = ti.Vector([-1, -1, -1, -1], dt=ti.i32)
        base = tid * 4
        p_added = 0.0
        
        # Get todo from A array (considering flip)
        todo = 0
        if flip == 0:
            todo = ndnr[tid]
        else:
            todo = ndnr_[tid]
        
        
        # EXACT CUDA loop: for (int i=0; i < todo; i++) but with i-- when removing
        i = 0
        while i < todo:
            if donors[i] == -1:
                if flip == 0:
                    donors[i] = dnr[base + i]
                else:
                    donors[i] = dnr_[base + i]
            
            did = donors[i]
            
            flip_did = get_src_flag(src, did, iter)
            
            # Get values from C array (original X/Y with flip_did)
            C_ndnr_val = 0
            C_p_val = 0.0
            
            if flip_did == 0:
                C_ndnr_val = ndnr[did]
                C_p_val = p[did]
            else:
                C_ndnr_val = ndnr_[did]
                C_p_val = p_[did]
            
            if C_ndnr_val <= 1:
                if worked == 0:
                    if flip == 0:
                        p_added = p[tid]
                    else:
                        p_added = p_[tid]
                worked = 1
                
                p_added += C_p_val
                
                if C_ndnr_val == 0:
                    todo -= 1
                    todo_offset = base + todo
                    if flip == 0:
                        donors[i] = dnr[todo_offset]
                    else:
                        donors[i] = dnr_[todo_offset]
                    # Don't increment i - CUDA does i-- which keeps i same
                else:
                    did_offset = did * 4
                    if flip_did == 0:
                        donors[i] = dnr[did_offset]
                    else:
                        donors[i] = dnr_[did_offset]
                    i += 1
            else:
                i += 1
        
        if worked == 1:
            # Write to B arrays
            if flip == 0:
                ndnr_[tid] = todo
                p_[tid] = p_added
                for j in range(todo):
                    dnr_[base + j] = donors[j]
            else:
                ndnr[tid] = todo
                p[tid] = p_added
                for j in range(todo):
                    dnr[base + j] = donors[j]
            
            update_src_flag(src, tid, iter, flip)


@ti.kernel
def fuse_final_result(A: ti.template(), src: ti.template(), B: ti.template(), n: int, iter: int):
    """1:1 port of CUDA fuse kernel"""
    for tid in range(n):
        if tid >= n:
            continue
        
        if get_src_flag(src, tid, iter):
            A[tid] = B[tid]


# ============================================================================
# TREE ALGORITHM INTERFACE FUNCTIONS
# ============================================================================

def run_tree_accum_upward_rake_compress(flow_fields: BasicFlowFields):
    """
    Run upward tree accumulation using rake-compress algorithm
    1:1 port of CUDA tree accumulation with proper Taichi memory management
    """
    N = flow_fields.N
    logN = int(math.ceil(math.log2(float(N))))
    
    # Initialize arrays - EXACT CUDA initialization
    @ti.kernel
    def init_arrays():
        for i in range(N):
            flow_fields.ndnr0[i] = 0
            flow_fields.src_tree[i] = 0
    
    init_arrays()
    
    # Build donor arrays
    rcv_to_donor_mapping(flow_fields.rcv, flow_fields.dnr0, flow_fields.ndnr0, N, flow_fields.res)
    
    # EXACT CUDA algorithm: for (int i=0; i < logN; i++)
    for i in range(logN):
        rake_compress_accumulation(flow_fields.dnr0, flow_fields.ndnr0, flow_fields.p, flow_fields.src_tree,
                                  flow_fields.dnr_0, flow_fields.ndnr_0, flow_fields.p_0_tree, N, i)
    
    # EXACT CUDA fuse call
    fuse_final_result(flow_fields.p, flow_fields.src_tree, flow_fields.p_0_tree, N, logN + 1)