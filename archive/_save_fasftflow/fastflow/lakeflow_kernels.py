"""
Lakeflow (Depression Routing) kernels - 1:1 CUDA port with proper Taichi structure
All kernels declared globally with template() arguments
"""
import taichi as ti
import math
from .data_structures import LakeflowFields, BasicFlowFields
from .scatter_min_kernels import run_scatter_min_atomic
from .parallel_scan import inclusive_scan
import numpy as np

@ti.kernel
def convert_to_int32(rcv_i32:ti.template(), rcv:ti.template(), W_i32:ti.template(), W:ti.template()):
    for i in rcv_i32:
        rcv_i32[i] = rcv[i]
        W_i32[i] = ti.i32(W[i])

@ti.kernel
def swaprcv(rcv:ti.template(),rcv_:ti.template()):
    for i in rcv:
        temp = rcv[i]
        rcv[i] = rcv_[i]
        rcv_[i] = temp


# ============================================================================
# PATH ACCUMULATION KERNELS (1:1 CUDA PORT)
# ============================================================================

@ti.kernel
def flow_cuda_path_accum_upward_kernel1(rcv: ti.template(), W: ti.template(), p: ti.template(), 
                                        rcv_: ti.template(), W_: ti.template(), p_: ti.template(), n: int):
    """1:1 port of CUDA upward path accumulation kernel 1 - EXACT CUDA types"""
    for tid in range(n):
        if tid < n and W[tid] > 0.001:
            if p[tid] > 0.001:
                p_[tid] = rcv[tid]  # CUDA: p_[tid] = rcv[tid] (int)
            else:
                p_[tid] = -42
            W_[tid] = W[tid] * W[rcv[tid]]
            rcv_[tid] = rcv[rcv[tid]]


@ti.kernel
def flow_cuda_path_accum_upward_kernel2(rcv: ti.template(), W: ti.template(), p: ti.template(), 
                                        rcv_: ti.template(), W_: ti.template(), p_: ti.template(), n: int):
    """1:1 port of CUDA upward path accumulation kernel 2"""
    for tid in range(n):
        if tid < n and W[tid] > 0.001:
            if p_[tid] != -42:
                p_idx = p_[tid]
                if p_idx >= 0 and p_idx < n:
                    p[p_idx] = 1.0
            W[tid] = W_[tid]
            rcv[tid] = rcv_[tid]


# ============================================================================
# BASIN MANAGEMENT KERNELS (1:1 CUDA PORT)
# ============================================================================

@ti.kernel
def indexed_set_id(N: int, locs: ti.template(), offset: int, dst: ti.template()):
    """1:1 port of CUDA indexed ID setting kernel"""
    for id in range(N):
        if id >= N:
            continue
        dst[locs[id]] = id + offset


@ti.kernel
def comp_basin_edgez(basin: ti.template(), z: ti.template(), bignum: float, res: int, basin_edgez: ti.template()):
    """1:1 port of CUDA basin edge computation kernel"""
    for x, y in ti.ndrange((1, res-1), (1, res-1)):
        loc = y * res + x

        ref = basin[loc]
        bhix = basin[loc+1]
        blox = basin[loc-1]
        bhiy = basin[loc + res]
        bloy = basin[loc - res]

        val = bignum  # Initialize val
        if (ref <= 0 or 
            (ref == bhix and ref == blox and ref == bhiy and ref == bloy)):
            val = bignum
        else:
            val = z[loc]

        nval = bignum
        if ref != bloy:
            nval = min(nval, z[loc - res])
        if ref != bhiy:
            nval = min(nval, z[loc + res])
        if ref != blox:
            nval = min(nval, z[loc - 1])
        if ref != bhix:
            nval = min(nval, z[loc + 1])

        basin_edgez[loc] = max(nval, val)


@ti.kernel
def create_basin2(basin_edgez: ti.template(), basin: ti.template(), bignum: float, basin2: ti.template(), N: int):
    """1:1 port of CUDA basin2 creation kernel (equivalent to torch.where)"""
    nonzero_basin2 = 0
    for i in range(N):
        if basin_edgez[i] == bignum:
            basin2[i] = 0
        else:
            basin2[i] = basin[i]  # Both are now ti.i64
            nonzero_basin2 += 1
    if N > 0:
        print(f"  DEBUG: create_basin2 produced {nonzero_basin2} non-zero basin2 values out of {N}")


@ti.kernel
def propag_basin_route_all(N: int, basin_route: ti.template()):
    """1:1 port of CUDA basin route propagation kernel"""
    changes = 0
    for i in basin_route:
        if basin_route[i] == basin_route[basin_route[i]]:
            continue
        basin_route[i] = basin_route[basin_route[i]]


@ti.kernel
def propag_basin_route_lm(N_: ti.template(), keep: ti.template(), p_lm: ti.template(), basin_route: ti.template()):
    """1:1 port of CUDA basin route propagation for local minima"""
    N = N_[0]
    for id in range(N):
        if id >= N or id == 0:
            continue
        lmid = p_lm[keep[id]-1]
        if id >= N or basin_route[lmid] == basin_route[basin_route[lmid]]:
            continue
        basin_route[lmid] = basin_route[basin_route[lmid]]


@ti.kernel
def update_all_basins(N: int, basin: ti.template(), basin_route: ti.template()):
    """1:1 port of CUDA basin update kernel"""
    max_basin_nonzero = 0
    for id in range(N):
        if id >= N:
            continue
        basin[id] = basin[basin_route[id]]
        if basin[id] > 0 and id > max_basin_nonzero:
            max_basin_nonzero = id
        if id == N - 1:  # Last iteration
            print(f"  DEBUG: update_all_basins max_nonzero_basin_id={max_basin_nonzero}, N={N}")


@ti.kernel
def update_basin_route(N_: ti.template(), keep: ti.template(), p_lm: ti.template(), 
                      b_rcv: ti.template(), basin_route: ti.template(), basin: ti.template()):
    """1:1 port of CUDA basin route update kernel"""
    N = N_[0]
    for id in range(N):
        if id >= N or id == 0:
            continue

        keep_id = keep[id]
        b_rcv_keep = b_rcv[keep_id]
        lm_from = p_lm[keep_id-1]
        if b_rcv_keep == 0:
            basin[lm_from] = 0
            continue

        basin_route[lm_from] = p_lm[b_rcv_keep-1]


# ============================================================================
# PATH COMPUTATION KERNELS (1:1 CUDA PORT)
# ============================================================================

@ti.kernel
def compute_p_b_rcv(N: int, p: ti.template(), z: ti.template(), basin: ti.template(), bignum: float, res: int, 
                   p_rcv: ti.template(), b_rcv: ti.template(), b: ti.template()):
    """1:1 port of CUDA p_b_rcv computation kernel"""
    for id in range(N):
        if id >= N:
            continue

        loc = p[id]
        pn_arr = ti.Vector([loc + 1, loc - 1, loc + res, loc - res])
        minpnz = ti.f32(3.4028235e+37)  # FLT_MAX equivalent
        minpnz_n = 0
        mintest = 0

        for i in range(4):
            # EXACT CUDA logic: int pn = (id == 0) ? 0 : pn_arr[i];
            pn = 0 if id == 0 else pn_arr[i]
            
            # EXACT CUDA: NO bounds checking - direct memory access
            # Must match CUDA behavior exactly to get same results
            basintest = basin[pn]
            pnz = bignum if basintest == basin[loc] else max(z[pn], z[loc])
            
            if (pnz < minpnz) or ((pnz == minpnz) and (basintest < mintest)):
                minpnz = pnz
                minpnz_n = pn
                mintest = basintest

        p_rcv[id] = minpnz_n
        b_rcv[id] = mintest
        b[id] = basin[loc]


@ti.kernel
def set_keep_b(N: int, b: ti.template(), b_rcv: ti.template(), keep_b: ti.template()):
    """1:1 port of CUDA keep_b generation kernel"""
    for id in range(N):
        if id >= N:
            continue
            
        b_id = b[id]
        # EXACT CUDA: bool do_keep = ((b_rcv[b_rcv[b_id]] == b_id) && (b_rcv[b_id] > b_id)) ? false : true;
        do_keep = not ((b_rcv[b_rcv[b_id]] == b_id) and (b_rcv[b_id] > b_id))
        keep_b[id] = 1 if do_keep else 0


@ti.kernel
def set_keep(N: int, b: ti.template(), keep_b: ti.template(), offset: ti.template(), keep: ti.template()):
    """1:1 port of CUDA keep array setting kernel"""
    for id in range(N):
        if id >= N:
            continue
        if keep_b[id]:
            # EXACT CUDA: keep[offset[id] - 1] = b[id];
            keep[offset[id] - 1] = b[id]


# ============================================================================
# CARVING KERNELS (1:1 CUDA PORT)
# ============================================================================

@ti.kernel
def init_reverse(N_: ti.template(), keep: ti.template(), p: ti.template(), reverse_path: ti.template()):
    """1:1 port of CUDA reverse path initialization kernel"""
    N: ti.i64 = N_[0]
    max_target: ti.i64 = 0
    for id in range(N):
        if id == 0 or id >= N:
            continue
        keep_id: ti.i64 = keep[id]
        target_idx: ti.i64 = p[keep_id]
        # EXACT CUDA: reverse_path[p[keep_id]] = 1.0f;
        reverse_path[target_idx] = 1.0
        if target_idx > max_target:
            max_target = target_idx
        if id <= 5:  # Debug first few
            print(f"    init_reverse: id={id}, keep_id={keep_id}, target_idx={target_idx}")
    print(f"  DEBUG: init_reverse max_target_idx={max_target}, grid_size={reverse_path.shape[0]}")


@ti.kernel
def final1(N: int, reverse_path: ti.template(), W: ti.template(), rcv: ti.template(), rcv_: ti.template()):
    """1:1 port of CUDA final1 carving kernel - EXACT CUDA with race conditions"""
    for id in range(N):
        if id >= N:
            continue
        if reverse_path[id] > 0.0 and rcv[id] != id:
            # EXACT CUDA: rcv_[rcv[id]] = id - race conditions allowed
            rcv_[rcv[id]] = id
                    
        if reverse_path[id] > 0.0 and rcv[id] == id:
            W[id] = 1.0


@ti.kernel
def final2(N_: ti.template(), keep: ti.template(), p_rcv: ti.template(), p: ti.template(), rcv_: ti.template()):
    """1:1 port of CUDA final2 carving kernel - writes to rcv_ (parameter name matches CUDA call)"""
    N = N_[0]
    changes = 0
    overwrites = 0
    for id in range(N):
        if id == 0 or id >= N:
            continue
        keep_id = keep[id]
        
        # EXACT CUDA: rcv[p[keep_id]] = p_rcv[keep_id];
        target_idx = p[keep_id]
        new_rcv = p_rcv[keep_id]
        old_rcv = rcv_[target_idx]
        
        if old_rcv != new_rcv:
            changes += 1
            # DEBUG: Track if final2 overwrites final1's changes
            if changes <= 5:
                print(f"    final2: rcv_[{target_idx}] {old_rcv} -> {new_rcv}")
                # Check if this could create a cycle after the change
                if target_idx < rcv_.shape[0] and new_rcv < rcv_.shape[0]:
                    print(f"      before: rcv_[{new_rcv}] = {rcv_[new_rcv]}")
                    print(f"      after setting rcv_[{target_idx}]={new_rcv}, check if rcv_[{new_rcv}]=={target_idx}")
            
        rcv_[target_idx] = new_rcv
        
    if changes > 0:
        print(f"  DEBUG: final2 made {changes} changes")


# ============================================================================
# INITIALIZATION KERNELS (1:1 CUDA PORT)
# ============================================================================

@ti.kernel
def init_lakeflow(N: int, rcv: ti.template(), rcv_: ti.template(), W: ti.template(), W_: ti.template(), 
                 basin_edgez: ti.template(), bignum: float, reverse_path: ti.template()):
    """1:1 port of CUDA lakeflow initialization kernel"""
    for i in rcv:

        rcv_[i] = rcv[i]
        W_[i] = W[i]
        basin_edgez[i] = bignum
        reverse_path[i] = 0

@ti.kernel
def init_minh(S: int, minh: ti.template()):
    """Initialize minh array for basins"""
    for i in minh:
        minh[i] = 1e10


@ti.kernel
def get_final_count(keep_offset: ti.template(), S: int, final_count: ti.template()):
    """1:1 port of CUDA final count extraction kernel"""
    final_count[0] = keep_offset[S]


@ti.kernel  
def copy_result_back(N: int, rcv_src: ti.template(), rcv_dst: ti.template()):
    """Copy result back from working array to main array"""
    changes = 0
    for i in range(N):
        old_val = rcv_dst[i]
        new_val = rcv_src[i]
        
        if old_val != new_val:
            changes += 1
                    
        rcv_dst[i] = new_val
        
    if changes > 0:
        print(f"  DEBUG: copy_result_back copied {changes} changes")


# ============================================================================
# LAKEFLOW ALGORITHM INTERFACE FUNCTION
# ============================================================================

# EXACT CUDA: basin_route = rcv.clone() - initialize basin_route from rcv
@ti.kernel 
def init_basin_route_from_rcv(basin_route:ti.template(),rcv:ti.template()):
    for i in rcv:
        basin_route[i] = rcv[i]

@ti.kernel
def count_S(bound:ti.template(), rcv:ti.template()) -> int:
    S=0
    for i in rcv:
        if(rcv[i] == i and bound[i] == 0):
            ti.atomic_add(S,1)
    return S


def run_lakeflow_algorithm(flow_fields: BasicFlowFields, lakeflow_fields: LakeflowFields, 
                          bignum: float, carve: bool, num_iter: int):
    """
    Run complete lakeflow (depression routing) algorithm
    1:1 port of CUDA lakeflow with proper Taichi memory management
    """
    N = flow_fields.N
    S = count_S(flow_fields.bound,flow_fields.rcv)
    print(f"TOCHECK::S={S}")
    lakeflow_fields.S = S
    res = flow_fields.res
    
    logN = int(math.ceil(math.log2(float(N))))
    logS = int(math.ceil(math.log2(float(S))))

    # Initialize all fields
    init_lakeflow(N, flow_fields.rcv, flow_fields.rcv_, flow_fields.W, flow_fields.W_, 
                 lakeflow_fields.basin_edgez, bignum, lakeflow_fields.reverse_path)

    init_minh(S + 10, lakeflow_fields.minh)  # Initialize minh array for S+10 basins
    
    
    init_basin_route_from_rcv(lakeflow_fields.basin_route, flow_fields.rcv)
    # raise ValueError('HERE')

    # Basin route propagation
    propag_iter = logN
    for i in range(propag_iter):
        propag_basin_route_all(N, lakeflow_fields.basin_route)

    print(f"TOCHECKBASIN::{np.unique(lakeflow_fields.basin_route.to_numpy())}")
    # Basin setup
    indexed_set_id(S, lakeflow_fields.p_lm, 1, lakeflow_fields.basin)
    
    update_all_basins(N, lakeflow_fields.basin, lakeflow_fields.basin_route)

    # Basin edge computation
    comp_basin_edgez(lakeflow_fields.basin, flow_fields.z, bignum, res, lakeflow_fields.basin_edgez)
    create_basin2(lakeflow_fields.basin_edgez, lakeflow_fields.basin, bignum, lakeflow_fields.basin2, N)
    
    # Scatter-min atomic operation
    run_scatter_min_atomic(flow_fields.z, lakeflow_fields.basin_edgez, lakeflow_fields.basin2, 
                          flow_fields.rcv, lakeflow_fields.minh, lakeflow_fields.argminh, 
                          lakeflow_fields.argbasin_, lakeflow_fields.nbasin_)

    lakeflow_fields.minh[0] = 0.0
    lakeflow_fields.argminh[0] = 0

    # Path computation
    compute_p_b_rcv(S + 1, lakeflow_fields.argminh, flow_fields.z, lakeflow_fields.basin, 
                   bignum, res, lakeflow_fields.p_rcv, lakeflow_fields.b_rcv, lakeflow_fields.b)
    
    # Keep array generation and inclusive scan
    # EXACT CUDA: Process ALL S+1 elements like CUDA does (no artificial caps)
    actual_size = S + 1
    set_keep_b(actual_size, lakeflow_fields.b, lakeflow_fields.b_rcv, lakeflow_fields.keep_b)
    inclusive_scan(lakeflow_fields.keep_b, lakeflow_fields.keep_offset, actual_size)
    set_keep(actual_size, lakeflow_fields.b, lakeflow_fields.keep_b, lakeflow_fields.keep_offset, lakeflow_fields.keep)

    # Get final count
    get_final_count(lakeflow_fields.keep_offset, S, lakeflow_fields.final_count)

    # Update basin route for final count
    update_basin_route(lakeflow_fields.final_count, lakeflow_fields.keep, lakeflow_fields.p_lm, 
                      lakeflow_fields.b_rcv, lakeflow_fields.basin_route, lakeflow_fields.basin)

    # Basin route propagation for local minima
    for i in range(logS):
        propag_basin_route_lm(lakeflow_fields.final_count, lakeflow_fields.keep, 
                             lakeflow_fields.p_lm, lakeflow_fields.basin_route)

    # Carving (if enabled)
    if carve:
        print(f"  DEBUG: Starting carving with final_count={lakeflow_fields.final_count[0]}")
        # Initialize reverse path
        init_reverse(lakeflow_fields.final_count, lakeflow_fields.keep, 
                    lakeflow_fields.argminh, lakeflow_fields.reverse_path)
        
        # EXACT CUDA: Convert to int32 for path accumulation
        # auto rcv2 = rcv.to(torch::kInt32); auto W2 = W.clone();
        
        convert_to_int32(flow_fields.rcv_i32, flow_fields.rcv, flow_fields.W_i32, flow_fields.W)
        
        # Path accumulation upward - EXACT CUDA with int32 arrays
        for i in range(logN):
            flow_cuda_path_accum_upward_kernel1(flow_fields.rcv_i32, flow_fields.W_i32, lakeflow_fields.reverse_path, 
                                               flow_fields.rcv_temp, flow_fields.W_temp, flow_fields.p_temp, N)
            flow_cuda_path_accum_upward_kernel2(flow_fields.rcv_i32, flow_fields.W_i32, lakeflow_fields.reverse_path, 
                                               flow_fields.rcv_temp, flow_fields.W_temp, flow_fields.p_temp, N)

        # Final carving - EXACT CUDA: both write to rcv_ then swap happens OUTSIDE lakeflow
        final1(N, lakeflow_fields.reverse_path, flow_fields.W, flow_fields.rcv, flow_fields.rcv_)
        print(f"  DEBUG: About to call final2 with final_count={lakeflow_fields.final_count[0]}")
        # CRITICAL: final2 ALSO writes to rcv_ (same as CUDA line 408: rcv_ptr_)
        final2(lakeflow_fields.final_count, lakeflow_fields.keep, lakeflow_fields.p_rcv, 
              lakeflow_fields.argminh, flow_fields.rcv_)
        

        swaprcv(flow_fields.rcv,flow_fields.rcv_)
        # EXACT CUDA BEHAVIOR: The Python wrapper does rcv_, rcv = rcv, rcv_ AFTER lakeflow_cuda returns
        # DO NOT copy back here - leave carving results in rcv_ for external swap
        print(f"  DEBUG: Carving completed - results in rcv_, waiting for swap")

    return lakeflow_fields.basin, lakeflow_fields.argminh, lakeflow_fields.p_rcv, lakeflow_fields.keep_b, 0, lakeflow_fields.reverse_path, flow_fields.W