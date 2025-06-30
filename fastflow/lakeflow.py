import taichi as ti
import math
from scatter_min import run_flow_cuda_scatter_min_atomic
from parallel_scan import inclusive_scan

@ti.kernel
def flow_cuda_path_accum_upward_kernel1(rcv: ti.template(), W: ti.template(), p: ti.template(), rcv_: ti.template(), W_: ti.template(), p_: ti.template(), n: int):
    for tid in range(n):
        if tid < n and W[tid] > 0.001:
            if p[tid] > 0.001:
                p_[tid] = rcv[tid]
            else:
                p_[tid] = -42
            W_[tid] = W[tid] * W[rcv[tid]]
            rcv_[tid] = rcv[rcv[tid]]


@ti.kernel
def flow_cuda_path_accum_upward_kernel2(rcv: ti.template(), W: ti.template(), p: ti.template(), rcv_: ti.template(), W_: ti.template(), p_: ti.template(), n: int):
    for tid in range(n):
        if tid < n and W[tid] > 0.001:
            if p_[tid] != -42:
                p[p_[tid]] = 1.0
            W[tid] = W_[tid]
            rcv[tid] = rcv_[tid]


@ti.kernel
def indexed_set_id(N: int, locs: ti.template(), offset: int, dst: ti.template()):
    for id in range(N):
        if id >= N:
            continue
        dst[locs[id]] = id + offset


@ti.kernel
def comp_basin_edgez(basin: ti.template(), z: ti.template(), bignum: float, res: int, basin_edgez: ti.template()):
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
def compute_p_b_rcv(N: int, p: ti.template(), z: ti.template(), basin: ti.template(), bignum: float, res: int, p_rcv: ti.template(), b_rcv: ti.template(), b: ti.template()):
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
            # RESTORING EXACT CUDA LOGIC even if it seems strange
            pn = 0 if id == 0 else pn_arr[i]
            
            # CUDA EQUIVALENT: Simulate CUDA's out-of-bounds behavior  
            basintest = 0
            pnz = bignum
            
            if pn >= 0 and pn < basin.shape[0]:
                basintest = basin[pn]
                pnz = bignum if basintest == basin[loc] else max(z[pn], z[loc])
            else:
                # CUDA-like: Out of bounds access gets boundary values
                basintest = 0  # Simulate boundary basin value
                pnz = bignum    # Simulate high cost for out-of-bounds

            if (pnz < minpnz) or ((pnz == minpnz) and (basintest < mintest)):
                minpnz = pnz
                minpnz_n = pn
                mintest = basintest

        p_rcv[id] = minpnz_n
        b_rcv[id] = mintest
        b[id] = basin[loc]


@ti.kernel
def set_keep_b(N: int, b: ti.template(), b_rcv: ti.template(), keep_b: ti.template()):
    for id in range(N):
        if id >= N:
            continue
            
        b_id = b[id]
        # EXACT CUDA: bool do_keep = ((b_rcv[b_rcv[b_id]] == b_id) && (b_rcv[b_id] > b_id)) ? false : true;
        do_keep = not ((b_rcv[b_rcv[b_id]] == b_id) and (b_rcv[b_id] > b_id))
        keep_b[id] = 1 if do_keep else 0


@ti.kernel
def set_keep(N: int, b: ti.template(), keep_b: ti.template(), offset: ti.template(), keep: ti.template()):
    for id in range(N):
        if id >= N:
            continue
        if keep_b[id]:
            # EXACT CUDA: keep[offset[id] - 1] = b[id];
            keep[offset[id] - 1] = b[id]


@ti.kernel
def init_reverse(N_: ti.template(), keep: ti.template(), p: ti.template(), reverse_path: ti.template()):
    N = N_[0]
    for id in range(N):
        if id == 0 or id >= N:
            continue
        keep_id = keep[id]
        # EXACT CUDA: reverse_path[p[keep_id]] = 1.0f;
        reverse_path[p[keep_id]] = 1.0


@ti.kernel
def final1(N: int, reverse_path: ti.template(), W: ti.template(), rcv: ti.template(), rcv_: ti.template()):
    for id in range(N):
        if id >= N:
            continue
        if reverse_path[id] > 0.0 and rcv[id] != id:
            rcv_[rcv[id]] = id
        if reverse_path[id] > 0.0 and rcv[id] == id:
            W[id] = 1.0


@ti.kernel
def final2(N_: ti.template(), keep: ti.template(), p_rcv: ti.template(), p: ti.template(), rcv: ti.template()):
    N = N_[0]
    changes = 0
    for id in range(N):
        if id == 0 or id >= N:
            continue
        keep_id = keep[id]
        # EXACT CUDA: rcv[p[keep_id]] = p_rcv[keep_id];
        old_val = rcv[p[keep_id]]
        new_val = p_rcv[keep_id]
        if old_val != new_val:
            changes += 1
        rcv[p[keep_id]] = p_rcv[keep_id]
    if changes > 0:
        print(f"  DEBUG: final2 made {changes} changes out of {N} processed")


@ti.kernel
def init(N: int, rcv: ti.template(), rcv_: ti.template(), W: ti.template(), W_: ti.template(), basin_edgez: ti.template(), bignum: float, reverse_path: ti.template(), minh: ti.template()):
    for id in range(N):
        if id >= N:
            continue
        rcv_[id] = rcv[id]
        W_[id] = W[id]
        basin_edgez[id] = bignum
        reverse_path[id] = 0
        minh[id] = 1e10


@ti.kernel
def propag_basin_route_all(N: int, basin_route: ti.template()):
    for id in range(N):
        if id >= N or basin_route[id] == basin_route[basin_route[id]]:
            continue
        basin_route[id] = basin_route[basin_route[id]]


@ti.kernel
def propag_basin_route_lm(N_: ti.template(), keep: ti.template(), p_lm: ti.template(), basin_route: ti.template()):
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
    for id in range(N):
        if id >= N:
            continue
        basin[id] = basin[basin_route[id]]


@ti.kernel
def update_basin_route(N_: ti.template(), keep: ti.template(), p_lm: ti.template(), b_rcv: ti.template(), basin_route: ti.template(), basin: ti.template()):
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


@ti.kernel
def create_basin2(basin_edgez: ti.template(), basin: ti.template(), bignum: float, basin2: ti.template(), N: int):
    for i in range(N):
        if basin_edgez[i] == bignum:
            basin2[i] = 0
        else:
            basin2[i] = basin[i]


# inclusive_scan is now imported from parallel_scan.py


@ti.kernel
def get_final_count(keep_offset: ti.template(), S: int, final_count: ti.template()):
    final_count[0] = keep_offset[S]


def declare_lakeflow_cuda(N: int, S: int):
    rcv_0 = ti.field(ti.i64, shape=N)
    W_0 = ti.field(ti.f32, shape=N)
    p_0 = ti.field(ti.i64, shape=N)
    basin2 = ti.field(ti.i64, shape=N)
    keep_offset = ti.field(ti.i64, shape=N+1)
    keep = ti.field(ti.i64, shape=N+1)
    final_count = ti.field(ti.i64, shape=1)
    return rcv_0, W_0, p_0, basin2, keep_offset, keep, final_count

def run_lakeflow_cuda(N: int, S: int, res: int, B: int, p_lm: ti.template(), rcv: ti.template(), rcv_: ti.template(), W: ti.template(), W_: ti.template(), basin: ti.template(), basin_route: ti.template(), basin_edgez: ti.template(), bound_ind: ti.template(), bignum: float, z: ti.template(), argminh: ti.template(), minh: ti.template(), p_rcv: ti.template(), b_rcv: ti.template(), b: ti.template(), keep_add_space: ti.template(), carve: bool, reverse_path: ti.template(), num_iter: int, lakeflow_fields: tuple, scatter_fields: tuple):

    # Unpack pre-allocated fields
    rcv_0, W_0, p_0, basin2, keep_offset, keep, final_count = lakeflow_fields
    argbasin_, nbasin_ = scatter_fields
    
    logN = int(math.ceil(math.log2(float(N))))
    logS = int(math.ceil(math.log2(float(S))))

    init(N, rcv, rcv_, W, W_, basin_edgez, bignum, reverse_path, minh)

    propag_iter = logN if num_iter == 0 else 1
    for i in range(propag_iter):
        propag_basin_route_all(N, basin_route)

    indexed_set_id(S, p_lm, 1, basin)
    update_all_basins(N, basin, basin_route)

    comp_basin_edgez(basin, z, bignum, res, basin_edgez)

    # Create basin2 equivalent to torch.where(basin_edgez == bignum, 0, basin)
    create_basin2(basin_edgez, basin, bignum, basin2, N)
    
    run_flow_cuda_scatter_min_atomic(z, basin_edgez, basin2, rcv, minh, argminh, argbasin_, nbasin_)

    minh[0] = 0.0
    argminh[0] = 0

    compute_p_b_rcv(S + 1, argminh, z, basin, bignum, res, p_rcv, b_rcv, b)

    # Manual inclusive scan implementation
    keep_b = keep_add_space
    
    set_keep_b(S + 1, b, b_rcv, keep_b)
    
    # Inclusive scan
    inclusive_scan(keep_b, keep_offset, S + 1)
    
    set_keep(S + 1, b, keep_b, keep_offset, keep)

    # Get final count
    get_final_count(keep_offset, S, final_count)

    update_basin_route(final_count, keep, p_lm, b_rcv, basin_route, basin)

    for i in range(logS):
        propag_basin_route_lm(final_count, keep, p_lm, basin_route)

    if carve:
        init_reverse(final_count, keep, argminh, reverse_path)
        
        for i in range(logN):
            flow_cuda_path_accum_upward_kernel1(rcv, W, reverse_path, rcv_0, W_0, p_0, N)
            flow_cuda_path_accum_upward_kernel2(rcv, W, reverse_path, rcv_0, W_0, p_0, N)

        

        final1(N, reverse_path, W, rcv, rcv_)
        # DEBUG: Print final_count before calling final2
        print(f"  Final count for carving: {final_count[0]}")
        final2(final_count, keep, p_rcv, argminh, rcv_)  # EXACT CUDA: use rcv_ like CUDA uses rcv_ptr_
        
        # Copy result back from rcv_ to rcv to match CUDA behavior
        @ti.kernel
        def copy_result_back():
            for i in range(N):
                rcv[i] = rcv_[i]
        copy_result_back()

    return basin, argminh, p_rcv, keep_add_space, 0, reverse_path, W