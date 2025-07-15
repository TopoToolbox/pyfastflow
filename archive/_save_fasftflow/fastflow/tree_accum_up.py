import taichi as ti
import math

@ti.func
def swap(A: ti.template(), B: ti.template(), idx: int):
    C = B[idx]
    B[idx] = A[idx]
    A[idx] = C


@ti.func
def getSrc(src: ti.template(), id: int, iter: int):
    entry = src[id]
    flip = entry < 0
    
    flip = (not flip) if abs(entry) == (iter + 1) else flip
    return flip


@ti.func
def updateSrc(src: ti.template(), tid: int, iter: int, flip: int):
    src[tid] = (1 if flip else -1) * (iter + 1)


@ti.kernel
def fuse(A: ti.template(), src: ti.template(), B: ti.template(), n: int, iter: int):
    for tid in range(n):
        if tid >= n:
            continue
        
        if getSrc(src, tid, iter):
            A[tid] = B[tid]

@ti.kernel
def copy_accumulated_values(p: ti.template(), p_0: ti.template(), n: int):
    """Copy accumulated values from p_0 to p where accumulation occurred"""
    for tid in range(n):
        if p_0[tid] > 0.0:  # If accumulation happened here
            p[tid] = p_0[tid]


@ti.kernel
def rcv2donor(rcv: ti.template(), dnr: ti.template(), ndnr: ti.template(), n: int, res: int):
    for tid in range(n):
        if tid < n and rcv[tid] != tid:
            rcv_idx = ti.cast(rcv[tid], ti.i64)  # Cast to i64
            old_val = ti.atomic_add(ndnr[rcv_idx], 1)
            dnr[rcv_idx * 4 + old_val] = tid


@ti.kernel
def rake_compress_accum(dnr: ti.template(), ndnr: ti.template(), p: ti.template(), src: ti.template(), dnr_: ti.template(), ndnr_: ti.template(), p_: ti.template(), n: int, iter: int):
    for tid in range(n):
        if tid >= n:
            continue

        # EXACT CUDA STATE mechanism
        # STATE A = {dnr, ndnr, p}
        # STATE B = {dnr_, ndnr_, p_}
        # STATE X = A, Y = B
        
        flip = getSrc(src, tid, iter)
        
        # if(flip) swap(A, B) - this determines which arrays to use as A and B
        # Need to use explicit conditionals for Taichi
        worked = 0
        donors = ti.Vector([-1, -1, -1, -1], dt=ti.i64)
        base = tid * 4
        p_added = 0.0
        
        # Get todo from A array (considering flip)
        todo = 0
        if flip == 0:
            todo = ndnr[tid]
        else:
            todo = ndnr_[tid]
        
        # Initialize donors from A array (considering flip)
        for i in range(min(todo, 4)):
            if donors[i] == -1:
                if flip == 0:
                    donors[i] = dnr[base + i]
                else:
                    donors[i] = dnr_[base + i]
        
        # EXACT CUDA loop: for (int i=0; i < todo; i++) but with i-- when removing
        i = 0
        while i < todo:
            did = donors[i]
            
            # STATE C = X, STATE D = Y
            # if(getSrc(src, did, iter)) swap(C, D)
            flip_did = getSrc(src, did, iter)
            
            # Get values from C array (original X/Y with flip_did)
            C_ndnr_val = 0
            C_p_val = 0.0
            
            # C = X if !flip_did else Y, where X=original {dnr, ndnr, p}, Y=original {dnr_, ndnr_, p_}
            if flip_did == 0:
                # C = original X = {dnr, ndnr, p}
                C_ndnr_val = ndnr[did]
                C_p_val = p[did]
            else:
                # C = original Y = {dnr_, ndnr_, p_}
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
                    # EXACT CUDA: donors[i--] = A.dnr[base + --todo]
                    todo -= 1
                    if flip == 0:
                        donors[i] = dnr[base + todo]
                    else:
                        donors[i] = dnr_[base + todo]
                    # Don't increment i (equivalent to i-- then loop increment)
                else:
                    # EXACT CUDA: donors[i] = C.dnr[did * 4]
                    if flip_did == 0:
                        donors[i] = dnr[did * 4]
                    else:
                        donors[i] = dnr_[did * 4]
                    i += 1
            else:
                i += 1
        
        if worked == 1:
            # EXACT CUDA: Write to B arrays
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
            
            # updateSrc(src, tid, iter, flip)
            updateSrc(src, tid, iter, flip)


def declare_flow_cuda_tree_accum_upward_rake_compress(n: int):
    dnr0 = ti.field(ti.i64, shape=n*4)
    dnr_0 = ti.field(ti.i64, shape=n*4)
    ndnr0 = ti.field(ti.i64, shape=n)
    ndnr_0 = ti.field(ti.i64, shape=n)
    p_0 = ti.field(ti.f32, shape=n)
    src = ti.field(ti.i64, shape=n)  # Use i64 instead of i32
    return dnr0, dnr_0, ndnr0, ndnr_0, p_0, src

def run_flow_cuda_tree_accum_upward_rake_compress(rcv: ti.template(), W: ti.template(), p: ti.template(), tree_up_fields: tuple):
    n = p.shape[0]
    logn = int(math.ceil(math.log2(float(n))))

    dnr0, dnr_0, ndnr0, ndnr_0, p_0, src = tree_up_fields

    # Initialize arrays - EXACT CUDA initialization
    ndnr0.fill(0)
    src.fill(0)
    
    res = int(math.sqrt(float(n)))

    # Build donor arrays
    rcv2donor(rcv, dnr0, ndnr0, n, res)

    # EXACT CUDA algorithm: for (int i=0; i < logn; i++)
    for i in range(logn):
        rake_compress_accum(dnr0, ndnr0, p, src, dnr_0, ndnr_0, p_0, n, i)

    # EXACT CUDA fuse call
    fuse(p, src, p_0, n, logn + 1)

    return p

# For backward compatibility - DEPRECATED: Creates fields in loop!
# Use run_flow_cuda_tree_accum_upward_rake_compress with pre-allocated fields instead
def flow_cuda_tree_accum_upward_rake_compress(rcv: ti.template(), W: ti.template(), p: ti.template()):
    n = p.shape[0]
    tree_up_fields = declare_flow_cuda_tree_accum_upward_rake_compress(n)  # ⚠️ CREATES FIELDS - NOT RECOMMENDED
    return run_flow_cuda_tree_accum_upward_rake_compress(rcv, W, p, tree_up_fields)