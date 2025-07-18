import taichi as ti
import math

# EXACT 1:1 port of CUDA tree accumulation

@ti.func
def getSrc_fixed(src: ti.template(), id: int, iter: int):
    entry = src[id]
    flip = entry < 0
    flip = (not flip) if ti.abs(entry) == (iter + 1) else flip
    return flip

@ti.func  
def updateSrc_fixed(src: ti.template(), tid: int, iter: int, flip: int):
    src[tid] = (1 if flip else -1) * (iter + 1)

@ti.kernel
def fuse_fixed(A: ti.template(), src: ti.template(), B: ti.template(), n: int, iter: int):
    for tid in range(n):
        if getSrc_fixed(src, tid, iter):
            A[tid] = B[tid]

@ti.kernel  
def rcv2donor_fixed(rcv: ti.template(), dnr: ti.template(), ndnr: ti.template(), n: int):
    for tid in range(n):
        if tid < n and rcv[tid] != tid:
            rcv_idx = ti.cast(rcv[tid], ti.i64)
            old_val = ti.atomic_add(ndnr[rcv_idx], 1)
            dnr[rcv_idx * 4 + old_val] = tid

@ti.kernel
def rake_compress_accum_fixed(
    dnr: ti.template(), ndnr: ti.template(), p: ti.template(), src: ti.template(),
    dnr_: ti.template(), ndnr_: ti.template(), p_: ti.template(), n: int, iter: int):
    
    for tid in range(n):
        if tid >= n:
            continue
            
        flip = getSrc_fixed(src, tid, iter)
        
        # Select arrays based on flip
        A_dnr = dnr_ if flip else dnr
        A_ndnr = ndnr_ if flip else ndnr  
        A_p = p_ if flip else p
        
        B_dnr = dnr if flip else dnr_
        B_ndnr = ndnr if flip else ndnr_
        B_p = p if flip else p_
        
        worked = 0
        donors = ti.Vector([-1, -1, -1, -1], dt=ti.i64)
        todo = A_ndnr[tid]
        base = tid * 4
        p_added = 0.0
        
        # Initialize donors from A arrays
        for i in range(min(todo, 4)):
            donors[i] = A_dnr[base + i]
        
        # Process donors
        i = 0
        while i < todo and todo <= 4:
            did = donors[i]
            if did == -1:
                i += 1
                continue
                
            flip_did = getSrc_fixed(src, did, iter)
            
            # Select C arrays for donor
            C_ndnr = ndnr_ if flip_did else ndnr
            C_p = p_ if flip_did else p
            C_dnr = dnr_ if flip_did else dnr
            
            if C_ndnr[did] <= 1:
                if worked == 0:
                    p_added = A_p[tid]
                worked = 1
                
                p_added += C_p[did]
                
                if C_ndnr[did] == 0:
                    # Remove this donor
                    todo -= 1
                    if todo > i:
                        donors[i] = donors[todo]
                        donors[todo] = -1
                    else:
                        donors[i] = -1
                else:
                    # Replace with its first donor
                    donors[i] = C_dnr[did * 4]
                    i += 1
            else:
                i += 1
        
        if worked == 1:
            B_ndnr[tid] = todo
            B_p[tid] = p_added
            for j in range(min(todo, 4)):
                B_dnr[base + j] = donors[j]
            updateSrc_fixed(src, tid, iter, flip)

def fixed_tree_accum_upward_rake_compress(rcv: ti.template(), W: ti.template(), p: ti.template()):
    """Fixed tree accumulation that should match CUDA exactly"""
    n = p.shape[0]
    logn = int(math.ceil(math.log2(float(n))))
    
    # Declare working arrays
    dnr0 = ti.field(ti.i64, shape=n*4)
    dnr_0 = ti.field(ti.i64, shape=n*4)
    ndnr0 = ti.field(ti.i64, shape=n)
    ndnr_0 = ti.field(ti.i64, shape=n)
    p_0 = ti.field(ti.f32, shape=n)
    src = ti.field(ti.i64, shape=n)
    
    # Initialize
    ndnr0.fill(0)
    src.fill(0)
    
    # Build donor arrays
    rcv2donor_fixed(rcv, dnr0, ndnr0, n)
    
    # Rake-compress iterations
    for i in range(logn):
        rake_compress_accum_fixed(dnr0, ndnr0, p, src, dnr_0, ndnr_0, p_0, n, i)
    
    # Final fuse
    fuse_fixed(p, src, p_0, n, logn)
    
    return p