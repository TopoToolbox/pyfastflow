"""
1:1 port of src/cuda/core/tree_accum_up.cu
"""
import taichi as ti

@ti.func
def getSrc(src: ti.template(), id: int, iter: int) -> bool:
    """1:1 port of getSrc CUDA device function"""
    entry = src[id]
    flip = entry < 0
    flip = (not flip) if (abs(entry) == (iter + 1)) else flip
    return flip


@ti.func
def updateSrc(src: ti.template(), tid: int, iter: int, flip: bool):
    """1:1 port of updateSrc CUDA device function"""
    src[tid] = (1 if flip else -1) * (iter + 1)


@ti.kernel
def fuse(A: ti.template(), src: ti.template(), B: ti.template(), n: int, iter: int):
    """1:1 port of fuse CUDA kernel"""
    for tid in range(ti.i32(n)):
        if tid >= n:
            continue
        if getSrc(src, tid, iter):
            A[tid] = B[tid]


@ti.kernel
def rcv2donor(rcv: ti.template(), dnr: ti.template(), ndnr: ti.template(), n: int, res: int):
    """1:1 port of rcv2donor CUDA kernel"""
    for tid in range(ti.i32(n)):
        if tid < n and rcv[tid] != tid:
            old_val = ti.atomic_add(ndnr[ti.i32(rcv[tid])], 1)
            dnr[ti.i32(rcv[tid]) * 4 + old_val] = tid


@ti.kernel
def rake_compress_accum(dnr: ti.template(), ndnr: ti.template(), p: ti.template(), src: ti.template(),
                       dnr_: ti.template(), ndnr_: ti.template(), p_: ti.template(), n: int, iter: int):
    """1:1 port of rake_compress_accum CUDA kernel"""
    for tid in range(ti.i32(n)):
        if tid >= n:
            continue
            
        flip = getSrc(src, tid, iter)
        
        worked = False
        donors = ti.Vector([-1, -1, -1, -1])
        todo = ndnr[tid] if not flip else ndnr_[tid]
        base = tid * 4
        p_added = 0.0
        
        i = 0
        while i < todo and i <4:
            if donors[i] == -1:
                donors[i] = dnr[base + i] if not flip else dnr_[base + i]
            did = donors[i]
            
            flip_donor = getSrc(src, did, iter)
            ndnr_val = ndnr[did] if not flip_donor else ndnr_[did]
            
            if ndnr_val <= 1:
                if not worked:
                    p_added = p[tid] if not flip else p_[tid]
                worked = True
                
                p_val = p[did] if not flip_donor else p_[did]
                p_added += p_val
                
                if ndnr_val == 0:
                    todo -= 1
                    if todo > i:
                        donors[i] = dnr[base + todo] if not flip else dnr_[base + todo]
                    i -= 1
                else:
                    donors[i] = dnr[did * 4] if not flip_donor else dnr_[did * 4]
            i += 1
            
        if worked:
            if flip:
                ndnr[tid] = todo
                p[tid] = p_added
                for j in range(todo):
                    dnr[base + j] = donors[j]
            else:
                ndnr_[tid] = todo
                p_[tid] = p_added
                for j in range(todo):
                    dnr_[base + j] = donors[j]
            updateSrc(src, tid, iter, flip)