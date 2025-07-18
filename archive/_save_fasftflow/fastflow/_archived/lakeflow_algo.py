import math
import taichi as ti
import order
from lakeflow import declare_lakeflow_cuda
from scatter_min import declare_flow_cuda_scatter_min_atomic

@ti.kernel
def swap_fields(A:ti.template(), B:ti.template()):
    for j in A:
        temp = A[j]
        A[j] = B[j]
        B[j] = temp


@ti.kernel
def update_jump(K:ti.template(), keep:ti.template(), p_lm:ti.template(), rcv:ti.template(), p_rcv:ti.template(),W:ti.template()):
    for j in ti.ndrange((1, K)):
        keep_idx = keep[j]
        idx = p_lm[keep_idx-1]
        rcv[idx] = p_rcv[keep_idx]  # update rcv
        W[idx] = 1.0  # update W

@ti.kernel
def find_local_minima(N:ti.template(), bound:ti.template(),rcv:ti.template(),p_lm:ti.template()) -> int:
    count = 0
    for j in range(N):
        if not bound[j] and rcv[j] == j:
            p_lm[count] = j
            count += 1
    return count

def lakeflow_1d(z, bound, rcv, W, res, method='carve'):
    if method == 'none':
        return rcv, W

    logn = math.ceil(math.log2(rcv.shape[0]))
    big_num = 1e10  # should be greater than max terrain height
    basin = ti.field(ti.i64, shape=rcv.shape[0])
    basin.fill(0)
    
    # EXACT COPY - bound.nonzero() equivalent for 1D
    bound_ind = ti.field(ti.i64, shape=bound.shape[0])
    B = 0
    
    @ti.kernel
    def find_bound_nonzero() -> int:
        count = 0
        for i in range(bound.shape[0]):
            if bound[i]:
                bound_ind[count] = i
                count += 1
        return count
    
    B = find_bound_nonzero()

    N = rcv.shape[0]
    rcv_ = ti.field(ti.i64, shape=N)
    W_ = ti.field(ti.f32, shape=N)
    basin_route = ti.field(ti.i64, shape=N)
    basin_edgez = ti.field(ti.f32, shape=N)
    
    @ti.kernel
    def copy_fields():
        for i in range(N):
            rcv_[i] = rcv[i]
            W_[i] = W[i]
            basin_route[i] = rcv[i]

    copy_fields()

    minh_space = ti.field(ti.f32, shape=N+1)
    argminh_space = ti.field(ti.i64, shape=N+1)
    p_rcv_space = ti.field(ti.i64, shape=N+1)
    b_rcv_space = ti.field(ti.i64, shape=N+1)
    b_space = ti.field(ti.i64, shape=N+1)
    keep_space = ti.field(ti.i64, shape=3*(N+1))
    reverse_path = ti.field(ti.f32, shape=N)
    lakeflow_fields = declare_lakeflow_cuda(N, N//4)
    scatter_fields = declare_flow_cuda_scatter_min_atomic(N, N+1)

    carve_b = method == "carve"

    # EXACT COPY - torch.where((~bound.view(-1)) & (rcv.view(-1)==idt))[0]
    p_lm = ti.field(ti.i64, shape=N)
    
    

    for i in range(logn):
        
        S = 0
        

        S = find_local_minima(N, bound,rcv,p_lm)
        if S == 0:
            break
        
        # Use Taichi lakeflow_cuda function - EXACT 1:1 PORT
        basin, p, p_rcv, keep, K, reverse_path, W = order.lakeflow_cuda(
            N, S, res, B, p_lm, rcv, rcv_, W, W_, basin, basin_route, basin_edgez, bound_ind, big_num, z, 
            argminh_space, minh_space, p_rcv_space, b_rcv_space, b_space, keep_space, carve_b, reverse_path, 
            i, lakeflow_fields, scatter_fields)

        # rcv_, rcv = rcv, rcv_
        swap_fields(rcv_,rcv)

        if method == 'carve':
            pass
            
        elif method == 'jump':
            K = keep[N + 1 + S]
            update_jump(K, keep, p_lm, rcv, p_rcv,W)
        
        else:
            raise NotImplementedError("Choose among 'carve', 'jump' or 'none'")

    return rcv, W