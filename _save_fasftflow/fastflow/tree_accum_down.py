import taichi as ti
import math
from kernel import copy_data_kernel

@ti.kernel
def flow_cuda_tree_accum_downward_kernel(rcv: ti.template(), W: ti.template(), p: ti.template(), rcv_: ti.template(), W_: ti.template(), p_: ti.template(), n: int):
    for tid in range(n):
        if tid < n:
            p_[tid] = p[tid] + W[tid] * p[rcv[tid]]
            W_[tid] = W[tid] * W[rcv[tid]]
            rcv_[tid] = rcv[rcv[tid]]


def declare_flow_cuda_tree_accum_downward(n: int):
    rcv_0 = ti.field(ti.i64, shape=n)
    W_0 = ti.field(ti.f32, shape=n)
    p_0 = ti.field(ti.f32, shape=n)
    return rcv_0, W_0, p_0

def run_flow_cuda_tree_accum_downward(rcv: ti.template(), W: ti.template(), p: ti.template(), rcv_0: ti.template(), W_0: ti.template(), p_0: ti.template()):
    n = p.shape[0]
    logn = int(math.ceil(math.log2(float(n))))

    copy_data_kernel(rcv, rcv_0, n)
    copy_data_kernel(W, W_0, n)
    copy_data_kernel(p, p_0, n)

    for i in range(logn):
        flow_cuda_tree_accum_downward_kernel(rcv, W, p, rcv_0, W_0, p_0, n)
        copy_data_kernel(rcv_0, rcv, n)
        copy_data_kernel(W_0, W, n)
        copy_data_kernel(p_0, p, n)

    return p