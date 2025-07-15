import taichi as ti
import math
from kernel import copy_data_kernel

@ti.kernel
def flow_cuda_tree_max_downward_kernel(rcv: ti.template(), p: ti.template(), rcv_: ti.template(), p_: ti.template(), n: int):
    for tid in range(n):
        if tid < n and rcv[rcv[tid]] != rcv[tid]:
            p_[tid] = max(p[rcv[tid]], p[tid])
            rcv_[tid] = rcv[rcv[tid]]


def declare_flow_cuda_tree_max_downward(n: int):
    rcv_0 = ti.field(ti.i64, shape=n)
    p_0 = ti.field(ti.f32, shape=n)
    return rcv_0, p_0

def run_flow_cuda_tree_max_downward(rcv: ti.template(), p: ti.template(), rcv_0: ti.template(), p_0: ti.template()):
    n = p.shape[0]
    logn = int(math.ceil(math.log2(float(n))))

    copy_data_kernel(rcv, rcv_0, n)
    copy_data_kernel(p, p_0, n)

    for i in range(logn):
        flow_cuda_tree_max_downward_kernel(rcv, p, rcv_0, p_0, n)
        copy_data_kernel(rcv_0, rcv, n)
        copy_data_kernel(p_0, p, n)

    return p