"""
1:1 port of src/cuda/core/tree_accum_down.cu
"""
import taichi as ti

@ti.kernel
def copy_data(from_field: ti.template(), to_field: ti.template(), n: int):
    """1:1 port of copy_data CUDA kernel template"""
    for tid in range(ti.i32(n)):
        if tid < n:
            to_field[tid] = from_field[tid]


@ti.kernel
def flow_cuda_tree_accum_downward_kernel(rcv: ti.template(), W: ti.template(), p: ti.template(),
                                        rcv_: ti.template(), W_: ti.template(), p_: ti.template(), n: int):
    """1:1 port of flow_cuda_tree_accum_downward_kernel CUDA kernel"""
    for tid in range(ti.i32(n)):
        if tid < n:
            p_[tid] = p[tid] + W[tid] * p[rcv[tid]]
            W_[tid] = W[tid] * W[rcv[tid]]
            rcv_[tid] = rcv[rcv[tid]]