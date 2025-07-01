"""
1:1 port of src/cuda/core/tree_max_down.cu
"""
import taichi as ti

@ti.kernel
def flow_cuda_tree_max_downward_kernel(rcv: ti.template(), p: ti.template(), 
                                      rcv_: ti.template(), p_: ti.template(), n: int):
    """1:1 port of flow_cuda_tree_max_downward_kernel CUDA kernel"""
    for tid in range(n):
        if tid < n and rcv[rcv[tid]] != rcv[tid]:
            p_[tid] = ti.max(p[rcv[tid]], p[tid])
            rcv_[tid] = rcv[rcv[tid]]