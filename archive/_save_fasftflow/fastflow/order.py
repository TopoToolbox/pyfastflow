import taichi as ti
import numpy as np
from rcv import declare_rcv_matrix_cuda, run_rcv_matrix_cuda, declare_rcv_matrix_rand_cuda, run_rcv_matrix_rand_cuda
from tree_accum_up import flow_cuda_tree_accum_upward_rake_compress
from tree_max_down import declare_flow_cuda_tree_max_downward, run_flow_cuda_tree_max_downward
from lakeflow import declare_lakeflow_cuda, run_lakeflow_cuda
from erode_deposit import declare_erode_deposit_cuda, run_erode_deposit_cuda
from scatter_min import declare_flow_cuda_scatter_min_atomic, run_flow_cuda_scatter_min_atomic
from tree_accum_down import declare_flow_cuda_tree_accum_downward, run_flow_cuda_tree_accum_downward


@ti.kernel
def create_default_bounds(z: ti.template(), bound: ti.template()):
    res = z.shape[0]
    for i, j in ti.ndrange(res, res):
        bound[i, j] = (i == 0) or (i == res-1) or (j == 0) or (j == res-1)

@ti.kernel
def convert_bound_2d_to_1d(bound_2d: ti.template(), bound_1d: ti.template(), res: int):
    """Convert 2D boundary to 1D"""
    for i, j in ti.ndrange(res, res):
        bound_1d[i * res + j] = bound_2d[i, j]

@ti.kernel
def convert_z_2d_to_1d(z_2d: ti.template(), z_1d: ti.template(), res: int):
    """Convert 2D z field to 1D"""
    for i, j in ti.ndrange(res, res):
        z_1d[i * res + j] = z_2d[i, j]


def default_bounds_taichi(z: ti.template()):
    res = z.shape[0]
    bound = ti.field(ti.i64, shape=(res, res))  # Use i64 instead of bool
    create_default_bounds(z, bound)
    return bound


def rcv_matrix_1d(z: ti.template(), bound: ti.template()):
    # z and bound are already 1D - NO DUPLICATION
    N = z.shape[0]
    res = int(N**0.5)
    
    # Declare and run RCV
    rcv_fields = declare_rcv_matrix_cuda(res)
    rcv, W = run_rcv_matrix_cuda(z, bound, *rcv_fields)
    return rcv, W


def rcv_matrix_rand_1d(z: ti.template(), bound: ti.template(), rand_array: ti.template()):
    # z and bound are already 1D - NO DUPLICATION
    N = z.shape[0]
    res = int(N**0.5)
    
    # Declare and run RCV
    rcv_fields = declare_rcv_matrix_rand_cuda(res)
    rcv, W = run_rcv_matrix_rand_cuda(z, bound, rand_array, *rcv_fields)
    return rcv, W


def default_bounds_taichi_1d(z: ti.template(), res: int):
    bound = ti.field(ti.i64, shape=z.shape[0])
    create_default_bounds_1d(z, bound, res)
    return bound


@ti.kernel
def create_default_bounds_1d(z: ti.template(), bound: ti.template(), res: int):
    for i in range(z.shape[0]):
        ii = i // res
        jj = i % res
        bound[i] = 1 if (ii == 0 or ii == res-1 or jj == 0 or jj == res-1) else 0


def tree_accum_upward_(rcv: ti.template(), W: ti.template(), p: ti.template()):
    # Use simple version for debugging
    import simple_tree_accum
    return simple_tree_accum.flow_cuda_tree_accum_upward_rake_compress_simple(rcv, W, p)


def tree_max_downward_(rcv: ti.template(), p: ti.template()):
    tree_max_fields = declare_flow_cuda_tree_max_downward(p.shape[0])
    return run_flow_cuda_tree_max_downward(rcv, p, *tree_max_fields)


def lakeflow_cuda(N, S, res, B, p_lm, rcv, rcv_, W, W_, basin, basin_route, basin_edgez, bound_ind, big_num, z,
                 argmin_space, minh_space, p_rcv, b, b_rcv, keep_space, carve, reverse_path, num_iter, lakeflow_fields, scatter_fields):
    # lakeflow_fields = declare_lakeflow_cuda(N, N//4)
    # scatter_fields = declare_flow_cuda_scatter_min_atomic(N, N+1)
    return run_lakeflow_cuda(N, S, res, B, p_lm, rcv, rcv_, W, W_, basin, basin_route, basin_edgez, bound_ind, big_num, z,
                           argmin_space, minh_space, p_rcv, b, b_rcv, keep_space, carve, reverse_path, num_iter,
                           lakeflow_fields, scatter_fields)


def erode_deposit_cuda(z: ti.template(), bound: ti.template(), rcv: ti.template(), drain: ti.template(), Qs: ti.template(), 
                      dt: float, dx: float, k_spl: float, k_t: float, k_h: float, k_d: float, m: float):
    erode_fields = declare_erode_deposit_cuda(z.shape[0])
    tree_fields = declare_flow_cuda_tree_accum_downward(z.shape[0])
    return run_erode_deposit_cuda(z, bound, rcv, drain, Qs, dt, dx, k_spl, k_t, k_h, k_d, m, erode_fields, tree_fields)