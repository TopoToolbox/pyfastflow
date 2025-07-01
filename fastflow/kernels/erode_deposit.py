"""
1:1 port of src/cuda/core/erode_deposit.cu
"""
import taichi as ti

@ti.kernel
def erode_deposit_kernel(z: ti.template(), bound: ti.template(), drain: ti.template(), Qs: ti.template(),
                        dt: float, dx: float, k_spl: float, k_t: float, k_h: float, k_d: float, m: float,
                        pe: ti.template(), We: ti.template(), N: int):
    """1:1 port of erode_deposit_kernel CUDA kernel"""
    for idx in range(ti.i32(N)):
        if idx >= N:
            continue
            
        if not bound[idx]:
            clamp_drain = ti.min(k_d / drain[idx], 1.0)
            z[idx] += dt * clamp_drain * Qs[idx]
        
        K = dt / dx * (k_spl * ti.pow(drain[idx], m) + k_t + k_h * ti.pow(drain[idx], -m))
        pe[idx] = z[idx] / (1.0 + K)
        We[idx] = K / (1.0 + K)
        
        if bound[idx]:
            pe[idx] = z[idx]
            We[idx] = 0.0