import taichi as ti
from tree_accum_down import run_flow_cuda_tree_accum_downward

@ti.kernel
def erode_deposit_kernel(z: ti.template(), bound: ti.template(), drain: ti.template(), Qs: ti.template(), 
                        dt: float, dx: float, k_spl: float, k_t: float, k_h: float, k_d: float, m: float,
                        pe: ti.template(), We: ti.template(), N: int):
    for idx in range(N):
        if idx >= N:
            continue

        if not bound[idx]:
            clamp_drain = min(k_d / drain[idx], 1.0)
            z[idx] += dt * clamp_drain * Qs[idx]

        K = dt / dx * (k_spl * ti.pow(drain[idx], m) + k_t + k_h * ti.pow(drain[idx], -m))
        pe[idx] = z[idx] / (1.0 + K)
        We[idx] = K / (1.0 + K)

        if bound[idx]:
            pe[idx] = z[idx]
            We[idx] = 0.0


def declare_erode_deposit_cuda(N: int):
    pe = ti.field(ti.f32, shape=N)
    We = ti.field(ti.f32, shape=N)
    return pe, We

def run_erode_deposit_cuda(z: ti.template(), bound: ti.template(), rcv: ti.template(), drain: ti.template(), Qs: ti.template(),
                          dt: float, dx: float, k_spl: float, k_t: float, k_h: float, k_d: float, m: float,
                          pe: ti.template(), We: ti.template(), tree_fields: tuple):
    N = z.shape[0]

    erode_deposit_kernel(z, bound, drain, Qs, dt, dx, k_spl, k_t, k_h, k_d, m, pe, We, N)

    rcv_0, W_0, p_0 = tree_fields
    z = run_flow_cuda_tree_accum_downward(rcv, We, pe, rcv_0, W_0, p_0)
    return z