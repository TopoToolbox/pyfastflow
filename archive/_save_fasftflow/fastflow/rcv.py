import taichi as ti
import math

@ti.kernel
def make_rcv(z: ti.template(), res: int, N: int, boundary: ti.template(), rcv: ti.template(), W: ti.template()):
    for id in range(N):
        if id >= N:  # Extra safety check
            continue
        x = id % res
        y = id // res

        # Check flow directions (only flow downhill) - exact match to CUDA
        x_incr = (x > 0) and (z[id] > z[id - 1])
        x_decr = (x < (res - 1)) and (z[id] > z[id + 1])
        y_incr = (y > 0) and (z[id] > z[id - res])
        y_decr = (y < (res - 1)) and (z[id] > z[id + res])

        bound = boundary[id]

        is_rcv0 = x_incr and (not bound)
        is_rcv1 = x_decr and (not bound) 
        is_rcv2 = y_incr and (not bound)
        is_rcv3 = y_decr and (not bound)
        
        # EXACT match to CUDA logic
        rcv0 = id + (-1 if is_rcv0 else 0)
        rcv1 = id + (1 if is_rcv1 else 0)
        rcv2 = id + (-res if is_rcv2 else 0)
        rcv3 = id + (res if is_rcv3 else 0)

        # Calculate weights as elevation differences
        W0 = z[id] - z[rcv0]
        W1 = z[id] - z[rcv1]  
        W2 = z[id] - z[rcv2]
        W3 = z[id] - z[rcv3]

        sum_val = max(1e-7, (W0 + W1 + W2 + W3))
        W0 = W0 / sum_val
        W1 = W1 / sum_val
        W2 = W2 / sum_val
        W3 = W3 / sum_val

        rcvmax = rcv0
        Wmax = W0
        if W1 > Wmax:
            Wmax = W1
            rcvmax = rcv1
        if W2 > Wmax:
            Wmax = W2
            rcvmax = rcv2
        if W3 > Wmax:
            Wmax = W3
            rcvmax = rcv3

        rcv[id] = rcvmax
        W[id] = ti.ceil(Wmax)


def declare_rcv_matrix_cuda(res: int):
    N = res * res
    rcv = ti.field(ti.i64, shape=N)
    W = ti.field(ti.f32, shape=N)
    return rcv, W

def run_rcv_matrix_cuda(z: ti.template(), bound: ti.template(), rcv: ti.template(), W: ti.template()):
    N = z.shape[0]  # z is 1D, so N is its length
    res = int(N**0.5)  # res is sqrt(N) for square grid
    print(f"RCV Debug: N={N}, res={res}, res*res={res*res}")
    make_rcv(z, res, N, bound, rcv, W)
    return rcv, W


@ti.kernel
def make_rcv_rand(z: ti.template(), res: int, N: int, boundary: ti.template(), rcv: ti.template(), W: ti.template(), rand_array: ti.template()):
    for id in range(N):
        if id >= N:  # Extra safety check
            continue
        x = id % res
        y = id // res

        # Check flow directions (only flow downhill)  
        x_incr = (x > 0) and (z[id] > z[id - 1])
        x_decr = (x < (res - 1)) and (z[id] > z[id + 1])
        y_incr = (y > 0) and (z[id] > z[id - res])
        y_decr = (y < (res - 1)) and (z[id] > z[id + res])

        bound = boundary[id]

        is_rcv0 = x_incr and (not bound)
        is_rcv1 = x_decr and (not bound)
        is_rcv2 = y_incr and (not bound)
        is_rcv3 = y_decr and (not bound)
        
        # EXACT match to CUDA logic
        rcv0 = id + (-1 if is_rcv0 else 0)
        rcv1 = id + (1 if is_rcv1 else 0)
        rcv2 = id + (-res if is_rcv2 else 0)
        rcv3 = id + (res if is_rcv3 else 0)

        # Calculate weights as elevation differences
        W0 = z[id] - z[rcv0]
        W1 = z[id] - z[rcv1]
        W2 = z[id] - z[rcv2]
        W3 = z[id] - z[rcv3]

        sum_val = max(1e-7, (W0 + W1 + W2 + W3))
        W0 = W0 / sum_val
        W1 = W1 / sum_val
        W2 = W2 / sum_val
        W3 = W3 / sum_val

        W1 += W0
        W2 += W1
        W3 += W2

        rcv_result = rcv0  # Initialize with default
        rand_num = rand_array[id]
        if W0 > rand_num:
            rcv_result = rcv0
        elif W1 > rand_num:
            rcv_result = rcv1
        elif W2 > rand_num:
            rcv_result = rcv2
        else:
            rcv_result = rcv3
            
        rcv[id] = rcv_result
        W[id] = W3


def declare_rcv_matrix_rand_cuda(res: int):
    N = res * res
    rcv = ti.field(ti.i64, shape=N)
    W = ti.field(ti.f32, shape=N)
    return rcv, W

def run_rcv_matrix_rand_cuda(z: ti.template(), bound: ti.template(), rand_array: ti.template(), rcv: ti.template(), W: ti.template()):
    N = z.shape[0]  # z is 1D, so N is its length
    res = int(N**0.5)  # res is sqrt(N) for square grid
    make_rcv_rand(z, res, N, bound, rcv, W, rand_array)
    return rcv, W