"""
RCV (Receiver) calculation kernels - 1:1 CUDA port with proper Taichi structure
All kernels declared globally with template() arguments
"""
import taichi as ti
from .data_structures import BasicFlowFields


# ============================================================================
# CORE RCV KERNELS (1:1 CUDA PORT)
# ============================================================================

@ti.kernel
def make_rcv_deterministic(z: ti.template(), res: int, N: int, boundary: ti.template(), 
                          rcv: ti.template(), W: ti.template()):
    """
    1:1 port of CUDA make_rcv kernel - deterministic steepest descent
    EXACT match to CUDA logic for receiver calculation
    """
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


@ti.kernel
def make_rcv_randomized(z: ti.template(), res: int, N: int, boundary: ti.template(), 
                       rcv: ti.template(), W: ti.template(), rand_array: ti.template()):
    """
    1:1 port of CUDA make_rcv_rand kernel - randomized proportional flow
    EXACT match to CUDA logic for weighted random receiver selection
    """
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

        # Cumulative distribution for random selection
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
        W[id] = W3  # Total cumulative weight


@ti.kernel
def create_default_bounds_1d(z: ti.template(), bound: ti.template(), res: int):
    """
    1:1 port of CUDA boundary condition kernel
    Sets boundary flags for edge cells of the grid
    """
    for i in range(z.shape[0]):
        ii = i // res
        jj = i % res
        bound[i] = 1 if (ii == 0 or ii == res-1 or jj == 0 or jj == res-1) else 0


# ============================================================================
# RCV COMPUTATION INTERFACE FUNCTIONS
# ============================================================================

def run_rcv_deterministic(flow_fields: BasicFlowFields):
    """
    Run deterministic (steepest descent) receiver calculation
    Uses pre-allocated fields from BasicFlowFields data bag
    """
    N = flow_fields.N
    res = flow_fields.res
    
    make_rcv_deterministic(flow_fields.z, res, N, flow_fields.bound, 
                          flow_fields.rcv, flow_fields.W)


def run_rcv_randomized(flow_fields: BasicFlowFields):
    """
    Run randomized (proportional) receiver calculation
    Uses pre-allocated fields from BasicFlowFields data bag
    """
    N = flow_fields.N
    res = flow_fields.res
    
    make_rcv_randomized(flow_fields.z, res, N, flow_fields.bound, 
                       flow_fields.rcv, flow_fields.W, flow_fields.rand_array)


def setup_default_bounds(flow_fields: BasicFlowFields):
    """
    Set up default boundary conditions (edges are outlets)
    Uses pre-allocated fields from BasicFlowFields data bag
    """
    create_default_bounds_1d(flow_fields.z, flow_fields.bound, flow_fields.res)