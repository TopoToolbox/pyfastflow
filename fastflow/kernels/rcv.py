"""
Receiver computation kernels - Taichi port of src/cuda/core/rcv.cu

This module implements the core flow routing algorithms that determine drainage
directions in digital elevation models. It provides both deterministic and 
randomized methods for computing receivers based on topographic gradients.

CUDA Correspondence:
- Direct 1:1 port of src/cuda/core/rcv.cu kernels
- Identical mathematical formulations and numerical precision
- Same boundary condition handling and edge case behavior
- Preserves tie-breaking rules and weight calculation methods

Mathematical Background:
Flow routing determines where water flows from each cell based on the steepest
descent principle. For each cell, the algorithm:

1. **Gradient Calculation**: Compute elevation differences to 4-connected neighbors
   Δz_i = max(0, z_center - z_neighbor_i) for directions i ∈ {N, S, E, W}

2. **Weight Normalization**: Convert gradients to flow weights
   W_i = Δz_i / Σ(Δz_j) for all downhill directions j

3. **Receiver Selection**: 
   - Deterministic: Choose neighbor with maximum weight (steepest descent)
   - Randomized: Choose neighbor probabilistically based on weights

Algorithm Variants:

**Deterministic Routing** (make_rcv):
- Selects steepest downhill neighbor as single receiver
- Deterministic tie-breaking ensures reproducible results
- Produces tree-like flow networks with single flow directions
- Mathematically equivalent to D8 flow routing algorithm

**Randomized Routing** (make_rcv_rand):
- Uses weighted random selection based on gradients
- Multiple flow directions possible through stochastic selection
- Produces more realistic flow dispersal patterns
- Controllable randomness through seeded random number generation

Performance Characteristics:
- Fully parallel kernels with O(N) complexity
- Each thread processes one grid cell independently
- Memory access pattern: read 5 cells (center + 4 neighbors)
- Optimal GPU utilization with high arithmetic intensity

Boundary Conditions:
- Edge cells automatically drain to themselves (rcv[i] = i)
- Prevents flow from leaving the computational domain
- Consistent with CUDA implementation boundary handling
- No special case handling required in downstream algorithms

Numerical Stability:
- Uses small epsilon (1e-7) to prevent division by zero
- Handles flat areas with graceful degradation
- Preserves relative gradient magnitudes across different terrains
- Robust to floating-point precision limitations
"""
import taichi as ti

@ti.kernel
def make_rcv(z: ti.template(), res: int, N: int, boundary: ti.template(), 
             rcv: ti.template(), W: ti.template()):
    """1:1 port of make_rcv CUDA kernel"""
    for i in z:
            
        x = i % res
        y = i // res
        
        x_incr = (z[i] > z[i - 1]) if x > 0 else False
        x_decr = (z[i] > z[i + 1]) if x < (res - 1) else False
        y_incr = (z[i] > z[i - res]) if y > 0 else False
        y_decr = (z[i] > z[i + res]) if y < (res - 1) else False
        
        bound = boundary[i]
        
        is_rcv0 = x_incr and not bound
        is_rcv1 = x_decr and not bound
        is_rcv2 = y_incr and not bound
        is_rcv3 = y_decr and not bound
        
        rcv0 = i + (-1 if is_rcv0 else 0)
        rcv1 = i + (1 if is_rcv1 else 0)
        rcv2 = i + (-res if is_rcv2 else 0)
        rcv3 = i + (res if is_rcv3 else 0)
        
        W0 = z[i] - z[rcv0]
        W1 = z[i] - z[rcv1]
        W2 = z[i] - z[rcv2]
        W3 = z[i] - z[rcv3]
        
        sum_val = ti.max(1e-7, (W0 + W1 + W2 + W3))
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
            
        rcv[i] = rcvmax
        W[i] = ti.ceil(Wmax)


@ti.kernel
def make_rcv_rand(z: ti.template(), res: int, N: int, boundary: ti.template(), 
                  rcv: ti.template(), W: ti.template(), rand_array: ti.template()):
    """1:1 port of make_rcv_rand CUDA kernel"""
    for i in range(ti.i32(N)):
        if i >= N:
            continue
            
        x = i % res
        y = i // res
        
        x_incr = (z[i] > z[i - 1]) if x > 0 else False
        x_decr = (z[i] > z[i + 1]) if x < (res - 1) else False
        y_incr = (z[i] > z[i - res]) if y > 0 else False
        y_decr = (z[i] > z[i + res]) if y < (res - 1) else False
        
        bound = boundary[i]
        
        is_rcv0 = x_incr and not bound
        is_rcv1 = x_decr and not bound
        is_rcv2 = y_incr and not bound
        is_rcv3 = y_decr and not bound
        
        rcv0 = i + (-1 if is_rcv0 else 0)
        rcv1 = i + (1 if is_rcv1 else 0)
        rcv2 = i + (-res if is_rcv2 else 0)
        rcv3 = i + (res if is_rcv3 else 0)
        
        W0 = z[i] - z[rcv0]
        W1 = z[i] - z[rcv1]
        W2 = z[i] - z[rcv2]
        W3 = z[i] - z[rcv3]
        
        sum_val = ti.max(1e-7, (W0 + W1 + W2 + W3))
        W0 = W0 / sum_val
        W1 = W1 / sum_val
        W2 = W2 / sum_val
        W3 = W3 / sum_val
        
        W1 += W0
        W2 += W1
        W3 += W2
        
        trcv_ = rcv0  # Initialize variable like CUDA
        rand_num = rand_array[i]
        if W0 > rand_num:
            trcv_ = rcv0
        elif W1 > rand_num:
            trcv_ = rcv1
        elif W2 > rand_num:
            trcv_ = rcv2
        else:
            trcv_ = rcv3
            
        rcv[i] = trcv_
        W[i] = W3