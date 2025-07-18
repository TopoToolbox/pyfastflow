import taichi as ti
import numpy as np

@ti.kernel
def simple_flow_accumulation(rcv: ti.template(), drain: ti.template(), n: int, max_iter: int):
    """Simple iterative flow accumulation - direct 1:1 port approach"""
    for iteration in range(max_iter):
        converged = 1
        for i in range(n):
            if rcv[i] != i:  # Not a sink
                rcv_idx = rcv[i]
                old_val = drain[rcv_idx]
                new_val = old_val + drain[i]
                if ti.abs(new_val - old_val) > 1e-6:
                    converged = 0
                drain[rcv_idx] = new_val
                drain[i] = 0.0  # Reset current cell
        
        if converged == 1:
            break

@ti.kernel 
def reset_to_unit(drain: ti.template(), n: int):
    """Reset all cells to unit drainage"""
    for i in range(n):
        drain[i] = 1.0

def simple_drainage_accumulation(rcv: ti.template(), drain: ti.template()):
    """Simple drainage accumulation that should match CUDA tree_accum_upward"""
    n = rcv.shape[0]
    
    # Reset drainage to unit values
    reset_to_unit(drain, n)
    
    # Simple accumulation by following receivers
    max_iterations = 100
    for iteration in range(max_iterations):
        old_drain = ti.field(ti.f32, shape=n)
        
        # Copy current state
        for i in range(n):
            old_drain[i] = drain[i]
        
        # Accumulate
        changed = False
        for i in range(n):
            if rcv[i] != i:  # Not boundary/sink
                receiver = rcv[i]
                if drain[receiver] < drain[i] + old_drain[receiver]:
                    drain[receiver] = drain[i] + old_drain[receiver] 
                    changed = True
        
        if not changed:
            break
    
    return drain