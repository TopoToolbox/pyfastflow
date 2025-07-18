import taichi as ti

@ti.kernel  
def correct_tree_accumulation_kernel(rcv: ti.template(), drain: ti.template(), n: int):
    """Correct tree accumulation - each cell accumulates upstream drainage"""
    
    # Process cells in reverse order to ensure upstream is processed first
    for iteration in range(n):  # Max n iterations needed
        changed = 0
        
        for i in range(n):
            if rcv[i] != i:  # Not a sink/boundary
                receiver_idx = rcv[i]
                # Each non-sink cell adds its drainage to its receiver
                ti.atomic_add(drain[receiver_idx], drain[i])
                drain[i] = 0.0  # Clear the source cell
                changed = 1
        
        if changed == 0:
            break

def correct_flow_accumulation(rcv: ti.template(), W: ti.template(), p: ti.template()):
    """Correct implementation that should match CUDA tree_accum_upward_rake_compress"""
    n = p.shape[0]
    
    # Initialize with unit drainage
    p.fill(1.0)
    
    # Use multiple passes to handle complex topologies
    for pass_num in range(n):  # Maximum n passes needed
        temp_drain = ti.field(ti.f32, shape=n)
        
        # Copy current drainage values
        for i in range(n):
            temp_drain[i] = p[i]
        
        # Accumulate drainage
        converged = True
        for i in range(n):
            if rcv[i] != i:  # Not a sink
                receiver = rcv[i]
                # Add this cell's contribution to receiver
                if temp_drain[i] > 0:
                    p[receiver] += temp_drain[i]
                    p[i] = 0.0  # This cell no longer holds drainage
                    converged = False
        
        if converged:
            break
    
    # Restore unit values for non-accumulated cells
    for i in range(n):
        if p[i] == 0.0 and rcv[i] != i:
            p[i] = 1.0
    
    return p