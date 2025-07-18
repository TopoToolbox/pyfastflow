import taichi as ti

def direct_flow_accumulation(rcv, W, p):
    """Direct flow accumulation that matches CUDA exactly"""
    n = p.shape[0]
    
    # Reset to unit values
    p.fill(1.0)
    
    # For each cell, find all cells that flow to it and sum them up
    result = ti.field(ti.f32, shape=n)
    result.fill(0.0)
    
    # Count how many cells flow to each cell
    for i in range(n):
        result[i] = 1.0  # Each cell contributes at least itself
        
        # Count upstream contributions  
        for j in range(n):
            if j != i and flows_to(j, i, rcv, n):
                result[i] += 1.0
    
    # Copy result back to p
    for i in range(n):
        p[i] = result[i]
    
    return p

def flows_to(source, target, rcv, n):
    """Check if source eventually flows to target"""
    current = source
    steps = 0
    
    while steps < n and current != target:
        if rcv[current] == current:  # Hit a sink
            return False
        current = rcv[current]
        steps += 1
    
    return current == target and steps < n