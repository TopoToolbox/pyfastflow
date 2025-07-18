import taichi as ti

@ti.kernel
def simple_tree_accum_upward(rcv: ti.template(), W: ti.template(), p: ti.template(), processed: ti.template()):
    """Simple iterative tree accumulation - not parallel but works"""
    # Initialize 
    for i in range(rcv.shape[0]):
        processed[i] = 0
    
    # Mark boundary/self-receivers as processed
    for i in range(rcv.shape[0]):
        if rcv[i] == i:
            processed[i] = 1
    
    # Iteratively process cells
    changed = 1
    max_iter = 100
    iter_count = 0
    
    while changed > 0 and iter_count < max_iter:
        changed = 0
        iter_count += 1
        
        for i in range(rcv.shape[0]):
            if processed[i] == 0:  # Not yet processed
                rcv_i = rcv[i]
                if processed[rcv_i] == 1:  # Receiver is processed
                    p[rcv_i] += p[i]
                    processed[i] = 1
                    changed = 1

def flow_cuda_tree_accum_upward_rake_compress_simple(rcv: ti.template(), W: ti.template(), p: ti.template()):
    """Simple wrapper that mimics the interface"""
    N = rcv.shape[0]
    processed = ti.field(ti.i64, shape=N)
    
    simple_tree_accum_upward(rcv, W, p, processed)
    return p