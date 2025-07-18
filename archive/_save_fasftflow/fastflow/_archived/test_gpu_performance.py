import taichi as ti
import numpy as np
import order
import time

# Switch to GPU for performance testing
ti.init(arch=ti.gpu, debug=False)

def test_performance():
    print("Testing GPU performance with large grid...")
    
    # Large grid for performance testing
    res = 128  # 128x128 = 16,384 cells
    N = res * res
    
    print(f"Grid size: {res}x{res} = {N:,} cells")
    
    # Create large fields
    z_1d = ti.field(ti.f32, shape=N)
    bound_1d = ti.field(ti.i64, shape=N)
    
    @ti.kernel
    def init_large_terrain():
        for i in range(N):
            x = i % res
            y = i // res
            # Create realistic terrain with noise
            base_height = 1000.0 - 2.0 * float(x + y)  # General slope
            noise = (ti.random() - 0.5) * 100.0  # Random variations
            z_1d[i] = base_height + noise
            
            # Boundary conditions
            bound_1d[i] = 1 if (x == 0 or x == res-1 or y == 0 or y == res-1) else 0
    
    print("Initializing terrain...")
    start_time = time.time()
    init_large_terrain()
    init_time = time.time() - start_time
    print(f"Terrain initialization: {init_time:.3f} seconds")
    
    # Test RCV computation performance
    print("Testing RCV computation...")
    rcv_fields = order.declare_rcv_matrix_cuda(res)
    
    start_time = time.time()
    rcv, W = order.rcv_matrix(z_1d, bound_1d, rcv_fields)
    rcv_time = time.time() - start_time
    print(f"RCV computation: {rcv_time:.3f} seconds ({N/rcv_time/1000:.1f}K cells/sec)")
    
    # Test tree accumulation performance
    print("Testing tree accumulation...")
    drain = ti.field(ti.f32, shape=N)
    drain.fill(1.0)
    
    start_time = time.time()
    drain_result = order.tree_accum_upward_(rcv, W, drain)
    tree_time = time.time() - start_time
    print(f"Tree accumulation: {tree_time:.3f} seconds ({N/tree_time/1000:.1f}K cells/sec)")
    
    # Check results
    total_flow = sum(drain_result[i] for i in range(N))
    max_flow = max(drain_result[i] for i in range(N))
    
    print(f"Total flow: {total_flow:.1f}")
    print(f"Max flow: {max_flow:.1f}")
    print(f"Average flow: {total_flow/N:.2f}")
    
    # Check boundary flow
    boundary_flow = 0.0
    for i in range(N):
        if bound_1d[i] == 1:
            boundary_flow += drain_result[i]
    
    percentage = (boundary_flow / total_flow) * 100
    print(f"Boundary flow: {boundary_flow:.1f} ({percentage:.1f}%)")
    
    return {
        'res': res,
        'N': N,
        'rcv_time': rcv_time,
        'tree_time': tree_time,
        'total_flow': total_flow,
        'boundary_percentage': percentage
    }

if __name__ == "__main__":
    results = test_performance()
    print("\n" + "="*50)
    print("PERFORMANCE SUMMARY")
    print("="*50)
    print(f"Grid: {results['res']}x{results['res']} ({results['N']:,} cells)")
    print(f"RCV: {results['rcv_time']:.3f}s ({results['N']/results['rcv_time']/1000:.1f}K cells/sec)")
    print(f"Tree: {results['tree_time']:.3f}s ({results['N']/results['tree_time']/1000:.1f}K cells/sec)")
    print(f"Flow routing: {results['boundary_percentage']:.1f}% reaches boundary")