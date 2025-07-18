import taichi as ti
import numpy as np
import order
from lakeflow_algo import lakeflow
import time

# Use GPU for realistic performance
ti.init(arch=ti.gpu, debug=False)

def test_full_pipeline():
    print("Testing complete pipeline with depression routing...")
    
    res = 64  # 64x64 = 4,096 cells for manageable test
    N = res * res
    
    print(f"Grid size: {res}x{res} = {N:,} cells")
    
    # Create terrain with local minima (depressions)
    z_2d = ti.field(ti.f32, shape=(res, res))
    z_1d = ti.field(ti.f32, shape=N)
    bound_2d = ti.field(ti.i64, shape=(res, res))
    bound_1d = ti.field(ti.i64, shape=N)
    
    @ti.kernel
    def create_terrain_with_depressions():
        for i, j in ti.ndrange(res, res):
            # Base terrain slopes toward bottom-right
            base_height = 1000.0 - 3.0 * float(i + j)
            
            # Add some depressions (local minima)
            # Depression 1: around (res//4, res//4)
            dep1_x, dep1_y = res//4, res//4
            dist1 = ((i - dep1_x)**2 + (j - dep1_y)**2)**0.5
            depression1 = -50.0 * ti.exp(-dist1**2 / 16.0)
            
            # Depression 2: around (3*res//4, res//2)
            dep2_x, dep2_y = 3*res//4, res//2
            dist2 = ((i - dep2_x)**2 + (j - dep2_y)**2)**0.5
            depression2 = -30.0 * ti.exp(-dist2**2 / 9.0)
            
            # Add small random noise
            noise = (ti.random() - 0.5) * 5.0
            
            final_height = base_height + depression1 + depression2 + noise
            
            # Ensure boundary cells are lower (outlets)
            is_boundary = (i == 0 or i == res-1 or j == 0 or j == res-1)
            if is_boundary:
                final_height -= 100.0  # Make boundaries much lower
            
            z_2d[i, j] = final_height
            bound_2d[i, j] = 1 if is_boundary else 0
            
            # Convert to 1D
            idx = i * res + j
            z_1d[idx] = final_height
            bound_1d[idx] = bound_2d[i, j]
    
    print("Creating terrain with depressions...")
    create_terrain_with_depressions()
    
    # Pre-allocate all needed fields
    print("Pre-allocating fields...")
    rcv_fields = order.declare_rcv_matrix_cuda(res)
    lakeflow_fields = order.declare_lakeflow_cuda(N, N//4)
    scatter_fields = order.declare_flow_cuda_scatter_min_atomic(N, N+1)
    
    # Step 1: Basic RCV computation
    print("Step 1: Computing receivers...")
    start_time = time.time()
    rcv, W = order.rcv_matrix(z_1d, bound_1d, rcv_fields)
    rcv_time = time.time() - start_time
    print(f"  RCV computation: {rcv_time:.3f}s")
    
    # Test basic flow accumulation without depression routing
    print("Step 2: Basic flow accumulation (no depression routing)...")
    drain_basic = ti.field(ti.f32, shape=N)
    drain_basic.fill(1.0)
    
    start_time = time.time()
    drain_basic_result = order.tree_accum_upward_(rcv, W, drain_basic)
    basic_time = time.time() - start_time
    
    # Calculate basic flow statistics
    total_basic = sum(drain_basic_result[i] for i in range(N))
    boundary_basic = sum(drain_basic_result[i] for i in range(N) if bound_1d[i] == 1)
    basic_percentage = (boundary_basic / total_basic) * 100
    
    print(f"  Basic accumulation: {basic_time:.3f}s")
    print(f"  Basic boundary flow: {basic_percentage:.1f}%")
    
    # Step 3: Depression routing
    print("Step 3: Depression routing...")
    start_time = time.time()
    try:
        rcv_routed, W_routed = lakeflow(z_1d, bound_1d, rcv, W, res, method='carve',
                                       lakeflow_fields=lakeflow_fields, scatter_fields=scatter_fields)
        lakeflow_time = time.time() - start_time
        print(f"  Depression routing: {lakeflow_time:.3f}s")
        
        # Step 4: Flow accumulation with depression routing
        print("Step 4: Flow accumulation (with depression routing)...")
        drain_routed = ti.field(ti.f32, shape=N)
        drain_routed.fill(1.0)
        
        start_time = time.time()
        drain_routed_result = order.tree_accum_upward_(rcv_routed, W_routed, drain_routed)
        routed_time = time.time() - start_time
        
        # Calculate routed flow statistics
        total_routed = sum(drain_routed_result[i] for i in range(N))
        boundary_routed = sum(drain_routed_result[i] for i in range(N) if bound_1d[i] == 1)
        routed_percentage = (boundary_routed / total_routed) * 100
        
        print(f"  Routed accumulation: {routed_time:.3f}s")
        print(f"  Routed boundary flow: {routed_percentage:.1f}%")
        
        success = True
        
    except Exception as e:
        print(f"  Depression routing FAILED: {e}")
        lakeflow_time = 0
        routed_percentage = 0
        success = False
    
    # Summary
    total_time = rcv_time + basic_time + lakeflow_time + (routed_time if success else 0)
    
    print("\n" + "="*60)
    print("FULL PIPELINE RESULTS")
    print("="*60)
    print(f"Grid: {res}x{res} ({N:,} cells)")
    print(f"Total time: {total_time:.3f}s")
    print(f"  RCV computation: {rcv_time:.3f}s ({N/rcv_time/1000:.1f}K cells/sec)")
    print(f"  Basic flow: {basic_time:.3f}s")
    print(f"  Depression routing: {lakeflow_time:.3f}s" + (" (FAILED)" if not success else ""))
    print(f"Flow to boundary:")
    print(f"  Without depression routing: {basic_percentage:.1f}%")
    if success:
        print(f"  With depression routing: {routed_percentage:.1f}%")
        improvement = routed_percentage - basic_percentage
        print(f"  Improvement: +{improvement:.1f} percentage points")
    
    return {
        'success': success,
        'basic_percentage': basic_percentage,
        'routed_percentage': routed_percentage if success else 0,
        'total_time': total_time
    }

if __name__ == "__main__":
    results = test_full_pipeline()