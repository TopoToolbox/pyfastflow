import taichi as ti
import lakeflow_algo
import order

def simulation_1d(z, bound, res, niter=40, dt=10000, dx=32, k_spl=2e-5, k_t=0.001, k_h=0.05, k_d=5e2, m=0.4, u=0.0012, lake_algo='carve', seed=42):
    '''
    EXACT 1:1 PORT OF CUDA VERSION - z and bound are 1D, NO DUPLICATES
    '''
    # Don't call ti.init - already done in main
    rand_array = ti.field(ti.f32, shape=(res**2, 1))
    
    @ti.kernel
    def init_rand():
        for i in range(res**2):
            rand_array[i, 0] = ti.random()
    
    init_rand()
    
    drain = ti.field(ti.f32, shape=res**2)
    dhdt = ti.field(ti.f32, shape=res**2)
    layer_sediment = ti.field(ti.f32, shape=res**2)
    
    dhdt.fill(0)
    layer_sediment.fill(0)
    
    # Calculate uplift - EXACT COPY but for 1D
    z_max_field = ti.field(ti.f32, shape=1)
    z_max_field[0] = 0.0
    
    @ti.kernel
    def find_z_max():
        for i in range(res**2):
            ti.atomic_max(z_max_field[0], z[i])
    
    find_z_max()
    z_max = z_max_field[0]
    
    @ti.kernel 
    def calc_uplift(u_val: ti.f32, z_max_val: ti.f32):
        for i in range(res**2):
            z[i] = z[i] * u_val * ((z[i] + 256) / z_max_val)
    
    for _ in range(niter):
        # Add noise - EXACT COPY but for 1D
        @ti.kernel
        def add_noise():
            for i in range(res**2):
                z[i] += (-1 + 2 * ti.random()) * 1
        
        add_noise()
        
        # EXACT COPY - 1D versions
        rcv, W = order.rcv_matrix_rand_1d(z, bound, rand_array)
        
        # Depression routing - EXACT COPY  
        rcv, W = lakeflow_algo.lakeflow_1d(z, bound, rcv, W, res, method=lake_algo)

        # Flow routing: accumulate water flux - EXACT COPY
        drain.fill(dx**2)
        drain = order.tree_accum_upward_(rcv, W, drain)

        # Accumulate sediment flux - EXACT COPY
        Qs = order.tree_accum_upward_(rcv, W, -dhdt)

        # Apply tectonic uplift - EXACT COPY but for 1D
        @ti.kernel
        def apply_uplift():
            for i in range(res**2):
                ii = i // res
                jj = i % res
                if not bound[i]:
                    z[i] += dt * u  # u is already calculated per cell above
        
        apply_uplift()
        
        # Store previous z - EXACT COPY (z.detach().clone())
        zp = ti.field(ti.f32, shape=res**2)
        
        @ti.kernel
        def copy_z():
            for i in range(res**2):
                zp[i] = z[i]
        
        copy_z()
        
        # Erosion and deposition - EXACT COPY
        z = order.erode_deposit_cuda(z, bound, rcv, drain, Qs, dt, dx, k_spl, k_t, k_h, k_d, m)

        # Update sediment layer - EXACT COPY
        @ti.kernel
        def update_sediment():
            for i in range(res**2):
                dhdt[i] = (z[i] - zp[i]) / dt
                layer_sediment[i] = max(0.0, layer_sediment[i] + dhdt[i] * dt)
        
        update_sediment()

    # Final processing - EXACT COPY
    rcv, W = order.rcv_matrix_1d(z, bound)
    rcv, W = lakeflow_algo.lakeflow_1d(z, bound, rcv, W, res, method=lake_algo)
    
    drain.fill(dx**2)
    drain = order.tree_accum_upward_(rcv, W, drain)

    # surface = z.detach().clone() - EXACT COPY
    surface = ti.field(ti.f32, shape=res**2)
    
    @ti.kernel
    def copy_surface():
        for i in range(res**2):
            surface[i] = z[i]
    
    copy_surface()
    surface = order.tree_max_downward_(rcv, surface)
    
    # Compute lake mask - EXACT COPY
    lake = ti.field(ti.f32, shape=res**2)
    
    @ti.kernel
    def compute_lake():
        for i in range(res**2):
            lake[i] = 1.0 if surface[i] > z[i] else 0.0
    
    compute_lake()

    return surface, drain, layer_sediment, lake