import taichi as ti
import lakeflow_algo
import order
import simulation
import matplotlib.pyplot as plt

ti.init(arch=ti.gpu, debug=True)

# Initialize everything as 1D from start - NO DUPLICATES
res = 64
z = ti.field(ti.f32, shape=res*res)
bound = ti.field(ti.i64, shape=res*res)

@ti.kernel
def initialize_terrain_1d():
    """Initialize terrain with random perturbations"""
    for i in range(res*res):
        z[i] = (1.0 + 0.01 * ti.random()) * 1e3

def test_depression(z, bound):
    '''
    Demonstrates the depression routing process.
    '''
    rcv, W = order.rcv_matrix_1d(z, bound)
    # Enable lakeflow for depression routing 
    rcv, W = lakeflow_algo.lakeflow_1d(z, bound, rcv, W, res, method='carve')
    return rcv, W


def test_flow(z, rcv, W):
    '''
    Demonstrates the flow routing process.
    '''
    drain = ti.field(ti.f32, shape=res*res)
    drain.fill(1.0)
    drain = order.tree_accum_upward_(rcv, W, drain)
    return drain


@ti.kernel
def compute_boundary_drain_sum(drain: ti.template(), res_val: ti.i64) -> ti.f32:
    '''
    Computes sum of drainage at boundaries
    '''
    total = 0.0
    
    # Top and bottom rows
    for j in range(res_val):
        total += drain[0 * res_val + j] + drain[(res_val-1) * res_val + j]
    
    # Left and right columns (excluding corners to avoid double counting)
    for i in range(1, res_val-1):
        total += drain[i * res_val + 0] + drain[i * res_val + (res_val-1)]
    
    return total


def test_drain_at_bound(z, drain):
    '''
    Checks if 100% of the discharge flows out of the boundary.
    '''
    N = z.shape[0]
    res_val = int(N**0.5)  # Calculate res from 1D array size
    boundary_sum = compute_boundary_drain_sum(drain, res_val)
    total_cells = N
    drain_at_bound = boundary_sum / total_cells * 100
    return drain_at_bound


def test_simulation(z, bound):
    '''
    Runs the simulation.
    '''
    return simulation.simulation_1d(z, bound, res, dt=4e4)


@ti.kernel
def initialize_terrain(z: ti.template()):
    """Initialize terrain with random perturbations"""
    for i, j in ti.ndrange(z.shape[0], z.shape[1]):
        z[i, j] = (1.0 + 0.01 * ti.random()) * 1e3


# Initialize with random perturbations  
initialize_terrain_1d()
bound = order.default_bounds_taichi_1d(z, res)

# Debug field shapes
print(f"z.shape = {z.shape}")
print(f"bound.shape = {bound.shape}")

# Perform depression routing to obtain receivers and weights, ensuring all flow paths terminate at the boundaries.
rcv, W = test_depression(z, bound)

# Debug RCV values for interior cells
interior_start = 65  # Second row, second column (should be interior)
rcv_sample = ti.field(ti.i64, shape=10)
bound_sample = ti.field(ti.i64, shape=10)
z_sample = ti.field(ti.f32, shape=10)

@ti.kernel
def sample_interior():
    for i in range(10):
        idx = interior_start + i
        rcv_sample[i] = rcv[idx]
        bound_sample[i] = bound[idx]
        z_sample[i] = z[idx]

sample_interior()
print("Interior RCV values:", [rcv_sample[i] for i in range(10)])
print("Interior bound values:", [bound_sample[i] for i in range(10)])
print("Interior Z values:", [z_sample[i] for i in range(10)])

# Check if all cells are being marked as boundaries
boundary_count = ti.field(ti.i64, shape=1)
@ti.kernel
def count_boundaries():
    boundary_count[0] = 0
    for i in range(bound.shape[0]):
        if bound[i] == 1:
            ti.atomic_add(boundary_count[0], 1)

count_boundaries()
print(f"Total boundary cells: {boundary_count[0]} out of {bound.shape[0]}")

# Calculate the water discharge across the terrain based on the flow routing.
drain = test_flow(z, rcv, W)

# Verify that the discharge at the boundary is 100% (all water exits the terrain).
print('Discharge leaving bounds = ', test_drain_at_bound(z, drain), '%')

# Reshape results to 2D for visualization (ONLY PLACE TO DO RESHAPING)
@ti.kernel
def convert_1d_to_2d_drain(drain_1d: ti.template(), drain_2d: ti.template(), res: int):
    for i, j in ti.ndrange(res, res):
        drain_2d[i, j] = drain_1d[i * res + j]

@ti.kernel  
def convert_1d_to_2d_z(z_1d: ti.template(), z_2d: ti.template(), res: int):
    for i, j in ti.ndrange(res, res):
        z_2d[i, j] = z_1d[i * res + j]

# Create 2D versions for visualization
z_2d = ti.field(ti.f32, shape=(res, res))
drain_2d = ti.field(ti.f32, shape=(res, res))

convert_1d_to_2d_z(z, z_2d, res)
convert_1d_to_2d_drain(drain, drain_2d, res)
ti.sync()

# plt.imshow(drain_2d.to_numpy())
# plt.show()
print("Basic RCV and flow routing working!")
print(f"Final terrain shape: {z_2d.shape}")
print(f"Final drain shape: {drain_2d.shape}")