import taichi as ti
import math
import time

tmp    = None
change = None


@ti.kernel
def init_fill_dem(z_original: ti.template(), z_filled: ti.template(), nx: ti.i32, ny: ti.i32):
    """
    Initializes the DEM for the filling process.

    Boundary cells are set to their original elevation. Interior cells are
    initialized to a very high value, simulating a flooded landscape contained
    by its boundaries.
    """
    for i in z_original:
        row, col = i // nx, i % nx
        if row == 0 or row == ny - 1 or col == 0 or col == nx - 1:
            z_filled[i] = z_original[i]
        else:
            z_filled[i] = 1e9  # A large float for 'infinity'

@ti.kernel
def fill_step(z_original: ti.template(), z_filled: ti.template(), changes: ti.template(), nx: ti.i32, ny: ti.i32):
    """
    Performs one relaxation pass to drain the flooded landscape.

    This kernel iterates over all interior cells. It lowers each cell's elevation
    to the minimum of its neighbors, but no lower than the original terrain height.
    This process is repeated until convergence, leaving depressions filled to their
    spill point.
    """
    for row, col in ti.ndrange((1, ny - 1), (1, nx - 1)):
        i = row * nx + col
        # Find the minimum elevation among the 4 neighbors in the current filled grid
        min_neighbor_elev = min(z_filled[i+1],
                                z_filled[i-1],
                                z_filled[i+nx],
                                z_filled[i-nx])

        # The new elevation cannot be lower than the original terrain.
        new_elev = max(z_original[i], min_neighbor_elev)+1e-4

        # If this change lowers the elevation (from infinity or a previous high value),
        # update the grid and flag that a change occurred.
        if new_elev < z_filled[i]:
            z_filled[i] = new_elev
            changes[None] = 1


def fill_dem(z,  nx, ny, N_iterations=None):
    '''
    '''
    global tmp, change

    if(tmp is None):
        tmp = ti.field(ti.f32, shape=(ny*nx))
        change = ti.field(ti.u8, shape = ())
        change.fill(0)


    if(N_iterations is None):
        N_iterations = round(math.log2(nx*ny))+1

    st = time.time()
    init_fill_dem(z,tmp,nx,ny)
    for i in range(nx*ny):
        change[None] = 0
        fill_step(z, tmp, change, nx, ny)
        if change[None] == 0:
            break

    z.copy_from(tmp)

    stop = time.time()
    print(f'filled in {stop - st}')
