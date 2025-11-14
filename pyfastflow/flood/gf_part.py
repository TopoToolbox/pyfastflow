import taichi as ti
import pyfastflow.flow as flow
import pyfastflow.grid as grid
from .. import constants as cte

# ---------------- Particle structure ----------------
Particle = ti.types.struct(pos=ti.i32, hw=cte.FLOAT_TYPE_TI)

# ---------------- Helpers ----------------
@ti.func
def is_valid_index(i: ti.i32) -> ti.u1:
    return (i != -1) and (not flow.neighbourer_flat.can_leave_domain(i)) and (not flow.neighbourer_flat.nodata(i))

@ti.func
def rand_i() -> ti.i32:
    ix = ti.min(ti.i32(ti.random() * cte.NX), cte.NX - 1)
    iy = ti.min(ti.i32(ti.random() * cte.NY), cte.NY - 1)
    return iy * cte.NX + ix

@ti.func
def steepest_next_z(i: ti.i32, z: ti.template(), stoch_exp: cte.FLOAT_TYPE_TI) -> ti.i32:
    best_j = -1
    best_w = 0.0
    vi = z[i]
    for k in range(4):
        j = flow.neighbourer_flat.neighbour(i, k)
        if not is_valid_index(j):
            continue
        vj = z[j]
        if vj < vi:
            slope = (vi - vj) / cte.DX
            coeff = 1.0
            if stoch_exp > 0.0:
                coeff = ti.pow(ti.random(), stoch_exp)
            w = slope * coeff
            if w > best_w:
                best_w = w
                best_j = j
    return best_j

@ti.func
def steepest_next_zh(i: ti.i32, z: ti.template(), h: ti.template(), stoch_exp: cte.FLOAT_TYPE_TI) -> ti.i32:
    best_j = -1
    best_w = 0.0
    vi = z[i] + h[i]
    for k in range(4):
        j = flow.neighbourer_flat.neighbour(i, k)
        if not is_valid_index(j):
            continue
        vj = z[j] + h[j]
        if vj < vi:
            slope = (vi - vj) / cte.DX
            coeff = 1.0
            if stoch_exp > 0.0:
                coeff = ti.pow(ti.random(), stoch_exp)
            w = slope * coeff
            if w > best_w:
                best_w = w
                best_j = j
    return best_j

# ---------------- API ----------------
@ti.func
def spawn(part: ti.template()):
    i = rand_i()
    while not is_valid_index(i):
        i = rand_i()
    part.pos = i

@ti.func
def move_on_z(part: ti.template(), stoch_exp: cte.FLOAT_TYPE_TI, z: ti.template()) -> ti.u1:
    if not is_valid_index(part.pos):
        return False
    jn = steepest_next_z(part.pos, z, stoch_exp)
    moved = False
    if jn != -1:
        part.pos = jn
        moved = True
    return moved

@ti.func
def move_on_zh(part: ti.template(), stoch_exp: cte.FLOAT_TYPE_TI, z: ti.template(), h: ti.template()) -> ti.u1:
    if not is_valid_index(part.pos):
        return False
    jn = steepest_next_zh(part.pos, z, h, stoch_exp)
    moved = False
    if jn != -1:
        part.pos = jn
        moved = True
    return moved

@ti.func
def steepest_next_zh_d8(i: ti.i32, z: ti.template(), h: ti.template(), stoch_exp: cte.FLOAT_TYPE_TI, use_d8: ti.u1) -> ti.i32:
    """Find steepest downhill neighbor on z+h surface with D4 or D8 connectivity."""
    best_j = -1
    best_w = 0.0
    vi = z[i] + h[i]
    n_neighbors = 8 if use_d8 else 4
    for k in range(8):
        if k >= n_neighbors:
            break
        j = flow.neighbourer_flat.neighbour(i, k)
        if not is_valid_index(j):
            continue
        vj = z[j] + h[j]
        if vj < vi:
            slope = (vi - vj) / cte.DX
            coeff = 1.0
            if stoch_exp > 0.0:
                coeff = ti.pow(ti.random(), stoch_exp)
            w = slope * coeff
            if w > best_w:
                best_w = w
                best_j = j
    return best_j

@ti.func
def find_lowest_neighbor_zh(i: ti.i32, z: ti.template(), h: ti.template(), use_d8: ti.u1) -> cte.FLOAT_TYPE_TI:
    """Find the lowest z+h value among neighbors."""
    lowest = -1.  # Start with current cell
    n_neighbors = 8 if use_d8 else 4
    for k in range(8):
        if k >= n_neighbors:
            break
        j = flow.neighbourer_flat.neighbour(i, k)
        if not is_valid_index(j):
            continue
        vj = z[j] + h[j]
        if vj < lowest or lowest == -1.:
            lowest = vj
    return lowest

@ti.func
def is_local_minima_zh(i: ti.i32, z: ti.template(), h: ti.template(), use_d8: ti.u1) -> ti.u1:
    """Check if current cell is a local minima on z+h surface."""
    vi = z[i] + h[i]
    n_neighbors = 8 if use_d8 else 4
    ret = True
    for k in range(8):
        if k >= n_neighbors:
            break
        j = flow.neighbourer_flat.neighbour(i, k)
        if not is_valid_index(j):
            continue
        vj = z[j] + h[j]
        if vj < vi:
            ret = False  # Found lower neighbor, not a minima
            break
    return ret  # All neighbors are higher or equal




@ti.kernel
def init_particles(particles: ti.template()):
    """
    Initialize all particles by spawning them at random valid locations.

    Args:
        particles: Particle field to initialize
    """
    for p_idx in particles:
        spawn(particles[p_idx])


@ti.kernel
def move_particles_with_collision(
    particles: ti.template(),
    z: ti.template(),
    h: ti.template(),
    occupancy: ti.template(),  # ti.u8 field for collision detection
    stoch_exp: cte.FLOAT_TYPE_TI,
    use_d8: ti.u1
):
    """
    Move particles on z+h surface with collision detection and local minima handling.

    Algorithm:
    - Each particle moves downhill on z+h surface
    - If in local minima, raise h to escape
    - Particles cannot share cells (collision detection via atomic occupancy field)
    - Particles respawn if they reach domain boundaries
    - Ensures at least one move per particle

    Args:
        particles: Particle field with pos and hw attributes
        z: Elevation field
        h: Water depth field
        occupancy: ti.u8 field for tracking cell occupation (same shape as z)
        stoch_exp: Stochasticity exponent for movement (0.0 = deterministic)
        use_d8: If True use D8 connectivity, else D4
    """
    # Reset occupancy field at start
    for idx in occupancy:
        occupancy[idx] = ti.u8(0)

    # Move each particle
    for p_idx in particles:
        current_pos = particles[p_idx].pos

        moved_at_least_once = False

        # Try to move up to 100 times
        for step in range(1000):

            # Check if next position is boundary or nodata
            if flow.neighbourer_flat.can_leave_domain(current_pos) or flow.neighbourer_flat.nodata(current_pos):
                # Respawn and continue
                spawn(particles[p_idx])
                current_pos = particles[p_idx].pos
                old_count = ti.atomic_add(occupancy[current_pos], ti.u8(1))

                if(old_count == 0):
                    break
                    
                continue

            # Check if in local minima
            if is_local_minima_zh(current_pos, z, h, use_d8):
                # Raise h to escape minima
                lowest_neighbor = find_lowest_neighbor_zh(current_pos, z, h, use_d8)
                current_zh = z[current_pos] + h[current_pos]
            
                # Raise h atomically to lowest_neighbor + 5e-3
                target_h = lowest_neighbor + 5e-3 - z[current_pos]
                ti.atomic_max(h[current_pos], target_h)

            # Find next cell
            next_pos = steepest_next_zh_d8(current_pos, z, h, stoch_exp, use_d8)

            if next_pos == -1:
                spawn(particles[p_idx])
                current_pos = particles[p_idx].pos
                continue


            # Move successful - no collision
            current_pos = next_pos
            moved_at_least_once = True
        

            # Ensure particle is in valid location to start
            if not is_valid_index(next_pos):
                spawn(particles[p_idx])
                current_pos = particles[p_idx].pos
                continue

            # Try to occupy next cell
            old_count = ti.atomic_add(occupancy[current_pos], ti.u8(1))

            if old_count == 0:
                break
            else:
                spawn(particles[p_idx])
                current_pos = particles[p_idx].pos
                
                old_count = ti.atomic_add(occupancy[current_pos], ti.u8(1))

                if(old_count == 0):
                    break

        # Update particle position
        particles[p_idx].pos = current_pos




@ti.kernel
def burn2pos(particles:ti.template(), npart:ti.types.ndarray(dtype=ti.i32, ndim=2)):

    for i,j in npart:
        npart[i,j] = 0

    for i in particles:
        r,c = grid.neighbourer_flat.rc_from_i(particles[i].pos)
        npart[r,c] += 1