"""
Vectorized 2D grid neighbouring operations for flow calculations with parameters.

Provides efficient grid navigation with different boundary conditions:
- Normal: flow stops at boundaries
- Periodic EW: wraps around East-West borders
- Periodic NS: wraps around North-South borders

This module takes nx and ny as function parameters instead of using constants - more flexible, less efficient.

Author: B.G.
"""

import taichi as ti

from .. import constants as cte

#########################################
###### GENERAL UTILITIES ################
#########################################


@ti.func
def rc_from_i_param(i: ti.i32, nx: ti.i32):
    """
    Convert vectorized index to row,col coordinates.

    Args:
            i: Vectorized grid index
            nx: Grid width

    Returns:
            tuple: (row, col) coordinates

    Author: B.G.
    """
    return i // nx, i % nx


@ti.func
def i_from_rc_param(row: ti.i32, col: ti.i32, nx: ti.i32):
    """
    Convert row,col coordinates to vectorized index.

    Args:
            row: Row coordinate
            col: Column coordinate
            nx: Grid width

    Returns:
            int: Vectorized grid index

    Author: B.G.
    """
    return row * nx + col


@ti.func
def is_on_edge_param(i: ti.i32, nx: ti.i32, ny: ti.i32):
    """
    Check if node is on grid boundary.

    Args:
            i: Vectorized grid index
            nx: Grid width
            ny: Grid height

    Returns:
            bool: True if node is on any edge

    Author: B.G.
    """
    res = False
    ty, tx = rc_from_i_param(i, nx)

    if tx == 0 or tx == nx - 1 or ty == 0 or ty == ny - 1:
        res = True

    return res


@ti.func
def which_edge_param(i: ti.i32, nx: ti.i32, ny: ti.i32) -> ti.u8:
    """
    Classify edge position: 0=interior, 1-8=specific edge/corner.

    Layout: |1|2|2|2|3|
            |4|0|0|0|5|
            |4|0|0|0|5|
            |4|0|0|0|5|
            |6|7|7|7|8|
    """

    res: ti.u8 = 0

    ty, tx = rc_from_i_param(i, nx)

    if i == 0:
        res = ti.u8(1)
    elif i < nx - 1:
        res = ti.u8(2)
    elif i == nx - 1:
        res = ti.u8(3)
    elif i < nx * ny - nx:
        if tx == 0:
            res = ti.u8(4)
        elif tx == nx - 1:
            res = ti.u8(5)
    elif ty == ny - 1:
        if tx == 0:
            res = ti.u8(6)
        elif tx == nx - 1:
            res = ti.u8(8)
        else:
            res = ti.u8(7)

    return res


@ti.kernel
def fill_edges_param(edges: ti.template(), nx: ti.i32, ny: ti.i32):
    for i in edges:
        edges[i] = which_edge_param(i, nx, ny)


#########################################
###### NORMAL BOUNDARIES ################
#########################################


# Raw neighbouring functions - no boundary checks
@ti.func
def top_n_param(i: ti.i32, nx: ti.i32):
    """Get top neighbor index."""
    return i - nx


@ti.func
def left_n_param(i: ti.i32, nx: ti.i32):
    """Get left neighbor index."""
    return i - 1


@ti.func
def right_n_param(i: ti.i32, nx: ti.i32):
    """Get right neighbor index."""
    return i + 1


@ti.func
def bottom_n_param(i: ti.i32, nx: ti.i32):
    """Get bottom neighbor index."""
    return i + nx


@ti.func
def validate_link_n_param(i: ti.i32, tdir: ti.template(), nx: ti.i32, ny: ti.i32):
    """Check if link is valid for normal boundaries (flow stops at edges)."""
    edge = which_edge_param(i, nx, ny)
    res = True
    if edge > 0:
        if tdir == 0 and edge <= 3:  # Top direction blocked at top edge
            res = False
        elif tdir == 1 and (
            edge == 1 or edge == 4 or edge == 6
        ):  # Left direction blocked at left edge
            res = False
        elif tdir == 2 and (
            edge == 3 or edge == 5 or edge == 8
        ):  # Right direction blocked at right edge
            res = False
        elif tdir == 3 and (edge >= 6):  # Bottom direction blocked at bottom edge
            res = False
    return res


@ti.func
def neighbour_n_param(i: ti.i32, tdir: ti.template(), nx: ti.i32, ny: ti.i32):
    """Get neighbor with normal boundary validation."""
    j: ti.i32 = -1
    if tdir == 0:
        j = top_n_param(i, nx)
    elif tdir == 1:
        j = left_n_param(i, nx)
    elif tdir == 2:
        j = right_n_param(i, nx)
    else:
        j = bottom_n_param(i, nx)

    res = -1
    if validate_link_n_param(i, tdir, nx, ny):
        res = j
    else:
        res = -1
    return res


@ti.func
def can_leave_domain_n_param(i: ti.i32, nx: ti.i32, ny: ti.i32):
    """Check if flow can leave domain at this node."""
    return which_edge_param(i, nx, ny) > 0


#########################################
###### PERIODIC EW BOUNDARIES ###########
#########################################


# Raw neighbouring functions - periodic East-West wrapping
@ti.func
def top_pew_param(i: ti.i32, nx: ti.i32):
    """Get top neighbor index."""
    return i - nx


@ti.func
def left_pew_param(i: ti.i32, nx: ti.i32):
    """Get left neighbor with EW wrapping."""
    row, col = rc_from_i_param(i, nx)
    res = -1
    if col > 0:
        res = i - 1
    else:
        res = i + nx - 1
    return res


@ti.func
def right_pew_param(i: ti.i32, nx: ti.i32):
    """Get right neighbor with EW wrapping."""
    row, col = rc_from_i_param(i, nx)
    res = -1
    if col < nx - 1:
        res = i + 1
    else:
        res = i - nx + 1
    return res


@ti.func
def bottom_pew_param(i: ti.i32, nx: ti.i32):
    """Get bottom neighbor index."""
    return i + nx


@ti.func
def validate_link_pew_param(i: ti.i32, tdir: ti.template(), nx: ti.i32, ny: ti.i32):
    """Check if link is valid for periodic EW boundaries (only NS blocked)."""
    edge = which_edge_param(i, nx, ny)
    res = True
    if edge > 0:
        if tdir == 0 and edge <= 3:  # Top direction blocked at top edge
            res = False
        elif tdir == 3 and edge >= 6:  # Bottom direction blocked at bottom edge
            res = False
    return res


@ti.func
def neighbour_pew_param(i: ti.i32, tdir: ti.template(), nx: ti.i32, ny: ti.i32):
    """Get neighbor with periodic EW boundary validation."""
    j: ti.i32 = -1
    if tdir == 0:
        j = top_pew_param(i, nx)
    elif tdir == 1:
        j = left_pew_param(i, nx)
    elif tdir == 2:
        j = right_pew_param(i, nx)
    else:
        j = bottom_pew_param(i, nx)

    res = -1
    if validate_link_pew_param(i, tdir, nx, ny):
        res = j

    return res


@ti.func
def can_leave_domain_pew_param(i: ti.i32, nx: ti.i32, ny: ti.i32):
    """Check if flow can leave domain (only through top/bottom edges)."""
    edge = which_edge_param(i, nx, ny)
    return (edge <= 3 or edge >= 6) and edge != 0


#########################################
###### PERIODIC NS BOUNDARIES ###########
#########################################


# Raw neighbouring functions - periodic North-South wrapping
@ti.func
def top_pns_param(i: ti.i32, nx: ti.i32, ny: ti.i32):
    """Get top neighbor with NS wrapping."""
    row, col = rc_from_i_param(i, nx)
    res = -1
    if row > 0:
        res = i - nx
    else:
        res = i + nx * (ny - 1)
    return res


@ti.func
def left_pns_param(i: ti.i32, nx: ti.i32):
    """Get left neighbor index."""
    return i - 1


@ti.func
def right_pns_param(i: ti.i32, nx: ti.i32):
    """Get right neighbor index."""
    return i + 1


@ti.func
def bottom_pns_param(i: ti.i32, nx: ti.i32, ny: ti.i32):
    """Get bottom neighbor with NS wrapping."""
    row, col = rc_from_i_param(i, nx)
    res = -1
    if row < ny - 1:
        res = i + nx
    else:
        res = i - nx * (ny - 1)
    return res


@ti.func
def validate_link_pns_param(i: ti.i32, tdir: ti.template(), nx: ti.i32, ny: ti.i32):
    """Check if link is valid for periodic NS boundaries (only EW blocked)."""
    edge = which_edge_param(i, nx, ny)
    res = True
    if edge > 0:
        if tdir == 1 and (
            edge == 1 or edge == 4 or edge == 6
        ):  # Left direction blocked at left edge
            res = False
        elif tdir == 2 and (
            edge == 3 or edge == 5 or edge == 8
        ):  # Right direction blocked at right edge
            res = False
    return res


@ti.func
def neighbour_pns_param(i: ti.i32, tdir: ti.template(), nx: ti.i32, ny: ti.i32):
    """Get neighbor with periodic NS boundary validation."""
    j: ti.i32 = -1
    if tdir == 0:
        j = top_pns_param(i, nx, ny)
    elif tdir == 1:
        j = left_pns_param(i, nx)
    elif tdir == 2:
        j = right_pns_param(i, nx)
    else:
        j = bottom_pns_param(i, nx, ny)

    res = -1
    if validate_link_pns_param(i, tdir, nx, ny):
        res = j
    else:
        res = -1
    return res


@ti.func
def can_leave_domain_pns_param(i: ti.i32, nx: ti.i32, ny: ti.i32):
    """Check if flow can leave domain (only through left/right edges including corners)."""
    edge = which_edge_param(i, nx, ny)
    return edge == 1 or edge == 3 or edge == 4 or edge == 5 or edge == 6 or edge == 8


#########################################
###### CUSTOMS BOUNDARIES ###############
#########################################

# Custom = per node field of boundary codes for fine grained management
# NOTE: the boundary field is kept as a global variable in constant to keep consistent exposed interface
# Boundary codes:
## 0: No Data
## 1: normal node (cannot leave the domain)
## 3: can leave the domain
## 7: can only enter (special boundary for hydro - will act as normal node for the rest)
## 9: periodic (! risky, make sure you have the opposite direction and are on a border)


# Raw neighbouring functions - custom wrapping
@ti.func
def top_custom_param(i: ti.i32, nx: ti.i32, ny: ti.i32):
    """Get top neighbor with NS wrapping."""
    node = -1
    tb = cte.boundaries[i]
    if tb == 9:
        node = top_pns_param(i, nx, ny)
    elif tb > 0:
        node = top_n_param(i, nx)

    if node > -1 and node < nx * ny:
        if cte.boundaries[node] == 0:
            node = -1
    return node


@ti.func
def left_custom_param(i: ti.i32, nx: ti.i32, ny: ti.i32):
    """Get left neighbor index."""
    node = -1
    tb = cte.boundaries[i]
    if tb == 9:
        node = left_pew_param(i, nx)
    elif tb > 0:
        node = left_n_param(i, nx)
    if node > -1 and node < nx * ny:
        if cte.boundaries[node] == 0:
            node = -1
    return node


@ti.func
def right_custom_param(i: ti.i32, nx: ti.i32, ny: ti.i32):
    """Get right neighbor index."""
    node = -1
    tb = cte.boundaries[i]
    if tb == 9:
        node = right_pew_param(i, nx)
    elif tb > 0:
        node = right_n_param(i, nx)
    if node > -1 and node < nx * ny:
        if cte.boundaries[node] == 0:
            node = -1
    return node


@ti.func
def bottom_custom_param(i: ti.i32, nx: ti.i32, ny: ti.i32):
    """Get bottom neighbor index."""
    node = -1
    tb = cte.boundaries[i]
    if tb == 9:
        node = bottom_pns_param(i, nx, ny)
    elif tb > 0:
        node = bottom_n_param(i, nx)

    if node > -1 and node < nx * ny:
        if cte.boundaries[node] == 0:
            node = -1

    return node


@ti.func
def validate_link_custom_param(i: ti.i32, tdir: ti.template(), nx: ti.i32, ny: ti.i32):
    """Check if link is valid for periodic NS boundaries (only EW blocked)."""
    tb = cte.boundaries[i]
    res = True
    if tb == 0:
        res = False
    elif tb != 9:
        res = validate_link_n_param(i, tdir, nx, ny)
    else:
        edge = which_edge_param(i, nx, ny)
        if edge == 0:
            res = validate_link_n_param(i, tdir, nx, ny)
        elif edge <= 3 or edge >= 6:
            res = validate_link_pns_param(i, tdir, nx, ny)
        else:
            res = validate_link_pew_param(i, tdir, nx, ny)
    return res


@ti.func
def neighbour_custom_param(i: ti.i32, tdir: ti.template(), nx: ti.i32, ny: ti.i32):
    """Get neighbor with periodic NS boundary validation."""
    j: ti.i32 = -1
    if tdir == 0:
        j = top_custom_param(i, nx, ny)
    elif tdir == 1:
        j = left_custom_param(i, nx, ny)
    elif tdir == 2:
        j = right_custom_param(i, nx, ny)
    else:
        j = bottom_custom_param(i, nx, ny)

    res = -1
    if validate_link_custom_param(i, tdir, nx, ny) and j > -1:
        res = j
        if cte.boundaries[j] == 0:
            res = -1
    else:
        res = -1
    return res


@ti.func
def can_leave_domain_custom_param(i: ti.i32, nx: ti.i32, ny: ti.i32):
    """Check if flow can leave domain (only through left/right edges)."""
    return cte.boundaries[i] == 3


#########################################
###### EXPOSED FUNCTIONS ################
#########################################


def top_param(i: ti.i32, nx: ti.i32, ny: ti.i32):
    """Get top neighbor - switches between boundary modes."""
    res = -1
    if ti.static(cte.BOUND_MODE == 0):
        res = top_n_param(i, nx)
    elif ti.static(cte.BOUND_MODE == 1):
        res = top_pew_param(i, nx)
    elif ti.static(cte.BOUND_MODE == 2):
        res = top_pns_param(i, nx, ny)
    elif ti.static(cte.BOUND_MODE == 3):
        res = top_custom_param(i, nx, ny)
    return res


def left_param(i: ti.i32, nx: ti.i32, ny: ti.i32):
    """Get left neighbor - switches between boundary modes."""
    res = -1
    if ti.static(cte.BOUND_MODE == 0):
        res = left_n_param(i, nx)
    elif ti.static(cte.BOUND_MODE == 1):
        res = left_pew_param(i, nx)
    elif ti.static(cte.BOUND_MODE == 2):
        res = left_pns_param(i, nx)
    elif ti.static(cte.BOUND_MODE == 3):
        res = left_custom_param(i, nx, ny)
    return res


def right_param(i: ti.i32, nx: ti.i32, ny: ti.i32):
    """Get right neighbor - switches between boundary modes."""
    res = -1
    if ti.static(cte.BOUND_MODE == 0):
        res = right_n_param(i, nx)
    elif ti.static(cte.BOUND_MODE == 1):
        res = right_pew_param(i, nx)
    elif ti.static(cte.BOUND_MODE == 2):
        res = right_pns_param(i, nx)
    elif ti.static(cte.BOUND_MODE == 3):
        res = right_custom_param(i, nx, ny)
    return res


def bottom_param(i: ti.i32, nx: ti.i32, ny: ti.i32):
    """Get bottom neighbor - switches between boundary modes."""
    res = -1
    if ti.static(cte.BOUND_MODE == 0):
        res = bottom_n_param(i, nx)
    elif ti.static(cte.BOUND_MODE == 1):
        res = bottom_pew_param(i, nx)
    elif ti.static(cte.BOUND_MODE == 2):
        res = bottom_pns_param(i, nx, ny)
    elif ti.static(cte.BOUND_MODE == 3):
        res = bottom_custom_param(i, nx, ny)
    return res


def validate_link_param(i: ti.i32, tdir: ti.template(), nx: ti.i32, ny: ti.i32):
    """Validate link direction - switches between boundary modes."""
    res = -1
    if ti.static(cte.BOUND_MODE == 0):
        res = validate_link_n_param(i, tdir, nx, ny)
    elif ti.static(cte.BOUND_MODE == 1):
        res = validate_link_pew_param(i, tdir, nx, ny)
    elif ti.static(cte.BOUND_MODE == 2):
        res = validate_link_pns_param(i, tdir, nx, ny)
    elif ti.static(cte.BOUND_MODE == 3):
        res = validate_link_custom_param(i, tdir, nx, ny)
    return res


def neighbour_param(i: ti.i32, tdir: ti.template(), nx: ti.i32, ny: ti.i32):
    """
    Get validated neighbor - switches between boundary modes.

    Args:
            i: Vectorized grid index
            tdir: Direction template (0=top, 1=left, 2=right, 3=bottom)
            nx: Grid width
            ny: Grid height

    Returns:
            int: Neighbor index if valid, -1 if blocked by boundary

    Author: B.G.
    """
    res = -1
    if ti.static(cte.BOUND_MODE == 0):
        res = neighbour_n_param(i, tdir, nx, ny)
    elif ti.static(cte.BOUND_MODE == 1):
        res = neighbour_pew_param(i, tdir, nx, ny)
    elif ti.static(cte.BOUND_MODE == 2):
        res = neighbour_pns_param(i, tdir, nx, ny)
    elif ti.static(cte.BOUND_MODE == 3):
        res = neighbour_custom_param(i, tdir, nx, ny)

    return res


def can_leave_domain_param(i: ti.i32, nx: ti.i32, ny: ti.i32):
    """
    Check if flow can leave domain - switches between boundary modes.

    Args:
            i: Vectorized grid index
            nx: Grid width
            ny: Grid height

    Returns:
            bool: True if flow can leave domain at this node

    Author: B.G.
    """
    res = -1
    if ti.static(cte.BOUND_MODE == 0):
        res = can_leave_domain_n_param(i, nx, ny)
    elif ti.static(cte.BOUND_MODE == 1):
        res = can_leave_domain_pew_param(i, nx, ny)
    elif ti.static(cte.BOUND_MODE == 2):
        res = can_leave_domain_pns_param(i, nx, ny)
    elif ti.static(cte.BOUND_MODE == 3):
        res = can_leave_domain_custom_param(i, nx, ny)
    return res


# Main API functions - automatically switch based on cte.BOUND_MODE
def compile_neighbourer_param():
    global top_param, left_param, right_param, bottom_param, validate_link_param, neighbour_param, can_leave_domain_param

    top_param = ti.func(top_param)
    left_param = ti.func(left_param)
    right_param = ti.func(right_param)
    bottom_param = ti.func(bottom_param)
    validate_link_param = ti.func(validate_link_param)
    neighbour_param = ti.func(neighbour_param)
    can_leave_domain_param = ti.func(can_leave_domain_param)


@ti.kernel
def flow_out_nodes_param(outnodes: ti.template(), nx: ti.i32, ny: ti.i32):
    for i in outnodes:
        outnodes[i] = can_leave_domain_param(i, nx, ny)