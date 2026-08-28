"""
2D grid neighbouring operations for flow calculations with (i,j,k) indexing and D8 connectivity.

Provides efficient 2D grid navigation with different boundary conditions:
- Normal: flow stops at boundaries
- Periodic EW: wraps around East-West borders
- Periodic NS: wraps around North-South borders

This module uses 2D indexing (i, j) and direction k instead of flat indexing.
D8 connectivity includes both cardinal and diagonal directions:
Direction ordering: 0=topleft, 1=top, 2=topright, 3=left, 4=right, 5=bottomleft, 6=bottom, 7=bottomright

Author: B.G.
"""

import taichi as ti

from .. import constants as cte

#########################################
###### GENERAL UTILITIES ################
#########################################


@ti.func
def rc_from_i_2D_d8(i: ti.i32):
    """
    Convert vectorized index to row,col coordinates.

    Args:
            i: Vectorized grid index

    Returns:
            tuple: (row, col) coordinates

    Author: B.G.
    """
    return i // cte.NX, i % cte.NX


@ti.func
def i_from_rc_2D_d8(row: ti.i32, col: ti.i32):
    """
    Convert row,col coordinates to vectorized index.

    Args:
            row: Row coordinate
            col: Column coordinate

    Returns:
            int: Vectorized grid index

    Author: B.G.
    """
    return row * cte.NX + col


@ti.func
def is_on_edge_2D_d8(i: ti.i32):
    """
    Check if node is on grid boundary.

    Args:
            i: Vectorized grid index

    Returns:
            bool: True if node is on any edge

    Author: B.G.
    """
    res = False
    ty, tx = rc_from_i_2D_d8(i)

    if tx == 0 or tx == cte.NX - 1 or ty == 0 or ty == cte.NY - 1:
        res = True

    return res


@ti.func
def which_edge_2D_d8(i: ti.i32) -> ti.u8:
    """
    Classify edge position: 0=interior, 1-8=specific edge/corner.

    Layout: |1|2|2|2|3|
            |4|0|0|0|5|
            |4|0|0|0|5|
            |4|0|0|0|5|
            |6|7|7|7|8|
    """

    res: ti.u8 = 0

    ty, tx = rc_from_i_2D_d8(i)

    if i == 0:
        res = ti.u8(1)
    elif i < cte.NX - 1:
        res = ti.u8(2)
    elif i == cte.NX - 1:
        res = ti.u8(3)
    elif i < cte.NX * cte.NY - cte.NX:
        if tx == 0:
            res = ti.u8(4)
        elif tx == cte.NX - 1:
            res = ti.u8(5)
    elif ty == cte.NY - 1:
        if tx == 0:
            res = ti.u8(6)
        elif tx == cte.NX - 1:
            res = ti.u8(8)
        else:
            res = ti.u8(7)

    return res


@ti.kernel
def fill_edges_2D_d8(edges: ti.template()):
    for i in edges:
        edges[i] = which_edge_2D_d8(i)


#########################################
###### NORMAL BOUNDARIES ################
#########################################


# Raw neighbouring functions - no boundary checks
@ti.func
def topleft_n_2D_d8(i: ti.i32):
    """Get topleft neighbor index."""
    return i - cte.NX - 1


@ti.func
def top_n_2D_d8(i: ti.i32):
    """Get top neighbor index."""
    return i - cte.NX


@ti.func
def topright_n_2D_d8(i: ti.i32):
    """Get topright neighbor index."""
    return i - cte.NX + 1


@ti.func
def left_n_2D_d8(i: ti.i32):
    """Get left neighbor index."""
    return i - 1


@ti.func
def right_n_2D_d8(i: ti.i32):
    """Get right neighbor index."""
    return i + 1


@ti.func
def bottomleft_n_2D_d8(i: ti.i32):
    """Get bottomleft neighbor index."""
    return i + cte.NX - 1


@ti.func
def bottom_n_2D_d8(i: ti.i32):
    """Get bottom neighbor index."""
    return i + cte.NX


@ti.func
def bottomright_n_2D_d8(i: ti.i32):
    """Get bottomright neighbor index."""
    return i + cte.NX + 1


@ti.func
def validate_link_n_2D_d8(i: ti.i32, tdir: ti.template()):
    """Check if link is valid for normal boundaries (flow stops at edges)."""
    edge = which_edge_2D_d8(i)
    res = True
    if edge > 0:
        if tdir == 0:  # topleft
            if edge == 1 or edge == 2 or edge == 4 or edge == 3:  # blocked at top or left edges
                res = False
        elif tdir == 1:  # top
            if edge <= 3:  # blocked at top edge
                res = False
        elif tdir == 2:  # topright
            if edge == 1 or edge == 2 or edge == 3 or edge == 5:  # blocked at top or right edges
                res = False
        elif tdir == 3:  # left
            if edge == 1 or edge == 4 or edge == 6:  # blocked at left edge
                res = False
        elif tdir == 4:  # right
            if edge == 3 or edge == 5 or edge == 8:  # blocked at right edge
                res = False
        elif tdir == 5:  # bottomleft
            if edge == 1 or edge == 4 or edge == 6 or edge == 7:  # blocked at bottom or left edges
                res = False
        elif tdir == 6:  # bottom
            if edge >= 6:  # blocked at bottom edge
                res = False
        elif tdir == 7:  # bottomright
            if edge == 3 or edge == 5 or edge == 7 or edge == 8:  # blocked at bottom or right edges
                res = False
    return res


@ti.func
def neighbour_n_2D_d8(i: ti.i32, tdir: ti.template()):
    """Get neighbor with normal boundary validation."""
    j: ti.i32 = -1
    if tdir == 0:
        j = topleft_n_2D_d8(i)
    elif tdir == 1:
        j = top_n_2D_d8(i)
    elif tdir == 2:
        j = topright_n_2D_d8(i)
    elif tdir == 3:
        j = left_n_2D_d8(i)
    elif tdir == 4:
        j = right_n_2D_d8(i)
    elif tdir == 5:
        j = bottomleft_n_2D_d8(i)
    elif tdir == 6:
        j = bottom_n_2D_d8(i)
    else:
        j = bottomright_n_2D_d8(i)

    res = -1
    if validate_link_n_2D_d8(i, tdir):
        res = j
    else:
        res = -1
    return res


@ti.func
def can_leave_domain_n_2D_d8(i: ti.i32):
    """Check if flow can leave domain at this node."""
    return which_edge_2D_d8(i) > 0


#########################################
###### PERIODIC EW BOUNDARIES ###########
#########################################


# Raw neighbouring functions - periodic East-West wrapping
@ti.func
def topleft_pew_2D_d8(i: ti.i32):
    """Get topleft neighbor with EW wrapping."""
    row, col = rc_from_i_2D_d8(i)
    res = -1
    if col > 0:
        res = i - cte.NX - 1
    else:
        res = i - 1
    return res


@ti.func
def top_pew_2D_d8(i: ti.i32):
    """Get top neighbor index."""
    return i - cte.NX


@ti.func
def topright_pew_2D_d8(i: ti.i32):
    """Get topright neighbor with EW wrapping."""
    row, col = rc_from_i_2D_d8(i)
    res = -1
    if col < cte.NX - 1:
        res = i - cte.NX + 1
    else:
        res = i - 2 * cte.NX + 1
    return res


@ti.func
def left_pew_2D_d8(i: ti.i32):
    """Get left neighbor with EW wrapping."""
    row, col = rc_from_i_2D_d8(i)
    res = -1
    if col > 0:
        res = i - 1
    else:
        res = i + cte.NX - 1
    return res


@ti.func
def right_pew_2D_d8(i: ti.i32):
    """Get right neighbor with EW wrapping."""
    row, col = rc_from_i_2D_d8(i)
    res = -1
    if col < cte.NX - 1:
        res = i + 1
    else:
        res = i - cte.NX + 1
    return res


@ti.func
def bottomleft_pew_2D_d8(i: ti.i32):
    """Get bottomleft neighbor with EW wrapping."""
    row, col = rc_from_i_2D_d8(i)
    res = -1
    if col > 0:
        res = i + cte.NX - 1
    else:
        res = i + 2 * cte.NX - 1
    return res


@ti.func
def bottom_pew_2D_d8(i: ti.i32):
    """Get bottom neighbor index."""
    return i + cte.NX


@ti.func
def bottomright_pew_2D_d8(i: ti.i32):
    """Get bottomright neighbor with EW wrapping."""
    row, col = rc_from_i_2D_d8(i)
    res = -1
    if col < cte.NX - 1:
        res = i + cte.NX + 1
    else:
        res = i + 1
    return res


@ti.func
def validate_link_pew_2D_d8(i: ti.i32, tdir: ti.template()):
    """Check if link is valid for periodic EW boundaries (only NS blocked)."""
    edge = which_edge_2D_d8(i)
    res = True
    if edge > 0:
        if tdir == 0 or tdir == 1 or tdir == 2:  # top directions blocked at top edge
            if edge <= 3:
                res = False
        elif tdir == 5 or tdir == 6 or tdir == 7:  # bottom directions blocked at bottom edge
            if edge >= 6:
                res = False
    return res


@ti.func
def neighbour_pew_2D_d8(i: ti.i32, tdir: ti.template()):
    """Get neighbor with periodic EW boundary validation."""
    j: ti.i32 = -1
    if tdir == 0:
        j = topleft_pew_2D_d8(i)
    elif tdir == 1:
        j = top_pew_2D_d8(i)
    elif tdir == 2:
        j = topright_pew_2D_d8(i)
    elif tdir == 3:
        j = left_pew_2D_d8(i)
    elif tdir == 4:
        j = right_pew_2D_d8(i)
    elif tdir == 5:
        j = bottomleft_pew_2D_d8(i)
    elif tdir == 6:
        j = bottom_pew_2D_d8(i)
    else:
        j = bottomright_pew_2D_d8(i)

    res = -1
    if validate_link_pew_2D_d8(i, tdir):
        res = j

    return res


@ti.func
def can_leave_domain_pew_2D_d8(i: ti.i32):
    """Check if flow can leave domain (only through top/bottom edges)."""
    edge = which_edge_2D_d8(i)
    return (edge <= 3 or edge >= 6) and edge != 0


#########################################
###### PERIODIC NS BOUNDARIES ###########
#########################################


# Raw neighbouring functions - periodic North-South wrapping
@ti.func
def topleft_pns_2D_d8(i: ti.i32):
    """Get topleft neighbor with NS wrapping."""
    row, col = rc_from_i_2D_d8(i)
    res = -1
    if row > 0:
        res = i - cte.NX - 1
    else:
        res = i + cte.NX * (cte.NY - 1) - 1
    return res


@ti.func
def top_pns_2D_d8(i: ti.i32):
    """Get top neighbor with NS wrapping."""
    row, col = rc_from_i_2D_d8(i)
    res = -1
    if row > 0:
        res = i - cte.NX
    else:
        res = i + cte.NX * (cte.NY - 1)
    return res


@ti.func
def topright_pns_2D_d8(i: ti.i32):
    """Get topright neighbor with NS wrapping."""
    row, col = rc_from_i_2D_d8(i)
    res = -1
    if row > 0:
        res = i - cte.NX + 1
    else:
        res = i + cte.NX * (cte.NY - 1) + 1
    return res


@ti.func
def left_pns_2D_d8(i: ti.i32):
    """Get left neighbor index."""
    return i - 1


@ti.func
def right_pns_2D_d8(i: ti.i32):
    """Get right neighbor index."""
    return i + 1


@ti.func
def bottomleft_pns_2D_d8(i: ti.i32):
    """Get bottomleft neighbor with NS wrapping."""
    row, col = rc_from_i_2D_d8(i)
    res = -1
    if row < cte.NY - 1:
        res = i + cte.NX - 1
    else:
        res = i - cte.NX * (cte.NY - 1) - 1
    return res


@ti.func
def bottom_pns_2D_d8(i: ti.i32):
    """Get bottom neighbor with NS wrapping."""
    row, col = rc_from_i_2D_d8(i)
    res = -1
    if row < cte.NY - 1:
        res = i + cte.NX
    else:
        res = i - cte.NX * (cte.NY - 1)
    return res


@ti.func
def bottomright_pns_2D_d8(i: ti.i32):
    """Get bottomright neighbor with NS wrapping."""
    row, col = rc_from_i_2D_d8(i)
    res = -1
    if row < cte.NY - 1:
        res = i + cte.NX + 1
    else:
        res = i - cte.NX * (cte.NY - 1) + 1
    return res


@ti.func
def validate_link_pns_2D_d8(i: ti.i32, tdir: ti.template()):
    """Check if link is valid for periodic NS boundaries (only EW blocked)."""
    edge = which_edge_2D_d8(i)
    res = True
    if edge > 0:
        if tdir == 0 or tdir == 3 or tdir == 5:  # left directions blocked at left edge
            if edge == 1 or edge == 4 or edge == 6:
                res = False
        elif tdir == 2 or tdir == 4 or tdir == 7:  # right directions blocked at right edge
            if edge == 3 or edge == 5 or edge == 8:
                res = False
    return res


@ti.func
def neighbour_pns_2D_d8(i: ti.i32, tdir: ti.template()):
    """Get neighbor with periodic NS boundary validation."""
    j: ti.i32 = -1
    if tdir == 0:
        j = topleft_pns_2D_d8(i)
    elif tdir == 1:
        j = top_pns_2D_d8(i)
    elif tdir == 2:
        j = topright_pns_2D_d8(i)
    elif tdir == 3:
        j = left_pns_2D_d8(i)
    elif tdir == 4:
        j = right_pns_2D_d8(i)
    elif tdir == 5:
        j = bottomleft_pns_2D_d8(i)
    elif tdir == 6:
        j = bottom_pns_2D_d8(i)
    else:
        j = bottomright_pns_2D_d8(i)

    res = -1
    if validate_link_pns_2D_d8(i, tdir):
        res = j
    else:
        res = -1
    return res


@ti.func
def can_leave_domain_pns_2D_d8(i: ti.i32):
    """Check if flow can leave domain (only through left/right edges including corners)."""
    edge = which_edge_2D_d8(i)
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
def topleft_custom_2D_d8(i: ti.i32):
    """Get topleft neighbor with custom wrapping."""
    node = -1
    tb = cte.boundaries[i]
    if tb == 9:
        node = topleft_pns_2D_d8(i) if which_edge_2D_d8(i) <= 3 or which_edge_2D_d8(i) >= 6 else topleft_pew_2D_d8(i)
    elif tb > 0:
        node = topleft_n_2D_d8(i)

    if node > -1 and node < cte.NX * cte.NY:
        if cte.boundaries[node] == 0:
            node = -1
    return node


@ti.func
def top_custom_2D_d8(i: ti.i32):
    """Get top neighbor with custom wrapping."""
    node = -1
    tb = cte.boundaries[i]
    if tb == 9:
        node = top_pns_2D_d8(i)
    elif tb > 0:
        node = top_n_2D_d8(i)

    if node > -1 and node < cte.NX * cte.NY:
        if cte.boundaries[node] == 0:
            node = -1
    return node


@ti.func
def topright_custom_2D_d8(i: ti.i32):
    """Get topright neighbor with custom wrapping."""
    node = -1
    tb = cte.boundaries[i]
    if tb == 9:
        node = topright_pns_2D_d8(i) if which_edge_2D_d8(i) <= 3 or which_edge_2D_d8(i) >= 6 else topright_pew_2D_d8(i)
    elif tb > 0:
        node = topright_n_2D_d8(i)

    if node > -1 and node < cte.NX * cte.NY:
        if cte.boundaries[node] == 0:
            node = -1
    return node


@ti.func
def left_custom_2D_d8(i: ti.i32):
    """Get left neighbor with custom wrapping."""
    node = -1
    tb = cte.boundaries[i]
    if tb == 9:
        node = left_pew_2D_d8(i)
    elif tb > 0:
        node = left_n_2D_d8(i)
    if node > -1 and node < cte.NX * cte.NY:
        if cte.boundaries[node] == 0:
            node = -1
    return node


@ti.func
def right_custom_2D_d8(i: ti.i32):
    """Get right neighbor with custom wrapping."""
    node = -1
    tb = cte.boundaries[i]
    if tb == 9:
        node = right_pew_2D_d8(i)
    elif tb > 0:
        node = right_n_2D_d8(i)
    if node > -1 and node < cte.NX * cte.NY:
        if cte.boundaries[node] == 0:
            node = -1
    return node


@ti.func
def bottomleft_custom_2D_d8(i: ti.i32):
    """Get bottomleft neighbor with custom wrapping."""
    node = -1
    tb = cte.boundaries[i]
    if tb == 9:
        node = bottomleft_pns_2D_d8(i) if which_edge_2D_d8(i) <= 3 or which_edge_2D_d8(i) >= 6 else bottomleft_pew_2D_d8(i)
    elif tb > 0:
        node = bottomleft_n_2D_d8(i)

    if node > -1 and node < cte.NX * cte.NY:
        if cte.boundaries[node] == 0:
            node = -1

    return node


@ti.func
def bottom_custom_2D_d8(i: ti.i32):
    """Get bottom neighbor with custom wrapping."""
    node = -1
    tb = cte.boundaries[i]
    if tb == 9:
        node = bottom_pns_2D_d8(i)
    elif tb > 0:
        node = bottom_n_2D_d8(i)

    if node > -1 and node < cte.NX * cte.NY:
        if cte.boundaries[node] == 0:
            node = -1

    return node


@ti.func
def bottomright_custom_2D_d8(i: ti.i32):
    """Get bottomright neighbor with custom wrapping."""
    node = -1
    tb = cte.boundaries[i]
    if tb == 9:
        node = bottomright_pns_2D_d8(i) if which_edge_2D_d8(i) <= 3 or which_edge_2D_d8(i) >= 6 else bottomright_pew_2D_d8(i)
    elif tb > 0:
        node = bottomright_n_2D_d8(i)

    if node > -1 and node < cte.NX * cte.NY:
        if cte.boundaries[node] == 0:
            node = -1

    return node


@ti.func
def validate_link_custom_2D_d8(i: ti.i32, tdir: ti.template()):
    """Check if link is valid for custom boundaries."""
    tb = cte.boundaries[i]
    res = True
    if tb == 0:
        res = False
    elif tb != 9:
        res = validate_link_n_2D_d8(i, tdir)
    else:
        edge = which_edge_2D_d8(i)
        if edge == 0:
            res = validate_link_n_2D_d8(i, tdir)
        elif edge <= 3 or edge >= 6:
            res = validate_link_pns_2D_d8(i, tdir)
        else:
            res = validate_link_pew_2D_d8(i, tdir)
    return res


@ti.func
def neighbour_custom_2D_d8(i: ti.i32, tdir: ti.template()):
    """Get neighbor with custom boundary validation."""
    j: ti.i32 = -1
    if tdir == 0:
        j = topleft_custom_2D_d8(i)
    elif tdir == 1:
        j = top_custom_2D_d8(i)
    elif tdir == 2:
        j = topright_custom_2D_d8(i)
    elif tdir == 3:
        j = left_custom_2D_d8(i)
    elif tdir == 4:
        j = right_custom_2D_d8(i)
    elif tdir == 5:
        j = bottomleft_custom_2D_d8(i)
    elif tdir == 6:
        j = bottom_custom_2D_d8(i)
    else:
        j = bottomright_custom_2D_d8(i)

    res = -1
    if validate_link_custom_2D_d8(i, tdir) and j > -1:
        res = j
        if cte.boundaries[j] == 0:
            res = -1
    else:
        res = -1
    return res


@ti.func
def can_leave_domain_custom_2D_d8(i: ti.i32):
    """Check if flow can leave domain with custom boundaries."""
    return cte.boundaries[i] == 3


#########################################
###### EXPOSED FUNCTIONS ################
#########################################


def topleft_2D_d8(i: ti.i32):
    """Get topleft neighbor - switches between boundary modes."""
    res = -1
    if ti.static(cte.BOUND_MODE == 0):
        res = topleft_n_2D_d8(i)
    elif ti.static(cte.BOUND_MODE == 1):
        res = topleft_pew_2D_d8(i)
    elif ti.static(cte.BOUND_MODE == 2):
        res = topleft_pns_2D_d8(i)
    elif ti.static(cte.BOUND_MODE == 3):
        res = topleft_custom_2D_d8(i)
    return res


def top_2D_d8(i: ti.i32):
    """Get top neighbor - switches between boundary modes."""
    res = -1
    if ti.static(cte.BOUND_MODE == 0):
        res = top_n_2D_d8(i)
    elif ti.static(cte.BOUND_MODE == 1):
        res = top_pew_2D_d8(i)
    elif ti.static(cte.BOUND_MODE == 2):
        res = top_pns_2D_d8(i)
    elif ti.static(cte.BOUND_MODE == 3):
        res = top_custom_2D_d8(i)
    return res


def topright_2D_d8(i: ti.i32):
    """Get topright neighbor - switches between boundary modes."""
    res = -1
    if ti.static(cte.BOUND_MODE == 0):
        res = topright_n_2D_d8(i)
    elif ti.static(cte.BOUND_MODE == 1):
        res = topright_pew_2D_d8(i)
    elif ti.static(cte.BOUND_MODE == 2):
        res = topright_pns_2D_d8(i)
    elif ti.static(cte.BOUND_MODE == 3):
        res = topright_custom_2D_d8(i)
    return res


def left_2D_d8(i: ti.i32):
    """Get left neighbor - switches between boundary modes."""
    res = -1
    if ti.static(cte.BOUND_MODE == 0):
        res = left_n_2D_d8(i)
    elif ti.static(cte.BOUND_MODE == 1):
        res = left_pew_2D_d8(i)
    elif ti.static(cte.BOUND_MODE == 2):
        res = left_pns_2D_d8(i)
    elif ti.static(cte.BOUND_MODE == 3):
        res = left_custom_2D_d8(i)
    return res


def right_2D_d8(i: ti.i32):
    """Get right neighbor - switches between boundary modes."""
    res = -1
    if ti.static(cte.BOUND_MODE == 0):
        res = right_n_2D_d8(i)
    elif ti.static(cte.BOUND_MODE == 1):
        res = right_pew_2D_d8(i)
    elif ti.static(cte.BOUND_MODE == 2):
        res = right_pns_2D_d8(i)
    elif ti.static(cte.BOUND_MODE == 3):
        res = right_custom_2D_d8(i)
    return res


def bottomleft_2D_d8(i: ti.i32):
    """Get bottomleft neighbor - switches between boundary modes."""
    res = -1
    if ti.static(cte.BOUND_MODE == 0):
        res = bottomleft_n_2D_d8(i)
    elif ti.static(cte.BOUND_MODE == 1):
        res = bottomleft_pew_2D_d8(i)
    elif ti.static(cte.BOUND_MODE == 2):
        res = bottomleft_pns_2D_d8(i)
    elif ti.static(cte.BOUND_MODE == 3):
        res = bottomleft_custom_2D_d8(i)
    return res


def bottom_2D_d8(i: ti.i32):
    """Get bottom neighbor - switches between boundary modes."""
    res = -1
    if ti.static(cte.BOUND_MODE == 0):
        res = bottom_n_2D_d8(i)
    elif ti.static(cte.BOUND_MODE == 1):
        res = bottom_pew_2D_d8(i)
    elif ti.static(cte.BOUND_MODE == 2):
        res = bottom_pns_2D_d8(i)
    elif ti.static(cte.BOUND_MODE == 3):
        res = bottom_custom_2D_d8(i)
    return res


def bottomright_2D_d8(i: ti.i32):
    """Get bottomright neighbor - switches between boundary modes."""
    res = -1
    if ti.static(cte.BOUND_MODE == 0):
        res = bottomright_n_2D_d8(i)
    elif ti.static(cte.BOUND_MODE == 1):
        res = bottomright_pew_2D_d8(i)
    elif ti.static(cte.BOUND_MODE == 2):
        res = bottomright_pns_2D_d8(i)
    elif ti.static(cte.BOUND_MODE == 3):
        res = bottomright_custom_2D_d8(i)
    return res


def validate_link_2D_d8(i: ti.i32, tdir: ti.template()):
    """Validate link direction - switches between boundary modes."""
    res = -1
    if ti.static(cte.BOUND_MODE == 0):
        res = validate_link_n_2D_d8(i, tdir)
    elif ti.static(cte.BOUND_MODE == 1):
        res = validate_link_pew_2D_d8(i, tdir)
    elif ti.static(cte.BOUND_MODE == 2):
        res = validate_link_pns_2D_d8(i, tdir)
    elif ti.static(cte.BOUND_MODE == 3):
        res = validate_link_custom_2D_d8(i, tdir)
    return res


def neighbour_2D_d8(i: ti.i32, tdir: ti.template()):
    """
    Get validated neighbor - switches between boundary modes.

    Args:
            i: Vectorized grid index
            tdir: Direction template (0=topleft, 1=top, 2=topright, 3=left, 4=right, 5=bottomleft, 6=bottom, 7=bottomright)

    Returns:
            int: Neighbor index if valid, -1 if blocked by boundary

    Author: B.G.
    """
    res = -1
    if ti.static(cte.BOUND_MODE == 0):
        res = neighbour_n_2D_d8(i, tdir)
    elif ti.static(cte.BOUND_MODE == 1):
        res = neighbour_pew_2D_d8(i, tdir)
    elif ti.static(cte.BOUND_MODE == 2):
        res = neighbour_pns_2D_d8(i, tdir)
    elif ti.static(cte.BOUND_MODE == 3):
        res = neighbour_custom_2D_d8(i, tdir)

    return res


def can_leave_domain_2D_d8(i: ti.i32):
    """
    Check if flow can leave domain - switches between boundary modes.

    Args:
            i: Vectorized grid index

    Returns:
            bool: True if flow can leave domain at this node

    Author: B.G.
    """
    res = -1
    if ti.static(cte.BOUND_MODE == 0):
        res = can_leave_domain_n_2D_d8(i)
    elif ti.static(cte.BOUND_MODE == 1):
        res = can_leave_domain_pew_2D_d8(i)
    elif ti.static(cte.BOUND_MODE == 2):
        res = can_leave_domain_pns_2D_d8(i)
    elif ti.static(cte.BOUND_MODE == 3):
        res = can_leave_domain_custom_2D_d8(i)
    return res


# Main API functions - automatically switch based on cte.BOUND_MODE
def compile_neighbourer_2D_d8():
    global topleft_2D_d8, top_2D_d8, topright_2D_d8, left_2D_d8, right_2D_d8, bottomleft_2D_d8, bottom_2D_d8, bottomright_2D_d8, validate_link_2D_d8, neighbour_2D_d8, can_leave_domain_2D_d8

    topleft_2D_d8 = ti.func(topleft_2D_d8)
    top_2D_d8 = ti.func(top_2D_d8)
    topright_2D_d8 = ti.func(topright_2D_d8)
    left_2D_d8 = ti.func(left_2D_d8)
    right_2D_d8 = ti.func(right_2D_d8)
    bottomleft_2D_d8 = ti.func(bottomleft_2D_d8)
    bottom_2D_d8 = ti.func(bottom_2D_d8)
    bottomright_2D_d8 = ti.func(bottomright_2D_d8)
    validate_link_2D_d8 = ti.func(validate_link_2D_d8)
    neighbour_2D_d8 = ti.func(neighbour_2D_d8)
    can_leave_domain_2D_d8 = ti.func(can_leave_domain_2D_d8)


@ti.kernel
def flow_out_nodes_2D_d8(outnodes: ti.template()):
    for i in outnodes:
        outnodes[i] = can_leave_domain_2D_d8(i)