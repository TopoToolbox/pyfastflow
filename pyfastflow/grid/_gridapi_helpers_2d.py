from types import SimpleNamespace
import math

import taichi as ti


def build_2d_helpers(gridctx):
    """
    Build the row/col helper family for one GridContext.

    Boundary handling is driven only by the edge mode (normal / periodic_EW /
    periodic_NS). If ``has_bcs`` is enabled on the context, nodata / outlet
    codes are folded in statically through the internal ``bcs`` field.

    The returned namespace is meant to be merged into ``gridctx.tfunc``.

    Author: B.G (02/2026)
    """
    helpers = SimpleNamespace()

    # Same capture logic as the flat helper set, but with row/col signatures.
    nx = int(gridctx.nx)
    ny = int(gridctx.ny)
    dx = float(gridctx.dx)
    boundary_mode = gridctx.boundary_mode
    d8 = gridctx.topology == "D8"
    n_neighbours = 8 if d8 else 4
    bcs = gridctx.bcs
    has_bcs = bool(gridctx.has_bcs)
    sqrt2dx = math.sqrt(2.0) * dx

    def _direction_delta(k):
        if d8:
            if k == 0:
                return -1, -1
            if k == 1:
                return -1, 0
            if k == 2:
                return -1, 1
            if k == 3:
                return 0, -1
            if k == 4:
                return 0, 1
            if k == 5:
                return 1, -1
            if k == 6:
                return 1, 0
            return 1, 1
        if k == 0:
            return -1, 0
        if k == 1:
            return 0, -1
        if k == 2:
            return 0, 1
        return 1, 0

    def _cardinal_to_k(which):
        if d8:
            if which == 0:
                return 1
            if which == 1:
                return 3
            if which == 2:
                return 4
            return 6
        return which

    @ti.func
    def _index(row: ti.i32, col: ti.i32) -> ti.i32:
        """Return flat row-major node index from row and column. Author: B.G (02/2026)"""
        return row * ti.static(nx) + col

    @ti.func
    def _in_bounds(row: ti.i32, col: ti.i32) -> ti.i32:
        """Return 1 when row and col lie inside the grid extent. Author: B.G (02/2026)"""
        return ti.cast(
            row >= 0 and row < ti.static(ny) and col >= 0 and col < ti.static(nx), ti.i32
        )

    @ti.func
    def _bc_code(row: ti.i32, col: ti.i32) -> ti.u8:
        """Return the optional internal boundary code for a node. Author: B.G (02/2026)"""
        if ti.static(has_bcs):
            return bcs[_index(row, col)]
        return ti.u8(1)

    @ti.func
    def _wrap_axis(row: ti.i32, col: ti.i32) -> ti.types.vector(2, ti.i32):
        """Apply edge-mode wrapping to a candidate neighbour coordinate. Author: B.G (02/2026)"""
        # Periodic modes wrap on one axis only. Normal mode leaves the candidate untouched.
        wrapped_row = row
        wrapped_col = col
        if ti.static(boundary_mode == "periodic_EW"):
            if wrapped_col < 0:
                wrapped_col += ti.static(nx)
            elif wrapped_col >= ti.static(nx):
                wrapped_col -= ti.static(nx)
        elif ti.static(boundary_mode == "periodic_NS"):
            if wrapped_row < 0:
                wrapped_row += ti.static(ny)
            elif wrapped_row >= ti.static(ny):
                wrapped_row -= ti.static(ny)
        return ti.Vector([wrapped_row, wrapped_col])

    @ti.func
    def _is_move_allowed(row: ti.i32, col: ti.i32, k: ti.template()) -> ti.i32:
        """Return 1 when the kth move is allowed from row, col. Author: B.G (02/2026)"""
        # This is the geometric gate: edges come from boundary_mode, nodata comes later.
        dr, dc = ti.static(_direction_delta(k))
        test_row = row + dr
        test_col = col + dc

        allowed = 1
        if ti.static(boundary_mode == "normal"):
            allowed = _in_bounds(test_row, test_col)
        elif ti.static(boundary_mode == "periodic_EW"):
            allowed = ti.cast(test_row >= 0 and test_row < ti.static(ny), ti.i32)
        else:
            allowed = ti.cast(test_col >= 0 and test_col < ti.static(nx), ti.i32)

        if ti.static(has_bcs):
            if _bc_code(row, col) == ti.u8(0):
                allowed = 0
        return allowed

    @ti.func
    def _is_target_valid(row: ti.i32, col: ti.i32) -> ti.i32:
        """Return 1 when a candidate target node can be used. Author: B.G (02/2026)"""
        #   __
        #  / _)
        # .-^^-.  Optional bcs only filters node validity; it no longer changes edge wrapping.
        #/|    |\\
        valid = _in_bounds(row, col)
        if ti.static(has_bcs):
            if valid == 1 and _bc_code(row, col) == ti.u8(0):
                valid = 0
        return valid

    @ti.func
    def is_active_2d(row: ti.i32, col: ti.i32) -> ti.i32:
        """Return 1 when the node is active under the optional internal mask. Author: B.G (02/2026)"""
        active = 1
        if ti.static(has_bcs):
            code = _bc_code(row, col)
            active = ti.cast(code == ti.u8(1) or code == ti.u8(3), ti.i32)
        return active

    @ti.func
    def nodata_2d(row: ti.i32, col: ti.i32) -> ti.i32:
        """Return 1 when the node is nodata under the optional internal mask. Author: B.G (02/2026)"""
        return 1 - is_active_2d(row, col)

    @ti.func
    def neighbour_raw_2d(row: ti.i32, col: ti.i32, k: ti.template()):
        """Return the raw kth neighbour coordinate without validity checks. Author: B.G (02/2026)"""
        # Raw neighbours are purely topological: edge mode wrapping only, no validity filtering.
        dr, dc = ti.static(_direction_delta(k))
        return _wrap_axis(row + dr, col + dc)

    @ti.func
    def neighbour_2d(row: ti.i32, col: ti.i32, k: ti.template()):
        """Return the valid kth neighbour coordinate, or [-1, -1] if blocked. Author: B.G (02/2026)"""
        # 2D helpers return coordinate vectors; invalid moves map to [-1, -1].
        invalid = ti.Vector([-1, -1])
        out = invalid
        if _is_move_allowed(row, col, k) == 1:
            candidate = neighbour_raw_2d(row, col, k)
            if _is_target_valid(candidate[0], candidate[1]) == 1:
                out = candidate
        return out

    @ti.func
    def is_on_edge_2d(row: ti.i32, col: ti.i32) -> ti.i32:
        """Return 1 when the node behaves as an edge node. Author: B.G (02/2026)"""
        # Edge meaning stays tied to the edge mode; bcs only adds local blocked-neighbour edges.
        edge = 0
        if ti.static(boundary_mode != "periodic_EW"):
            if col == 0 or col == ti.static(nx) - 1:
                edge = 1
        if ti.static(boundary_mode != "periodic_NS"):
            if row == 0 or row == ti.static(ny) - 1:
                edge = 1

        if ti.static(has_bcs):
            if edge == 0:
                for k in ti.static(range(n_neighbours)):
                    if edge == 0:
                        candidate = neighbour_2d(row, col, k)
                        if candidate[0] == -1:
                            edge = 1
        return edge

    @ti.func
    def which_edge_2d(row: ti.i32, col: ti.i32) -> ti.i32:
        """Return edge code for the node, or -1 for interior. Author: B.G (02/2026)"""
        # Compact code: 0 top, 1 left, 2 right, 3 bottom, -1 interior.
        edge = -1

        if ti.static(boundary_mode != "periodic_NS"):
            if row == 0:
                edge = 0
            elif row == ti.static(ny) - 1:
                edge = 3
        if edge == -1 and ti.static(boundary_mode != "periodic_EW"):
            if col == 0:
                edge = 1
            elif col == ti.static(nx) - 1:
                edge = 2

        if ti.static(has_bcs):
            if edge == -1:
                for which in ti.static(range(4)):
                    if edge == -1:
                        k = ti.static(_cardinal_to_k(which))
                        candidate = neighbour_2d(row, col, k)
                        if candidate[0] == -1:
                            edge = which
        return edge

    @ti.func
    def can_out_2d(row: ti.i32, col: ti.i32) -> ti.i32:
        """Return 1 when the node is an outlet under current settings. Author: B.G (02/2026)"""
        if ti.static(has_bcs):
            return ti.cast(_bc_code(row, col) == ti.u8(3), ti.i32)

        can = 0
        if ti.static(boundary_mode == "normal"):
            can = is_on_edge_2d(row, col)
        elif ti.static(boundary_mode == "periodic_EW"):
            edge = which_edge_2d(row, col)
            can = ti.cast(edge == 0 or edge == 3, ti.i32)
        else:
            edge = which_edge_2d(row, col)
            can = ti.cast(edge == 1 or edge == 2, ti.i32)
        return can

    @ti.func
    def neighbours_raw_2d(row: ti.i32, col: ti.i32):
        """Return all raw neighbours in stencil order. Author: B.G (02/2026)"""
        if ti.static(d8):
            return ti.Matrix.rows(
                [
                    neighbour_raw_2d(row, col, 0),
                    neighbour_raw_2d(row, col, 1),
                    neighbour_raw_2d(row, col, 2),
                    neighbour_raw_2d(row, col, 3),
                    neighbour_raw_2d(row, col, 4),
                    neighbour_raw_2d(row, col, 5),
                    neighbour_raw_2d(row, col, 6),
                    neighbour_raw_2d(row, col, 7),
                ]
            )
        return ti.Matrix.rows(
            [
                neighbour_raw_2d(row, col, 0),
                neighbour_raw_2d(row, col, 1),
                neighbour_raw_2d(row, col, 2),
                neighbour_raw_2d(row, col, 3),
            ]
        )

    @ti.func
    def neighbours_2d(row: ti.i32, col: ti.i32):
        """Return all checked neighbours in stencil order. Author: B.G (02/2026)"""
        if ti.static(d8):
            return ti.Matrix.rows(
                [
                    neighbour_2d(row, col, 0),
                    neighbour_2d(row, col, 1),
                    neighbour_2d(row, col, 2),
                    neighbour_2d(row, col, 3),
                    neighbour_2d(row, col, 4),
                    neighbour_2d(row, col, 5),
                    neighbour_2d(row, col, 6),
                    neighbour_2d(row, col, 7),
                ]
            )
        return ti.Matrix.rows(
            [
                neighbour_2d(row, col, 0),
                neighbour_2d(row, col, 1),
                neighbour_2d(row, col, 2),
                neighbour_2d(row, col, 3),
            ]
        )

    @ti.func
    def dist_from_k_2d(k: ti.template()) -> ti.f32:
        """Return stencil distance associated with direction k. Author: B.G (02/2026)"""
        if ti.static(d8 and k in (0, 2, 5, 7)):
            return ti.cast(ti.static(sqrt2dx), ti.f32)
        return ti.cast(ti.static(dx), ti.f32)

    @ti.func
    def dist_between_nodes_2d(
        row_a: ti.i32, col_a: ti.i32, row_b: ti.i32, col_b: ti.i32
    ) -> ti.f32:
        """Return local distance between adjacent nodes, or -1 otherwise. Author: B.G (02/2026)"""
        # Local stencil distance helper only; non-neighbour pairs return -1.
        out = ti.cast(-1.0, ti.f32)
        if row_b >= 0 and col_b >= 0:
            dr = ti.abs(row_b - row_a)
            dc = ti.abs(col_b - col_a)

            if ti.static(boundary_mode == "periodic_NS"):
                dr = ti.min(dr, ti.static(ny) - dr)
            if ti.static(boundary_mode == "periodic_EW"):
                dc = ti.min(dc, ti.static(nx) - dc)

            if dr == 0 and dc == 1:
                out = ti.cast(ti.static(dx), ti.f32)
            elif dr == 1 and dc == 0:
                out = ti.cast(ti.static(dx), ti.f32)
            elif ti.static(d8):
                if dr == 1 and dc == 1:
                    out = ti.cast(ti.static(sqrt2dx), ti.f32)
        return out

    # Public 2D helper surface exposed through gridctx.tfunc.
    helpers.can_out_2d = can_out_2d
    helpers.is_active_2d = is_active_2d
    helpers.nodata_2d = nodata_2d
    helpers.neighbour_2d = neighbour_2d
    helpers.neighbours_2d = neighbours_2d
    helpers.neighbour_raw_2d = neighbour_raw_2d
    helpers.neighbours_raw_2d = neighbours_raw_2d
    helpers.dist_from_k_2d = dist_from_k_2d
    helpers.dist_between_nodes_2d = dist_between_nodes_2d
    helpers.is_on_edge_2d = is_on_edge_2d
    helpers.which_edge_2d = which_edge_2d

    def _bind_direction(name, k):
        # Direction aliases keep handwritten kernels short and readable.
        @ti.func
        def checked(row: ti.i32, col: ti.i32):
            """Return checked neighbour for one fixed direction. Author: B.G (02/2026)"""
            return neighbour_2d(row, col, ti.static(k))

        @ti.func
        def raw(row: ti.i32, col: ti.i32):
            """Return raw neighbour for one fixed direction. Author: B.G (02/2026)"""
            return neighbour_raw_2d(row, col, ti.static(k))

        setattr(helpers, f"{name}_2d", checked)
        setattr(helpers, f"{name}_raw_2d", raw)

    if d8:
        direction_names = [
            "topleft",
            "top",
            "topright",
            "left",
            "right",
            "bottomleft",
            "bottom",
            "bottomright",
        ]
    else:
        direction_names = ["top", "left", "right", "bottom"]

    for direction_k, direction_name in enumerate(direction_names):
        _bind_direction(direction_name, direction_k)

    return helpers
