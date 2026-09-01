import taichi as ti

from .. import constants as cte


@ti.kernel
def tectonic_uplift_kernel(z: ti.template()):
    """
    Apply ``dt * uplift_rate`` to all non-outlet nodes.

    Author: B.G (02/2026)
    """
    for i in z:
        if not gridctx.tfunc.can_out_flat(i):
            z[i] += get_dt(i) * get_uplift_rate(i)


@ti.kernel
def uplift_baselevel_kernel(z: ti.template()):
    """
    Apply ``dt * uplift_rate`` only to outlet nodes.

    Author: B.G (02/2026)
    """
    for i in z:
        if gridctx.tfunc.can_out_flat(i):
            z[i] += get_dt(i) * get_uplift_rate(i)


@ti.kernel
def init_erode_spl_kernel(
    z: ti.template(),
    z_work: ti.template(),
    z_aux: ti.template(),
    alpha: ti.template(),
    alpha_aux: ti.template(),
    area: ti.template(),
    receivers: ti.template(),
):
    """
    Initialize the implicit SPL sweep buffers.

    The current cleaned implicit form matches the linear stream-power update
    used in the legacy code path and therefore assumes the slope exponent is
    effectively one.

    Author: B.G (02/2026)
    """
    for i in z:
        z_work[i] = z[i]
        z_aux[i] = z[i]
        alpha[i] = ti.cast(0.0, cte.FLOAT_TYPE_TI)
        alpha_aux[i] = ti.cast(0.0, cte.FLOAT_TYPE_TI)

        rec = receivers[i]
        if rec == i or gridctx.tfunc.can_out_flat(i):
            continue

        dist = flowctx.tfunc.dist_between_nodes_corrected(i, rec)
        if dist <= ti.cast(0.0, cte.FLOAT_TYPE_TI):
            continue

        coeff = (
            get_K_bedrock(i)
            * ti.pow(area[i], get_m_exp(i))
            * get_dt(i)
            / dist
        )
        alpha[i] = coeff / (ti.cast(1.0, cte.FLOAT_TYPE_TI) + coeff)
        z_work[i] = z[i] / (ti.cast(1.0, cte.FLOAT_TYPE_TI) + coeff)
        z_aux[i] = z_work[i]


@ti.kernel
def iteration_erode_spl_kernel(
    z_work: ti.template(),
    z_aux: ti.template(),
    rec_work: ti.template(),
    rec_aux: ti.template(),
    alpha: ti.template(),
    alpha_aux: ti.template(),
):
    """
    Perform one pointer-jumping implicit SPL iteration.

    Author: B.G (02/2026)
    """
    for i in rec_work:
        rec = rec_work[i]
        z_work[i] += alpha[i] * z_aux[rec]
        alpha_aux[i] = alpha[i] * alpha[rec]
        rec_aux[i] = rec_work[rec]

    for i in rec_work:
        if gridctx.tfunc.can_out_flat(i):
            continue
        z_aux[i] = z_work[i]
        rec_work[i] = rec_aux[i]
        alpha[i] = alpha_aux[i]


@ti.kernel
def copy_flat_kernel(src: ti.template(), dst: ti.template()):
    """
    Copy one flat field into another.

    Author: B.G (02/2026)
    """
    for i in src:
        dst[i] = src[i]


@ti.kernel
def flat_to_grid_2d_kernel(src: ti.template(), dst: ti.template()):
    """
    Copy one flat row-major field into a 2D raster field.

    Author: B.G (02/2026)
    """
    for row, col in dst:
        dst[row, col] = src[row * ti.static(gridctx.nx) + col]


@ti.kernel
def grid_2d_to_flat_kernel(src: ti.template(), dst: ti.template()):
    """
    Copy one 2D raster field back into a flat row-major field.

    Author: B.G (02/2026)
    """
    for row, col in src:
        dst[row * ti.static(gridctx.nx) + col] = src[row, col]


@ti.kernel
def transpose_grid_2d_kernel(src: ti.template(), dst: ti.template()):
    """
    Transpose one 2D raster into another 2D field.

    Author: B.G (02/2026)
    """
    for row, col in src:
        dst[col, row] = src[row, col]


@ti.kernel
def build_hillslope_fixed_mask_kernel(fixed_mask: ti.template()):
    """
    Build the ADI fixed-cell mask in the current orientation.

    Fixed cells are:
    - inactive / nodata,
    - outlets / baselevels,
    - geometric boundaries on non-periodic axes.

    Author: B.G (02/2026)
    """
    for row, col in fixed_mask:
        orig_row = row
        orig_col = col
        if ti.static(transposed):
            orig_row = col
            orig_col = row

        idx = orig_row * ti.static(gridctx.nx) + orig_col
        fixed = ti.cast(0, ti.u8)

        if gridctx.tfunc.nodata_flat(idx) == 1:
            fixed = ti.u8(1)
        if gridctx.tfunc.can_out_flat(idx) == 1:
            fixed = ti.u8(1)

        if ti.static(gridctx.boundary_mode != "periodic_NS"):
            if orig_row == 0 or orig_row == ti.static(gridctx.ny) - 1:
                fixed = ti.u8(1)
        if ti.static(gridctx.boundary_mode != "periodic_EW"):
            if orig_col == 0 or orig_col == ti.static(gridctx.nx) - 1:
                fixed = ti.u8(1)

        fixed_mask[row, col] = fixed


@ti.kernel
def assemble_hillslope_adi_row_system_kernel(
    z_in: ti.template(),
    fixed_mask: ti.template(),
    a: ti.template(),
    b: ti.template(),
    c: ti.template(),
    rhs: ti.template(),
):
    """
    Assemble one ADI row system in the current orientation.

    Rows are implicit; the transverse axis is explicit. Fixed or blocked
    neighbours are replaced by the center value so no flux crosses a masked
    cell or a non-periodic boundary.

    Author: B.G (02/2026)
    """
    n_rows = ti.static(gridctx.ny if not transposed else gridctx.nx)
    n_cols = ti.static(gridctx.nx if not transposed else gridctx.ny)
    periodic_line = ti.static(
        gridctx.boundary_mode == ("periodic_EW" if not transposed else "periodic_NS")
    )
    periodic_transverse = ti.static(
        gridctx.boundary_mode == ("periodic_NS" if not transposed else "periodic_EW")
    )
    dx2 = ti.cast(ti.static(gridctx.dx * gridctx.dx), cte.FLOAT_TYPE_TI)

    for row, col in z_in:
        orig_row = row
        orig_col = col
        if ti.static(transposed):
            orig_row = col
            orig_col = row
        idx = orig_row * ti.static(gridctx.nx) + orig_col

        center = z_in[row, col]
        if fixed_mask[row, col] != 0:
            a[row, col] = ti.cast(0.0, cte.FLOAT_TYPE_TI)
            b[row, col] = ti.cast(1.0, cte.FLOAT_TYPE_TI)
            c[row, col] = ti.cast(0.0, cte.FLOAT_TYPE_TI)
            rhs[row, col] = center
            continue

        r = (
            ti.cast(0.5, cte.FLOAT_TYPE_TI)
            * get_kappa_bedrock(idx)
            * get_dt(idx)
            / dx2
        )

        up = center
        down = center

        if ti.static(periodic_transverse):
            up_row = row - 1
            if up_row < 0:
                up_row += ti.static(n_rows)
            if fixed_mask[up_row, col] == 0:
                up = z_in[up_row, col]

            down_row = row + 1
            if down_row >= ti.static(n_rows):
                down_row -= ti.static(n_rows)
            if fixed_mask[down_row, col] == 0:
                down = z_in[down_row, col]
        else:
            if row > 0 and fixed_mask[row - 1, col] == 0:
                up = z_in[row - 1, col]
            if row < ti.static(n_rows) - 1 and fixed_mask[row + 1, col] == 0:
                down = z_in[row + 1, col]

        blocked_left = ti.cast(1, ti.i32)
        blocked_right = ti.cast(1, ti.i32)

        if ti.static(periodic_line):
            left_col = col - 1
            if left_col < 0:
                left_col += ti.static(n_cols)
            blocked_left = ti.cast(fixed_mask[row, left_col] != 0, ti.i32)

            right_col = col + 1
            if right_col >= ti.static(n_cols):
                right_col -= ti.static(n_cols)
            blocked_right = ti.cast(fixed_mask[row, right_col] != 0, ti.i32)
        else:
            if col > 0:
                blocked_left = ti.cast(fixed_mask[row, col - 1] != 0, ti.i32)
            if col < ti.static(n_cols) - 1:
                blocked_right = ti.cast(fixed_mask[row, col + 1] != 0, ti.i32)

        left_coeff = ti.cast(0.0, cte.FLOAT_TYPE_TI)
        right_coeff = ti.cast(0.0, cte.FLOAT_TYPE_TI)
        diag_coeff = ti.cast(1.0, cte.FLOAT_TYPE_TI)

        if blocked_left == 0:
            left_coeff = -r
            diag_coeff += r
        if blocked_right == 0:
            right_coeff = -r
            diag_coeff += r

        a[row, col] = left_coeff
        b[row, col] = diag_coeff
        c[row, col] = right_coeff
        rhs[row, col] = center + r * (up - ti.cast(2.0, cte.FLOAT_TYPE_TI) * center + down)


@ti.kernel
def solve_tridiagonal_rows_kernel(
    a: ti.template(),
    b: ti.template(),
    c: ti.template(),
    rhs: ti.template(),
    cp: ti.template(),
    dp: ti.template(),
    out: ti.template(),
):
    """
    Solve one batch of independent tridiagonal row systems with Thomas.

    Author: B.G (02/2026)
    """
    n_rows = ti.static(gridctx.ny if not transposed else gridctx.nx)
    n_cols = ti.static(gridctx.nx if not transposed else gridctx.ny)

    for row in range(n_rows):
        denom = b[row, 0]
        cp[row, 0] = ti.cast(0.0, cte.FLOAT_TYPE_TI)
        if ti.static(n_cols > 1):
            cp[row, 0] = c[row, 0] / denom
        dp[row, 0] = rhs[row, 0] / denom

        for col in range(1, n_cols):
            denom = b[row, col] - a[row, col] * cp[row, col - 1]
            cp[row, col] = ti.cast(0.0, cte.FLOAT_TYPE_TI)
            if col < n_cols - 1:
                cp[row, col] = c[row, col] / denom
            dp[row, col] = (rhs[row, col] - a[row, col] * dp[row, col - 1]) / denom

        out[row, n_cols - 1] = dp[row, n_cols - 1]
        for rev in range(n_cols - 1):
            col = n_cols - 2 - rev
            out[row, col] = dp[row, col] - cp[row, col] * out[row, col + 1]


@ti.kernel
def solve_cyclic_rows_kernel(
    a: ti.template(),
    b: ti.template(),
    c: ti.template(),
    rhs: ti.template(),
    cp: ti.template(),
    dp: ti.template(),
    y: ti.template(),
    z_aux: ti.template(),
    out: ti.template(),
):
    """
    Solve one batch of cyclic tridiagonal row systems with Sherman-Morrison.

    The wrap coefficients come from ``a[row, 0]`` and ``c[row, last]``.

    Author: B.G (02/2026)
    """
    n_rows = ti.static(gridctx.ny if not transposed else gridctx.nx)
    n_cols = ti.static(gridctx.nx if not transposed else gridctx.ny)

    for row in range(n_rows):
        alpha = a[row, 0]
        beta = c[row, n_cols - 1]
        gamma = -b[row, 0]

        # First tridiagonal solve: T y = rhs.
        denom = b[row, 0] - gamma
        cp[row, 0] = ti.cast(0.0, cte.FLOAT_TYPE_TI)
        if ti.static(n_cols > 1):
            cp[row, 0] = c[row, 0] / denom
        dp[row, 0] = rhs[row, 0] / denom
        for col in range(1, n_cols):
            diag = b[row, col]
            superdiag = c[row, col]
            if col == n_cols - 1:
                diag -= alpha * beta / gamma
                superdiag = ti.cast(0.0, cte.FLOAT_TYPE_TI)
            denom = diag - a[row, col] * cp[row, col - 1]
            cp[row, col] = ti.cast(0.0, cte.FLOAT_TYPE_TI)
            if col < n_cols - 1:
                cp[row, col] = superdiag / denom
            dp[row, col] = (rhs[row, col] - a[row, col] * dp[row, col - 1]) / denom

        y[row, n_cols - 1] = dp[row, n_cols - 1]
        for rev in range(n_cols - 1):
            col = n_cols - 2 - rev
            y[row, col] = dp[row, col] - cp[row, col] * y[row, col + 1]

        # Second tridiagonal solve: T z = u.
        denom = b[row, 0] - gamma
        cp[row, 0] = ti.cast(0.0, cte.FLOAT_TYPE_TI)
        if ti.static(n_cols > 1):
            cp[row, 0] = c[row, 0] / denom
        dp[row, 0] = gamma / denom
        for col in range(1, n_cols):
            diag = b[row, col]
            superdiag = c[row, col]
            rhs_u = ti.cast(0.0, cte.FLOAT_TYPE_TI)
            if col == n_cols - 1:
                diag -= alpha * beta / gamma
                superdiag = ti.cast(0.0, cte.FLOAT_TYPE_TI)
                rhs_u = alpha
            denom = diag - a[row, col] * cp[row, col - 1]
            cp[row, col] = ti.cast(0.0, cte.FLOAT_TYPE_TI)
            if col < n_cols - 1:
                cp[row, col] = superdiag / denom
            dp[row, col] = (rhs_u - a[row, col] * dp[row, col - 1]) / denom

        z_aux[row, n_cols - 1] = dp[row, n_cols - 1]
        for rev in range(n_cols - 1):
            col = n_cols - 2 - rev
            z_aux[row, col] = dp[row, col] - cp[row, col] * z_aux[row, col + 1]

        factor = (
            y[row, 0] + beta * y[row, n_cols - 1] / gamma
        ) / (
            ti.cast(1.0, cte.FLOAT_TYPE_TI)
            + z_aux[row, 0]
            + beta * z_aux[row, n_cols - 1] / gamma
        )
        for col in range(n_cols):
            out[row, col] = y[row, col] - factor * z_aux[row, col]
