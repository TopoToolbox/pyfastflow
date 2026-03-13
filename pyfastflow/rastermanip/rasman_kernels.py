"""
Generic raster manipulation helpers and kernels for ``RasManContext``.

All kernels operate on flat row-major fields and always receive explicit source
and target dimensions.

Author: B.G (02/2026)
"""

import taichi as ti

from .. import constants as cte

BOUNDARY_CLAMP = 0
BOUNDARY_WRAP = 1
BOUNDARY_REFLECT = 2


@ti.func
def wrap_index(i: ti.i32, n: ti.i32) -> ti.i32:
    """
    Wrap one index into ``[0, n)``.

    Author: B.G (02/2026)
    """
    return ti.math.mod(i, n)


@ti.func
def reflect_index(i: ti.i32, n: ti.i32) -> ti.i32:
    """
    Reflect one index into ``[0, n)``.

    Author: B.G (02/2026)
    """
    period = ti.max(1, 2 * (n - 1))
    x = ti.math.mod(i, period)
    if x < 0:
        x += period
    if x >= n:
        x = period - x
    return ti.min(ti.max(x, 0), n - 1)


@ti.func
def resolve_index(i: ti.i32, n: ti.i32, mode: ti.i32) -> ti.i32:
    """
    Resolve one index according to boundary mode.

    Author: B.G (02/2026)
    """
    out = i
    if not (0 <= i < n):
        if mode == BOUNDARY_CLAMP:
            out = ti.min(ti.max(i, 0), n - 1)
        elif mode == BOUNDARY_WRAP:
            out = wrap_index(i, n)
        else:
            out = reflect_index(i, n)
    return out


@ti.func
def cubic_interpolate(
    v0: cte.FLOAT_TYPE_TI,
    v1: cte.FLOAT_TYPE_TI,
    v2: cte.FLOAT_TYPE_TI,
    v3: cte.FLOAT_TYPE_TI,
    t: cte.FLOAT_TYPE_TI,
) -> cte.FLOAT_TYPE_TI:
    """
    Cubic interpolation between four values.

    Author: B.G (02/2026)
    """
    a = -0.5 * v0 + 1.5 * v1 - 1.5 * v2 + 0.5 * v3
    b = v0 - 2.5 * v1 + 2.0 * v2 - 0.5 * v3
    c = -0.5 * v0 + 0.5 * v2
    d = v1
    return ((a * t + b) * t + c) * t + d


@ti.func
def sinc(x: cte.FLOAT_TYPE_TI) -> cte.FLOAT_TYPE_TI:
    """
    Normalized sinc function.

    Author: B.G (02/2026)
    """
    ax = ti.abs(x)
    out = ti.cast(1.0, cte.FLOAT_TYPE_TI)
    if ax >= ti.cast(1e-6, cte.FLOAT_TYPE_TI):
        pix = ti.math.pi * x
        out = ti.sin(pix) / pix
    return out


@ti.func
def lanczos_weight(x: cte.FLOAT_TYPE_TI, a: ti.i32) -> cte.FLOAT_TYPE_TI:
    """
    Lanczos windowed-sinc weight.

    Author: B.G (02/2026)
    """
    ax = ti.abs(x)
    out = ti.cast(0.0, cte.FLOAT_TYPE_TI)
    if ax < ti.cast(a, cte.FLOAT_TYPE_TI):
        out = sinc(x) * sinc(x / ti.cast(a, cte.FLOAT_TYPE_TI))
    return out


@ti.func
def sample_nearest(
    source_flat: ti.template(),
    nx_src: ti.i32,
    ny_src: ti.i32,
    x: cte.FLOAT_TYPE_TI,
    y: cte.FLOAT_TYPE_TI,
    boundary_mode: ti.i32,
) -> cte.FLOAT_TYPE_TI:
    """
    Nearest-neighbor sample from a flat row-major source.

    Author: B.G (02/2026)
    """
    ix = ti.cast(ti.floor(x + 0.5), ti.i32)
    iy = ti.cast(ti.floor(y + 0.5), ti.i32)
    ix = resolve_index(ix, nx_src, boundary_mode)
    iy = resolve_index(iy, ny_src, boundary_mode)
    return source_flat[iy * nx_src + ix]


@ti.func
def sample_bilinear(
    source_flat: ti.template(),
    nx_src: ti.i32,
    ny_src: ti.i32,
    x: cte.FLOAT_TYPE_TI,
    y: cte.FLOAT_TYPE_TI,
    boundary_mode: ti.i32,
) -> cte.FLOAT_TYPE_TI:
    """
    Bilinear sample from a flat row-major source.

    Author: B.G (02/2026)
    """
    base_x = ti.floor(x, dtype=ti.i32)
    base_y = ti.floor(y, dtype=ti.i32)
    fx = x - ti.cast(base_x, cte.FLOAT_TYPE_TI)
    fy = y - ti.cast(base_y, cte.FLOAT_TYPE_TI)

    out = ti.cast(0.0, cte.FLOAT_TYPE_TI)
    for dy in ti.static(range(2)):
        for dx in ti.static(range(2)):
            ix = resolve_index(base_x + dx, nx_src, boundary_mode)
            iy = resolve_index(base_y + dy, ny_src, boundary_mode)
            w = (1.0 - fx if dx == 0 else fx) * (1.0 - fy if dy == 0 else fy)
            out += source_flat[iy * nx_src + ix] * w
    return out


@ti.func
def sample_bicubic(
    source_flat: ti.template(),
    nx_src: ti.i32,
    ny_src: ti.i32,
    x: cte.FLOAT_TYPE_TI,
    y: cte.FLOAT_TYPE_TI,
    boundary_mode: ti.i32,
) -> cte.FLOAT_TYPE_TI:
    """
    Bicubic sample from a flat row-major source.

    Author: B.G (02/2026)
    """
    base_x = ti.floor(x, dtype=ti.i32)
    base_y = ti.floor(y, dtype=ti.i32)
    fx = x - ti.cast(base_x, cte.FLOAT_TYPE_TI)
    fy = y - ti.cast(base_y, cte.FLOAT_TYPE_TI)

    row_vals = ti.Vector(
        [
            ti.cast(0.0, cte.FLOAT_TYPE_TI),
            ti.cast(0.0, cte.FLOAT_TYPE_TI),
            ti.cast(0.0, cte.FLOAT_TYPE_TI),
            ti.cast(0.0, cte.FLOAT_TYPE_TI),
        ]
    )
    for j in ti.static(range(4)):
        jj = resolve_index(base_y + j - 1, ny_src, boundary_mode)
        samples = ti.Vector(
            [
                ti.cast(0.0, cte.FLOAT_TYPE_TI),
                ti.cast(0.0, cte.FLOAT_TYPE_TI),
                ti.cast(0.0, cte.FLOAT_TYPE_TI),
                ti.cast(0.0, cte.FLOAT_TYPE_TI),
            ]
        )
        for i in ti.static(range(4)):
            ii = resolve_index(base_x + i - 1, nx_src, boundary_mode)
            samples[i] = source_flat[jj * nx_src + ii]
        row_vals[j] = cubic_interpolate(samples[0], samples[1], samples[2], samples[3], fx)
    return cubic_interpolate(row_vals[0], row_vals[1], row_vals[2], row_vals[3], fy)


@ti.func
def sample_lanczos8(
    source_flat: ti.template(),
    nx_src: ti.i32,
    ny_src: ti.i32,
    x: cte.FLOAT_TYPE_TI,
    y: cte.FLOAT_TYPE_TI,
    boundary_mode: ti.i32,
) -> cte.FLOAT_TYPE_TI:
    """
    Lanczos sample with support ``a=4`` (8x8 neighborhood).

    Author: B.G (02/2026)
    """
    a = ti.static(4)
    base_x = ti.floor(x, dtype=ti.i32)
    base_y = ti.floor(y, dtype=ti.i32)
    sum_w = ti.cast(0.0, cte.FLOAT_TYPE_TI)
    sum_v = ti.cast(0.0, cte.FLOAT_TYPE_TI)

    for oy in ti.static(range(8)):
        sy = base_y + oy - 3
        wy = lanczos_weight(y - ti.cast(sy, cte.FLOAT_TYPE_TI), a)
        iy = resolve_index(sy, ny_src, boundary_mode)
        for ox in ti.static(range(8)):
            sx = base_x + ox - 3
            wx = lanczos_weight(x - ti.cast(sx, cte.FLOAT_TYPE_TI), a)
            w = wx * wy
            ix = resolve_index(sx, nx_src, boundary_mode)
            sum_w += w
            sum_v += w * source_flat[iy * nx_src + ix]

    out = source_flat[
        resolve_index(base_y, ny_src, boundary_mode) * nx_src
        + resolve_index(base_x, nx_src, boundary_mode)
    ]
    if ti.abs(sum_w) > ti.cast(1e-12, cte.FLOAT_TYPE_TI):
        out = sum_v / sum_w
    return out


@ti.func
def source_coord_1d(i_t: ti.i32, n_src: ti.i32, n_t: ti.i32) -> cte.FLOAT_TYPE_TI:
    """
    Center-preserving source coordinate for one target index.

    Author: B.G (02/2026)
    """
    return (ti.cast(i_t, cte.FLOAT_TYPE_TI) + 0.5) * ti.cast(n_src, cte.FLOAT_TYPE_TI) / ti.cast(n_t, cte.FLOAT_TYPE_TI) - 0.5


@ti.func
def box_bounds_1d(i_t: ti.i32, n_src: ti.i32, n_t: ti.i32) -> ti.types.vector(2, ti.i32):
    """
    Integer source bounds ``[start, end)`` for one target bin.

    Author: B.G (02/2026)
    """
    x0 = ti.cast(i_t, cte.FLOAT_TYPE_TI) * ti.cast(n_src, cte.FLOAT_TYPE_TI) / ti.cast(n_t, cte.FLOAT_TYPE_TI)
    x1 = ti.cast(i_t + 1, cte.FLOAT_TYPE_TI) * ti.cast(n_src, cte.FLOAT_TYPE_TI) / ti.cast(n_t, cte.FLOAT_TYPE_TI)
    start = ti.floor(x0, dtype=ti.i32)
    end = ti.floor(x1 - ti.cast(1e-6, cte.FLOAT_TYPE_TI), dtype=ti.i32) + 1
    start = ti.max(0, ti.min(start, n_src - 1))
    end = ti.max(start + 1, ti.min(end, n_src))
    return ti.Vector([start, end])


@ti.kernel
def two_d_to_flat_kernel(source_2d: ti.template(), target_flat: ti.template(), nx: ti.i32, ny: ti.i32):
    """
    Copy a 2D field to flat row-major layout.

    Author: B.G (02/2026)
    """
    for j, i in source_2d:
        if j < ny and i < nx:
            target_flat[j * nx + i] = source_2d[j, i]


@ti.kernel
def flat_to_2d_kernel(source_flat: ti.template(), target_2d: ti.template(), nx: ti.i32, ny: ti.i32):
    """
    Copy a flat row-major field to 2D layout.

    Author: B.G (02/2026)
    """
    for j, i in target_2d:
        if j < ny and i < nx:
            target_2d[j, i] = source_flat[j * nx + i]


@ti.kernel
def upscale_nearest_kernel(
    source_flat: ti.template(),
    target_flat: ti.template(),
    nx_src: ti.i32,
    ny_src: ti.i32,
    nx_t: ti.i32,
    ny_t: ti.i32,
    boundary_mode: ti.i32,
):
    """
    Generic nearest-neighbor upscaling kernel.

    Author: B.G (02/2026)
    """
    for idx in target_flat:
        j_t = idx // nx_t
        i_t = idx % nx_t
        if j_t >= ny_t or i_t >= nx_t:
            continue
        x = source_coord_1d(i_t, nx_src, nx_t)
        y = source_coord_1d(j_t, ny_src, ny_t)
        target_flat[idx] = sample_nearest(source_flat, nx_src, ny_src, x, y, boundary_mode)


@ti.kernel
def upscale_bilinear_kernel(
    source_flat: ti.template(),
    target_flat: ti.template(),
    nx_src: ti.i32,
    ny_src: ti.i32,
    nx_t: ti.i32,
    ny_t: ti.i32,
    boundary_mode: ti.i32,
):
    """
    Generic bilinear upscaling kernel.

    Author: B.G (02/2026)
    """
    for idx in target_flat:
        j_t = idx // nx_t
        i_t = idx % nx_t
        if j_t >= ny_t or i_t >= nx_t:
            continue
        x = source_coord_1d(i_t, nx_src, nx_t)
        y = source_coord_1d(j_t, ny_src, ny_t)
        target_flat[idx] = sample_bilinear(source_flat, nx_src, ny_src, x, y, boundary_mode)


@ti.kernel
def upscale_bicubic_kernel(
    source_flat: ti.template(),
    target_flat: ti.template(),
    nx_src: ti.i32,
    ny_src: ti.i32,
    nx_t: ti.i32,
    ny_t: ti.i32,
    boundary_mode: ti.i32,
):
    """
    Generic bicubic upscaling kernel.

    Author: B.G (02/2026)
    """
    for idx in target_flat:
        j_t = idx // nx_t
        i_t = idx % nx_t
        if j_t >= ny_t or i_t >= nx_t:
            continue
        x = source_coord_1d(i_t, nx_src, nx_t)
        y = source_coord_1d(j_t, ny_src, ny_t)
        target_flat[idx] = sample_bicubic(source_flat, nx_src, ny_src, x, y, boundary_mode)


@ti.kernel
def upscale_lanczos8_kernel(
    source_flat: ti.template(),
    target_flat: ti.template(),
    nx_src: ti.i32,
    ny_src: ti.i32,
    nx_t: ti.i32,
    ny_t: ti.i32,
    boundary_mode: ti.i32,
):
    """
    Generic Lanczos (a=4) upscaling kernel.

    Author: B.G (02/2026)
    """
    for idx in target_flat:
        j_t = idx // nx_t
        i_t = idx % nx_t
        if j_t >= ny_t or i_t >= nx_t:
            continue
        x = source_coord_1d(i_t, nx_src, nx_t)
        y = source_coord_1d(j_t, ny_src, ny_t)
        target_flat[idx] = sample_lanczos8(source_flat, nx_src, ny_src, x, y, boundary_mode)


@ti.kernel
def downscale_mean_kernel(
    source_flat: ti.template(),
    target_flat: ti.template(),
    nx_src: ti.i32,
    ny_src: ti.i32,
    nx_t: ti.i32,
    ny_t: ti.i32,
):
    """
    Downscale using mean over source bins.

    Author: B.G (02/2026)
    """
    for idx in target_flat:
        j_t = idx // nx_t
        i_t = idx % nx_t
        if j_t >= ny_t or i_t >= nx_t:
            continue

        bx = box_bounds_1d(i_t, nx_src, nx_t)
        by = box_bounds_1d(j_t, ny_src, ny_t)

        total = ti.cast(0.0, cte.FLOAT_TYPE_TI)
        count = 0
        for y in range(by[0], by[1]):
            base = y * nx_src
            for x in range(bx[0], bx[1]):
                total += source_flat[base + x]
                count += 1

        target_flat[idx] = total / ti.cast(ti.max(1, count), cte.FLOAT_TYPE_TI)


@ti.kernel
def downscale_min_kernel(
    source_flat: ti.template(),
    target_flat: ti.template(),
    nx_src: ti.i32,
    ny_src: ti.i32,
    nx_t: ti.i32,
    ny_t: ti.i32,
):
    """
    Downscale using minimum over source bins.

    Author: B.G (02/2026)
    """
    for idx in target_flat:
        j_t = idx // nx_t
        i_t = idx % nx_t
        if j_t >= ny_t or i_t >= nx_t:
            continue

        bx = box_bounds_1d(i_t, nx_src, nx_t)
        by = box_bounds_1d(j_t, ny_src, ny_t)

        out = ti.cast(1e30, cte.FLOAT_TYPE_TI)
        for y in range(by[0], by[1]):
            base = y * nx_src
            for x in range(bx[0], bx[1]):
                out = ti.min(out, source_flat[base + x])
        target_flat[idx] = out


@ti.kernel
def downscale_max_kernel(
    source_flat: ti.template(),
    target_flat: ti.template(),
    nx_src: ti.i32,
    ny_src: ti.i32,
    nx_t: ti.i32,
    ny_t: ti.i32,
):
    """
    Downscale using maximum over source bins.

    Author: B.G (02/2026)
    """
    for idx in target_flat:
        j_t = idx // nx_t
        i_t = idx % nx_t
        if j_t >= ny_t or i_t >= nx_t:
            continue

        bx = box_bounds_1d(i_t, nx_src, nx_t)
        by = box_bounds_1d(j_t, ny_src, ny_t)

        out = ti.cast(-1e30, cte.FLOAT_TYPE_TI)
        for y in range(by[0], by[1]):
            base = y * nx_src
            for x in range(bx[0], bx[1]):
                out = ti.max(out, source_flat[base + x])
        target_flat[idx] = out


@ti.kernel
def downscale_percentile_kernel(
    source_flat: ti.template(),
    target_flat: ti.template(),
    nx_src: ti.i32,
    ny_src: ti.i32,
    nx_t: ti.i32,
    ny_t: ti.i32,
    percentile: cte.FLOAT_TYPE_TI,
    n_iter: ti.i32,
):
    """
    Downscale using approximate percentile via value-space bisection.

    This avoids local sorting buffers and works for arbitrary source bin sizes.

    Author: B.G (02/2026)
    """
    for idx in target_flat:
        j_t = idx // nx_t
        i_t = idx % nx_t
        if j_t >= ny_t or i_t >= nx_t:
            continue

        bx = box_bounds_1d(i_t, nx_src, nx_t)
        by = box_bounds_1d(j_t, ny_src, ny_t)

        vmin = ti.cast(1e30, cte.FLOAT_TYPE_TI)
        vmax = ti.cast(-1e30, cte.FLOAT_TYPE_TI)
        n = 0
        for y in range(by[0], by[1]):
            base = y * nx_src
            for x in range(bx[0], bx[1]):
                v = source_flat[base + x]
                vmin = ti.min(vmin, v)
                vmax = ti.max(vmax, v)
                n += 1

        if n <= 1 or vmax <= vmin:
            target_flat[idx] = vmin
            continue

        lo = vmin
        hi = vmax
        rank = ti.cast(percentile, cte.FLOAT_TYPE_TI) * (ti.cast(n - 1, cte.FLOAT_TYPE_TI) / 100.0)

        it = 0
        while it < n_iter:
            mid = 0.5 * (lo + hi)
            count_le = 0
            for y in range(by[0], by[1]):
                base = y * nx_src
                for x in range(bx[0], bx[1]):
                    if source_flat[base + x] <= mid:
                        count_le += 1

            if ti.cast(count_le, cte.FLOAT_TYPE_TI) <= rank:
                lo = mid
            else:
                hi = mid
            it += 1

        target_flat[idx] = 0.5 * (lo + hi)


@ti.kernel
def downscale_median_kernel(
    source_flat: ti.template(),
    target_flat: ti.template(),
    nx_src: ti.i32,
    ny_src: ti.i32,
    nx_t: ti.i32,
    ny_t: ti.i32,
    n_iter: ti.i32,
):
    """
    Downscale using median (exact rank definition via two order statistics).

    For even bin sizes, returns the average of middle-low and middle-high.

    Author: B.G (02/2026)
    """
    for idx in target_flat:
        j_t = idx // nx_t
        i_t = idx % nx_t
        if j_t >= ny_t or i_t >= nx_t:
            continue

        bx = box_bounds_1d(i_t, nx_src, nx_t)
        by = box_bounds_1d(j_t, ny_src, ny_t)

        vmin = ti.cast(1e30, cte.FLOAT_TYPE_TI)
        vmax = ti.cast(-1e30, cte.FLOAT_TYPE_TI)
        n = 0
        for y in range(by[0], by[1]):
            base = y * nx_src
            for x in range(bx[0], bx[1]):
                v = source_flat[base + x]
                vmin = ti.min(vmin, v)
                vmax = ti.max(vmax, v)
                n += 1

        if n <= 1 or vmax <= vmin:
            target_flat[idx] = vmin
            continue

        k1 = (n - 1) // 2
        k2 = n // 2

        lo1 = vmin
        hi1 = vmax
        it1 = 0
        while it1 < n_iter:
            mid = 0.5 * (lo1 + hi1)
            count_le = 0
            for y in range(by[0], by[1]):
                base = y * nx_src
                for x in range(bx[0], bx[1]):
                    if source_flat[base + x] <= mid:
                        count_le += 1
            if count_le <= k1:
                lo1 = mid
            else:
                hi1 = mid
            it1 += 1
        v1 = 0.5 * (lo1 + hi1)

        lo2 = vmin
        hi2 = vmax
        it2 = 0
        while it2 < n_iter:
            mid = 0.5 * (lo2 + hi2)
            count_le = 0
            for y in range(by[0], by[1]):
                base = y * nx_src
                for x in range(bx[0], bx[1]):
                    if source_flat[base + x] <= mid:
                        count_le += 1
            if count_le <= k2:
                lo2 = mid
            else:
                hi2 = mid
            it2 += 1
        v2 = 0.5 * (lo2 + hi2)

        target_flat[idx] = 0.5 * (v1 + v2)
