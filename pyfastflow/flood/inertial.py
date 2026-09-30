"""Conservative CuPy local-inertia flood program on D4 or D8 grids.

``topology="D4"`` with the ``"bates"`` regulator keeps only the fixed bed,
depth, and two staggered cardinal flux grids: about 16 bytes per cell.
``topology="D8"`` additionally allocates the two unique diagonal link grids,
``qxy`` (south-east) and ``qyx`` (south-west), for about 24 bytes per cell.
The diffusive regulator modes allocate a second set of flux grids for the
provisional Bates update; D4 and D8 then use about 24 and 40 bytes per cell,
respectively. D4 never allocates diagonal grids.

All live-cell links are stored exactly once and contribute with opposite signs
to their two endpoints. D8 uses the requested regular-octagon convention:
every one of its eight links has width ``dx / 2``; cardinal and diagonal link
lengths are respectively ``dx`` and ``sqrt(2) * dx``. The depth control
volume remains ``dx * dx``.
"""

from pyfastflow.core import Backend, KernelBuilder
from pyfastflow.core.context.program import Dim, ProgramBuilder
from pyfastflow.grid import make_grid_group, make_grid_parameters
from pyfastflow.grid._mask import GridMaskAccessor


def _cupy_only(be: Backend) -> None:
    if be.name != "cupy":
        raise ValueError(f"InertialFloodProgram is cupy-only, got {be.name!r}")


def _grid_plan(frozen, values):
    """Bind grid leaves by their exact paths and local values by leaf name."""
    bound = frozen.build()
    try:
        plan = {}
        for address in bound.addresses():
            name = ".".join(address)
            if len(address) >= 2 and address[-2] == "grid":
                plan[name] = "grid"
            elif address[-1] in values:
                plan[name] = values[address[-1]]
        return plan
    finally:
        bound.close()


def _diagonal_shape(config):
    """Allocate diagonal links only for the D8 recipe."""
    if config["topology"] == "D8":
        return (config["ny"], config["nx"])
    return (0, 0)


def _regulated_qx_shape(config):
    if config["regulator"] != "bates":
        return (config["ny"], config["nx"] + 1)
    return (0, 0)


def _regulated_qy_shape(config):
    if config["regulator"] != "bates":
        return (config["ny"] + 1, config["nx"])
    return (0, 0)


def _regulated_diagonal_shape(config):
    if config["regulator"] != "bates" and config["topology"] == "D8":
        return (config["ny"], config["nx"])
    return (0, 0)


def _inlet_cell_shape(config):
    """Per-cell inlet fields (mask + held depth); allocated only when inlet=True."""
    if config["inlet"]:
        return (config["ny"], config["nx"])
    return (0, 0)


def _inlet_qx_shape(config):
    if config["inlet"]:
        return (config["ny"], config["nx"] + 1)
    return (0, 0)


def _inlet_qy_shape(config):
    if config["inlet"]:
        return (config["ny"] + 1, config["nx"])
    return (0, 0)


def _reset_factory(be: Backend, _bundles, config):
    _cupy_only(be)
    nx, ny = int(config["nx"]), int(config["ny"])
    n = nx * ny
    nqx, nqy = ny * (nx + 1), (ny + 1) * nx
    diagonal = config["topology"] == "D8"
    args = ", float* qxy, float* qyx" if diagonal else ""
    clear_diagonal = f"if (i < {n}) qxy[i] = qyx[i] = 0.0f;" if diagonal else ""
    return KernelBuilder(
        f'''extern "C" __global__ void inertial_reset(
                float* h, float* qx, float* qy{args}) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < {n}) h[i] = 0.0f;
            if (i < {nqx}) qx[i] = 0.0f;
            if (i < {nqy}) qy[i] = 0.0f;
            {clear_diagonal}
        }}''',
        domain=max(n, nqx, nqy),
    ).freeze()


def _flux_factory(be: Backend, bundles, config):
    """Build the unfiltered Bates link update.

    Regulated variants write this provisional field to the existing ``*_reg``
    buffers.  A subsequent pass may smooth whole provisional link values, so
    the pressure-gradient and friction terms always remain a Bates update.
    """
    _cupy_only(be)
    nx, ny = int(config["nx"]), int(config["ny"])
    n = nx * ny
    d8 = config["topology"] == "D8"
    periodic_x = config["boundary"] == "periodic_EW"
    periodic_y = config["boundary"] == "periodic_NS"
    filtered = config["regulator"] != "bates"
    left_k, up_k = (3, 1) if d8 else (1, 0)
    q_type = "const float*" if filtered else "float*"
    diagonal_args = f", {q_type} qxy, {q_type} qyx" if d8 else ""
    regulated_args = (
        ", float* qx_reg, float* qy_reg"
        + (", float* qxy_reg, float* qyx_reg" if d8 else "")
        if filtered else ""
    )
    qx_out = "qx_reg" if filtered else "qx"
    qy_out = "qy_reg" if filtered else "qy"
    qxy_out = "qxy_reg" if filtered else "qxy"
    qyx_out = "qyx_reg" if filtered else "qyx"

    diagonal_update = f'''
            int se = $ctx.grid.neighbour(i, 7)$;
            if (se >= 0) {{
                float qxy_old = qxy[i];
                {qxy_out}[i] = inertial_update(qxy_old,
                    h[i], h[se], z[i], z[se],
                    dt, gravity, manning, froude_limit, transfer_fraction,
                    min_hflow, dx * 1.4142135623730951f, link_width, cell_area);
            }} else {{
                {qxy_out}[i] = ($ctx.grid.can_out(i)$ && (x == {nx - 1} || y == {ny - 1}))
                    ? boundary_flux : 0.0f;
            }}
            int sw = $ctx.grid.neighbour(i, 5)$;
            if (sw >= 0) {{
                float qyx_old = qyx[i];
                {qyx_out}[i] = inertial_update(qyx_old,
                    h[i], h[sw], z[i], z[sw],
                    dt, gravity, manning, froude_limit, transfer_fraction,
                    min_hflow, dx * 1.4142135623730951f, link_width, cell_area);
            }} else {{
                {qyx_out}[i] = ($ctx.grid.can_out(i)$ && (x == 0 || y == {ny - 1}))
                    ? boundary_flux : 0.0f;
            }}''' if d8 else ""
    inactive_clear = f"{qxy_out}[i] = {qyx_out}[i] = 0.0f;" if d8 else ""
    return KernelBuilder(
        f'''__device__ __forceinline__ float inertial_update(
                float q_old, float h_a, float h_b,
                float z_a, float z_b,
                float dt, float gravity, float manning, float froude_limit,
                float transfer_fraction, float min_hflow, float link_length,
                float link_width, float cell_area) {{
            float hflow = 0.5f * (h_a + h_b);
            if (!(hflow > min_hflow)) return 0.0f;
            float slope = ((z_b + h_b) - (z_a + h_a)) / link_length;
            float h10_3 = hflow * hflow * hflow * cbrtf(hflow);
            float friction = 1.0f + gravity * hflow * dt * manning * manning
                * fabsf(q_old) / fmaxf(h10_3, 1.0e-12f);
            float q = (q_old - gravity * hflow * dt * slope) / friction;
            float q_froude = hflow * sqrtf(gravity * hflow) * froude_limit;
            q = fminf(fmaxf(q, -q_froude), q_froude);
            float donor_h = q >= 0.0f ? h_a : h_b;
            float q_transfer = transfer_fraction * donor_h * cell_area
                / (link_width * fmaxf(dt, 1.0e-12f));
            return fminf(fmaxf(q, -q_transfer), q_transfer);
        }}

        extern "C" __global__ void inertial_update_flux(
                const float* z, const float* h, {q_type} qx, {q_type} qy{diagonal_args}{regulated_args}) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            int x = i % {nx};
            int y = i / {nx};
            int fx = y * ({nx} + 1) + x;
            int fy = y * {nx} + x;
            if (!$ctx.grid.is_active(i)$) {{
                {qx_out}[fx] = {qy_out}[fy] = 0.0f;
                if (x == {nx - 1}) {qx_out}[fx + 1] = 0.0f;
                if (y == {ny - 1}) {qy_out}[fy + {nx}] = 0.0f;
                {inactive_clear}
                return;
            }}
            float dt = $ctx.dt.get(0)$;
            float gravity = $ctx.gravity.get(0)$;
            float manning = $ctx.manning.get(0)$;
            float froude_limit = $ctx.froude_limit.get(0)$;
            float transfer_fraction = $ctx.transfer_fraction.get(0)$;
            float min_hflow = $ctx.min_flow_depth.get(0)$;
            float boundary_flux = $ctx.boundary_flux.get(0)$;
            float dx = $ctx.grid.DX.get(0)$;
            float link_width = dx * {"0.5f" if d8 else "1.0f"};
            float cell_area = dx * dx;

            int west = $ctx.grid.neighbour(i, {left_k})$;
            if (west >= 0) {{
                float qx_old = qx[fx];
                {qx_out}[fx] = inertial_update(qx_old,
                    h[west], h[i], z[west], z[i],
                    dt, gravity, manning, froude_limit, transfer_fraction,
                    min_hflow, dx, link_width, cell_area);
            }} else {{
                {qx_out}[fx] = (x == 0 && $ctx.grid.can_out(i)$) ? -boundary_flux : 0.0f;
            }}
            if (x == {nx - 1}) {{
                {qx_out}[fx + 1] = {"0.0f" if periodic_x else "($ctx.grid.can_out(i)$ ? boundary_flux : 0.0f)"};
            }}

            int north = $ctx.grid.neighbour(i, {up_k})$;
            if (north >= 0) {{
                float qy_old = qy[fy];
                {qy_out}[fy] = inertial_update(qy_old,
                    h[north], h[i], z[north], z[i],
                    dt, gravity, manning, froude_limit, transfer_fraction,
                    min_hflow, dx, link_width, cell_area);
            }} else {{
                {qy_out}[fy] = (y == 0 && $ctx.grid.can_out(i)$) ? -boundary_flux : 0.0f;
            }}
            if (y == {ny - 1}) {{
                {qy_out}[fy + {nx}] = {"0.0f" if periodic_y else "($ctx.grid.can_out(i)$ ? boundary_flux : 0.0f)"};
            }}
            {diagonal_update}
        }}''',
        domain=n,
    ).compose("grid", bundles["grid"]).freeze()


def _regulate_flux_factory(be: Backend, bundles, config):
    """Optionally filter complete provisional link fluxes before continuity.

    ``q_upwind``/``q_centered`` weight the provisional link against its two
    parallel neighbours with one fixed ``regulator_theta`` (de Almeida et al.,
    2012). ``s_upwind``/``s_centered`` keep the same upwind/centered
    construction but replace that fixed weight with the per-interface adaptive
    factor ``theta = 1 - (dt / link_length) * min(|q| / hflow,
    sqrt(gravity * hflow))``, clamped to ``[0, 1]``, where ``hflow`` is the
    interface flow depth and ``q`` the provisional interface discharge
    (Sridharan et al., 2020).

    Author: B.G (09/2026)
    """
    _cupy_only(be)
    if config["regulator"] == "bates":
        return KernelBuilder(
            'extern "C" __global__ void inertial_regulate_flux() {}', domain=1,
        ).freeze()

    nx, ny = int(config["nx"]), int(config["ny"])
    n = nx * ny
    d8 = config["topology"] == "D8"
    periodic_x = config["boundary"] == "periodic_EW"
    periodic_y = config["boundary"] == "periodic_NS"
    regulator_mode = {"q_upwind": 1, "q_centered": 2,
                      "s_upwind": 1, "s_centered": 2}[config["regulator"]]
    adaptive = config["regulator"] in ("s_upwind", "s_centered")
    diag = "1.4142135623730951f"

    # s-schemes: adaptive per-interface theta from the local hydraulic state;
    # q-schemes: one fixed, clamped regulator_theta reused at every face.
    theta_helper = (f'''__device__ __forceinline__ float inertial_theta(
                float q, float hflow, float dt, float gravity, float link_length) {{
            if (!(hflow > 0.0f)) return 1.0f;
            float velocity = fabsf(q) / hflow;
            float celerity = sqrtf(gravity * hflow);
            float th = 1.0f - (dt / link_length) * fminf(velocity, celerity);
            return fminf(fmaxf(th, 0.0f), 1.0f);
        }}

        ''' if adaptive else "")
    h_arg = ", const float* h" if adaptive else ""
    if adaptive:
        theta_setup = ("float dt = $ctx.dt.get(0)$;\n"
                       "            float gravity = $ctx.gravity.get(0)$;\n"
                       "            float dx = $ctx.grid.DX.get(0)$;")
        theta_x = "inertial_theta(qx_reg[fx], 0.5f * (h[west] + h[i]), dt, gravity, dx)"
        theta_y = "inertial_theta(qy_reg[fy], 0.5f * (h[north] + h[i]), dt, gravity, dx)"
        theta_xy = f"inertial_theta(qxy_reg[i], 0.5f * (h[i] + h[se]), dt, gravity, dx * {diag})"
        theta_yx = f"inertial_theta(qyx_reg[i], 0.5f * (h[i] + h[sw]), dt, gravity, dx * {diag})"
    else:
        theta_setup = "float theta = fminf(fmaxf($ctx.regulator_theta.get(0)$, 0.0f), 1.0f);"
        theta_x = theta_y = theta_xy = theta_yx = "theta"

    diagonal_args = ", float* qxy, float* qyx, const float* qxy_reg, const float* qyx_reg" if d8 else ""
    diagonal_update = f'''
            int se = $ctx.grid.neighbour(i, 7)$;
            int nw = $ctx.grid.neighbour(i, 0)$;
            qxy[i] = se >= 0
                ? inertial_filter(qxy_reg[i],
                    nw >= 0 ? qxy_reg[nw] : qxy_reg[i],
                    qxy_reg[se], {theta_xy}, {regulator_mode})
                : qxy_reg[i];
            int sw = $ctx.grid.neighbour(i, 5)$;
            int ne = $ctx.grid.neighbour(i, 2)$;
            qyx[i] = sw >= 0
                ? inertial_filter(qyx_reg[i],
                    ne >= 0 ? qyx_reg[ne] : qyx_reg[i],
                    qyx_reg[sw], {theta_yx}, {regulator_mode})
                : qyx_reg[i];''' if d8 else ""
    inactive_clear = "qxy[i] = qyx[i] = 0.0f;" if d8 else ""
    return KernelBuilder(
        f'''__device__ __forceinline__ float inertial_filter(
                float q, float q_previous, float q_next, float theta,
                int regulator_mode) {{
            float neighbour_q = regulator_mode == 1
                ? (q >= 0.0f ? q_previous : q_next)
                : 0.5f * (q_previous + q_next);
            return theta * q + (1.0f - theta) * neighbour_q;
        }}

        {theta_helper}extern "C" __global__ void inertial_regulate_flux(
                float* qx, float* qy, const float* qx_reg, const float* qy_reg{diagonal_args}{h_arg}) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            int x = i % {nx};
            int y = i / {nx};
            int fx = y * ({nx} + 1) + x;
            int fy = y * {nx} + x;
            if (!$ctx.grid.is_active(i)$) {{
                qx[fx] = qy[fy] = 0.0f;
                if (x == {nx - 1}) qx[fx + 1] = 0.0f;
                if (y == {ny - 1}) qy[fy + {nx}] = 0.0f;
                {inactive_clear}
                return;
            }}
            {theta_setup}
            int west = $ctx.grid.neighbour(i, {3 if d8 else 1})$;
            qx[fx] = west >= 0
                ? inertial_filter(qx_reg[fx],
                    qx_reg[x > 0 ? fx - 1 : ({int(periodic_x)} ? fx + {nx - 1} : fx)],
                    qx_reg[x < {nx - 1} ? fx + 1 : ({int(periodic_x)} ? y * ({nx} + 1) : fx + 1)],
                    {theta_x}, {regulator_mode})
                : qx_reg[fx];
            if (x == {nx - 1}) qx[fx + 1] = qx_reg[fx + 1];

            int north = $ctx.grid.neighbour(i, {1 if d8 else 0})$;
            qy[fy] = north >= 0
                ? inertial_filter(qy_reg[fy],
                    qy_reg[y > 0 ? fy - {nx} : ({int(periodic_y)} ? ({ny - 1}) * {nx} + x : fy)],
                    qy_reg[y < {ny - 1} ? fy + {nx} : ({int(periodic_y)} ? x : fy + {nx})],
                    {theta_y}, {regulator_mode})
                : qy_reg[fy];
            if (y == {ny - 1}) qy[fy + {nx}] = qy_reg[fy + {nx}];
            {diagonal_update}
        }}''',
        domain=n,
    ).compose("grid", bundles["grid"]).freeze()


def _depth_factory(be: Backend, bundles, config):
    _cupy_only(be)
    nx, ny = int(config["nx"]), int(config["ny"])
    n = nx * ny
    d8 = config["topology"] == "D8"
    periodic_x = config["boundary"] == "periodic_EW"
    periodic_y = config["boundary"] == "periodic_NS"
    outlet_depth = config["outlet_depth"]
    outlet_sink = (
        "" if outlet_depth is not None else
        "h_new -= dt * $ctx.outlet_discharge.get(0)$ / (dx * dx);"
    )
    outlet_constraint = (
        f"if ($ctx.grid.can_out(i)$) h[i] = {float(outlet_depth):.9e}f;"
        if outlet_depth is not None else ""
    )
    diagonal_args = ", const float* qxy, const float* qyx" if d8 else ""
    inlet = config["inlet"]
    inlet_args = ", const float* inlet_mask, const float* inlet_depth" if inlet else ""
    # a band (inlet) cell is a prescribed boundary: its depth is HELD at the
    # parent's value, so it never runs the continuity update (its faces are the
    # fixed-flux inlet, stamped by inertial_apply_inlet).
    inlet_hold = ("if (inlet_mask[i] != 0.0f) { h[i] = inlet_depth[i]; return; }"
                  if inlet else "")
    diagonal_flux = '''
            int nw = $ctx.grid.neighbour(i, 0)$;
            int ne = $ctx.grid.neighbour(i, 2)$;
            float q_nw = nw >= 0 ? qxy[nw] : 0.0f;
            float q_ne = ne >= 0 ? qyx[ne] : 0.0f;
            divergence += 0.5f * dx * (q_nw + q_ne - qxy[i] - qyx[i]);''' if d8 else ""
    return KernelBuilder(
        f'''extern "C" __global__ void inertial_update_depth(
                float* h, const float* qx, const float* qy{diagonal_args}{inlet_args}) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            if (!$ctx.grid.is_active(i)$) {{
                h[i] = 0.0f;
                return;
            }}
            {inlet_hold}
            int x = i % {nx};
            int y = i / {nx};
            float dx = $ctx.grid.DX.get(0)$;
            float q_w = qx[y * ({nx} + 1) + x];
            float q_e = (x == {nx - 1} && {int(periodic_x)})
                ? qx[y * ({nx} + 1)] : qx[y * ({nx} + 1) + x + 1];
            float q_n = qy[y * {nx} + x];
            float q_s = (y == {ny - 1} && {int(periodic_y)})
                ? qy[x] : qy[(y + 1) * {nx} + x];
            float divergence = dx * (q_w - q_e + q_n - q_s);
            {diagonal_flux}
            float dt = $ctx.dt.get(0)$;
            float h_new = h[i] + dt * (divergence / (dx * dx))
                + dt * $ctx.rainfall.get(i)$;
            if (y == 0) {{
                h_new += dt * $ctx.top_inflow.get(0)$ / (dx * dx);
            }}
            if ($ctx.grid.can_out(i)$) {{
                {outlet_sink}
            }}
            h[i] = fmaxf(h_new, $ctx.min_depth.get(0)$);
            {outlet_constraint}
        }}''',
        domain=n,
    ).compose("grid", bundles["grid"]).freeze()


def _apply_inlet_factory(be: Backend, bundles, config):
    """Stamp the prescribed inlet fluxes onto every face touching a band (inlet)
    cell, AFTER the flux + regulator passes and BEFORE continuity, so the depth
    update sees exactly the parent's qx/qy across the band. No-op when inlet is
    off. This is the per-face fixed-flux inlet BC: a band cell contributes its
    parent-computed momentum to the child, nothing else."""
    _cupy_only(be)
    if not config["inlet"]:
        return KernelBuilder(
            'extern "C" __global__ void inertial_apply_inlet() {}', domain=1,
        ).freeze()
    nx, ny = int(config["nx"]), int(config["ny"])
    n = nx * ny
    d8 = config["topology"] == "D8"
    left_k, up_k = (3, 1) if d8 else (1, 0)
    return KernelBuilder(
        f'''extern "C" __global__ void inertial_apply_inlet(
                float* qx, float* qy, const float* qx_fix, const float* qy_fix,
                const float* inlet_mask, const float* flux_mask) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            if (!$ctx.grid.is_active(i)$) return;
            int x = i % {nx};
            int y = i / {nx};
            int fx = y * ({nx} + 1) + x;
            int fy = y * {nx} + x;
            // a face is stamped if either side is a prescribed-flux cell -- band
            // (inlet_mask, depth also HELD) or lateral source (flux_mask, depth
            // FREE to relax from its Manning seed).
            bool me = inlet_mask[i] != 0.0f || flux_mask[i] != 0.0f;
            int west = $ctx.grid.neighbour(i, {left_k})$;
            bool wbc = west >= 0 && (inlet_mask[west] != 0.0f || flux_mask[west] != 0.0f);
            if (me || wbc) qx[fx] = qx_fix[fx];
            if (me && x == {nx - 1}) qx[fx + 1] = qx_fix[fx + 1];
            int north = $ctx.grid.neighbour(i, {up_k})$;
            bool nbc = north >= 0 && (inlet_mask[north] != 0.0f || flux_mask[north] != 0.0f);
            if (me || nbc) qy[fy] = qy_fix[fy];
            if (me && y == {ny - 1}) qy[fy + {nx}] = qy_fix[fy + {nx}];
        }}''',
        domain=n,
    ).compose("grid", bundles["grid"]).freeze()


def build_inertial_flood_program() -> type:
    """Build a compact D4/D8 local-inertia flood program for CuPy.

    ``boundary``, ``outlet``, and ``nodata`` use the shared grid conventions.
    ``outlet_depth`` is an optional constant Dirichlet water depth applied to
    every grid outlet, whether outlets are edge-derived or mask-defined.
    ``regulator`` selects whether the complete provisional Bates flux is
    spatially filtered before the continuity step. ``"q_upwind"`` takes its
    neighbour from the upstream parallel link; ``"q_centered"`` averages both
    parallel links. ``regulator_theta`` is the provisional-flux weight, so
    one exactly recovers Bates at one and lower values add more diffusion.
    ``"s_upwind"``/``"s_centered"`` reuse those two constructions but replace
    ``regulator_theta`` with an adaptive per-interface weight derived from the
    local flow depth and discharge (Sridharan et al., 2020); ``regulator_theta``
    is then unused.
    ``boundary_flux`` is a uniform outward unit discharge on eligible normal
    grid-edge links. ``outlet_discharge`` is an optional per-outlet-cell
    external sink in m3/s, useful for interior cells selected by
    ``outlet='mask'``. ``top_inflow`` is a uniform m3/s source applied only
    to active cells in the top row. All three are zero by default.
    """
    b = ProgramBuilder("InertialFloodProgram")
    b.dim("ny").dim("nx")
    b.config("ny").config("nx").config("dx", default=1.0)
    b.config("topology", choices=("D4", "D8"), default="D4")
    b.config("boundary", choices=("normal", "periodic_EW", "periodic_NS"),
             default="normal")
    b.config("outlet", choices=("edge", "mask"), default="edge")
    b.config("nodata", choices=(False, True), default=False)
    b.config("outlet_depth", default=None)
    b.config("regulator",
             choices=("bates", "q_upwind", "q_centered", "s_upwind", "s_centered"),
             default="bates")
    # per-face fixed-flux inlet BC: band cells whose faces carry a prescribed
    # qx/qy (e.g. an already-solved upstream tile's flux) and whose depth is held.
    b.config("inlet", choices=(False, True), default=False)

    shape = (Dim("ny"), Dim("nx"))
    b.param("dt", "scalar", "f32", value=1.0e-3)
    b.param("rainfall", "auto", "f32", value=0.0, shape=shape)
    b.param("manning", "scalar", "f32", value=0.033)
    b.param("gravity", "scalar", "f32", value=9.81)
    b.param("froude_limit", "scalar", value=1.0, dtype="f32")
    b.param("transfer_fraction", "scalar", "f32", value=0.2)
    b.param("regulator_theta", "scalar", "f32", value=0.9)
    b.param("min_flow_depth", "scalar", "f32", value=1.0e-4)
    b.param("min_depth", "scalar", "f32", value=0.0)
    b.param("boundary_flux", "scalar", "f32", value=0.0)
    b.param("outlet_discharge", "scalar", "f32", value=0.0)
    b.param("top_inflow", "scalar", "f32", value=0.0)

    b.data("z", "f32", shape, role="input", shape_source=True)
    b.data("h", "f32", shape, role="state")
    b.data("qx", "f32", (Dim("ny"), Dim("nx") + 1), role="state")
    b.data("qy", "f32", (Dim("ny") + 1, Dim("nx")), role="state")
    b.data("qxy", "f32", _diagonal_shape, role="state")
    b.data("qyx", "f32", _diagonal_shape, role="state")
    b.data("qx_reg", "f32", _regulated_qx_shape, role="internal")
    b.data("qy_reg", "f32", _regulated_qy_shape, role="internal")
    b.data("qxy_reg", "f32", _regulated_diagonal_shape, role="internal")
    b.data("qyx_reg", "f32", _regulated_diagonal_shape, role="internal")
    # inlet BC fields (allocated only when inlet=True): band mask, held depth,
    # and the prescribed staggered face fluxes.
    b.data("inlet_mask", "f32", _inlet_cell_shape, role="input")
    b.data("inlet_depth", "f32", _inlet_cell_shape, role="input")
    b.data("qx_fix", "f32", _inlet_qx_shape, role="input")
    b.data("qy_fix", "f32", _inlet_qy_shape, role="input")
    # lateral-source mask: faces get their prescribed flux stamped like the band,
    # but the depth is NOT held (it relaxes from a Manning warm-start seed).
    b.data("flux_mask", "f32", _inlet_cell_shape, role="input")

    def grid_structure(be, *, topology, boundary, outlet, nodata, **_):
        _cupy_only(be)
        return make_grid_group(be, topology=topology, boundary=boundary,
                               outlet=outlet, nodata=nodata)

    def grid_parameters(be, pool, *, nx, ny, dx, topology, outlet, nodata, **_):
        if float(dx) <= 0.0:
            raise ValueError("InertialFloodProgram requires dx > 0")
        return make_grid_parameters(be, pool, nx, ny, dx, topology=topology,
                                    outlet=outlet, nodata=nodata)

    b.bundle("grid", grid_structure, grid_parameters, dims=("nx", "ny"),
             config=("dx", "topology", "boundary", "outlet", "nodata"))

    reset_values = {"h": "h", "qx": "qx", "qy": "qy", "qxy": "qxy", "qyx": "qyx"}
    flux_values = {
        "z": "z", "h": "h", "qx": "qx", "qy": "qy", "qxy": "qxy", "qyx": "qyx",
        "qx_reg": "qx_reg", "qy_reg": "qy_reg", "qxy_reg": "qxy_reg", "qyx_reg": "qyx_reg",
        "dt": "dt", "gravity": "gravity", "manning": "manning",
        "froude_limit": "froude_limit", "transfer_fraction": "transfer_fraction",
        "min_flow_depth": "min_flow_depth", "boundary_flux": "boundary_flux",
    }
    depth_values = {
        "h": "h", "qx": "qx", "qy": "qy", "qxy": "qxy", "qyx": "qyx",
        "qx_reg": "qx_reg", "qy_reg": "qy_reg", "qxy_reg": "qxy_reg", "qyx_reg": "qyx_reg",
        "dt": "dt", "rainfall": "rainfall", "min_depth": "min_depth",
        "outlet_discharge": "outlet_discharge", "top_inflow": "top_inflow",
        "inlet_mask": "inlet_mask", "inlet_depth": "inlet_depth",
    }
    regulate_values = {
        "qx": "qx", "qy": "qy", "qxy": "qxy", "qyx": "qyx",
        "qx_reg": "qx_reg", "qy_reg": "qy_reg", "qxy_reg": "qxy_reg", "qyx_reg": "qyx_reg",
        "regulator_theta": "regulator_theta",
        "h": "h", "dt": "dt", "gravity": "gravity",
    }
    apply_inlet_values = {
        "qx": "qx", "qy": "qy", "qx_fix": "qx_fix", "qy_fix": "qy_fix",
        "inlet_mask": "inlet_mask", "flux_mask": "flux_mask",
    }
    for topology in ("D4", "D8"):
        b.add(f"reset_{topology}", _reset_factory,
              bind=lambda f, be, values=reset_values: _grid_plan(f, values))
        b.add(f"update_flux_{topology}", _flux_factory,
              bind=lambda f, be, values=flux_values: _grid_plan(f, values))
        b.add(f"update_depth_{topology}", _depth_factory,
              bind=lambda f, be, values=depth_values: _grid_plan(f, values))
        b.add(f"regulate_flux_{topology}", _regulate_flux_factory,
              bind=lambda f, be, values=regulate_values: _grid_plan(f, values))
        b.add(f"apply_inlet_{topology}", _apply_inlet_factory,
              bind=lambda f, be, values=apply_inlet_values: _grid_plan(f, values))
    b.dispatch("reset", on="topology", cases={"D4": "reset_D4", "D8": "reset_D8"})
    b.dispatch("update_flux", on="topology", cases={"D4": "update_flux_D4", "D8": "update_flux_D8"})
    b.dispatch("update_depth", on="topology", cases={"D4": "update_depth_D4", "D8": "update_depth_D8"})
    b.dispatch("regulate_flux", on="topology", cases={"D4": "regulate_flux_D4", "D8": "regulate_flux_D8"})
    b.dispatch("apply_inlet", on="topology", cases={"D4": "apply_inlet_D4", "D8": "apply_inlet_D8"})
    b.pipeline("step", ("update_flux", "regulate_flux", "apply_inlet", "update_depth"))

    program = b.freeze()
    program.outlet_mask = property(
        lambda self: GridMaskAccessor(self, "OUTLET_MASK", "outlet='mask'"),
    )
    program.nodata_mask = property(
        lambda self: GridMaskAccessor(self, "NODATA_MASK", "nodata=True"),
    )
    return program


InertialFloodProgram = build_inertial_flood_program()

__all__ = ["InertialFloodProgram", "build_inertial_flood_program"]
