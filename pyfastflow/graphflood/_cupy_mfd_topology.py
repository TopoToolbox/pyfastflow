"""
cupy-only MFD topology construction for make_graphflood's kind="vanilla_mfd" -
computes the per-node downslope receiver bitmask (`dirs`), normalized
slope-proportional weights (`mfd_w`) and indegree ../flow's persistent_mfd
accumulation (_cupy_mfd_accum.py) needs to run over an already-built MFD
graph, from a filled elevation surface. Unported anywhere else in the
package before this - ../../CLAUDE.md's own state notes flag this as the
one piece "still entirely unported": legacy's only MFD code
(pyfastflow/flow/flow_mfd_kernels.py) is a completely different Jacobi
power-iteration scheme with dense per-node routing_weights and no bitmask/
indegree at all, so this is new work, not a port.

cupy-only, no closure-backend equivalent, for the same structural reason
_cupy_mfd_accum.py itself is cupy-only: persistent_mfd's grid-wide barrier
needs raw CUDA primitives no closure-backend kernel model expresses - the
topology this module builds only exists to feed that accumulation kernel,
so there is no reason for a closure-backend variant to exist independently.

`filled` is the caller-supplied elevation surface this operates on -
make_graphflood's kind="vanilla_mfd" always passes the fill_reconstruct
surface (the resolved, monotonic z+h fill), never bare z or z+h directly:
MFD's per-node weight split needs every node to have somewhere to send
water, which an unresolved depression does not guarantee. `dist` is the
companion per-cell perturbation ../graphflood/_cupy_reconstruct_epsilon.py
accumulates along the `parent` chain - a strictly-toward-the-outlet, ULP-
scaled tie-break carrier. The slope helper's own second additive term (its
`h` argument) is exactly where it belongs:
`slope(filled[i], dist[i], filled[j], dist[j], k)` computes
`((filled[i] - filled[j]) + (dist[i] - dist[j])) / d` - the real relief
drop plus the perturbation drop, in one call. Inside a resolved depression
the `filled` drop is exactly 0 for every neighbour pair (a real lake
surface IS flat), and the `dist` drop - evaluated at its own ~1e-7
magnitude, never folded up into `filled`'s magnitude where it would round
away - is what gives every flat cell a downslope edge toward the outlet.
On genuine relief the `dist` drop is negligible against the real drop, so
it perturbs neither the direction set nor the weights there.

A can_out node gets `dirs[i] = 0` (no outgoing directions at all, mask
never set) and an all-zero `mfd_w` row - the same "this is where routing
stops" role `rec[i] = i` plays for SFD receivers, just expressed as "sends
nowhere" instead of "sends to itself" (persistent_mfd's own frontier/
indegree walk has no self-loop concept to exploit the SFD convention with).

Author: B.G (08/2026)
"""

from ..core.context.builder import KernelBuilder
from ..core.context.frozen import FrozenKernel
from ..core.context.slot import SlotKind
from ..core.pool.base import new_uid
from ..flow._cupy_receivers import build_distance_slope_helpers


def _find_param_paths(frozen, leaf_name: str, prefix: tuple = ()) -> list:
    """Every relative dotted path under `frozen`'s composed subtree whose PARAM slot is named `leaf_name`."""
    paths = []
    if leaf_name in frozen.slots.names(SlotKind.PARAM):
        paths.append(".".join(prefix + (leaf_name,)))
    for name, child in frozen.composed.items():
        paths.extend(_find_param_paths(child, leaf_name, prefix + (name,)))
    return paths


def _share_leaf(builder, canonical: str) -> None:
    """Declare every occurrence of PARAM `canonical` in `builder`'s composed subtree shared with its own top-level slot."""
    paths = []
    for name, child in builder.composed.items():
        paths.extend(_find_param_paths(child, canonical, (name,)))
    if paths:
        builder.share(canonical, *paths)


def build_mfd_topology(*, grid, n_flat: int, topology: str, diagonal_partition_correction: bool) -> dict:
    """
    Three FrozenKernels: "dirs_weights" (data args (filled, dist, dirs,
    mfd_w): the bitmask/weight computation described in the module
    docstring),
    "indegree_reset" (data arg (indegree,): indegree[i] = 0 - must run
    before "indegree_count" every call, dirs/mfd_w/indegree all being
    recomputed fresh every GraphFlood step as the surface evolves) and
    "indegree_count" (data args (dirs, indegree): atomic-adds 1 into
    indegree[neighbour] for every bit set in dirs[i], via
    `ctx.grid.neighbour_raw` - trusted the same way _cupy_mfd_accum.py's
    own persistent kernel trusts it, since every bit `dirs` sets already
    passed a bounds-checked `ctx.grid.neighbour` in "dirs_weights").

    A caller runs all three, in order, every step, then
    `init_frontier_mfd` (../flow/_cupy_mfd_accum.py, host-side) to compact
    the zero-indegree cells into a frontier before launching
    `build_persistent_mfd`'s "accum" kernel.

    Parameters
    ----------
    grid : FrozenGroup
    n_flat : int
    topology : str
        "D4" or "D8" - sizes `mfd_w` (`n_flat * n_neighbours`) at the
        caller's own allocation, not here; only affects `slope`'s diagonal
        correction.
    diagonal_partition_correction : bool

    Returns
    -------
    dict
        {"dirs_weights": FrozenKernel, "indegree_reset": FrozenKernel,
        "indegree_count": FrozenKernel}.

    Author: B.G (08/2026)
    """
    slope = build_distance_slope_helpers(
        grid, topology=topology, diagonal_partition_correction=diagonal_partition_correction
    )["slope_from_values_k"]
    t = f"gfm{new_uid()}"

    dirs_weights_body = f"""
extern "C" __global__ void {t}_mfd_dirs_weights(const float* filled, const float* dist, unsigned char* dirs, float* mfd_w) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    int nk = $ctx.grid.N_NEIGHBOURS.get(0)$;
    unsigned char mask = 0;
    float slopes[8];
    float sum_s = 0.0f;
    if (!$ctx.grid.can_out(i)$) {{
        for (int k = 0; k < nk; k++) {{
            int j = $ctx.grid.neighbour(i, k)$;
            float s = 0.0f;
            if (j != -1) {{
                s = $ctx.slope(filled[i], dist[i], filled[j], dist[j], k)$;
                if (s < 0.0f) s = 0.0f;
            }}
            slopes[k] = s;
            if (s > 0.0f) {{ mask |= (1 << k); sum_s += s; }}
        }}
    }}
    for (int k = 0; k < nk; k++) {{
        mfd_w[i * nk + k] = sum_s > 0.0f ? slopes[k] / sum_s : 0.0f;
    }}
    dirs[i] = mask;
}}
"""

    dirs_weights_kb = KernelBuilder()
    grid_param_names = grid.slots.names(SlotKind.PARAM)
    for name in grid_param_names:
        dirs_weights_kb.wire_param(name)
    dirs_weights_kb.compose("grid", grid).compose("slope", slope)
    dirs_weights_kb.wire_data("filled").wire_data("dist").wire_data("dirs").wire_data("mfd_w")
    for name in grid_param_names:
        _share_leaf(dirs_weights_kb, name)

    indegree_reset: FrozenKernel = (
        KernelBuilder()
        .wire_data("indegree")
        .ingest(
            f"""
extern "C" __global__ void {t}_mfd_indegree_reset(int* indegree) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    indegree[i] = 0;
}}
"""
        )
    )

    indegree_count_kb = KernelBuilder()
    for name in grid_param_names:
        indegree_count_kb.wire_param(name)
    indegree_count_kb.compose("grid", grid)
    indegree_count_kb.wire_data("dirs").wire_data("indegree")
    for name in grid_param_names:
        _share_leaf(indegree_count_kb, name)
    indegree_count_body = f"""
extern "C" __global__ void {t}_mfd_indegree_count(const unsigned char* dirs, int* indegree) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    unsigned char mask = dirs[i];
    int nk = $ctx.grid.N_NEIGHBOURS.get(0)$;
    for (int k = 0; k < nk; k++) {{
        if (!(mask & (1 << k))) continue;
        int j = $ctx.grid.neighbour_raw(i, k)$;
        atomicAdd(&indegree[j], 1);
    }}
}}
"""

    return {
        "dirs_weights": dirs_weights_kb.ingest(dirs_weights_body),
        "indegree_reset": indegree_reset,
        "indegree_count": indegree_count_kb.ingest(indegree_count_body),
    }
