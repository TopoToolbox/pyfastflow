"""Shared CuPy Program factories for grid and reconstruction steps."""

import math

from pyfastflow.core import Backend, HostBlockBuilder, KernelBuilder, SequenceBuilder
from pyfastflow.flow import make_fill_reconstruct
from pyfastflow.flow._cupy_reconstruct_epsilon import build_hops_init, build_hops_jump

BLOCK = 256


def _cupy_only(be: Backend) -> None:
    if be.name != "cupy":
        raise ValueError(f"this program is cupy-only, got {be.name!r}")

def _noop_factory(be, _bundles, _config):
    """Return an explicit no-op so ``none`` remains a normal dispatch case."""
    _cupy_only(be)

    def noop(ctx):
        pass

    return HostBlockBuilder(noop).freeze()

def _grid_leaf_plan(frozen, values):
    """Make an exact Program bind map from unique leaf names plus grid leaves."""
    bound = frozen.build()
    try:
        plan = {}
        for address in bound.addresses():
            if len(address) >= 2 and address[-2] == "grid":
                plan[".".join(address)] = "grid"
            elif address[-1] in values:
                plan[".".join(address)] = values[address[-1]]
        return plan
    finally:
        bound.close()

def _reconstruct_epsilon_factory(be, bundles, config):
    """Build reconstruction, epsilon distance, and parent-to-rec as one sequence."""
    _cupy_only(be)
    nx, ny = config["nx"], config["ny"]
    n = nx * ny
    max_passes = 4 * max(nx, ny)
    if n < max_passes + 2:
        raise ValueError("reconstruct_epsilon requires nx*ny >= 4*max(nx, ny)+2")

    deps = make_fill_reconstruct(be, bundles["grid"], nx=nx, ny=ny)
    pass_p, active_p = bundles.param("pass_index"), bundles.param("active")

    def zero_pass(ctx): ctx.P.set(0)
    def bump_pass(ctx): ctx.P.set(int(ctx.P.read()) + 1)
    def zero_active(ctx): ctx.ACTIVE.set(0)
    def converged(ctx): return int(ctx.ACTIVE.read()) == 0

    reset = KernelBuilder(f'''extern "C" __global__ void reset_reconstruct(
            int* counters, int* queued_gen) {{
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= {n}) return;
        counters[i] = 0;
        queued_gen[i] = -1;
    }}''', domain=n).freeze()
    copy_parent = KernelBuilder(f'''extern "C" __global__ void copy_parent(
            const int* parent, int* rec) {{
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < {n}) rec[i] = parent[i];
    }}''', domain=n).freeze()
    hops_init = build_hops_init(n_flat=n)
    hops_jump = build_hops_jump(n_flat=n)
    hops_rounds = math.ceil(math.log2(max(2, n))) + 1
    if hops_rounds % 2: hops_rounds += 1

    sb = SequenceBuilder()
    sb.add("reset", reset)
    for name in ("init_filled", "sweep_row_lr", "sweep_row_rl", "sweep_col_tb", "sweep_col_bt", "frontier_init"):
        sb.add(name, deps[name])
    sb.add("zero_pass", HostBlockBuilder(zero_pass).freeze())
    sb.add("zero_active", HostBlockBuilder(zero_active).freeze())
    sb.add("relax", deps["relax"])
    sb.add("bump_pass", HostBlockBuilder(bump_pass).freeze())
    sb.add("converged", HostBlockBuilder(converged).freeze())
    sb.add("hops_init", hops_init)
    sb.add("hops_forward", hops_jump)
    sb.add("hops_backward", hops_jump)
    sb.add("copy_parent", copy_parent)
    for name in ("reset", "init_filled", "sweep_row_lr", "sweep_row_rl", "sweep_col_tb", "sweep_col_bt", "frontier_init", "zero_pass"):
        sb.step(name)
    sb.loop(body=("zero_active", "relax", "bump_pass"), max_times=max_passes, until="converged")
    sb.step("hops_init")
    sb.loop(body=("hops_forward", "hops_backward"), max_times=hops_rounds // 2)
    sb.step("copy_parent")
    return sb.freeze()


def _reconstruct_epsilon_plan(frozen, _be):
    """Bind shared reconstruction scratch and its ping-pong distance buffers."""
    values = {
        "z": "z", "filled": "filled", "parent": "parent",
        "frontier": "frontier", "counters": "counters",
        "queued_gen": "queued_gen", "active": "active.handle", "P": "pass_index",
        "ACTIVE": "active", "dist": "epsilon_distance", "anc": "epsilon_ancestor",
        "dist_in": "epsilon_distance", "dist_out": "epsilon_distance_work",
        "anc_in": "epsilon_ancestor", "anc_out": "epsilon_ancestor_work",
        "rec": "rec",
    }
    plan = _grid_leaf_plan(frozen, values)
    plan.update({
        "hops_forward.dist_in": "epsilon_distance",
        "hops_forward.dist_out": "epsilon_distance_work",
        "hops_forward.anc_in": "epsilon_ancestor",
        "hops_forward.anc_out": "epsilon_ancestor_work",
        "hops_backward.dist_in": "epsilon_distance_work",
        "hops_backward.dist_out": "epsilon_distance",
        "hops_backward.anc_in": "epsilon_ancestor_work",
        "hops_backward.anc_out": "epsilon_ancestor",
    })
    return plan
