"""Conditioning, topology, and accumulation steps for the MFD Program."""

from pyfastflow.core import HostBlockBuilder, KernelBuilder, SequenceBuilder
from pyfastflow.flow import (
    make_accumulation, make_depression_solver, make_depressions,
    make_mfd_topology, make_receivers,
)
from pyfastflow.flow._program_cupy import BLOCK, _cupy_only, _grid_leaf_plan


def _cordonnier_factory(reroute):
    def factory(be, bundles, config):
        _cupy_only(be)
        n = config["nx"] * config["ny"]
        deps = make_depressions(
            be, bundles["grid"], bundles.param("ndep"), method="optimized",
            reroute=reroute, n_flat=n,
        )
        return make_depression_solver(
            be, deps, bundles.bundle_params("grid"), method="optimized",
            reroute=reroute, n_flat=n, block_size=BLOCK,
        )[0]

    return factory

def _route_factory(be, bundles, _config):
    _cupy_only(be)
    return make_receivers(
        be, bundles["grid"], topology="D8", mode="steepest"
    )["receivers"]

def _snapshot_factory(be, bundles, config):
    _cupy_only(be)
    return make_mfd_topology(
        be, bundles["grid"], method="cordonnier_rank",
        n_flat=config["nx"] * config["ny"], topology="D8",
        diagonal_partition_correction=True,
        quantized_weight=config["quantized_weight"],
    )["snapshot_receivers"]

def _rank_factory(be, bundles, config):
    _cupy_only(be)
    return make_mfd_topology(
        be, bundles["grid"], method="cordonnier_rank",
        n_flat=config["nx"] * config["ny"], topology="D8",
        diagonal_partition_correction=True,
        quantized_weight=config["quantized_weight"],
    )["receiver_rank"]

def _cordonnier_fill_factory(be, bundles, config):
    _cupy_only(be)
    return make_mfd_topology(
        be, bundles["grid"], method="cordonnier_fill",
        n_flat=config["nx"] * config["ny"], topology="D8",
        diagonal_partition_correction=True,
        quantized_weight=config["quantized_weight"],
    )["receiver_fill"]

def _rank_plan(_frozen, _be):
    return {
        "init.rec": "rec", "init.ancestor": "rank_ancestor", "init.rank": "rank",
        "forward.ancestor_in": "rank_ancestor", "forward.rank_in": "rank",
        "forward.ancestor_out": "rank_ancestor_alt", "forward.rank_out": "rank_alt",
        "backward.ancestor_in": "rank_ancestor_alt", "backward.rank_in": "rank_alt",
        "backward.ancestor_out": "rank_ancestor", "backward.rank_out": "rank",
    }

def _cordonnier_fill_plan(_frozen, _be):
    return {
        "init.rec": "rec", "init.z": "z",
        "init.ancestor": "rank_ancestor", "init.rank": "rank",
        "init.filled": "filled",
        "forward.ancestor_in": "rank_ancestor",
        "forward.rank_in": "rank", "forward.filled_in": "filled",
        "forward.ancestor_out": "rank_ancestor_alt",
        "forward.rank_out": "rank_alt", "forward.filled_out": "z_prime",
        "backward.ancestor_in": "rank_ancestor_alt",
        "backward.rank_in": "rank_alt", "backward.filled_in": "z_prime",
        "backward.ancestor_out": "rank_ancestor",
        "backward.rank_out": "rank", "backward.filled_out": "filled",
    }

def _raw_topology_factory(be, bundles, config):
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    clear = KernelBuilder(
        f'''extern "C" __global__ void clear_flat_distance(float* dist) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < {n}) dist[i] = 0.0f;
        }}''',
        domain=n,
    ).freeze()
    topology = make_mfd_topology(
        be, bundles["grid"], method="surface", n_flat=n, topology="D8",
        diagonal_partition_correction=True,
        quantized_weight=config["quantized_weight"],
    )
    sb = SequenceBuilder()
    sb.add("clear_distance", clear)
    sb.add("dirs_weights", topology["dirs_weights"])
    sb.add("indegree_reset", topology["indegree_reset"])
    sb.add("indegree_count", topology["indegree_count"])
    sb.step("clear_distance").step("dirs_weights")
    sb.step("indegree_reset").step("indegree_count")
    return sb.freeze()

def _surface_topology_factory(be, bundles, config):
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    topology = make_mfd_topology(
        be, bundles["grid"], method="surface", n_flat=n, topology="D8",
        diagonal_partition_correction=True,
        quantized_weight=config["quantized_weight"],
    )
    sb = SequenceBuilder()
    for name in ("dirs_weights", "indegree_reset", "indegree_count"):
        sb.add(name, topology[name]).step(name)
    return sb.freeze()

def _rank_topology_factory(be, bundles, config):
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    topology = make_mfd_topology(
        be, bundles["grid"], method="cordonnier_rank", n_flat=n,
        topology="D8", diagonal_partition_correction=True,
        quantized_weight=config["quantized_weight"],
    )
    sb = SequenceBuilder()
    for name in ("dirs_weights", "indegree_reset", "indegree_count"):
        sb.add(name, topology[name]).step(name)
    return sb.freeze()

def _fill_topology_factory(be, bundles, config):
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    topology = make_mfd_topology(
        be, bundles["grid"], method="cordonnier_fill", n_flat=n,
        topology="D8", diagonal_partition_correction=True,
        quantized_weight=config["quantized_weight"],
    )
    sb = SequenceBuilder()
    for name in ("dirs_weights", "indegree_reset", "indegree_count"):
        sb.add(name, topology[name]).step(name)
    return sb.freeze()

def _topology_plan(kind):
    def plan(frozen, _be):
        values = {
            "dist": "epsilon_distance", "dirs": "directions",
            "mfd_w": "weights", "indegree": "indegree",
        }
        if kind == "raw":
            values.update({"filled": "z"})
        elif kind == "surface":
            values.update({"filled": "filled"})
        elif kind == "fill":
            values.update({"filled": "filled", "rank": "rank"})
        else:
            values.update({
                "z": "z", "rec_initial": "rec_initial", "rec": "rec",
                "rank": "rank",
            })
        return _grid_leaf_plan(frozen, values)

    return plan

def _accumulation_factory(be, bundles, config):
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    parts = make_accumulation(
        be, bundles["grid"], method="persistent_mfd", n_flat=n,
        n_neighbours=8,
        quantized_weight=config["quantized_weight"],
    )

    def prepare_frontier(ctx, indegree, frontier0, count, barrier):
        from pyfastflow.flow._cupy_mfd_accum import init_frontier_mfd

        ready = init_frontier_mfd(indegree.array, frontier0.array)
        count.array[0] = ready
        count.array[1] = 0
        barrier.array[0] = 0

    prepare = HostBlockBuilder(prepare_frontier).freeze()
    sb = SequenceBuilder()
    sb.add("q_init", parts["q_init"])
    sb.add("prepare_frontier", prepare)
    sb.add("accum", parts["accum"])
    sb.step("q_init").step("prepare_frontier").step("accum")
    return sb.freeze()

def _accumulation_plan(frozen, _be):
    return _grid_leaf_plan(frozen, {
        "SOURCE": "source", "accum": "drainage", "dirs": "directions",
        "mfd_w": "weights", "indegree": "indegree",
        "frontier0": "mfd_frontier0", "frontier1": "mfd_frontier1",
        "count": "mfd_count", "barrier": "mfd_barrier",
    })
