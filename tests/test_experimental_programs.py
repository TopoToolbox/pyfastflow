"""Device-free contract checks for experimental Program recipes."""

from pyfastflow.noise import PerlinNoiseProgram
from pyfastflow.graphflood import (
    GraphFloodParticles, GraphFloodRelax, GraphFloodVanilla,
)
from pyfastflow.flow import MFDFlowProgram, SFDFlowProgram
from pyfastflow.visu import HillshadeProgram
from pyfastflow.experimental.golem import GolemNoSedProgram, GolemSedProgram


def test_perlin_program_public_recipe():
    recipe = PerlinNoiseProgram._recipe
    assert recipe.name == "PerlinNoiseProgram"
    assert set(recipe.dims) == {"nx", "ny"}
    assert set(recipe.sequences) == {"generate", "add_southward_slope"}
    assert recipe.data["noise"].role == "output"


def test_sfd_program_high_level_choices():
    recipe = SFDFlowProgram._recipe
    assert recipe.config["local_minima"].choices == (
        "none", "reconstruct_epsilon", "cordonnier_carve", "cordonnier_jump",
    )
    assert recipe.config["accumulation"].choices == (
        "rake_compress", "pointer_jump_push", "pj",
    )
    assert set(recipe.dispatch["resolve_minima"].cases) == set(recipe.config["local_minima"].choices)
    assert set(recipe.dispatch["accumulate"].cases) == set(recipe.config["accumulation"].choices)
    assert recipe.data["z"].shape_source
    assert {"nx", "ny"} <= set(recipe.config)
    assert recipe.data["rec"].role == "output"
    assert recipe.data["drainage"].role == "output"


def test_mfd_program_high_level_choices():
    recipe = MFDFlowProgram._recipe
    minima = (
        "none", "reconstruct_epsilon", "cordonnier_carve",
        "fill_cordonnier",
    )
    assert recipe.config["local_minima"].choices == minima
    assert set(recipe.dispatch["resolve_minima"].cases) == set(minima)
    assert set(recipe.dispatch["prepare_mfd_surface"].cases) == set(minima)
    assert set(recipe.dispatch["build_topology"].cases) == set(minima)
    assert recipe.data["z"].shape_source
    assert recipe.data["drainage"].role == "output"
    assert recipe.config["quantized_weight"].default is True
    assert set(recipe.sequences) >= {
        "route", "snapshot_receivers", "compute_rank",
        "compute_cordonnier_fill", "resolve_none",
        "resolve_reconstruct_epsilon", "resolve_carve",
        "topology_none", "topology_reconstruct",
        "topology_rank", "topology_fill", "accumulate",
    }


def test_hillshade_program_public_recipe():
    recipe = HillshadeProgram._recipe
    assert recipe.config["method"].choices == ("hillshade", "multishade")
    assert set(recipe.dispatch["render"].cases) == {"hillshade", "multishade"}
    assert recipe.data["z"].role == "input"
    assert recipe.data["image"].role == "output"


_TOPOLOGY = (
    "make_surface", "route_local_minima", "snapshot_local_minima",
    "resolve_minima", "prepare_mfd_surface", "refresh_hydraulic_surface",
    "build_topology", "prepare_frontier", "accumulate",
)


def test_graphflood_programs_share_recipe():
    for program in (GraphFloodVanilla, GraphFloodRelax, GraphFloodParticles):
        recipe = program._recipe
        assert recipe.name == program.__name__
        assert {"dx", "topology", "boundary", "outlet", "nodata",
                "friction_law", "mfd_local_minima"} <= set(recipe.config)
        assert recipe.config["mfd_local_minima"].choices == (
            "rank_cordonnier", "fill_cordonnier", "carve_cordonnier",
            "reconstruct_epsilon",
        )
        assert recipe.config["mfd_local_minima"].default == "carve_cordonnier"
        assert "quantized_weight" not in recipe.config
        assert recipe.params["precipitation"].mode == "auto"
        assert {"friction_coefficient", "friction_exponent",
                "carve_slope_min"} <= set(recipe.params)
        assert recipe.data["z"].shape_source
        assert recipe.data["h"].role == "state"
        assert recipe.data["Qi"].role == recipe.data["Qo"].role == "output"


def test_graphflood_vanilla_recipe():
    recipe = GraphFloodVanilla._recipe
    assert recipe.params["dt"].mode == "auto"
    assert recipe.pipelines["run"].steps == _TOPOLOGY + ("update_depth",)
    assert recipe.pipelines["run_transient"].steps == (
        "make_surface", "transport_transient",
    )


def test_graphflood_relax_recipe():
    recipe = GraphFloodRelax._recipe
    assert recipe.config["analytical_solver"].choices == ("local", "bottom_up")
    assert {"relaxation", "warmup_relaxation", "depth_cap"} <= set(
        recipe.params)
    assert recipe.pipelines["run"].steps == _TOPOLOGY + ("update_depth",)
    assert recipe.pipelines["_warmup_pass"].steps == _TOPOLOGY + (
        "update_depth_capped",
    )


def test_graphflood_particles_recipe():
    recipe = GraphFloodParticles._recipe
    assert {"n_particles", "relaxation", "propagate", "spawn_pad",
            "walk_steps", "source_percentile", "warmup_relaxation",
            "depth_cap"} <= set(recipe.params)
    assert recipe.params["propagate"].value == 0.5


def test_golem_nosed_program_public_recipe():
    recipe = GolemNoSedProgram._recipe
    assert recipe.name == "GolemNoSedProgram"
    assert recipe.config["topology"].choices == ("D4", "D8")
    assert recipe.config["local_minima"].choices == (
        "none", "reconstruct_epsilon", "cordonnier_carve", "cordonnier_jump",
    )
    assert recipe.data["z"].role == "state"
    assert recipe.data["z"].shape_source
    assert {"rec", "drainage_area", "erosion_rate"} <= set(recipe.data)
    assert recipe.pipelines["run_n_step"].steps == (
        "apply_uplift", "restore_outlet_z", "route", "resolve_minima",
        "accumulate", "scale_drainage_area", "fluvial", "restore_outlet_z",
        "hillslope", "restore_outlet_z",
    )


def test_golem_sed_program_public_recipe():
    recipe = GolemSedProgram._recipe
    assert recipe.name == "GolemSedProgram"
    assert recipe.config["topology"].choices == ("D4", "D8")
    assert recipe.config["hillslope_model"].choices == ("none", "linear_implicit")
    assert recipe.config["hillslope_model"].default == "none"
    assert recipe.config["fluvial_model"].choices == (
        "space", "shared_stream_power",
    )
    assert {"detachment_erodibility", "transport_erodibility"} <= set(recipe.params)
    assert recipe.data["z"].role == "state"
    assert recipe.data["sediment_thickness"].role == "state"
    assert {
        "rock_erosion_rate", "sediment_entrainment_rate",
        "sediment_source_rate", "sediment_flux", "deposition_rate",
        "sediment_export_rate",
    } <= set(recipe.data)
