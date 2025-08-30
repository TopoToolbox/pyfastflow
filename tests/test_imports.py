"""
Import tests for all PyFastFlow modules and submodules.

These tests ensure that all modules can be imported without errors,
which is crucial for detecting import-related issues early.

Tests are marked with @pytest.mark.import for selective running.
"""
import pytest


class TestMainPackageImports:
    """Test imports for the main pyfastflow package."""

    @pytest.mark.importtest
    def test_main_package_import(self):
        """Test that the main pyfastflow package can be imported."""
        import pyfastflow
        assert hasattr(pyfastflow, '__version__') or hasattr(pyfastflow, '__all__')

    @pytest.mark.importtest
    def test_constants_import(self):
        """Test that constants module can be imported."""
        import pyfastflow.constants
        assert pyfastflow.constants is not None


class TestCLIImports:
    """Test imports for CLI modules."""

    @pytest.mark.importtest
    def test_cli_init_import(self):
        """Test CLI package import."""
        import pyfastflow.cli
        assert pyfastflow.cli is not None

    @pytest.mark.importtest
    def test_cli_raster_commands_import(self):
        """Test raster commands module import."""
        import pyfastflow.cli.raster_commands
        assert pyfastflow.cli.raster_commands is not None
        assert hasattr(pyfastflow.cli.raster_commands, 'raster2npy')


class TestErodepImports:
    """Test imports for erosion/deposition modules."""

    @pytest.mark.importtest
    def test_erodep_init_import(self):
        """Test erodep package import."""
        import pyfastflow.erodep
        assert pyfastflow.erodep is not None
        # Test that main functions are available
        assert hasattr(pyfastflow.erodep, 'SPL')
        assert hasattr(pyfastflow.erodep, 'block_uplift')

    @pytest.mark.importtest
    def test_erodep_spl_import(self):
        """Test SPL archive module import."""
        import pyfastflow.erodep.archive.SPL
        assert pyfastflow.erodep.archive.SPL is not None

    @pytest.mark.importtest
    def test_erodep_uplift_import(self):
        """Test uplift module import."""
        import pyfastflow.erodep.uplift
        assert pyfastflow.erodep.uplift is not None

    @pytest.mark.importtest
    def test_erodep_fluvial_deposition_import(self):
        """Test fluvial deposition module import."""
        import pyfastflow.erodep.fluvial_deposition
        assert pyfastflow.erodep.fluvial_deposition is not None

    @pytest.mark.importtest
    def test_erodep_spl_no_sed_import(self):
        """Test SPL no sediment module import."""
        import pyfastflow.erodep.spl_no_sed
        assert pyfastflow.erodep.spl_no_sed is not None

    @pytest.mark.importtest
    def test_erodep_spl_sed_import(self):
        """Test SPL with sediment module import."""
        import pyfastflow.erodep.spl_sed
        assert pyfastflow.erodep.spl_sed is not None


class TestFloodImports:
    """Test imports for flood modeling modules."""

    @pytest.mark.importtest
    def test_flood_init_import(self):
        """Test flood package import."""
        import pyfastflow.flood
        assert pyfastflow.flood is not None
        # Test main classes are available
        assert hasattr(pyfastflow.flood, 'Flooder')

    @pytest.mark.importtest
    def test_flood_gf_fields_import(self):
        """Test GraphFlood fields module import."""
        import pyfastflow.flood.gf_fields
        assert pyfastflow.flood.gf_fields is not None

    @pytest.mark.importtest
    def test_flood_gf_hydrodynamics_import(self):
        """Test GraphFlood hydrodynamics module import."""
        import pyfastflow.flood.gf_hydrodynamics
        assert pyfastflow.flood.gf_hydrodynamics is not None

    @pytest.mark.importtest
    def test_flood_gf_ls_import(self):
        """Test GraphFlood LisFlood module import."""
        import pyfastflow.flood.gf_ls
        assert pyfastflow.flood.gf_ls is not None


class TestFlowImports:
    """Test imports for flow routing modules."""

    @pytest.mark.importtest
    def test_flow_init_import(self):
        """Test flow package import."""
        import pyfastflow.flow
        assert pyfastflow.flow is not None
        # Test main API classes are available
        assert hasattr(pyfastflow.flow, 'FlowRouter')
        assert hasattr(pyfastflow.flow, 'GridField')

    @pytest.mark.importtest
    def test_flow_flowfields_import(self):
        """Test flowfields module import."""
        import pyfastflow.flow.flowfields
        assert pyfastflow.flow.flowfields is not None

    @pytest.mark.importtest
    def test_flow_receivers_import(self):
        """Test receivers module import."""
        import pyfastflow.flow.receivers
        assert pyfastflow.flow.receivers is not None

    @pytest.mark.importtest
    def test_flow_downstream_propag_import(self):
        """Test downstream propagation module import."""
        import pyfastflow.flow.downstream_propag
        assert pyfastflow.flow.downstream_propag is not None

    @pytest.mark.importtest
    def test_flow_fill_topo_import(self):
        """Test topology filling module import."""
        import pyfastflow.flow.fill_topo
        assert pyfastflow.flow.fill_topo is not None

    @pytest.mark.importtest
    def test_flow_lakeflow_import(self):
        """Test lake flow module import."""
        import pyfastflow.flow.lakeflow
        assert pyfastflow.flow.lakeflow is not None

    @pytest.mark.importtest
    def test_flow_level_acc_import(self):
        """Test level accumulation module import."""
        import pyfastflow.flow.level_acc
        assert pyfastflow.flow.level_acc is not None

    @pytest.mark.importtest
    def test_flow_f32_i32_struct_import(self):
        """Test f32/i32 struct module import."""
        import pyfastflow.flow.f32_i32_struct
        assert pyfastflow.flow.f32_i32_struct is not None


class TestGeneralAlgorithmsImports:
    """Test imports for general algorithms modules."""

    @pytest.mark.importtest
    def test_general_algorithms_init_import(self):
        """Test general algorithms package import."""
        import pyfastflow.general_algorithms
        assert pyfastflow.general_algorithms is not None
        # Test main functions are available
        assert hasattr(pyfastflow.general_algorithms, 'atan')

    @pytest.mark.importtest
    def test_math_utils_import(self):
        """Test math utilities module import."""
        import pyfastflow.general_algorithms.math_utils
        assert pyfastflow.general_algorithms.math_utils is not None

    @pytest.mark.importtest
    def test_parallel_scan_import(self):
        """Test parallel scan module import."""
        import pyfastflow.general_algorithms.parallel_scan
        assert pyfastflow.general_algorithms.parallel_scan is not None

    @pytest.mark.importtest
    def test_pingpong_import(self):
        """Test pingpong module import."""
        import pyfastflow.general_algorithms.pingpong
        assert pyfastflow.general_algorithms.pingpong is not None

    @pytest.mark.importtest
    def test_slope_tools_import(self):
        """Test slope tools module import."""
        import pyfastflow.general_algorithms.slope_tools
        assert pyfastflow.general_algorithms.slope_tools is not None

    @pytest.mark.importtest
    def test_util_taichi_import(self):
        """Test Taichi utilities module import."""
        import pyfastflow.general_algorithms.util_taichi
        assert pyfastflow.general_algorithms.util_taichi is not None


class TestGridImports:
    """Test imports for grid management modules."""

    @pytest.mark.importtest
    def test_grid_init_import(self):
        """Test grid package import."""
        import pyfastflow.grid
        assert pyfastflow.grid is not None

    @pytest.mark.importtest
    def test_grid_gridfields_import(self):
        """Test gridfields module import."""
        import pyfastflow.grid.gridfields
        assert pyfastflow.grid.gridfields is not None

    @pytest.mark.importtest
    def test_grid_neighbourer_flat_import(self):
        """Test neighbourer flat module import."""
        import pyfastflow.grid.neighbourer_flat
        assert pyfastflow.grid.neighbourer_flat is not None

    @pytest.mark.importtest
    def test_grid_hswrapper_import(self):
        """Test hillshading wrapper module import."""
        import pyfastflow.grid._hswrapper
        assert pyfastflow.grid._hswrapper is not None


class TestIOImports:
    """Test imports for I/O modules."""

    @pytest.mark.importtest
    def test_io_init_import(self):
        """Test I/O package import."""
        import pyfastflow.io
        assert pyfastflow.io is not None
        # Test main functions are available
        assert hasattr(pyfastflow.io, 'raster_to_numpy')

    @pytest.mark.importtest
    def test_io_ttbwrp_import(self):
        """Test TopoToolbox wrapper module import."""
        import pyfastflow.io.ttbwrp
        assert pyfastflow.io.ttbwrp is not None


class TestMiscImports:
    """Test imports for miscellaneous utility modules."""

    @pytest.mark.importtest
    def test_misc_init_import(self):
        """Test misc package import."""
        import pyfastflow.misc
        assert pyfastflow.misc is not None
        # Test main functions are available
        assert hasattr(pyfastflow.misc, 'load_raster_save_numpy')

    @pytest.mark.importtest
    def test_misc_raster_utils_import(self):
        """Test raster utilities module import."""
        import pyfastflow.misc.raster_utils
        assert pyfastflow.misc.raster_utils is not None


class TestPoolImports:
    """Test imports for memory pool modules."""

    @pytest.mark.importtest
    def test_pool_init_import(self):
        """Test pool package import."""
        import pyfastflow.pool
        assert pyfastflow.pool is not None

    @pytest.mark.importtest
    def test_pool_pool_import(self):
        """Test pool module import."""
        import pyfastflow.pool.pool
        assert pyfastflow.pool.pool is not None


class TestVisuImports:
    """Test imports for visualization modules."""

    @pytest.mark.importtest
    def test_visu_init_import(self):
        """Test visualization package import."""
        import pyfastflow.visu
        assert pyfastflow.visu is not None
        # Test main classes are available
        assert hasattr(pyfastflow.visu, 'SurfaceViewer')

    @pytest.mark.importtest
    def test_visu_hillshading_import(self):
        """Test hillshading module import."""
        import pyfastflow.visu.hillshading
        assert pyfastflow.visu.hillshading is not None

    @pytest.mark.importtest
    def test_visu_live_import(self):
        """Test live visualization module import."""
        import pyfastflow.visu.live
        assert pyfastflow.visu.live is not None


class TestAPIAccessibility:
    """Test that key API components are accessible from top-level imports."""

    @pytest.mark.importtest
    def test_flow_router_accessibility(self):
        """Test FlowRouter is accessible from pf.flow."""
        import pyfastflow as pf
        assert hasattr(pf.flow, 'FlowRouter')

    @pytest.mark.importtest
    def test_grid_field_accessibility(self):
        """Test GridField is accessible from pf.flow."""
        import pyfastflow as pf
        assert hasattr(pf.flow, 'GridField')

    @pytest.mark.importtest
    def test_flooder_accessibility(self):
        """Test Flooder is accessible from pf.flood."""
        import pyfastflow as pf
        assert hasattr(pf.flood, 'Flooder')

    @pytest.mark.importtest
    def test_cli_command_accessibility(self):
        """Test CLI command is accessible."""
        import pyfastflow as pf
        assert hasattr(pf.cli.raster_commands, 'raster2npy')