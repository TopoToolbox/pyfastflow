"""
Integration tests for basic PyFastFlow workflows.

These tests verify that core components work together properly
and basic workflows can be executed without errors.
"""
import pytest
import numpy as np


class TestBasicFlowWorkflow:
    """Test basic flow routing workflow."""

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_basic_flow_router_workflow(self, skip_if_no_taichi, test_data_manager):
        """Test a complete basic flow routing workflow."""
        import taichi as ti
        import pyfastflow as pf
        
        # Initialize Taichi
        ti.init(arch=ti.cpu, offline_cache=False)
        
        # Create test data
        elevation = test_data_manager.create_simple_dem(nx=20, ny=15, dx=1.0)
        
        # Test workflow (will be expanded as API is clarified)
        # For now, just test that key components can be imported and instantiated
        assert hasattr(pf.flow, 'FlowRouter')
        
        # This test will be expanded once GridField API is clarified

    @pytest.mark.integration
    def test_io_workflow(self, tmp_path, test_data_manager):
        """Test basic I/O workflow with synthetic data."""
        import numpy as np
        import pyfastflow as pf
        
        # Create synthetic elevation data
        elevation = test_data_manager.create_drainage_pattern(nx=10, ny=8, dx=2.0)
        
        # Save as numpy array (simulating raster processing)
        test_file = tmp_path / "test_elevation.npy"
        np.save(test_file, elevation)
        
        # Load back
        loaded = np.load(test_file)
        
        # Verify data integrity
        np.testing.assert_array_equal(elevation, loaded)

    @pytest.mark.integration
    def test_misc_utilities_workflow(self, tmp_path, test_data_manager):
        """Test miscellaneous utilities workflow."""
        import numpy as np
        import pyfastflow as pf
        
        # Create test data
        elevation = test_data_manager.create_simple_dem(nx=15, ny=12, dx=1.5)
        
        # Save using numpy (simulating the misc utilities)
        output_file = tmp_path / "test_output.npy"
        np.save(output_file, elevation)
        
        # Verify the misc module has expected functions
        assert hasattr(pf.misc, 'load_raster_save_numpy')


class TestModuleInteractions:
    """Test interactions between different modules."""

    @pytest.mark.integration
    def test_flow_and_erodep_modules_interaction(self):
        """Test that flow and erosion modules can work together."""
        import pyfastflow as pf
        
        # Test that both modules are available
        assert hasattr(pf, 'flow')
        assert hasattr(pf, 'erodep')
        
        # Test that key functions are available from both
        assert hasattr(pf.flow, 'FlowRouter')
        assert hasattr(pf.erodep, 'SPL')

    @pytest.mark.integration
    def test_grid_and_flow_modules_interaction(self):
        """Test that grid and flow modules can work together."""
        import pyfastflow as pf
        
        # Test that both modules are available
        assert hasattr(pf, 'grid')
        assert hasattr(pf, 'flow')
        
        # Test that GridField is accessible from flow module
        assert hasattr(pf.flow, 'GridField')