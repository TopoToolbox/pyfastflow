"""
Unit tests for Taichi neighbouring operations.
"""
import pytest
import numpy as np


class TestNeighbourerFlat:
    """Test flat neighbourer operations."""

    @pytest.mark.unit
    def test_neighbourer_flat_imports(self):
        """Test that core flat neighbourer functions can be imported."""
        from pyfastflow.grid.neighbourer_flat import (
            neighbour, can_leave_domain, top, bottom, left, right,
            top_n, bottom_n, left_n, right_n,
            neighbour_n, can_leave_domain_n,
            top_pew, bottom_pew, left_pew, right_pew,
            neighbour_pew, can_leave_domain_pew,
            top_pns, bottom_pns, left_pns, right_pns,
            neighbour_pns, can_leave_domain_pns,
            neighbour_custom, can_leave_domain_custom,
            top_custom, bottom_custom, left_custom, right_custom
        )
        
        # Test all functions are callable
        functions = [
            neighbour, can_leave_domain, top, bottom, left, right,
            top_n, bottom_n, left_n, right_n,
            neighbour_n, can_leave_domain_n,
            top_pew, bottom_pew, left_pew, right_pew,
            neighbour_pew, can_leave_domain_pew,
            top_pns, bottom_pns, left_pns, right_pns,
            neighbour_pns, can_leave_domain_pns,
            neighbour_custom, can_leave_domain_custom,
            top_custom, bottom_custom, left_custom, right_custom
        ]
        
        for func in functions:
            assert callable(func)

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_basic_neighbours(self, skip_if_no_taichi):
        """Test basic neighbour calculations."""
        import taichi as ti
        from pyfastflow.grid.neighbourer_flat import top_n, bottom_n, left_n, right_n
        
        ti.init(arch=ti.cpu)
        
        # Test with a small 3x3 grid (nx=3, ny=3)
        @ti.kernel
        def test_neighbours():
            # Center cell (i=4 in flat indexing for 3x3 grid)
            center_i = 4
            
            # Test basic directions using _n functions
            top_i = top_n(center_i)
            bottom_i = bottom_n(center_i)
            left_i = left_n(center_i)
            right_i = right_n(center_i)
            
            # For 3x3 grid, center cell (1,1) = index 4
            # Top should be (0,1) = index 1
            # Bottom should be (2,1) = index 7
            # Left should be (1,0) = index 3
            # Right should be (1,2) = index 5
            assert top_i == 1
            assert bottom_i == 7
            assert left_i == 3
            assert right_i == 5
        
        test_neighbours()

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_boundary_conditions(self, skip_if_no_taichi):
        """Test boundary condition handling."""
        import taichi as ti
        from pyfastflow.grid.neighbourer_flat import can_leave_domain_n
        
        ti.init(arch=ti.cpu)
        
        @ti.kernel
        def test_boundaries():
            # Test edge cells in 3x3 grid
            # Top-left corner (0,0) = index 0
            corner_i = 0
            
            # Should be able to leave domain from corner (normal boundaries)
            can_leave = can_leave_domain_n(corner_i)
            
            # Corner cells can leave domain in normal boundary mode
            assert can_leave == True or can_leave == 1
        
        test_boundaries()


class TestNeighbourer2D:
    """Test 2D neighbourer operations."""

    @pytest.mark.unit
    def test_neighbourer_2d_imports(self):
        """Test that all 2D neighbourer functions can be imported."""
        from pyfastflow.grid.neighbourer_2D import (
            neighbour_2D, can_leave_domain_2D, top_2D, bottom_2D, left_2D, right_2D,
            top_n_2D, bottom_n_2D, left_n_2D, right_n_2D
        )
        
        functions = [
            neighbour_2D, can_leave_domain_2D, top_2D, bottom_2D, left_2D, right_2D,
            top_n_2D, bottom_n_2D, left_n_2D, right_n_2D
        ]
        
        for func in functions:
            assert callable(func)

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_2d_indexing(self, skip_if_no_taichi):
        """Test 2D indexing calculations."""
        import taichi as ti
        from pyfastflow.grid.neighbourer_2D import rc_from_i_2D, i_from_rc_2D
        
        ti.init(arch=ti.cpu)
        
        @ti.kernel
        def test_2d_neighbours():
            # Test conversion between flat and 2D indexing
            center_i = 4  # Center of 3x3 grid
            
            # Convert to row,col
            row, col = rc_from_i_2D(center_i)
            
            # Should be (1,1) for center of 3x3 grid
            assert row == 1 and col == 1
            
            # Convert back to flat index
            flat_i = i_from_rc_2D(row, col)
            assert flat_i == center_i
        
        test_2d_neighbours()


class TestNeighbourerFlatParam:
    """Test parameterized flat neighbourer operations."""

    @pytest.mark.unit
    def test_neighbourer_flat_param_imports(self):
        """Test that all parameterized flat neighbourer functions can be imported."""
        from pyfastflow.grid.neighbourer_flat_param import (
            neighbour_param, can_leave_domain_param, top_param, bottom_param, left_param, right_param,
            top_n_param, bottom_n_param, left_n_param, right_n_param
        )
        
        functions = [
            neighbour_param, can_leave_domain_param, top_param, bottom_param, left_param, right_param,
            top_n_param, bottom_n_param, left_n_param, right_n_param
        ]
        
        for func in functions:
            assert callable(func)

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_parameterized_neighbours(self, skip_if_no_taichi):
        """Test parameterized neighbour calculations."""
        import taichi as ti
        from pyfastflow.grid.neighbourer_flat_param import top_n_param, bottom_n_param, left_n_param, right_n_param
        
        ti.init(arch=ti.cpu)
        
        @ti.kernel
        def test_param_neighbours():
            # Test with 4x4 grid (nx=4, ny=4)
            nx, ny = 4, 4
            center_i = 5  # Center cell (1,1) in 4x4 grid
            
            top_i = top_n_param(center_i, nx)
            bottom_i = bottom_n_param(center_i, nx)
            left_i = left_n_param(center_i, nx)
            right_i = right_n_param(center_i, nx)
            
            # For 4x4 grid, cell (1,1) = index 5
            # Top should be (0,1) = index 1
            # Bottom should be (2,1) = index 9
            # Left should be (1,0) = index 4
            # Right should be (1,2) = index 6
            assert top_i == 1
            assert bottom_i == 9
            assert left_i == 4
            assert right_i == 6
        
        test_param_neighbours()


class TestNeighbourer2DParam:
    """Test parameterized 2D neighbourer operations."""

    @pytest.mark.unit
    def test_neighbourer_2d_param_imports(self):
        """Test that all parameterized 2D neighbourer functions can be imported."""
        from pyfastflow.grid.neighbourer_2D_param import (
            neighbour_2D_param, can_leave_domain_2D_param, top_2D_param, bottom_2D_param, 
            left_2D_param, right_2D_param, top_n_2D_param, bottom_n_2D_param,
            left_n_2D_param, right_n_2D_param
        )
        
        functions = [
            neighbour_2D_param, can_leave_domain_2D_param, top_2D_param, bottom_2D_param, 
            left_2D_param, right_2D_param, top_n_2D_param, bottom_n_2D_param,
            left_n_2D_param, right_n_2D_param
        ]
        
        for func in functions:
            assert callable(func)

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_2d_param_indexing(self, skip_if_no_taichi):
        """Test parameterized 2D indexing calculations."""
        import taichi as ti
        from pyfastflow.grid.neighbourer_2D_param import rc_from_i_2D_param, i_from_rc_2D_param
        
        ti.init(arch=ti.cpu)
        
        @ti.kernel
        def test_2d_param_neighbours():
            # Test with 5x5 grid
            nx, ny = 5, 5
            center_i = 12  # Center cell (2,2) in 5x5 grid
            
            # Convert to row,col with parameters
            row, col = rc_from_i_2D_param(center_i, nx)
            
            # Should be (2,2) for center of 5x5 grid
            assert row == 2 and col == 2
            
            # Convert back to flat index
            flat_i = i_from_rc_2D_param(row, col, nx)
            assert flat_i == center_i
        
        test_2d_param_neighbours()


class TestPeriodicBoundaries:
    """Test periodic boundary conditions."""

    @pytest.mark.unit
    def test_periodic_imports(self):
        """Test periodic boundary function imports."""
        from pyfastflow.grid.neighbourer_flat import (
            neighbour_pew, neighbour_pns,
            top_pew, bottom_pew, left_pew, right_pew,
            top_pns, bottom_pns, left_pns, right_pns
        )
        
        functions = [
            neighbour_pew, neighbour_pns,
            top_pew, bottom_pew, left_pew, right_pew,
            top_pns, bottom_pns, left_pns, right_pns
        ]
        
        for func in functions:
            assert callable(func)

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_periodic_ew_wrapping(self, skip_if_no_taichi):
        """Test East-West periodic wrapping."""
        import taichi as ti
        from pyfastflow.grid.neighbourer_flat import left_pew, right_pew
        
        ti.init(arch=ti.cpu)
        
        @ti.kernel
        def test_ew_wrapping():
            # Test with 3x3 grid
            # Left edge cell (1,0) = index 3
            left_edge_i = 3
            
            # Going left from left edge should wrap to right edge
            wrapped_left = left_pew(left_edge_i)
            
            # Should wrap to (1,2) = index 5
            assert wrapped_left == 5
            
            # Right edge cell (1,2) = index 5
            right_edge_i = 5
            
            # Going right from right edge should wrap to left edge
            wrapped_right = right_pew(right_edge_i)
            
            # Should wrap to (1,0) = index 3
            assert wrapped_right == 3
        
        test_ew_wrapping()


class TestCustomBoundaries:
    """Test custom boundary conditions."""

    @pytest.mark.unit
    def test_custom_imports(self):
        """Test custom boundary function imports."""
        from pyfastflow.grid.neighbourer_flat import (
            neighbour_custom, can_leave_domain_custom,
            top_custom, bottom_custom, left_custom, right_custom
        )
        
        functions = [
            neighbour_custom, can_leave_domain_custom,
            top_custom, bottom_custom, left_custom, right_custom
        ]
        
        for func in functions:
            assert callable(func)

    @pytest.mark.unit
    @pytest.mark.gpu 
    @pytest.mark.slow
    def test_custom_boundary_logic(self, skip_if_no_taichi):
        """Test custom boundary condition logic - skipped as it requires boundary setup."""
        # Custom boundary functions require global boundary arrays to be set up
        # This test just verifies the functions are imported correctly
        # Full testing would need integration test with proper Grid setup
        pytest.skip("Custom boundary functions require global boundary array setup")