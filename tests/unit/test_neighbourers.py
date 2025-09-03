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


class TestNeighbourerFlatD8:
    """Test flat D8 neighbourer operations."""

    @pytest.mark.unit
    def test_neighbourer_flat_d8_imports(self):
        """Test that core flat D8 neighbourer functions can be imported."""
        from pyfastflow.grid.neighbourer_flat_d8 import (
            neighbour_d8, can_leave_domain_d8, 
            topleft_d8, top_d8, topright_d8, left_d8, right_d8, bottomleft_d8, bottom_d8, bottomright_d8,
            topleft_n_d8, top_n_d8, topright_n_d8, left_n_d8, right_n_d8, bottomleft_n_d8, bottom_n_d8, bottomright_n_d8,
            neighbour_n_d8, can_leave_domain_n_d8,
            topleft_pew_d8, top_pew_d8, topright_pew_d8, left_pew_d8, right_pew_d8, bottomleft_pew_d8, bottom_pew_d8, bottomright_pew_d8,
            neighbour_pew_d8, can_leave_domain_pew_d8,
            topleft_pns_d8, top_pns_d8, topright_pns_d8, left_pns_d8, right_pns_d8, bottomleft_pns_d8, bottom_pns_d8, bottomright_pns_d8,
            neighbour_pns_d8, can_leave_domain_pns_d8,
            neighbour_custom_d8, can_leave_domain_custom_d8,
            topleft_custom_d8, top_custom_d8, topright_custom_d8, left_custom_d8, right_custom_d8, bottomleft_custom_d8, bottom_custom_d8, bottomright_custom_d8
        )
        
        # Test all functions are callable
        functions = [
            neighbour_d8, can_leave_domain_d8,
            topleft_d8, top_d8, topright_d8, left_d8, right_d8, bottomleft_d8, bottom_d8, bottomright_d8,
            topleft_n_d8, top_n_d8, topright_n_d8, left_n_d8, right_n_d8, bottomleft_n_d8, bottom_n_d8, bottomright_n_d8,
            neighbour_n_d8, can_leave_domain_n_d8,
            topleft_pew_d8, top_pew_d8, topright_pew_d8, left_pew_d8, right_pew_d8, bottomleft_pew_d8, bottom_pew_d8, bottomright_pew_d8,
            neighbour_pew_d8, can_leave_domain_pew_d8,
            topleft_pns_d8, top_pns_d8, topright_pns_d8, left_pns_d8, right_pns_d8, bottomleft_pns_d8, bottom_pns_d8, bottomright_pns_d8,
            neighbour_pns_d8, can_leave_domain_pns_d8,
            neighbour_custom_d8, can_leave_domain_custom_d8,
            topleft_custom_d8, top_custom_d8, topright_custom_d8, left_custom_d8, right_custom_d8, bottomleft_custom_d8, bottom_custom_d8, bottomright_custom_d8
        ]
        
        for func in functions:
            assert callable(func)

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_basic_d8_neighbours(self, skip_if_no_taichi):
        """Test basic D8 neighbour calculations."""
        import taichi as ti
        from pyfastflow.grid.neighbourer_flat_d8 import (
            topleft_n_d8, top_n_d8, topright_n_d8, left_n_d8, 
            right_n_d8, bottomleft_n_d8, bottom_n_d8, bottomright_n_d8
        )
        
        ti.init(arch=ti.cpu)
        
        # Test with a small 3x3 grid (nx=3, ny=3)
        @ti.kernel
        def test_d8_neighbours():
            # Center cell (i=4 in flat indexing for 3x3 grid)
            center_i = 4
            
            # Test all 8 directions using _n functions
            topleft_i = topleft_n_d8(center_i)
            top_i = top_n_d8(center_i)
            topright_i = topright_n_d8(center_i)
            left_i = left_n_d8(center_i)
            right_i = right_n_d8(center_i)
            bottomleft_i = bottomleft_n_d8(center_i)
            bottom_i = bottom_n_d8(center_i)
            bottomright_i = bottomright_n_d8(center_i)
            
            # For 3x3 grid, center cell (1,1) = index 4
            # Neighbors should be:
            # 0 1 2
            # 3 4 5
            # 6 7 8
            assert topleft_i == 0
            assert top_i == 1
            assert topright_i == 2
            assert left_i == 3
            assert right_i == 5
            assert bottomleft_i == 6
            assert bottom_i == 7
            assert bottomright_i == 8
        
        test_d8_neighbours()

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_d8_boundary_conditions(self, skip_if_no_taichi):
        """Test D8 boundary condition handling."""
        import taichi as ti
        from pyfastflow.grid.neighbourer_flat_d8 import validate_link_n_d8
        
        ti.init(arch=ti.cpu)
        
        @ti.kernel
        def test_d8_boundaries():
            # Test corner cell (0,0) = index 0
            corner_i = 0
            
            # Test diagonal directions at corner - should be blocked
            # Direction 0 = topleft - blocked at corner
            topleft_valid = validate_link_n_d8(corner_i, 0)
            # Direction 2 = topright - blocked at top edge  
            topright_valid = validate_link_n_d8(corner_i, 2)
            
            # Both should be False/0 at corner
            assert topleft_valid == False or topleft_valid == 0
            assert topright_valid == False or topright_valid == 0
        
        test_d8_boundaries()


class TestNeighbourer2DD8:
    """Test 2D D8 neighbourer operations."""

    @pytest.mark.unit
    def test_neighbourer_2d_d8_imports(self):
        """Test that all 2D D8 neighbourer functions can be imported."""
        from pyfastflow.grid.neighbourer_2D_d8 import (
            neighbour_2D_d8, can_leave_domain_2D_d8,
            topleft_2D_d8, top_2D_d8, topright_2D_d8, left_2D_d8, right_2D_d8, bottomleft_2D_d8, bottom_2D_d8, bottomright_2D_d8,
            topleft_n_2D_d8, top_n_2D_d8, topright_n_2D_d8, left_n_2D_d8, right_n_2D_d8, bottomleft_n_2D_d8, bottom_n_2D_d8, bottomright_n_2D_d8
        )
        
        functions = [
            neighbour_2D_d8, can_leave_domain_2D_d8,
            topleft_2D_d8, top_2D_d8, topright_2D_d8, left_2D_d8, right_2D_d8, bottomleft_2D_d8, bottom_2D_d8, bottomright_2D_d8,
            topleft_n_2D_d8, top_n_2D_d8, topright_n_2D_d8, left_n_2D_d8, right_n_2D_d8, bottomleft_n_2D_d8, bottom_n_2D_d8, bottomright_n_2D_d8
        ]
        
        for func in functions:
            assert callable(func)

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_2d_d8_indexing(self, skip_if_no_taichi):
        """Test 2D D8 indexing calculations."""
        import taichi as ti
        from pyfastflow.grid.neighbourer_2D_d8 import rc_from_i_2D_d8, i_from_rc_2D_d8
        
        ti.init(arch=ti.cpu)
        
        @ti.kernel
        def test_2d_d8_neighbours():
            # Test conversion between flat and 2D indexing
            center_i = 4  # Center of 3x3 grid
            
            # Convert to row,col
            row, col = rc_from_i_2D_d8(center_i)
            
            # Should be (1,1) for center of 3x3 grid
            assert row == 1 and col == 1
            
            # Convert back to flat index
            flat_i = i_from_rc_2D_d8(row, col)
            assert flat_i == center_i
        
        test_2d_d8_neighbours()


class TestNeighbourerFlatParamD8:
    """Test parameterized flat D8 neighbourer operations."""

    @pytest.mark.unit
    def test_neighbourer_flat_param_d8_imports(self):
        """Test that all parameterized flat D8 neighbourer functions can be imported."""
        from pyfastflow.grid.neighbourer_flat_param_d8 import (
            neighbour_param_d8, can_leave_domain_param_d8,
            topleft_param_d8, top_param_d8, topright_param_d8, left_param_d8, right_param_d8, bottomleft_param_d8, bottom_param_d8, bottomright_param_d8,
            topleft_n_param_d8, top_n_param_d8, topright_n_param_d8, left_n_param_d8, right_n_param_d8, bottomleft_n_param_d8, bottom_n_param_d8, bottomright_n_param_d8
        )
        
        functions = [
            neighbour_param_d8, can_leave_domain_param_d8,
            topleft_param_d8, top_param_d8, topright_param_d8, left_param_d8, right_param_d8, bottomleft_param_d8, bottom_param_d8, bottomright_param_d8,
            topleft_n_param_d8, top_n_param_d8, topright_n_param_d8, left_n_param_d8, right_n_param_d8, bottomleft_n_param_d8, bottom_n_param_d8, bottomright_n_param_d8
        ]
        
        for func in functions:
            assert callable(func)

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_parameterized_d8_neighbours(self, skip_if_no_taichi):
        """Test parameterized D8 neighbour calculations."""
        import taichi as ti
        from pyfastflow.grid.neighbourer_flat_param_d8 import (
            topleft_n_param_d8, top_n_param_d8, topright_n_param_d8, left_n_param_d8, 
            right_n_param_d8, bottomleft_n_param_d8, bottom_n_param_d8, bottomright_n_param_d8
        )
        
        ti.init(arch=ti.cpu)
        
        @ti.kernel
        def test_param_d8_neighbours():
            # Test with 4x4 grid (nx=4, ny=4)
            nx, ny = 4, 4
            center_i = 5  # Center cell (1,1) in 4x4 grid
            
            topleft_i = topleft_n_param_d8(center_i, nx)
            top_i = top_n_param_d8(center_i, nx)
            topright_i = topright_n_param_d8(center_i, nx)
            left_i = left_n_param_d8(center_i, nx)
            right_i = right_n_param_d8(center_i, nx)
            bottomleft_i = bottomleft_n_param_d8(center_i, nx)
            bottom_i = bottom_n_param_d8(center_i, nx)
            bottomright_i = bottomright_n_param_d8(center_i, nx)
            
            # For 4x4 grid, cell (1,1) = index 5
            # Neighbors should be:
            # 0  1  2  3
            # 4  5  6  7
            # 8  9 10 11
            # 12 13 14 15
            assert topleft_i == 0
            assert top_i == 1
            assert topright_i == 2
            assert left_i == 4
            assert right_i == 6
            assert bottomleft_i == 8
            assert bottom_i == 9
            assert bottomright_i == 10
        
        test_param_d8_neighbours()


class TestNeighbourer2DParamD8:
    """Test parameterized 2D D8 neighbourer operations."""

    @pytest.mark.unit
    def test_neighbourer_2d_param_d8_imports(self):
        """Test that all parameterized 2D D8 neighbourer functions can be imported."""
        from pyfastflow.grid.neighbourer_2D_param_d8 import (
            neighbour_2D_param_d8, can_leave_domain_2D_param_d8,
            topleft_2D_param_d8, top_2D_param_d8, topright_2D_param_d8, left_2D_param_d8, right_2D_param_d8, 
            bottomleft_2D_param_d8, bottom_2D_param_d8, bottomright_2D_param_d8,
            topleft_n_2D_param_d8, top_n_2D_param_d8, topright_n_2D_param_d8, left_n_2D_param_d8, 
            right_n_2D_param_d8, bottomleft_n_2D_param_d8, bottom_n_2D_param_d8, bottomright_n_2D_param_d8
        )
        
        functions = [
            neighbour_2D_param_d8, can_leave_domain_2D_param_d8,
            topleft_2D_param_d8, top_2D_param_d8, topright_2D_param_d8, left_2D_param_d8, right_2D_param_d8, 
            bottomleft_2D_param_d8, bottom_2D_param_d8, bottomright_2D_param_d8,
            topleft_n_2D_param_d8, top_n_2D_param_d8, topright_n_2D_param_d8, left_n_2D_param_d8, 
            right_n_2D_param_d8, bottomleft_n_2D_param_d8, bottom_n_2D_param_d8, bottomright_n_2D_param_d8
        ]
        
        for func in functions:
            assert callable(func)

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_2d_param_d8_indexing(self, skip_if_no_taichi):
        """Test parameterized 2D D8 indexing calculations."""
        import taichi as ti
        from pyfastflow.grid.neighbourer_2D_param_d8 import rc_from_i_2D_param_d8, i_from_rc_2D_param_d8
        
        ti.init(arch=ti.cpu)
        
        @ti.kernel
        def test_2d_param_d8_neighbours():
            # Test with 5x5 grid
            nx, ny = 5, 5
            center_i = 12  # Center cell (2,2) in 5x5 grid
            
            # Convert to row,col with parameters
            row, col = rc_from_i_2D_param_d8(center_i, nx)
            
            # Should be (2,2) for center of 5x5 grid
            assert row == 2 and col == 2
            
            # Convert back to flat index
            flat_i = i_from_rc_2D_param_d8(row, col, nx)
            assert flat_i == center_i
        
        test_2d_param_d8_neighbours()


class TestPeriodicBoundariesD8:
    """Test D8 periodic boundary conditions."""

    @pytest.mark.unit
    def test_periodic_d8_imports(self):
        """Test D8 periodic boundary function imports."""
        from pyfastflow.grid.neighbourer_flat_d8 import (
            neighbour_pew_d8, neighbour_pns_d8,
            topleft_pew_d8, top_pew_d8, topright_pew_d8, left_pew_d8, right_pew_d8, bottomleft_pew_d8, bottom_pew_d8, bottomright_pew_d8,
            topleft_pns_d8, top_pns_d8, topright_pns_d8, left_pns_d8, right_pns_d8, bottomleft_pns_d8, bottom_pns_d8, bottomright_pns_d8
        )
        
        functions = [
            neighbour_pew_d8, neighbour_pns_d8,
            topleft_pew_d8, top_pew_d8, topright_pew_d8, left_pew_d8, right_pew_d8, bottomleft_pew_d8, bottom_pew_d8, bottomright_pew_d8,
            topleft_pns_d8, top_pns_d8, topright_pns_d8, left_pns_d8, right_pns_d8, bottomleft_pns_d8, bottom_pns_d8, bottomright_pns_d8
        ]
        
        for func in functions:
            assert callable(func)

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_periodic_d8_ew_wrapping(self, skip_if_no_taichi):
        """Test D8 East-West periodic wrapping including diagonals."""
        import taichi as ti
        from pyfastflow.grid.neighbourer_flat_d8 import topleft_pew_d8, topright_pew_d8, bottomleft_pew_d8, bottomright_pew_d8
        
        ti.init(arch=ti.cpu)
        
        @ti.kernel
        def test_d8_ew_wrapping():
            # Test with 3x3 grid
            # Left edge cell (1,0) = index 3
            left_edge_i = 3
            
            # Going topleft from left edge should wrap eastward
            wrapped_topleft = topleft_pew_d8(left_edge_i)
            # Should wrap to (0,2) = index 2
            assert wrapped_topleft == 2
            
            # Going bottomleft from left edge should wrap eastward
            wrapped_bottomleft = bottomleft_pew_d8(left_edge_i)  
            # Should wrap to (2,2) = index 8
            assert wrapped_bottomleft == 8
            
            # Right edge cell (1,2) = index 5
            right_edge_i = 5
            
            # Going topright from right edge should wrap westward
            wrapped_topright = topright_pew_d8(right_edge_i)
            # Should wrap to (0,0) = index 0
            assert wrapped_topright == 0
            
            # Going bottomright from right edge should wrap westward
            wrapped_bottomright = bottomright_pew_d8(right_edge_i)
            # Should wrap to (2,0) = index 6
            assert wrapped_bottomright == 6
        
        test_d8_ew_wrapping()

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_periodic_d8_ns_wrapping(self, skip_if_no_taichi):
        """Test D8 North-South periodic wrapping including diagonals."""
        import taichi as ti
        from pyfastflow.grid.neighbourer_flat_d8 import topleft_pns_d8, topright_pns_d8, bottomleft_pns_d8, bottomright_pns_d8
        
        ti.init(arch=ti.cpu)
        
        @ti.kernel
        def test_d8_ns_wrapping():
            # Test with 3x3 grid
            # Top edge cell (0,1) = index 1
            top_edge_i = 1
            
            # Going topleft from top edge should wrap southward
            wrapped_topleft = topleft_pns_d8(top_edge_i)
            # Should wrap to (2,0) = index 6
            assert wrapped_topleft == 6
            
            # Going topright from top edge should wrap southward
            wrapped_topright = topright_pns_d8(top_edge_i)
            # Should wrap to (2,2) = index 8  
            assert wrapped_topright == 8
            
            # Bottom edge cell (2,1) = index 7
            bottom_edge_i = 7
            
            # Going bottomleft from bottom edge should wrap northward
            wrapped_bottomleft = bottomleft_pns_d8(bottom_edge_i)
            # Should wrap to (0,0) = index 0
            assert wrapped_bottomleft == 0
            
            # Going bottomright from bottom edge should wrap northward
            wrapped_bottomright = bottomright_pns_d8(bottom_edge_i)
            # Should wrap to (0,2) = index 2
            assert wrapped_bottomright == 2
        
        test_d8_ns_wrapping()


class TestD8DirectionValidation:
    """Test D8 direction validation and neighbour function."""

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_d8_neighbour_function(self, skip_if_no_taichi):
        """Test the main D8 neighbour function with direction parameter."""
        import taichi as ti
        from pyfastflow.grid.neighbourer_flat_d8 import neighbour_n_d8
        
        ti.init(arch=ti.cpu)
        
        @ti.kernel
        def test_d8_neighbour():
            # Test with 3x3 grid, center cell = index 4
            center_i = 4
            
            # Test all 8 directions
            # 0=topleft, 1=top, 2=topright, 3=left, 4=right, 5=bottomleft, 6=bottom, 7=bottomright
            for direction in range(8):
                neighbor_i = neighbour_n_d8(center_i, direction)
                
                # All should be valid from center cell
                assert neighbor_i != -1
                
                # Verify correct neighbors
                if direction == 0:  # topleft
                    assert neighbor_i == 0
                elif direction == 1:  # top
                    assert neighbor_i == 1
                elif direction == 2:  # topright
                    assert neighbor_i == 2
                elif direction == 3:  # left
                    assert neighbor_i == 3
                elif direction == 4:  # right
                    assert neighbor_i == 5
                elif direction == 5:  # bottomleft
                    assert neighbor_i == 6
                elif direction == 6:  # bottom
                    assert neighbor_i == 7
                elif direction == 7:  # bottomright
                    assert neighbor_i == 8
        
        test_d8_neighbour()