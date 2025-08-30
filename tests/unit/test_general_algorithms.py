"""
Unit tests for general algorithms.
"""
import pytest
import numpy as np


class TestMathUtils:
    """Test mathematical utilities."""

    @pytest.mark.unit
    def test_atan_import(self):
        """Test that atan function can be imported."""
        from pyfastflow.general_algorithms.math_utils import atan
        assert callable(atan)

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_atan_basic_functionality(self, skip_if_no_taichi):
        """Test basic atan functionality with Taichi."""
        import taichi as ti
        from pyfastflow.general_algorithms.math_utils import atan
        
        # This test would require Taichi field setup
        # For now, just test that the function exists
        assert callable(atan)


class TestPingPong:
    """Test ping-pong utilities."""

    @pytest.mark.unit
    def test_pingpong_functions_import(self):
        """Test that ping-pong functions can be imported."""
        from pyfastflow.general_algorithms.pingpong import fuse, getSrc, updateSrc
        
        assert callable(fuse)
        assert callable(getSrc)
        assert callable(updateSrc)


class TestParallelScan:
    """Test parallel scan algorithms."""

    @pytest.mark.unit
    def test_inclusive_scan_import(self):
        """Test that inclusive_scan can be imported."""
        from pyfastflow.general_algorithms.parallel_scan import inclusive_scan
        assert callable(inclusive_scan)