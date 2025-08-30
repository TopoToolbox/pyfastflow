"""
Unit tests for CLI functionality.
"""
import pytest
from click.testing import CliRunner


class TestCLIRasterCommands:
    """Test CLI raster conversion commands."""

    @pytest.fixture
    def runner(self):
        """Provide Click test runner."""
        return CliRunner()

    @pytest.mark.unit
    def test_raster2npy_help(self, runner):
        """Test that raster2npy command shows help."""
        from pyfastflow.cli.raster_commands import raster2npy
        
        result = runner.invoke(raster2npy, ['--help'])
        assert result.exit_code == 0
        assert 'Convert a raster file to numpy array format' in result.output

    @pytest.mark.unit
    def test_raster2npy_requires_args(self, runner):
        """Test that raster2npy requires arguments."""
        from pyfastflow.cli.raster_commands import raster2npy
        
        result = runner.invoke(raster2npy, [])
        assert result.exit_code != 0  # Should fail without arguments