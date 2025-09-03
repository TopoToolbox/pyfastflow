"""Unit tests for CLI functionality."""

import numpy as np
import pytest
from click.testing import CliRunner


class TestCLIRasterCommands:
    """Test CLI raster conversion and manipulation commands."""

    @pytest.fixture
    def runner(self):
        """Provide Click test runner."""
        return CliRunner()

    @pytest.mark.unit
    def test_raster2npy_help(self, runner):
        """Test that raster2npy command shows help."""
        from pyfastflow.cli.raster_commands import raster2npy

        result = runner.invoke(raster2npy, ["--help"])
        assert result.exit_code == 0
        assert "Convert a raster file to numpy array format" in result.output

    @pytest.mark.unit
    def test_raster2npy_requires_args(self, runner):
        """Test that raster2npy requires arguments."""
        from pyfastflow.cli.raster_commands import raster2npy

        result = runner.invoke(raster2npy, [])
        assert result.exit_code != 0  # Should fail without arguments

    @pytest.mark.unit
    def test_rastermanip_help(self, runner):
        """Ensure rastermanip CLI commands show help text."""
        from pyfastflow.cli.rastermanip_commands import raster_downscale, raster_upscale

        res_up = runner.invoke(raster_upscale, ["--help"])
        assert res_up.exit_code == 0
        assert "Double the resolution" in res_up.output

        res_down = runner.invoke(raster_downscale, ["--help"])
        assert res_down.exit_code == 0
        assert "Halve the resolution" in res_down.output

    @pytest.mark.unit
    def test_upscale_downscale_roundtrip(self, runner, tmp_path):
        """Test raster upscaling and downscaling round-trip."""
        from pyfastflow.cli.rastermanip_commands import (  # noqa: I001
            raster_downscale,
            raster_upscale,
        )

        pytest.importorskip("rasterio")
        from rasterio.crs import CRS
        from rasterio.transform import array_bounds, from_origin

        ttb = pytest.importorskip("topotoolbox")

        # create small raster
        data = np.array([[1, 2], [3, 4]], dtype=np.float32)
        go = ttb.GridObject()
        go.z = data
        go.cellsize = 1.0
        go.transform = from_origin(0, 0, 1.0, 1.0)
        go.georef = CRS.from_epsg(4326)
        go.bounds = array_bounds(go.rows, go.columns, go.transform)
        in_path = tmp_path / "in.tif"
        ttb.write_tif(go, in_path)

        # upscale
        up_path = tmp_path / "up.tif"
        result = runner.invoke(raster_upscale, [str(in_path), str(up_path)])
        assert result.exit_code == 0
        up = ttb.read_tif(up_path)
        assert up.rows == 4
        assert up.columns == 4

        # downscale
        down_path = tmp_path / "down.tif"
        result = runner.invoke(
            raster_downscale, [str(up_path), str(down_path), "--method", "mean"]
        )
        assert result.exit_code == 0
        down = ttb.read_tif(down_path)
        assert down.rows == 2
        assert down.columns == 2
