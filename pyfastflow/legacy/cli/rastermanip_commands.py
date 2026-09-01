"""CLI commands for raster resampling using PyFastFlow's rastermanip utilities."""

import sys

import click

import pyfastflow as pf

try:
    import topotoolbox as ttb
    from rasterio.transform import Affine, array_bounds
    TOPOTOOLBOX_AVAILABLE = True
except ImportError:  # pragma: no cover - handled in commands
    TOPOTOOLBOX_AVAILABLE = False
    ttb = None
    Affine = None
    array_bounds = None


def _require_topotoolbox():
    """Ensure TopoToolbox is available."""
    if not TOPOTOOLBOX_AVAILABLE:
        raise ImportError(
            "TopoToolbox is required for this command. Install with: pip install topotoolbox"
        )


@click.command()
@click.argument("input_raster", type=click.Path(exists=True))
@click.argument("output_raster", type=click.Path())
@click.option(
    "--method",
    "-m",
    type=click.Choice(["nearest", "bilinear", "bicubic", "lanczos"]),
    default="bicubic",
    show_default=True,
    help="Interpolation method for upscaling",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def raster_upscale(input_raster, output_raster, method, verbose):
    """Double the resolution of INPUT_RASTER and save to OUTPUT_RASTER."""
    try:
        _require_topotoolbox()
        if verbose:
            click.echo(
                f"Upscaling raster '{input_raster}' -> '{output_raster}' with method='{method}'"
            )
        grid = ttb.read_tif(input_raster)
        rasmanctx = pf.rastermanip.RasManContext()
        result = rasmanctx.double_resolution(
            grid.z,
            method=method,
            as_numpy=True,
            output_layout="2d",
        )
        grid.z = result.astype("float32")
        grid.cellsize = grid.cellsize / 2
        t = grid.transform
        grid.transform = Affine(t.a / 2, t.b, t.c, t.d, t.e / 2, t.f)
        grid.bounds = array_bounds(grid.rows, grid.columns, grid.transform)
        ttb.write_tif(grid, output_raster)
        if verbose:
            click.echo("Upscaling completed successfully!")
    except Exception as e:  # pragma: no cover - error handling path
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@click.command()
@click.argument("input_raster", type=click.Path(exists=True))
@click.argument("output_raster", type=click.Path())
@click.option(
    "--method",
    "-m",
    type=click.Choice(["mean", "median", "min", "max", "percentile"]),
    default="mean",
    show_default=True,
    help="Aggregation method for downscaling",
)
@click.option(
    "--percentile",
    "-p",
    default=50.0,
    show_default=True,
    type=float,
    help="Percentile value when method='percentile'",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def raster_downscale(input_raster, output_raster, method, percentile, verbose):
    """Halve the resolution of INPUT_RASTER and save to OUTPUT_RASTER."""
    try:
        _require_topotoolbox()
        if verbose:
            click.echo(
                f"Downscaling raster '{input_raster}' -> '{output_raster}' using method='{method}'"
            )
        grid = ttb.read_tif(input_raster)
        rasmanctx = pf.rastermanip.RasManContext()
        result = rasmanctx.halve_resolution(
            grid.z,
            method=method,
            percentile=percentile,
            as_numpy=True,
            output_layout="2d",
        )
        grid.z = result.astype("float32")
        grid.cellsize = grid.cellsize * 2
        t = grid.transform
        grid.transform = Affine(t.a * 2, t.b, t.c, t.d, t.e * 2, t.f)
        grid.bounds = array_bounds(grid.rows, grid.columns, grid.transform)
        ttb.write_tif(grid, output_raster)
        if verbose:
            click.echo("Downscaling completed successfully!")
    except Exception as e:  # pragma: no cover - error handling path
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


__all__ = ["raster_upscale", "raster_downscale"]
