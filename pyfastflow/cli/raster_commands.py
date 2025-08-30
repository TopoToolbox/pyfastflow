"""
Raster Processing CLI Commands for PyFastFlow

Command line interface for raster file processing operations,
particularly conversion between raster formats and numpy arrays.

Author: B.G.
"""

import sys

import click

import pyfastflow as pf


@click.command()
@click.argument("input_raster", type=click.Path(exists=True))
@click.argument("output_npy", type=click.Path())
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def raster2npy(input_raster, output_npy, verbose):
    """
    Convert a raster file to numpy array format.

    Loads a raster file (GeoTIFF, ASCII grid, etc.) using TopoToolbox
    and saves the elevation values as a .npy file for use in PyFastFlow
    simulations.

    INPUT_RASTER: Path to input raster file
    OUTPUT_NPY: Path for output .npy file

    Examples:

        # Convert GeoTIFF to numpy
        pff-raster2npy elevation.tif elevation.npy

        # With verbose output
        pff-raster2npy -v dem.tif terrain.npy
    """
    try:
        if verbose:
            click.echo(
                f"Converting raster '{input_raster}' to numpy array '{output_npy}'..."
            )

        # Use PyFastFlow's misc module to perform conversion
        pf.misc.load_raster_save_numpy(input_raster, output_npy)

        if verbose:
            click.echo("Conversion completed successfully!")
        else:
            click.echo(f"Successfully converted '{input_raster}' -> '{output_npy}'")

    except ImportError as e:
        click.echo(f"Error: Missing dependency - {e}", err=True)
        click.echo("Install topotoolbox with: pip install topotoolbox", err=True)
        sys.exit(1)

    except FileNotFoundError:
        click.echo(f"Error: Input file '{input_raster}' not found", err=True)
        sys.exit(1)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    raster2npy()
