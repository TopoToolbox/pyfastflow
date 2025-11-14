"""
DEM to PNG Conversion CLI Commands for PyFastFlow

Command line interface for converting DEM files to PNG format.

Author: B.G.
"""

import sys

import click
import numpy as np
from PIL import Image


@click.command()
@click.argument("input_dem", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default=None,
    help="Output PNG filename (default: input name with .png extension)",
)
@click.option(
    "--uint",
    is_flag=True,
    default=False,
    help="Save as uint8 (0-255), otherwise save as float (0-1)",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def dem2png(input_dem, output, uint, verbose):
    """
    Convert a DEM file to PNG format.

    Loads a DEM file using TopoToolbox (supports GeoTIFF, ASCII grid, etc.)
    or numpy array (.npy) and saves it as a single-band PNG image.

    INPUT_DEM: Path to input DEM file (.tif, .npy, etc.)

    Examples:

        # Convert DEM to float PNG (0-1)
        pff-dem2png elevation.tif

        # Convert numpy array to uint8 PNG (0-255)
        pff-dem2png elevation.npy --uint

        # Specify output name
        pff-dem2png elevation.tif -o terrain.png

        # With verbose output
        pff-dem2png -v dem.tif --uint
    """
    try:
        if verbose:
            click.echo(f"Loading DEM from '{input_dem}'...")

        # Check if input is a numpy array or raster file
        if input_dem.endswith('.npy'):
            # Load numpy array directly
            dem_data = np.load(input_dem)
        else:
            # Import topotoolbox here to give better error message if missing
            import topotoolbox as ttb
            # Load the DEM using topotoolbox
            dem_data = ttb.read_tif(input_dem)

        # Determine output filename
        if output is None:
            output = input_dem.rsplit(".", 1)[0] + ".png"

        if verbose:
            click.echo(f"Processing DEM data (shape: {dem_data.shape})...")

        # Normalize the data to 0-1 range
        dem_min = np.nanmin(dem_data)
        dem_max = np.nanmax(dem_data)

        if dem_min == dem_max:
            click.echo("Warning: DEM has constant values", err=True)
            normalized = np.zeros_like(dem_data)
        else:
            normalized = (dem_data - dem_min) / (dem_max - dem_min)

        # Handle NaN values (set to 0)
        normalized = np.nan_to_num(normalized, nan=0.0)

        if uint:
            # Convert to uint8 (0-255)
            img_data = (normalized * 255).astype(np.uint8)
            mode = "L"  # 8-bit grayscale
        else:
            # Convert to uint16 (0-65535) for better precision in PNG
            img_data = (normalized * 65535).astype(np.uint16)
            mode = "I;16"  # 16-bit grayscale

        # Create and save PIL image
        img = Image.fromarray(img_data, mode=mode)

        if verbose:
            click.echo(f"Saving PNG to '{output}'...")

        img.save(output)

        if verbose:
            click.echo(f"Conversion completed! Mode: {mode}, Range: {dem_min}-{dem_max}")
        else:
            click.echo(f"Converted '{input_dem}' -> '{output}'")

    except ImportError as e:
        click.echo(f"Error: Missing dependency - {e}", err=True)
        click.echo("Install topotoolbox with: pip install topotoolbox", err=True)
        click.echo("Install pillow with: pip install pillow", err=True)
        sys.exit(1)

    except FileNotFoundError:
        click.echo(f"Error: Input file '{input_dem}' not found", err=True)
        sys.exit(1)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    dem2png()
