"""
TopoToolbox Integration Module for PyFastFlow

This module provides seamless integration between PyFastFlow and PyTopoToolbox,
enabling easy data exchange and workflow integration between the two packages.
PyTopoToolbox provides comprehensive DEM analysis tools while PyFastFlow offers
high-performance GPU-accelerated flow modeling.

Key Features:
- Convert TopoToolbox GridObj to PyFastFlow Grid
- Direct raster file loading into PyFastFlow format
- Preserve metadata and coordinate system information
- Enable combined analysis workflows

Dependencies:
- topotoolbox: Required for all functions in this module
  Install with: pip install topotoolbox

Author: B.G.
"""

import pyfastflow as pf

try:
    import topotoolbox as ttb

    TOPOTOOLBOX_AVAILABLE = True
except ImportError:
    TOPOTOOLBOX_AVAILABLE = False
    ttb = None


def _check_topotoolbox():
    """Check if topotoolbox is available and raise informative error if not."""
    if not TOPOTOOLBOX_AVAILABLE:
        raise ImportError(
            "TopoToolbox is required for io functionality but not installed. "
            "Install with: pip install topotoolbox or pip install pyfastflow[topotoolbox]"
        )


def gridobj_to_grid(go):
    """
    Convert a TopoToolbox GridObj to a PyFastFlow Grid object.

    Transfers elevation data, grid dimensions, and coordinate system information
    from a TopoToolbox GridObj to a PyFastFlow Grid for use in flow modeling
    and landscape evolution simulations.

    Args:
        go (topotoolbox.GridObj): TopoToolbox GridObj containing elevation data
            Must have attributes: columns, rows, cellsize, z (elevation array)

    Returns:
        pf.grid.Grid: PyFastFlow Grid object ready for flow routing

    Note:
        - Original TopoToolbox GridObj is preserved in metadata for reference
        - Uses 'normal' boundary conditions by default (flow can exit at edges)
        - Grid spacing (cellsize) must be uniform in both x and y directions
        - Elevation data is automatically converted to 1D format for GPU processing

    Example:
        import topotoolbox as ttb
        import pyfastflow as pf

        # Load DEM with TopoToolbox
        dem_ttb = ttb.load_dem('elevation.tif')

        # Convert to PyFastFlow Grid
        grid = pf.io.gridobj_to_grid(dem_ttb)

        # Use with PyFastFlow flow router
        router = pf.flow.FlowRouter(grid)
        router.compute_receivers()

    Author: B.G.
    """
    _check_topotoolbox()

    # Validate input
    if (
        not hasattr(go, "columns")
        or not hasattr(go, "rows")
        or not hasattr(go, "cellsize")
    ):
        raise ValueError(
            "Invalid GridObj: missing required attributes (columns, rows, cellsize)"
        )

    if not hasattr(go, "z"):
        raise ValueError("Invalid GridObj: missing elevation data (z attribute)")

    # Create PyFastFlow Grid with TopoToolbox data
    grid = pf.grid.Grid(
        nx=go.columns,
        ny=go.rows,
        dx=go.cellsize,
        z=go.z,
        boundary_mode="normal",
        boundaries=None,
        metadata={"ttb_grid": go},
    )

    return grid


def raster_to_grid(fname):
    """
    Load a raster file directly into a PyFastFlow Grid object.

    Convenience function that combines TopoToolbox's raster reading capabilities
    with PyFastFlow Grid creation. Supports all raster formats handled by
    TopoToolbox (GeoTIFF, ASCII, etc.).

    Args:
        fname (str): Path to raster file (GeoTIFF, ASCII grid, etc.)
            Must be a format supported by TopoToolbox

    Returns:
        pf.grid.Grid: PyFastFlow Grid object ready for flow modeling

    Raises:
        FileNotFoundError: If the specified file doesn't exist
        ValueError: If the raster file cannot be read or has invalid data
        ImportError: If topotoolbox is not installed

    Note:
        - Preserves coordinate system and metadata from original raster
        - Automatically handles different raster formats and projections
        - Grid spacing is derived from raster cell size
        - Uses TopoToolbox's robust raster reading under the hood

    Example:
        import pyfastflow as pf

        # Load DEM directly from file
        grid = pf.io.raster_to_grid('path/to/elevation.tif')

        # Create flow router and run simulation
        router = pf.flow.FlowRouter(grid)
        router.compute_receivers()
        router.accumulate_constant_Q(1.0)

        # Get drainage area
        drainage_area = router.get_Q() * grid.dx * grid.dx

    Author: B.G.
    """
    _check_topotoolbox()

    # Load raster using TopoToolbox
    try:
        go = ttb.read_tif(fname)
    except Exception as e:
        raise ValueError(f"Failed to read raster file '{fname}': {e}")

    # Convert to PyFastFlow Grid
    return gridobj_to_grid(go)


# Legacy function aliases for backward compatibility
def gridobj_to_gridfield(go):
    """
    Legacy alias for gridobj_to_grid().

    DEPRECATED: Use gridobj_to_grid() instead.
    This function is maintained for backward compatibility only.
    """
    import warnings

    warnings.warn(
        "gridobj_to_gridfield() is deprecated. Use gridobj_to_grid() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return gridobj_to_grid(go)


def raster_to_gridfield(fname):
    """
    Legacy alias for raster_to_grid().

    DEPRECATED: Use raster_to_grid() instead.
    This function is maintained for backward compatibility only.
    """
    import warnings

    warnings.warn(
        "raster_to_gridfield() is deprecated. Use raster_to_grid() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return raster_to_grid(fname)


def raster_to_numpy(fname):
    """
    Load a raster file and return its values as a numpy 2D array.

    Simple utility function that loads a raster file using TopoToolbox
    and returns only the elevation values as a 2D numpy array, without
    the additional metadata and Grid wrapper.

    Args:
        fname (str): Path to raster file (GeoTIFF, ASCII grid, etc.)
            Must be a format supported by TopoToolbox

    Returns:
        numpy.ndarray: 2D array containing raster values

    Raises:
        FileNotFoundError: If the specified file doesn't exist
        ValueError: If the raster file cannot be read or has invalid data
        ImportError: If topotoolbox is not installed

    Example:
        import pyfastflow as pf
        import numpy as np

        # Load elevation values as numpy array
        elevation = pf.io.raster_to_numpy('path/to/elevation.tif')

        # Save as .npy file
        np.save('elevation.npy', elevation)

    Author: B.G.
    """
    _check_topotoolbox()

    # Load raster using TopoToolbox
    try:
        go = ttb.read_tif(fname)
    except Exception as e:
        raise ValueError(f"Failed to read raster file '{fname}': {e}")

    # Return numpy array of values
    return go.z
