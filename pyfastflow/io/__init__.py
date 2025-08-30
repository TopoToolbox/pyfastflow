"""
Input/Output and Data Integration Module for PyFastFlow

This submodule provides seamless integration between PyFastFlow and external
geospatial data sources and analysis packages. Enables easy data exchange,
file I/O operations, and workflow integration with the broader geospatial
Python ecosystem.

Core Integration:
- TopoToolbox: Seamless data exchange with PyTopoToolbox for DEM analysis
- Raster I/O: Direct loading of GeoTIFF, ASCII, and other raster formats
- Grid conversion: Convert between different grid data structures
- Metadata preservation: Maintain coordinate systems and projection information

Key Features:
- Optional dependency management: Graceful handling of missing packages
- Error checking: Robust validation of input data and formats
- Legacy compatibility: Backward-compatible function aliases
- Workflow integration: Enable combined analysis workflows

Available Functions:
- gridobj_to_grid: Convert TopoToolbox GridObj to PyFastFlow Grid
- raster_to_grid: Load raster files directly into PyFastFlow format
- gridobj_to_gridfield: [DEPRECATED] Legacy alias for gridobj_to_grid
- raster_to_gridfield: [DEPRECATED] Legacy alias for raster_to_grid

Dependencies:
Optional dependencies are handled gracefully - functions will raise informative
errors if required packages are not installed:

- topotoolbox: Required for TopoToolbox integration
  Install with: pip install topotoolbox or pip install pyfastflow[topotoolbox]

Usage Examples:

Basic TopoToolbox Integration:
    import pyfastflow as pf
    import topotoolbox as ttb

    # Load DEM with TopoToolbox
    dem_ttb = ttb.load_dem('elevation.tif')

    # Convert to PyFastFlow for high-performance simulation
    grid = pf.io.gridobj_to_grid(dem_ttb)
    router = pf.flow.FlowRouter(grid)

    # Run PyFastFlow simulation
    router.compute_receivers()
    router.accumulate_constant_Q(1.0)
    drainage_area = router.get_Q() * grid.dx ** 2

Direct Raster Loading:
    import pyfastflow as pf

    # Load DEM directly from file
    grid = pf.io.raster_to_grid('path/to/elevation.tif')

    # Immediate PyFastFlow usage
    router = pf.flow.FlowRouter(grid)
    flooder = pf.flood.Flooder(router)
    flooder.run_LS(N=1000)

Combined Workflow Example:
    import pyfastflow as pf
    import topotoolbox as ttb

    # TopoToolbox preprocessing
    dem = ttb.load_dem('raw_elevation.tif')
    dem_filled = ttb.fillsinks(dem)
    streams_ttb = ttb.streamnet(dem_filled, threshold=1000)

    # Convert to PyFastFlow for simulation
    grid = pf.io.gridobj_to_grid(dem_filled)
    router = pf.flow.FlowRouter(grid)

    # High-performance landscape evolution
    alpha_ = ti.field(ti.f32, shape=(grid.nx*grid.ny,))
    alpha_.fill(1e-5)

    for timestep in range(1000):
        router.compute_receivers()
        pf.erodep.SPL(router, alpha_, alpha_)

    # Back to TopoToolbox for analysis
    evolved_dem = ttb.GridObj(router.get_Z().reshape(grid.rshp))

Error Handling:
The module provides informative error messages when optional dependencies
are missing:

    try:
        grid = pf.io.raster_to_grid('elevation.tif')
    except ImportError as e:
        print("TopoToolbox not installed:", e)
        # Fallback to manual grid creation
        elevation = np.loadtxt('elevation.asc')
        grid = pf.grid.Grid(nx, ny, dx, elevation)

Scientific Applications:
This integration enables powerful workflows combining TopoToolbox's mature
DEM analysis capabilities with PyFastFlow's high-performance simulation:

- Preprocessing: Use TopoToolbox for data cleaning, coordinate transformations
- Simulation: Use PyFastFlow for large-scale, GPU-accelerated modeling
- Analysis: Return to TopoToolbox for specialized geomorphometric analysis
- Validation: Compare results between different algorithmic implementations

Author: B.G.
"""

# Import functions with proper error handling
from .ttbwrp import (
    TOPOTOOLBOX_AVAILABLE,
    gridobj_to_grid,
    gridobj_to_gridfield,  # Deprecated alias
    raster_to_grid,
    raster_to_gridfield,  # Deprecated alias
    raster_to_numpy,
)

# Export public API
__all__ = [
    # Primary functions
    "gridobj_to_grid",
    "raster_to_grid",
    "raster_to_numpy",
    # Legacy aliases (deprecated)
    "gridobj_to_gridfield",
    "raster_to_gridfield",
    # Utility
    "TOPOTOOLBOX_AVAILABLE",
]
