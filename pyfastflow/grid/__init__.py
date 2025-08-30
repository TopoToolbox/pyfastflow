"""
Grid management and spatial data structures for PyFastFlow.

This submodule provides data structures for managing 2D regular grids used in
geomorphological and hydrological modeling. Handles grid geometry, boundary
conditions, and coordinate systems for all PyFastFlow algorithms.

Core Classes:
- Grid: 2D regular grid with elevation data and boundary condition management

Key Features:
- Multiple boundary condition modes (normal, periodic EW/NS, custom per-node)
- Pool-based GPU memory management for elevation data
- Global constant configuration for kernel compilation
- Neighbor system integration for boundary-aware computations
- Support for custom boundary masks and periodic boundaries
- Built-in hillshading methods for terrain visualization
- Automatic temporary field management for GPU operations

Boundary Condition Modes:
- 'normal': Open boundaries where flow can exit at all edges
- 'periodic_EW': Periodic East-West boundaries (wraps left-right)
- 'periodic_NS': Periodic North-South boundaries (wraps top-bottom)
- 'custom': Per-node boundary conditions using boundary code array

Usage:
    import pyfastflow as pf
    import numpy as np

    # Create elevation data
    nx, ny, dx = 256, 256, 30.0
    elevation = np.random.rand(ny, nx) * 1000

    # Create grid with normal boundaries
    grid = pf.grid.Grid(nx, ny, dx, elevation, boundary_mode='normal')

    # Create grid with custom boundaries
    boundaries = np.ones((ny, nx), dtype=np.uint8)
    boundaries[0, :] = 3   # Top edge can drain out
    boundaries[-1, :] = 3  # Bottom edge can drain out
    boundaries[:, 0] = 1   # Left edge is closed
    boundaries[:, -1] = 1  # Right edge is closed

    custom_grid = pf.grid.Grid(
        nx, ny, dx, elevation,
        boundary_mode='custom',
        boundaries=boundaries
    )

    # Use grid with flow router
    router = pf.flow.FlowRouter(grid)

    # Built-in hillshading for visualization
    hillshade = grid.hillshade()                           # Perfect defaults - no config needed
    dramatic_hs = grid.hillshade(style='dramatic')         # Quick predefined styles
    multi_hs = grid.hillshade(multidirectional=True)       # Enhanced multidirectional
    custom_hs = grid.hillshade(altitude_deg=60, azimuth_deg=120, z_factor=1.5)

Author: B.G.
"""

from .gridfields import Grid

# Export main classes
__all__ = ["Grid"]
