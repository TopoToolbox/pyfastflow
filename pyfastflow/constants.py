"""
Global constants and configuration parameters for PyFastFlow.

This module defines compile-time constants and runtime parameters used throughout
PyFastFlow for grid dimensions, boundary conditions, hydrodynamic parameters, and
landscape evolution settings. Constants are centralized here to ensure consistency
across all GPU kernels and computational modules.

Performance Note:
Compile-time constants (marked as such) are embedded in GPU kernels during compilation
and can improve performance by up to 30% compared to runtime parameters by enabling
compiler optimizations and reducing memory accesses.

Constant Categories:
- Grid Constants: Domain size, spacing, and boundary condition modes
- Hydrodynamic Constants: Manning's roughness, precipitation, time steps
- Erosion Constants: Stream Power Law parameters, uplift, deposition
- General Constants: Physical constants (gravity, etc.)

Boundary Condition Modes:
- BOUND_MODE = 0: Open boundaries (flow can exit at all edges)
- BOUND_MODE = 1: Periodic East-West (wraps left-right borders)
- BOUND_MODE = 2: Periodic North-South (wraps top-bottom borders)
- BOUND_MODE = 3: Custom per-node boundaries (requires boundary array)

Custom Boundary Codes (when BOUND_MODE = 3):
- 0: No Data (invalid/masked node)
- 1: Normal interior node (cannot leave domain)
- 3: Outlet node (can leave domain)
- 7: Inlet node (can only receive flow, acts normal otherwise)
- 9: Periodic connection (use with caution)

Usage:
    import pyfastflow.constants as cte

    # Access constants in Python code
    grid_size = cte.NX * cte.NY
    cell_area = cte.DX * cte.DX

    # Constants are automatically available in Taichi kernels
    @ti.kernel
    def my_kernel():
        for i in range(cte.NX * cte.NY):
            # Use constants directly
            cell_area = cte.DX ** 2

    # Custom boundary setup
    import numpy as np
    boundaries = np.ones((cte.NY, cte.NX), dtype=np.uint8)
    boundaries[0, :] = 3   # Top edge: outlets
    boundaries[-1, :] = 3  # Bottom edges: outlets
    cte.init_custom_boundaries(boundaries.flatten())

Author: B.G.
"""

import numpy as np
import taichi as ti

#########################################
###### UTILS CONSTANTS ##################
#########################################

INITIALISED = False


#########################################
###### GRID CONSTANTS ###################
#########################################

# Grid spacing (uniform cell size in meters)
# Compile-time constant: embedded in GPU kernels for performance
DX = 1.0

# Number of columns in the grid (x-direction)
# Compile-time constant: affects kernel compilation and memory layout
NX = 512

# Number of rows in the grid (y-direction)
# Compile-time constant: affects kernel compilation and memory layout
NY = 512

# Enable stochastic receiver selection for flow routing uncertainty quantification
# When True, enables probabilistic flow directions instead of deterministic steepest descent
RAND_RCV = False

# Boundary condition mode:
# 0 -> open boundaries (flow can exit at edges)
# 1 -> periodic East-West (wraps around left-right borders)
# 2 -> periodic North-South (wraps around top-bottom borders)
# 3 -> custom boundaries (per-node boundary codes)
#      Custom boundary codes:
#      0: No Data (invalid node)
#      1: Normal node (cannot leave domain)
#      3: Can leave domain (outlet)
#      7: Can only enter (inlet - acts as normal for other operations)
#      9: Periodic (risky - ensure opposite direction exists on border)
BOUND_MODE = 0

# Global field for custom boundary conditions (initialized when needed)
boundaries = None
_snodetree_boundaries = None


def init_custom_boundaries(tboundaries: np.ndarray):
    """
    Initialize custom boundary conditions from numpy array.

    Args:
            tboundaries: Boundary code array (uint8) with shape (NX*NY,)

    Note:
            Sets BOUND_MODE to 3 and creates Taichi field for boundary codes

    Author: B.G.
    """
    global boundaries, _snodetree_boundaries, BOUND_MODE
    # Clean up existing boundary field if it exists
    if _snodetree_boundaries is not None:
        try:
            _snodetree_boundaries.destroy()
        except (AttributeError, RuntimeError):
            pass
        _snodetree_boundaries = None

    # Create new boundary field
    fb1 = ti.FieldsBuilder()
    boundaries = ti.field(dtype=ti.u8)
    fb1.dense(ti.i, NX * NY).place(boundaries)
    _snodetree_boundaries = fb1.finalize()  # Finalize field structure
    boundaries.from_numpy(tboundaries)  # Copy boundary data
    BOUND_MODE = 3  # Switch to custom boundary mode


#########################################
###### HYDRODYNAMIC CONSTANTS ###########
#########################################

# Default precipitation rate for constant rainfall scenarios (m/s)
# Converts 10 mm/hr to m/s: effective precipitation rate
PREC = 10 * 1e-3 / 3600  # 10 mm/hr = 2.78e-6 m/s

# Manning's roughness coefficient for flow resistance (dimensionless)
# Default value suitable for natural channels and overland flow
MANNING = 0.033  # Typical for grassed surfaces and natural channels

# Edge slope for boundary nodes that can drain out of the domain (dimensionless)
# Used to compute flow velocity at domain outlets
EDGESW = 1e-2  # 1% slope

## GraphFlood-specific parameters ##

# Time step for GraphFlood hydrodynamic solver (seconds)
# Implicit solver allows larger time steps than explicit methods
DT_HYDRO = 5e-3  # 5 milliseconds

## LisFlood-specific parameters ##

# Time step for LisFlood explicit solver (seconds)
# Limited by CFL condition for numerical stability
DT_HYDRO_LS = 1e-1  # 0.1 seconds

# Minimum water depth threshold for flow computations (meters)
# Below this depth, flow is considered negligible
HFLOW_THRESHOLD = 1e-3  # 1 mm

# Froude number limit for flow stability in shallow water equations
# Fr = v/sqrt(gh) where v=velocity, g=gravity, h=depth
FROUDE_LIMIT = 1.0  # Subcritical flow limit


#########################################
###### LANDSCAPE EVOLUTION CONSTANTS ####
#########################################

# Deposition coefficient for transport-limited erosion (dimensionless)
# Controls the rate of sediment deposition when transport capacity is exceeded
KD = 1e-2

# Bedrock erodibility coefficient for Stream Power Law (m^(1-2m)/s)
# Units depend on drainage area exponent m: [L^(1-2m)/T]
# Default value suitable for moderate bedrock strength
KR = 2e-5
KS = 3e-5

# Time step for landscape evolution models (years)
# Large time steps possible due to implicit numerical schemes
DT_SPL = 1e3  # 1000 years per time step

# Drainage area exponent in Stream Power Law: E = K * A^m * S^n
# Controls sensitivity of erosion to drainage area (discharge proxy)
MEXP = 0.45  # Typical values range from 0.3 to 0.7

# Slope exponent in Stream Power Law: E = K * A^m * S^n
# Controls sensitivity of erosion to local slope
# Currently unused in implementation (implicit in receiver selection)
NEXP = 1.0  # Theoretical value, not used in current algorithms

#########################################
###### PHYSICAL CONSTANTS ###############
#########################################

# Gravitational acceleration (m/s²)
# Used in shallow water equations and slope calculations
GRAVITY = 9.81  # Standard Earth gravity
