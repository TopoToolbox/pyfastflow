import taichi as ti
import numpy as np
from .. import constants as cte
import pyfastflow as pf

class Grid:
	"""
	Grid data structure for 2D regular grids with elevation data and boundary conditions.
	
	Manages a 2D regular grid for geomorphological and hydrological modeling. Handles
	elevation data storage using pool-based GPU fields and configures global constants
	for boundary conditions and grid parameters.
	
	The Grid class serves as the foundation for flow routing, flood modeling, and
	landscape evolution by providing consistent grid geometry and boundary handling
	across all PyFastFlow algorithms.
	
	Attributes:
		nx (int): Number of grid columns (x-direction)
		ny (int): Number of grid rows (y-direction)  
		dx (float): Grid spacing in meters (uniform cell size)
		rshp (tuple): Reshape tuple (ny, nx) for converting 1D to 2D arrays
		z (TPField): Elevation field allocated from pool (GPU field)
		boundary_mode (str): Boundary condition mode string
		metadata (dict): Optional metadata for transforms, projections, etc.
		
	Methods:
		hillshade(): Comprehensive hillshading with single/multi-directional options and predefined styles
		
	Author: B.G.
	"""

	def __init__(self, nx:int, ny:int, dx:float, z:np.ndarray, boundary_mode = 'normal', boundaries = None, metadata = None):
		"""
		Initialize Grid with dimensions, elevation data, and boundary conditions.
		
		Creates a regular 2D grid with specified dimensions and elevation data.
		Configures global constants used throughout PyFastFlow and allocates
		GPU memory for elevation storage using the pool system.
		
		Args:
			nx (int): Number of grid columns (x-direction)
			ny (int): Number of grid rows (y-direction)
			dx (float): Grid spacing in meters (uniform cell size)
			z (np.ndarray): Elevation data as 2D numpy array shape (ny, nx)
			boundary_mode (str, optional): Boundary condition mode. Default: 'normal'
				- 'normal': Open boundaries (flow can exit at edges)
				- 'periodic_EW': Periodic East-West (wraps left-right)
				- 'periodic_NS': Periodic North-South (wraps top-bottom)
				- 'custom': Custom per-node boundaries (requires boundaries array)
			boundaries (np.ndarray, optional): Custom boundary codes when boundary_mode='custom'.
				Shape (ny, nx) with uint8 codes: 0=no data, 1=normal, 3=outlet, 7=inlet, 9=periodic
			metadata (dict, optional): Additional metadata for transforms, projections, etc.
				
		Note:
			- Sets global constants (cte.NX, cte.NY, cte.DX, cte.BOUND_MODE) used by all kernels
			- Elevation data is stored in pool-allocated GPU field for efficient computation
			- Custom boundaries are initialized in global state if boundary_mode='custom'
			- Triggers neighbor system compilation for the specified boundary conditions
			
		Example:
			elevation = np.random.rand(100, 100) * 1000  # 100x100 grid, 0-1000m elevation
			grid = Grid(100, 100, 30.0, elevation, boundary_mode='normal')
			
		Author: B.G.
		"""

		# Store grid parameters
		self.nx = nx  # Number of columns
		self.ny = ny  # Number of rows
		self.dx = dx  # Grid spacing
		self.rshp = (ny,nx)  # Reshape tuple for converting 1D arrays to 2D

		self.boundary_mode = boundary_mode
		self.boundaries = boundaries  # Store boundary array for custom mode

		# Future placeholder to store for example the transform or the OG ttb grid
		self.metadata = metadata

		# Set global constants that affect all flow computations
		cte.NX = nx  # Grid width
		cte.NY = ny  # Grid height
		cte.DX = dx  # Grid spacing

		# Fetching from pool
		self.z = pf.pool.taipool.get_tpfield(dtype = ti.f32, shape = (nx*ny) )
		self.z.field.from_numpy(z.ravel())

		# ====== GLOBAL CONFIGURATION SETUP ======
			
		# Set boundary mode based on string parameter
		cte.BOUND_MODE = 0 if self.boundary_mode == 'normal' else (
			1 if self.boundary_mode == 'periodic_EW' else (
				2 if self.boundary_mode == 'periodic_NS' else (
					3 if self.boundary_mode == 'custom' else 0
				)
			)
		)
		
		# Initialize custom boundary conditions if specified
		if(cte.BOUND_MODE == 3):
			cte.init_custom_boundaries(self.boundaries.ravel())


		pf.grid.neighbourer_flat.compile_neighbourer()

	def hillshade(self, altitude_deg=45.0, azimuth_deg=315.0, z_factor=1.0, 
	              multidirectional=False, azimuths_deg=None, weights=None, style=None):
		"""
		Generate hillshading for terrain visualization with comprehensive options.
		
		Creates hillshaded relief visualization with automatic GPU memory management.
		Supports single-direction, multidirectional, and predefined styles. Perfect
		default parameters provide excellent results without any configuration.
		
		Args:
			altitude_deg (float, optional): Sun altitude angle in degrees (0-90°).
				Higher values create steeper lighting. Default: 45.0
			azimuth_deg (float, optional): Sun azimuth angle in degrees (0-360°).
				0° = North, 90° = East, 180° = South, 270° = West. Default: 315.0 (NW)
			z_factor (float, optional): Vertical exaggeration factor.
				Values > 1.0 enhance terrain relief. Default: 1.0
			multidirectional (bool, optional): Enable multidirectional lighting for enhanced
				detail. Uses 4 cardinal directions by default. Default: False
			azimuths_deg (list, optional): Custom azimuth angles for multidirectional lighting.
				Automatically enables multidirectional mode. Default: [315, 45, 135, 225]
			weights (list, optional): Relative weights for each azimuth direction.
				Must match length of azimuths_deg. Default: Equal weights
			style (str, optional): Predefined style for quick results. Options:
				'default', 'dramatic', 'subtle', 'east', 'south', 'multi', 'multi_smooth'.
				Overrides other parameters when specified. Default: None
				
		Returns:
			np.ndarray: 2D hillshade array with shape (ny, nx) and values in [0, 1] range.
				0 = fully shadowed, 1 = fully illuminated
				
		Note:
			- Uses pool-based temporary fields for efficient GPU memory management
			- All temporary fields are automatically allocated and released
			- Respects grid boundary conditions (normal, periodic, custom)
			- Parameter precedence: style > multidirectional/azimuths_deg > single-direction
			
		Examples:
			import matplotlib.pyplot as plt
			
			# Perfect defaults - no parameters needed
			hillshade = grid.hillshade()
			plt.imshow(hillshade, cmap='gray')
			
			# Quick predefined styles
			dramatic = grid.hillshade(style='dramatic')
			multi = grid.hillshade(style='multi')
			
			# Custom single-direction lighting
			east_light = grid.hillshade(altitude_deg=60, azimuth_deg=90, z_factor=2.0)
			
			# Multidirectional with default 4 directions
			enhanced = grid.hillshade(multidirectional=True, z_factor=1.5)
			
			# Custom multidirectional with specific azimuths and weights
			custom = grid.hillshade(
				azimuths_deg=[315, 45, 135, 225],
				weights=[0.4, 0.2, 0.2, 0.2],  # Emphasize NW light
				altitude_deg=60
			)
			
		Author: B.G.
		"""
		from ._hswrapper import compute_hillshade
		return compute_hillshade(self, altitude_deg, azimuth_deg, z_factor,
		                        multidirectional, azimuths_deg, weights, style)