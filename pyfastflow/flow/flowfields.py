"""
High-level FlowRouter class for GPU-accelerated flow routing.

Provides an object-oriented interface for performing hydrological flow routing
on digital elevation models. Handles field allocation, boundary conditions,
and orchestrates the flow routing pipeline including receiver computation,
lake flow processing, and flow accumulation.

Author: B.G.
"""

import taichi as ti
import numpy as np
import math
from . import environment as env
from .. import constants as cte
from .. import general_algorithms as gena
from . import downstream_propag as dpr
from . import lakeflow as lf
from . import util_taichi as ut
from . import fill_topo as fl
from . import receivers as rcv




class FlowRouter:
	"""
	High-level interface for GPU-accelerated flow routing computations.
	
	Handles field allocation, boundary conditions, and orchestrates the complete
	flow routing pipeline including receiver computation, lake flow processing,
	and flow accumulation using parallel algorithms.
	
	Author: B.G.
	"""

	def __init__(self, nx, ny, dx, boundary_mode = 'normal', boundaries = None, lakeflow = True, stochastic_receivers = False):
		"""
		Initialize FlowRouter with grid parameters and boundary conditions.
		
		Args:
			nx: Number of grid columns
			ny: Number of grid rows
			dx: Grid spacing
			boundary_mode: 'normal', 'periodic_EW', 'periodic_NS', or 'custom'
			boundaries: Custom boundary array (if boundary_mode='custom')
			lakeflow: Enable lake flow processing for depression handling
			stochastic_receivers: Enable stochastic receiver selection
			
		Author: B.G.
		"""

		# Store grid parameters
		self.nx = nx  # Number of columns
		self.ny = ny  # Number of rows
		self.dx = dx  # Grid spacing
		self.rshp = (ny,nx)  # Reshape tuple for converting 1D arrays to 2D
		
		# Store configuration parameters
		self.boundary_mode = boundary_mode  # Boundary condition type
		self.boundaries = boundaries  # Custom boundary array (if applicable)
		self.lakeflow = lakeflow  # Enable depression handling
		self.stochastic_receivers = stochastic_receivers  # Enable random receiver selection

		# Initialize Taichi field builders for memory management
		self.fb_flow = ti.FieldsBuilder()  # Builder for flow computation fields
		self.fb_lake = None  # Builder for lake flow fields
		self.snodetree_flow = None  # Will hold finalized flow field structure
		self.snodetree_lake = None  # Will hold finalized lake field structure

		# ====== CORE FLOW COMPUTATION FIELDS ======
		# These fields are required for all flow routing operations
		
		# Elevation field - input topography
		self.z = ti.field(ti.f32)
		self.fb_flow.dense(ti.i,(nx*ny)).place(self.z)
		
		# Flow accumulation fields (primary and ping-pong buffer)
		self.Q = ti.field(ti.f32)  # Primary accumulation values
		self.fb_flow.dense(ti.i,(nx*ny)).place(self.Q)
		self.Q_ = ti.field(ti.f32)  # Alternate buffer for rake-compress
		self.fb_flow.dense(ti.i,(nx*ny)).place(self.Q_)
		
		# Gradient field - stores steepest descent gradients
		self.gradient = ti.field(ti.f32)
		self.fb_flow.dense(ti.i,(nx*ny)).place(self.gradient)

		# ====== FLOW GRAPH FIELDS ======
		# These fields represent the drainage network structure
		
		# Receiver field - downstream flow direction for each node
		self.receivers = ti.field(ti.i32)
		self.fb_flow.dense(ti.i,(nx*ny)).place(self.receivers)
		
		# Ping-pong state tracking for rake-compress algorithm
		self.src = ti.field(ti.i32)  # Tracks which buffer to use per node
		self.fb_flow.dense(ti.i,(nx*ny)).place(self.src)
		
		# Donor counting fields (primary and alternate for ping-pong)
		self.ndonors = ti.field(ti.i32)  # Number of upstream donors per node
		self.fb_flow.dense(ti.i,(nx*ny)).place(self.ndonors)
		self.ndonors_ = ti.field(ti.i32)  # Alternate buffer for ping-pong
		self.fb_flow.dense(ti.i,(nx*ny)).place(self.ndonors_)

		# Donor list fields (each node can have up to 4 donors)
		self.donors = ti.field(ti.i32)  # Primary donor lists (4 per node)
		self.fb_flow.dense(ti.i, (nx*ny*4)).place(self.donors)
		self.donors_ = ti.field(ti.i32)  # Alternate donor lists for ping-pong
		self.fb_flow.dense(ti.i, (nx*ny*4)).place(self.donors_)

		# Finalize the flow field structure (allocates GPU memory)
		self.snodetree_flow = self.fb_flow.finalize()

		# Initialize gradient field to zero
		self.gradient.fill(0.)

		# ====== OPTIONAL LAKE FLOW FIELDS ======
		# These fields are only allocated if depression handling is enabled
		if self.lakeflow:

			self.fb_lake = ti.FieldsBuilder()

			# Modified elevation field for lake outlet computation
			self.z_ = ti.field(ti.f32)
			self.fb_lake.dense(ti.i,(nx*ny)).place(self.z_)
			self.z__ = ti.field(ti.f32)
			self.fb_lake.dense(ti.i,(nx*ny)).place(self.z__)

			# Additional receiver arrays for lake flow processing
			self.receivers_ = ti.field(ti.i32)  # Temporary receiver storage
			self.fb_lake.dense(ti.i,(nx*ny)).place(self.receivers_)
			self.receivers__ = ti.field(ti.i32)  # Second temporary receiver array
			self.fb_lake.dense(ti.i,(nx*ny)).place(self.receivers__)

			# Basin identification fields
			self.bid = ti.field(ti.i32)  # Basin ID for each node
			self.fb_lake.dense(ti.i,(nx*ny)).place(self.bid)
			self.basin_saddlenode = ti.field(ti.i32)  # Saddle node per basin
			self.fb_lake.dense(ti.i,(nx*ny)).place(self.basin_saddlenode)

			# Packed saddle and outlet information (uses f32+i32 packing)
			self.outlet = ti.field(ti.i64)  # Packed outlet info per basin
			self.fb_lake.dense(ti.i,(nx*ny)).place(self.outlet)
			self.basin_saddle = ti.field(ti.i64)  # Packed saddle info per basin
			self.fb_lake.dense(ti.i,(nx*ny)).place(self.basin_saddle)

			# Border detection and carving algorithm fields
			self.is_border = ti.field(ti.u1)  # Marks basin border nodes
			self.fb_lake.dense(ti.i,(nx*ny)).place(self.is_border)
			self.tag = ti.field(ti.u1)  # Primary tagging for carving
			self.fb_lake.dense(ti.i,(nx*ny)).place(self.tag)
			self.tag_ = ti.field(ti.u1)  # Alternate tagging for carving
			self.fb_lake.dense(ti.i,(nx*ny)).place(self.tag_)
			self.rerouted = ti.field(ti.u1)  # Keep track of rerouted nodes
			self.fb_lake.dense(ti.i,(nx*ny)).place(self.rerouted)

			# Finalize lake flow field structure
			self.snodetree_lake = self.fb_lake.finalize()

			# Convergence flag for iterative algorithms (single scalar)
			self.change = ti.field(ti.u1, shape = ())

		# ====== GLOBAL CONFIGURATION SETUP ======
		# Set global constants that affect all flow computations
		cte.NX = nx  # Grid width
		cte.NY = ny  # Grid height
		cte.DX = dx  # Grid spacing
		cte.RAND_RCV = True if self.stochastic_receivers else False
		
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

		# Initialize the FastFlow environment (compiles neighbor functions)
		env.initialise()
		


	def free(self):
		"""
		Free allocated GPU memory and destroy field structures.
		
		Properly cleans up all allocated Taichi fields to prevent memory leaks.
		Should be called when the FlowRouter is no longer needed.
		
		Author: B.G.
		"""
		# Free core flow computation fields
		self.snodetree_flow.destroy()
		
		# Free lake flow fields if they were allocated
		if(self.lakeflow):
			self.snodetree_lake.destroy()

	def set_z(self, z):
		"""
		Set elevation data from numpy array.
		
		Converts 2D numpy elevation array to 1D Taichi field format.
		Automatically handles type conversion to float32.
		
		Args:
			z: 2D numpy array of elevation values (ny, nx)
			
		Author: B.G.
		"""
		self.z.from_numpy(z.ravel().astype(np.float32))

	def compute_receivers(self):
		"""
		Compute steepest descent receivers for all grid nodes.
		
		Determines the downstream flow direction for each node based on
		steepest descent algorithm. Results are stored in self.receivers
		and gradients in self.gradient.
		
		Author: B.G.
		"""
		rcv.compute_receivers(self.z, self.receivers, self.gradient)

	def recompute_receivers_ignore_rerouted(self):
		"""
		Compute steepest descent receivers for all grid nodes.
		
		Determines the downstream flow direction for each node based on
		steepest descent algorithm. Results are stored in self.receivers
		and gradients in self.gradient.
		
		Author: B.G.
		"""
		rcv.compute_receivers_ignore_rerouted(self.z, self.receivers, self.gradient, self.rerouted)

	def reroute_flow(self, carve = True):
		"""
		Process lake flow to handle depressions and closed basins.
		
		Implements depression handling algorithms to route flow through
		or around topographic depressions. Supports both carving and
		filling approaches.
		
		Args:
			carve: Use carving (True) or filling (False) for depression handling
			       Carving creates channels through saddle points
			       Filling jumps flow directly to basin outlets
			
		Raises:
			RuntimeError: If lakeflow was not enabled during initialization
			
		Author: B.G.
		"""
		if(self.lakeflow == False):
			raise RuntimeError('Flow field not compiled for lakeflow')

		# Call the main lake flow routing algorithm
		lf.reroute_flow(self.bid, self.receivers, self.receivers_, self.receivers__,
        self.z, self.z_, self.is_border, self.outlet, self.basin_saddle, 
        self.basin_saddlenode, self.tag, self.tag_, self.change, self.rerouted, carve = carve)

	def accumulate_constant_Q(self, value, area = True):
		"""
		Accumulate constant flow values using parallel rake-compress algorithm.
		
		Performs flow accumulation with uniform input values using the
		rake-and-compress algorithm for efficient parallel computation.
		
		Args:
			value: Constant flow value to accumulate at each node
			area: If True, multiply by grid cell area (dx²) for area-based flow
			      If False, use raw value for unit-based flow
			
		Note:
			Results are stored in self.Q and can be accessed via get_Q()
			
		Author: B.G.
		"""
		# Calculate number of iterations needed (log₂ of grid size)
		logn = math.ceil(math.log2(self.nx*self.ny))+1

		# Initialize arrays for rake-compress algorithm
		self.ndonors.fill(0)  # Reset donor counts
		self.src.fill(0)      # Reset ping-pong state
		
		# Initialize flow values (multiply by area if requested)
		self.Q.fill(value*(cte.DX * cte.DX if area else 1.))
		
		# Build donor-receiver relationships from receiver array
		dpr.rcv2donor(self.receivers, self.donors, self.ndonors)

		# Rake-compress iterations for parallel tree accumulation
		# Each iteration doubles the effective path length being compressed
		for i in range(logn+1):
			dpr.rake_compress_accum(self.donors, self.ndonors, self.Q, self.src,
			                   self.donors_, self.ndonors_, self.Q_, i)

		# Final fuse step to consolidate results from ping-pong buffers
		# Merge accumulated values from working arrays back to primary array
		gena.fuse(self.Q, self.src, self.Q_, logn)

	def accumulate_custom_donwstream(self, Acc:ti.template()):
		"""
		Acc needs to be accumulated to the OG value to accumulate
		"""
		# Calculate number of iterations needed (log₂ of grid size)
		logn = math.ceil(math.log2(self.nx*self.ny))+1

		# Initialize arrays for rake-compress algorithm
		self.ndonors.fill(0)  # Reset donor counts
		self.src.fill(0)      # Reset ping-pong state

		# Build donor-receiver relationships from receiver array
		dpr.rcv2donor(self.receivers, self.donors, self.ndonors)

		# Rake-compress iterations for parallel tree accumulation
		# Each iteration doubles the effective path length being compressed
		for i in range(logn+1):
			dpr.rake_compress_accum(self.donors, self.ndonors, Acc, self.src,
			                   self.donors_, self.ndonors_, self.Q_, i)

		# Final fuse step to consolidate results from ping-pong buffers
		# Merge accumulated values from working arrays back to primary array
		gena.fuse(Acc, self.src, self.Q_, logn)


	def accumulate_constant_Q_stochastic(self, value, area = True, N = 4):
		"""
		Accumulate constant flow values using parallel rake-compress algorithm.
		
		Performs flow accumulation with uniform input values using the
		rake-and-compress algorithm for efficient parallel computation.
		
		Args:
			value: Constant flow value to accumulate at each node
			area: If True, multiply by grid cell area (dx²) for area-based flow
			      If False, use raw value for unit-based flow
			
		Note:
			Results are stored in self.Q and can be accessed via get_Q()
			
		Author: B.G.
		"""
		self.z_.fill(0.)

		# Calculate number of iterations needed (log₂ of grid size)
		logn = math.ceil(math.log2(self.nx*self.ny))+1
		for __ in range(N):
			self.compute_receivers()

			# Initialize arrays for rake-compress algorithm
			self.ndonors.fill(0)  # Reset donor counts
			self.src.fill(0)      # Reset ping-pong state
			
			# Initialize flow values (multiply by area if requested)
			self.Q.fill(value*(cte.DX * cte.DX if area else 1.))
			
			# Build donor-receiver relationships from receiver array
			dpr.rcv2donor(self.receivers, self.donors, self.ndonors)

			# Rake-compress iterations for parallel tree accumulation
			# Each iteration doubles the effective path length being compressed
			for i in range(logn+1):
				dpr.rake_compress_accum(self.donors, self.ndonors, self.Q, self.src,
				                   self.donors_, self.ndonors_, self.Q_, i)

			# Final fuse step to consolidate results from ping-pong buffers
			# Merge accumulated values from working arrays back to primary array
			gena.fuse(self.Q, self.src, self.Q_, logn)

			ut.add_B_to_weighted_A(self.z_, self.Q, 1./N)

		self.Q.copy_from(self.z_)



	def fill_z(self, epsilon=1e-3):
		fl.topofill(self, epsilon=epsilon, custom_z = None)

	def rec2rec_(self, second = False):

		self.receivers_.copy_from(self.receivers)
		if second:
			self.receivers__.copy_from(self.receivers)

	def get_Q(self):
		"""
		Get flow accumulation results as 2D numpy array.
		
		Returns:
			numpy.ndarray: 2D array of flow accumulation values (ny, nx)
			
		Author: B.G.
		"""
		return self.Q.to_numpy().reshape(self.rshp)


	def get_Z(self):
		"""
		Get flow accumulation results as 2D numpy array.
		
		Returns:
			numpy.ndarray: 2D array of flow accumulation values (ny, nx)
			
		Author: B.G.
		"""
		return self.z.to_numpy().reshape(self.rshp)
