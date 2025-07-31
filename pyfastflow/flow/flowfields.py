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
import pyfastflow.general_algorithms.util_taichi as ut
from . import fill_topo as fl
from . import receivers as rcv
import pyfastflow as pf




class FlowRouter:
	"""
	High-level interface for GPU-accelerated flow routing computations.
	
	Handles field allocation, boundary conditions, and orchestrates the complete
	flow routing pipeline including receiver computation, lake flow processing,
	and flow accumulation using parallel algorithms.
	
	Author: B.G.
	"""

	def __init__(self, grid, lakeflow = True):
		"""
		Initialize FlowRouter with grid parameters and boundary conditions.
		
		Args:
			grid: the GridField dobject of the assiciated grid
			lakeflow: Enable lake flow processing for depression handling
			
		Author: B.G.
		"""

		self.grid = grid

		# Store configuration parameters
		self.lakeflow = lakeflow  # Enable depression handling

		# Flow accumulation fields (primary and ping-pong buffer)
		self.Q = pf.pool.taipool.get_tpfield(dtype = ti.f32, shape = (self.nx*self.ny) )
		
		# Receiver field - downstream flow direction for each node
		self.receivers = pf.pool.taipool.get_tpfield(dtype = ti.i32, shape = (self.nx*self.ny) )

		

	@property
	def nx(self):
		return self.grid.nx

	@property
	def ny(self):
		return self.grid.ny

	@property
	def dx(self):
		return self.grid.dx

	@property
	def rshp(self):
		return self.grid.rshp


	def compute_receivers(self):
		"""
		Compute steepest descent receivers for all grid nodes.
		
		Determines the downstream flow direction for each node based on
		steepest descent algorithm. Results are stored in self.receivers
		and gradients in self.gradient.
		
		Author: B.G.
		"""
		rcv.compute_receivers(self.grid.z, self.receivers)

	def compute_stochastic_receivers(self):
		"""
		Compute steepest descent receivers for all grid nodes.
		
		Determines the downstream flow direction for each node based on
		steepest descent algorithm. Results are stored in self.receivers
		and gradients in self.gradient.
		
		Author: B.G.
		"""
		rcv.compute_receivers_stochastic(self.grid.z, self.receivers)

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

		# Querying fields from the pool
		bid              = pf.pool.taipool.get_tpfield(dtype = ti.i32, shape = (self.nx*self.ny) )
		receivers_       = pf.pool.taipool.get_tpfield(dtype = ti.i32, shape = (self.nx*self.ny) )
		receivers__      = pf.pool.taipool.get_tpfield(dtype = ti.i32, shape = (self.nx*self.ny) )
		z_               = pf.pool.taipool.get_tpfield(dtype = ti.f32, shape = (self.nx*self.ny) )
		is_border        = pf.pool.taipool.get_tpfield(dtype = ti.u1,  shape = (self.nx*self.ny) )
		outlet           = pf.pool.taipool.get_tpfield(dtype = ti.i64, shape = (self.nx*self.ny) )
		basin_saddle     = pf.pool.taipool.get_tpfield(dtype = ti.i64, shape = (self.nx*self.ny) )
		basin_saddlenode = pf.pool.taipool.get_tpfield(dtype = ti.i32, shape = (self.nx*self.ny) )
		tag              = pf.pool.taipool.get_tpfield(dtype = ti.u1,  shape = (self.nx*self.ny) )
		tag_             = pf.pool.taipool.get_tpfield(dtype = ti.u1,  shape = (self.nx*self.ny) )
		change 			 = pf.pool.taipool.get_tpfield(dtype = ti.i32, shape =        ()         )
		rerouted         = pf.pool.taipool.get_tpfield(dtype = ti.u1,  shape = (self.nx*self.ny) )

		# Call the main lake flow routing algorithm
		lf.reroute_flow(bid, self.receivers, receivers_, receivers__,
        self.grid.z, z_, is_border, outlet, basin_saddle, 
        basin_saddlenode, tag, tag_, change, rerouted, carve = carve)

        bid.release()
		receivers_.release()
		receivers__.release()
		z_.release()
		is_border.release()
		outlet.release()
		basin_saddle.release()
		basin_saddlenode.release()
		tag.release()
		tag_.release()
		change.release()
		rerouted.release()

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

		# Get temporary fields from pool for rake-compress algorithm
		ndonors  = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(self.nx*self.ny))
		donors   = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(self.nx*self.ny*4))
		src      = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(self.nx*self.ny))
		donors_  = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(self.nx*self.ny*4))
		ndonors_ = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(self.nx*self.ny))
		Q_       = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(self.nx*self.ny))

		# Initialize arrays for rake-compress algorithm
		ndonors.fill(0)  # Reset donor counts
		src.fill(0)      # Reset ping-pong state
		
		# Initialize flow values (multiply by area if requested)
		self.Q.fill(value*(cte.DX * cte.DX if area else 1.))
		
		# Build donor-receiver relationships from receiver array
		dpr.rcv2donor(self.receivers, donors, ndonors)

		# Rake-compress iterations for parallel tree accumulation
		# Each iteration doubles the effective path length being compressed
		for i in range(logn+1):
			dpr.rake_compress_accum(donors, ndonors, self.Q, src,
			                   donors_, ndonors_, Q_, i)

		# Final fuse step to consolidate results from ping-pong buffers
		# Merge accumulated values from working arrays back to primary array
		gena.fuse(self.Q, src, Q_, logn)

		# Release all temporary fields back to pool
		ndonors.release()
		donors.release()
		src.release()
		donors_.release()
		ndonors_.release()
		Q_.release()

	def accumulate_custom_donwstream(self, Acc:ti.template()):
		"""
		Acc needs to be accumulated to the OG value to accumulate
		
		Author: B.G.
		"""
		# Calculate number of iterations needed (log₂ of grid size)
		logn = math.ceil(math.log2(self.nx*self.ny))+1

		# Get temporary fields from pool for rake-compress algorithm
		ndonors    = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(self.nx*self.ny))
		donors     = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(self.nx*self.ny*4))
		src        = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(self.nx*self.ny))
		donors_    = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(self.nx*self.ny*4))
		ndonors_   = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(self.nx*self.ny))
		Q_         = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(self.nx*self.ny))

		# Initialize arrays for rake-compress algorithm
		ndonors.fill(0)  # Reset donor counts
		src.fill(0)      # Reset ping-pong state

		# Build donor-receiver relationships from receiver array
		dpr.rcv2donor(self.receivers, donors, ndonors)

		# Rake-compress iterations for parallel tree accumulation
		# Each iteration doubles the effective path length being compressed
		for i in range(logn+1):
			dpr.rake_compress_accum(donors, ndonors, Acc, src,
			                   donors_, ndonors_, Q_, i)

		# Final fuse step to consolidate results from ping-pong buffers
		# Merge accumulated values from working arrays back to primary array
		gena.fuse(Acc, src, Q_, logn)

		# Release all temporary fields back to pool
		ndonors.release()
		donors.release()
		src.release()
		donors_.release()
		ndonors_.release()
		Q_.release()


	def accumulate_constant_Q_stochastic(self, value, area = True, N = 4):
		"""
		Accumulate constant flow values using parallel rake-compress algorithm.
		
		Performs flow accumulation with uniform input values using the
		rake-and-compress algorithm for efficient parallel computation.
		
		Args:
			value: Constant flow value to accumulate at each node
			area: If True, multiply by grid cell area (dx²) for area-based flow
			      If False, use raw value for unit-based flow
			N: Number of stochastic iterations
			
		Note:
			Results are stored in self.Q and can be accessed via get_Q()
			
		Author: B.G.
		"""
		# Get temporary accumulation field
		fullQ = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(self.nx*self.ny))
		fullQ.fill(0.)

		# Calculate number of iterations needed (log₂ of grid size)
		logn = math.ceil(math.log2(self.nx*self.ny))+1
		
		# Get temporary fields from pool for rake-compress algorithm
		ndonors  = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(self.nx*self.ny))
		donors   = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(self.nx*self.ny*4))
		src      = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(self.nx*self.ny))
		donors_  = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(self.nx*self.ny*4))
		ndonors_ = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(self.nx*self.ny))
		Q_       = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(self.nx*self.ny))


		for __ in range(N):
			self.compute_stochastic_receivers()

			# Initialize arrays for rake-compress algorithm
			ndonors.fill(0)  # Reset donor counts
			src.fill(0)      # Reset ping-pong state
			
			# Initialize flow values (multiply by area if requested)
			self.Q.fill(value*(cte.DX * cte.DX if area else 1.))
			
			# Build donor-receiver relationships from receiver array
			dpr.rcv2donor(self.receivers, donors, ndonors)

			# Rake-compress iterations for parallel tree accumulation
			# Each iteration doubles the effective path length being compressed
			for i in range(logn+1):
				dpr.rake_compress_accum(donors, ndonors, self.Q, src,
				                   donors_, ndonors_, Q_, i)

			# Final fuse step to consolidate results from ping-pong buffers
			# Merge accumulated values from working arrays back to primary array
			gena.fuse(self.Q, src, Q_, logn)

			ut.add_B_to_weighted_A(fullQ, self.Q, 1./N)

		# Release temporary fields for this iteration
		ndonors.release()
		donors.release()
		src.release()
		donors_.release()
		ndonors_.release()
		Q_.release()

		self.Q.copy_from(fullQ)
		fullQ.release()



	def fill_z(self, epsilon=1e-3):
		"""
		Fill topographic depressions to ensure proper flow routing.
		
		Args:
			epsilon: Small elevation increment for depression filling
			
		Author: B.G.
		"""
		fl.topofill(self, epsilon=epsilon, custom_z = None)

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
		Get elevation data as 2D numpy array.
		
		Returns:
			numpy.ndarray: 2D array of elevation values (ny, nx)
			
		Author: B.G.
		"""
		return self.grid.z.to_numpy().reshape(self.rshp)

	def get_receivers(self):
		"""
		Get receiver data as 2D numpy array.
		
		Returns:
			numpy.ndarray: 2D array of receiver indices (ny, nx)
			
		Author: B.G.
		"""
		return self.receivers.to_numpy().reshape(self.rshp)

	def destroy(self):
		"""
		Release all pooled fields and free GPU memory.
		
		Should be called when finished with the FlowRouter to ensure
		proper cleanup of GPU resources. After calling this method,
		the FlowRouter should not be used.
		
		Author: B.G.
		"""
		# Release core fields back to pool
		if hasattr(self, 'Q') and self.Q is not None:
			self.Q.release()
			self.Q = None
			
		if hasattr(self, 'receivers') and self.receivers is not None:
			self.receivers.release() 
			self.receivers = None

	def __del__(self):
		"""
		Destructor - automatically release fields when object is deleted.
		
		Author: B.G.
		"""
		try:
			self.destroy()
		except:
			pass  # Ignore errors during destruction
