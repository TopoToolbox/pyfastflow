"""
Flood field management and data structures for 2D shallow water flow.

This module provides the Flooder class which manages field allocation and 
orchestrates the 2D shallow water flow computation pipeline. It handles
GPU memory allocation using Taichi's field system and provides high-level
interface for flood modeling workflows.

The Flooder class integrates with FastFlow's flow routing system to provide
initial conditions and boundary handling for hydrodynamic simulations.
Uses Manning's equation for friction and supports precipitation input.

Key Features:
- GPU memory management for flow depth fields
- Integration with FastFlow flow routing
- Configurable hydrodynamic parameters
- Simple interface for flood simulation workflows

Author: B.G.
"""

import taichi as ti
import pyfastflow as pf
from .. import constants as cte
from . import gf_ls as ls


class Flooder:
	"""
	High-level interface for 2D shallow water flow simulation.
	
	The Flooder class provides a complete workflow for flood modeling using
	GPU-accelerated shallow water equations. It integrates with FastFlow's
	flow routing system and manages field allocation for hydrodynamic variables.
	
	The class handles:
	- GPU memory allocation for flow depth fields
	- Parameter configuration for Manning's friction and precipitation
	- Integration with flow routing for initial conditions
	- Execution of the shallow water flow computation pipeline
	
	Args:
		router: FastFlow router object providing grid and flow routing
		precipitation_rates: Precipitation rate in m/s (default: 10e-3/3600)
		manning: Manning's roughness coefficient (default: 1e-3)
		edge_slope: Boundary slope for edge conditions (default: 1e-2)
	
	Attributes:
		nx (int): Number of grid columns
		ny (int): Number of rows  
		dx (float): Grid cell size
		rshp (tuple): Reshape tuple for converting 1D to 2D arrays
		router: Reference to the FastFlow router
		h (ti.field): Flow depth field
		dh (ti.field): Flow depth change field
		fb (ti.FieldsBuilder): Taichi field builder for memory management
		snodetree: Finalized field structure
	
	Author: B.G.
	"""

	def __init__(self, router, precipitation_rates = 10e-3/3600, manning=0.033, edge_slope = 1e-2, dt_hydro = 1e-3, dt_hydro_ls = None):
		"""
		Initialize the Flooder with grid parameters and hydrodynamic settings.
		
		Args:
			router: FastFlow router object providing grid and flow routing
			precipitation_rates: Precipitation rate in m/s (default: 10e-3/3600)
			manning: Manning's roughness coefficient (default: 1e-3)  
			edge_slope: Boundary slope for edge conditions (default: 1e-2)
		"""

		self.router = router
		self.grid = router.grid

		# Saving the original elevation (to cpu)
		self.og_z = self.grid.z.to_numpy()

		# ====== CORE FLOW COMPUTATION FIELDS ======
		# These fields are required for all flow routing operations
		
		self.h = pf.pool.taipool.get_tpfield(dtype = ti.f32, shape = (self.nx*self.ny) )
		self.dh = pf.pool.taipool.get_tpfield(dtype = ti.f32, shape = (self.nx*self.ny) )
		self.nQ = pf.pool.taipool.get_tpfield(dtype = ti.f32, shape = (self.nx*self.ny) )
		self.qx = None
		self.qy = None

		self.h.fill(0.)
		self.dh.fill(0.)
		self.nQ.fill(0.)
		self.nQ_init = False

		self.verbose = False

		cte.PREC = precipitation_rates
		cte.MANNING = manning
		cte.EDGESW = edge_slope
		cte.DT_HYDRO = dt_hydro
		cte.DT_HYDRO_LS = dt_hydro if dt_hydro_ls is None else dt_hydro_ls
		# cte.RAND_RCV = True


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



	def run_graphflood(self, N=10, N_stochastic = 4, N_diffuse = 0, temporal_filtering = 0.):
		"""
		Execute graphflood simulation workflow.
		
		Performs the following steps:
		1. Compute flow receivers using steepest descent
		2. Reroute flow through lakes and depressions
		3. Fill topography with current water depth
		4. Diffuse discharge field for multiple flow paths
		5. Apply shallow water equations with Manning's friction
		
		Args:
			N (int): Number of iterations (currently not used, defaults to 10)
		"""
		z_          = pf.pool.taipool.get_tpfield(dtype = ti.f32, shape = (self.nx*self.ny) )
		Q_          = pf.pool.taipool.get_tpfield(dtype = ti.f32, shape = (self.nx*self.ny) )
		receivers_  = pf.pool.taipool.get_tpfield(dtype = ti.i32, shape = (self.nx*self.ny) )
		receivers__ = pf.pool.taipool.get_tpfield(dtype = ti.i32, shape = (self.nx*self.ny) )

		for _ in range(N):

			if(self.verbose):
				print('Running iteration', _)

			pf.general_algorithms.util_taichi.add_B_to_A(self.grid.z, self.h)

			# Compute steepest descent receivers for flow routing
			self.router.compute_receivers()
			
			# Handle flow routing through lakes and depressions
			self.router.reroute_flow()

			# fills with water
			pf.flow.fill_z_add_delta(self.grid.z,self.h, z_,self.router.receivers, receivers_, receivers__, epsilon=1e-3)
			
			# Accumulate with N stochastic routes if N_stochastic > 0else normal, accumulation
			if(N_stochastic > 0):
				self.router.accumulate_constant_Q_stochastic(cte.PREC, area = True, N=N_stochastic)
			else:
				self.router.accumulate_constant_Q(cte.PREC, area = True)

			# Diffuse as multiple flow N times
			for __ in range(N_diffuse):
				pf.flood.diffuse_Q_constant_prec(self.grid.z, self.router.Q, Q_)

			# z is filled with h, I wanna remove the wxtra z
			pf.general_algorithms.util_taichi.add_B_to_weighted_A(self.grid.z, self.h,-1.)


			if(temporal_filtering == 0.):
				# Apply shallow water equations with Manning's friction
				pf.flood.graphflood_core_cte_mannings(self.h, self.grid.z, self.dh, self.router.receivers, self.router.Q)
			else:
				if(self.nQ_init == False):
					self.nQ.copy_from(self.router.Q)
				else:
					pf.general_algorithms.util_taichi.weighted_mean_B_in_A(self.nQ, self.router.Q, temporal_filtering)

				pf.flood.graphflood_core_cte_mannings(self.h, self.grid.z, self.dh, self.router.receivers, self.nQ)

		z_.release()
		Q_.release()
		receivers_.release()
		receivers__.release()



	def run_LS(self, N=1000, input_mode = 'constant_prec', mode = None):
		"""
		Execute LisFlood shallow water flow simulation using Bates et al. 2010 method.
		
		Runs the LisFlood algorithm for N iterations using separate bed elevation
		and water depth fields. The method alternates between flow routing
		(computing discharge) and depth update (applying continuity equation).
		
		Uses dedicated qx and qy discharge fields for numerical stability.
		
		Args:
			N (int): Number of LisFlood iterations to perform (default: 1000)
		
		Note:
			This method maintains bed elevation (router.z) constant and only
			updates water depth (self.h) to avoid floating-point precision issues.
		
		Author: B.G.
		"""

		# Initialize LisFlood-specific discharge fields on first call
		if self.qx is None :
			# Create x-direction discharge field
			self.qx = pf.pool.taipool.get_tpfield(dtype = ti.f32, shape = (self.nx*self.ny) )
			self.qy = pf.pool.taipool.get_tpfield(dtype = ti.f32, shape = (self.nx*self.ny) )
			
			self.qx.fill(0.)
			self.qy.fill(0.)

		# Main LisFlood iteration loop
		for _ in range(N):
			
			if input_mode == 'constant_prec':
				# Optional: Add precipitation as water depth increase
				ls.init_LS_on_hw_from_constant_effective_prec(self.h, self.grid.z)
			elif input_mode == 'custom_func':
				mode()

			# Step 1: Compute discharge based on water surface gradients
			ls.flow_route(self.h, self.grid.z, self.qx, self.qy)
			
			# Step 2: Update water depths based on discharge divergence
			ls.depth_update(self.h, self.grid.z, self.qx, self.qy)


	def set_h(self, val):
		"""
		Set flow depth field from numpy array.
		
		Args:
			val (numpy.ndarray): Flow depth values to set
		"""
		self.h.from_numpy(val.ravel())
		# self.grid.z.from_numpy(self.og_z.ravel())
		# pf.general_algorithms.util_taichi.add_B_to_A(self.grid.z, self.h)

	def get_h(self):
		"""
		Get current flow depth field as numpy array.
		
		Returns:
			numpy.ndarray: Flow depth values reshaped to 2D grid
		"""
		return self.h.to_numpy().reshape(self.rshp)

	def get_qx(self):
		"""
		Get current x-direction discharge field as numpy array.
		
		Returns:
			numpy.ndarray: X-direction discharge values (m²/s) reshaped to 2D grid
		"""
		return self.qx.to_numpy().reshape(self.rshp)


	def get_qy(self):
		"""
		Get current y-direction discharge field as numpy array.
		
		Returns:
			numpy.ndarray: Y-direction discharge values (m²/s) reshaped to 2D grid
		"""
		return self.qy.to_numpy().reshape(self.rshp)


	def get_dh(self):
		"""
		Get current depth change field as numpy array.
		
		Returns:
			numpy.ndarray: Flow depth change values (m) reshaped to 2D grid
		"""
		return self.dh.to_numpy().reshape(self.rshp)


	def get_Q(self):
		"""
		Get current discharge field from router as numpy array.
		
		Returns:
			numpy.ndarray: Discharge values (m³/s) reshaped to 2D grid
		"""
		return self.router.Q.to_numpy().reshape(self.rshp)

	def destroy(self):
		"""
		Release all pooled fields and free GPU memory.
		
		Should be called when finished with the Flooder to ensure
		proper cleanup of GPU resources. After calling this method,
		the Flooder should not be used.
		
		Author: B.G.
		"""
		# Release core flood fields back to pool
		if hasattr(self, 'h') and self.h is not None:
			self.h.release()
			self.h = None
			
		if hasattr(self, 'dh') and self.dh is not None:
			self.dh.release()
			self.dh = None
			
		if hasattr(self, 'nQ') and self.nQ is not None:
			self.nQ.release()
			self.nQ = None
			
		# Release LisFlood discharge fields if they exist
		if hasattr(self, 'qx') and self.qx is not None:
			self.qx.release()
			self.qx = None
			
		if hasattr(self, 'qy') and self.qy is not None:
			self.qy.release()
			self.qy = None

	def __del__(self):
		"""
		Destructor - automatically release fields when object is deleted.
		
		Author: B.G.
		"""
		try:
			self.destroy()
		except:
			pass  # Ignore errors during destruction
