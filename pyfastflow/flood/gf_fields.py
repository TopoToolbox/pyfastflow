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


		# Store grid parameters
		self.nx = router.nx  # Number of columns
		self.ny = router.ny  # Number of rows
		self.dx = router.dx  # Grid spacing
		self.rshp = router.rshp  # Reshape tuple for converting 1D arrays to 2D

		self.router = router

		# Saving the original elevation (to cpu)
		self.og_z = self.router.z.to_numpy()


		# Initialize Taichi field builders for memory management
		self.fb = ti.FieldsBuilder()  # Builder for flow computation fields
		self.snodetree = None  # Will hold finalized flow field structure

		# ====== CORE FLOW COMPUTATION FIELDS ======
		# These fields are required for all flow routing operations
		
		# flow depth
		self.h = ti.field(ti.f32)
		self.fb.dense(ti.i,(self.nx*self.ny)).place(self.h)
		self.dh = ti.field(ti.f32)
		self.fb.dense(ti.i,(self.nx*self.ny)).place(self.dh)

		# Finalize the flow field structure (allocates GPU memory)
		self.snodetree = self.fb.finalize()

		self.h.fill(0.)
		self.dh.fill(0.)

		self.fb_LS = None
		self.snodetree_LS = None
		self.qx = None
		self.qy = None


		cte.PREC = precipitation_rates
		cte.MANNING = manning
		cte.EDGESW = edge_slope
		cte.DT_HYDRO = dt_hydro
		cte.DT_HYDRO_LS = dt_hydro if dt_hydro_ls is None else dt_hydro_ls
		# cte.RAND_RCV = True


	def run_graphflood(self, N=10):
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

		for _ in range(N):

			pf.flow.ut.add_B_to_A(self.router.z, self.h)

			# Compute steepest descent receivers for flow routing
			self.router.compute_receivers()
			
			# Handle flow routing through lakes and depressions
			self.router.reroute_flow()

			pf.flow.fill_z_add_delta(self.router.z,self.h,self.router.z_,self.router.receivers,self.router.receivers_,self.router.receivers__, epsilon=1e-3)
			
			self.router.accumulate_constant_Q_stochastic(cte.PREC, area = True, N=4)

			pf.flood.diffuse_Q_constant_prec(self.router.z, self.router.Q, self.router.Q_)

			pf.flow.ut.add_B_to_weighted_A(self.router.z, self.h,-1.)

# 
			# Test 1
			# Fill topography accounting for water depth
			# pf.flow.fill_z_add_delta(self.router.z,self.h,self.router.z_,self.router.receivers,self.router.receivers_,self.router.receivers__, epsilon=1e-3)

			# self.router.accumulate_constant_Q(cte.PREC, area = True)

			# Diffuse discharge field to simulate multiple flow paths
			# for i in range(5):
			# 	pf.flood.diffuse_Q_constant_prec(self.router.z, self.router.Q, self.router.Q_)



			# Apply shallow water equations with Manning's friction
			pf.flood.graphflood_core_cte_mannings(self.h,self.router.z,self.dh, self.router.receivers, self.router.Q)


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
			print('INIT')  # Debug output for field initialization
			self.fb_LS = ti.FieldsBuilder()
			self.snodetree_LS = None
			
			# Create x-direction discharge field
			self.qx = ti.field(ti.f32)
			self.fb_LS.dense(ti.i,(self.nx*self.ny)).place(self.qx)
			
			# Create y-direction discharge field  
			self.qy = ti.field(ti.f32)
			self.fb_LS.dense(ti.i,(self.nx*self.ny)).place(self.qy)
			
			# Finalize field structure and initialize to zero
			self.snodetree_LS = self.fb_LS.finalize()
			self.qx.fill(0.)
			self.qy.fill(0.)

		# Main LisFlood iteration loop
		for _ in range(N):
			
			if input_mode == 'constant_prec':
				# Optional: Add precipitation as water depth increase
				ls.init_LS_on_hw_from_constant_effective_prec(self.h, self.router.z)
			elif input_mode == 'custom_func':
				mode()

			# Step 1: Compute discharge based on water surface gradients
			ls.flow_route(self.h, self.router.z, self.qx, self.qy)
			
			# Step 2: Update water depths based on discharge divergence
			ls.depth_update(self.h, self.router.z, self.qx, self.qy)


	def set_h(self, val):
		"""
		Set flow depth field from numpy array.
		
		Args:
			val (numpy.ndarray): Flow depth values to set
		"""
		self.h.from_numpy(val.ravel())
		# self.router.z.from_numpy(self.og_z.ravel())
		# pf.flow.ut.add_B_to_A(self.router.z, self.h)

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
