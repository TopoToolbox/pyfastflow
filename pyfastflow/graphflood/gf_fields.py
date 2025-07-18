"""
GraphFlood field management and data structures for 2D shallow water flow.

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

	def __init__(self, router, precipitation_rates = 10e-3/3600, manning=0.033, edge_slope = 1e-2):
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


		# Initialize Taichi field builders for memory management
		self.fb = ti.FieldsBuilder()  # Builder for flow computation fields
		self.snodetree_flow = None  # Will hold finalized flow field structure

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



		

		cte.PREC = precipitation_rates
		cte.MANNING = manning
		cte.EDGESW = edge_slope


	def run_graphflood(self, N=10):
		"""
		Execute graphfloodflood simulation workflow.
		
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
			# Compute steepest descent receivers for flow routing
			self.router.compute_receivers()
			
			# Handle flow routing through lakes and depressions
			self.router.reroute_flow()
			
			# Fill topography accounting for water depth
			pf.flow.fill_z_add_delta(self.router.z,self.h,self.router.z_,self.router.receivers,self.router.receivers_,self.router.receivers__, epsilon=1e-3)

			# Diffuse discharge field to simulate multiple flow paths
			for i in range(5):
				pf.graphflood.diffuse_Q_constant_prec(self.router.z, self.router.Q, self.router.Q_)

			# Apply shallow water equations with Manning's friction
			pf.graphflood.graphflood_core_cte_mannings(self.h,self.router.z,self.dh, self.router.receivers, self.router.Q)

	def set_h(self, val):
		"""
		Set flow depth field from numpy array.
		
		Args:
			val (numpy.ndarray): Flow depth values to set
		"""
		self.h.from_numpy(val.ravel())

	def get_h(self):
		"""
		Get current flow depth field as numpy array.
		
		Returns:
			numpy.ndarray: Flow depth values reshaped to 2D grid
		"""
		return self.h.to_numpy().reshape(self.rshp)
