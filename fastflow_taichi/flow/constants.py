"""
Compile-time constants for Taichi kernels.

Defines grid dimensions, boundary conditions, and other compile-time parameters
that affect kernel compilation. Changes require recompilation of affected kernels.

This centralized approach ensures consistent constants across all kernels
and can improve performance by up to 30% compared to runtime parameters.

Author: B.G.
"""

import taichi as ti
import numpy as np


#########################################
###### GRID CONSTANTS ###################
#########################################

# Grid spacing (uniform cell size)
DX = 1.

# Number of columns in the grid
NX = 512

# Number of rows in the grid
NY = 512

# Enable stochastic receiver selection (randomized flow routing)
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
	global boundaries,_snodetree_boundaries, BOUND_MODE
	# Clean up existing boundary field if it exists
	if _snodetree_boundaries is not None:
		try:
			_snodetree_boundaries.destroy()
		except:
			pass
		_snodetree_boundaries = None
	
	# Create new boundary field
	fb1 = ti.FieldsBuilder()
	boundaries = ti.field(dtype=ti.u8)
	fb1.dense(ti.i, NX*NY).place(boundaries)
	_snodetree_boundaries = fb1.finalize()  # Finalize field structure
	boundaries.from_numpy(tboundaries)  # Copy boundary data
	BOUND_MODE = 3  # Switch to custom boundary mode