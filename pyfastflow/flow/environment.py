"""
Environment initialization and management for FastFlow Taichi.

Handles system initialization, boundary condition setup, and resource management
for the FastFlow Taichi flow routing system. Manages the compilation of
boundary-aware neighbor functions and global state.

Author: B.G.
"""

import taichi as ti
from . import constants as cte
from . import neighbourer_flat as nei


def initialise():
	"""
	Initialize the FastFlow Taichi environment.
	
	Sets up boundary conditions, compiles neighbor functions, and initializes
	the global state. Must be called before using any flow routing functions.
	
	Raises:
		RuntimeError: If already initialized
		
	Author: B.G.
	"""
	if(cte.INITIALISED):
		raise RuntimeError("FastFlow Taichi already initialized")

	# Create dummy boundary field for non-custom boundary modes
	if(cte.BOUND_MODE != 3):
		cte.boundaries = ti.field(ti.u1, shape = (10))
	
	# Compile boundary-aware neighbor functions with static conditions
	nei.compile_neighbourer()

	# Mark as initialized
	cte.INITIALISED = True


def reboot():
	"""
	Reset the Taichi environment and reinitialize.
	
	Clears all GPU memory and resets the system state. Use when changing
	fundamental parameters like grid size or boundary conditions.
	
	Author: B.G.
	"""
	ti.reset()
	cte.INITIALISED = False