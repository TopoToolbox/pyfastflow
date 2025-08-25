'''
Sets fo function to manage tectonic uplift with the landscape evolution model

B.G.
'''


import taichi as ti
import numpy as np
import pyfastflow as pf
import math


@ti.kernel
def block_uplift(z:ti.template(), rate:ti.f32):
	"""
	Apply uniform block uplift to the topography.
	
	Adds vertical motion to all interior nodes (excluding boundary nodes
	that can drain out of the domain). This simulates tectonic uplift
	or other large-scale vertical motions.
	
	Args:
		z (ti.template): Topographic elevation field to be uplifted
		rate (ti.f32): Uplift rate in m/year (positive values = uplift)
	
	Note:
		Boundary nodes that can leave the domain are excluded from uplift
		to maintain consistent boundary conditions.
	
	Author: B.G.
	"""
	for i in z:
		# Only apply uplift to interior nodes (not boundary outlets)
		if(pf.flow.neighbourer_flat.can_leave_domain(i) == False):
			z[i] += rate * pf.constants.DT_SPL  # Apply uplift over time step


@ti.kernel
def ext_uplift_nobl(z:ti.template(), rate:ti.template()):
	"""
	Apply spatially-varying uplift without boundary checking.
	
	Applies variable uplift rates across the domain without excluding 
	boundary nodes. This allows uplift to be applied uniformly including
	at domain edges where flow can exit.
	
	Args:
		z (ti.template): Topographic elevation field to be uplifted
		rate (ti.template): Spatially-varying uplift rate field (m/year)
	
	Note:
		Unlike block_uplift(), this function applies uplift to ALL nodes
		including boundary outlets, which may be desired for certain
		tectonic scenarios or boundary condition setups.
	
	Author: B.G.
	"""
	for i in z:
		z[i] += rate[i] * pf.constants.DT_SPL  # Apply uplift over time step


@ti.kernel
def ext_uplift_bl(z:ti.template(), rate:ti.template()):
	"""
	Apply spatially-varying uplift with boundary checking.
	
	Applies variable uplift rates across the domain while excluding
	boundary nodes that can drain out of the domain. This maintains
	consistent boundary conditions for flow routing.
	
	Args:
		z (ti.template): Topographic elevation field to be uplifted  
		rate (ti.template): Spatially-varying uplift rate field (m/year)
	
	Note:
		Boundary nodes that can leave the domain are excluded from uplift
		to preserve boundary condition stability. Use ext_uplift_nobl()
		if uplift is needed at all nodes including boundaries.
	
	Author: B.G.
	"""
	for i in z:
		# Only apply uplift to interior nodes (not boundary outlets)
		if(pf.flow.neighbourer_flat.can_leave_domain(i) == False):
			z[i] += rate[i] * pf.constants.DT_SPL  # Apply uplift over time step

