"""
Topographic depression filling algorithms for flow routing.

This module implements various depression filling algorithms used in hydrological 
modeling to create depressionless surfaces for flow routing computation.

Author: B.G.
"""

import taichi as ti
import math
from .. import constants as cte
import pyfastflow.general_algorithms.util_taichi as ut
import pyfastflow as pf


@ti.kernel
def _topofill(z:ti.template(), z_:ti.template(), rec_:ti.template(), rec__:ti.template(), epsilon:ti.template(), iteration : ti.template()):
	"""
	Single iteration of topographic filling algorithm.
	
	Performs one iteration of the topographic filling process, updating elevations 
	to ensure flow connectivity while preserving original topography where possible.
	
	Args:
		z: Original elevation field
		z_: Working elevation field (modified in place)
		rec_: Current receiver field (modified in place)
		rec__: Next receiver field (working buffer)
		epsilon: Small increment value for elevation adjustment
		iteration: Current iteration number
	"""
	# Update elevations and receiver connections
	for i in z:
		rec__[i] = rec_[rec_[i]]
		
		# Skip if node is its own receiver or already properly connected
		if(i == rec_[i] or (z_[i]>z[rec_[i]] and rec_[rec_[i]] == rec_[i])):
			continue

		# Apply elevation adjustment with exponential increment
		z_[i] = ti.max(z_[i], z_[rec_[i]] + (ti.math.pow(2,iteration-1)) * epsilon)

	# Update receiver field for next iteration
	for i in rec_:
		rec_[i] = rec__[i]


def topofill(flow_field, epsilon=1e-3, custom_z = None):
	"""
	Fill depressions using topographic filling algorithm.
	
	Implements an iterative topographic filling algorithm that preserves original 
	topography while ensuring flow connectivity. Uses exponential increments to 
	create a depressionless surface.
	
	Args:
		flow_field: FlowRouter object containing elevation and receiver fields
		epsilon: Small increment value for elevation adjustment (default: 1e-6)
		custom_z: Optional custom elevation field for output (default: None)
	"""
	# Initialize receiver working arrays
	receivers_  = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(flow_field.nx*flow_field.ny))
	receivers__ = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(flow_field.nx*flow_field.ny))
	z_ = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(flow_field.nx*flow_field.ny))
	z__ = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(flow_field.nx*flow_field.ny))
	receivers_.copy_from(flow_field.receivers)
	receivers__.copy_from(flow_field.receivers)
	

	# Process with internal elevation fields or custom output
	if(custom_z is None):
		# Use internal elevation buffers
		z_.copy_from(flow_field.grid.z)
		z__.copy_from(flow_field.grid.z)
		for it in range(math.ceil(math.log2(cte.NX*cte.NY))):
			_topofill(flow_field.grid.z, z_, receivers_, receivers__, epsilon, it+1)
		flow_field.grid.z.copy_from(z_)
	else:
		# Use custom elevation field for output
		custom_z.copy_from(flow_field.grid.z)
		for it in range(math.ceil(math.log2(cte.NX*cte.NY))):
			_topofill(flow_field.grid.z, custom_z, receivers_, receivers__, epsilon, it+1)

	# Release temporary fields back to pool
	receivers_.release()
	receivers__.release()
	z_.release()
	z__.release()


@ti.kernel
def _apply_fill_z_add_delta(z:ti.template(), h:ti.template(), z_:ti.template()):
	"""
	Apply filling adjustments and track delta changes.
	
	Updates the elevation field with filled values and adds the elevation 
	difference to a separate height field for tracking adjustments.
	
	Args:
		z: Original elevation field (modified in place)
		h: Height adjustment field (modified in place)
		z_: Filled elevation field
	"""
	for i in z:
		dh = z_[i] - z[i]  # Calculate elevation difference
		h[i]+= dh          # Add difference to height field
		z[i]=z_[i]         # Update elevation with filled value

def fill_z_add_delta(zh,h,z_,receivers,receivers_,receivers__, epsilon=1e-3):
	"""
	Fill elevations and adds height adjustments to ext. field.

	Used in flood analysis to fill the topo with water or sed and not bedrock.
	
	Performs topographic filling on the elevation field and adds the elevation 
	differences to a separate height field for tracking the adjustments made.
	
	Args:
		zh: Combined elevation field (modified in place)
		h: Height adjustment field (modified in place)
		z_: Working elevation field
		receivers: Receiver field
		receivers_: Working receiver field
		receivers__: Second working receiver field
		epsilon: Small increment value for elevation adjustment (default: 1e-4)
	"""
	# Initialize working arrays
	z_.copy_from(zh)
	receivers_.copy_from(receivers)
	receivers__.copy_from(receivers)
	
	# Perform iterative filling
	for it in range(math.ceil(math.log2(cte.NX*cte.NY))):
		_topofill(zh, z_, receivers_, receivers__, epsilon, it)

	# Apply changes and track deltas
	_apply_fill_z_add_delta(zh,h,z_)


