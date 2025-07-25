"""
Lake flow algorithms for depression handling in flow routing.

Implements algorithms for identifying and routing flow through depressions (lakes)
in digital elevation models. Uses parallel algorithms for basin identification,
saddle point detection, and flow rerouting through lake outlets.

Based on priority-flood and carving algorithms for hydrological flow routing.

Author: B.G.
"""

import numpy as np
import taichi as ti
import math
from .. import constants as cte
from . import neighbourer_flat as nei
from . import util_taichi as ut
from .f32_i32_struct import pack_float_index, unpack_float_index
from .util_taichi import swap_arrays



@ti.kernel
def depression_counter(rec:ti.template())->int:
	"""
	Count the number of depression (pit) nodes in the grid.
	
	Args:
		rec: Receiver array (nodes draining to themselves are pits)
		
	Returns:
		int: Number of depression nodes found
		
	Author: B.G.
	"""
	Ndep = 0
	for i in rec:
		# A pit is a non-edge node that drains to itself and cannot leave domain
		if(rec[i] == i):
			if nei.can_leave_domain(i) == False:
				ti.atomic_add(Ndep, 1)

	return Ndep

@ti.kernel
def basin_id_init(bid:ti.template()):
	"""
	Initialize basin IDs for each cell.
	
	Args:
		bid: Basin ID array to initialize
		
	Note:
		Non-edge cells get unique IDs (i+1), edge cells get ID 0
		
	Author: B.G.
	"""
	for i in bid:
		# Each internal cell gets its own basin ID, edge cells share ID 0
		bid[i] = (i+1) if nei.can_leave_domain(i) == False else 0

@ti.kernel
def propagate_basin(bid:ti.template(), rec_:ti.template()):
	"""
	Propagate basin IDs upstream using pointer jumping technique.
	
	Args:
		bid: Basin ID array to propagate
		rec_: Receiver array for pointer jumping
		
	Note:
		Uses pointer jumping to accelerate basin ID propagation
		
	Author: B.G.
	"""
	for i in bid:
		# Pointer jumping: compress receiver chains and propagate basin IDs
		bid[i] = bid[rec_[i]]  # Propagate basin ID from receiver
		rec_[i] = rec_[rec_[i]]  # Pointer jumping: skip intermediate receivers

def basin_identification(bid:ti.template(), rec:ti.template(), rec_:ti.template(), N:int):
	"""
	Standalone function for basin identification.
	
	Args:
		bid: Basin ID array to compute
		rec: Original receiver array
		rec_: Temporary receiver array for processing
		N: Total number of nodes in grid
		
	Note:
		Performs complete basin identification using pointer jumping
		
	Author: B.G.
	"""
	rec_.copy_from(rec)  # Initialize working copy
	basin_id_init(bid)  # Initialize basin IDs
	# Iterate log2(N) times to ensure full propagation
	for _ in range(math.ceil(math.log2(N))+1):
		propagate_basin(bid, rec_)



@ti.kernel
def saddlesort(bid:ti.template(), is_border:ti.template(), z_prime:ti.template(), basin_saddle:ti.template(), 
	basin_saddlenode:ti.template(), outlet:ti.template(), z:ti.template()):
	"""
	Algorithm 3: Identify saddle points and outlets for each basin.
	
	Args:
		bid: Basin ID array
		is_border: Boolean array marking basin border cells
		z_prime: Modified elevation array for outlet computation
		basin_saddle: Packed saddle information per basin
		basin_saddlenode: Saddle node indices per basin
		outlet: Packed outlet information per basin
		z: Original elevation array
		
	Note:
		Implements 3-pass algorithm to find basin saddles and outlets
		
	Author: B.G.
	"""
	
	# Generic invalid value for packing
	invalid = pack_float_index(1e8,42)

	####################################
	# First pass: Identify basin borders and compute z_prime
	####################################
	for i in z:
		# Domain boundary nodes: use original elevation
		if(nei.can_leave_domain(i)):
			z_prime[i] = z[i]
			continue

		# Initialize for internal nodes
		is_border[i] = False
		z_prime[i] = 1e9  # High value for non-border cells
		zn = 1e9  # Minimum neighbor elevation
		
		# Check all 4 neighbors
		for k in range(4):
			j = nei.neighbour(i,k)

			if(j == -1):  # Invalid neighbor
				continue

			# If neighbor belongs to different basin, this is a border cell
			if(bid[j]!=bid[i]):
				is_border[i] = True
				zn = ti.min(zn,z[j])  # Track minimum neighbor elevation

		# For border cells, z_prime is max of own elevation and min neighbor elevation
		if(is_border[i]):
			z_prime[i] = ti.max(z[i], zn)

	####################################
	# Second pass: Find saddle points for each basin
	####################################
	
	# Initialize basin data structures
	for i in bid:
		basin_saddle[i] = invalid
		outlet[i] = invalid
		basin_saddlenode[i] = -1

	# Find minimum saddle elevation for each basin
	for i in bid:
		# Skip non-border nodes
		if(is_border[i] == False):
			continue

		tbid:ti.i32 = bid[i]  # Current basin ID
		res:ti.i64 = invalid  # Best saddle candidate

		# Check all neighbors for saddle candidates
		for k in range(4):
			j = nei.neighbour(i,k)

			if(j == -1):
				continue

			# If neighbor is in different basin, this is a saddle candidate
			if(bid[j] != tbid):
				candidate:ti.i64 = pack_float_index(z_prime[i],bid[j])
				res = ti.min(res,candidate)  # Find minimum saddle

		if(res == invalid):
			continue

		# Atomically update basin's minimum saddle
		ti.atomic_min(basin_saddle[bid[i]], res)

	####################################
	# Third pass: Identify actual saddle nodes
	####################################
	for i in bid:
		# Skip non-border nodes and edge basins
		if(is_border[i] == False or bid[i] == 0):
			continue

		# Get the minimum saddle for this basin
		target_z, target_b = unpack_float_index(basin_saddle[bid[i]])

		# Check if this node is the actual saddle point
		ishere = False
		for k in range(4):
			j = nei.neighbour(i,k)

			if(j == -1):
				continue
			# This is the saddle if neighbor basin matches target and elevation matches
			if(bid[j] == target_b and z_prime[i] == target_z):
				ishere = True

		if(ishere):
			basin_saddlenode[bid[i]] = i



	####################################
	# Fourth pass: Compute outlet nodes for each basin
	####################################
	for i in bid:
		# Skip edge basins and basins without saddles
		if(i==0 or basin_saddle[i] == invalid):
			continue

		tbid:ti.i32 = i  # Current basin ID
		node = basin_saddlenode[tbid]  # Saddle node for this basin
		tz = 1e9  # Best outlet elevation
		rec = -1  # Best outlet node

		# Find lowest neighbor of saddle node in different basin
		for k in range(4):
			j = nei.neighbour(node,k)

			if(j == -1):
				continue
			# Look for neighbors in different basins with lower elevation
			if(bid[j] != tbid and tz > z[j]):
				tz = z[j]
				rec = j

		if(rec > -1):
			# Store outlet using lexicographic packing (elevation, node)
			candidate:ti.i64 = pack_float_index(tz,rec)
			ti.atomic_min(outlet[tbid], candidate)

	####################################
	# Fifth pass: Remove cycles in basin graph
	####################################
	for i in bid:
		bid_d = i

		# Skip edge basins and basins without outlets
		if(bid_d == 0 or outlet[bid_d] == invalid):
			continue

		# Get outlet information for current basin
		temp, recout = unpack_float_index(outlet[bid_d])
		bid_d_prime = bid[recout]  # Basin that current basin drains to

		if(bid_d_prime == 0):
			continue
		
		# Get outlet information for the target basin
		temp, recoutdprime = unpack_float_index(outlet[bid_d_prime])
		bid_d_prime_prime = bid[recoutdprime]  # Basin that target basin drains to

		bid_saddle_of_d = bid_d

		# Detect and remove cycles: if target's target drains back to current basin
		if(bid_d_prime_prime == bid_saddle_of_d):
			if(bid_d_prime < bid_saddle_of_d):
				# Remove the cycle by invalidating the outlet
				outlet[bid_d]           = invalid
				basin_saddle[bid_d]     = invalid
				basin_saddlenode[bid_d] = -1


@ti.kernel
def reroute_jump(rec:ti.template(), outlet:ti.template(), rerouted:ti.template()):
	"""
	Reroute flow by jumping directly to basin outlets (filling approach).
	
	Args:
		rec: Receiver array to modify
		outlet: Basin outlet information (packed elevation,node)
		
	Note:
		This implements the "filling" approach where flow jumps directly to outlets
		
	Author: B.G.
	"""
	invalid = pack_float_index(1e8,42)

	for i in rerouted:
		rerouted[i] = False

	for i in rec:
		if(outlet[i] == invalid):
			continue
		# Extract outlet node and redirect flow
		temp, rrec = unpack_float_index(outlet[i])
		rec[i-1] = rrec  # Redirect to outlet node
		rerouted[i-1] = True


@ti.kernel
def init_reroute_carve(tag:ti.template(), tag_:ti.template(), saddlenode:ti.template()):
	"""
	Initialize tagging for carving algorithm.
	
	Args:
		tag: Primary tagging array
		tag_: Secondary tagging array
		saddlenode: Saddle node indices per basin
		
	Note:
		Tags all saddle nodes to start carving propagation
		
	Author: B.G.
	"""
	# Initialize all tags to False
	for i in tag:	
		tag[i] = False

	# Tag all saddle nodes as starting points for carving
	for i in tag:	
		if saddlenode[i] == -1:
			continue
		tag[saddlenode[i]] = True
		
	# Copy to secondary array
	for i in tag:
		tag_[i] = tag[i]

@ti.kernel
def iteration_reroute_carve(tag:ti.template(), tag_:ti.template(), rec:ti.template(), rec_:ti.template(), change:ti.template()):
	"""
	Single iteration of carving algorithm with pointer jumping.
	
	Args:
		tag: Primary tagging array
		tag_: Secondary tagging array  
		rec: Primary receiver array
		rec_: Secondary receiver array
		change: Convergence flag
		
	Note:
		Uses pointer jumping to accelerate carving propagation
		Implements corrected Algorithm 4 from paper
		
	Author: B.G.
	"""
	# First pass: propagate tags and copy receivers
	for i in tag:
		if(tag[i]):
			tag_[rec[i]] = True  # Propagate tag to receiver

		rec_[i] = rec[i]  # Copy receiver array

	# Second pass: pointer jumping and convergence check
	for i in tag:
		rec[i] = rec_[rec_[i]]  # Pointer jumping: skip intermediate receivers

		# Check for convergence
		if(tag[i] != tag_[i]):
			change[None] = True
		tag[i] = tag_[i]  # Update tag array


@ti.kernel
def finalise_reroute_carve(rec:ti.template(), rec_:ti.template(), tag:ti.template(), saddlenode:ti.template(), outlet:ti.template(), rerouted:ti.template()):
	"""
	Finalize carving by creating bidirectional links and connecting outlets.
	
	Args:
		rec: Primary receiver array to finalize
		rec_: Secondary receiver array  
		tag: Tagging array
		saddlenode: Saddle node indices per basin
		outlet: Basin outlet information
		
	Note:
		Creates carved channels and connects saddle nodes to outlets
		
	Author: B.G.
	"""
	invalid = pack_float_index(1e8,42)

	# Copy final receiver state
	for i in rec:
		rec[i] = rec_[i]
		
	# Create bidirectional links for carved channels
	for i in rec:
		if tag[rec_[i]] and tag[i] and i != rec_[i]:
			rec[rec_[i]] = i  # Create reverse link
			rerouted[rec_[i]] = True

	# Connect saddle nodes to their outlets
	for i in rec:
		if outlet[i] != invalid:
			temp, node = unpack_float_index(outlet[i])
			rec[saddlenode[i]] = node  # Direct connection from saddle to outlet
			rerouted[saddlenode[i]] = True


def reroute_carve(rec, rec_, rec__, tag, tag_, saddlenode, outlet, change:ti.template(), rerouted:ti.template()):
	"""
	Main carving algorithm that creates channels through saddle points.
	
	Args:
		rec: Primary receiver array
		rec_: Temporary receiver array
		rec__: Secondary temporary receiver array
		tag: Primary tagging array
		tag_: Secondary tagging array
		saddlenode: Saddle node indices per basin
		outlet: Basin outlet information
		change: Convergence flag
		
	Note:
		Implements iterative carving with pointer jumping until convergence
		
	Author: B.G.
	"""
	init_reroute_carve(tag, tag_, saddlenode)  # Initialize tagging
	change[None] = True
	it = 0
	rec.copy_from(rec_)  # Initialize working copies
	rec__.copy_from(rec_)
	
	# Iterate until convergence
	# while change[None]:
	for i in range(math.ceil(math.log2(cte.NX*cte.NY))+1):
		it += 1
		# change[None] = False
		iteration_reroute_carve(tag, tag_, rec, rec_, change)

	# Finalize the carving process
	finalise_reroute_carve(rec, rec__, tag, saddlenode, outlet, rerouted)






@ti.kernel
def count_N_valid(arr:ti.template()) -> int:
	"""
	Count number of valid entries in packed array.
	
	Args:
		arr: Array with packed values to count
		
	Returns:
		int: Number of valid (non-invalid) entries
		
	Author: B.G.
	"""
	invalid = pack_float_index(1e8,42)
	Ninv = 0
	for i in arr:
		if(arr[i] != invalid):
			Ninv += 1
	return Ninv


def reroute_flow(bid:ti.template(), rec:ti.template(), rec_:ti.template(), rec__:ti.template(),
	z:ti.template(), z_prime:ti.template(),	is_border:ti.template(), outlet:ti.template(), basin_saddle:ti.template(), 
	basin_saddlenode:ti.template(), tag:ti.template(), tag_:ti.template(), change:ti.template(), rerouted:ti.template(), carve = True):
	"""
	Main lake flow algorithm implementing depression jumping and carving.
	
	Args:
		bid: Basin ID array
		rec: Primary receiver array
		rec_: Temporary receiver array
		rec__: Secondary temporary receiver array
		z: Elevation array
		z_prime: Modified elevation array for lake outlets
		is_border: Boolean array marking basin border cells
		outlet: Packed outlet information per basin
		basin_saddle: Packed saddle information per basin
		basin_saddlenode: Saddle node indices per basin
		tag: Tagging array for carving algorithm
		tag_: Temporary tagging array
		change: Flag for iteration convergence
		carve: Whether to use carving (True) or jumping (False)
		
	Author: B.G.
	"""
	
	rec_.copy_from(rec)  # Initialize working copy
	N = cte.NX * cte.NY  # Total number of nodes
	Ndep = depression_counter(rec)  # Count initial depressions
	rerouted.fill(False)

	# print(f'Ndep:{Ndep}')
	if(Ndep == 0):
		return  # No depressions to process

	# Main iterative loop - process depressions until convergence
	for _ in range(math.ceil(math.log2(Ndep))+1):
		Ndep_bis = depression_counter(rec_)  # Count remaining depressions
		# print(f'HERE::{Ndep_bis}')

		####################################
		# Algorithm 2: Basin identification
		####################################
		basin_id_init(bid)  # Initialize basin IDs
		rec__.copy_from(rec_)
		# Propagate basin IDs upstream using pointer jumping
		for _ in range((math.ceil(math.log2(N))+1)):
			propagate_basin(bid, rec__)
		if Ndep_bis == 0:
			# print(f'HERE::{Ndep_bis}')
			break  # All depressions resolved

		####################################
		# Algorithm 3: Computing basin graph
		####################################
		# Find saddle points and outlets for each basin
		saddlesort(bid, is_border, z_prime, basin_saddle, basin_saddlenode, outlet, z)

		# Apply flow rerouting strategy
		if(carve):
			# Carving: create channels through saddle points
			reroute_carve(rec, rec_, rec__, tag, tag_, basin_saddlenode, outlet, change, rerouted)
			rec_.copy_from(rec)
		else:
			# Filling: jump flow directly to outlets
			reroute_jump(rec_, outlet, rerouted)
		# print (np.unique(rec_.to_numpy() - nump).shape)





	# Final basin identification and cleanup
	basin_id_init(bid)
	return
	rec__.copy_from(rec_)
	for _ in range(math.ceil(math.log2(N))):
		propagate_basin(bid, rec__)

	# Swap arrays to return result in rec
	swap_arrays(rec, rec_)






# End of file
