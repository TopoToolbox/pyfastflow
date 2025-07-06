import numpy as np
import taichi as ti
import math


# BACKUP: Previous approach that worked for pack/unpack but not lexicographic ordering
# @ti.func
# def pack_float_index_v1(val: ti.f32, idx: ti.i32) -> ti.i64:
#     # Transform float for lexicographic ordering
#     float_bits = ti.bit_cast(val, ti.u32)
#     
#     # Standard IEEE 754 lexicographic transformation
#     if float_bits >= ti.u32(0x80000000):  # negative
#         # Flip all bits for negative numbers
#         float_bits = ~float_bits
#     else:  # positive
#         # Flip sign bit for positive numbers
#         float_bits = float_bits | ti.u32(0x80000000)
#     
#     return (ti.cast(float_bits, ti.i64) << 32) | ti.cast(idx, ti.i64)
#
# @ti.func
# def unpack_float_index_v1(packed: ti.i64):
#     # Extract components
#     float_bits = ti.cast(packed >> 32, ti.u32)
#     idx = ti.cast(packed & ti.i64(0xFFFFFFFF), ti.i32)
#     
#     # Reverse the transformation
#     if float_bits >= ti.u32(0x80000000):  # was positive
#         # Remove sign bit
#         float_bits = float_bits & ti.u32(0x7FFFFFFF)
#     else:  # was negative
#         # Flip all bits back
#         float_bits = ~float_bits
#     
#     return ti.bit_cast(float_bits, ti.f32), idx


@ti.kernel
def swap_arrays(array1: ti.template(), array2: ti.template(), N: int):
    """
    Swap contents of two arrays of the same type and size
    After this operation: array1 contains original array2, array2 contains original array1
    """
    for i in range(ti.i32(N)):
        temp = array1[i]
        array1[i] = array2[i]
        array2[i] = temp

@ti.func
def flip_float_bits(f: ti.f32) -> ti.u32:
    u = ti.bit_cast(f, ti.u32)
    # REVERSED IEEE 754 transformation to get ascending order
    return ti.select(u & ti.u32(0x80000000) != 0, u ^ ti.u32(0x80000000), ~u)

@ti.func 
def unflip_float_bits(u: ti.u32) -> ti.f32:
    # Reverse the REVERSED transformation
    restored = ti.select(u & ti.u32(0x80000000) != 0, ~u, u ^ ti.u32(0x80000000))
    return ti.bit_cast(restored, ti.f32)

@ti.func
def pack_float_index(f: ti.f32, i: ti.i32) -> ti.i64:
    f_enc = flip_float_bits(f)
    i_enc = ti.bit_cast(i, ti.u32)
    
    # Pack: float in upper 32, index in lower 32
    packed = (ti.cast(f_enc, ti.i64) << 32) | ti.cast(i_enc, ti.i64)
    
    # Flip only the upper 32 bits (float part) to reverse float ordering
    # Keep lower 32 bits (index part) unchanged for correct index ordering
    flipped_upper = (~packed) & (ti.i64(0xFFFFFFFF) << 32)
    unchanged_lower = packed & ti.i64(0xFFFFFFFF)
    
    return flipped_upper | unchanged_lower

@ti.func
def unpack_float_index(packed: ti.i64) -> tuple:
    # Reverse the selective flipping
    # Flip back only the upper 32 bits, keep lower 32 unchanged
    flipped_upper = (~packed) & (ti.i64(0xFFFFFFFF) << 32)
    unchanged_lower = packed & ti.i64(0xFFFFFFFF)
    unflipped = flipped_upper | unchanged_lower
    
    # Simple unpack with corrected bit transformation
    f_enc = ti.cast(unflipped >> 32, ti.u32)
    i_enc = ti.cast(unflipped & ti.i64(0xFFFFFFFF), ti.u32)
    
    f = unflip_float_bits(f_enc)
    i = ti.bit_cast(i_enc, ti.i32)
    
    return f, i

@ti.kernel
def unpack_full_float_index(arr:ti.template(), ft:ti.template(), it:ti.template()):

	for i in arr:
		tft, tit = unpack_float_index(arr[i])
		ft[i]    = tft
		it[i]    = tit

@ti.kernel
def pack_full_float_index(arr:ti.template(), ft:ti.template(), it:ti.template()):

	for i in arr:
		arr[i] = pack_float_index(ft[i],it[i])


@ti.func
def on_edge(i:int, nx:int, ny:int):
	'''
	'''
	col = i // nx
	ret = False
	if(i < nx or i >= nx*ny - nx or col == 0 or col == nx-1):
		ret = True
	return ret


@ti.func
def neighbour(i:int, k:int, nx:int, ny:int):
	return i-nx if k==0 else (i-1 if k==1 else (i+1 if k==2 else i+nx))


@ti.kernel
def count_Ndep(rec:ti.template(), edges:ti.template())->int:
	'''
	 Atmically sums the number of pit nodes
	'''
	Ndep = 0
	for i in rec:
		if(edges[i] == False and rec[i] == i):
			Ndep += 1 
	return Ndep

@ti.kernel
def basin_id_init(bid:ti.template(), edges:ti.template()):
	'''
	assign to each cell its own value except the outflow edges that are all one big basin	
	'''
	for i in bid:
		bid[i] = (i+1) if(edges[i] == False) else 0

@ti.kernel
def propagate_basin(bid:ti.template(), rec_:ti.template(), edges:ti.template(), active_basin:ti.template()):
	'''
	Propagate the basin ID upstream
	'''

	for i in bid:
		bid[i] = bid[rec_[i]]
		rec_[i] = rec_[rec_[i]]
		# active_basin[i+1] = (bid[i]==i+1)

def basin_identification(bid:ti.template(), rec:ti.template(), rec_:ti.template(), edges:ti.template(), active_basin:ti.template(), N:int):
	'''
	'''
	rec_.copy_from(rec)
	active_basin.fill(False)
	basin_id_init(bid, edges)
	for _ in range(math.ceil(math.log2(N))+1):
		propagate_basin(bid, rec_, edges, active_basin)


@ti.kernel
def border_id_edgez(z:ti.template(), bid:ti.template(), is_border:ti.template(), z_prime:ti.template(), nx:int, ny:int):
	'''
	first part of Algorithm 3:
	identifying cells and calulating z'
	'''
	for i in z:
		if(on_edge(i,nx,ny)):
			continue

		is_border[i] = False
		z_prime[i] = z[i]
		zn = 1e9
		for k in range(4):
			j = neighbour(i,k,nx,ny)

			if(bid[j]!=bid[i]):
				is_border[i] = True
				if(z[j]<zn):
					zn = z[j]

		if(is_border[i]):
			z_prime[i] = ti.max(z_prime[i], zn)
		else:
			z_prime[i] = 1e9		


@ti.kernel
def saddlesort(bid:ti.template(), is_border:ti.template(), z_prime:ti.template(), basin_saddle:ti.template(), 
	basin_saddlenode:ti.template(), active_basin:ti.template(), outlet:ti.template(), z:ti.template(), 
	nx:int, ny:int):
	'''
	Second aprt of Algorithm 3:
	Identifying sadlles for each basin ID
	'''
	
	invalid = pack_float_index(1e8,42)

	for i in bid:
		basin_saddle[i] = invalid
		outlet[i] = invalid
		basin_saddlenode[i] = -1

	for i in bid:
		
		# Ignoring no border nodes or basins draining to da edges
		if(is_border[i] == False or bid[i] == 0):
			continue

		# Basin Identifyer
		tbid:ti.i32 = nx*ny

		# Local Prime
		tx = z_prime[i]

		for k in range(4):
			j = neighbour(i,k,nx,ny)
			if(bid[j] != tbid):
				tbid = ti.min(bid[j],tbid)


		# Lexicographic atomic min
		candidate:ti.i64 = pack_float_index(tx,tbid)
		# debug_f, debug_i = unpack_float_index(candidate)
		# print(f'{tx} {tbid} -> {debug_f} {debug_i}')
		ti.atomic_min(basin_saddle[bid[i]], candidate)

	for i in bid:

		if(is_border[i] == False or bid[i] == 0):
			continue

		target_z, target_b = unpack_float_index(basin_saddle[bid[i]])

		if(z_prime[i] != target_z):
			continue
		# print('HERE')
		ishere=False
		for k in range(4):
			j = neighbour(i,k,nx,ny)
			if(bid[j] == target_b):
				ishere = True
		if(ishere):
			basin_saddlenode[bid[i]] = i
			# print('here')

	for i in bid:

		# Ignoring no border nodes or basins draining to da edges
		if(i==0 or basin_saddle[i] == invalid):
			# print('BAGUL')
			continue

		# Basin Identifyer
		tbid:ti.i32 = i
		# target_z, temp = unpack_float_index(basin_saddle[tbid])


		# if(z_prime[temp] != target_z):
		# 	continue

		node = basin_saddlenode[i]
		tz = 1e9
		rec = -1
		for k in range(4):
			j = neighbour(node,k,nx,ny)
			if(bid[j] != tbid and tz > z[j]):
				tz = z[j]
				rec = j

		if(rec > -1):
			print(rec,'here')
			# Lexicographic atomic min
			candidate:ti.i64 = pack_float_index(tz,rec)
			ti.atomic_min(outlet[tbid], candidate)


	# Remove the cycles part of the thingy
	for i in bid:

		bid_d = i
		# print('A')
		# if(active_basin[bid_d] == False or bid_d == 0):
		if(bid_d == 0 or outlet[bid_d] == invalid):
			continue
		# print('B')

		temp, recout = unpack_float_index(outlet[bid_d])
		bid_d_prime = bid[recout]
		print(bid_d,bid_d_prime, "<---THAT")
		if(bid_d_prime == 0):
			continue
		temp, recoutdprime = unpack_float_index(outlet[bid_d_prime])
		bid_d_prime_prime = bid[recoutdprime]

		temp, bid_saddle_of_d = unpack_float_index(basin_saddle[bid_d])

		# if the magical mixture of conditions is met then delete the edge
		if(bid_d_prime_prime == bid_saddle_of_d):
			if(bid_d_prime < bid_saddle_of_d):
				outlet[bid_d]       = invalid
				basin_saddle[bid_d] = invalid
				basin_saddlenode[bid_d] = -1


@ti.kernel
def reroute_jump(rec:ti.template(), outlet:ti.template()):

	invalid = pack_float_index(1e8,42)


	for i in rec:
		if(outlet[i] == invalid):
			continue
		# print('YOLO')
		temp, rrec = unpack_float_index(outlet[i])
		rec[i-1] = rrec


@ti.kernel
def count_N_valid(arr:ti.template()) -> int:
	invalid = pack_float_index(1e8,42)
	Ninv = 0
	for i in arr:
		if(arr[i] != invalid):
			Ninv += 1
	return Ninv


def _reroute_flow(bid:ti.template(), rec:ti.template(), rec_:ti.template(), rec__:ti.template(),
	edges:ti.template(), active_basin:ti.template(), z:ti.template(), z_prime:ti.template(),
	is_border:ti.template(), outlet:ti.template(), basin_saddle:ti.template(), basin_saddlenode:ti.template(),
	nx: int, ny: int):
	
	rec_.copy_from(rec)
	N = nx*ny
	Ndep = count_Ndep(rec,edges) # works

	nump = rec.to_numpy()
	# nump = rec.to_numpy().reshape(ny,nx)[1:-1,1:-1].ravel()
	# arr = np.arange(N).reshape(ny,nx)[1:-1,1:-1].ravel()
	# print(f"DEBUG::{Ndep} depressions found initially, should be {nump[nump==arr].shape[0]}")
	print(f"DEBUG::{Ndep} depressions found initially")

	for _ in range(math.ceil(math.log2(Ndep))+1):
	# for _ in range(1):
		Ndep = count_Ndep(rec_, edges)
		print(Ndep)

		####################################
		# Algorithm 2: Basin identification
		####################################
		active_basin.fill(False)
		basin_id_init(bid, edges)
		rec__.copy_from(rec_)
		for _ in range(math.ceil(math.log2(N))+1):
			propagate_basin(bid, rec__, edges, active_basin)


		####################################
		# Algorithm 3: Computing basin graph
		####################################

		border_id_edgez(z, bid, is_border, z_prime, nx, ny)

		saddlesort(bid, is_border, z_prime, basin_saddle, basin_saddlenode, active_basin, outlet, z, nx, ny)

		# print(f'{count_N_valid(basin_saddle)} -- {count_N_valid(outlet)}')
		
		reroute_jump(rec_, outlet)
		# print (np.unique(rec_.to_numpy() - nump).shape)





	active_basin.fill(False)
	basin_id_init(bid, edges)
	rec__.copy_from(rec_)
	for _ in range(math.ceil(math.log2(N))+1):
		propagate_basin(bid, rec__, edges, active_basin)

	swap_arrays(rec, rec_, N)
	# rec, rec_ = rec_, rec

	# print (np.unique(rec.to_numpy() - nump))






# End of file
