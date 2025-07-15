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
			z_prime[i] = z[i]
			continue

		is_border[i] = False
		z_prime[i] = 1e9
		zn = 1e9
		for k in range(4):
			j = neighbour(i,k,nx,ny)

			if(bid[j]!=bid[i]):
				is_border[i] = True
				zn = ti.min(zn,z[j])

		if(is_border[i]):
			z_prime[i] = ti.max(z[i], zn)
		# else:
		# 	z_prime[i] = 1e9		


@ti.kernel
def saddlesort(bid:ti.template(), is_border:ti.template(), z_prime:ti.template(), basin_saddle:ti.template(), 
	basin_saddlenode:ti.template(), active_basin:ti.template(), outlet:ti.template(), z:ti.template(), 
	nx:int, ny:int):
	'''
	Second aprt of Algorithm 3:
	Identifying sadlles for each basin ID
	'''
	
	# Generic invalid value
	invalid = pack_float_index(1e8,42)

	# 
	for i in bid:
		basin_saddle[i] = invalid
		outlet[i] = invalid
		basin_saddlenode[i] = -1

	for i in bid:
		
		# Ignoring no border nodes or basins draining to da edges
		if(is_border[i] == False):
			continue

		# if z_prime[i]>1e8:
		# 	print('!!!!!!')

		# Basin Identifyer
		tbid:ti.i32 = bid[i]

		# Local Prime
		res:ti.i64 = invalid

		for k in range(4):
			j = neighbour(i,k,nx,ny)

			if(bid[j] != tbid ):
				candidate:ti.i64 = pack_float_index(z_prime[i],bid[j])
				res = ti.min(res,candidate)

		if(res == invalid):
			continue

		ti.atomic_min(basin_saddle[bid[i]], res)

	for i in bid:

		if(is_border[i] == False or bid[i] == 0):
			continue

		target_z, target_b = unpack_float_index(basin_saddle[bid[i]])

		

		# print('HERE')
		ishere = False
		for k in range(4):
			j = neighbour(i,k,nx,ny)
			if(bid[j] == target_b and z_prime[i] == target_z):
			# if(bid[j] != bid[i] and z_prime[j] == target_z ):
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

		node = basin_saddlenode[tbid]
		tz = 1e9
		rec = -1
		for k in range(4):
			j = neighbour(node,k,nx,ny)
			if(bid[j] != tbid and tz > z[j]):
				tz = z[j]
				rec = j

		if(rec > -1):
			# Lexicographic atomic min
			candidate:ti.i64 = pack_float_index(tz,rec)
			ti.atomic_min(outlet[tbid], candidate)
		# else:
		# 	print('YOLO')


	# Remove the cycles part of the thingy
	for i in bid:

		bid_d = i

		# if(active_basin[bid_d] == False or bid_d == 0):
		if(bid_d == 0 or outlet[bid_d] == invalid):
			continue

		temp, recout = unpack_float_index(outlet[bid_d])
		# temp1,temp2, recout = unpack_float_index(outlet[bid_d])
		bid_d_prime = bid[recout]
		# print(bid_d, z[basin_saddlenode[bid_d]],'--->',bid_d_prime, temp, "<---THAT")

		if(bid_d_prime == 0):
			continue
		# print('B')
		
		temp, recoutdprime = unpack_float_index(outlet[bid_d_prime])
		bid_d_prime_prime = bid[recoutdprime]

		# temp, bid_saddle_of_d = unpack_float_index(basin_saddle[bid_d])
		bid_saddle_of_d = bid_d

		# if the magical mixture of conditions is met then delete the edge
		if(bid_d_prime_prime == bid_saddle_of_d):
			# print("B")
			if(bid_d_prime < bid_saddle_of_d):
				outlet[bid_d]           = invalid
				basin_saddle[bid_d]     = invalid
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
def init_reroute_carve(tag:ti.template(), tag_:ti.template(), saddlenode:ti.template()):
	'''
	'''
	# invalid = pack_float_index(1e8,42)

	# First pass
	for i in tag:	
		tag[i] = False
		# tag_[i] = False

	for i in tag:	
		if saddlenode[i] == -1:
			continue
		tag[saddlenode[i]] = True
		# tag_[saddlenode[i]] = True
	for i in tag:
		tag_[i] = tag[i]

@ti.kernel
def iteration_reroute_carve(tag:ti.template(), tag_:ti.template(), rec:ti.template(), rec_:ti.template(), change:ti.template()):
	'''
	OPTIMISATION:
	Tehre should be a ping pong scheme
	Algorithm 4 in paper ligne 15 and 16 should be switched (error in the paper)
	so then beware of race condition between recs
	
	'''

	for i in tag:
		if(tag[i]):
			tag_[rec[i]] = True

		rec_[i] = rec[i]

	for i in tag:
		# if(tag_[i]):
		rec[i] = rec_[rec_[i]]

		if(tag[i] != tag_[i]):
			change[None] = True
		tag[i] = tag_[i]

	# for i in tag:
	# 	if(tag[i]):
	# 		tag_[rec[i]] = True
	# 		# rec[i] = rec[rec[rec[rec[rec[i]]]]]
	# 		# rec[i] = rec[rec[rec[i]]]
	# 		temp = i
	# 		for k in range(10):
	# 			temp = rec[temp]
	# 			tag_[temp] = True

	# 		rec[i] = temp
	# 		tag_[temp] = True

	# 		# rec[i] = rec_[rec_[rec_[i]]]

	# 		# temp = rec[i]
	# 		# tag[temp] = True
	# 		# temp = i
	# 		# for k in range(50):
	# 		# 	temp = rec_[temp]
	# 		# 	tag[temp] = True


	# 		# 	tag[rec_[rec_[i]]] = True
	# 		# 	tag[rec_[rec_[rec_[i]]]] = True
	# 		# # rec[i] = rec_[rec_[i]]
	# for i in tag:

	# 	if(tag[i] != tag_[i]):
	# 		change[None] = True

	# 	tag[i]=tag_[i]


@ti.kernel
def finalise_reroute_carve(rec:ti.template(), rec_:ti.template(), tag:ti.template(), saddlenode:ti.template(), outlet:ti.template()):
	'''
	'''
	invalid = pack_float_index(1e8,42)

	for i in rec:
		rec[i] = rec_[i]
	for i in rec:
		if tag[rec_[i]] and tag[i] and i != rec_[i]:
			rec[rec_[i]] = i

	for i in rec:
		if outlet[i] != invalid:
			temp, node = unpack_float_index(outlet[i])
			rec[saddlenode[i]] = node

def reroute_carve(rec, rec_, rec__, tag, tag_, saddlenode, outlet, N, change:ti.template()):
	'''
	'''

	init_reroute_carve(tag, tag_, saddlenode)
	# for _ in range(math.ceil(math.log2(N))+1):
	change[None] = True
	it = 0
	rec.copy_from(rec_)
	rec__.copy_from(rec_)
	while change[None]:
		it += 1
		change[None] = False
		iteration_reroute_carve(tag, tag_, rec, rec_, change)

	# print('converged in', it, 'vs', math.ceil(math.log2(N))+1)
	finalise_reroute_carve(rec, rec__, tag, saddlenode, outlet)






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
	is_border:ti.template(), outlet:ti.template(), basin_saddle:ti.template(), basin_saddlenode:ti.template(), tag:ti.template(), tag_:ti.template(), change:ti.template(),
	nx: int, ny: int, carve = True):
	
	rec_.copy_from(rec)
	N = nx*ny
	Ndep = count_Ndep(rec,edges) # works
	if(Ndep == 0):
		return

	# nump = rec.to_numpy()
	# nump = rec.to_numpy().reshape(ny,nx)[1:-1,1:-1].ravel()
	# arr = np.arange(N).reshape(ny,nx)[1:-1,1:-1].ravel()
	# print(f"DEBUG::{Ndep} depressions found initially, should be {nump[nump==arr].shape[0]}")
	# print(f"DEBUG::{Ndep} depressions found initially")

	# flag_carve = False
	for _ in range(math.ceil(math.log2(Ndep))+1):
	# for _ in range(1):
		Ndep_bis = count_Ndep(rec_, edges)
		# print('Number of depressions :',Ndep_bis, '\n\n')


		####################################
		# Algorithm 2: Basin identification
		####################################
		active_basin.fill(False)
		basin_id_init(bid, edges)
		rec__.copy_from(rec_)
		for _ in range((math.ceil(math.log2(N))+1)):
			propagate_basin(bid, rec__, edges, active_basin)

		if Ndep_bis == 0:
			break

		####################################
		# Algorithm 3: Computing basin graph
		####################################

		border_id_edgez(z, bid, is_border, z_prime, nx, ny)

		saddlesort(bid, is_border, z_prime, basin_saddle, basin_saddlenode, active_basin, outlet, z, nx, ny)

		# print(f'{count_N_valid(basin_saddle)} -- {count_N_valid(outlet)}')
		if(carve):
			
			# flag_carve = not flag_carve

			# if(flag_carve):
				# reroute_carve(rec, rec_, tag, basin_saddlenode, outlet, N)
			# else:
				# reroute_carve(rec_, rec, tag, basin_saddlenode, outlet, N)
			# rec.copy_from(rec_)
			reroute_carve(rec, rec_, rec__, tag, tag_, basin_saddlenode, outlet, N, change)
			# swap_arrays(rec_, rec, N)
			rec_.copy_from(rec)
			
		else:
			reroute_jump(rec_, outlet)
		# print (np.unique(rec_.to_numpy() - nump).shape)





	active_basin.fill(False)
	basin_id_init(bid, edges)
	rec__.copy_from(rec_)
	for _ in range(math.ceil(math.log2(N))):
		propagate_basin(bid, rec__, edges, active_basin)

	# swap_arrays(rec, rec_, N) if (flag_carve == False or carve == False) else 0
	swap_arrays(rec, rec_, N) #if (flag_carve == False or carve == False) else 0
	# rec, rec_ = rec_, rec

	# print (np.unique(rec.to_numpy() - nump))






# End of file
