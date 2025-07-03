import numpy as np
import taichi as ti
import math


@ti.func
def pack_float_index(val: ti.f32, idx: ti.i32) -> ti.i64:
    float_bits = ti.bit_cast(val, ti.u32)
    
    # Standard IEEE 754 to lexicographic ordering transformation
    # This ensures negative values are smaller than positive values
    mask = ti.u32(0x80000000)
    result_bits = ti.u32(0)
    if (float_bits & mask) != 0:
        # Negative: flip all bits to reverse order
        result_bits = ~float_bits
    else:
        # Positive: flip sign bit to put above all negative values
        result_bits = float_bits ^ mask
    
    # Cast to signed i32 for correct atomic_min behavior
    signed_val = ti.cast(result_bits, ti.i32)
    # For negative floats, we need to ensure they map to negative packed values
    extended_val = ti.cast(signed_val, ti.i64)
    final_val = (extended_val << 32) | ti.cast(idx, ti.i64)
    
    # For negative floats, ensure the final result is negative
    if (float_bits & mask) != 0:
        # This was a negative float, force the packed value to be negative
        # by setting the sign bit of the final i64
        final_val = final_val | (ti.i64(1) << 63)
    
    return final_val

@ti.func
def unpack_float_index(packed: ti.i64):
    # Check if this was originally a negative float (i64 sign bit set)
    was_negative = (packed >> 32) < 0
    
    # Extract the upper 32 bits and get the lower 32 bits for reconstruction
    upper_bits = packed >> 32
    idx = ti.cast(packed & ti.u64(0xFFFFFFFF), ti.i32)
    
    # Reverse the transformation
    float_bits = ti.u32(0)
    
    # If it was negative, we need to extract the actual transformed bits
    if was_negative:
        # The upper 32 bits contain our transformed value, but when cast to u32,
        # the sign bit from our i64 operation affects the result
        # We need to mask out just the sign bit in the 32-bit portion
        packed_bits = ti.cast(upper_bits, ti.u32) & ti.u32(0x7FFFFFFF)
        # Reverse the bit flip: original = ~transformed
        float_bits = ~packed_bits
    else:
        # For positive values, just cast and restore
        packed_bits = ti.cast(upper_bits, ti.u32)
        # Reverse the XOR with sign bit
        float_bits = packed_bits ^ ti.u32(0x80000000)
    
    return ti.bit_cast(float_bits, ti.f32), idx



@ti.func
def on_edge(i:int, nx:int, ny:int):
	'''
	'''
	col = i // nx
	if(i < nx or i >= nx*ny - nx or col == 0 or col == nx-1):
		return True
	return False


@ti.func
def neighbour(i:int, k:int, nx:int, ny:int):
	return i-nx if k==0 else (i-1 if k==1 else (i+1 if k==2 else i+nx))


@ti.kernel
def basin_id_init(bid:ti.template(), edges:ti.template()):
	'''
	assign to each cell its own value except the outflow edges that are all one big basin	
	'''
	for i in bid:
		bid[i] = i if(edges[i] == False) else -1

@ti.kernel
def propagate_basin(bid:ti.template(), rec_:ti.template(), edges:ti.template(), active_basin:ti.template()):
	'''
	Propagate the basin ID upstream
	'''

	for i in bid:
		bid[i] = bid[rec_[i]]
		rec_[i] = rec_[rec_[i]]
		active_basin[i] = bid==[i]

def basin_identification(bid:ti.template(), rec:ti.template(), rec_:ti.template(), edges:ti.template(), N:int):
	'''
	'''
	rec_.copy_from(rec)
	basin_id_init(bid, edges)
	for _ in range(math.ceil(math.log2(N))+1):
		propagate_basin(bid, rec_, edges)


@ti.kernel
def border_id_edgez(z:ti.template(), bid:ti.template(), is_border:ti.template(), border_z:ti.template(), nx:int, ny:int):
	for i in z:
		if(on_edge(i,nx,ny)):
			continue

		is_border[i] = False
		border_z[i] = z[i]
		zn = 1e10
		
		for k in range(4):
			j = neighbour(i,k,nx,ny)
			if(on_edge(j,nx,ny)):
				continue
			if(bid[j]!=bid[i]):
				is_border[i] = True
				zn = ti.min(zn,z[j])
		if(is_border[i]):
			border_z[i] = ti.min(border_z[i], zn)			


@ti.kernel
def saddlesort(bid:ti.template(), is_border:ti.template(), border_z:ti.template(), basin_saddle:ti.template()):

	for i in bid:
		basin_saddle[i] = ti.i64(1e9)

	for i in bid:
		
		if(is_border[i] == False):
			continue

		tbid:ti.i32 = bid[i]
		tx = border_z[i]
		candidate:ti.i64 = pack_float_index(tx,tbid)
		ti.atomic_min(basin_saddle[bid], candidate)


	for i in bid:
		
		if(is_border[i] == False):
			continue

		tbid:ti.i32 = bid[i]
		tx = border_z[i]
		candidate:ti.i64 = pack_float_index(tx,tbid)
		ti.atomic_min(basin_saddle[bid], candidate)






# End of file
