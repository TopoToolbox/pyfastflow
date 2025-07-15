import taichi as ti
import numpy as np

ti.init(ti.gpu)

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

@ti.kernel
def find_min(values: ti.template(), indices: ti.template(), result: ti.template()):
    result[None] = pack_float_index(values[0], indices[0])
    for i in range(1, values.shape[0]):
        candidate = pack_float_index(values[i], indices[i])
        ti.atomic_min(result[None], candidate)

@ti.kernel
def setup(values: ti.template(), indices: ti.template()):
    for i in range(values.shape[0]):
        values[i] = ti.random() * 200.0 - 100.0
        indices[i] = i
    values[56] = -99.5
    indices[56] = 999

@ti.kernel
def get_result(result: ti.template()) -> (ti.f32, ti.i32):
    return unpack_float_index(result[None])

@ti.kernel 
def debug_pack(val: ti.f32, idx: ti.i32) -> ti.i64:
    return pack_float_index(val, idx)

@ti.kernel
def debug_pack_verbose(val: ti.f32, idx: ti.i32) -> (ti.i64, ti.i64, ti.i64):
    float_bits = ti.bit_cast(val, ti.u32)
    mask = ti.u32(0x80000000)
    result_bits = ti.u32(0)
    if (float_bits & mask) != 0:
        result_bits = ~float_bits
    else:
        result_bits = float_bits ^ mask
    signed_val = ti.cast(result_bits, ti.i32)
    extended_val = ti.cast(signed_val, ti.i64)
    if (float_bits & mask) != 0:
        extended_val = extended_val | (ti.i64(1) << 63)
    final_val = (extended_val << 32) | ti.cast(idx, ti.i64)
    return ti.cast(signed_val, ti.i64), extended_val, final_val

@ti.kernel
def debug_unpack(packed: ti.i64) -> (ti.f32, ti.i32):
    return unpack_float_index(packed)

@ti.kernel  
def debug_unpack_verbose(packed: ti.i64) -> (ti.i64, ti.u32, ti.u32):
    upper_bits = packed >> 32
    was_negative = upper_bits < 0
    
    packed_bits = ti.cast(upper_bits, ti.u32)
    
    float_bits = ti.u32(0)
    if was_negative:
        float_bits = ~packed_bits
    else:
        float_bits = packed_bits ^ ti.u32(0x80000000)
    
    return upper_bits, packed_bits, float_bits

@ti.kernel
def debug_bits(val: ti.f32) -> (ti.u32, ti.u32, ti.i32):
    float_bits = ti.bit_cast(val, ti.u32)
    mask = ti.u32(0x80000000)
    result_bits = ti.u32(0)
    if (float_bits & mask) != 0:
        result_bits = ~float_bits
    else:
        result_bits = float_bits ^ mask
    signed_val = ti.cast(result_bits, ti.i32)
    return float_bits, result_bits, signed_val

# Test
N = 100
values = ti.field(ti.f32, shape=N)
indices = ti.field(ti.i32, shape=N)
result = ti.field(ti.i64, shape=())

setup(values, indices)

# Simple test with just -99.5 to verify packing/unpacking
packed_neg = debug_pack(-99.5, 999)
unpacked_neg = debug_unpack(packed_neg)
print(f"Pack/unpack test: -99.5 -> {packed_neg} -> {unpacked_neg}")

# Detailed debug
upper_bits, packed_bits, float_bits = debug_unpack_verbose(packed_neg)
print(f"Unpack debug: upper={upper_bits}, packed={packed_bits}, float={float_bits}")
print(f"Expected float_bits for -99.5: 3267821568")
print(f"Expected packed_bits: 1027145727")

find_min(values, indices, result)
val, idx = get_result(result)
print(f"Atomic: val={val}, idx={idx}")

# Verification
vals_np = values.to_numpy()
indices_np = indices.to_numpy()
min_idx = np.argmin(vals_np)
print(f"NumPy:  val={vals_np[min_idx]}, idx={indices_np[min_idx]}")

# Manual lexicographic check
min_val = vals_np[0]
min_corr_idx = indices_np[0]
for i in range(1, N):
    if vals_np[i] < min_val or (vals_np[i] == min_val and indices_np[i] < min_corr_idx):
        min_val = vals_np[i]
        min_corr_idx = indices_np[i]
print(f"Manual: val={min_val}, idx={min_corr_idx}")

# Check if they match
if abs(val - min_val) < 1e-6 and idx == min_corr_idx:
    print("SUCCESS: Atomic lexicographic argmin matches expected result!")
else:
    print(f"FAILED: Expected ({min_val}, {min_corr_idx}), got ({val}, {idx})")