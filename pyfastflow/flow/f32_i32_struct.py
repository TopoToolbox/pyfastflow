"""
Float-Int packing for lexicographic atomic operations.

Provides bit manipulation functions to pack f32 and i32 into a single i64 structure.
Enables atomic comparison with float priority and int tiebreaker for lexicographic argmin.
Uses IEEE 754 bit manipulation to ensure correct ordering in packed form.

Author: B.G.
"""

import taichi as ti


@ti.func
def flip_float_bits(f: ti.f32) -> ti.u32:
    """
    Transform float bits for lexicographic ordering in packed structure.

    Args:
        f: Float value to transform

    Returns:
        uint32: Bit-transformed representation for correct ordering

    Author: B.G.
    """
    u = ti.bit_cast(f, ti.u32)  # Cast float to uint32 to access bits
    # IEEE 754 transformation: flip sign bit for negatives, invert all bits for positives
    # This ensures negative floats order correctly before positive ones
    return ti.select(u & ti.u32(0x80000000) != 0, u ^ ti.u32(0x80000000), ~u)


@ti.func
def unflip_float_bits(u: ti.u32) -> ti.f32:
    """
    Reverse the float bit transformation to restore original value.

    Args:
        u: Bit-transformed uint32 representation

    Returns:
        float: Original float value

    Author: B.G.
    """
    # Reverse the IEEE 754 transformation applied in flip_float_bits
    # For negatives (MSB=1): invert all bits, for positives: flip sign bit
    restored = ti.select(u & ti.u32(0x80000000) != 0, ~u, u ^ ti.u32(0x80000000))
    return ti.bit_cast(restored, ti.f32)  # Cast back to float


@ti.func
def pack_float_index(f: ti.f32, i: ti.i32) -> ti.i64:
    """
    Pack float and index into single 64-bit value for lexicographic comparison.

    Args:
        f: Float value (primary comparison key)
        i: Index value (secondary comparison key)

    Returns:
        int64: Packed value suitable for atomic lexicographic argmin

    Author: B.G.
    """
    f_enc = flip_float_bits(f)  # Transform float for correct ordering
    i_enc = ti.bit_cast(i, ti.u32)  # Cast index to unsigned for bit operations

    # Pack: float in upper 32 bits, index in lower 32 bits
    packed = (ti.cast(f_enc, ti.i64) << 32) | ti.cast(i_enc, ti.i64)

    # Flip only the upper 32 bits (float part) to reverse float ordering
    # This ensures smaller floats produce smaller packed values
    flipped_upper = (~packed) & (ti.i64(0xFFFFFFFF) << 32)
    unchanged_lower = packed & ti.i64(0xFFFFFFFF)  # Keep index bits unchanged

    return flipped_upper | unchanged_lower


@ti.func
def unpack_float_index(packed: ti.i64) -> tuple:
    """
    Unpack 64-bit value back to original float and index values.

    Args:
        packed: Packed 64-bit value from pack_float_index

    Returns:
        tuple: (float, index) - original values

    Author: B.G.
    """
    # Reverse the selective flipping applied during packing
    # Flip back only the upper 32 bits (float part), keep lower 32 bits unchanged
    flipped_upper = (~packed) & (ti.i64(0xFFFFFFFF) << 32)
    unchanged_lower = packed & ti.i64(0xFFFFFFFF)
    unflipped = flipped_upper | unchanged_lower

    # Extract float and index from their respective bit ranges
    f_enc = ti.cast(unflipped >> 32, ti.u32)  # Upper 32 bits -> float
    i_enc = ti.cast(unflipped & ti.i64(0xFFFFFFFF), ti.u32)  # Lower 32 bits -> index

    f = unflip_float_bits(f_enc)  # Restore original float value
    i = ti.bit_cast(i_enc, ti.i32)  # Restore original index value

    return f, i


@ti.kernel
def unpack_full_float_index(arr: ti.template(), ft: ti.template(), it: ti.template()):
    """
    Unpack entire array of packed float-index values into separate arrays.

    Args:
            arr: Array of packed 64-bit values
            ft: Output array for float values
            it: Output array for index values

    Author: B.G.
    """
    for i in arr:
        tft, tit = unpack_float_index(arr[i])  # Unpack each element
        ft[i] = tft  # Store float value
        it[i] = tit  # Store index value


@ti.kernel
def pack_full_float_index(arr: ti.template(), ft: ti.template(), it: ti.template()):
    """
    Pack entire arrays of float and index values into single packed array.

    Args:
            arr: Output array for packed 64-bit values
            ft: Input array of float values
            it: Input array of index values

    Author: B.G.
    """
    for i in arr:
        arr[i] = pack_float_index(
            ft[i], it[i]
        )  # Pack float and index into single value
