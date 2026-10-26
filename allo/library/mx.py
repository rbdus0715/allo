# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=used-before-assignment, unsubscriptable-object, unsupported-assignment-operation, chained-comparison, no-value-for-parameter

import re

import allo
from ..ir.utils import MockBuffer
from ..ir.types import Int, UInt, int32, uint8, float32, ConstExpr


def _mx_quantize_elem_int_f32[
    Ty
](v_f32: float32, shared_field: uint8) -> "UInt(Ty.elem_bits)":
    bits: int32 = v_f32.bitcast()
    sign: UInt(1) = bits[31:32]
    exp_field: Int(9) = bits[23:31]
    full_mant: Int(25) = (1 << 23) | bits[0:23]
    total_shift: Int(10) = (23 - Ty.max_unbiased_exp) + shared_field - exp_field

    mag: Int(Ty.elem_bits) = 0
    if total_shift < 24:
        mag = full_mant >> total_shift

    rounded: Int(Ty.elem_bits) = mag
    if sign == 1:
        rounded = -mag
    if rounded > Ty.max_int:
        rounded = Ty.max_int
    if rounded < -Ty.max_int:
        rounded = -Ty.max_int
    return rounded


def _mx_quantize_elem_fp_f32[
    Ty
](v_f32: float32, shared_field: uint8) -> "UInt(Ty.elem_bits)":
    # OCP MX: v / 2^(shared exponent - emax), round-to-nearest-even, saturating
    bits: int32 = v_f32.bitcast()
    sign: int32 = int(bits[31:32])
    exp_field: int32 = int(bits[23:31])
    full_mant: int32 = (1 << 23) | int(bits[0:23])
    code: int32 = exp_field - int(shared_field) + Ty.max_unbiased_exp + Ty.bias
    shift: int32 = 23 - Ty.mantissa_bits
    base: int32 = 0
    if code >= 1:  # normal: q in [2^m, 2^(m+1)] sits on exponent code - 1
        base = (code - 1) << Ty.mantissa_bits
    else:  # subnormal: q in [0, 2^m]; q == 2^m carries into exponent code 1
        shift = shift + 1 - code

    mag: int32 = 0
    if exp_field != 0 and shift <= 25:  # larger shifts round to zero
        q: int32 = full_mant >> shift
        rem: int32 = full_mant & ((1 << shift) - 1)
        half: int32 = 1 << (shift - 1)
        round_up: int32 = 0
        if rem > half:
            round_up = 1
        if rem == half:
            round_up = q & 1
        mag = base + q + round_up
    if mag > Ty.max_code:
        mag = Ty.max_code
    result: UInt(Ty.elem_bits) = (sign << (Ty.elem_bits - 1)) | mag
    return result


def schedule_mx_quantize_block_f32(s):
    s.unroll("mx_quantize_block_f32:i1")


def mx_quantize_block_f32[Ty, K](x: "float32[K]") -> "Ty":

    exp_fields: uint8[K]
    with allo.meta_for(K) as i0:
        bits_i: int32 = x[i0].bitcast()
        exp_fields[i0] = bits_i[23:31]

    with allo.meta_for(K.bit_length() - 1) as stage:
        stride: ConstExpr[int32] = K >> (stage + 1)
        with allo.meta_for(stride) as lane:
            if int(exp_fields[lane + stride]) > int(exp_fields[lane]):
                exp_fields[lane] = exp_fields[lane + stride]

    max_exp_field: int32 = exp_fields[0]

    scale_wide: int32 = max_exp_field - Ty.max_unbiased_exp
    scale_wide = min(scale_wide, 254)
    scale_wide = max(scale_wide, 0)
    scale_field: uint8 = scale_wide
    max_exp_field_u8: uint8 = max_exp_field

    word: Ty = 0
    word[Ty.bits - 8 : Ty.bits] = scale_field

    for i1 in range(K):
        elem: UInt(Ty.elem_bits) = 0
        with allo.meta_if(Ty.is_float):
            elem = _mx_quantize_elem_fp_f32[Ty](x[i1], max_exp_field_u8)
        with allo.meta_else():
            elem = _mx_quantize_elem_int_f32[Ty](x[i1], max_exp_field_u8)
        word[i1 * Ty.elem_bits : (i1 + 1) * Ty.elem_bits] = elem

    return word


def mx_pack_words(arr, bits):
    """Packs an array's raw bytes into a 1D array of bits-wide words."""
    import numpy as np  # pylint: disable=import-outside-toplevel
    from ..utils import get_np_struct_type  # pylint: disable=import-outside-toplevel

    raw = np.ascontiguousarray(arr).view(np.uint8).reshape(-1)
    assert (
        raw.size % (bits // 8) == 0
    ), f"{raw.size} bytes is not a whole number of {bits}-bit words"
    return raw.view(get_np_struct_type(bits))


def mx_pack_f32_blocks(x):
    """Splits a float32 [R, C] array (C % 32 == 0) into the two 512b-word
    arrays nn.mx_unpack_f32 reads: A0 holds floats 0-15 of every 32-float
    block, A1 floats 16-31, so one block arrives per cycle over two ports."""
    import numpy as np  # pylint: disable=import-outside-toplevel

    x = np.ascontiguousarray(x, dtype=np.float32)
    R, C = x.shape
    assert C % 32 == 0, f"C={C} must be a multiple of 32"
    halves = x.reshape(R, C // 32, 2, 16)
    return tuple(
        mx_pack_words(np.ascontiguousarray(halves[:, :, h]), 512).reshape(R, C // 32)
        for h in range(2)
    )


def mx_dequantize_np(Ty, elems, scales):
    """Host inverse of mx_quantize, in float64: (elements, e8m0 scales along the
    last axis) -> values. MXFP elements are bit patterns, MXINT the integers."""
    import numpy as np  # pylint: disable=import-outside-toplevel

    e = np.asarray(elems, dtype=np.int64)
    if Ty.is_float:
        m, eb = Ty.mantissa_bits, Ty.elem_bits
        codes = e & ((1 << eb) - 1)
        exp = (codes >> m) & ((1 << Ty.exp_bits) - 1)
        mant = codes & ((1 << m) - 1)
        mag = np.where(
            exp == 0,
            mant * 2.0 ** (1 - Ty.bias - m),
            (1 + mant / 2.0**m) * 2.0 ** (exp - Ty.bias),
        )
        vals = np.where(codes >> (eb - 1), -mag, mag)
    else:
        vals = e.astype(np.float64)
    scale = np.repeat(np.asarray(scales, dtype=np.int64), Ty.block_size, axis=-1)
    return vals * 2.0 ** (scale - 127)


def _mx_word_dtype(Ty):
    # numpy dtype of one Ty element as the LLVM backend lays it out
    from ..utils import (  # pylint: disable=import-outside-toplevel
        get_llvm_wide_int_struct_type,
        get_np_struct_type,
    )

    if Ty.bits > 64:
        return get_llvm_wide_int_struct_type(Ty.bits)
    return get_np_struct_type(max(8, 1 << (Ty.bits - 1).bit_length()))


def mx_pack_np(Ty, x, axis=-1):
    """Host MX quantization into an array of Ty words (scale | elements), the
    operand of nn.mx_linear2d / nn.mx_conv2d: blocks run along `axis`, which
    is zero-padded to whole blocks and becomes the block axis (length NB)."""
    import numpy as np  # pylint: disable=import-outside-toplevel

    BS, EB = Ty.block_size, Ty.elem_bits
    x = np.moveaxis(np.asarray(x, dtype=np.float32), axis, -1)
    lead, K = x.shape[:-1], x.shape[-1]
    NB = -(-K // BS)
    rows = np.zeros((int(np.prod(lead)), NB * BS), np.float32)
    rows[:, :K] = x.reshape(-1, K)
    elems, scales = _mx_quantize_np(Ty, rows)
    codes = elems.astype(np.int64).reshape(-1, NB, BS) & ((1 << EB) - 1)
    bits = np.zeros((rows.shape[0], NB, Ty.bits), np.uint8)
    bits[..., : BS * EB] = ((codes[..., None] >> np.arange(EB)) & 1).reshape(
        *codes.shape[:2], -1
    )
    bits[..., BS * EB :] = (scales.astype(np.int64)[..., None] >> np.arange(8)) & 1
    raw = np.packbits(bits, axis=-1, bitorder="little")
    dtype = _mx_word_dtype(Ty)
    words = np.zeros(raw.shape[:-1] + (dtype.itemsize,), np.uint8)
    words[..., : raw.shape[-1]] = raw
    words = words.view(dtype)[..., 0].reshape(*lead, NB)
    return np.ascontiguousarray(np.moveaxis(words, -1, axis))


def mx_pack_channels_np(Ty, x):
    """Host counterpart of nn.mx_quantize_channels: NCHW (or a conv kernel
    [Cout, Cin, Kh, Kw]) -> Ty[N, NB * H * W], blocks along the channels."""
    return mx_pack_np(Ty, x, axis=1).reshape(len(x), -1)


def mx_unpack_np(Ty, q, axis=-1, length=None):
    """Inverse of mx_pack_np, in float64: Ty words -> values along `axis`
    (`length` of them, default all NB * block_size)."""
    import numpy as np  # pylint: disable=import-outside-toplevel

    BS, EB = Ty.block_size, Ty.elem_bits
    q = np.moveaxis(np.asarray(q), axis, -1)
    raw = np.ascontiguousarray(q).view(np.uint8).reshape(*q.shape, -1)
    bits = np.unpackbits(raw, axis=-1, bitorder="little")[..., : Ty.bits]
    bits = bits.astype(np.int64)
    codes = (bits[..., : BS * EB].reshape(*q.shape, BS, EB) << np.arange(EB)).sum(-1)
    scales = (bits[..., BS * EB :] << np.arange(8)).sum(-1)
    elems = codes - ((codes >> (EB - 1)) << EB)  # sign-extend
    vals = mx_dequantize_np(Ty, elems.reshape(*q.shape[:-1], -1), scales)
    vals = vals[..., : length if length is not None else vals.shape[-1]]
    return np.moveaxis(vals, -1, axis)


def _mx_quantize_np(Ty, x):
    """numpy MX quantization of a [R, K] array, bit-exact with the hardware quantizer."""
    import numpy as np  # pylint: disable=import-outside-toplevel

    x = np.ascontiguousarray(x, dtype=np.float32)
    R, K = x.shape
    BS = Ty.block_size
    assert K % BS == 0, f"K={K} must be a multiple of block_size={BS}"

    bits = x.view(np.uint32).astype(np.int64).reshape(R, K // BS, BS)
    exp_field = (bits >> 23) & 0xFF
    max_exp = exp_field.max(axis=2, keepdims=True)
    full_mant = (bits & 0x7FFFFF) | (1 << 23)
    if Ty.is_float:  # mirrors _mx_quantize_elem_fp_f32
        m, eb = Ty.mantissa_bits, Ty.elem_bits
        code = exp_field - max_exp + Ty.max_unbiased_exp + Ty.bias
        shift = 23 - m + np.maximum(0, 1 - code)
        s = np.minimum(shift, 40)
        q = full_mant >> s
        rem, half = full_mant & ((1 << s) - 1), 1 << (s - 1)
        q = q + ((rem > half) | ((rem == half) & ((q & 1) == 1)))
        mag = (np.maximum(code - 1, 0) << m) + q
        mag = np.where((exp_field != 0) & (shift <= 25), mag, 0)
        mag = np.minimum(mag, Ty.max_code)
        codes = ((bits >> 31) << (eb - 1)) | mag
        elems = codes - ((codes >> (eb - 1)) << eb)  # same bits, as a signed int
    else:  # mirrors _mx_quantize_elem_int_f32
        total_shift = (23 - Ty.max_unbiased_exp) + max_exp - exp_field
        mag = np.where(total_shift < 24, full_mant >> np.minimum(total_shift, 63), 0)
        elems = np.clip(np.where(bits >> 31, -mag, mag), -Ty.max_int, Ty.max_int)
    scales = np.clip(max_exp[..., 0] - Ty.max_unbiased_exp, 0, 254)
    elem_dtype = np.dtype(f"int{max(8, 1 << (Ty.elem_bits - 1).bit_length())}")
    return elems.astype(elem_dtype), scales.astype(np.uint8)


def mx_quantize(Ty, x, axis=-1):
    """Quantizes a 2D array into an MX operand: (elements, e8m0 scales).

    MXINT elements are the integers themselves; MXFP elements are their bit
    patterns, stored as same-width signed integers (e.g. Int(8) for mxfp8).
    """
    import numpy as np  # pylint: disable=import-outside-toplevel

    x = np.asarray(x, dtype=np.float32)
    assert x.ndim == 2, "mx_quantize takes a 2D array"
    if axis in (0, -2):
        elems, scales = mx_quantize(Ty, x.T)
        return np.ascontiguousarray(elems.T), np.ascontiguousarray(scales.T)
    elems, scales = _mx_quantize_np(Ty, x)
    return elems.reshape(x.shape), scales


def _mx_get_scale[Ty](word: "Ty") -> uint8:
    return word[Ty.bits - 8 : Ty.bits]


def _mx_acc_bits(Ty, K):
    # FP: fixed-point product width (see _mx_fp_mul); INT: plain product width
    product_bits = Ty.block_accum_bits if Ty.is_float else 2 * Ty.elem_bits
    return product_bits + (K - 1).bit_length()


def _mx_fp_mul[
    Ty, K
](a_i: "UInt(Ty.elem_bits)", b_i: "UInt(Ty.elem_bits)") -> "Int(_mx_acc_bits(Ty, K))":
    """Exact product of two MXFP elements, in units of 2^Ty.dot_exp_offset."""
    a: int32 = int(a_i)
    b: int32 = int(b_i)
    exp_mask: int32 = (1 << Ty.exp_bits) - 1
    mant_mask: int32 = (1 << Ty.mantissa_bits) - 1
    a_exp_field: int32 = (a >> Ty.mantissa_bits) & exp_mask
    b_exp_field: int32 = (b >> Ty.mantissa_bits) & exp_mask

    # normal: implicit leading 1, exponent field - bias
    # subnormal (field 0): no leading 1, exponent pinned to 1 - bias
    a_exp: int32 = 1 - Ty.bias
    a_mant: int32 = a & mant_mask
    if a_exp_field != 0:
        a_exp = a_exp_field - Ty.bias
        a_mant = (1 << Ty.mantissa_bits) | (a & mant_mask)
    b_exp: int32 = 1 - Ty.bias
    b_mant: int32 = b & mant_mask
    if b_exp_field != 0:
        b_exp = b_exp_field - Ty.bias
        b_mant = (1 << Ty.mantissa_bits) | (b & mant_mask)

    # >= 0: both exponents bottom out at 1 - bias (the accumulator's radix point)
    shift_amt: int32 = (a_exp + b_exp) - 2 * (1 - Ty.bias)
    magnitude: "Int(_mx_acc_bits(Ty, K))" = a_mant * b_mant
    magnitude = magnitude << shift_amt
    term: "Int(_mx_acc_bits(Ty, K))" = magnitude
    if ((a ^ b) >> (Ty.elem_bits - 1)) & 1 == 1:
        term = -magnitude
    return term


def mx_block_dot[Ty, K](data_a: "Ty", data_b: "Ty") -> float32:
    scale_a: int32 = int(_mx_get_scale[Ty](data_a))
    scale_b: int32 = int(_mx_get_scale[Ty](data_b))

    mul_i: "Int(_mx_acc_bits(Ty, K))[K]"
    for j0 in range(K):
        with allo.meta_if(Ty.is_float):
            a_f: UInt(Ty.elem_bits) = data_a[
                j0 * Ty.elem_bits : (j0 + 1) * Ty.elem_bits
            ]
            b_f: UInt(Ty.elem_bits) = data_b[
                j0 * Ty.elem_bits : (j0 + 1) * Ty.elem_bits
            ]
            mul_i[j0] = _mx_fp_mul[Ty, K](a_f, b_f)
        with allo.meta_else():
            a_i: Int(Ty.elem_bits) = data_a[j0 * Ty.elem_bits : (j0 + 1) * Ty.elem_bits]
            b_i: Int(Ty.elem_bits) = data_b[j0 * Ty.elem_bits : (j0 + 1) * Ty.elem_bits]
            mul_i[j0] = a_i * b_i

    acc_i: "Int(_mx_acc_bits(Ty, K))" = 0
    for j1 in range(K):
        acc_i = acc_i + mul_i[j1]
    total: float32 = float(acc_i)

    total_bits: int32 = total.bitcast()
    sign: UInt(1) = total_bits[31:32]
    exp_t: int32 = int(total_bits[23:31])
    mant_t: Int(24) = total_bits[0:23]

    # -254: the two E8M0 biases; dot_exp_offset: the products' radix point (FP)
    new_exp: int32 = exp_t + scale_a + scale_b - 254 + Ty.dot_exp_offset

    result_bits: int32 = 0
    if exp_t == 0 or scale_a == 0 or scale_b == 0:
        result_bits = 0
    elif new_exp <= 0:
        result_bits = int(sign) << 31
    elif new_exp >= 255:
        result_bits = (int(sign) << 31) | (255 << 23)
    else:
        result_bits = (int(sign) << 31) | (new_exp << 23) | mant_t
    return result_bits.bitcast()


def schedule_mx_block_dot(s):
    s.unroll("mx_block_dot:j0")
    s.unroll("mx_block_dot:j1")


# ---- MX matrices (Ty[R, NB]: an R x K matrix in MX blocks along K) and the
# stages MX layers are built from ----


def mx_quantize_rows[
    Ty, TyX, R, K
](X: "TyX[R, K]") -> "Ty[R, (K + Ty.block_size - 1) // Ty.block_size]":
    # float32 [R, K] -> MX matrix: one packed scale | elems word per block along
    # K (zero-padded to whole blocks); X is read once, row by row
    BS: ConstExpr[int32] = Ty.block_size
    NB: ConstExpr[int32] = (K + BS - 1) // BS
    Q: Ty[R, NB]
    for r, b in allo.grid(R, NB):
        blk: float32[BS]
        for e in range(BS):
            fq: float32 = 0.0
            if b * BS + e < K:
                fq = X[r, b * BS + e]
            blk[e] = fq
        Q[r, b] = mx_quantize_block_f32[Ty, BS](blk)
    return Q


def mx_quantize_cols[
    Ty, TyX, R, K
](X: "TyX[K, R]") -> "Ty[R, (K + Ty.block_size - 1) // Ty.block_size]":
    # float32 [K, R] -> MX matrix of X^T: blocks run down the columns, e.g.
    # V in attention's P @ V (mx_linear2d reduces along the block axis)
    BS: ConstExpr[int32] = Ty.block_size
    NB: ConstExpr[int32] = (K + BS - 1) // BS
    Q: Ty[R, NB]
    for r, b in allo.grid(R, NB):
        blk: float32[BS]
        for e in range(BS):
            fq: float32 = 0.0
            if b * BS + e < K:
                fq = X[b * BS + e, r]
            blk[e] = fq
        Q[r, b] = mx_quantize_block_f32[Ty, BS](blk)
    return Q


def _mx_dequantize_elem[Ty](e: "UInt(Ty.elem_bits)", scale: int32) -> float32:
    # value = mag * 2^sh, built by adjusting the exponent field of float(mag)
    v: int32 = int(e)
    neg: int32 = (v >> (Ty.elem_bits - 1)) & 1
    mag: int32 = 0
    sh: int32 = scale - 127
    with allo.meta_if(Ty.is_float):
        exp_field: int32 = (v >> Ty.mantissa_bits) & ((1 << Ty.exp_bits) - 1)
        mag = v & ((1 << Ty.mantissa_bits) - 1)
        sh = sh + 1 - Ty.bias - Ty.mantissa_bits  # subnormal: 0.m * 2^(1 - bias)
        if exp_field != 0:  # normal: 1.m * 2^(exp_field - bias)
            mag = mag | (1 << Ty.mantissa_bits)
            sh = sh + exp_field - 1
    with allo.meta_else():
        mag = v
        if neg == 1:  # two's complement
            mag = (1 << Ty.elem_bits) - v
    f: float32 = float(mag)
    bits: int32 = f.bitcast()
    new_exp: int32 = int(bits[23:31]) + sh
    result_bits: int32 = 0
    if mag != 0 and new_exp >= 255:
        result_bits = (neg << 31) | (255 << 23)
    elif mag != 0 and new_exp > 0:
        result_bits = (neg << 31) | (new_exp << 23) | (bits & 0x7FFFFF)
    return result_bits.bitcast()


def mx_dequantize[Ty, R, K](Q: "Ty[R, K // Ty.block_size]") -> "float32[R, K]":
    # MX matrix -> float32 [R, K] (debugging, mixing with other precisions)
    BS: ConstExpr[int32] = Ty.block_size
    EB: ConstExpr[int32] = Ty.elem_bits
    X: float32[R, K]
    for r, b in allo.grid(R, K // BS):
        w: Ty = Q[r, b]
        scale: int32 = int(w[Ty.bits - 8 : Ty.bits])
        for e in range(BS):
            el: UInt(Ty.elem_bits) = w[e * EB : (e + 1) * EB]
            X[r, b * BS + e] = _mx_dequantize_elem[Ty](el, scale)
    return X


def schedule_mx(s):
    """Unrolls the block-level loops inside the MX primitives a kernel uses."""
    for fn, loops in (
        ("mx_block_dot", ("j0", "j1")),
        ("mx_quantize_block_f32", ("i1",)),
    ):
        if s._find_function(fn, error=False) is not None:
            for loop in loops:
                s.unroll(f"{fn}:{loop}")
    return s


def partition_block_lanes(s, buf, length, BS, dim=2):
    """Partitions `buf` along `dim` (of `length` elements) into one bank per
    block lane, so the BS elements of a block are read or written in one cycle."""
    if length <= BS:
        s.partition(buf, dim=dim)
    else:
        s.partition(buf, partition_type=2, dim=dim, factor=BS)


def mx_unpack_f32[
    R, C
](A0: "UInt(512)[R, C // 32]", A1: "UInt(512)[R, C // 32]", X: "float32[R, C]"):
    # DRAM -> chip at one 32-float block per cycle: each block arrives as two
    # 512b words, one per port (A0: floats 0-15, A1: 16-31). C % 32 == 0;
    # mx.mx_pack_f32_blocks builds A0/A1 on the host. X is an output
    # argument, so the consumer's banking of it reaches this function too.
    for r, i in allo.grid(R, C // 32):
        lo: UInt(512) = A0[r, i]
        hi: UInt(512) = A1[r, i]
        with allo.meta_for(16) as e:
            lb: int32 = lo[e * 32 : (e + 1) * 32]
            hb: int32 = hi[e * 32 : (e + 1) * 32]
            X[r, i * 32 + e] = lb.bitcast()
            X[r, i * 32 + 16 + e] = hb.bitcast()


def schedule_mx_unpack_f32(s):
    # X's block-lane banks come from the consuming layer's schedule (its
    # partition of the same buffer propagates here); a second one would clash
    s.pipeline(f"{s.top_func_name}:i")
    return s


def schedule_mx_quantize_rows(s):
    """One block per cycle: X is banked by block lane."""
    Ty, _, _, K = s.inst_list[:4]
    schedule_mx(s)
    top = s.top_func_name
    partition_block_lanes(s, MockBuffer(top, "X"), K, Ty.block_size)
    s.pipeline(f"{top}:b")
    return s


def mx_quantize_channels[
    Ty, TyX, B, C, H, W
](X: "TyX[B, C, H, W]") -> "Ty[B, (C + Ty.block_size - 1) // Ty.block_size * H * W]":
    # float32 NCHW -> MX blocks along the channels (zero-padded to whole
    # blocks), one row per image: block (cb, h, w) at column (cb * H + h) * W + w
    BS: ConstExpr[int32] = Ty.block_size
    NB: ConstExpr[int32] = (C + BS - 1) // BS
    Q: Ty[B, NB * H * W]
    for n, cb, h, w in allo.grid(B, NB, H, W):
        blk: float32[BS]
        for e in range(BS):
            fx: float32 = 0.0
            if cb * BS + e < C:
                fx = X[n, cb * BS + e, h, w]
            blk[e] = fx
        Q[n, (cb * H + h) * W + w] = mx_quantize_block_f32[Ty, BS](blk)
    return Q


def schedule_mx_quantize_channels(s):
    """One block per cycle: X is banked by block lane along the channels."""
    Ty, _, _, C = s.inst_list[:4]
    top = s.top_func_name
    schedule_mx(s)
    partition_block_lanes(s, MockBuffer(top, "X"), C, Ty.block_size)
    s.pipeline(f"{top}:w")
    return s


MX_ACC_LATENCY = 6


def mx_matmul[
    Ty, M, N, K
](
    Aq: "Ty[M, (K + Ty.block_size - 1) // Ty.block_size]",
    Bq: "Ty[N, (K + Ty.block_size - 1) // Ty.block_size]",
) -> "float32[M, N]":
    BS: ConstExpr[int32] = Ty.block_size
    NB: ConstExpr[int32] = (K + BS - 1) // BS
    P: ConstExpr[int32] = 4 - 3 * (N >= MX_ACC_LATENCY)
    C: float32[M, N]
    acc: float32[N] = 0.0
    aw: Ty[P]
    for m in range(M):
        for g, n in allo.grid((NB + P - 1) // P, N):
            d: float32 = 0.0
            with allo.meta_for(P) as j:
                if g * P + j < NB:
                    if n == 0:
                        aw[j] = Aq[m, g * P + j]
                    d += mx_block_dot[Ty, BS](aw[j], Bq[n, g * P + j])
            prev: float32 = 0.0
            if g > 0:
                prev = acc[n]
            total: float32 = prev + d
            acc[n] = total
            if g == (NB - 1) // P:
                C[m, n] = total
    return C


def pipeline_mx_matmul(s, func, N):
    s.pipeline(f"{func}:n", initiation_interval=1 if N >= MX_ACC_LATENCY else 4)


def schedule_mx_matmul(s):
    schedule_mx(s)
    pipeline_mx_matmul(s, s.top_func_name, s.inst_list[2])
    return s


def bias_add[Ty, M, N](X: "float32[M, N]", bias: "Ty[N]") -> "Ty[M, N]":
    # X + bias per row; X is read once, row by row
    Z: Ty[M, N]
    for m, n in allo.grid(M, N):
        Z[m, n] = X[m, n] + bias[n]
    return Z


def merge_rows[Ty, B, L, D](X: "Ty[B, L, D]") -> "Ty[B * L, D]":
    # [B, L, D] -> [B * L, D], read and written in order
    Y: Ty[B * L, D]
    for b, i, d in allo.grid(B, L, D):
        Y[b * L + i, d] = X[b, i, d]
    return Y


def split_rows[Ty, B, L, D](X: "Ty[B * L, D]") -> "Ty[B, L, D]":
    # [B * L, D] -> [B, L, D], read and written in order
    Y: Ty[B, L, D]
    for b, i, d in allo.grid(B, L, D):
        Y[b, i, d] = X[b * L + i, d]
    return Y


def mx_conv_windows[
    Ty, B, C, H, W, Kh, Kw, Oh, Ow, Sh, Sw, Pd0, Pd1
](
    Xq: "Ty[B, (C + Ty.block_size - 1) // Ty.block_size * H * W]",
) -> "Ty[B * Oh * Ow, (C + Ty.block_size - 1) // Ty.block_size * Kh * Kw]":
    # MX image rows -> one row per output pixel holding its receptive field,
    # blocks in (channel block, kh, kw) order and zero blocks for the padding,
    # written in order, one block per iteration
    NB: ConstExpr[int32] = (C + Ty.block_size - 1) // Ty.block_size
    Xw: Ty[B * Oh * Ow, NB * Kh * Kw]
    for n, oh, ow, cb, kh, kw in allo.grid(B, Oh, Ow, NB, Kh, Kw):
        ih: int32 = oh * Sh + kh - Pd0
        iw: int32 = ow * Sw + kw - Pd1
        blk: Ty = 0
        if ih >= 0 and ih < H and iw >= 0 and iw < W:
            blk = Xq[n, (cb * H + ih) * W + iw]
        Xw[(n * Oh + oh) * Ow + ow, (cb * Kh + kh) * Kw + kw] = blk
    return Xw


def rows_to_nchw[Ty, B, C, H, W](Z: "Ty[B * H * W, C]") -> "Ty[B, C, H, W]":
    # one row per pixel (row-major, read in order) -> NCHW
    Y: Ty[B, C, H, W]
    for n, h, w, c in allo.grid(B, H, W, C):
        Y[n, c, h, w] = Z[(n * H + h) * W + w, c]
    return Y


def patch_extern_c_for_class_return_types(kernel_cpp_path):
    # issue #603 (https://github.com/alloy-lang/allo/issues/603)
    with open(kernel_cpp_path, encoding="utf-8") as f:
        lines = f.readlines()
    out = []
    i = 0
    sig_re = re.compile(r"^ap_u?int<\d+>\s+\w+\(")
    patched = []
    while i < len(lines):
        line = lines[i]
        if sig_re.match(line):
            out.append('} // extern "C"\n')
            out.append(line)
            fn_name = line.split()[1].split("(")[0]
            i += 1
            while not lines[i].startswith("}"):
                out.append(lines[i])
                i += 1
            out.append(lines[i])
            out.append('extern "C" {\n')
            patched.append(fn_name)
            i += 1
        else:
            out.append(line)
            i += 1
    with open(kernel_cpp_path, "w", encoding="utf-8") as f:
        f.writelines(out)
    return patched
