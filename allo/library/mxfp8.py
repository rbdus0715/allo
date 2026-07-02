"""MXFP8 block format: Python reference + Allo kernels."""

# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=used-before-assignment, unsubscriptable-object, missing-function-docstring

import math

import numpy as np

from .. import mxfp8_ops
from ..ir.types import uint8, int32, float32, mxfp8

MXFP8_BLOCK_SIZE = 32
E4M3_BIAS = 7
E8M0_BIAS = 127
MX_INT_W = 24


# ---------------------------------------------------------------------------
# Python reference (golden model for verification)
# ---------------------------------------------------------------------------


def ref_decode_e8m0(u8: int) -> float:
    u8 = int(u8) & 0xFF
    if u8 == 0 or u8 == 0xFF:
        return float("nan")
    return float(2.0 ** (u8 - E8M0_BIAS))


def ref_encode_e8m0(scale: float) -> int:
    if scale <= 0:
        return 0
    exp = int(math.ceil(math.log2(scale))) + E8M0_BIAS
    return max(1, min(254, exp))


def ref_decode_e4m3(u8: int) -> float:
    u8 = int(u8) & 0xFF
    sign = (u8 >> 7) & 1
    exp = (u8 >> 3) & 0xF
    mant = u8 & 0x7

    if exp == 15 and mant == 7:
        return float("nan")

    if exp == 0:
        val = mant / 8.0 * 2.0 ** (1 - E4M3_BIAS)
    else:
        val = (1.0 + mant / 8.0) * 2.0 ** (exp - E4M3_BIAS)
    return -val if sign else val


def ref_encode_e4m3(f: float) -> int:
    if math.isnan(f):
        return 0x7F
    if f == 0:
        return 0

    sign = 1 if f < 0 else 0
    abs_f = abs(f)

    exp = int(math.floor(math.log2(abs_f))) + E4M3_BIAS

    if exp <= 0:
        exp = 0
        mant = int(round((abs_f / 2.0 ** (1 - E4M3_BIAS)) * 8))
    else:
        mant = int(round((abs_f / 2.0 ** (exp - E4M3_BIAS) - 1.0) * 8))
        if mant == 8:
            mant = 0
            exp += 1

    if exp >= 15:
        exp = 14
        mant = 7

    return (sign << 7) | (exp << 3) | mant


def ref_decode_block(scale_u8: int, data: np.ndarray) -> np.ndarray:
    s = ref_decode_e8m0(scale_u8)
    if math.isnan(s):
        return np.full(len(data), np.nan, dtype=np.float32)
    return np.array([ref_decode_e4m3(x) * s for x in data], dtype=np.float32)


def ref_encode_block(data: np.ndarray) -> tuple[int, np.ndarray]:
    max_val = np.max(np.abs(data))
    scale_u8 = ref_encode_e8m0(max_val)
    s = ref_decode_e8m0(scale_u8)

    scaled_data = data / s if s != 0 else data
    encoded = np.array([ref_encode_e4m3(x) for x in scaled_data], dtype=np.uint8)
    return scale_u8, encoded


def _sign_extend(val: int, bits: int) -> int:
    val = int(val) & ((1 << bits) - 1)
    if val & (1 << (bits - 1)):
        val -= 1 << bits
    return val


def ref_unpack_mx_elem(e4m3_byte: int, block_scale_u8: int) -> tuple[int, int]:
    """Unpack one MXFP8 element to (signed mantissa, combined scale byte)."""
    u8 = int(e4m3_byte) & 0xFF
    block_scale = int(block_scale_u8) & 0xFF
    if u8 == 0 or block_scale == 0:
        return 0, 0

    sign = (u8 >> 7) & 1
    exp = (u8 >> 3) & 0xF
    mant = u8 & 0x7
    if exp == 15 and mant == 7:
        return 0, 0

    if exp == 0:
        op = -mant if sign else mant
        scale = block_scale - 9
    else:
        op = -(8 + mant) if sign else (8 + mant)
        scale = block_scale + exp - E4M3_BIAS
    return _sign_extend(op, MX_INT_W), scale & 0xFF


def ref_nrm_to_float(op: int, scale: int) -> float:
    if op == 0:
        return 0.0
    return _sign_extend(op, MX_INT_W) * (2.0 ** (scale - E8M0_BIAS - 3))


def ref_add_nrm(
    op0: int, op1: int, scale0: int, scale1: int, int_w: int = MX_INT_W
) -> tuple[int, int]:
    """Hardware-style MX mantissa addition with scale alignment (add_nrm)."""
    op0 = _sign_extend(op0, int_w)
    op1 = _sign_extend(op1, int_w)
    scale0 = int(scale0) & 0xFF
    scale1 = int(scale1) & 0xFF

    if scale0 < scale1:
        op_lrg, op_sml = op1, op0
        scale_lrg, _scale_sml = scale1, scale0
    else:
        op_lrg, op_sml = op0, op1
        scale_lrg, _scale_sml = scale0, scale1

    scale_diff = (scale_lrg - _scale_sml) & 0xFF
    if scale_diff > 3:
        shift_amt = scale_diff - 3
        all_ones = (1 << int_w) - 1
        sticky_mask = (~(all_ones << shift_amt)) & all_ones
        sticky = 1 if (op_sml & sticky_mask) else 0
    else:
        sticky = 0

    aug_sml = _sign_extend(op_sml << 3, int_w + 4)
    if scale_diff < int_w + 4:
        aug_sml = _sign_extend(aug_sml >> scale_diff, int_w + 4)
    else:
        aug_sml = 0
    aug_sml = (aug_sml & ~1) | sticky

    aug_lrg = _sign_extend(op_lrg << 3, int_w + 4)
    total = _sign_extend(aug_lrg + aug_sml, int_w + 4)
    if total == 0:
        return 0, scale_lrg

    rnd_bit = (total >> 3) & 1
    sticky_bits = total & 0x7
    lsb = (total >> 4) & 1
    inc = rnd_bit and (sticky_bits != 0 or lsb)
    out = _sign_extend((total >> 3) + (1 if inc else 0), int_w)

    limit = 1 << (int_w - 1)
    scale_adj = scale_lrg
    while out >= limit or out < -limit:
        dropped = out & 1
        out = _sign_extend(out >> 1, int_w)
        if dropped and out != 0:
            out = _sign_extend(out + (1 if out > 0 else -1), int_w)
        scale_adj = (scale_adj + 1) & 0xFF
    return out, scale_adj


def ref_block_add(
    scale1: int, data1: np.ndarray, scale2: int, data2: np.ndarray
) -> tuple[int, np.ndarray]:
    buf = np.zeros(len(data1), dtype=np.float32)
    for i, (d1, d2) in enumerate(zip(data1, data2)):
        o0, sc0 = ref_unpack_mx_elem(d1, scale1)
        o1, sc1 = ref_unpack_mx_elem(d2, scale2)
        out, osc = ref_add_nrm(o0, o1, sc0, sc1)
        buf[i] = ref_nrm_to_float(out, osc)
    return ref_encode_block(buf)


# ---------------------------------------------------------------------------
# Allo kernels (native MXFP8 ops; compile to LLVM / HLS)
# ---------------------------------------------------------------------------


def decode_e8m0(u8: uint8) -> float32:
    return mxfp8_ops.decode_e8m0(u8)


def encode_e8m0(scale: float32) -> uint8:
    return mxfp8_ops.encode_e8m0(scale)


def decode_e4m3(u8: uint8) -> float32:
    return mxfp8_ops.decode_e4m3(u8)


def encode_e4m3(f: float32) -> uint8:
    return mxfp8_ops.encode_e4m3(f)


def mxfp8_decode_block[BS: int32](scale: uint8, data: mxfp8[BS], out: float32[BS]):
    mxfp8_ops.decode_mxfp8_block(scale, data, out)


def mxfp8_encode_block[BS: int32](
    data: float32[BS], scale_out: uint8[1], data_out: mxfp8[BS]
):
    mxfp8_ops.encode_mxfp8_block(data, scale_out, data_out)


def mxfp8_block_add[BS: int32](
    scale_a: uint8,
    data_a: mxfp8[BS],
    scale_b: uint8,
    data_b: mxfp8[BS],
    scale_out: uint8[1],
    data_out: mxfp8[BS],
):
    mxfp8_ops.block_add_mxfp8(
        scale_a, data_a, scale_b, data_b, scale_out, data_out
    )


def mxfp8_block_matmul[BS: int32](
    scale_a: uint8,
    data_a: mxfp8[BS],
    scale_b: uint8,
    data_b: mxfp8[BS],
    scale_out: uint8[1],
    data_out: mxfp8[BS],
):
    mxfp8_ops.block_matmul_mxfp8(
        scale_a, data_a, scale_b, data_b, scale_out, data_out
    )


def schedule_mxfp8_block_add(s):
    # Native block_add_mxfp8 lowers to a single intrinsic without inner loops.
    return s
