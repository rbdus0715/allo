"""MXFP8 block format: Python reference + Allo kernels."""

# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=used-before-assignment, unsubscriptable-object, missing-function-docstring

import math

import numpy as np

from .. import dsl
from ..ir.types import uint8, int32, float32

MXFP8_BLOCK_SIZE = 32
E4M3_BIAS = 7
E8M0_BIAS = 127


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


def ref_block_add(
    scale1: int, data1: np.ndarray, scale2: int, data2: np.ndarray
) -> tuple[int, np.ndarray]:
    arr1 = ref_decode_block(scale1, data1)
    arr2 = ref_decode_block(scale2, data2)
    arr_sum = arr1 + arr2
    return ref_encode_block(arr_sum)


# ---------------------------------------------------------------------------
# Allo kernels (compile to LLVM / HLS)
# ---------------------------------------------------------------------------


def decode_e8m0(u8: uint8) -> float32:
    u: int32 = int(u8)
    result: float32 = 0.0
    if u != 0 and u != 255:
        exp: int32 = u - E8M0_BIAS
        result = 2.0**float(exp)
    return result


def encode_e8m0(scale: float32) -> uint8:
    result: int32 = 0
    if scale > 0.0:
        result = 254
        for e in range(1, 255):
            if result == 254:
                p: float32 = 2.0 ** float(e - E8M0_BIAS)
                if p >= scale:
                    result = e
    return result


def decode_e4m3(u8: uint8) -> float32:
    u: int32 = int(u8)
    sign: int32 = (u >> 7) & 1
    exp: int32 = (u >> 3) & 15
    mant: int32 = u & 7

    val: float32 = 0.0
    if exp == 15 and mant == 7:
        val = 0.0
    elif exp == 0:
        val = float(mant) / 8.0 * (2.0 ** float(1 - E4M3_BIAS))
    else:
        val = (1.0 + float(mant) / 8.0) * (2.0 ** float(exp - E4M3_BIAS))

    if sign == 1:
        val = 0.0 - val
    return val


def encode_e4m3(f: float32) -> uint8:
    packed: int32 = 0
    if f != 0.0:
        sign: int32 = 0
        abs_f: float32 = f
        if f < 0.0:
            sign = 1
            abs_f = 0.0 - f

        unbiased_exp: int32 = -20
        for e in range(-20, 16):
            if abs_f >= 2.0 ** float(e):
                unbiased_exp = e

        exp_field: int32 = unbiased_exp + E4M3_BIAS
        mant: int32 = 0

        if exp_field <= 0:
            exp_field = 0
            divisor: float32 = 2.0 ** float(1 - E4M3_BIAS)
            mant = int(abs_f / divisor * 8.0 + 0.5)
        else:
            divisor = 2.0 ** float(unbiased_exp)
            mant = int((abs_f / divisor - 1.0) * 8.0 + 0.5)
            if mant == 8:
                mant = 0
                exp_field += 1

        if exp_field >= 15:
            exp_field = 14
            mant = 7

        packed = (sign << 7) | (exp_field << 3) | mant
    return packed


def mxfp8_decode_block[BS: int32](
    scale: uint8, data: uint8[BS], out: float32[BS]
):
    s: float32 = decode_e8m0(scale)
    for i in dsl.grid(BS, name="decode"):
        out[i] = decode_e4m3(data[i]) * s


def mxfp8_encode_block[BS: int32](
    data: float32[BS], scale_out: uint8[1], data_out: uint8[BS]
):
    max_val: float32 = 0.0
    for i in dsl.grid(BS, name="find_max"):
        v: float32 = data[i]
        av: float32 = v
        if v < 0.0:
            av = 0.0 - v
        if av > max_val:
            max_val = av

    scale_out[0] = encode_e8m0(max_val)
    s: float32 = decode_e8m0(scale_out[0])

    for i in dsl.grid(BS, name="encode"):
        scaled: float32 = data[i]
        if s != 0.0:
            scaled = data[i] / s
        data_out[i] = encode_e4m3(scaled)


def mxfp8_block_add[BS: int32](
    scale_a: uint8,
    data_a: uint8[BS],
    scale_b: uint8,
    data_b: uint8[BS],
    scale_out: uint8[1],
    data_out: uint8[BS],
):
    buf: float32[BS]
    sa: float32 = decode_e8m0(scale_a)
    sb: float32 = decode_e8m0(scale_b)
    for i in dsl.grid(BS, name="decode_add"):
        buf[i] = decode_e4m3(data_a[i]) * sa + decode_e4m3(data_b[i]) * sb
    mxfp8_encode_block[BS](buf, scale_out, data_out)


def schedule_mxfp8_block_add(s):
    assert s.top_func_name == "mxfp8_block_add"
    loops = s.get_loops(s.top_func_name)
    s.pipeline(loops["decode_add"]["i"])
    encode_loops = s.get_loops("mxfp8_encode_block")
    s.pipeline(encode_loops["find_max"]["i"])
    s.pipeline(encode_loops["encode"]["i"])
    return s
