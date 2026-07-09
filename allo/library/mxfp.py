# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=used-before-assignment, unsubscriptable-object

import allo
from ..ir.types import UInt, int32, uint8, float32


def _mx_quantize_elem_fp[Ty](v: float32, shared_exp: int32) -> "UInt(Ty.elem_bits)":
    # Requantize one float32 element into Ty's (1 + exp_bits + mantissa_bits)
    # narrow FP element, given the block's shared (already-clamped) exponent.
    # Follows Algorithm 1 of arXiv:2310.10537: v is divided by the block
    # scale (implicitly, via exponent subtraction) then clamped/rounded
    # (round-to-nearest-even) into the element format, preserving Inf/NaN
    # for formats that define them.
    bits: int32 = v.bitcast()
    sign: int32 = bits[31:32]
    exp_field: int32 = bits[23:31]
    mant32: int32 = bits[0:23]

    # top exponent code is reserved for Inf (E5M2-style); top mantissa code
    # at the top exponent is reserved for NaN only when Inf isn't also
    # reserved there (E4M3-style) -- see MXFP.max_unbiased_exp for the
    # matching derivation used when computing the block's shared exponent.
    max_exp_code: int32 = (1 << Ty.exp_bits) - 1
    with allo.meta_if(Ty.has_inf):
        max_exp_code = (1 << Ty.exp_bits) - 2
    max_mant_code: int32 = (1 << Ty.mantissa_bits) - 1
    with allo.meta_if(Ty.has_nan and not Ty.has_inf):
        max_mant_code = (1 << Ty.mantissa_bits) - 2

    out: UInt(Ty.elem_bits) = 0
    is_special: int32 = 0

    with allo.meta_if(Ty.has_nan or Ty.has_inf):
        if exp_field == 255:
            is_special = 1
            with allo.meta_if(Ty.has_nan and Ty.has_inf):
                # E5M2-style: NaN if mantissa nonzero, else Inf
                if mant32 != 0:
                    out = (
                        (sign << (Ty.exp_bits + Ty.mantissa_bits))
                        | (((1 << Ty.exp_bits) - 1) << Ty.mantissa_bits)
                        | ((1 << Ty.mantissa_bits) - 1)
                    )
                else:
                    out = (sign << (Ty.exp_bits + Ty.mantissa_bits)) | (
                        ((1 << Ty.exp_bits) - 1) << Ty.mantissa_bits
                    )
            with allo.meta_else():
                # E4M3-style: only NaN is defined (mantissa all-ones at the
                # top exponent code); an input Inf has no encoding, saturate.
                if mant32 != 0:
                    out = (
                        (sign << (Ty.exp_bits + Ty.mantissa_bits))
                        | (((1 << Ty.exp_bits) - 1) << Ty.mantissa_bits)
                        | ((1 << Ty.mantissa_bits) - 1)
                    )
                else:
                    out = (
                        (sign << (Ty.exp_bits + Ty.mantissa_bits))
                        | (max_exp_code << Ty.mantissa_bits)
                        | max_mant_code
                    )

    if is_special == 0:
        if exp_field == 0:
            # subnormal/zero float32 input -> zero (per Algorithm 1's note)
            out = sign << (Ty.exp_bits + Ty.mantissa_bits)
        else:
            unbiased_exp: int32 = exp_field - 127
            target_exp: int32 = unbiased_exp - shared_exp + Ty.bias
            full_mant: int32 = (1 << 23) | mant32

            # Round to Ty.mantissa_bits (round-to-nearest-even). Widths only
            # depend on compile-time constants, so this is safe regardless
            # of how large/small target_exp is; overflow (including the
            # rounding-carry case, and the E4M3 top-exponent NaN-reserved
            # mantissa pattern) is caught once at the end by the final clamp.
            result_exp: int32 = 0
            result_mant: int32 = 0
            if target_exp <= 0:
                total_shift: int32 = (23 - Ty.mantissa_bits) + (1 - target_exp)
                if total_shift > 24:
                    result_exp = 0
                    result_mant = 0
                else:
                    kept: int32 = full_mant >> total_shift
                    remainder: int32 = full_mant & ((1 << total_shift) - 1)
                    halfpoint: int32 = 1 << (total_shift - 1)
                    round_up: int32 = 0
                    if remainder > halfpoint:
                        round_up = 1
                    elif remainder == halfpoint:
                        if (kept & 1) == 1:
                            round_up = 1
                    if round_up == 1:
                        kept = kept + 1
                    if kept >= (1 << Ty.mantissa_bits):
                        result_exp = 1
                        result_mant = 0
                    else:
                        result_exp = 0
                        result_mant = kept
            else:
                total_shift: int32 = 23 - Ty.mantissa_bits
                kept: int32 = full_mant >> total_shift
                remainder: int32 = full_mant & ((1 << total_shift) - 1)
                halfpoint: int32 = 1 << (total_shift - 1)
                round_up: int32 = 0
                if remainder > halfpoint:
                    round_up = 1
                elif remainder == halfpoint:
                    if (kept & 1) == 1:
                        round_up = 1
                if round_up == 1:
                    kept = kept + 1
                if kept >= (1 << (Ty.mantissa_bits + 1)):
                    result_exp = target_exp + 1
                    result_mant = 0
                else:
                    result_exp = target_exp
                    result_mant = kept & ((1 << Ty.mantissa_bits) - 1)

            if (result_exp > max_exp_code) or (
                result_exp == max_exp_code and result_mant > max_mant_code
            ):
                result_exp = max_exp_code
                result_mant = max_mant_code

            out = (
                (sign << (Ty.exp_bits + Ty.mantissa_bits))
                | (result_exp << Ty.mantissa_bits)
                | result_mant
            )
    return out


def _mx_quantize_elem_int[Ty](v: float32, shared_exp: int32) -> "UInt(Ty.elem_bits)":
    # Requantize one float32 element into Ty's elem_bits-wide two's
    # complement integer, given the block's shared (already-clamped)
    # exponent. round-half-away-from-zero, clamp on overflow.
    bits: int32 = v.bitcast()
    exp_field: int32 = bits[23:31]
    max_val: int32 = (1 << (Ty.elem_bits - 1)) - 1
    min_val: int32 = -(1 << (Ty.elem_bits - 1))

    rounded: int32 = 0
    if exp_field == 255:
        # Inf/NaN input has no integer encoding: saturate by sign
        sign: int32 = bits[31:32]
        if sign == 1:
            rounded = min_val
        else:
            rounded = max_val
    elif exp_field == 0:
        rounded = 0
    else:
        scale_bits: int32 = (shared_exp + 127) << 23
        scale: float32 = scale_bits.bitcast()
        scaled: float32 = v / scale
        if scaled >= 0.0:
            rounded = int(scaled + 0.5)
        else:
            rounded = int(scaled - 0.5)
        rounded = min(rounded, max_val)
        rounded = max(rounded, min_val)
    return rounded


def mx_quantize_block[Ty, K](x: "float32[K]") -> ("uint8", "Ty"):
    # Encode step (not part of the no-dequant dot-product datapath): one
    # K-element float32 block -> (E8M0 scale, packed Ty word). Shared
    # exponent is found from the max IEEE exponent field among the block's
    # elements (a bit-extraction, not a real log2), per Algorithm 1.
    max_exp_field: int32 = 0
    for i in range(K):
        bits_i: int32 = x[i].bitcast()
        exp_field_i: int32 = bits_i[23:31]
        # ignore subnormal/zero (0) and Inf/NaN (255) float32 inputs when
        # searching for the block's shared exponent: neither contributes a
        # meaningful magnitude to scale the block against (Inf/NaN pass
        # through their own encoding downstream regardless of the scale).
        if exp_field_i > max_exp_field and exp_field_i != 255:
            max_exp_field = exp_field_i

    shared_exp: int32 = max_exp_field - 127 - Ty.max_unbiased_exp
    shared_exp = min(shared_exp, 127)
    shared_exp = max(shared_exp, -127)

    scale_field: uint8 = shared_exp + 127

    word: Ty = 0
    word[Ty.bits - 8 : Ty.bits] = scale_field

    for i in range(K):
        elem: UInt(Ty.elem_bits) = 0
        with allo.meta_if(Ty.is_float):
            elem = _mx_quantize_elem_fp[Ty](x[i], shared_exp)
        with allo.meta_else():
            elem = _mx_quantize_elem_int[Ty](x[i], shared_exp)
        word[i * Ty.elem_bits : (i + 1) * Ty.elem_bits] = elem

    return scale_field, word


def mx_quantize[Ty, K, N](
    X: "float32[N]", scales: "uint8[N // K]", data: "Ty[N // K]"
):
    # out-params (not a tuple return): Allo does not support a plain
    # function returning multiple array-typed (memref) results and
    # destructuring them at the call site, only scalar tuple-returns.
    for b in range(N // K):
        blk: float32[K]
        for i in range(K):
            blk[i] = X[b * K + i]
        s, w = mx_quantize_block[Ty, K](blk)
        scales[b] = s
        data[b] = w
