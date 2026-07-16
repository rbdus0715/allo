# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=used-before-assignment, unsubscriptable-object

import allo
from ..ir.types import Int, UInt, int32, uint8, uint16, float32, bfloat16


def _bf16_to_f32(v: bfloat16) -> float32:
    bits16: uint16 = v.bitcast()
    bits32: int32 = int(bits16) << 16
    return bits32.bitcast()


def _mx_quantize_elem_fp[Ty](v: bfloat16, shared_exp: int32) -> "UInt(Ty.elem_bits)":
    # Requantize one bfloat16 element into Ty's (1 + exp_bits + mantissa_bits)
    # narrow FP element, given the block's shared (already-clamped) exponent.
    # Follows Algorithm 1 of arXiv:2310.10537: v is divided by the block
    # scale (implicitly, via exponent subtraction) then clamped/rounded
    # (round-to-nearest-even) into the element format. Pure bit manipulation
    # (no floating-point arithmetic on v itself), so this operates directly
    # on bf16's own (1 + 8 + 7)-bit layout -- same 8-bit exponent field/127
    # bias as float32, just a narrower 7-bit mantissa -- no float32 widening
    # needed (contrast _mx_quantize_elem_int, which does a real division).
    bits: uint16 = v.bitcast()
    sign: int32 = bits[15:16]
    exp_field: int32 = bits[7:15]
    mant16: int32 = bits[0:7]

    max_exp_code: int32 = (1 << Ty.exp_bits) - 1
    max_mant_code: int32 = (1 << Ty.mantissa_bits) - 1

    out: UInt(Ty.elem_bits) = 0

    if exp_field == 0:
        # subnormal/zero bf16 input -> zero (per Algorithm 1's note)
        out = sign << (Ty.exp_bits + Ty.mantissa_bits)
    else:
        unbiased_exp: int32 = exp_field - 127
        target_exp: int32 = unbiased_exp - shared_exp + Ty.bias
        full_mant: int32 = (1 << 7) | mant16

        result_exp: int32 = 0
        result_mant: int32 = 0
        extra_shift: int32 = max(0, 1 - target_exp)
        total_shift: int32 = (7 - Ty.mantissa_bits) + extra_shift
        if total_shift > 8:
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

            if target_exp <= 0:
                if kept >= (1 << Ty.mantissa_bits):
                    result_exp = 1
                    result_mant = 0
                else:
                    result_exp = 0
                    result_mant = kept
            else:
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


def _mx_quantize_elem_int[Ty](v: bfloat16, shared_exp: int32) -> "UInt(Ty.elem_bits)":
    v_f32: float32 = _bf16_to_f32(v)
    bits: int32 = v_f32.bitcast()
    exp_field: int32 = bits[23:31]
    max_val: int32 = (1 << (Ty.elem_bits - 1)) - 1
    min_val: int32 = -(1 << (Ty.elem_bits - 1))

    rounded: int32 = 0
    if exp_field == 0:
        rounded = 0
    else:
        scale_bits: int32 = (shared_exp + 127) << 23
        scale: float32 = scale_bits.bitcast()
        scaled: float32 = v_f32 / scale
        if scaled >= 0.0:
            rounded = int(scaled + 0.5)
        else:
            rounded = int(scaled - 0.5)
        rounded = min(rounded, max_val)
        rounded = max(rounded, min_val)
    return rounded


def mx_quantize_block[Ty, K](x: "bfloat16[K]") -> ("uint8", "Ty"):
    max_exp_field: int32 = 0
    for i0 in range(K):
        bits_i: uint16 = x[i0].bitcast()
        exp_field_i: int32 = bits_i[7:15]
        if exp_field_i > max_exp_field:
            max_exp_field = exp_field_i

    shared_exp: int32 = max_exp_field - 127 - Ty.max_unbiased_exp
    shared_exp = min(shared_exp, 127)
    shared_exp = max(shared_exp, -127)

    scale_field: uint8 = shared_exp + 127

    word: Ty = 0
    word[Ty.bits - 8 : Ty.bits] = scale_field

    for i1 in range(K):
        elem: UInt(Ty.elem_bits) = 0
        with allo.meta_if(Ty.is_float):
            elem = _mx_quantize_elem_fp[Ty](x[i1], shared_exp)
        with allo.meta_else():
            elem = _mx_quantize_elem_int[Ty](x[i1], shared_exp)
        word[i1 * Ty.elem_bits : (i1 + 1) * Ty.elem_bits] = elem

    return scale_field, word


def schedule_mx_quantize_block(s):
    # dev-plan Fig 1 "parallel structure": pipelining a reduction/elementwise
    # loop lets Vitis HLS's own scheduler overlap iterations (parallel-like
    # multiplier throughput) and, for the max_exp_field reduction, rebalance
    # the dependency chain into a tree internally to still hit II=1.
    s.pipeline("mx_quantize_block:i0")
    s.pipeline("mx_quantize_block:i1")


def mx_quantize[Ty, K, N](
    X: "bfloat16[N]", scales: "uint8[N // K]", data: "uint8[N // K, K]"
):
    for b in range(N // K):
        blk: bfloat16[K]
        for i0 in range(K):
            blk[i0] = X[b * K + i0]
        s, w = mx_quantize_block[Ty, K](blk)
        scales[b] = s
        for i1 in range(K):
            data[b, i1] = w[i1 * Ty.elem_bits : (i1 + 1) * Ty.elem_bits]


def schedule_mx_quantize(s):
    s.pipeline("mx_quantize:i0")
    s.pipeline("mx_quantize:i1")
    # Also pipeline the outer per-block loop itself: without this, each
    # block's mx_quantize_block call must fully drain before the next
    # block starts, leaving the inner loops' own pipelines idle between
    # blocks instead of kept continuously fed.
    s.pipeline("mx_quantize:b")


def _mx_scale_to_float32(scale: uint8) -> float32:
    # E8M0 (127-biased power-of-two) scale -> float32 value 2**(scale-127),
    # built via repeated doubling/halving (exact in binary, no rounding).
    shift: int32 = int(scale) - 127
    val: float32 = 1.0
    if shift >= 0:
        s: int32 = 0
        while s < shift:
            val = val * 2.0
            s = s + 1
    else:
        s: int32 = 0
        while s < -shift:
            val = val / 2.0
            s = s + 1
    return val


def _mx_dequantize_elem_fp[Ty](bits: "UInt(Ty.elem_bits)") -> float32:
    # Reconstruct one Ty-encoded (1 + exp_bits + mantissa_bits) narrow FP
    # element back to float32: value = (-1)**sign * significand * 2**exponent,
    # built via repeated doubling/halving (exact in binary) rather than
    # float32 bit reconstruction, so subnormal Ty elements (exp_field == 0)
    # need no separate leading-zero-count renormalize step.
    sign: int32 = bits[Ty.elem_bits - 1 : Ty.elem_bits]
    exp_field: int32 = bits[Ty.mantissa_bits : Ty.elem_bits - 1]
    mant: int32 = bits[0 : Ty.mantissa_bits]

    sig: float32 = float(mant) / float(1 << Ty.mantissa_bits)
    exp_val: int32 = 1 - Ty.bias
    if exp_field != 0:
        sig = sig + 1.0
        exp_val = exp_field - Ty.bias

    val: float32 = sig
    if exp_val >= 0:
        s: int32 = 0
        while s < exp_val:
            val = val * 2.0
            s = s + 1
    else:
        s: int32 = 0
        while s < -exp_val:
            val = val / 2.0
            s = s + 1

    if sign == 1:
        val = -val
    return val


def _mx_pack_word[Ty, K](scale: uint8, elems: "uint8[K]") -> "Ty":
    # Constructor for the packed Ty word: scale in the top 8 bits, the K
    # per-element bytes below it (same layout mx_quantize_block builds).
    # Callers pass a plain uint8[K] byte array (never a Ty[...] array --
    # see mx_quantize's docstring for why that crashes the MLIR JIT), so
    # the wide word only ever exists as this function's scalar local.
    word: Ty = 0
    word[Ty.bits - 8 : Ty.bits] = scale
    for i in range(K):
        word[i * Ty.elem_bits : (i + 1) * Ty.elem_bits] = elems[i]
    return word


def _mx_get_scale[Ty](word: "Ty") -> uint8:
    return word[Ty.bits - 8 : Ty.bits]


def mx_block_dot[Ty, K](data_a: "Ty", data_b: "Ty") -> float32:
    # Simple (non-Kulisch) block dot product: dequantize each element back
    # to float32 and accumulate with ordinary float32 multiply-add. Every
    # add rounds (unlike an exact/Kulisch integer accumulator), trading
    # per-block accumulation accuracy for a much simpler datapath -- no
    # wide fixed-point accumulator sizing, no separate cross-block
    # scale-alignment step (mx_normalize_add) needed either.
    #
    # data_a/data_b are the full packed Ty word (scale in the top 8 bits,
    # elements in the rest -- the layout mx_quantize_block already builds),
    # not scale+data passed separately.
    scale_a: uint8 = _mx_get_scale[Ty](data_a)
    scale_b: uint8 = _mx_get_scale[Ty](data_b)
    scale_a_val: float32 = _mx_scale_to_float32(scale_a)
    scale_b_val: float32 = _mx_scale_to_float32(scale_b)

    acc: float32 = 0.0
    for i in range(K):
        with allo.meta_if(Ty.is_float):
            a_i: UInt(Ty.elem_bits) = data_a[i * Ty.elem_bits : (i + 1) * Ty.elem_bits]
            b_i: UInt(Ty.elem_bits) = data_b[i * Ty.elem_bits : (i + 1) * Ty.elem_bits]
            a_val: float32 = _mx_dequantize_elem_fp[Ty](a_i)
            b_val: float32 = _mx_dequantize_elem_fp[Ty](b_i)
            acc = acc + a_val * b_val
        with allo.meta_else():
            a_i: Int(Ty.elem_bits) = data_a[i * Ty.elem_bits : (i + 1) * Ty.elem_bits]
            b_i: Int(Ty.elem_bits) = data_b[i * Ty.elem_bits : (i + 1) * Ty.elem_bits]
            a_val: float32 = float(a_i)
            b_val: float32 = float(b_i)
            acc = acc + a_val * b_val

    return acc * scale_a_val * scale_b_val


def schedule_mx_block_dot(s):
    # pipelining the K-element dequantize/multiply/accumulate loop overlaps
    # the per-element dequantizers and multipliers across iterations and
    # lets Vitis HLS's reduction-variable optimization rebalance the acc +=
    # dependency chain into a tree internally, hitting II=1 despite the
    # apparent sequential dependency in the source.
    s.pipeline("mx_block_dot:i")


def mx_dot_general[Ty, K, N](
    scales_a: "uint8[N // K]",
    data_a: "uint8[N // K, K]",
    scales_b: "uint8[N // K]",
    data_b: "uint8[N // K, K]",
) -> float32:
    # Cross-block reduction: each block's dequantized dot product
    # (mx_block_dot, already a plain float32) is added directly into a
    # running float32 total -- ordinary float32 addition already handles
    # aligning different blocks' magnitudes, so no separate scale-tracking
    # or exponent-alignment step is needed here.
    #
    # data_a/data_b hold per-element raw bytes, not `Ty[N // K]` arrays --
    # see mx_quantize's comment for why a live array of the wide packed Ty
    # word crashes the MLIR JIT ExecutionEngine. Each block's packed word
    # is rebuilt (via _mx_pack_word) as a scalar local immediately before
    # use, never stored into an array.
    total: float32 = 0.0

    for b in range(N // K):
        block_a: uint8[K]
        block_b: uint8[K]
        for i in range(K):
            block_a[i] = data_a[b, i]
            block_b[i] = data_b[b, i]
        word_a: Ty = _mx_pack_word[Ty, K](scales_a[b], block_a)
        word_b: Ty = _mx_pack_word[Ty, K](scales_b[b], block_b)
        total = total + mx_block_dot[Ty, K](word_a, word_b)

    return total


def schedule_mx_dot_general(s):
    # Pipeline the word-reconstruction loop (see its own comment on why the
    # packed Ty word is rebuilt from bytes here rather than stored in an
    # array). mx_block_dot's own K-loop is pipelined separately by
    # schedule_mx_block_dot when that's scheduled as part of the same
    # kernel; this is the N//K cross-block loop's own inner K-loop.
    s.pipeline("mx_dot_general:i")
    # Also pipeline the  outer per-block (N // K) loop -- it calls
    # mx_block_dot once per block; without this, each block's full compute
    # (including mx_block_dot's own internally pipelined K-loop) must fully
    # drain before the next block's starts, rather than overlapping
    # consecutive blocks through the same hardware.
    s.pipeline("mx_dot_general:b")


def dot_product[Ty, K, N](A: "bfloat16[N]", B: "bfloat16[N]") -> float32:
    scales_a: uint8[N // K]
    scales_b: uint8[N // K]
    data_a: uint8[N // K, K]
    data_b: uint8[N // K, K]
    mx_quantize[Ty, K, N](A, scales_a, data_a)
    mx_quantize[Ty, K, N](B, scales_b, data_b)

    # Inlined from mx_dot_general: dot_product itself builds each block's
    # packed Ty word (via _mx_pack_word, as a scalar local -- never as an
    # array, see mx_quantize's docstring for why) and hands it directly to
    # mx_block_dot.
    total: float32 = 0.0
    for b in range(N // K):
        block_a: uint8[K]
        block_b: uint8[K]
        for i in range(K):
            block_a[i] = data_a[b, i]
            block_b[i] = data_b[b, i]
        word_a: Ty = _mx_pack_word[Ty, K](scales_a[b], block_a)
        word_b: Ty = _mx_pack_word[Ty, K](scales_b[b], block_b)
        total = total + mx_block_dot[Ty, K](word_a, word_b)

    return total
