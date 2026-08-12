# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=used-before-assignment, unsubscriptable-object

import allo
from ..ir.types import Int, UInt, int32, uint8, uint16, float32, bfloat16
from .._mlir.exceptions import AlloValueError


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


def mx_quantize_block[Ty, K](x: "bfloat16[K]") -> "Ty":
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

    return word


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
        w: Ty = mx_quantize_block[Ty, K](blk)
        scales[b] = _mx_get_scale[Ty](w)
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
    # E8M0 (127-biased power-of-two) scale -> float32 value 2**(scale-127).
    # A float32 with biased exponent field = scale and zero sign/mantissa
    # is exactly 2**(scale-127) by the IEEE754 encoding itself, so the
    # value is assembled with a single shift + bitcast -- no data-dependent
    # loop trip count, unlike the previous repeated-doubling/halving
    # version, which made mx_block_dot's latency variable and blocked
    # Vitis HLS from pipelining mx_dot_general's per-block loop around it.
    # (At the extreme low clamp, scale == 0 now yields 0.0 rather than the
    # previous version's subnormal 2**-127 -- a ~1e-39 discrepancy with no
    # measurable effect on any dot product.)
    bits: int32 = int(scale) << 23
    return bits.bitcast()


def _mx_pack_word[Ty, K](scale: uint8, elems: "uint8[K]") -> "Ty":
    word: Ty = 0
    word[Ty.bits - 8 : Ty.bits] = scale
    for i in range(K):
        word[i * Ty.elem_bits : (i + 1) * Ty.elem_bits] = elems[i]
    return word


def  _mx_get_scale[Ty](word: "Ty") -> uint8:
    return word[Ty.bits - 8 : Ty.bits]


def _mx_acc_bits(Ty, K):
    if Ty.is_float:
        return Ty.block_accum_bits + (K - 1).bit_length()
    return 2 * Ty.elem_bits + (K - 1).bit_length()


def _mx_fp_mul[Ty, K](
    a_i: "UInt(Ty.elem_bits)", b_i: "UInt(Ty.elem_bits)"
) -> "Int(_mx_acc_bits(Ty, K))":
    a_sign: int32 = a_i[Ty.elem_bits - 1 : Ty.elem_bits]
    a_exp_field: int32 = a_i[Ty.mantissa_bits : Ty.elem_bits - 1]
    a_mant: int32 = a_i[0 : Ty.mantissa_bits]
    a_exp: int32 = 1 - Ty.bias
    if a_exp_field != 0:
        a_mant = (1 << Ty.mantissa_bits) | a_mant
        a_exp = a_exp_field - Ty.bias

    b_sign: int32 = b_i[Ty.elem_bits - 1 : Ty.elem_bits]
    b_exp_field: int32 = b_i[Ty.mantissa_bits : Ty.elem_bits - 1]
    b_mant: int32 = b_i[0 : Ty.mantissa_bits]
    b_exp: int32 = 1 - Ty.bias
    if b_exp_field != 0:
        b_mant = (1 << Ty.mantissa_bits) | b_mant
        b_exp = b_exp_field - Ty.bias

    prod_mant: int32 = a_mant * b_mant
    shift_amt: int32 = (a_exp + b_exp) - 2 * (1 - Ty.bias)
    prod_mant_wide: "Int(_mx_acc_bits(Ty, K))" = prod_mant
    magnitude: "Int(_mx_acc_bits(Ty, K))" = prod_mant_wide << shift_amt

    term: "Int(_mx_acc_bits(Ty, K))" = magnitude
    if (a_sign ^ b_sign) == 1:
        term = -magnitude
    return term


def mx_block_dot[Ty, K](data_a: "Ty", data_b: "Ty") -> float32:
    scale_a: uint8 = _mx_get_scale[Ty](data_a)
    scale_b: uint8 = _mx_get_scale[Ty](data_b)
    scale_a_val: float32 = _mx_scale_to_float32(scale_a)
    scale_b_val: float32 = _mx_scale_to_float32(scale_b)

    total: float32 = 0.0
    with allo.meta_if(Ty.is_float):
        mul: "Int(_mx_acc_bits(Ty, K))[K]"
        for j0 in range(K):
            a_i: UInt(Ty.elem_bits) = data_a[j0 * Ty.elem_bits : (j0 + 1) * Ty.elem_bits]
            b_i: UInt(Ty.elem_bits) = data_b[j0 * Ty.elem_bits : (j0 + 1) * Ty.elem_bits]
            mul[j0] = _mx_fp_mul[Ty, K](a_i, b_i)

        acc: "Int(_mx_acc_bits(Ty, K))" = 0
        for j1 in range(K):
            acc = acc + mul[j1]

        pow2_val: float32 = 2.0 ** (2 * (1 - Ty.bias) - 2 * Ty.mantissa_bits)
        total = float(acc) * pow2_val
    with allo.meta_else():
        mul_i: "Int(_mx_acc_bits(Ty, K))[K]"
        for j0 in range(K):
            a_i: Int(Ty.elem_bits) = data_a[j0 * Ty.elem_bits : (j0 + 1) * Ty.elem_bits]
            b_i: Int(Ty.elem_bits) = data_b[j0 * Ty.elem_bits : (j0 + 1) * Ty.elem_bits]
            a_wide: "Int(_mx_acc_bits(Ty, K))" = a_i
            b_wide: "Int(_mx_acc_bits(Ty, K))" = b_i
            mul_i[j0] = a_wide * b_wide

        acc_i: "Int(_mx_acc_bits(Ty, K))" = 0
        for j1 in range(K):
            acc_i = acc_i + mul_i[j1]
        total = float(acc_i)

    return total * scale_a_val * scale_b_val


def schedule_mx_block_dot(s):
    s.unroll("mx_block_dot:j0")
    s.unroll("mx_block_dot:j1")


def block_dot_product[Ty, K](A: "bfloat16[K]", B: "bfloat16[K]") -> float32:
    word_a: Ty = mx_quantize_block[Ty, K](A)
    word_b: Ty = mx_quantize_block[Ty, K](B)
    return mx_block_dot[Ty, K](word_a, word_b)


def schedule_block_dot_product(s):
    schedule_mx_quantize_block(s)
    schedule_mx_block_dot(s)


def mx_dot_general[Ty, K, N, P](
    scales_a: "uint8[N // K]",
    data_a: "uint8[N // K, K]",
    scales_b: "uint8[N // K]",
    data_b: "uint8[N // K, K]",
) -> float32:
    # P independent partial-sum accumulators (one per lane) break the
    # loop-carried float-add dependency that a single running
    # `total = total + block_dot` would put on one register: each lane's
    # own accumulate is still a loop-carried float add across outer trips
    # i and i+1, but with P lanes the same slot is only revisited every P
    # outer trips instead of every one. `j` is unrolled, so `partials[j]`
    # is a compile-time-constant index for each of the P lanes -- no
    # data-dependent index for Vitis's dependence analysis to reason
    # about, and no manual `#pragma HLS dependence` patch needed (see
    # test/mxint8_dot_general_partial_check.py and its P=4 sibling
    # mxint8_dot_general_partial_p4_check.py for the derivation).
    #
    # P should be picked to match mx_block_dot's own achievable
    # throughput on the target device/frequency, not just the float
    # adder's raw latency -- Vitis's scheduler resolves the exact
    # per-cycle dependency, which can be tighter than that rule of thumb
    # (measured: P=4 already reaches II=4 for mxint8 at K=32 on
    # u55c/300MHz, half the DSPs of the naive P=8 choice, at the same
    # overall latency).
    partials: float32[P]
    for p0 in range(P):
        partials[p0] = 0.0

    for i in range(N // K // P):
        for j in range(P):
            b: int32 = i * P + j
            block_a: uint8[K]
            block_b: uint8[K]
            for k in range(K):
                block_a[k] = data_a[b, k]
                block_b[k] = data_b[b, k]
            word_a: Ty = _mx_pack_word[Ty, K](scales_a[b], block_a)
            word_b: Ty = _mx_pack_word[Ty, K](scales_b[b], block_b)
            partials[j] = partials[j] + mx_block_dot[Ty, K](word_a, word_b)

    total: float32 = 0.0
    for r in range(P):
        total = total + partials[r]
    return total


def schedule_mx_dot_general(s, P):
    schedule_mx_block_dot(s)
    # Each lane's own accumulate is still a loop-carried float add across
    # outer trips i and i+1 (P trips apart) -- see mx_dot_general's
    # docstring for why this must be P, not the default 1.
    s.pipeline("mx_dot_general:i", initiation_interval=P)
    s.unroll("mx_dot_general:j")
    s.unroll("mx_dot_general:k")
    s.unroll("mx_dot_general:p0")
    s.unroll("mx_dot_general:r")
    s.partition("mx_dot_general:partials", dim=0)
    for name in ("data_a", "data_b"):
        try:
            s.partition(f"mx_dot_general:{name}", dim=2)
        except AlloValueError:
            pass


def dot_product[Ty, K, N](A: "bfloat16[N]", B: "bfloat16[N]") -> float32:
    scales_a: uint8[N // K]
    scales_b: uint8[N // K]
    data_a: uint8[N // K, K]
    data_b: uint8[N // K, K]
    mx_quantize[Ty, K, N](A, scales_a, data_a)
    mx_quantize[Ty, K, N](B, scales_b, data_b)

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
