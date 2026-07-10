# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=used-before-assignment, unsubscriptable-object

import allo
from ..ir.types import Int, UInt, int32, uint8, uint1, float32

# extra guard bits mx_normalize_add widens its intermediate sum by, so a
# single addition's worst-case 1-bit magnitude growth never truncates the
# alignment-shift's own rounded result before the final renormalize step.
# Must be a plain Python int (not an Allo-typed kernel variable): it is
# used inside Int(...) bit-width expressions, which only accept
# compile-time constants.
_NORM_ADD_GUARD = 3


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
    X: "float32[N]", scales: "uint8[N // K]", data: "uint8[N // K, K]"
):
    # out-params (not a tuple return): Allo does not support a plain
    # function returning multiple array-typed (memref) results and
    # destructuring them at the call site, only scalar tuple-returns.
    #
    # `data` holds per-element raw bytes, not an array of the packed Ty
    # word (`Ty[N // K]`): a local/live memref whose element type is wider
    # than 64 bits and has 3+ elements crashes the MLIR JIT's
    # ExecutionEngine at MLIRContext teardown (StorageUniquer heap
    # corruption -- reproduced with plain UInt(72)[N] independent of any
    # mxfp-specific logic). The packed Ty word is only ever materialized
    # as a scalar local (`w` below), never stored into an array.
    for b in range(N // K):
        blk: float32[K]
        for i in range(K):
            blk[i] = X[b * K + i]
        s, w = mx_quantize_block[Ty, K](blk)
        scales[b] = s
        for i in range(K):
            data[b, i] = w[i * Ty.elem_bits : (i + 1) * Ty.elem_bits]


def mx_block_dot[Ty, K](
    scale_a: uint8, data_a: "Ty", scale_b: uint8, data_b: "Ty"
) -> ("uint8", "Int(Ty.final_accum_bits)", "uint1", "uint1"):
    # Fig 1's block dot product: integer multiply (k x multipliers) +
    # Kulisch (exact, no per-term rounding) accumulation, no
    # dequantization to float anywhere. Uses final_accum_bits (not the
    # narrower block_accum_bits) as the accumulator width: block_accum_bits
    # only bounds a single product term, but summing up to block_size of
    # them (all same sign, all near the format's max magnitude is a valid,
    # if adversarial, input) needs ceil(log2(block_size)) extra guard bits
    # to stay overflow-free -- confirmed empirically, since block_accum_bits
    # alone overflows on a same-sign, max-magnitude stress input.
    # Scales are E8M0 (biased by 127, representing a power of two), so
    # combining them means adding the *unbiased* exponents, not the raw
    # fields: (scale_a-127) + (scale_b-127) + 127 = scale_a + scale_b - 127.
    combined_scale: int32 = int(scale_a) + int(scale_b) - 127

    # eexp_min_single is the reference point the per-element significand
    # products below are shifted against so every product lands at a
    # non-negative bit offset within the fixed-width accumulator (see
    # shift_amount below). Folding 2*eexp_min_single into the returned
    # scale here undoes that shift, so (scale_out, acc) together still
    # satisfy true_dot_value == acc * 2**(scale_out - 127), exactly like
    # a single quantized MXFP value -- callers don't need to know this
    # implementation detail of how the accumulator was aligned.
    eexp_min_single: int32 = 0
    with allo.meta_if(Ty.is_float):
        eexp_min_single = 1 - Ty.bias - Ty.mantissa_bits
        combined_scale = combined_scale + 2 * eexp_min_single

    combined_scale = min(combined_scale, 254)
    combined_scale = max(combined_scale, 0)
    scale_out: uint8 = combined_scale

    acc: Int(Ty.final_accum_bits) = 0
    has_nan: uint1 = 0
    has_inf: uint1 = 0

    for i in range(K):
        with allo.meta_if(Ty.is_float):
            a_i: UInt(Ty.elem_bits) = data_a[i * Ty.elem_bits : (i + 1) * Ty.elem_bits]
            b_i: UInt(Ty.elem_bits) = data_b[i * Ty.elem_bits : (i + 1) * Ty.elem_bits]
            sign_a: int32 = a_i[Ty.elem_bits - 1 : Ty.elem_bits]
            ec_a: int32 = a_i[Ty.mantissa_bits : Ty.elem_bits - 1]
            m_a: int32 = a_i[0 : Ty.mantissa_bits]
            sign_b: int32 = b_i[Ty.elem_bits - 1 : Ty.elem_bits]
            ec_b: int32 = b_i[Ty.mantissa_bits : Ty.elem_bits - 1]
            m_b: int32 = b_i[0 : Ty.mantissa_bits]

            # Sticky NaN/Inf detection: threaded as separate flags rather
            # than encoded in the integer accumulator, since ordinary
            # two's-complement add/shift cannot propagate IEEE-style
            # special values (see mxfp_dev_plan design notes).
            is_special_pair: int32 = 0
            with allo.meta_if(Ty.has_nan or Ty.has_inf):
                all_ones: int32 = (1 << Ty.exp_bits) - 1
                a_special: int32 = 0
                b_special: int32 = 0
                with allo.meta_if(Ty.has_nan and Ty.has_inf):
                    # E5M2-style: the whole top exponent code is reserved
                    # (either Inf or NaN), regardless of mantissa.
                    if ec_a == all_ones:
                        a_special = 1
                    if ec_b == all_ones:
                        b_special = 1
                with allo.meta_else():
                    # E4M3-style: only the single all-ones-mantissa pattern
                    # at the top exponent code is NaN; every other mantissa
                    # there (e.g. 448's encoding) is an ordinary normal
                    # value and must not be treated as special.
                    nan_mant: int32 = (1 << Ty.mantissa_bits) - 1
                    if ec_a == all_ones and m_a == nan_mant:
                        a_special = 1
                    if ec_b == all_ones and m_b == nan_mant:
                        b_special = 1

                if a_special == 1 or b_special == 1:
                    is_special_pair = 1
                    with allo.meta_if(Ty.has_nan and Ty.has_inf):
                        # E5M2-style: mantissa nonzero at the reserved
                        # exponent code means NaN, else Inf.
                        a_nan: int32 = 0
                        if a_special == 1 and m_a != 0:
                            a_nan = 1
                        b_nan: int32 = 0
                        if b_special == 1 and m_b != 0:
                            b_nan = 1
                        if a_nan == 1 or b_nan == 1:
                            has_nan = 1
                        else:
                            has_inf = 1
                    with allo.meta_else():
                        # E4M3-style: the reserved code only ever means
                        # NaN (no Inf encoding exists in this format).
                        has_nan = 1

            if is_special_pair == 0:
                ec_a_eff: int32 = ec_a
                if ec_a_eff == 0:
                    ec_a_eff = 1
                ec_b_eff: int32 = ec_b
                if ec_b_eff == 0:
                    ec_b_eff = 1

                sig_a: int32 = m_a
                if ec_a != 0:
                    sig_a = (1 << Ty.mantissa_bits) | m_a
                sig_b: int32 = m_b
                if ec_b != 0:
                    sig_b = (1 << Ty.mantissa_bits) | m_b

                # unified effective exponent (true magnitude = sig * 2**eexp)
                # for both normal (ec>0) and subnormal (ec==0) elements
                eexp_a: int32 = ec_a_eff - Ty.bias - Ty.mantissa_bits
                eexp_b: int32 = ec_b_eff - Ty.bias - Ty.mantissa_bits

                shift_amount: int32 = (eexp_a + eexp_b) - 2 * eexp_min_single
                product_mag: Int(Ty.final_accum_bits) = sig_a * sig_b
                product_mag = product_mag << shift_amount

                if (sign_a ^ sign_b) == 1:
                    acc = acc - product_mag
                else:
                    acc = acc + product_mag
        with allo.meta_else():
            a_i: Int(Ty.elem_bits) = data_a[i * Ty.elem_bits : (i + 1) * Ty.elem_bits]
            b_i: Int(Ty.elem_bits) = data_b[i * Ty.elem_bits : (i + 1) * Ty.elem_bits]
            acc = acc + int(a_i) * int(b_i)

    return scale_out, acc, has_nan, has_inf


def mx_normalize_add[Ty](
    scale0: uint8, op0: "Int(Ty.final_accum_bits)", scale1: uint8, op1: "Int(Ty.final_accum_bits)"
) -> ("uint8", "Int(Ty.final_accum_bits)"):
    # Fig 2: a floating-point-adder-like normalising adder, but combining
    # whole Kulisch block-dot results rather than per-element mantissas.
    # Sort by scale, align (barrel-shift + round-to-nearest-even) the
    # smaller-scale operand into the larger-scale operand's frame, add
    # losslessly in a widened (BW+GUARD) frame, then round+renormalize
    # the (at most 1-bit-wider) sum back down to BW bits -- exactly like
    # a mantissa-carry pushing a floating-point adder's exponent up by 1.
    #
    # Parametrized by Ty (not a bare bit-width) only because Allo's
    # generic-subscript resolver doesn't accept an attribute expression
    # like mx_normalize_add[Ty.final_accum_bits] at the call site -- Ty
    # itself, a bare name, works fine, so BW is derived internally instead.
    # NOTE: bare literal `1` defaults to int32, so `1 << shift_amount` would
    # silently overflow/wrap for shift_amount >= 32 (very real here: BW can
    # be into the 40s). Every wide shift below starts from this
    # already-widened constant instead of a bare `1`.
    wide_one: Int(Ty.final_accum_bits + _NORM_ADD_GUARD) = 1
    BW: int32 = Ty.final_accum_bits

    hi_scale: int32 = int(scale0)
    hi_op: Int(Ty.final_accum_bits + _NORM_ADD_GUARD) = op0
    lo_scale: int32 = int(scale1)
    lo_op: Int(Ty.final_accum_bits + _NORM_ADD_GUARD) = op1
    if int(scale1) > int(scale0):
        hi_scale = int(scale1)
        hi_op = op1
        lo_scale = int(scale0)
        lo_op = op0

    diff: int32 = hi_scale - lo_scale

    # NOTE: round-then-add (rounding the shifted-down lo_op on its own,
    # before adding it to hi_op) is a classic double-rounding bug -- e.g.
    # hi_op=-215 (scale134), lo_op=-1438 (scale132, diff=2): the exact
    # combined value is -574.5 (scale134 units), a tie whose correct
    # round-to-even result is -574 (even). Rounding lo_op/4=-359.5 to -360
    # (even) *first*, then adding -215, gives -575 (odd) instead -- wrong
    # by a full unit at this scale. Fix: add hi_op + floor(lo_op >> diff)
    # *unrounded* first, keep a guard/sticky pair from the discarded bits,
    # and make the single round-to-nearest-even decision against the
    # SUM's own parity, not the shifted operand's parity in isolation.
    total: Int(Ty.final_accum_bits + _NORM_ADD_GUARD) = 0
    if diff > 0:
        if diff >= BW + _NORM_ADD_GUARD:
            total = hi_op
        else:
            kept: Int(Ty.final_accum_bits + _NORM_ADD_GUARD) = lo_op >> diff
            remainder: Int(Ty.final_accum_bits + _NORM_ADD_GUARD) = lo_op & (
                (wide_one << diff) - 1
            )
            halfpoint: Int(Ty.final_accum_bits + _NORM_ADD_GUARD) = wide_one << (
                diff - 1
            )
            total = hi_op + kept
            round_up: int32 = 0
            if remainder > halfpoint:
                round_up = 1
            elif remainder == halfpoint:
                if (total & 1) == 1:
                    round_up = 1
            if round_up == 1:
                total = total + 1
    else:
        total = hi_op + lo_op

    max_val: Int(Ty.final_accum_bits + _NORM_ADD_GUARD) = (
        wide_one << (BW - 1)
    ) - 1
    min_val: Int(Ty.final_accum_bits + _NORM_ADD_GUARD) = -(wide_one << (BW - 1))

    out_scale: int32 = hi_scale
    result: Int(Ty.final_accum_bits + _NORM_ADD_GUARD) = total
    if total > max_val or total < min_val:
        # a single addition can grow the magnitude by at most one bit;
        # shift right by 1 (round-to-nearest-even) and bump the scale,
        # mirroring a floating-point adder's post-normalize step.
        bit0: Int(Ty.final_accum_bits + _NORM_ADD_GUARD) = total & 1
        result = total >> 1
        if bit0 == 1:
            if (result & 1) == 1:
                result = result + 1
        out_scale = hi_scale + 1

    out_scale = min(out_scale, 254)
    out_scale = max(out_scale, 0)
    scale_out: uint8 = out_scale
    out: Int(Ty.final_accum_bits) = result
    return scale_out, out


def mx_dot_general[Ty, K, N](
    scales_a: "uint8[N // K]",
    data_a: "uint8[N // K, K]",
    scales_b: "uint8[N // K]",
    data_b: "uint8[N // K, K]",
) -> ("uint8", "Int(Ty.final_accum_bits)", "uint1", "uint1"):
    # Cross-block reduction (Fig 1 + Fig 2 combined): each block pair's
    # mx_block_dot result is folded into a running (scale, acc) total via
    # mx_normalize_add, with has_nan/has_inf sticky-OR'd in alongside
    # (kept a separate, much simpler concern from the magnitude path, see
    # mx_block_dot). No dequantization to float anywhere in this reduction.
    #
    # data_a/data_b hold per-element raw bytes, not `Ty[N // K]` arrays --
    # see mx_quantize's comment for why a live array of the wide packed Ty
    # word crashes the MLIR JIT ExecutionEngine. Each block's packed word
    # is rebuilt as a scalar local (word_a/word_b) immediately before use.
    acc_scale: uint8 = 0
    acc_val: Int(Ty.final_accum_bits) = 0
    has_nan: uint1 = 0
    has_inf: uint1 = 0

    for b in range(N // K):
        word_a: Ty = 0
        word_b: Ty = 0
        for i in range(K):
            word_a[i * Ty.elem_bits : (i + 1) * Ty.elem_bits] = data_a[b, i]
            word_b[i * Ty.elem_bits : (i + 1) * Ty.elem_bits] = data_b[b, i]
        blk_scale, blk_acc, blk_nan, blk_inf = mx_block_dot[Ty, K](
            scales_a[b], word_a, scales_b[b], word_b
        )
        if b == 0:
            acc_scale = blk_scale
            acc_val = blk_acc
        else:
            acc_scale, acc_val = mx_normalize_add[Ty](
                acc_scale, acc_val, blk_scale, blk_acc
            )
        if blk_nan == 1:
            has_nan = 1
        if blk_inf == 1:
            has_inf = 1

    return acc_scale, acc_val, has_nan, has_inf


def mx_to_float32[Ty](
    scale: uint8, val: "Int(Ty.final_accum_bits)", has_nan: uint1, has_inf: uint1
) -> float32:
    # Boundary-only float32 conversion (Fig 1/2's four-port (scale, acc,
    # has_nan, has_inf) design): the only place an MXFP dot-product result
    # ever leaves the integer domain. has_inf always maps to +Inf here --
    # mx_block_dot skips a special element's product term entirely rather
    # than folding a signed Inf into acc (see its is_special_pair branch),
    # so no sign is ever tracked for Inf through the accumulator. This is
    # a documented simplification, not full IEEE-754 Inf-sign propagation.
    NAN_BITS: int32 = 0x7FC00000
    INF_BITS: int32 = 0x7F800000
    out: float32 = 0.0
    if has_nan == 1:
        out = NAN_BITS.bitcast()
    elif has_inf == 1:
        out = INF_BITS.bitcast()
    else:
        shift: int32 = int(scale) - 127
        acc_f: float32 = float(val)
        if shift >= 0:
            s: int32 = 0
            while s < shift:
                acc_f = acc_f * 2.0
                s = s + 1
        else:
            s: int32 = 0
            while s < -shift:
                acc_f = acc_f / 2.0
                s = s + 1
        out = acc_f
    return out


def dot_product[Ty, K, N](A: "float32[N]", B: "float32[N]") -> float32:
    # Top-level user-facing kernel: plain float32[N] operands in, a single
    # float32 scalar out. Quantize -> block_dot (folded into dot_general's
    # cross-block reduction) -> to_float32, with no dequantization to
    # float anywhere in between -- everything from mx_quantize through
    # mx_dot_general stays in packed-integer/E8M0-scale form.
    scales_a: uint8[N // K]
    scales_b: uint8[N // K]
    data_a: uint8[N // K, K]
    data_b: uint8[N // K, K]
    mx_quantize[Ty, K, N](A, scales_a, data_a)
    mx_quantize[Ty, K, N](B, scales_b, data_b)
    out_scale, acc, has_nan, has_inf = mx_dot_general[Ty, K, N](
        scales_a, data_a, scales_b, data_b
    )
    return mx_to_float32[Ty](out_scale, acc, has_nan, has_inf)
