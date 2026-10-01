# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=used-before-assignment, unsubscriptable-object

import allo
import allo.dataflow as df
from ..ir.types import Int, UInt, int32, uint8, uint16, float32, Stream, ConstExpr
from .mxint8 import _mx_pack_word, _mx_get_scale


def _mx_quantize_elem_fp_bf16[Ty](v_bf16: uint16, shared_field: uint8) -> "UInt(Ty.elem_bits)":
    sign: UInt(1) = v_bf16[15:16]
    exp_field: uint8 = v_bf16[7:15]

    local_exp: int32 = (
        int(v_bf16[7:15]) - int(shared_field) + Ty.bias + Ty.max_unbiased_exp
    )
    max_exp_code: int32 = (1 << Ty.exp_bits) - 1
    mant_shift: int32 = 7 - Ty.mantissa_bits

    out_exp: int32 = 0
    out_mant: int32 = 0
    if exp_field != 0 and local_exp >= 1:
        out_exp = local_exp
        round_bit: int32 = (int(v_bf16[0:7]) >> (mant_shift - 1)) & 1
        out_mant = (int(v_bf16[0:7]) >> mant_shift) + round_bit
        if out_mant == (1 << Ty.mantissa_bits):
            out_mant = 0
            out_exp = out_exp + 1
        if out_exp > max_exp_code:
            out_exp = max_exp_code
            out_mant = (1 << Ty.mantissa_bits) - 1
    result_bits: int32 = (
        (int(sign) << (Ty.exp_bits + Ty.mantissa_bits))
        | (out_exp << Ty.mantissa_bits)
        | out_mant
    )
    result: UInt(Ty.elem_bits) = result_bits
    return result


def schedule_mx_quantize_block_fp_bf16(s):
    s.unroll("mx_quantize_block_fp_bf16:i1")


def mx_quantize_block_fp_bf16[Ty, K](x: "uint16[K]") -> "Ty":

    exp_fields: uint8[K]
    with allo.meta_for(K) as i0:
        exp_fields[i0] = x[i0][7:15]

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
        elem: UInt(Ty.elem_bits) = _mx_quantize_elem_fp_bf16[Ty](x[i1], max_exp_field_u8)
        word[i1 * Ty.elem_bits : (i1 + 1) * Ty.elem_bits] = elem

    return word


def _mx_fp_mul_bf16[Ty](
    a_i: "UInt(Ty.elem_bits)", b_i: "UInt(Ty.elem_bits)"
) -> "Int(Ty.final_accum_bits)":
    a_exp_field: uint8 = a_i[Ty.mantissa_bits : Ty.elem_bits - 1]
    b_exp_field: uint8 = b_i[Ty.mantissa_bits : Ty.elem_bits - 1]
    a_sign: UInt(1) = a_i[Ty.elem_bits - 1 : Ty.elem_bits]
    b_sign: UInt(1) = b_i[Ty.elem_bits - 1 : Ty.elem_bits]

    # normal element: implicit leading 1, exponent = field - bias.
    # subnormal (exp field == 0): no leading 1, exponent pinned to 1 - bias.
    a_exp: int32 = 1 - Ty.bias
    a_mant: int32 = int(a_i[0 : Ty.mantissa_bits])
    if a_exp_field != 0:
        a_exp = int(a_i[Ty.mantissa_bits : Ty.elem_bits - 1]) - Ty.bias
        a_mant = (1 << Ty.mantissa_bits) | int(a_i[0 : Ty.mantissa_bits])

    b_exp: int32 = 1 - Ty.bias
    b_mant: int32 = int(b_i[0 : Ty.mantissa_bits])
    if b_exp_field != 0:
        b_exp = int(b_i[Ty.mantissa_bits : Ty.elem_bits - 1]) - Ty.bias
        b_mant = (1 << Ty.mantissa_bits) | int(b_i[0 : Ty.mantissa_bits])

    prod_mant: int32 = a_mant * b_mant
    # shift_amt is always >= 0: a_exp/b_exp bottom out at 1-Ty.bias
    # (subnormal*subnormal), which is exactly the accumulator's reference
    # point (shift_amt == 0 there).
    shift_amt: int32 = (a_exp + b_exp) - 2 * (1 - Ty.bias)

    magnitude: "Int(Ty.final_accum_bits)" = prod_mant
    magnitude = magnitude << shift_amt

    term: "Int(Ty.final_accum_bits)" = magnitude
    if (a_sign ^ b_sign) == 1:
        term = -magnitude
    return term


def mx_block_dot_fp[Ty, K](data_a: "Ty", data_b: "Ty") -> float32:
    scale_a: int32 = int(_mx_get_scale[Ty](data_a))
    scale_b: int32 = int(_mx_get_scale[Ty](data_b))

    mul_i: "Int(Ty.final_accum_bits)[K]"
    for j0 in range(K):
        a_i: UInt(Ty.elem_bits) = data_a[j0 * Ty.elem_bits : (j0 + 1) * Ty.elem_bits]
        b_i: UInt(Ty.elem_bits) = data_b[j0 * Ty.elem_bits : (j0 + 1) * Ty.elem_bits]
        mul_i[j0] = _mx_fp_mul_bf16[Ty](a_i, b_i)

    acc_i: "Int(Ty.final_accum_bits)" = 0
    for j1 in range(K):
        acc_i = acc_i + mul_i[j1]
    total: float32 = float(acc_i)

    total_bits: int32 = total.bitcast()
    sign: UInt(1) = total_bits[31:32]
    exp_t: int32 = int(total_bits[23:31])
    mant_t: UInt(23) = total_bits[0:23]

    # mul_i's fixed radix point sits at 2**(2*(1-Ty.bias) - 2*Ty.mantissa_bits)
    # (see _mx_fp_mul_bf16) -- a compile-time constant, folded in here
    # alongside the same -254 (= -127 - 127) double-bias correction
    # mx_block_dot (mxint8.py) uses for its own two scale fields.
    new_exp: int32 = (
        exp_t + scale_a + scale_b - 254 + (2 * (1 - Ty.bias) - 2 * Ty.mantissa_bits)
    )

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


def schedule_mx_block_dot_fp(s):
    s.unroll("mx_block_dot_fp:j0")
    s.unroll("mx_block_dot_fp:j1")


def make_mxfp8_dot_general_dataflow_bf16(Ty, K, NB, P, depth=4):
    # Same 512b-word, no-half-split layout as mxint8_bf16.py's
    # make_mx_dot_general_dataflow_bf16: OCP MXFP8 elements are 8 bits (both
    # e4m3 and e5m2: 1 + exp_bits + mantissa_bits == 8), so a K=32 block is
    # byte-identical in packed size to an mxint8 block.
    WORD_BITS = 512
    HALF = WORD_BITS // 16  # bf16 elements per 512b word

    @df.region()
    def top(
        A0: "UInt(WORD_BITS)[NB]",
        B0: "UInt(WORD_BITS)[NB]",
        out: "uint16[1]",
    ):
        pipe_a_raw: Stream[uint16[K], depth]
        pipe_b_raw: Stream[uint16[K], depth]
        pipe_a_q: Stream["uint8[K + 1]", depth]
        pipe_b_q: Stream["uint8[K + 1]", depth]

        @df.kernel(mapping=[1], args=[A0])
        def read_a(local_A0: "UInt(WORD_BITS)[NB]"):
            for b in range(NB):
                blk: uint16[K]
                word: UInt(WORD_BITS) = local_A0[b]
                for j in range(HALF):
                    blk[j] = word[j * 16 : (j + 1) * 16]
                pipe_a_raw.put(blk)

        @df.kernel(mapping=[1], args=[B0])
        def read_b(local_B0: "UInt(WORD_BITS)[NB]"):
            for b in range(NB):
                blk: uint16[K]
                word: UInt(WORD_BITS) = local_B0[b]
                for j in range(HALF):
                    blk[j] = word[j * 16 : (j + 1) * 16]
                pipe_b_raw.put(blk)

        @df.kernel(mapping=[1])
        def quantize_ab():
            for b in range(NB):
                blk_a: uint16[K] = pipe_a_raw.get()
                word_a: Ty = mx_quantize_block_fp_bf16[Ty, K](blk_a)
                bundle_a: uint8[K + 1]
                bundle_a[0] = word_a[Ty.bits - 8 : Ty.bits]
                for k in range(K):
                    bundle_a[k + 1] = word_a[k * Ty.elem_bits : (k + 1) * Ty.elem_bits]
                pipe_a_q.put(bundle_a)

                blk_b: uint16[K] = pipe_b_raw.get()
                word_b: Ty = mx_quantize_block_fp_bf16[Ty, K](blk_b)
                bundle_b: uint8[K + 1]
                bundle_b[0] = word_b[Ty.bits - 8 : Ty.bits]
                for k in range(K):
                    bundle_b[k + 1] = word_b[k * Ty.elem_bits : (k + 1) * Ty.elem_bits]
                pipe_b_q.put(bundle_b)

        @df.kernel(mapping=[1], args=[out])
        def dot_product_stage(local_out: "uint16[1]"):
            partials: float32[P]
            for p0 in range(P):
                partials[p0] = 0.0

            for i in range(NB // P):
            #pragma pipeline II=P(4)
                for j in range(P):
                #pragma unroll
                    bundle_a: uint8[K + 1] = pipe_a_q.get()
                    bundle_b: uint8[K + 1] = pipe_b_q.get()
                    scale_a: uint8 = bundle_a[0]
                    scale_b: uint8 = bundle_b[0]
                    block_a: uint8[K]
                    block_b: uint8[K]
                    for k in range(K):
                        block_a[k] = bundle_a[k + 1]
                        block_b[k] = bundle_b[k + 1]
                    word_a: Ty = _mx_pack_word[Ty, K](scale_a, block_a)
                    word_b: Ty = _mx_pack_word[Ty, K](scale_b, block_b)
                    partials[j] = partials[j] + mx_block_dot_fp[Ty, K](word_a, word_b)

            total: float32 = 0.0
            for r in range(P):
                total = total + partials[r]

            # Narrow the f32 accumulator to bf16 (top 16 bits, round-to-
            # nearest-even) -- bf16 shares f32's sign/exponent layout, so
            # this is a pure bit truncation, no float hardware needed.
            total_bits: int32 = total.bitcast()
            total_bits_u: UInt(32) = total_bits
            lsb: UInt(1) = total_bits_u[16:17]
            rounding_bias: UInt(32) = 0x7FFF + lsb
            rounded: UInt(32) = total_bits_u + rounding_bias
            local_out[0] = rounded[16:32]

    s = df.customize(top, opt_default=False)
    schedule_mx_block_dot_fp(s)
    schedule_mx_quantize_block_fp_bf16(s)
    s.pipeline("read_a_0:j")
    s.pipeline("read_b_0:j")
    s.pipeline("read_a_0:b")
    s.pipeline("read_b_0:b")
    s.pipeline("quantize_ab_0:b")
    s.unroll("quantize_ab_0:k")
    s.pipeline("dot_product_stage_0:i", initiation_interval=P)
    s.unroll("dot_product_stage_0:j")
    s.unroll("dot_product_stage_0:k")
    s.unroll("dot_product_stage_0:p0")
    s.unroll("dot_product_stage_0:r")
    s.partition("dot_product_stage_0:partials", dim=0)
    return s
