# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=used-before-assignment, unsubscriptable-object

import allo
import allo.dataflow as df
from ..ir.types import Int, UInt, int32, uint8, uint16, float32, Stream, ConstExpr
from .mxint8 import _mx_pack_word, mx_block_dot, schedule_mx_block_dot


def _mx_quantize_elem_int_bf16[Ty](v_bf16: uint16, shared_field: uint8) -> "UInt(Ty.elem_bits)":
    sign: UInt(1) = v_bf16[15:16]
    exp_field: Int(9) = v_bf16[7:15]
    full_mant: Int(9) = (1 << 7) | v_bf16[0:7]
    total_shift: Int(10) = (7 - Ty.max_unbiased_exp) + shared_field - exp_field

    mag: Int(Ty.elem_bits) = 0
    if total_shift < 8:
        mag = full_mant >> total_shift

    rounded: Int(Ty.elem_bits) = mag
    if sign == 1:
        rounded = -mag
    if rounded > 127:
        rounded = 127
    if rounded < -127:
        rounded = -127
    return rounded


def schedule_mx_quantize_block_bf16(s):
    s.unroll("mx_quantize_block_bf16:i1")


def mx_quantize_block_bf16[Ty, K](x: "uint16[K]") -> "Ty":

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
        elem: UInt(Ty.elem_bits) = _mx_quantize_elem_int_bf16[Ty](x[i1], max_exp_field_u8)
        word[i1 * Ty.elem_bits : (i1 + 1) * Ty.elem_bits] = elem

    return word


def make_mx_dot_general_dataflow_bf16(Ty, K, NB, P, depth=4):
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
                word_a: Ty = mx_quantize_block_bf16[Ty, K](blk_a)
                bundle_a: uint8[K + 1]
                bundle_a[0] = word_a[Ty.bits - 8 : Ty.bits]
                for k in range(K):
                    bundle_a[k + 1] = word_a[k * Ty.elem_bits : (k + 1) * Ty.elem_bits]
                pipe_a_q.put(bundle_a)

                blk_b: uint16[K] = pipe_b_raw.get()
                word_b: Ty = mx_quantize_block_bf16[Ty, K](blk_b)
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
                for j in range(P):
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
                    partials[j] = partials[j] + mx_block_dot[Ty, K](word_a, word_b)

            total: float32 = 0.0
            for r in range(P):
                total = total + partials[r]

            total_bits: int32 = total.bitcast()
            total_bits_u: UInt(32) = total_bits
            lsb: UInt(1) = total_bits_u[16:17]
            rounding_bias: UInt(32) = 0x7FFF + lsb
            rounded: UInt(32) = total_bits_u + rounding_bias
            local_out[0] = rounded[16:32]

    s = df.customize(top, opt_default=False)
    schedule_mx_block_dot(s)
    schedule_mx_quantize_block_bf16(s)
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
