# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=used-before-assignment, unsubscriptable-object

import re

import allo.dataflow as df
from ..ir.types import Int, UInt, int32, uint8, float32, Stream


def _mx_quantize_elem_int_f32[Ty](v_f32: float32, shared_field: uint8) -> "UInt(Ty.elem_bits)":
    bits: int32 = v_f32.bitcast()
    sign: UInt(1) = bits[31:32]
    exp_field: uint8 = bits[23:31]
    mant: UInt(24) = bits[0:23]

    deficit: uint8 = shared_field - exp_field
    total_shift: uint8 = (23 - Ty.max_unbiased_exp) + deficit
    full_mant: UInt(24) = (1 << 23) | mant

    mag: Int(Ty.elem_bits) = 0
    if total_shift < 24:
        mag = full_mant >> total_shift

    rounded: Int(Ty.elem_bits) = mag
    if sign == 1:
        rounded = -mag
    if rounded > 127:
        rounded = 127
    if rounded < -127:
        rounded = -127
    return rounded


def schedule_mx_quantize_block_f32(s):
    s.unroll("mx_quantize_block_f32:i0")
    s.unroll("mx_quantize_block_f32:i1")


def mx_quantize_block_f32[Ty, K](x: "float32[K]") -> "Ty":
    max_exp_field: int32 = 0
    for i0 in range(K):
        bits_i: int32 = x[i0].bitcast()
        exp_field_i: uint8 = bits_i[23:31]
        if exp_field_i > max_exp_field:
            max_exp_field = exp_field_i

    scale_wide: int32 = max_exp_field - Ty.max_unbiased_exp
    scale_wide = min(scale_wide, 254)
    scale_wide = max(scale_wide, 0)
    scale_field: uint8 = scale_wide
    max_exp_field_u8: uint8 = max_exp_field

    word: Ty = 0
    word[Ty.bits - 8 : Ty.bits] = scale_field

    for i1 in range(K):
        elem: UInt(Ty.elem_bits) = _mx_quantize_elem_int_f32[Ty](x[i1], max_exp_field_u8)
        word[i1 * Ty.elem_bits : (i1 + 1) * Ty.elem_bits] = elem

    return word


def _mx_scale_to_float32(scale: uint8) -> float32:
    bits: int32 = int(scale) << 23
    return bits.bitcast()


def _mx_pack_word[Ty, K](scale: uint8, elems: "uint8[K]") -> "Ty":
    word: Ty = 0
    word[Ty.bits - 8 : Ty.bits] = scale
    for i in range(K):
        word[i * Ty.elem_bits : (i + 1) * Ty.elem_bits] = elems[i]
    return word


def _mx_get_scale[Ty](word: "Ty") -> uint8:
    return word[Ty.bits - 8 : Ty.bits]


def _mx_acc_bits(Ty, K):
    return 2 * Ty.elem_bits + (K - 1).bit_length()


def mx_block_dot[Ty, K](data_a: "Ty", data_b: "Ty") -> float32:
    scale_a: uint8 = _mx_get_scale[Ty](data_a)
    scale_b: uint8 = _mx_get_scale[Ty](data_b)
    scale_a_val: float32 = _mx_scale_to_float32(scale_a)
    scale_b_val: float32 = _mx_scale_to_float32(scale_b)

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
    total: float32 = float(acc_i)

    return total * scale_a_val * scale_b_val


def schedule_mx_block_dot(s):
    s.unroll("mx_block_dot:j0")
    s.unroll("mx_block_dot:j1")


def patch_extern_c_for_class_return_types(kernel_cpp_path):
    # remove extern "C" block and add it back with the function signature
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


def make_mx_dot_general_dataflow(Ty, K, NB, P, depth=4):
    N = K * NB
    WORD_BITS = 512
    HALF = WORD_BITS // 32  # float32 elements per 512b half

    @df.region()
    def top(
        A0: "UInt(WORD_BITS)[NB]",
        A1: "UInt(WORD_BITS)[NB]",
        B0: "UInt(WORD_BITS)[NB]",
        B1: "UInt(WORD_BITS)[NB]",
        out: "float32[1]",
    ):
        pipe_a_raw: Stream[float32[K], depth]
        pipe_b_raw: Stream[float32[K], depth]
        pipe_a_q: Stream["uint8[K + 1]", depth]
        pipe_b_q: Stream["uint8[K + 1]", depth]

        @df.kernel(mapping=[1], args=[A0, A1])
        def read_a(local_A0: "UInt(WORD_BITS)[NB]", local_A1: "UInt(WORD_BITS)[NB]"):
            for b in range(NB):
                blk: float32[K]
                word_lo: UInt(WORD_BITS) = local_A0[b]
                for j0 in range(HALF):
                    bits: int32 = word_lo[j0 * 32 : (j0 + 1) * 32]
                    blk[j0] = bits.bitcast()
                word_hi: UInt(WORD_BITS) = local_A1[b]
                for j1 in range(HALF):
                    bits: int32 = word_hi[j1 * 32 : (j1 + 1) * 32]
                    blk[HALF + j1] = bits.bitcast()
                pipe_a_raw.put(blk)

        @df.kernel(mapping=[1], args=[B0, B1])
        def read_b(local_B0: "UInt(WORD_BITS)[NB]", local_B1: "UInt(WORD_BITS)[NB]"):
            for b in range(NB):
                blk: float32[K]
                word_lo: UInt(WORD_BITS) = local_B0[b]
                for j0 in range(HALF):
                    bits: int32 = word_lo[j0 * 32 : (j0 + 1) * 32]
                    blk[j0] = bits.bitcast()
                word_hi: UInt(WORD_BITS) = local_B1[b]
                for j1 in range(HALF):
                    bits: int32 = word_hi[j1 * 32 : (j1 + 1) * 32]
                    blk[HALF + j1] = bits.bitcast()
                pipe_b_raw.put(blk)

        @df.kernel(mapping=[1])
        def quantize_ab():
            for b in range(NB):
                blk_a: float32[K] = pipe_a_raw.get()
                word_a: Ty = mx_quantize_block_f32[Ty, K](blk_a)
                bundle_a: uint8[K + 1]
                bundle_a[0] = word_a[Ty.bits - 8 : Ty.bits]
                for k in range(K):
                    bundle_a[k + 1] = word_a[k * Ty.elem_bits : (k + 1) * Ty.elem_bits]
                pipe_a_q.put(bundle_a)

                blk_b: float32[K] = pipe_b_raw.get()
                word_b: Ty = mx_quantize_block_f32[Ty, K](blk_b)
                bundle_b: uint8[K + 1]
                bundle_b[0] = word_b[Ty.bits - 8 : Ty.bits]
                for k in range(K):
                    bundle_b[k + 1] = word_b[k * Ty.elem_bits : (k + 1) * Ty.elem_bits]
                pipe_b_q.put(bundle_b)

        @df.kernel(mapping=[1], args=[out])
        def dot_product_stage(local_out: "float32[1]"):
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
            local_out[0] = total

    s = df.customize(top, opt_default=False)
    schedule_mx_block_dot(s)
    schedule_mx_quantize_block_f32(s)
    s.pipeline("read_a_0:j0")
    s.pipeline("read_b_0:j0")
    s.pipeline("read_a_0:j1")
    s.pipeline("read_b_0:j1")
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
