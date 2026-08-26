# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=used-before-assignment, unsubscriptable-object

import re

import allo
import allo.dataflow as df
from ..ir.types import Int, UInt, int32, uint8, uint16, float32, bfloat16, Stream
from .._mlir.exceptions import AlloValueError


def _bf16_to_f32(v: bfloat16) -> float32:
    bits16: uint16 = v.bitcast()
    bits32: int32 = int(bits16) << 16
    return bits32.bitcast()


def _mx_quantize_elem_fp[Ty](v: bfloat16, shared_exp: int32) -> "UInt(Ty.elem_bits)":
    bits: uint16 = v.bitcast()
    sign: int32 = bits[15:16]
    exp_field: int32 = bits[7:15]
    mant16: int32 = bits[0:7]

    max_exp_code: int32 = (1 << Ty.exp_bits) - 1
    max_mant_code: int32 = (1 << Ty.mantissa_bits) - 1

    out: UInt(Ty.elem_bits) = 0

    if exp_field == 0:
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
    return _mx_quantize_elem_int_f32[Ty](v_f32, shared_exp)


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
    s.pipeline("mx_quantize_block:i0")
    s.pipeline("mx_quantize_block:i1")


def _mx_quantize_elem_int_f32[Ty](v_f32: float32, shared_exp: int32) -> "UInt(Ty.elem_bits)":
    bits: int32 = v_f32.bitcast()
    sign: int32 = bits[31:32]
    exp_field: int32 = bits[23:31]
    mant: int32 = bits[0:23]
    max_val: int32 = (1 << (Ty.elem_bits - 1)) - 1
    min_val: int32 = -(1 << (Ty.elem_bits - 1))

    rounded: int32 = 0
    if exp_field == 0:
        rounded = 0
    else:
        unbiased_exp: int32 = exp_field - 127
        target_exp: int32 = unbiased_exp - shared_exp
        full_mant: int32 = (1 << 23) | mant
        total_shift: int32 = 23 - target_exp

        mag: int32 = 0
        if total_shift <= 0:
            mag = max_val + 1
        elif total_shift > 30:
            mag = 0
        else:
            shifted: int32 = full_mant >> (total_shift - 1)
            kept: int32 = shifted >> 1
            round_up: int32 = shifted & 1

            if round_up == 1:
                kept = kept + 1
            mag = kept

        if sign == 1:
            rounded = -mag
        else:
            rounded = mag
        rounded = min(rounded, max_val)
        rounded = max(rounded, min_val)
    return rounded


def schedule_mx_quantize_block_f32(s):
    s.unroll("mx_quantize_block_f32:i0")
    s.unroll("mx_quantize_block_f32:i1")


def mx_quantize_block_f32[Ty, K](x: "float32[K]") -> "Ty":
    max_exp_field: int32 = 0
    for i0 in range(K):
        bits_i: int32 = x[i0].bitcast()
        exp_field_i: int32 = bits_i[23:31]
        if exp_field_i > max_exp_field:
            max_exp_field = exp_field_i

    shared_exp: int32 = max_exp_field - 127 - Ty.max_unbiased_exp
    shared_exp = min(shared_exp, 127)
    shared_exp = max(shared_exp, -127)

    scale_field: uint8 = shared_exp + 127

    word: Ty = 0
    word[Ty.bits - 8 : Ty.bits] = scale_field

    for i1 in range(K):
        elem: UInt(Ty.elem_bits) = _mx_quantize_elem_int_f32[Ty](x[i1], shared_exp)
        word[i1 * Ty.elem_bits : (i1 + 1) * Ty.elem_bits] = elem

    return word


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
    s.pipeline("mx_quantize:b")


def _mx_scale_to_float32(scale: uint8) -> float32:
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
