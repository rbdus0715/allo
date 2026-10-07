# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=used-before-assignment, unsubscriptable-object

import re

import allo
from ..ir.types import Int, UInt, int32, uint8, float32, Stream, ConstExpr


def _mx_quantize_elem_int_f32[
    Ty
](v_f32: float32, shared_field: uint8) -> "UInt(Ty.elem_bits)":
    bits: int32 = v_f32.bitcast()
    sign: UInt(1) = bits[31:32]
    exp_field: Int(9) = bits[23:31]
    full_mant: Int(25) = (1 << 23) | bits[0:23]
    total_shift: Int(10) = (23 - Ty.max_unbiased_exp) + shared_field - exp_field

    mag: Int(Ty.elem_bits) = 0
    if total_shift < 24:
        mag = full_mant >> total_shift

    rounded: Int(Ty.elem_bits) = mag
    if sign == 1:
        rounded = -mag
    if rounded > Ty.max_int:
        rounded = Ty.max_int
    if rounded < -Ty.max_int:
        rounded = -Ty.max_int
    return rounded


def _mx_quantize_elem_fp_f32[
    Ty
](v_f32: float32, shared_field: uint8) -> "UInt(Ty.elem_bits)":
    # OCP MX: v / 2^(shared exponent - emax), round-to-nearest-even, saturating
    bits: int32 = v_f32.bitcast()
    sign: int32 = int(bits[31:32])
    exp_field: int32 = int(bits[23:31])
    full_mant: int32 = (1 << 23) | int(bits[0:23])
    code: int32 = exp_field - int(shared_field) + Ty.max_unbiased_exp + Ty.bias
    shift: int32 = 23 - Ty.mantissa_bits
    base: int32 = 0
    if code >= 1:  # normal: q in [2^m, 2^(m+1)] sits on exponent code - 1
        base = (code - 1) << Ty.mantissa_bits
    else:  # subnormal: q in [0, 2^m]; q == 2^m carries into exponent code 1
        shift = shift + 1 - code

    mag: int32 = 0
    if exp_field != 0 and shift <= 25:  # larger shifts round to zero
        q: int32 = full_mant >> shift
        rem: int32 = full_mant & ((1 << shift) - 1)
        half: int32 = 1 << (shift - 1)
        round_up: int32 = 0
        if rem > half:
            round_up = 1
        if rem == half:
            round_up = q & 1
        mag = base + q + round_up
    if mag > Ty.max_code:
        mag = Ty.max_code
    result: UInt(Ty.elem_bits) = (sign << (Ty.elem_bits - 1)) | mag
    return result


def schedule_mx_quantize_block_f32(s):
    s.unroll("mx_quantize_block_f32:i1")


def mx_quantize_block_f32[Ty, K](x: "float32[K]") -> "Ty":

    exp_fields: uint8[K]
    with allo.meta_for(K) as i0:
        bits_i: int32 = x[i0].bitcast()
        exp_fields[i0] = bits_i[23:31]

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
        elem: UInt(Ty.elem_bits) = 0
        with allo.meta_if(Ty.is_float):
            elem = _mx_quantize_elem_fp_f32[Ty](x[i1], max_exp_field_u8)
        with allo.meta_else():
            elem = _mx_quantize_elem_int_f32[Ty](x[i1], max_exp_field_u8)
        word[i1 * Ty.elem_bits : (i1 + 1) * Ty.elem_bits] = elem

    return word


def mx_pack_words(arr, bits):
    """Packs an array's raw bytes into a 1D array of bits-wide words."""
    import numpy as np  # pylint: disable=import-outside-toplevel
    from ..utils import get_np_struct_type  # pylint: disable=import-outside-toplevel

    raw = np.ascontiguousarray(arr).view(np.uint8).reshape(-1)
    assert (
        raw.size % (bits // 8) == 0
    ), f"{raw.size} bytes is not a whole number of {bits}-bit words"
    return raw.view(get_np_struct_type(bits))


def _mx_quantize_np(Ty, x):
    """numpy MX quantization of a [R, K] array, bit-exact with the hardware quantizer."""
    import numpy as np  # pylint: disable=import-outside-toplevel

    x = np.ascontiguousarray(x, dtype=np.float32)
    R, K = x.shape
    BS = Ty.block_size
    assert K % BS == 0, f"K={K} must be a multiple of block_size={BS}"

    bits = x.view(np.uint32).astype(np.int64).reshape(R, K // BS, BS)
    exp_field = (bits >> 23) & 0xFF
    max_exp = exp_field.max(axis=2, keepdims=True)
    full_mant = (bits & 0x7FFFFF) | (1 << 23)
    if Ty.is_float:  # mirrors _mx_quantize_elem_fp_f32
        m, eb = Ty.mantissa_bits, Ty.elem_bits
        code = exp_field - max_exp + Ty.max_unbiased_exp + Ty.bias
        shift = 23 - m + np.maximum(0, 1 - code)
        s = np.minimum(shift, 40)
        q = full_mant >> s
        rem, half = full_mant & ((1 << s) - 1), 1 << (s - 1)
        q = q + ((rem > half) | ((rem == half) & ((q & 1) == 1)))
        mag = (np.maximum(code - 1, 0) << m) + q
        mag = np.where((exp_field != 0) & (shift <= 25), mag, 0)
        mag = np.minimum(mag, Ty.max_code)
        codes = ((bits >> 31) << (eb - 1)) | mag
        elems = codes - ((codes >> (eb - 1)) << eb)  # same bits, as a signed int
    else:  # mirrors _mx_quantize_elem_int_f32
        total_shift = (23 - Ty.max_unbiased_exp) + max_exp - exp_field
        mag = np.where(total_shift < 24, full_mant >> np.minimum(total_shift, 63), 0)
        elems = np.clip(np.where(bits >> 31, -mag, mag), -Ty.max_int, Ty.max_int)
    scales = np.clip(max_exp[..., 0] - Ty.max_unbiased_exp, 0, 254)
    elem_dtype = np.dtype(f"int{max(8, 1 << (Ty.elem_bits - 1).bit_length())}")
    return elems.astype(elem_dtype), scales.astype(np.uint8)


def mx_quantize(Ty, x, axis=-1):
    """Quantizes a 2D array into an MX operand: (elements, e8m0 scales).

    MXINT elements are the integers themselves; MXFP elements are their bit
    patterns, stored as same-width signed integers (e.g. Int(8) for mxfp8).
    """
    import numpy as np  # pylint: disable=import-outside-toplevel

    x = np.asarray(x, dtype=np.float32)
    assert x.ndim == 2, "mx_quantize takes a 2D array"
    if axis in (0, -2):
        elems, scales = mx_quantize(Ty, x.T)
        return np.ascontiguousarray(elems.T), np.ascontiguousarray(scales.T)
    elems, scales = _mx_quantize_np(Ty, x)
    return elems.reshape(x.shape), scales


def mx_pack_elems(Ty, elems):
    """Bit-packs [..., block_size] MX elements into payload_bits-wide words."""
    import numpy as np  # pylint: disable=import-outside-toplevel

    from ..utils import get_np_struct_type  # pylint: disable=import-outside-toplevel

    PB = Ty.payload_bits
    word_bits = max(8, 1 << (PB - 1).bit_length())  # LLVM pads iN to a power of 2
    elems = np.asarray(elems, dtype=np.int64)
    lanes = (elems[..., None] >> np.arange(Ty.elem_bits)) & 1  # two's complement bits
    lanes = lanes.reshape(-1, PB).astype(np.uint8)
    lanes = np.pad(lanes, ((0, 0), (0, word_bits - PB)))
    raw = np.packbits(lanes, axis=-1, bitorder="little")
    if word_bits <= 64:
        return raw.view(np.dtype(f"uint{word_bits}")).reshape(-1)
    return raw.view(get_np_struct_type(word_bits)).reshape(-1)


def mx_quantize_weights(Ty, W, Tn=None):
    """Quantizes a [N, K] weight into tile-ordered MX elements and scales."""
    import numpy as np  # pylint: disable=import-outside-toplevel

    elems, scales = _mx_quantize_np(Ty, W)
    N, NB, BS = elems.shape
    Tn = N if Tn is None else Tn
    assert N % Tn == 0, f"N={N} must be a multiple of Tn={Tn}"
    elems = elems.reshape(N // Tn, Tn, NB, BS)
    scales = scales.reshape(N // Tn, Tn, NB)
    return (
        np.ascontiguousarray(elems.transpose(0, 2, 1, 3)),
        np.ascontiguousarray(scales.transpose(0, 2, 1)),
    )


def _mx_scale_to_float32(scale: uint8) -> float32:
    bits: int32 = int(scale) << 23
    return bits.bitcast()


def _mx_pack_word[Ty, K](scale: uint8, elems: "UInt(Ty.elem_bits)[K]") -> "Ty":
    word: Ty = 0
    word[Ty.bits - 8 : Ty.bits] = scale
    for i in range(K):
        word[i * Ty.elem_bits : (i + 1) * Ty.elem_bits] = elems[i]
    return word


def _mx_get_scale[Ty](word: "Ty") -> uint8:
    return word[Ty.bits - 8 : Ty.bits]


def _mx_acc_bits(Ty, K):
    # FP: fixed-point product width (see _mx_fp_mul); INT: plain product width
    product_bits = Ty.block_accum_bits if Ty.is_float else 2 * Ty.elem_bits
    return product_bits + (K - 1).bit_length()


def _mx_fp_mul[
    Ty, K
](a_i: "UInt(Ty.elem_bits)", b_i: "UInt(Ty.elem_bits)") -> "Int(_mx_acc_bits(Ty, K))":
    """Exact product of two MXFP elements, in units of 2^Ty.dot_exp_offset."""
    a: int32 = int(a_i)
    b: int32 = int(b_i)
    exp_mask: int32 = (1 << Ty.exp_bits) - 1
    mant_mask: int32 = (1 << Ty.mantissa_bits) - 1
    a_exp_field: int32 = (a >> Ty.mantissa_bits) & exp_mask
    b_exp_field: int32 = (b >> Ty.mantissa_bits) & exp_mask

    # normal: implicit leading 1, exponent field - bias
    # subnormal (field 0): no leading 1, exponent pinned to 1 - bias
    a_exp: int32 = 1 - Ty.bias
    a_mant: int32 = a & mant_mask
    if a_exp_field != 0:
        a_exp = a_exp_field - Ty.bias
        a_mant = (1 << Ty.mantissa_bits) | (a & mant_mask)
    b_exp: int32 = 1 - Ty.bias
    b_mant: int32 = b & mant_mask
    if b_exp_field != 0:
        b_exp = b_exp_field - Ty.bias
        b_mant = (1 << Ty.mantissa_bits) | (b & mant_mask)

    # >= 0: both exponents bottom out at 1 - bias (the accumulator's radix point)
    shift_amt: int32 = (a_exp + b_exp) - 2 * (1 - Ty.bias)
    magnitude: "Int(_mx_acc_bits(Ty, K))" = a_mant * b_mant
    magnitude = magnitude << shift_amt
    term: "Int(_mx_acc_bits(Ty, K))" = magnitude
    if ((a ^ b) >> (Ty.elem_bits - 1)) & 1 == 1:
        term = -magnitude
    return term


def mx_block_dot[Ty, K](data_a: "Ty", data_b: "Ty") -> float32:
    scale_a: int32 = int(_mx_get_scale[Ty](data_a))
    scale_b: int32 = int(_mx_get_scale[Ty](data_b))

    mul_i: "Int(_mx_acc_bits(Ty, K))[K]"
    for j0 in range(K):
        with allo.meta_if(Ty.is_float):
            a_f: UInt(Ty.elem_bits) = data_a[
                j0 * Ty.elem_bits : (j0 + 1) * Ty.elem_bits
            ]
            b_f: UInt(Ty.elem_bits) = data_b[
                j0 * Ty.elem_bits : (j0 + 1) * Ty.elem_bits
            ]
            mul_i[j0] = _mx_fp_mul[Ty, K](a_f, b_f)
        with allo.meta_else():
            a_i: Int(Ty.elem_bits) = data_a[j0 * Ty.elem_bits : (j0 + 1) * Ty.elem_bits]
            b_i: Int(Ty.elem_bits) = data_b[j0 * Ty.elem_bits : (j0 + 1) * Ty.elem_bits]
            mul_i[j0] = a_i * b_i

    acc_i: "Int(_mx_acc_bits(Ty, K))" = 0
    for j1 in range(K):
        acc_i = acc_i + mul_i[j1]
    total: float32 = float(acc_i)

    total_bits: int32 = total.bitcast()
    sign: UInt(1) = total_bits[31:32]
    exp_t: int32 = int(total_bits[23:31])
    mant_t: Int(24) = total_bits[0:23]

    # -254: the two E8M0 biases; dot_exp_offset: the products' radix point (FP)
    new_exp: int32 = exp_t + scale_a + scale_b - 254 + Ty.dot_exp_offset

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


def schedule_mx_block_dot(s):
    s.unroll("mx_block_dot:j0")
    s.unroll("mx_block_dot:j1")


def patch_extern_c_for_class_return_types(kernel_cpp_path):
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
    import allo.dataflow as df

    N = K * NB
    WORD_BITS = 512
    HALF = WORD_BITS // 32  # float32 elements per 512b half
    assert K == 2 * HALF, f"block_size={K}: A0/A1 hold exactly {HALF} floats each"
    LB = max(Ty.elem_bits, Ty.scale_bits)  # bundle lane: one element or the scale

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
        pipe_a_q: Stream["UInt(LB)[K + 1]", depth]
        pipe_b_q: Stream["UInt(LB)[K + 1]", depth]

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
                bundle_a: UInt(LB)[K + 1]
                bundle_a[0] = word_a[Ty.bits - 8 : Ty.bits]
                for k in range(K):
                    bundle_a[k + 1] = word_a[k * Ty.elem_bits : (k + 1) * Ty.elem_bits]
                pipe_a_q.put(bundle_a)

                blk_b: float32[K] = pipe_b_raw.get()
                word_b: Ty = mx_quantize_block_f32[Ty, K](blk_b)
                bundle_b: UInt(LB)[K + 1]
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
                    bundle_a: UInt(LB)[K + 1] = pipe_a_q.get()
                    bundle_b: UInt(LB)[K + 1] = pipe_b_q.get()
                    scale_a: uint8 = bundle_a[0]
                    scale_b: uint8 = bundle_b[0]
                    block_a: UInt(Ty.elem_bits)[K]
                    block_b: UInt(Ty.elem_bits)[K]
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
