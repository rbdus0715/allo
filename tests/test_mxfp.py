# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile

import numpy as np
import ml_dtypes
import pytest
import allo
import allo.backend.hls as hls
from allo.library.mxfp import (
    mx_quantize_block,
    mx_quantize,
    mx_block_dot,
    mx_dot_general,
    dot_product,
    make_mx_dot_general_dataflow,
    patch_extern_c_for_class_return_types,
    _mx_get_scale,
)
from allo.ir.types import float32, uint8, bfloat16
import allo.ir.types as T

K = 8

# HLS csynth builds write substantial scratch output; keep it inside the repo
# (gitignored) instead of the system /tmp.
_REPO_TMP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_tmp")
os.makedirs(_REPO_TMP_DIR, exist_ok=True)


def make_quantize_block_kernel(Ty):
    def kernel(x: bfloat16[K]) -> ("uint8[1]", "uint8[K]"):
        scale_out: uint8[1]
        w: Ty = mx_quantize_block[Ty, K](x)
        scale_out[0] = _mx_get_scale[Ty](w)
        elem_bits: uint8[K]
        for i in range(K):
            elem_bits[i] = w[i * Ty.elem_bits : (i + 1) * Ty.elem_bits]
        return scale_out, elem_bits

    kernel.__name__ = f"quantize_block_{Ty.name}"
    return kernel


def _mxint8_quantize_ref(x, elem_max_unbiased):
    amax = np.max(np.abs(x)) if len(x) else 0.0
    shared_exp = -127.0 if amax == 0 else np.floor(np.log2(amax)) - elem_max_unbiased
    scale_val = 2.0**shared_exp
    signed = np.zeros(len(x), dtype=np.int64)
    for i in range(len(x)):
        scaled = x[i] / scale_val
        s = int(scaled + 0.5) if scaled >= 0 else -int(-scaled + 0.5)
        signed[i] = max(-128, min(127, s))
    return int(shared_exp) + 127, scale_val, signed


def test_mx_quantize_block_mxint8():
    Ty = T.mxint8
    rng = np.random.default_rng(1)
    mod = allo.customize(make_quantize_block_kernel(Ty)).build()
    elem_max_unbiased = Ty.max_unbiased_exp  # elem_bits - 2 == 6

    for trial in range(20):
        x_f32 = (rng.standard_normal(K) * 2.0 ** rng.integers(-8, 8)).astype(np.float32)
        if trial == 0:
            x_f32[:] = 0.0
        elif trial == 1:
            x_f32[0] = 0.0
        x_bf16 = x_f32.astype(ml_dtypes.bfloat16)
        x = x_bf16.astype(np.float32)

        scale, elem_bits = mod(x_bf16)
        scale = int(np.asarray(scale).flatten()[0])
        elem_bits = np.asarray(elem_bits).flatten()

        ref_scale, _, ref_signed = _mxint8_quantize_ref(x, elem_max_unbiased)
        assert scale == ref_scale, f"trial {trial}: scale {scale} != {ref_scale}"

        for i in range(K):
            ours = int(elem_bits[i])
            ours_signed = ours - 256 if ours >= 128 else ours
            assert ours_signed == ref_signed[i], (
                f"trial {trial} elem {i}: {ours_signed} != {ref_signed[i]} (input={x[i]})"
            )


def test_mx_quantize_full_tensor():
    Ty = T.mxint8
    N = 16

    def kernel2(x: bfloat16[N], scales: uint8[N // K], data_bits: uint8[N // K, K]):
        mx_quantize[Ty, K, N](x, scales, data_bits)

    mod = allo.customize(kernel2).build()
    rng = np.random.default_rng(2)
    x_bf16 = ((rng.standard_normal(N) * 4.0).astype(np.float32)).astype(ml_dtypes.bfloat16)
    x = x_bf16.astype(np.float32)
    scales = np.zeros(N // K, dtype=np.uint8)
    data_bits = np.zeros((N // K, K), dtype=np.uint8)
    mod(x_bf16, scales, data_bits)

    elem_max_unbiased = Ty.max_unbiased_exp
    for b in range(N // K):
        ref_scale, _, ref_signed = _mxint8_quantize_ref(
            x[b * K : (b + 1) * K], elem_max_unbiased
        )
        assert int(scales[b]) == ref_scale
        for i in range(K):
            ours = int(data_bits[b, i])
            ours_signed = ours - 256 if ours >= 128 else ours
            assert ours_signed == ref_signed[i]


######################################################################
# mx_block_dot (multiply-accumulate in a fixed-width integer accumulator,
# convert to float32 only once at the end)
######################################################################

DOT_K = 16


def make_block_dot_kernel(Ty, K):
    def kernel(
        a: bfloat16[K], b: bfloat16[K]
    ) -> ("uint8[1]", "uint8[1]", "uint8[K]", "uint8[K]", "float32[1]"):
        sa: uint8[1]
        sb: uint8[1]
        a_bytes: uint8[K]
        b_bytes: uint8[K]
        data_a: Ty = mx_quantize_block[Ty, K](a)
        data_b: Ty = mx_quantize_block[Ty, K](b)
        sa[0] = _mx_get_scale[Ty](data_a)
        sb[0] = _mx_get_scale[Ty](data_b)
        for i in range(K):
            a_bytes[i] = data_a[i * Ty.elem_bits : (i + 1) * Ty.elem_bits]
            b_bytes[i] = data_b[i * Ty.elem_bits : (i + 1) * Ty.elem_bits]

        result: float32[1]
        result[0] = mx_block_dot[Ty, K](data_a, data_b)
        return sa, sb, a_bytes, b_bytes, result

    kernel.__name__ = f"block_dot_{Ty.name}"
    return kernel


def test_mx_block_dot_mxint8():
    Ty = T.mxint8
    mod = allo.customize(make_block_dot_kernel(Ty, DOT_K)).build()
    rng = np.random.default_rng(4)
    for trial in range(10):
        # small integers are exactly representable in bf16 (7 mantissa bits
        # covers magnitudes well past 100), so this bf16-truncation is lossless.
        a = rng.integers(-100, 100, DOT_K).astype(np.float32)
        b = rng.integers(-100, 100, DOT_K).astype(np.float32)
        _, _, _, _, result = mod(a.astype(ml_dtypes.bfloat16), b.astype(ml_dtypes.bfloat16))
        our_dot = float(np.asarray(result).flatten()[0])
        ref_dot = float(np.dot(a.astype(np.float64), b.astype(np.float64)))
        assert our_dot == pytest.approx(ref_dot, rel=1e-3), (
            f"trial {trial}: our={our_dot} ref={ref_dot}"
        )


######################################################################
# mx_dot_general (cross-block reduction)
######################################################################


def make_dot_general_kernel(Ty, K, NB):
    N = K * NB

    def kernel(
        a: bfloat16[N],
        b: bfloat16[N],
        scales_a: uint8[NB],
        scales_b: uint8[NB],
        a_bytes: uint8[NB, K],
        b_bytes: uint8[NB, K],
        result: float32[1],
    ):
        mx_quantize[Ty, K, N](a, scales_a, a_bytes)
        mx_quantize[Ty, K, N](b, scales_b, b_bytes)
        result[0] = mx_dot_general[Ty, K, N, NB](scales_a, a_bytes, scales_b, b_bytes)

    kernel.__name__ = f"dot_general_{Ty.name}"
    return kernel


def test_mx_dot_general_mxint8():
    Ty = T.mxint8
    K, NB = 8, 4
    N = K * NB
    rng = np.random.default_rng(5)
    mod = allo.customize(make_dot_general_kernel(Ty, K, NB)).build()
    elem_max_unbiased = Ty.max_unbiased_exp

    for trial in range(5):
        a = np.concatenate(
            [
                (rng.standard_normal(K) * 2.0 ** rng.integers(-10, 10)).astype(np.float32)
                for _ in range(NB)
            ]
        ).astype(ml_dtypes.bfloat16)
        b = np.concatenate(
            [
                (rng.standard_normal(K) * 2.0 ** rng.integers(-10, 10)).astype(np.float32)
                for _ in range(NB)
            ]
        ).astype(ml_dtypes.bfloat16)
        scales_a = np.zeros(NB, dtype=np.uint8)
        scales_b = np.zeros(NB, dtype=np.uint8)
        a_bytes = np.zeros((NB, K), dtype=np.uint8)
        b_bytes = np.zeros((NB, K), dtype=np.uint8)
        result = np.zeros(1, dtype=np.float32)
        mod(a, b, scales_a, scales_b, a_bytes, b_bytes, result)
        our_dot = float(result[0])

        a_f32 = a.astype(np.float32)
        b_f32 = b.astype(np.float32)
        ref_dot = 0.0
        for blk in range(NB):
            _, a_scale_val, a_signed = _mxint8_quantize_ref(
                a_f32[blk * K : (blk + 1) * K], elem_max_unbiased
            )
            _, b_scale_val, b_signed = _mxint8_quantize_ref(
                b_f32[blk * K : (blk + 1) * K], elem_max_unbiased
            )
            aq = a_signed * a_scale_val
            bq = b_signed * b_scale_val
            ref_dot += float(np.sum(aq * bq))

        mag = max(abs(our_dot), abs(ref_dot), 1.0)
        tol = mag * 2.0**-10
        assert our_dot == pytest.approx(ref_dot, rel=1e-3, abs=tol), (
            f"trial {trial}: our={our_dot} ref={ref_dot}"
        )


######################################################################
# dot_product (top-level kernel: bfloat16[N], bfloat16[N] -> float32)
######################################################################

DOT_PRODUCT_K = 32 
DOT_PRODUCT_NB = 8 

def _ref_dot_product_mxint8(a, b, K, Ty):
    NB = len(a) // K
    elem_max_unbiased = Ty.max_unbiased_exp
    total = 0.0
    for blk in range(NB):
        _, a_scale_val, a_signed = _mxint8_quantize_ref(
            a[blk * K : (blk + 1) * K], elem_max_unbiased
        )
        _, b_scale_val, b_signed = _mxint8_quantize_ref(
            b[blk * K : (blk + 1) * K], elem_max_unbiased
        )
        aq = a_signed * a_scale_val
        bq = b_signed * b_scale_val
        total += float(np.sum(aq * bq))
    return total


def test_dot_product_mxint8():
    Ty = T.mxint8
    K, NB = DOT_PRODUCT_K, DOT_PRODUCT_NB
    N = K * NB
    rng = np.random.default_rng(11)

    def kernel(a: bfloat16[N], b: bfloat16[N]) -> float32:
        return dot_product[Ty, K, N](a, b)

    mod = allo.customize(kernel).build()
    for trial in range(5):
        a_bf16 = (
            (rng.standard_normal(N) * 2.0 ** rng.integers(-8, 8))
            .astype(np.float32)
            .astype(ml_dtypes.bfloat16)
        )
        b_bf16 = (
            (rng.standard_normal(N) * 2.0 ** rng.integers(-8, 8))
            .astype(np.float32)
            .astype(ml_dtypes.bfloat16)
        )
        a = a_bf16.astype(np.float32)
        b = b_bf16.astype(np.float32)
        our_dot = float(mod(a_bf16, b_bf16))
        ref_dot = _ref_dot_product_mxint8(a, b, K, Ty)
        mag = max(abs(our_dot), abs(ref_dot), 1.0)
        ulp_guess = mag * 2.0**-8  
        assert our_dot == pytest.approx(ref_dot, rel=1e-3, abs=NB * ulp_guess), (
            f"trial {trial}: our={our_dot} ref={ref_dot}"
        )


def test_mx_dot_general_dataflow_mxint8():
    Ty = T.mxint8
    K = 32  # mxint8's actual block_size
    NB = 16
    P = 4
    s = make_mx_dot_general_dataflow(Ty, K, NB, P)

    if not hls.is_available("vitis_hls"):
        return

    # Which HLS mode to build/run. Override at invocation time, e.g.:
    #   ALLO_HLS_MODE=hw_emu python tests/test_mxfp.py
    #   ALLO_HLS_MODE=hw_emu pytest tests/test_mxfp.py -k dataflow_mxint8
    # ...or just edit the default here.
    mode = os.environ.get("ALLO_HLS_MODE", "csyn")  # csyn | sw_emu | hw_emu | hw
    # mode = "hw_emu"
    assert mode in {"csyn", "sw_emu", "hw_emu", "hw"}, f"unsupported mode {mode!r}"

    # sw_emu/hw_emu/hw go through the v++ Makefile flow, which needs XDEVICE
    # pointing at a platform .xpfm (same env var Allo already reads).
    if mode != "csyn" and "XDEVICE" not in os.environ:
        print(f"Skipping {mode} run: set XDEVICE to a platform .xpfm to run this mode")
        return

    with tempfile.TemporaryDirectory(dir=_REPO_TMP_DIR) as tmpdir:
        hls_mod = s.build(
            target="vitis_hls",
            mode=mode,
            project=tmpdir,
            wrap_io=True,
        )
        # issue #603 (https://github.com/alloy-lang/allo/issues/603)
        patched = patch_extern_c_for_class_return_types(f"{tmpdir}/kernel.cpp")
        assert patched, "expected _mx_pack_word/mx_quantize_block_f32 to need the patch"

        if mode == "csyn":
            hls_mod()
            csynth_rpt = os.path.join(
                tmpdir, "out.prj", "solution1", "syn", "report", "top_csynth.rpt"
            )
            assert os.path.isfile(csynth_rpt)
            with open(csynth_rpt, encoding="utf-8") as f:
                report = f.read()
            assert "dataflow" in report
        else:
            # sw_emu/hw_emu/hw can't be called with zero args like csyn -- pack
            # real float32 input blocks into the wide UInt(K*32)[NB] words.
            N = K * NB
            rng = np.random.default_rng(0)
            A = (rng.standard_normal(N) * 2.0 ** rng.integers(-4, 4, N)).astype(np.float32)
            B = (rng.standard_normal(N) * 2.0 ** rng.integers(-4, 4, N)).astype(np.float32)
            pack = lambda x: (
                np.frombuffer(x.tobytes(), dtype=np.uint8)
                .reshape(NB, K * 4)
                .copy()
                .view(f"V{K * 4}")
                .reshape(NB)
            )
            result = np.zeros((1,), dtype=np.float32)
            hls_mod(pack(A), pack(B), result)
            ref = float(np.dot(A.astype(np.float64), B.astype(np.float64)))
            print(f"[{mode}] result={result[0]} ref={ref}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--csyn", action="store_const", dest="mode", const="csyn")
    mode_group.add_argument(
        "--sw_emu", action="store_const", dest="mode", const="sw_emu"
    )
    mode_group.add_argument(
        "--hw_emu", action="store_const", dest="mode", const="hw_emu"
    )
    mode_group.add_argument("--hw", action="store_const", dest="mode", const="hw")
    parser.add_argument(
        "--xdevice", default=None, help="platform .xpfm path (sets XDEVICE)"
    )
    args, pytest_args = parser.parse_known_args()
    if args.mode:
        os.environ["ALLO_HLS_MODE"] = args.mode
    if args.xdevice:
        os.environ["XDEVICE"] = args.xdevice
    pytest.main([__file__, *pytest_args])
