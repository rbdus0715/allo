# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=unsubscriptable-object, no-value-for-parameter

# nn.mx_conv2d vs the same direct convolution in plain float32: both take one
# block of BS input channels per tap per cycle (cout innermost, pipelined).
# The MX operands come quantized (one word per block); float32 reads a block
# from BS banks.
#   pytest tests/comp_fp_mx_conv.py                                -> numerics
#   python tests/comp_fp_mx_conv.py --format mxint8_k32 -k csyn    -> + csynth

import os
import shutil

import numpy as np
import pytest
import allo
import allo.backend.hls as hls
import allo.ir.types as T
import allo.library.nn as nn
from allo.ir.types import int32, float32, ConstExpr
from allo.ir.utils import MockBuffer
from allo.library.nn import partition_block_lanes
from allo.library.mxint8 import mx_pack_np, patch_extern_c_for_class_return_types

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
BS = 32  # channels per block, as in the 32-element MX formats

# (B, Cin, Cout, H, K, S, P)
CONFIGS = {
    "small": (1, 4, 8, 8, 3, 1, 1),
    "medium": (1, 16, 16, 16, 3, 1, 1),
}

# MX formats to compare against float32; csyn takes one (--format)
FORMATS = {
    t.name: t
    for t in (T.MXInt(8, 32), T.MXInt(4, 16), T.MXFP(4, 3, 32), T.MXFP(2, 1, 32))
}


def fp32_conv2d[
    B, Cin, Cout, H, W, Kh, Kw, Oh, Ow, Sh, Sw, Pd0, Pd1
](
    inp: "float32[B, Cin, H, W]",
    kernel: "float32[Cout, Cin, Kh, Kw]",
    bias: "float32[Cout]",
) -> "float32[B, Cout, Oh, Ow]":
    # nn.mx_conv2d's loops with a float32 block dot of BS channels
    NB: ConstExpr[int32] = (Cin + BS - 1) // BS
    Z: float32[B, Cout, Oh, Ow]
    for n0, c0, oh0, ow0 in allo.grid(B, Cout, Oh, Ow):
        Z[n0, c0, oh0, ow0] = bias[c0]
    for n, cout, oh, ow, kh, kw, cb in allo.grid(B, Cout, Oh, Ow, Kh, Kw, NB):
        ih: int32 = oh * Sh + kh - Pd0
        iw: int32 = ow * Sw + kw - Pd1
        if ih >= 0 and ih < H and iw >= 0 and iw < W:
            d: float32 = 0.0
            with allo.meta_for(BS) as e:
                fx: float32 = 0.0
                if cb * BS + e < Cin:
                    fx = inp[n, cb * BS + e, ih, iw] * kernel[cout, cb * BS + e, kh, kw]
                d = d + fx
            Z[n, cout, oh, ow] += d
    return Z


def schedule_fp32_conv2d(s):
    """nn.schedule_mx_conv2d plus block-lane banks for inp/kernel."""
    _, Cin = s.inst_list[:2]
    top = s.top_func_name
    for arr in ("inp", "kernel"):
        partition_block_lanes(s, MockBuffer(top, arr), Cin, BS)
    s.pipeline(f"{top}:ow0")
    s.reorder(
        f"{top}:oh", f"{top}:ow", f"{top}:kh", f"{top}:kw", f"{top}:cb", f"{top}:cout"
    )
    s.pipeline(f"{top}:cout")
    return s


def _inst(B, Cin, Cout, H, K, S, P):
    Oh = (H + 2 * P - K) // S + 1
    return [B, Cin, Cout, H, H, K, K, Oh, Oh, S, S, P, P]


@pytest.mark.parametrize("Ty", FORMATS.values(), ids=list(FORMATS))
@pytest.mark.parametrize("name", list(CONFIGS))
def test_conv_numerics(name, Ty):
    B, Cin, Cout, H, K, S, P = CONFIGS[name]
    rng = np.random.default_rng(0)
    rand = lambda shape: (
        rng.standard_normal(shape) * 2.0 ** rng.integers(-4, 4, shape)
    ).astype(np.float32)
    inp, kernel, bias = rand((B, Cin, H, H)), rand((Cout, Cin, K, K)), rand((Cout,))
    inst = _inst(*CONFIGS[name])
    Oh = inst[8]
    exact = np.zeros((B, Cout, Oh, Oh)) + bias[None, :, None, None]
    xp = np.pad(inp.astype(np.float64), ((0, 0), (0, 0), (P, P), (P, P)))
    for oh, ow in np.ndindex(Oh, Oh):
        win = xp[:, :, oh * S : oh * S + K, ow * S : ow * S + K]
        exact[:, :, oh, ow] += np.einsum("bchw,ochw->bo", win, kernel)
    rel = lambda Z: np.linalg.norm(Z - exact) / np.linalg.norm(exact)
    build = lambda fn, i: allo.customize(fn, instantiate=i).build(target="llvm")
    Z_fp = build(fp32_conv2d, inst)(inp, kernel, bias)
    xq, wq = mx_pack_np(Ty, inp, axis=1), mx_pack_np(Ty, kernel, axis=1)
    Z_mx = build(nn.mx_conv2d, [Ty, *inst])(xq, wq, bias)
    print(
        f"\n[{name}] rel norm vs fp64: fp32 {rel(Z_fp):.2e}, {Ty.name} {rel(Z_mx):.2e}"
    )
    assert rel(Z_fp) < 1e-5
    assert rel(Z_mx) < (0.1 if Ty.elem_bits >= 8 else 0.5)


@pytest.mark.parametrize("name", list(CONFIGS))
def test_conv_csyn(name):
    Ty = FORMATS[os.environ.get("ALLO_MXFORMAT", "mxint8_k32")]
    inst = _inst(*CONFIGS[name])
    s_mx = nn.schedule_mx_conv2d(allo.customize(nn.mx_conv2d, instantiate=[Ty, *inst]))
    s_fp = schedule_fp32_conv2d(allo.customize(fp32_conv2d, instantiate=inst))
    for s in (s_mx, s_fp):
        assert "#pragma HLS pipeline" in str(s.build(target="vhls"))
    if os.environ.get("ALLO_HLSMODE", "csyn") != "csyn":
        return
    if not hls.is_available("vitis_hls"):
        pytest.skip("vitis_hls not available")
    root = os.environ.get("ALLO_PROJECT", os.path.join(_TESTS_DIR, "comp_fp_mx_conv"))
    for tag, s, top in ((Ty.name, s_mx, "mx_conv2d"), ("fp32", s_fp, "fp32_conv2d")):
        project = f"{root}_{name}_{tag}.prj"
        mod = s.build(target="vitis_hls", mode="csyn", project=project)
        # issue #603: extern "C" functions returning ap_int<N>
        patch_extern_c_for_class_return_types(f"{project}/kernel.cpp")
        mod()
        rpt = os.path.join(
            project, "out.prj", "solution1", "syn", "report", f"{top}_csynth.rpt"
        )
        assert os.path.isfile(rpt)
        print(f"[{name}] {tag}: {rpt}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project",
        default=os.path.join(_TESTS_DIR, "comp_fp_mx_conv"),
        help="Project root; projects are <root>_<config>_{<format>,fp32}.prj",
    )
    parser.add_argument("--format", choices=list(FORMATS), default="mxint8_k32")
    parser.add_argument("--clean", action="store_true", help="Clean the project dirs")
    args, pytest_args = parser.parse_known_args()
    os.environ["ALLO_PROJECT"] = args.project
    os.environ["ALLO_MXFORMAT"] = args.format
    if args.clean:
        for name in CONFIGS:
            for tag in (args.format, "fp32"):
                shutil.rmtree(f"{args.project}_{name}_{tag}.prj", ignore_errors=True)
    pytest.main([__file__, "-s", *pytest_args])
