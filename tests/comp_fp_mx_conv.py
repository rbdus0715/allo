# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=unsubscriptable-object, no-value-for-parameter

# nn.mx_conv2d (mxint8) vs the same dataflow in plain float32: im2col streams
# one 32-element block per token and the dot stage does one block dot per
# cycle in both; float32 just has no quantize stages.

import os
import shutil

import numpy as np
import pytest
import allo
import allo.backend.hls as hls
import allo.ir.types as T
import allo.library.nn as nn
from allo.ir.types import int32, float32, UInt, ConstExpr
from allo.ir.utils import MockBuffer
from allo.library.nn import (
    _mx_conv_rowblocks,
    _mx_conv_lb_im2col,
    partition_block_lanes,
    schedule_conv_lb_input,
)
from allo.library.mxint8 import (
    mx_pack_nhwc_blocks,
    patch_extern_c_for_class_return_types,
)

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
BS = 32  # block size, same as mxint8

# (B, Cin, Cout, H, K, S, P)
CONFIGS = {
    "small": (1, 4, 8, 8, 3, 1, 1),  # Cin*K*K = 36 -> 2 blocks
    "medium": (1, 16, 16, 16, 3, 1, 1),  # 144 -> 5 blocks
}


def _fp_conv_wpack[
    Cout, Cin, Kh, Kw, CKK, NBc, BS
](kernel: "float32[Cout, Cin, Kh, Kw]", WF: "UInt(32 * BS)[Cout, NBc]"):
    # flatten to [Cout, CKK] (zero-padded), one word per weight block
    # flatten and zero-pad in one pass (no separate zero-fill of kflat)
    kflat: float32[Cout, CKK]
    for fo, j in allo.grid(Cout, CKK):
        fk: float32 = 0.0
        if j < Cin * Kh * Kw:
            fk = kernel[fo, j // (Kh * Kw), (j // Kw) % Kh, j % Kw]
        kflat[fo, j] = fk
    for fo, b in allo.grid(Cout, NBc):
        word: UInt(32 * BS) = 0
        with allo.meta_for(BS) as e:
            fv: float32 = kflat[fo, b * BS + e]
            word[e * 32 : (e + 1) * 32] = fv.bitcast()
        WF[fo, b] = word


def _fp_conv_dot[
    B, Cout, Oh, Ow, NBc, BS
](
    P: "UInt(32 * BS)[B * Oh * Ow, NBc]",
    WF: "UInt(32 * BS)[Cout, NBc]",
    bias: "float32[Cout]",
    Z: "float32[B, Cout, Oh, Ow]",
):
    # same loop order as nn._mx_conv_dot: one float32 block dot per iteration
    acc: float32[Cout] = 0.0
    xw: UInt(32 * BS) = 0
    for n, oh, ow in allo.grid(B, Oh, Ow):
        for b, cout in allo.grid(NBc, Cout):
            if cout == 0:
                xw = P[(n * Oh + oh) * Ow + ow, b]
            ww: UInt(32 * BS) = WF[cout, b]
            d: float32 = 0.0
            with allo.meta_for(BS) as e:
                xb: int32 = xw[e * 32 : (e + 1) * 32]
                wb: int32 = ww[e * 32 : (e + 1) * 32]
                xf: float32 = xb.bitcast()
                wf: float32 = wb.bitcast()
                d = d + xf * wf
            prev: float32 = 0.0
            if b > 0:
                prev = acc[cout]
            total: float32 = prev + d
            acc[cout] = total
            if b == NBc - 1:
                Z[n, cout, oh, ow] = total + bias[cout]


def fp32_conv2d[
    B, Cin, Cout, H, W, Kh, Kw, Oh, Ow, Sh, Sw, Pd0, Pd1
](
    inp: "float32[B, Cin, H, W]",
    kernel: "float32[Cout, Cin, Kh, Kw]",
    bias: "float32[Cout]",
) -> "float32[B, Cout, Oh, Ow]":
    NBc: ConstExpr[int32] = (Cin * Kh * Kw + BS - 1) // BS
    CKK: ConstExpr[int32] = NBc * BS
    R: ConstExpr[int32] = B * Oh * Ow
    PW: ConstExpr[int32] = 32 * BS
    WF: UInt(PW)[Cout, NBc]
    _fp_conv_wpack[Cout, Cin, Kh, Kw, CKK, NBc, BS](kernel, WF)
    WB: ConstExpr[int32] = (W * Cin + BS - 1) // BS
    Q: UInt(PW)[B * H, WB]
    _mx_conv_rowblocks[B, Cin, H, W, BS, WB](inp, Q)
    P: UInt(PW)[R, NBc]
    _mx_conv_lb_im2col[B, Cin, H, W, Kh, Kw, Oh, Ow, Sh, Sw, Pd0, Pd1, BS, NBc, WB](
        Q, P
    )
    Z: float32[B, Cout, Oh, Ow]
    _fp_conv_dot[B, Cout, Oh, Ow, NBc, BS](P, WF, bias, Z)
    return Z


def schedule_fp32_conv2d(s, depth=4):
    """Same schedule as nn.schedule_mx_conv2d, minus the quantize stage."""
    _, Cin, _, _, W, Kh, Kw = s.inst_list[:7]
    NBc = (Cin * Kh * Kw + BS - 1) // BS
    top = s.top_func_name
    # weights: flatten one per cycle, then pack one block per cycle
    partition_block_lanes(s, MockBuffer("_fp_conv_wpack", "kflat"), NBc * BS, BS)
    s.pipeline("_fp_conv_wpack:j")
    s.pipeline("_fp_conv_wpack:b")
    schedule_conv_lb_input(s, top, Cin, W, BS, max(depth, NBc))
    s.pipeline("_fp_conv_dot:cout")
    s.to(MockBuffer(top, "P"), "_fp_conv_dot", depth=max(depth, NBc))
    s.dataflow(top)
    return s


# MX formats to compare against float32; csyn takes one (--format)
FORMATS = {
    t.name: t
    for t in (T.MXInt(8, 32), T.MXInt(4, 16), T.MXFP(4, 3, 32), T.MXFP(2, 1, 32))
}


def _mx_format():
    return FORMATS[os.environ.get("ALLO_MXFORMAT", "mxint8_k32")]


# DRAM boundary: the feature map arrives NHWC-packed as 512b words over two
# ports (one 32-float block per cycle), unpacked on chip by nn.mx_unpack_nhwc
def mx_conv_top[
    Ty, B, Cin, Cout, H, W, Kh, Kw, Oh, Ow, Sh, Sw, Pd0, Pd1
](
    I0: "UInt(512)[B * H, W * Cin // 32]",
    I1: "UInt(512)[B * H, W * Cin // 32]",
    kernel: "float32[Cout, Cin, Kh, Kw]",
    bias: "float32[Cout]",
) -> "float32[B, Cout, Oh, Ow]":
    inp: float32[B, Cin, H, W]
    nn.mx_unpack_nhwc[B, Cin, H, W](I0, I1, inp)
    return nn.mx_conv2d[Ty, B, Cin, Cout, H, W, Kh, Kw, Oh, Ow, Sh, Sw, Pd0, Pd1](
        inp, kernel, bias
    )


def fp32_conv_top[
    B, Cin, Cout, H, W, Kh, Kw, Oh, Ow, Sh, Sw, Pd0, Pd1
](
    I0: "UInt(512)[B * H, W * Cin // 32]",
    I1: "UInt(512)[B * H, W * Cin // 32]",
    kernel: "float32[Cout, Cin, Kh, Kw]",
    bias: "float32[Cout]",
) -> "float32[B, Cout, Oh, Ow]":
    inp: float32[B, Cin, H, W]
    nn.mx_unpack_nhwc[B, Cin, H, W](I0, I1, inp)
    return fp32_conv2d[B, Cin, Cout, H, W, Kh, Kw, Oh, Ow, Sh, Sw, Pd0, Pd1](
        inp, kernel, bias
    )


def _top_schedule(top, layer, inst):
    s = allo.customize(top, instantiate=inst)
    B, Cin, _, H, W = inst[-13:-8]  # the 13 conv dims, after Ty if any
    s.compose(nn.mx_unpack_nhwc, instantiate=[B, Cin, H, W])
    s.compose(layer)  # its inp banking reaches the unpack too
    s.dataflow(top.__name__)
    return s


def _inst(B, Cin, Cout, H, K, S, P):
    Oh = (H + 2 * P - K) // S + 1
    return [B, Cin, Cout, H, H, K, K, Oh, Oh, S, S, P, P]


def _rand(shape, rng):
    return (rng.standard_normal(shape) * 2.0 ** rng.integers(-4, 4, shape)).astype(
        np.float32
    )


def _exact(inp, kernel, bias, K, S, P):
    B, Cout = inp.shape[0], kernel.shape[0]
    Oh = (inp.shape[2] + 2 * P - K) // S + 1
    x = np.pad(inp, ((0, 0), (0, 0), (P, P), (P, P))).astype(np.float64)
    out = np.zeros((B, Cout, Oh, Oh))
    for oh, ow in np.ndindex(Oh, Oh):
        win = x[:, :, oh * S : oh * S + K, ow * S : ow * S + K]
        out[:, :, oh, ow] = np.einsum("bchw,ochw->bo", win, kernel.astype(np.float64))
    return out + bias[None, :, None, None]


@pytest.mark.parametrize("Ty", FORMATS.values(), ids=list(FORMATS))
@pytest.mark.parametrize("name", list(CONFIGS))
def test_conv_numerics(name, Ty):
    B, Cin, Cout, H, K, S, P = CONFIGS[name]
    rng = np.random.default_rng(0)
    inp = _rand((B, Cin, H, H), rng)
    kernel = _rand((Cout, Cin, K, K), rng)
    bias = _rand((Cout,), rng)
    inst = _inst(*CONFIGS[name])
    exact = _exact(inp, kernel, bias, K, S, P)
    rel = lambda Z: np.linalg.norm(Z - exact) / np.linalg.norm(exact)
    args = (*mx_pack_nhwc_blocks(inp), kernel, bias)
    Z_fp = allo.customize(fp32_conv_top, instantiate=inst).build(target="llvm")(*args)
    Z_mx = allo.customize(mx_conv_top, instantiate=[Ty, *inst]).build(target="llvm")(
        *args
    )
    # the packed interface changes nothing but how the input reaches the chip
    ref_mx = allo.customize(nn.mx_conv2d, instantiate=[Ty, *inst]).build(target="llvm")(
        inp, kernel, bias
    )
    np.testing.assert_array_equal(Z_mx, ref_mx)
    print(
        f"\n[{name}] rel norm err vs fp64: fp32 {rel(Z_fp):.2e}, "
        f"{Ty.name} {rel(Z_mx):.2e}"
    )
    assert rel(Z_fp) < 1e-5
    assert rel(Z_mx) < (0.1 if Ty.elem_bits >= 8 else 0.5)


def _schedules(name, Ty):
    inst = _inst(*CONFIGS[name])
    s_mx = nn.schedule_mx_conv2d(allo.customize(nn.mx_conv2d, instantiate=[Ty, *inst]))
    s_fp = schedule_fp32_conv2d(allo.customize(fp32_conv2d, instantiate=inst))
    return (
        _top_schedule(mx_conv_top, s_mx, [Ty, *inst]),
        _top_schedule(fp32_conv_top, s_fp, inst),
    )


@pytest.mark.parametrize("name", list(CONFIGS))
def test_conv_csyn(name):
    Ty = _mx_format()
    s_mx, s_fp = _schedules(name, Ty)
    for s in (s_mx, s_fp):
        assert "#pragma HLS dataflow" in str(s.build(target="vhls"))
    if os.environ.get("ALLO_HLSMODE", "csyn") != "csyn":
        return
    if not hls.is_available("vitis_hls"):
        pytest.skip("vitis_hls not available")
    root = os.environ.get("ALLO_PROJECT", os.path.join(_TESTS_DIR, "comp_fp_mx_conv"))
    for tag, s, top in (
        (Ty.name, s_mx, "mx_conv_top"),
        ("fp32", s_fp, "fp32_conv_top"),
    ):
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
