# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# nn.mx_conv2d: LLVM numerics per MX format, plus Vitis HLS runs:
#   pytest tests/test_mx_conv2d.py                       -> LLVM + HLS codegen
#   python tests/test_mx_conv2d.py --mode csyn|hw_emu    -> + Vitis HLS

import os
import shutil

import numpy as np
import pytest
import allo
import allo.backend.hls as hls
import allo.ir.types as T
import allo.library.nn as nn
from allo.ir.types import float32
from allo.library.mx import (
    mx_pack_channels_np,
    mx_unpack_np,
    patch_extern_c_for_class_return_types,
)

# MX formats as MXInt(elem_bits, block_size) / MXFP(exp_bits, mantissa_bits, block_size)
FORMATS = [T.MXInt(8, 32), T.MXInt(4, 16), T.MXFP(4, 3, 32), T.MXFP(2, 1, 32)]

# (B, Cin, Cout, H, K, S, P)
CONFIGS = [(1, 2, 3, 6, 4, 1, 0), (2, 3, 4, 7, 3, 2, 1), (1, 40, 8, 8, 3, 1, 1)]


@pytest.mark.parametrize("Ty", FORMATS, ids=lambda t: t.name)
@pytest.mark.parametrize("B, Cin, Cout, H, K, S, P", CONFIGS)
def test_mx_conv2d(Ty, B, Cin, Cout, H, K, S, P):
    rng = np.random.default_rng(0)
    rand = lambda shape: (
        rng.standard_normal(shape) * 2.0 ** rng.integers(-4, 4, shape)
    ).astype(np.float32)
    inp, kernel, bias = rand((B, Cin, H, H)), rand((Cout, Cin, K, K)), rand((Cout,))
    # weights packed on the host (blocks along the input channels); xq is the
    # same packing of inp, as the layer quantizes it, for the reference
    xq, wq = mx_pack_channels_np(Ty, inp), mx_pack_channels_np(Ty, kernel)
    Oh = (H + 2 * P - K) // S + 1
    inst = [Ty, float32, B, Cin, Cout, H, H, K, K, Oh, Oh, S, S, P, P]
    Z = allo.customize(nn.mx_conv2d, instantiate=inst).build(target="llvm")(
        inp, wq, bias
    )

    # reference: per output pixel, exact block dots in (channel block, kh, kw)
    # order (padding taps are 0), summed in float32 in groups of G blocks as
    # mx_matmul does (G = 4 for fewer than 6 output channels), then the bias
    BS, NB = Ty.block_size, -(-Cin // Ty.block_size)
    G = 1 if Cout >= 6 else 4
    xd = mx_unpack_np(Ty, xq.reshape(B, NB, H, H), axis=1)  # [B, NB * BS, H, W]
    wd = mx_unpack_np(Ty, wq.reshape(Cout, NB, K, K), axis=1)
    ref = np.zeros((B, Cout, Oh, Oh), np.float32)
    for n, oh, ow in np.ndindex(B, Oh, Oh):
        dots = []
        for cb, kh, kw in np.ndindex(NB, K, K):
            ih, iw = oh * S + kh - P, ow * S + kw - P
            dot = np.zeros(Cout, np.float32)
            if 0 <= ih < H and 0 <= iw < H:
                lanes = slice(cb * BS, (cb + 1) * BS)
                dot = (wd[:, lanes, kh, kw] @ xd[n, lanes, ih, iw]).astype(np.float32)
            dots.append(dot)
        acc = np.zeros(Cout, np.float32)
        for g in range(0, len(dots), G):
            part = np.zeros(Cout, np.float32)
            for dot in dots[g : g + G]:
                part += dot
            acc += part
        ref[n, :, oh, ow] = acc + bias
    np.testing.assert_array_equal(Z, ref)

    exact = np.zeros((B, Cout, Oh, Oh)) + bias[None, :, None, None]
    xp = np.pad(inp.astype(np.float64), ((0, 0), (0, 0), (P, P), (P, P)))
    for oh, ow in np.ndindex(Oh, Oh):
        win = xp[:, :, oh * S : oh * S + K, ow * S : ow * S + K]
        exact[:, :, oh, ow] += np.einsum("bchw,ochw->bo", win, kernel)
    rel = np.linalg.norm(Z - exact) / np.linalg.norm(exact)
    print(f"{Ty.name} mx_conv2d vs float64: rel norm {rel:.4f}")
    assert rel < (0.1 if Ty.elem_bits >= 8 else 0.5)  # loose for 4-bit elements

    s = allo.customize(nn.mx_conv2d, instantiate=inst)
    code = str(nn.schedule_mx_conv2d(s).build(target="vhls"))
    assert "#pragma HLS dataflow" in code  # windows -> mx_linear2d -> NCHW
    assert code.count("hls::stream<") >= 2


def test_mx_conv2d_hls():
    # Vitis HLS run of one config/format, when --mode (ALLO_HLSMODE) is given
    mode = os.environ.get("ALLO_HLSMODE")
    if not mode or not hls.is_available("vitis_hls"):
        return
    if mode != "csyn" and "XDEVICE" not in os.environ:
        print(f"Skipping {mode} run: set XDEVICE to a platform .xpfm to run this mode")
        return
    Ty = {t.name: t for t in FORMATS}[os.environ.get("ALLO_MXFORMAT", "mxint8_k32")]
    B, Cin, Cout, H, K, S, P = CONFIGS[-1]
    Oh = (H + 2 * P - K) // S + 1
    conv = [Ty, float32, B, Cin, Cout, H, H, K, K, Oh, Oh, S, S, P, P]

    def top(
        inp: "float32[B, Cin, H, H]",
        kernel: "float32[Cout, Cin, K, K]",
        bias: "float32[Cout]",
    ) -> "float32[B, Cout, Oh, Oh]":
        wq = nn.mx_quantize_channels[Ty, float32, Cout, Cin, K, K](kernel)
        return nn.mx_conv2d[Ty, float32, B, Cin, Cout, H, H, K, K, Oh, Oh, S, S, P, P](
            inp, wq, bias
        )

    s = allo.customize(top)
    s.compose(nn.mx_quantize_channels, instantiate=[Ty, float32, Cout, Cin, K, K])
    s.compose(nn.mx_conv2d, instantiate=conv)
    project = os.environ["ALLO_PROJECT"]
    hls_mod = s.build(target="vitis_hls", mode=mode, project=project)
    # issue #603: extern "C" functions returning ap_int<N> (the packed MX word)
    patch_extern_c_for_class_return_types(f"{project}/kernel.cpp")
    if mode == "csyn":
        hls_mod()
        return

    rng = np.random.default_rng(0)
    inp = rng.standard_normal((B, Cin, H, H)).astype(np.float32)
    kernel = rng.standard_normal((Cout, Cin, K, K)).astype(np.float32)
    bias = rng.standard_normal(Cout).astype(np.float32)
    Z = np.zeros((B, Cout, Oh, Oh), np.float32)
    hls_mod(inp, kernel, bias, Z)
    ref = allo.customize(top).build(target="llvm")(inp, kernel, bias)
    print(f"[{mode}] hw == LLVM bit-exact: {np.array_equal(Z, ref)}")
    np.testing.assert_array_equal(Z, ref)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["csyn", "sw_emu", "hw_emu", "hw"], default="csyn"
    )
    parser.add_argument(
        "--project",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "mx_conv2d.prj"
        ),
    )
    parser.add_argument(
        "--format", choices=[t.name for t in FORMATS], default="mxint8_k32"
    )
    parser.add_argument("--clean", action="store_true", help="Clean the project dir")

    args, pytest_args = parser.parse_known_args()

    os.environ["ALLO_HLSMODE"] = args.mode
    os.environ["ALLO_PROJECT"] = args.project
    os.environ["ALLO_MXFORMAT"] = args.format
    if args.clean:
        shutil.rmtree(args.project, ignore_errors=True)

    pytest.main([__file__, "-s", *pytest_args])
