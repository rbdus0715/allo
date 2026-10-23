# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil

import numpy as np
import pytest
import allo
import allo.backend.hls as hls
import allo.ir.types as T
import allo.library.nn as nn
from allo.library.mxint8 import mx_quantize, patch_extern_c_for_class_return_types

Ty = T.mxint8

# (B, Cin, Cout, H, K, S, P): Cin*K*K = 32 (one block), 27 -> 32, 36 -> 64
HLS_CONFIG = (1, 4, 8, 8, 3, 1, 1)
CONFIGS = [
    (1, 2, 3, 6, 4, 1, 0),
    (2, 3, 4, 7, 3, 2, 1),
    HLS_CONFIG,
    # line-buffer edge cases: stride 2 without padding, 1x1, Cin > block size
    (1, 8, 4, 9, 3, 2, 0),
    (1, 3, 2, 5, 1, 1, 0),
    (1, 40, 2, 4, 3, 1, 1),
]


def _rand(shape, rng):
    return (rng.standard_normal(shape) * 2.0 ** rng.integers(-4, 4, shape)).astype(
        np.float32
    )


def _inputs(B, Cin, Cout, H, K, seed=0):
    rng = np.random.default_rng(seed)
    return (
        _rand((B, Cin, H, H), rng),
        _rand((Cout, Cin, K, K), rng),
        _rand((Cout,), rng),
    )


def _im2col(inp, K, S, P, CKK):
    B, Cin, H, _ = inp.shape
    Oh = (H + 2 * P - K) // S + 1
    padded = np.pad(inp, ((0, 0), (0, 0), (P, P), (P, P)))
    patches = np.zeros((B * Oh * Oh, CKK), np.float32)
    for r, (n, oh, ow) in enumerate(np.ndindex(B, Oh, Oh)):
        win = padded[n, :, oh * S : oh * S + K, ow * S : ow * S + K]
        patches[r, : Cin * K * K] = win.reshape(-1)
    return patches, Oh


def _mx_conv2d_ref(inp, kernel, bias, K, S, P):
    """numpy model of mx_conv2d: MX blocks dotted exactly, float32 across blocks."""
    B, Cout = inp.shape[0], kernel.shape[0]
    BS = Ty.block_size
    NBc = (kernel[0].size + BS - 1) // BS
    patches, Oh = _im2col(inp, K, S, P, NBc * BS)
    kflat = np.zeros((Cout, NBc * BS), np.float32)
    kflat[:, : kernel[0].size] = kernel.reshape(Cout, -1)
    pe, ps = mx_quantize(Ty, patches)
    ke, ks = mx_quantize(Ty, kflat)
    pe = pe.astype(np.int64).reshape(-1, NBc, BS)
    ke = ke.astype(np.int64).reshape(Cout, NBc, BS)
    ps, ks = ps.astype(np.int64), ks.astype(np.int64)
    out = np.zeros((patches.shape[0], Cout), np.float32)
    for b in range(NBc):  # same block order as mx_dot
        acc = (pe[:, None, b, :] * ke[None, :, b, :]).sum(-1).astype(np.float32)
        sh = ps[:, None, b] + ks[None, :, b] - 254 + Ty.dot_exp_offset
        blk = np.ldexp(acc, sh).astype(np.float32)
        blk[(ps[:, None, b] == 0) | (ks[None, :, b] == 0)] = 0.0
        out += blk
    out += bias
    exact = patches.astype(np.float64) @ kflat.T.astype(np.float64) + bias
    to_nchw = lambda x: x.reshape(B, Oh, Oh, Cout).transpose(0, 3, 1, 2)
    return to_nchw(out), to_nchw(exact)


def _inst(B, Cin, Cout, H, K, S, P):
    Oh = (H + 2 * P - K) // S + 1
    return [Ty, B, Cin, Cout, H, H, K, K, Oh, Oh, S, S, P, P], Oh


@pytest.mark.parametrize("B, Cin, Cout, H, K, S, P", CONFIGS)
def test_mx_conv2d_llvm(B, Cin, Cout, H, K, S, P):
    inp, kernel, bias = _inputs(B, Cin, Cout, H, K)
    inst, _ = _inst(B, Cin, Cout, H, K, S, P)
    Z = allo.customize(nn.mx_conv2d, instantiate=inst).build(target="llvm")(
        inp, kernel, bias
    )
    ref, exact = _mx_conv2d_ref(inp, kernel, bias, K, S, P)
    np.testing.assert_array_equal(Z, ref)
    rel_norm = np.linalg.norm(Z - exact) / np.linalg.norm(exact)
    print(f"mx_conv2d vs float64: rel norm {rel_norm:.4f}")
    assert rel_norm < 0.1


def test_mx_conv2d_hls():
    B, Cin, Cout, H, K, S, P = HLS_CONFIG
    inst, Oh = _inst(B, Cin, Cout, H, K, S, P)
    s = allo.customize(nn.mx_conv2d, instantiate=inst)
    nn.schedule_mx_conv2d(s)
    code = str(s.build(target="vhls"))
    assert "#pragma HLS dataflow" in code
    assert code.count("#pragma HLS stream") >= 3  # P, XS, XE

    mode = os.environ.get("ALLO_HLSMODE")
    project = os.environ.get("ALLO_PROJECT")
    if mode is None or not hls.is_available("vitis_hls"):
        return
    if mode != "csyn" and "XDEVICE" not in os.environ:
        print(f"Skipping {mode} run: set XDEVICE to a platform .xpfm to run this mode")
        return

    hls_mod = s.build(target="vitis_hls", mode=mode, project=project)
    # issue #603: extern "C" functions returning ap_int<N> (the packed MX word)
    patch_extern_c_for_class_return_types(f"{project}/kernel.cpp")

    if mode == "csyn":
        hls_mod()
        rpt = os.path.join(
            project, "out.prj", "solution1", "syn", "report", "mx_conv2d_csynth.rpt"
        )
        assert os.path.isfile(rpt)
        return

    inp, kernel, bias = _inputs(B, Cin, Cout, H, K)
    Z = np.zeros((B, Cout, Oh, Oh), np.float32)
    hls_mod(inp, kernel, bias, Z)
    ref, _ = _mx_conv2d_ref(inp, kernel, bias, K, S, P)
    print(f"[{mode}] max abs diff vs ref: {np.abs(Z - ref).max()}")
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
    parser.add_argument("--clean", action="store_true", help="Clean the project dir")

    args, pytest_args = parser.parse_known_args()

    os.environ["ALLO_HLSMODE"] = args.mode
    os.environ["ALLO_PROJECT"] = args.project
    if args.clean:
        shutil.rmtree(args.project, ignore_errors=True)

    pytest.main([__file__, "-s", *pytest_args])
