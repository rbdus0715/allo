# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=unsubscriptable-object, no-value-for-parameter

# nn.mx_linear2d vs the same loop in plain float32: one block dot of 32
# elements per cycle in both. MX: X is quantized on chip, W arrives
# host-packed (one word per block); float32 reads a block from 32 banks.

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
from allo.library.nn import partition_block_lanes
from allo.library.mx import (
    mx_pack_f32_blocks,
    mx_pack_np,
    patch_extern_c_for_class_return_types,
)

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
BS = 32  # block size, same as mxint8

# (M, N, K): Z[M, N] = X[M, K] @ W[N, K]^T + b
CONFIGS = {
    "small": (16, 64, 128),  # 4 blocks per row
    "medium": (32, 256, 256),  # 8 blocks per row
}


def fp32_linear2d[
    M, N, K
](X: "float32[M, K]", W: "float32[N, K]", bias: "float32[N]") -> "float32[M, N]":
    # nn.mx_linear2d's loops with a float32 block dot of BS elements
    NB: ConstExpr[int32] = (K + BS - 1) // BS
    Z: float32[M, N]
    acc: float32[N] = 0.0
    for m in range(M):
        for b, n in allo.grid(NB, N):
            d: float32 = 0.0
            with allo.meta_for(BS) as e:
                fx: float32 = 0.0
                if b * BS + e < K:
                    fx = X[m, b * BS + e] * W[n, b * BS + e]
                d = d + fx
            prev: float32 = 0.0
            if b > 0:
                prev = acc[n]
            total: float32 = prev + d
            acc[n] = total
            if b == NB - 1:
                Z[m, n] = total + bias[n]
    return Z


def schedule_fp32_linear2d(s):
    """nn.schedule_mx_linear2d plus block-lane banks for X/W."""
    _, _, K = s.inst_list[:3]
    top = s.top_func_name
    for arr in ("X", "W"):
        partition_block_lanes(s, MockBuffer(top, arr), K, BS)
    s.pipeline(f"{top}:n")
    return s


# MX formats to compare against float32; csyn takes one (--format)
FORMATS = {
    t.name: t
    for t in (T.MXInt(8, 32), T.MXInt(4, 16), T.MXFP(4, 3, 32), T.MXFP(2, 1, 32))
}


def _mx_format():
    return FORMATS[os.environ.get("ALLO_MXFORMAT", "mxint8_k32")]


# DRAM boundary: float32 X (and fp32 W) arrive as 512b words over two ports
# (one 32-float block per cycle), unpacked on chip by nn.mx_unpack_f32
def mx_linear_top[
    Ty, M, N, K
](
    X0: "UInt(512)[M, K // 32]",
    X1: "UInt(512)[M, K // 32]",
    Wq: "Ty[N, K // Ty.block_size]",
    bias: "float32[N]",
) -> "float32[M, N]":
    X: float32[M, K]
    nn.mx_unpack_f32[M, K](X0, X1, X)
    return nn.mx_linear2d[Ty, float32, M, N, K](X, Wq, bias)


def fp32_linear_top[
    M, N, K
](
    X0: "UInt(512)[M, K // 32]",
    X1: "UInt(512)[M, K // 32]",
    W0: "UInt(512)[N, K // 32]",
    W1: "UInt(512)[N, K // 32]",
    bias: "float32[N]",
) -> "float32[M, N]":
    X: float32[M, K]
    W: float32[N, K]
    nn.mx_unpack_f32[M, K](X0, X1, X)
    nn.mx_unpack_f32[N, K](W0, W1, W)
    return fp32_linear2d[M, N, K](X, W, bias)


def _schedules(name, Ty):
    M, N, K = CONFIGS[name]
    s_mx = allo.customize(mx_linear_top, instantiate=[Ty, M, N, K])
    s_mx.compose(nn.mx_unpack_f32, instantiate=[M, K])
    # its X block-lane banks reach the unpack too
    s_mx.compose(nn.mx_linear2d, instantiate=[Ty, float32, M, N, K])
    s_mx.dataflow("mx_linear_top")
    s_fp = allo.customize(fp32_linear_top, instantiate=[M, N, K])
    s_fp.compose(nn.mx_unpack_f32, instantiate=[M, K])
    s_fp.compose(nn.mx_unpack_f32, instantiate=[N, K], id="1")
    s_fp.compose(
        schedule_fp32_linear2d(allo.customize(fp32_linear2d, instantiate=[M, N, K]))
    )
    s_fp.dataflow("fp32_linear_top")
    return s_mx, s_fp


def _rand(shape, rng):
    return (rng.standard_normal(shape) * 2.0 ** rng.integers(-4, 4, shape)).astype(
        np.float32
    )


@pytest.mark.parametrize("Ty", FORMATS.values(), ids=list(FORMATS))
@pytest.mark.parametrize("name", list(CONFIGS))
def test_linear_numerics(name, Ty):
    M, N, K = CONFIGS[name]
    rng = np.random.default_rng(0)
    X, W, bias = _rand((M, K), rng), _rand((N, K), rng), _rand((N,), rng)
    exact = X.astype(np.float64) @ W.astype(np.float64).T + bias
    rel = lambda Z: np.linalg.norm(Z - exact) / np.linalg.norm(exact)
    Z_fp = allo.customize(fp32_linear_top, instantiate=[M, N, K]).build(target="llvm")(
        *mx_pack_f32_blocks(X), *mx_pack_f32_blocks(W), bias
    )
    Wq = mx_pack_np(Ty, W)
    Z_mx = allo.customize(mx_linear_top, instantiate=[Ty, M, N, K]).build(
        target="llvm"
    )(*mx_pack_f32_blocks(X), Wq, bias)
    # the packed DRAM interface changes nothing but how X reaches the chip
    ref_mx = allo.customize(nn.mx_linear2d, instantiate=[Ty, float32, M, N, K]).build(
        target="llvm"
    )(X, Wq, bias)
    np.testing.assert_array_equal(Z_mx, ref_mx)
    print(
        f"\n[{name}] rel norm err vs fp64: fp32 {rel(Z_fp):.2e}, "
        f"{Ty.name} {rel(Z_mx):.2e}"
    )
    assert rel(Z_fp) < 1e-5
    assert rel(Z_mx) < (0.1 if Ty.elem_bits >= 8 else 0.5)


@pytest.mark.parametrize("name", list(CONFIGS))
def test_linear_csyn(name):
    Ty = _mx_format()
    s_mx, s_fp = _schedules(name, Ty)
    for s in (s_mx, s_fp):
        assert "#pragma HLS dataflow" in str(s.build(target="vhls"))
    if os.environ.get("ALLO_HLSMODE", "csyn") != "csyn":
        return
    if not hls.is_available("vitis_hls"):
        pytest.skip("vitis_hls not available")
    root = os.environ.get("ALLO_PROJECT", os.path.join(_TESTS_DIR, "comp_fp_mx_linear"))
    for tag, s, top in (
        (Ty.name, s_mx, "mx_linear_top"),
        ("fp32", s_fp, "fp32_linear_top"),
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
        default=os.path.join(_TESTS_DIR, "comp_fp_mx_linear"),
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
