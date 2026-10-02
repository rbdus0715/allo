# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import allo
import allo.ir.types as T
from allo.backend.simulator import LLVMOMPModule
import allo.library.nn as nn
from allo.library.nn import (
    mx_matmul,
    mx_linear2d,
    mx_linear2d_ref,
    mx_linear3d,
    make_mx_matmul_dataflow,
    make_mx_linear2d_dataflow,
    make_mx_linear2d_dataflow_prequant,
    mx_hbm_mapping,
    mx_pick_tile,
)
from allo.library.mxint8 import mx_quantize, mx_quantize_weights, mx_pack_words

Ty = T.mxint8

_REL_NORM_TOL = 0.1


def _rand(shape, rng):
    return (rng.standard_normal(shape) * 2.0 ** rng.integers(-4, 4, shape)).astype(
        np.float32
    )


def _assert_close(Z, ref, label):
    rel_err = np.abs(Z - ref) / np.maximum(np.abs(ref), 1e-6)
    rel_norm = np.linalg.norm(Z - ref) / np.linalg.norm(ref)
    print(
        f"{label}: max per-element rel err {rel_err.max():.4f}, rel norm {rel_norm:.4f}"
    )
    assert rel_norm < _REL_NORM_TOL, f"{label}: relative norm error {rel_norm} too high"


def test_mx_matmul():
    M, N, K = 4, 6, 64  # K = 2 * mxint8.block_size
    rng = np.random.default_rng(0)
    A = _rand((M, K), rng)
    B = _rand((N, K), rng)

    s = allo.customize(mx_matmul, instantiate=[Ty, M, N, K])
    mod = s.build(target="llvm")
    Z = mod(A, B)

    ref = A.astype(np.float64) @ B.astype(np.float64).T
    _assert_close(Z, ref, "mx_matmul")


def test_mx_linear2d():
    M, N, K = 4, 6, 64
    rng = np.random.default_rng(1)
    X = _rand((M, K), rng)
    W = _rand((N, K), rng)
    bias = _rand((N,), rng)

    s = allo.customize(mx_linear2d, instantiate=[Ty, M, N, K])
    mod = s.build(target="llvm")
    Z = mod(X, W, bias)

    ref = X.astype(np.float64) @ W.astype(np.float64).T + bias.astype(np.float64)
    _assert_close(Z, ref, "mx_linear2d")

    s_ref = allo.customize(mx_linear2d_ref, instantiate=[Ty, M, N, K])
    np.testing.assert_array_equal(Z, s_ref.build(target="llvm")(X, W, bias))

    from allo.ir.types import float32
    from allo.library import nn

    def kernel(
        X: "float32[M, K]", W: "float32[N, K]", bias: "float32[N]"
    ) -> "float32[M, N]":
        return nn.mx_linear2d[Ty, M, N, K](X, W, bias)

    s_k = allo.customize(kernel)
    np.testing.assert_array_equal(Z, s_k.build(target="llvm")(X, W, bias))
    assert "#pragma HLS dataflow" in str(s_k.build(target="vhls"))


def test_mx_linear3d():
    B, L, D, M = 2, 3, 64, 5
    rng = np.random.default_rng(2)
    X = _rand((B, L, D), rng)
    W = _rand((M, D), rng)
    bias = _rand((M,), rng)

    s = allo.customize(mx_linear3d, instantiate=[Ty, B, L, D, M])
    mod = s.build(target="llvm")
    Z = mod(X, W, bias)

    ref = np.einsum("bld,md->blm", X.astype(np.float64), W.astype(np.float64))
    ref = ref + bias.astype(np.float64)
    _assert_close(Z, ref, "mx_linear3d")


def test_allo_linear_mx_type_dispatch():
    from allo.ir.types import float32

    M, N, K = 4, 6, 64
    rng = np.random.default_rng(3)
    X = _rand((M, K), rng)
    W = _rand((N, K), rng)
    bias = _rand((N,), rng)
    ref = X.astype(np.float64) @ W.astype(np.float64).T + bias.astype(np.float64)

    def kernel_plain(
        X: "float32[M, K]", W: "float32[N, K]", bias: "float32[N]"
    ) -> "float32[M, N]":
        return allo.linear(X, W, bias)

    mod_plain = allo.customize(kernel_plain).build(target="llvm")
    Z_plain = mod_plain(X, W, bias)
    np.testing.assert_allclose(Z_plain, ref, atol=1e-3)

    def kernel_mx(
        X: "float32[M, K]", W: "float32[N, K]", bias: "float32[N]"
    ) -> "float32[M, N]":
        return allo.linear[Ty](X, W, bias)

    mod_mx = allo.customize(kernel_mx).build(target="llvm")
    Z_mx = mod_mx(X, W, bias)
    _assert_close(Z_mx, ref, "allo.linear[mxint8]")


def test_allo_linear_mx_operands():
    from allo.ir.types import float32, int8, e8m0

    M, N, K = 4, 6, 64
    NB = K // Ty.block_size
    rng = np.random.default_rng(5)
    X = _rand((M, K), rng)
    W = _rand((N, K), rng)
    B = _rand((K, N), rng)
    bias = _rand((N,), rng)
    ref_lin = allo.customize(mx_linear2d_ref, instantiate=[Ty, M, N, K]).build(
        target="llvm"
    )(X, W, bias)
    ref_mm = allo.customize(mx_matmul, instantiate=[Ty, M, N, K]).build(target="llvm")(
        X, np.ascontiguousarray(B.T)
    )

    def lin_q(
        Xq: "int8[M, K]",
        Xs: "e8m0[M, NB]",
        Wq: "int8[N, K]",
        Ws: "e8m0[N, NB]",
        bias: "float32[N]",
    ) -> "float32[M, N]":
        return allo.linear(Xq, Xs, Wq, Ws, bias)

    def lin_wq(
        X: "float32[M, K]", Wq: "int8[N, K]", Ws: "e8m0[N, NB]", bias: "float32[N]"
    ) -> "float32[M, N]":
        return allo.linear(X, Wq, Ws, bias)

    def mm_wq(
        A: "float32[M, K]", Bq: "int8[K, N]", Bs: "e8m0[NB, N]"
    ) -> "float32[M, N]":
        return allo.matmul(A, Bq, Bs)

    Xq, Xs = mx_quantize(Ty, X)
    Wq, Ws = mx_quantize(Ty, W)
    Bq, Bs = mx_quantize(Ty, B, axis=0)
    build = lambda f: allo.customize(f).build(target="llvm")
    np.testing.assert_array_equal(build(lin_q)(Xq, Xs, Wq, Ws, bias), ref_lin)
    np.testing.assert_array_equal(build(lin_wq)(X, Wq, Ws, bias), ref_lin)
    np.testing.assert_array_equal(build(mm_wq)(X, Bq, Bs), ref_mm)


@pytest.mark.parametrize(
    "M, N, K, P, Tn", [(1, 16, 64, 1, 4), (4, 16, 128, 2, 8), (5, 12, 96, 1, 3)]
)
def test_mx_operand_tiled(M, N, K, P, Tn):
    NB = K // Ty.block_size
    rng = np.random.default_rng(6)
    X = _rand((M, K), rng)
    W = _rand((N, K), rng)
    B = _rand((K, N), rng)
    bias = _rand((N,), rng)
    ref_lin = allo.customize(mx_linear2d_ref, instantiate=[Ty, M, N, K]).build(
        target="llvm"
    )(X, W, bias)
    ref_mm = allo.customize(mx_matmul, instantiate=[Ty, M, N, K]).build(target="llvm")(
        X, np.ascontiguousarray(B.T)
    )
    Xq, Xs = mx_quantize(Ty, X)
    Wq, Ws = mx_quantize(Ty, W)
    Bq, Bs = mx_quantize(Ty, B, axis=0)
    build = lambda f: allo.customize(f, instantiate=[Ty, M, N, K, NB, P, Tn]).build(
        target="llvm"
    )
    np.testing.assert_array_equal(build(nn.mx_linear2d_wq)(X, Wq, Ws, bias), ref_lin)
    np.testing.assert_array_equal(
        build(nn.mx_linear2d_q)(Xq, Xs, Wq, Ws, bias), ref_lin
    )
    np.testing.assert_array_equal(build(nn.mx_matmul_wq)(X, Bq, Bs), ref_mm)
    np.testing.assert_array_equal(build(nn.mx_matmul_q)(Xq, Xs, Bq, Bs), ref_mm)


def test_allo_mx_float_operands():
    from allo.ir.types import float32

    M, N, K = 4, 16, 128
    rng = np.random.default_rng(7)
    A = _rand((M, K), rng)
    B = _rand((K, N), rng)
    W = _rand((N, K), rng)
    bias = _rand((N,), rng)

    def mm(A: "float32[M, K]", B: "float32[K, N]") -> "float32[M, N]":
        return allo.matmul[Ty](A, B)

    def lin(
        X: "float32[M, K]", W: "float32[N, K]", bias: "float32[N]"
    ) -> "float32[M, N]":
        return allo.linear[Ty](X, W, bias)

    ref_mm = allo.customize(mx_matmul, instantiate=[Ty, M, N, K]).build(target="llvm")(
        A, np.ascontiguousarray(B.T)
    )
    ref_lin = allo.customize(mx_linear2d_ref, instantiate=[Ty, M, N, K]).build(
        target="llvm"
    )(A, W, bias)
    np.testing.assert_array_equal(allo.customize(mm).build(target="llvm")(A, B), ref_mm)
    np.testing.assert_array_equal(
        allo.customize(lin).build(target="llvm")(A, W, bias), ref_lin
    )
    assert "#pragma HLS dataflow" in str(allo.customize(mm).build(target="vhls"))


def test_mx_matmul_dataflow():
    M, N, K, P = 8, 6, 64, 2  # M > stream depth (4)
    rng = np.random.default_rng(0)
    A = _rand((M, K), rng)
    B = _rand((N, K), rng)

    s = make_mx_matmul_dataflow(Ty, M, N, K, P)
    mod = LLVMOMPModule(s.module, s.top_func_name)
    Z = np.zeros(M * N, dtype=np.float32)
    mod(mx_pack_words(A, 512), mx_pack_words(B, 512), Z)
    Z = Z.reshape(M, N)

    s_ref = allo.customize(mx_matmul, instantiate=[Ty, M, N, K])
    np.testing.assert_array_equal(Z, s_ref.build(target="llvm")(A, B))

    ref = A.astype(np.float64) @ B.astype(np.float64).T
    _assert_close(Z, ref, "mx_matmul_dataflow")


def test_mx_linear2d_dataflow():
    M, N, K, P = 8, 6, 64, 2  # M > stream depth (4)
    rng = np.random.default_rng(1)
    X = _rand((M, K), rng)
    W = _rand((N, K), rng)
    bias = _rand((N,), rng)

    s = make_mx_linear2d_dataflow(Ty, M, N, K, P)
    mod = LLVMOMPModule(s.module, s.top_func_name)
    Z = np.zeros(M * N, dtype=np.float32)
    mod(mx_pack_words(X, 512), mx_pack_words(W, 512), bias, Z)
    Z = Z.reshape(M, N)

    s_ref = allo.customize(mx_linear2d_ref, instantiate=[Ty, M, N, K])
    np.testing.assert_array_equal(Z, s_ref.build(target="llvm")(X, W, bias))

    ref = X.astype(np.float64) @ W.astype(np.float64).T + bias.astype(np.float64)
    _assert_close(Z, ref, "mx_linear2d_dataflow")


@pytest.mark.parametrize("M, Tn", [(8, None), (8, 2), (1, 2)])
def test_mx_linear2d_dataflow_prequant(M, Tn):
    N, K, P = 6, 64, 2
    rng = np.random.default_rng(4)
    X = _rand((M, K), rng)
    W = _rand((N, K), rng)
    bias = _rand((N,), rng)
    W[0, :32] = 0.0  # all-zero block
    W[1, 3] = 1e-40  # denormal next to normals
    W[2, 40] = 3.0e4  # one huge value flushes the rest of its block
    W[3, 7] = -W[3, 7]

    W_q, W_s = mx_quantize_weights(Ty, W, Tn)
    NB, tn = K // Ty.block_size, Tn or N
    assert W_q.shape == (N // tn, NB, tn, Ty.block_size)
    assert W_s.shape == (N // tn, NB, tn)

    X_w = mx_pack_words(X, 512)
    s = make_mx_linear2d_dataflow_prequant(Ty, M, N, K, P, Tn)
    Z = np.zeros(M * N, dtype=np.float32)
    LLVMOMPModule(s.module, s.top_func_name)(
        X_w, mx_pack_words(W_q, 256), W_s.reshape(-1), bias, Z
    )

    s_ref = make_mx_linear2d_dataflow(Ty, M, N, K, P)
    Z_ref = np.zeros(M * N, dtype=np.float32)
    LLVMOMPModule(s_ref.module, s_ref.top_func_name)(
        X_w, mx_pack_words(W, 512), bias, Z_ref
    )

    np.testing.assert_array_equal(Z, Z_ref)


def test_mx_pick_tile():
    assert mx_pick_tile(Ty, 4096, 4096, 2) == 512  # 512 * 128 blocks * 33 B < 4 MiB
    assert mx_pick_tile(Ty, 6, 64, 2) == 6
    assert mx_pick_tile(Ty, 6, 64, 2, budget_bytes=4 * 66) == 2


def test_mx_hbm_mapping():
    s = make_mx_linear2d_dataflow_prequant(Ty, 8, 6, 64, 2)
    assert mx_hbm_mapping(s) == {
        "X": "HBM[0]",
        "W_q": "HBM[1]",
        "W_s": "HBM[2]",
        "bias": "HBM[2]",
        "Z": "HBM[2]",
    }
    assert mx_hbm_mapping(s, base=4)["W_q"] == "HBM[5]"
    assert mx_hbm_mapping(s, memory="DDR")["X"] == "DDR[0]"  # Alveo U250


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
