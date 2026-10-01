# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import allo
import allo.ir.types as T
from allo.backend.simulator import LLVMOMPModule
from allo.library.nn import (
    mx_matmul,
    mx_linear2d,
    mx_linear3d,
    make_mx_matmul_dataflow,
    make_mx_linear2d_dataflow,
)

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


def test_mx_matmul_dataflow():
    # Same math as test_mx_matmul, but built through the streaming
    # (df.region/df.kernel) implementation instead of allo.customize.
    M, N, K, P = 4, 6, 64, 2
    rng = np.random.default_rng(0)
    A = _rand((M, K), rng)
    B = _rand((N, K), rng)

    s = make_mx_matmul_dataflow(Ty, M, N, K, P)
    mod = LLVMOMPModule(s.module, s.top_func_name)
    Z = np.zeros((M, N), dtype=np.float32)
    mod(A, B, Z)

    ref = A.astype(np.float64) @ B.astype(np.float64).T
    _assert_close(Z, ref, "mx_matmul_dataflow")


def test_mx_linear2d_dataflow():
    M, N, K, P = 4, 6, 64, 2
    rng = np.random.default_rng(1)
    X = _rand((M, K), rng)
    W = _rand((N, K), rng)
    bias = _rand((N,), rng)

    s = make_mx_linear2d_dataflow(Ty, M, N, K, P)
    mod = LLVMOMPModule(s.module, s.top_func_name)
    Z = np.zeros((M, N), dtype=np.float32)
    mod(X, W, bias, Z)

    ref = X.astype(np.float64) @ W.astype(np.float64).T + bias.astype(np.float64)
    _assert_close(Z, ref, "mx_linear2d_dataflow")


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
