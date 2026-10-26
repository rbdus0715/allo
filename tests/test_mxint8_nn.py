# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import allo
import allo.ir.types as T
from allo.ir.types import float32, int32, ConstExpr
import allo.library.nn as nn
from allo.library.mx import mx_quantize_block_f32, mx_block_dot, mx_quantize, mx_pack_np

Ty = T.mxint8

_REL_NORM_TOL = 0.1


def mx_matmul_ref[
    Ty, M, N, K
](A: "float32[M, K]", B: "float32[N, K]") -> "float32[M, N]":
    # A @ B^T, A and B quantized block by block: the plain loops that
    # nn.mx_linear2d / nn.mx_matmul must match
    BS: ConstExpr[int32] = Ty.block_size
    NB: ConstExpr[int32] = K // BS
    Z: float32[M, N]
    for i in range(M):
        for j in range(N):
            acc: float32 = 0.0
            for b in range(NB):
                blk_a: float32[BS]
                blk_b: float32[BS]
                for e in range(BS):
                    blk_a[e] = A[i, b * BS + e]
                    blk_b[e] = B[j, b * BS + e]
                qa: Ty = mx_quantize_block_f32[Ty, BS](blk_a)
                qb: Ty = mx_quantize_block_f32[Ty, BS](blk_b)
                acc += mx_block_dot[Ty, BS](qa, qb)
            Z[i, j] = acc
    return Z


def mx_linear2d_ref[
    Ty, M, N, K
](X: "float32[M, K]", W: "float32[N, K]", bias: "float32[N]") -> "float32[M, N]":
    Z = mx_matmul_ref[Ty, M, N, K](X, W)
    for i in range(M):
        for j in range(N):
            Z[i, j] = Z[i, j] + bias[j]
    return Z


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

    s = allo.customize(mx_matmul_ref, instantiate=[Ty, M, N, K])
    mod = s.build(target="llvm")
    Z = mod(A, B)

    ref = A.astype(np.float64) @ B.astype(np.float64).T
    _assert_close(Z, ref, "mx_matmul_ref")


def test_mx_linear2d():
    M, N, K = 4, 6, 64
    rng = np.random.default_rng(1)
    X = _rand((M, K), rng)
    W = _rand((N, K), rng)
    bias = _rand((N,), rng)

    s = allo.customize(nn.mx_linear2d, instantiate=[Ty, float32, M, N, K])
    mod = s.build(target="llvm")
    Z = mod(X, mx_pack_np(Ty, W), bias)

    ref = X.astype(np.float64) @ W.astype(np.float64).T + bias.astype(np.float64)
    _assert_close(Z, ref, "mx_linear2d")

    s_ref = allo.customize(mx_linear2d_ref, instantiate=[Ty, M, N, K])
    np.testing.assert_array_equal(Z, s_ref.build(target="llvm")(X, W, bias))

    def kernel(
        X: "float32[M, K]", W: "float32[N, K]", bias: "float32[N]"
    ) -> "float32[M, N]":
        Wq = nn.mx_quantize_rows[Ty, float32, N, K](W)
        return nn.mx_linear2d[Ty, float32, M, N, K](X, Wq, bias)

    s_k = allo.customize(kernel)
    np.testing.assert_array_equal(Z, s_k.build(target="llvm")(X, W, bias))


def test_mx_linear3d():
    B, L, D, M = 2, 3, 64, 5
    rng = np.random.default_rng(2)
    X = _rand((B, L, D), rng)
    W = _rand((M, D), rng)
    bias = _rand((M,), rng)

    inst = [Ty, float32, B, L, D, M]
    Wq = mx_pack_np(Ty, W)
    Z = allo.customize(nn.mx_linear3d, instantiate=inst).build(target="llvm")(
        X, Wq, bias
    )
    # the B x L rows through one mx_linear2d
    Z2 = allo.customize(nn.mx_linear2d, instantiate=[Ty, float32, B * L, M, D]).build(
        target="llvm"
    )(X.reshape(B * L, D), Wq, bias)
    np.testing.assert_array_equal(Z, Z2.reshape(B, L, M))
    ref = np.einsum("bld,md->blm", X.astype(np.float64), W.astype(np.float64))
    _assert_close(Z, ref + bias.astype(np.float64), "mx_linear3d")

    s = nn.schedule_mx_linear3d(allo.customize(nn.mx_linear3d, instantiate=inst))
    code = str(s.build(target="vhls"))
    assert code.count("#pragma HLS dataflow") >= 2  # mx_linear3d, mx_linear2d
    assert code.count("hls::stream<") >= 4  # X2, Z2, and Xq, C in mx_linear2d


def test_mx_linear2d_vs_float():
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
        Wq = nn.mx_quantize_rows[Ty, float32, N, K](W)
        return nn.mx_linear2d[Ty, float32, M, N, K](X, Wq, bias)

    mod_mx = allo.customize(kernel_mx).build(target="llvm")
    Z_mx = mod_mx(X, W, bias)
    _assert_close(Z_mx, ref, "nn.mx_linear2d[mxint8]")


# element width / block size other than mxint8's 8 / 32
CUSTOM_MX = [T.MXInt(4, 16), T.MXInt(6, 32), T.MXInt(8, 64), T.MXInt(12, 16)]


@pytest.mark.parametrize("Tc", CUSTOM_MX, ids=lambda t: t.name)
def test_mx_custom_format_linear(Tc):
    M, N, K = 3, 8, 4 * Tc.block_size
    rng = np.random.default_rng(7)
    X = _rand((M, K), rng)
    W = _rand((N, K), rng)
    bias = _rand((N,), rng)
    ref = allo.customize(mx_linear2d_ref, instantiate=[Tc, M, N, K]).build(
        target="llvm"
    )(X, W, bias)

    def lin_ff(
        X: "float32[M, K]", W: "float32[N, K]", bias: "float32[N]"
    ) -> "float32[M, N]":
        Wq = nn.mx_quantize_rows[Tc, float32, N, K](W)
        return nn.mx_linear2d[Tc, float32, M, N, K](X, Wq, bias)

    assert np.abs(mx_quantize(Tc, W)[0]).max() <= Tc.max_int
    build = lambda f: allo.customize(f).build(target="llvm")
    np.testing.assert_array_equal(build(lin_ff)(X, W, bias), ref)


MXFP_FORMATS = [
    T.mxfp8_e4m3,
    T.mxfp8_e5m2,
    T.mxfp6_e2m3,
    T.mxfp6_e3m2,
    T.mxfp4_e2m1,
    T.MXFP(4, 3, 16),  # non-default block size
]
_ML_DTYPES = {
    (4, 3): "float8_e4m3fn",
    (5, 2): "float8_e5m2",
    (2, 3): "float6_e2m3fn",
    (3, 2): "float6_e3m2fn",
    (2, 1): "float4_e2m1fn",
}


def _mxfp_decode(Ty, elems, scales):
    """Dequantizes MXFP (bit-pattern elements, e8m0 scales) to float64."""
    m, eb = Ty.mantissa_bits, Ty.elem_bits
    codes = np.asarray(elems, dtype=np.int64) & ((1 << eb) - 1)
    exp = (codes >> m) & ((1 << Ty.exp_bits) - 1)
    mant = codes & ((1 << m) - 1)
    mag = np.where(
        exp == 0,
        mant * 2.0 ** (1 - Ty.bias - m),
        (1 + mant / 2.0**m) * 2.0 ** (exp - Ty.bias),
    )
    vals = np.where(codes >> (eb - 1), -mag, mag)
    scale = 2.0 ** (np.repeat(scales.astype(np.int64), Ty.block_size, axis=1) - 127)
    return vals * scale


@pytest.mark.parametrize("Tf", MXFP_FORMATS, ids=lambda t: t.name)
def test_mxfp_quantize_matches_ocp(Tf):
    ml_dtypes = pytest.importorskip("ml_dtypes")
    dt = getattr(ml_dtypes, _ML_DTYPES[(Tf.exp_bits, Tf.mantissa_bits)], None)
    if dt is None:
        pytest.skip("ml_dtypes has no matching element type")
    R, BS = 4, Tf.block_size
    K = 4 * BS
    X = _rand((R, K), np.random.default_rng(9))
    elems, scales = mx_quantize(Tf, X)

    # OCP MX v1.0: shared exponent = floor(log2(amax)) - emax; elements are
    # x / 2^shared rounded to nearest even, saturated to the largest finite value
    blocks = X.reshape(R, K // BS, BS).astype(np.float64)
    shared = np.frexp(np.abs(blocks).max(axis=2))[1] - 1 - Tf.max_unbiased_exp
    np.testing.assert_array_equal(scales, shared + 127)
    scale = 2.0 ** shared[..., None]
    max_val = _mxfp_decode(Tf, np.full((1, BS), Tf.max_code), np.full((1, 1), 127))
    ref = np.clip(blocks / scale, -max_val[0, 0], max_val[0, 0]).astype(dt)
    ref = (ref.astype(np.float64) * scale).reshape(R, K)
    np.testing.assert_array_equal(_mxfp_decode(Tf, elems, scales), ref)


@pytest.mark.parametrize("Tf", MXFP_FORMATS, ids=lambda t: t.name)
def test_mxfp_linear(Tf):
    M, N, K = 3, 8, 2 * Tf.block_size
    rng = np.random.default_rng(10)
    X = _rand((M, K), rng)
    W = _rand((N, K), rng)
    bias = _rand((N,), rng)

    def lin_ff(
        X: "float32[M, K]", W: "float32[N, K]", bias: "float32[N]"
    ) -> "float32[M, N]":
        Wq = nn.mx_quantize_rows[Tf, float32, N, K](W)
        return nn.mx_linear2d[Tf, float32, M, N, K](X, Wq, bias)

    build = lambda f: allo.customize(f).build(target="llvm")
    Z = build(lin_ff)(X, W, bias)
    Xq, Xs = mx_quantize(Tf, X)
    Wq, Ws = mx_quantize(Tf, W)

    # block dots are exact; only the float32 sums across blocks round
    ref = _mxfp_decode(Tf, Xq, Xs) @ _mxfp_decode(Tf, Wq, Ws).T + bias
    assert np.linalg.norm(Z - ref) / np.linalg.norm(ref) < 1e-5


# ---- nn.mx_quantize_rows / nn.mx_dequantize round trip ----


@pytest.mark.parametrize(
    "Td", [T.mxint8, T.MXInt(4, 16), T.mxfp8_e4m3, T.mxfp4_e2m1], ids=lambda t: t.name
)
def test_mx_api_dequantize(Td):
    from allo.ir.types import float32

    R, K = 3, 2 * Td.block_size
    X = _rand((R, K), np.random.default_rng(15))

    def roundtrip(X: "float32[R, K]") -> "float32[R, K]":
        Q = nn.mx_quantize_rows[Td, float32, R, K](X)
        return nn.mx_dequantize[Td, R, K](Q)

    got = allo.customize(roundtrip).build(target="llvm")(X)
    elems, scales = mx_quantize(Td, X)
    if Td.is_float:
        ref = _mxfp_decode(Td, elems, scales)
    else:
        scale = np.repeat(scales.astype(np.int64), Td.block_size, axis=1) - 127
        ref = elems.astype(np.float64) * 2.0**scale
    np.testing.assert_array_equal(got, ref.astype(np.float32))


if __name__ == "__main__":
    pytest.main([__file__])
