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
    mx_hbm_mapping,
)
from allo.library.mxint8 import mx_quantize, mx_pack_words

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
        return nn.mx_linear2d[Ty, M, N, K](X, W, bias)

    mod_mx = allo.customize(kernel_mx).build(target="llvm")
    Z_mx = mod_mx(X, W, bias)
    _assert_close(Z_mx, ref, "nn.mx_linear2d[mxint8]")


@pytest.mark.parametrize("mode", ["q", "wq", "ff"])
def test_mx_gemm_compose_dataflow(mode):
    from allo.ir.types import float32, int8, e8m0

    M, N, K = 4, 16, 64
    NB = K // Ty.block_size
    inst = [Ty, M, N, K, NB, 0]

    def k_q(
        Xq: "int8[M, K]",
        Xs: "e8m0[M, NB]",
        Wq: "int8[N, K]",
        Ws: "e8m0[N, NB]",
        bias: "float32[N]",
    ) -> "float32[M, N]":
        return nn.mx_gemm_q[Ty, M, N, K, NB, 0](Xq, Xs, Wq, Ws, bias)

    def k_wq(
        X: "float32[M, K]", Wq: "int8[N, K]", Ws: "e8m0[N, NB]", bias: "float32[N]"
    ) -> "float32[M, N]":
        return nn.mx_gemm_wq[Ty, M, N, K, NB, 0](X, Wq, Ws, bias)

    def k_ff(
        X: "float32[M, K]", W: "float32[N, K]", bias: "float32[N]"
    ) -> "float32[M, N]":
        return nn.mx_gemm_ff[Ty, M, N, K, NB, 0](X, W, bias)

    s = allo.customize({"q": k_q, "wq": k_wq, "ff": k_ff}[mode])
    s.compose(getattr(nn, f"mx_gemm_{mode}"), instantiate=inst)
    code = str(s.build(target="vhls"))
    assert "#pragma HLS dataflow" in code
    assert "#pragma HLS stream" in code  # qx: X blocks -> block dot


def test_mx_linear_chain_dataflow():
    from allo.ir.types import float32, int8, e8m0

    M, K, N1, N2 = 4, 64, 32, 16
    NB1, NB2 = K // Ty.block_size, N1 // Ty.block_size
    rng = np.random.default_rng(11)
    X = _rand((M, K), rng)
    W1, b1 = _rand((N1, K), rng), _rand((N1,), rng)
    W2, b2 = _rand((N2, N1), rng), _rand((N2,), rng)
    W1q, W1s = mx_quantize(Ty, W1)
    W2q, W2s = mx_quantize(Ty, W2)

    def model(
        X: "float32[M, K]",
        W1q: "int8[N1, K]",
        W1s: "e8m0[N1, NB1]",
        b1: "float32[N1]",
        W2q: "int8[N2, N1]",
        W2s: "e8m0[N2, NB2]",
        b2: "float32[N2]",
    ) -> "float32[M, N2]":
        a = nn.mx_gemm_wq[Ty, M, N1, K, NB1, 0](X, W1q, W1s, b1)
        return nn.mx_gemm_wq[Ty, M, N2, N1, NB2, 0](a, W2q, W2s, b2)

    # numerics first: streams are HLS-only
    Z = allo.customize(model).build(target="llvm")(X, W1q, W1s, b1, W2q, W2s, b2)
    lin = lambda *d: allo.customize(nn.mx_gemm_wq, instantiate=[Ty, *d, 0]).build(
        target="llvm"
    )
    a = lin(M, N1, K, NB1)(X, W1q, W1s, b1)
    np.testing.assert_array_equal(Z, lin(M, N2, N1, NB2)(a, W2q, W2s, b2))

    # each layer: X blocks stream into the dot; the two layers run as a dataflow
    s = allo.customize(model)
    s.compose(nn.mx_gemm_wq, instantiate=[Ty, M, N1, K, NB1, 0])
    s.compose(nn.mx_gemm_wq, id="1", instantiate=[Ty, M, N2, N1, NB2, 0])
    s.dataflow("model")
    code = str(s.build(target="vhls"))
    assert code.count("#pragma HLS dataflow") >= 3  # model + one per layer


@pytest.mark.parametrize("M, N, K", [(1, 16, 64), (4, 16, 128), (5, 12, 96)])
def test_mx_gemm_variants(M, N, K):
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
    build = lambda f, *t: allo.customize(
        f, instantiate=[Ty, M, N, K, NB, *t]
    ).build(target="llvm")
    np.testing.assert_array_equal(build(nn.mx_gemm_wq, 0)(X, Wq, Ws, bias), ref_lin)
    np.testing.assert_array_equal(build(nn.mx_gemm_q, 0)(Xq, Xs, Wq, Ws, bias), ref_lin)
    np.testing.assert_array_equal(build(nn.mx_gemm_ff, 0)(X, W, bias), ref_lin)
    np.testing.assert_array_equal(build(nn.mx_matmul_wq)(X, Bq, Bs), ref_mm)
    np.testing.assert_array_equal(build(nn.mx_matmul_q)(Xq, Xs, Bq, Bs), ref_mm)
    np.testing.assert_array_equal(build(nn.mx_matmul_ff)(X, B), ref_mm)


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


# element width / block size other than mxint8's 8 / 32
CUSTOM_MX = [T.MXInt(4, 16), T.MXInt(6, 32), T.MXInt(8, 64), T.MXInt(12, 16)]


@pytest.mark.parametrize("Tc", CUSTOM_MX, ids=lambda t: t.name)
def test_mx_custom_format_linear(Tc):
    from allo.ir.types import float32, Int, e8m0

    M, N, K = 3, 8, 4 * Tc.block_size
    NB, EB = K // Tc.block_size, Tc.elem_bits
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
        return nn.mx_linear2d[Tc, M, N, K](X, W, bias)

    def lin_q(
        Xq: "Int(EB)[M, K]",
        Xs: "e8m0[M, NB]",
        Wq: "Int(EB)[N, K]",
        Ws: "e8m0[N, NB]",
        bias: "float32[N]",
    ) -> "float32[M, N]":
        return nn.mx_gemm_q[Tc, M, N, K, NB, 0](Xq, Xs, Wq, Ws, bias)

    Xq, Xs = mx_quantize(Tc, X)
    Wq, Ws = mx_quantize(Tc, W)
    assert np.abs(Wq).max() <= Tc.max_int
    build = lambda f: allo.customize(f).build(target="llvm")
    np.testing.assert_array_equal(build(lin_ff)(X, W, bias), ref)
    np.testing.assert_array_equal(build(lin_q)(Xq, Xs, Wq, Ws, bias), ref)


@pytest.mark.parametrize("Tc", [T.MXInt(4, 16), T.MXInt(6, 64)], ids=lambda t: t.name)
def test_mx_custom_format_dataflow(Tc):
    M, N, K, P = 4, 6, 2 * Tc.block_size, 2
    rng = np.random.default_rng(8)
    X = _rand((M, K), rng)
    W = _rand((N, K), rng)
    bias = _rand((N,), rng)
    X_w = mx_pack_words(X, 512)

    s_ref = make_mx_linear2d_dataflow(Tc, M, N, K, P)
    Z_ref = np.zeros(M * N, dtype=np.float32)
    LLVMOMPModule(s_ref.module, s_ref.top_func_name)(
        X_w, mx_pack_words(W, 512), bias, Z_ref
    )
    ref = allo.customize(mx_linear2d_ref, instantiate=[Tc, M, N, K]).build(
        target="llvm"
    )(X, W, bias)
    np.testing.assert_array_equal(Z_ref.reshape(M, N), ref)


def test_mx_hbm_mapping():
    s = make_mx_linear2d_dataflow(Ty, 8, 6, 64, 2)
    assert mx_hbm_mapping(s) == {
        "X": "HBM[0]",
        "W": "HBM[1]",
        "bias": "HBM[2]",
        "Z": "HBM[2]",
    }
    assert mx_hbm_mapping(s, base=4)["W"] == "HBM[5]"
    assert mx_hbm_mapping(s, memory="DDR")["X"] == "DDR[0]"  # Alveo U250


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])


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
    from allo.ir.types import float32, Int, e8m0

    M, N, K = 3, 8, 2 * Tf.block_size
    NB, EB = K // Tf.block_size, Tf.elem_bits
    rng = np.random.default_rng(10)
    X = _rand((M, K), rng)
    W = _rand((N, K), rng)
    bias = _rand((N,), rng)

    def lin_ff(
        X: "float32[M, K]", W: "float32[N, K]", bias: "float32[N]"
    ) -> "float32[M, N]":
        return nn.mx_linear2d[Tf, M, N, K](X, W, bias)

    def lin_q(
        Xq: "Int(EB)[M, K]",
        Xs: "e8m0[M, NB]",
        Wq: "Int(EB)[N, K]",
        Ws: "e8m0[N, NB]",
        bias: "float32[N]",
    ) -> "float32[M, N]":
        return nn.mx_gemm_q[Tf, M, N, K, NB, 0](Xq, Xs, Wq, Ws, bias)

    build = lambda f: allo.customize(f).build(target="llvm")
    Z = build(lin_ff)(X, W, bias)
    Xq, Xs = mx_quantize(Tf, X)
    Wq, Ws = mx_quantize(Tf, W)
    np.testing.assert_array_equal(build(lin_q)(Xq, Xs, Wq, Ws, bias), Z)

    # block dots are exact; only the float32 sums across blocks round
    ref = _mxfp_decode(Tf, Xq, Xs) @ _mxfp_decode(Tf, Wq, Ws).T + bias
    assert np.linalg.norm(Z - ref) / np.linalg.norm(ref) < 1e-5


# ---- MX matrix API: nn.mx_quantize / mx_pack / mx_dot / mx_dequantize ----


def test_mx_api_linear():
    from allo.ir.types import float32, int8, e8m0

    M, N, K = 4, 8, 64
    NB = K // Ty.block_size
    rng = np.random.default_rng(12)
    X, W, b = _rand((M, K), rng), _rand((N, K), rng), _rand((N,), rng)
    R = _rand((M, N), rng)
    Wq, Ws = mx_quantize(Ty, W)

    def user_linear(
        X: "float32[M, K]", Wq: "int8[N, K]", Ws: "e8m0[N, NB]", b: "float32[N]"
    ) -> "float32[M, N]":
        qx = nn.mx_quantize[Ty, M, K](X)
        qw = nn.mx_pack[Ty, N, K](Wq, Ws)
        Z: float32[M, N]
        for i, j in allo.grid(M, N):
            Z[i, j] = nn.mx_dot[Ty, NB](qx[i], qw[j]) + b[j]
        return Z

    def user_block(
        X: "float32[M, K]",
        Wq: "int8[N, K]",
        Ws: "e8m0[N, NB]",
        b: "float32[N]",
        R: "float32[M, N]",
    ) -> "float32[M, N]":
        qx = nn.mx_quantize[Ty, M, K](X)
        qw = nn.mx_pack[Ty, N, K](Wq, Ws)
        Z: float32[M, N]
        for i, j in allo.grid(M, N):
            y: float32 = nn.mx_dot[Ty, NB](qx[i], qw[j]) + b[j]
            Z[i, j] = max(y, 0.0) + R[i, j]
        return Z

    ref = allo.customize(nn.mx_gemm_wq, instantiate=[Ty, M, N, K, NB, 0]).build(
        target="llvm"
    )(X, Wq, Ws, b)
    Z = allo.customize(user_linear).build(target="llvm")(X, Wq, Ws, b)
    np.testing.assert_array_equal(Z, ref)
    Zb = allo.customize(user_block).build(target="llvm")(X, Wq, Ws, b, R)
    np.testing.assert_array_equal(Zb, np.maximum(ref, 0.0) + R)

    s = allo.customize(user_linear)
    nn.schedule_mx(s)
    s.pipeline("user_linear:j")
    code = str(s.build(target="vhls"))
    assert "#pragma HLS unroll" in code and "#pragma HLS pipeline" in code


def test_mx_api_conv2d_im2col():
    from allo.ir.types import float32, int8, e8m0

    Cin, H, W_, Kh, Kw, Cout = 2, 6, 6, 4, 4, 3
    Oh, Ow, CKK = H - Kh + 1, W_ - Kw + 1, Cin * Kh * Kw  # CKK = 32: one block
    NBc = CKK // Ty.block_size
    rng = np.random.default_rng(13)
    inp = _rand((Cin, H, W_), rng)
    Wf = _rand((Cout, Cin, Kh, Kw), rng)
    bias = _rand((Cout,), rng)
    Wq, Ws = mx_quantize(Ty, Wf.reshape(Cout, -1))

    def mx_conv2d(
        inp: "float32[Cin, H, W_]",
        Wq: "int8[Cout, CKK]",
        Ws: "e8m0[Cout, NBc]",
        bias: "float32[Cout]",
    ) -> "float32[Cout, Oh, Ow]":
        qw = nn.mx_pack[Ty, Cout, CKK](Wq, Ws)
        Y: float32[Cout, Oh, Ow]
        for oh, ow in allo.grid(Oh, Ow):
            patch: float32[1, CKK]
            for c, kh, kw in allo.grid(Cin, Kh, Kw):
                patch[0, (c * Kh + kh) * Kw + kw] = inp[c, oh + kh, ow + kw]
            qp = nn.mx_quantize[Ty, 1, CKK](patch)
            for co in range(Cout):
                Y[co, oh, ow] = nn.mx_dot[Ty, NBc](qp[0], qw[co]) + bias[co]
        return Y

    Y = allo.customize(mx_conv2d).build(target="llvm")(inp, Wq, Ws, bias)
    # reference: the same blocks through the GEMM, on host-side im2col patches
    patches = [
        inp[:, oh : oh + Kh, ow : ow + Kw].reshape(-1)
        for oh in range(Oh)
        for ow in range(Ow)
    ]
    P = np.stack(patches).astype(np.float32)
    gemm = allo.customize(
        nn.mx_gemm_wq, instantiate=[Ty, Oh * Ow, Cout, CKK, NBc, 0]
    ).build(target="llvm")
    np.testing.assert_array_equal(Y, gemm(P, Wq, Ws, bias).T.reshape(Cout, Oh, Ow))


def test_mx_api_attention_scores():
    from allo.ir.types import float32

    L, D = 8, 64
    NB = D // Ty.block_size
    c = np.float32(1.0 / np.sqrt(D))
    rng = np.random.default_rng(14)
    Q, K_ = _rand((L, D), rng), _rand((L, D), rng)

    def scores(Q: "float32[L, D]", K_: "float32[L, D]") -> "float32[L, L]":
        qq = nn.mx_quantize[Ty, L, D](Q)
        qk = nn.mx_quantize[Ty, L, D](K_)
        S: float32[L, L]
        for i, j in allo.grid(L, L):
            S[i, j] = nn.mx_dot[Ty, NB](qq[i], qk[j]) * c
        return S

    S = allo.customize(scores).build(target="llvm")(Q, K_)
    ref = allo.customize(nn.mx_gemm_ff, instantiate=[Ty, L, L, D, NB, 0]).build(
        target="llvm"
    )(Q, K_, np.zeros(L, np.float32))
    np.testing.assert_array_equal(S, ref * c)


@pytest.mark.parametrize(
    "Td", [T.mxint8, T.MXInt(4, 16), T.mxfp8_e4m3, T.mxfp4_e2m1], ids=lambda t: t.name
)
def test_mx_api_dequantize(Td):
    from allo.ir.types import float32

    R, K = 3, 2 * Td.block_size
    X = _rand((R, K), np.random.default_rng(15))

    def roundtrip(X: "float32[R, K]") -> "float32[R, K]":
        return nn.mx_dequantize[Td, R, K](nn.mx_quantize[Td, R, K](X))

    got = allo.customize(roundtrip).build(target="llvm")(X)
    elems, scales = mx_quantize(Td, X)
    if Td.is_float:
        ref = _mxfp_decode(Td, elems, scales)
    else:
        scale = np.repeat(scales.astype(np.int64), Td.block_size, axis=1) - 127
        ref = elems.astype(np.float64) * 2.0**scale
    np.testing.assert_array_equal(got, ref.astype(np.float32))


@pytest.mark.parametrize(
    "B, Cin, Cout, H, K, S, P",
    [(1, 2, 3, 6, 4, 1, 0), (2, 3, 4, 7, 3, 2, 1)],  # Cin*K*K = 32, and 27 -> 32
)
def test_mx_conv2d(B, Cin, Cout, H, K, S, P):
    Oh = (H + 2 * P - K) // S + 1
    BS = Ty.block_size
    NBc = (Cin * K * K + BS - 1) // BS
    CKK = NBc * BS
    rng = np.random.default_rng(16)
    inp = _rand((B, Cin, H, H), rng)
    kernel = _rand((Cout, Cin, K, K), rng)
    bias = _rand((Cout,), rng)
    inst = [Ty, B, Cin, Cout, H, H, K, K, Oh, Oh, S, S, P, P]
    Z = allo.customize(nn.mx_conv2d, instantiate=inst).build(target="llvm")(
        inp, kernel, bias
    )

    # reference: host-side im2col (zero-padded to CKK) through the MX GEMM
    padded = np.pad(inp, ((0, 0), (0, 0), (P, P), (P, P)))
    patches = np.zeros((B * Oh * Oh, CKK), np.float32)
    for r, (n, oh, ow) in enumerate(np.ndindex(B, Oh, Oh)):
        win = padded[n, :, oh * S : oh * S + K, ow * S : ow * S + K]
        patches[r, : Cin * K * K] = win.reshape(-1)
    kflat = np.zeros((Cout, CKK), np.float32)
    kflat[:, : Cin * K * K] = kernel.reshape(Cout, -1)
    gemm = allo.customize(
        nn.mx_gemm_ff, instantiate=[Ty, B * Oh * Oh, Cout, CKK, NBc, 0]
    ).build(target="llvm")
    ref = gemm(patches, kflat, bias).reshape(B, Oh, Oh, Cout).transpose(0, 3, 1, 2)
    np.testing.assert_array_equal(Z, ref)

    exact = (patches.astype(np.float64) @ kflat.T.astype(np.float64) + bias).reshape(
        B, Oh, Oh, Cout
    )
    _assert_close(Z, exact.transpose(0, 3, 1, 2), "mx_conv2d")

    s = allo.customize(nn.mx_conv2d, instantiate=inst)
    nn.schedule_mx_conv2d(s)
    assert "#pragma HLS pipeline" in str(s.build(target="vhls"))
