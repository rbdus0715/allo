# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# nn.mx_linear2d / nn.mx_matmul / nn.mx_quantize_cols and bfloat16 activations,
# on the LLVM backend.

import numpy as np
import pytest
import allo
import allo.ir.types as T
from allo.ir.types import float32, bfloat16
import allo.library.nn as nn
from test_mxint8_nn import mx_linear2d_ref
from allo.library.mx import mx_pack_np, mx_pack_channels_np, mx_unpack_np

M, N, K = 4, 16, 128

# MX element formats: integer and float elements, block sizes 32 and 16
FORMATS = [T.MXInt(8, 32), T.MXInt(4, 16), T.MXFP(4, 3, 32), T.MXFP(2, 1, 32)]


@pytest.mark.parametrize(
    "Tf, dims",
    [(FORMATS[0], (16, 64, 96))] + [(t, (M, N, K)) for t in FORMATS],
    ids=lambda x: getattr(x, "name", str(x)),
)
def test_mx_linear2d(Tf, dims):
    # X quantized on chip, W on the host: equals mx_linear2d_ref
    m, n, k = dims
    rng = np.random.default_rng(1)
    rand = lambda shape: (
        rng.standard_normal(shape) * 2.0 ** rng.integers(-4, 4, shape)
    ).astype(np.float32)
    X, W, bias = rand((m, k)), rand((n, k)), rand((n,))
    inst = [Tf, float32, m, n, k]
    Z = allo.customize(nn.mx_linear2d, instantiate=inst).build(target="llvm")(
        X, mx_pack_np(Tf, W), bias
    )
    ref = allo.customize(mx_linear2d_ref, instantiate=[Tf, m, n, k]).build(
        target="llvm"
    )(X, W, bias)
    np.testing.assert_array_equal(Z, ref)

    s = allo.customize(nn.mx_linear2d, instantiate=inst)
    code = str(nn.schedule_mx_linear2d(s).build(target="vhls"))
    assert "#pragma HLS pipeline" in code


@pytest.mark.parametrize("Tf", FORMATS, ids=lambda t: t.name)
def test_mx_matmul(Tf):
    # A @ B: A blocked along its rows, B down its columns (rows of B^T)
    m, k, n = 3, 40, 5
    rng = np.random.default_rng(3)
    A = rng.standard_normal((m, k)).astype(np.float32)
    B = rng.standard_normal((k, n)).astype(np.float32)
    bias = rng.standard_normal(n).astype(np.float32)
    Aq, Bq = mx_pack_np(Tf, A), mx_pack_np(Tf, B.T)
    C = allo.customize(nn.mx_matmul, instantiate=[Tf, m, n, k]).build(target="llvm")(
        Aq, Bq
    )
    # mx_linear2d quantizes A itself, then the same matmul and the bias
    Z = allo.customize(nn.mx_linear2d, instantiate=[Tf, float32, m, n, k]).build(
        target="llvm"
    )(A, Bq, bias)
    np.testing.assert_array_equal(Z, C + bias)
    rel = np.linalg.norm(C - A @ B) / np.linalg.norm(A @ B)
    assert rel < (0.1 if Tf.elem_bits >= 8 else 0.5)


@pytest.mark.parametrize("Tf", FORMATS, ids=lambda t: t.name)
def test_mx_quantize_cols(Tf):
    # blocks down the columns of a [K, R] array == host packing of its transpose
    k, r = 40, 3
    X = np.random.default_rng(2).standard_normal((k, r)).astype(np.float32)
    Q = allo.customize(nn.mx_quantize_cols, instantiate=[Tf, float32, r, k]).build(
        target="llvm"
    )(X)
    np.testing.assert_array_equal(
        mx_unpack_np(Tf, Q), mx_unpack_np(Tf, mx_pack_np(Tf, X.T))
    )


@pytest.mark.parametrize("Tf", FORMATS[:2], ids=lambda t: t.name)
def test_mx_layers_bfloat16(Tf):
    # bfloat16 activations (LLVM only: the HLS emitter has no bf16 type):
    # the float32 path on the same values, rounded to bfloat16
    import ml_dtypes

    bf = ml_dtypes.bfloat16
    rng = np.random.default_rng(4)
    run = lambda fn, inst, *args: allo.customize(fn, instantiate=inst).build(
        target="llvm"
    )(*args)
    same = lambda z16, z32: np.array_equal(
        np.asarray(z16).view(np.uint16), z32.astype(bf).view(np.uint16)
    )
    X, b = rng.standard_normal((4, 100)).astype(bf), rng.standard_normal(16).astype(bf)
    Wq = mx_pack_np(Tf, rng.standard_normal((16, 100)).astype(np.float32))
    z16 = run(nn.mx_linear2d, [Tf, bfloat16, 4, 16, 100], X, Wq, b)
    z32 = run(
        nn.mx_linear2d,
        [Tf, float32, 4, 16, 100],
        X.astype(np.float32),
        Wq,
        b.astype(np.float32),
    )
    assert same(z16, z32)
    conv = [1, 40, 8, 6, 6, 3, 3, 6, 6, 1, 1, 1, 1]
    x, cb = rng.standard_normal((1, 40, 6, 6)).astype(bf), rng.standard_normal(
        8
    ).astype(bf)
    wq = mx_pack_channels_np(Tf, rng.standard_normal((8, 40, 3, 3)).astype(np.float32))
    c16 = run(nn.mx_conv2d, [Tf, bfloat16, *conv], x, wq, cb)
    c32 = run(
        nn.mx_conv2d,
        [Tf, float32, *conv],
        x.astype(np.float32),
        wq,
        cb.astype(np.float32),
    )
    assert same(c16, c32)


if __name__ == "__main__":
    pytest.main([__file__])
