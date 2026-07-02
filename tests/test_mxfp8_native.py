# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import allo
from allo import mxfp8_ops
from allo.ir.types import mxfp8, uint8, float32, int32
from allo.library.mxfp8 import (
    MXFP8_BLOCK_SIZE,
    ref_decode_e4m3,
    ref_encode_e4m3,
    ref_decode_e8m0,
    ref_encode_e8m0,
    ref_block_add,
    ref_encode_block,
)


def test_mxfp8_type_roundtrip():
    from allo._mlir.dialects import allo as allo_d
    from allo._mlir.ir import Context
    from allo.utils import register_dialect

    with Context() as ctx:
        register_dialect(ctx)
        ty = mxfp8.build()
        assert allo_d.Mxfp8Type.isinstance(ty)
        assert allo_d.Mxfp8Type(ty).block_size == MXFP8_BLOCK_SIZE


def test_decode_e4m3_native():
    def kernel(u8: uint8) -> float32:
        return mxfp8_ops.decode_e4m3(u8)

    mod = allo.customize(kernel).build()
    for u in [0, 1, 42, 200]:
        assert abs(mod(int(u)) - ref_decode_e4m3(u)) < 1e-4


def test_encode_e4m3_native():
    def kernel(f: float32) -> uint8:
        return mxfp8_ops.encode_e4m3(f)

    mod = allo.customize(kernel).build()
    for f in [-2.0, -0.5, 0.0, 0.25, 1.0, 3.5]:
        u = mod(float(f))
        assert abs(ref_decode_e4m3(u) - f) < 0.5


def test_block_add_native():
    bs = MXFP8_BLOCK_SIZE

    def kernel[BS: int32](
        scale_a: uint8,
        data_a: uint8[BS],
        scale_b: uint8,
        data_b: uint8[BS],
        scale_out: uint8[1],
        data_out: uint8[BS],
    ):
        mxfp8_ops.block_add_mxfp8(
            scale_a, data_a, scale_b, data_b, scale_out, data_out
        )

    mod = allo.customize(kernel, instantiate=[bs]).build()
    rng = np.random.default_rng(0)
    data1 = rng.standard_normal(bs).astype(np.float32)
    data2 = rng.standard_normal(bs).astype(np.float32)
    s1, d1 = ref_encode_block(data1)
    s2, d2 = ref_encode_block(data2)
    so = np.zeros(1, dtype=np.uint8)
    dout = np.zeros(bs, dtype=np.uint8)
    mod(int(s1), d1, int(s2), d2, so, dout)
    rs, rd = ref_block_add(s1, d1, s2, d2)
    decoded = np.array([ref_decode_e4m3(x) * ref_decode_e8m0(so[0]) for x in dout])
    golden = np.array([ref_decode_e4m3(x) * ref_decode_e8m0(rs) for x in rd])
    assert np.allclose(decoded, golden, atol=0.5)


def test_block_add_native_vhls():
    bs = MXFP8_BLOCK_SIZE

    def kernel[BS: int32](
        scale_a: uint8,
        data_a: uint8[BS],
        scale_b: uint8,
        data_b: uint8[BS],
        scale_out: uint8[1],
        data_out: uint8[BS],
    ):
        mxfp8_ops.block_add_mxfp8(
            scale_a, data_a, scale_b, data_b, scale_out, data_out
        )

    s = allo.customize(kernel, instantiate=[bs])
    hls_mod = s.build(target="vhls")
    assert "allo_block_add_mxfp8" in hls_mod.hls_code
    assert "allo_add_nrm" in hls_mod.hls_code


def test_block_matmul_native():
    bs = MXFP8_BLOCK_SIZE

    def kernel[BS: int32](
        scale_a: uint8,
        data_a: uint8[BS],
        scale_b: uint8,
        data_b: uint8[BS],
        scale_out: uint8[1],
        data_out: uint8[BS],
    ):
        mxfp8_ops.block_matmul_mxfp8(
            scale_a, data_a, scale_b, data_b, scale_out, data_out
        )

    mod = allo.customize(kernel, instantiate=[bs]).build()
    rng = np.random.default_rng(1)
    data1 = rng.standard_normal(bs).astype(np.float32)
    data2 = rng.standard_normal(bs).astype(np.float32)
    s1, d1 = ref_encode_block(data1)
    s2, d2 = ref_encode_block(data2)
    so = np.zeros(1, dtype=np.uint8)
    dout = np.zeros(bs, dtype=np.uint8)
    mod(int(s1), d1, int(s2), d2, so, dout)
    dot = float(np.sum(data1 * data2))
    _, rd = ref_encode_block(np.array([dot], dtype=np.float32))
    decoded = ref_decode_e4m3(dout[0]) * ref_decode_e8m0(so[0])
    golden = ref_decode_e4m3(rd[0]) * ref_decode_e8m0(ref_encode_e8m0(abs(dot)))
    assert abs(decoded - dot) < max(0.5, abs(dot) * 0.5)
