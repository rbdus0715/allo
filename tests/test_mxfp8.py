# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import allo
from allo.library import mxfp8
from allo.library.mxfp8 import (
    MXFP8_BLOCK_SIZE,
    ref_decode_e4m3,
    ref_encode_e4m3,
    ref_decode_e8m0,
    ref_encode_e8m0,
    ref_decode_block,
    ref_encode_block,
    ref_block_add,
)


def test_e4m3_roundtrip():
    for f in [0.0, 0.5, 1.0, -2.0, 3.5]:
        u = ref_encode_e4m3(f)
        assert abs(ref_decode_e4m3(u) - f) < 0.5


def test_e8m0_power_of_two():
    assert ref_decode_e8m0(ref_encode_e8m0(4.0)) == pytest.approx(4.0)


def test_block_add_ref():
    a = np.random.randn(32).astype(np.float32)
    b = np.random.randn(32).astype(np.float32)
    sa, da = ref_encode_block(a)
    sb, db = ref_encode_block(b)
    so, do = ref_block_add(sa, da, sb, db)
    out = ref_decode_block(so, do)
    assert np.allclose(out, a + b, atol=0.5)


def test_decode_e4m3_kernel():
    s = allo.customize(mxfp8.decode_e4m3)
    mod = s.build()
    for f in [0.0, 0.5, 1.0, -2.0, 3.5]:
        u = ref_encode_e4m3(f)
        assert abs(mod(int(u)) - ref_decode_e4m3(u)) < 1e-5


def test_encode_e4m3_kernel():
    s = allo.customize(mxfp8.encode_e4m3)
    mod = s.build()
    for f in [0.0, 0.5, 1.0, -2.0, 3.5]:
        u = mod(float(f))
        assert abs(ref_decode_e4m3(u) - f) < 0.5


def test_mxfp8_block_add_kernel():
    bs = MXFP8_BLOCK_SIZE
    a = np.random.randn(bs).astype(np.float32)
    b = np.random.randn(bs).astype(np.float32)
    sa, da = ref_encode_block(a)
    sb, db = ref_encode_block(b)

    s = allo.customize(mxfp8.mxfp8_block_add, instantiate=[bs])
    mod = s.build()
    so = np.zeros(1, dtype=np.uint8)
    dout = np.zeros(bs, dtype=np.uint8)
    mod(int(sa), da, int(sb), db, so, dout)

    out = ref_decode_block(int(so[0]), dout)
    assert np.allclose(out, a + b, atol=0.5)


def test_mxfp8_block_add_vhls():
    bs = MXFP8_BLOCK_SIZE
    s = allo.customize(mxfp8.mxfp8_block_add, instantiate=[bs])
    hls_mod = s.build(target="vhls")
    assert "void mxfp8_block_add" in hls_mod.hls_code
    assert "uint8_t" in hls_mod.hls_code
    assert "float" in hls_mod.hls_code
