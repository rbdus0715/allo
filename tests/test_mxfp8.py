import numpy as np
import pytest
from allo.library.mxfp8 import (
    decode_e4m3, encode_e4m3,
    decode_e8m0, encode_e8m0,
    decode_block, encode_block, block_add,
    MXFP8_BLOCK_SIZE,
)

def test_e4m3_roundtrip():
    for f in [0.0, 0.5, 1.0, -2.0, 3.5]:
        u = encode_e4m3(f)
        assert abs(decode_e4m3(u) - f) < 0.5 

def test_e8m0_power_of_two():
    assert decode_e8m0(encode_e8m0(4.0)) == pytest.approx(4.0)

def test_block_add():
    a = np.random.randn(32).astype(np.float32)
    b = np.random.randn(32).astype(np.float32)
    sa, da = encode_block(a)
    sb, db = encode_block(b)
    so, do = block_add(sa, da, sb, db)
    out = decode_block(so, do)
    assert np.allclose(out, a + b, atol=0.5)