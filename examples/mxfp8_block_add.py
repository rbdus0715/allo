#!/usr/bin/env python3
"""MXFP8 Allo kernel example: simulate on LLVM and emit Vivado HLS C++."""

# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import allo
from allo.library import mxfp8
from allo.library.mxfp8 import MXFP8_BLOCK_SIZE, ref_encode_block, ref_decode_block


def main():
    bs = MXFP8_BLOCK_SIZE
    np.random.seed(0)

    a = np.random.randn(bs).astype(np.float32)
    b = np.random.randn(bs).astype(np.float32)
    sa, da = ref_encode_block(a)
    sb, db = ref_encode_block(b)

    # 1) Software simulation (LLVM)
    s = allo.customize(mxfp8.mxfp8_block_add, instantiate=[bs])
    mod = s.build()
    scale_out = np.zeros(1, dtype=np.uint8)
    data_out = np.zeros(bs, dtype=np.uint8)
    mod(int(sa), da, int(sb), db, scale_out, data_out)

    decoded = ref_decode_block(int(scale_out[0]), data_out)
    print("LLVM simulation result (first 4 elements):", decoded[:4])
    print("Expected a+b (first 4 elements):           ", (a + b)[:4])
    np.testing.assert_allclose(decoded, a + b, atol=0.5)
    print("LLVM simulation passed.")

    # 2) Emit Vivado HLS C++
    hls_mod = s.build(target="vhls")
    print("\n--- Generated HLS (excerpt) ---")
    lines = hls_mod.hls_code.splitlines()
    for line in lines[:40]:
        print(line)
    print("...")
    print(f"Total HLS lines: {len(lines)}")
    print("\nUse schedule_mxfp8_block_add(s) before build() to pipeline loops.")


if __name__ == "__main__":
    main()
