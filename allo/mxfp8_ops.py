# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native MXFP8 compiler intrinsics (lowered to allo.* MLIR ops)."""


def decode_e4m3(_u8):
    raise RuntimeError("decode_e4m3 is a compiler intrinsic")


def encode_e4m3(_f):
    raise RuntimeError("encode_e4m3 is a compiler intrinsic")


def decode_e8m0(_u8):
    raise RuntimeError("decode_e8m0 is a compiler intrinsic")


def encode_e8m0(_scale):
    raise RuntimeError("encode_e8m0 is a compiler intrinsic")


def decode_mxfp8_block(_scale, _data, _out):
    raise RuntimeError("decode_mxfp8_block is a compiler intrinsic")


def encode_mxfp8_block(_data, _scale_out, _data_out):
    raise RuntimeError("encode_mxfp8_block is a compiler intrinsic")


def block_add_mxfp8(_scale_a, _data_a, _scale_b, _data_b, _scale_out, _data_out):
    raise RuntimeError("block_add_mxfp8 is a compiler intrinsic")


def block_matmul_mxfp8(_scale_a, _data_a, _scale_b, _data_b, _scale_out, _data_out):
    raise RuntimeError("block_matmul_mxfp8 is a compiler intrinsic")
