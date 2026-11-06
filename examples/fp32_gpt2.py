# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=unsubscriptable-object, no-value-for-parameter

# The float32 baseline of examples/mx_gpt2.py: the same block and dataflow
# schedule, its linears a float32 block dot of BS elements per cycle (as many
# MACs as an MX block dot) instead of nn.mx_linear2d. Its weights arrive as
# two 512b words per 32 floats (mx_pack_f32_blocks), 32 floats per cycle like
# an MX word.
#   python examples/fp32_gpt2.py                      -> LLVM, vs float64
#   python examples/fp32_gpt2.py --mode csyn|hw_emu   -> one block on HLS

import numpy as np
import allo
import allo.library.nn as nn
from allo.ir.types import float32, int32, ConstExpr, UInt
from allo.ir.utils import MockBuffer
from allo.library.mx import partition_block_lanes, mx_pack_f32_blocks
from mx_gpt2 import L, D, H, F, VOCAB, DH, QKV, SCALE
from mx_gpt2 import init_weights, block_ref, gpt2

BS = 32  # float32 MACs per cycle in each linear


def fp32_linear2d[
    M, N, K
](X: "float32[M, K]", W: "float32[N, K]", bias: "float32[N]") -> "float32[M, N]":
    NB: ConstExpr[int32] = (K + BS - 1) // BS
    Z: float32[M, N]
    acc: float32[N] = 0.0
    for m in range(M):
        for b, n in allo.grid(NB, N):
            d: float32 = 0.0
            with allo.meta_for(BS) as e:
                fx: float32 = 0.0
                if b * BS + e < K:
                    fx = X[m, b * BS + e] * W[n, b * BS + e]
                d = d + fx
            prev: float32 = 0.0
            if b > 0:
                prev = acc[n]
            total: float32 = prev + d
            acc[n] = total
            if b == NB - 1:
                Z[m, n] = total + bias[n]
    return Z


def schedule_fp32_linear2d(s):
    _, _, K = s.inst_list[:3]
    top = s.top_func_name
    for arr in ("X", "W"):
        partition_block_lanes(s, MockBuffer(top, arr), K, BS)
    s.pipeline(f"{top}:n")
    return s


def fork2(X: "float32[L, D]", A: "float32[L, D]", B: "float32[L, D]"):
    for i, j in allo.grid(L, D):
        A[i, j] = X[i, j]
        B[i, j] = X[i, j]


def causal_softmax(S: "float32[L, L]") -> "float32[L, L]":
    P: float32[L, L] = 0.0
    mx: float32[L] = -1e30
    tot: float32[L] = 0.0
    for i, j in allo.grid(L, L):
        if j <= i and S[i, j] * SCALE > mx[i]:
            mx[i] = S[i, j] * SCALE
    for ie, je in allo.grid(L, L):
        if je <= ie:
            P[ie, je] = allo.exp(S[ie, je] * SCALE - mx[ie])
            tot[ie] += P[ie, je]
    for iz, jz in allo.grid(L, L):
        P[iz, jz] = P[iz, jz] / tot[iz]
    return P


def attention(qkv: "float32[L, QKV]") -> "float32[L, D]":
    Z: float32[L, D]
    for h in range(H):
        q: float32[L, DH]
        k: float32[L, DH]
        v: float32[L, DH]
        for r, c in allo.grid(L, DH):
            q[r, c] = qkv[r, h * DH + c]
            k[r, c] = qkv[r, D + h * DH + c]
            v[r, c] = qkv[r, 2 * D + h * DH + c]
        S: float32[L, L]
        for i, j in allo.grid(L, L):
            acc: float32 = 0.0
            for d in range(DH):
                acc += q[i, d] * k[j, d]
            S[i, j] = acc
        P = causal_softmax(S)
        for ip, dp in allo.grid(L, DH):
            accp: float32 = 0.0
            for jp in range(L):
                accp += P[ip, jp] * v[jp, dp]
            Z[ip, h * DH + dp] = accp
    return Z


# `block` is a reserved name in the generated HDL
def block_fp32(
    x: "float32[L, D]",
    ln1_g: "float32[D]",
    ln1_b: "float32[D]",
    w_qkv0: "UInt(512)[QKV, D // 32]",
    w_qkv1: "UInt(512)[QKV, D // 32]",
    b_qkv: "float32[QKV]",
    w_o0: "UInt(512)[D, D // 32]",
    w_o1: "UInt(512)[D, D // 32]",
    b_o: "float32[D]",
    ln2_g: "float32[D]",
    ln2_b: "float32[D]",
    w_fc10: "UInt(512)[F, D // 32]",
    w_fc11: "UInt(512)[F, D // 32]",
    b_fc1: "float32[F]",
    w_fc20: "UInt(512)[D, F // 32]",
    w_fc21: "UInt(512)[D, F // 32]",
    b_fc2: "float32[D]",
) -> "float32[L, D]":
    w_qkv: float32[QKV, D]
    nn.mx_unpack_f32[QKV, D](w_qkv0, w_qkv1, w_qkv)
    w_o: float32[D, D]
    nn.mx_unpack_f32[D, D](w_o0, w_o1, w_o)
    w_fc1: float32[F, D]
    nn.mx_unpack_f32[F, D](w_fc10, w_fc11, w_fc1)
    w_fc2: float32[D, F]
    nn.mx_unpack_f32[D, F](w_fc20, w_fc21, w_fc2)
    xa: float32[L, D]
    xb: float32[L, D]
    fork2(x, xa, xb)
    h1 = nn.layer_norm[float32, L, D](xa, ln1_g, ln1_b)
    qkv = fp32_linear2d[L, QKV, D](h1, w_qkv, b_qkv)
    a = attention(qkv)
    o = fp32_linear2d[L, D, D](a, w_o, b_o)
    x1 = nn.residual_add[float32, L, D](xb, o)
    x1a: float32[L, D]
    x1b: float32[L, D]
    fork2(x1, x1a, x1b)
    h2 = nn.layer_norm[float32, L, D](x1a, ln2_g, ln2_b)
    f = fp32_linear2d[L, F, D](h2, w_fc1, b_fc1)
    g = nn.GeLU[float32, L, F](f)
    # fc2 re-reads each row of g per output column, so g stays a buffer
    y = fp32_linear2d[L, D, F](g, w_fc2, b_fc2)
    return nn.residual_add[float32, L, D](x1b, y)


def schedule_block(s):
    for i, (n, k) in enumerate([(QKV, D), (D, D), (F, D), (D, F)]):
        s.compose(nn.mx_unpack_f32, instantiate=[n, k], id=str(i) if i else None)
        sub = allo.customize(fp32_linear2d, instantiate=[L, n, k])
        s.compose(schedule_fp32_linear2d(sub), id=str(i) if i else None)
    s.compose(nn.layer_norm, instantiate=[float32, L, D])
    s.compose(nn.GeLU, instantiate=[float32, L, F])
    s.pipeline("fork2:j")
    s.to(MockBuffer("block_fp32", "f"), "GeLU", depth=F)
    s.dataflow("block_fp32")
    return s


LINEARS = ("w_qkv", "w_o", "w_fc1", "w_fc2")


def block_args(x, p):
    """block_fp32's arguments: each linear's weights as two 512b word arrays."""
    args = [np.ascontiguousarray(x, np.float32)]
    for k, v in p.items():
        args += mx_pack_f32_blocks(v) if k in LINEARS else [np.ascontiguousarray(v)]
    return args


def main():
    layers, head = init_weights()
    tokens = np.random.default_rng(1).integers(0, VOCAB, L)
    ref = gpt2(tokens, layers, head, lambda x, p: block_ref(x.astype(np.float64), p))
    mod = allo.customize(block_fp32).build(target="llvm")
    run = lambda x, p: mod(*block_args(x, p))
    logits = gpt2(tokens, layers, head, run)
    rel = np.linalg.norm(logits - ref) / np.linalg.norm(ref)
    match = (logits.argmax(-1) == ref.argmax(-1)).mean()
    print(f"float32 GPT-2: logits rel err {rel:.2e}, top-1 match {match:.0%}")
    return rel


def run_hw(mode, project):
    """One block on Vitis HLS (csyn / hw_emu), checked against LLVM."""
    layers, head = init_weights()
    p = layers[0]
    tokens = np.random.default_rng(1).integers(0, VOCAB, L)
    x = np.ascontiguousarray(head["wte"][tokens] + head["wpe"], np.float32)
    args = block_args(x, p)
    ref = allo.customize(block_fp32).build(target="llvm")(*args)
    s = schedule_block(allo.customize(block_fp32))
    mod = s.build(target="vitis_hls", mode=mode, project=project)
    if mode == "csyn":
        mod()
        return None
    y = np.zeros((L, D), np.float32)
    mod(*args, y)
    rel = np.linalg.norm(y - ref) / np.linalg.norm(ref)
    print(f"[{mode}] float32 block: rel diff vs LLVM {rel:.2e}")
    return rel


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["llvm", "csyn", "hw_emu"], default="llvm")
    parser.add_argument("--project", default="fp32_gpt2_block.prj")
    args = parser.parse_args()
    if args.mode == "llvm":
        main()
    else:
        run_hw(args.mode, args.project)
