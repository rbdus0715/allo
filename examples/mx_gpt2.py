# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=unsubscriptable-object, no-value-for-parameter

# A small GPT-2 with MX arithmetic in its linear layers.
#
# The usual MX split: every Linear (QKV, attention out, MLP fc1/fc2) runs on
# MX operands, its activations quantized on chip (nn.mx_quantize_rows) and its
# weights packed once on the host (mx_pack_np). LayerNorm, GeLU, softmax and
# the residuals stay float32; embedding and the LM head run on the host.
# attn_mx=True also puts attention's Q @ K^T and P @ V on MX operands.
#
# One transformer block is the Allo kernel; the host runs it once per layer.
#   python examples/mx_gpt2.py                                -> LLVM, all formats
#   python examples/mx_gpt2.py --mode csyn|hw_emu [--attn-mx] -> one block on HLS

import re

import numpy as np
import allo
import allo.ir.types as T
import allo.library.nn as nn
from allo.ir.types import float32
from allo.library.mxint8 import mx_pack_np, patch_extern_c_for_class_return_types

# sequence length, model width, heads, MLP width, layers, vocabulary
L, D, H, F, N_LAYER, VOCAB = 8, 64, 2, 256, 2, 128
DH = D // H
QKV = 3 * D
SCALE = 1.0 / float(np.sqrt(DH))


def make_block(Ty, attn_mx):
    """Builds one pre-LN GPT-2 block on MX format Ty."""
    # annotations name Ty (captured: the body uses it), not a closure count
    ATTN_MX = int(attn_mx)

    def causal_softmax(S: "float32[L, L]") -> "float32[L, L]":
        # row softmax of S * SCALE over j <= i
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
        # causal multi-head attention on qkv = [Q | K | V]
        Z: float32[L, D]
        for h in range(H):
            q: float32[L, DH]
            k: float32[L, DH]
            v: float32[L, DH]
            for r, c in allo.grid(L, DH):
                q[r, c] = qkv[r, h * DH + c]
                k[r, c] = qkv[r, D + h * DH + c]
                v[r, c] = qkv[r, 2 * D + h * DH + c]
            # each branch writes its head's columns of Z (assigning a call's
            # result to an existing array would lower to an unsupported copy)
            with allo.meta_if(ATTN_MX):
                qq = nn.mx_quantize_rows[Ty, float32, L, DH](q)
                kq = nn.mx_quantize_rows[Ty, float32, L, DH](k)
                S1 = nn.mx_matmul[Ty, L, L, DH](qq, kq)  # Q @ K^T
                P1 = causal_softmax(S1)
                pq = nn.mx_quantize_rows[Ty, float32, L, L](P1)
                vq = nn.mx_quantize_cols[Ty, float32, DH, L](v)  # V^T blocked along L
                O1 = nn.mx_matmul[Ty, L, DH, L](pq, vq)  # P @ V
                for rm, cm in allo.grid(L, DH):
                    Z[rm, h * DH + cm] = O1[rm, cm]
            with allo.meta_else():
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

    def block(
        x: "float32[L, D]",
        ln1_g: "float32[D]",
        ln1_b: "float32[D]",
        w_qkv: "Ty[QKV, (D + Ty.block_size - 1) // Ty.block_size]",
        b_qkv: "float32[QKV]",
        w_o: "Ty[D, (D + Ty.block_size - 1) // Ty.block_size]",
        b_o: "float32[D]",
        ln2_g: "float32[D]",
        ln2_b: "float32[D]",
        w_fc1: "Ty[F, (D + Ty.block_size - 1) // Ty.block_size]",
        b_fc1: "float32[F]",
        w_fc2: "Ty[D, (F + Ty.block_size - 1) // Ty.block_size]",
        b_fc2: "float32[D]",
    ) -> "float32[L, D]":
        # x + attn(ln1(x)), then + mlp(ln2(.))
        h1 = nn.layer_norm[float32, L, D](x, ln1_g, ln1_b)
        qkv = nn.mx_linear2d[Ty, float32, L, QKV, D](h1, w_qkv, b_qkv)
        a = attention(qkv)
        o = nn.mx_linear2d[Ty, float32, L, D, D](a, w_o, b_o)
        x1 = nn.residual_add[float32, L, D](x, o)
        h2 = nn.layer_norm[float32, L, D](x1, ln2_g, ln2_b)
        f = nn.mx_linear2d[Ty, float32, L, F, D](h2, w_fc1, b_fc1)
        g = nn.GeLU[float32, L, F](f)
        y = nn.mx_linear2d[Ty, float32, L, D, F](g, w_fc2, b_fc2)
        return nn.residual_add[float32, L, D](x1, y)

    return block


def make_block_hw(Ty, attn_mx):
    """make_block with float32 weights quantized on chip, for hw_emu: Allo's
    HLS host does not yet lay out Ty words the way the AXI ports read them."""
    block = make_block(Ty, attn_mx)

    def block_hw(
        x: "float32[L, D]",
        ln1_g: "float32[D]",
        ln1_b: "float32[D]",
        w_qkv: "float32[QKV, D]",
        b_qkv: "float32[QKV]",
        w_o: "float32[D, D]",
        b_o: "float32[D]",
        ln2_g: "float32[D]",
        ln2_b: "float32[D]",
        w_fc1: "float32[F, D]",
        b_fc1: "float32[F]",
        w_fc2: "float32[D, F]",
        b_fc2: "float32[D]",
    ) -> "float32[L, D]":
        q_qkv = nn.mx_quantize_rows[Ty, float32, QKV, D](w_qkv)
        q_o = nn.mx_quantize_rows[Ty, float32, D, D](w_o)
        q_fc1 = nn.mx_quantize_rows[Ty, float32, F, D](w_fc1)
        q_fc2 = nn.mx_quantize_rows[Ty, float32, D, F](w_fc2)
        return block(
            x,
            ln1_g,
            ln1_b,
            q_qkv,
            b_qkv,
            q_o,
            b_o,
            ln2_g,
            ln2_b,
            q_fc1,
            b_fc1,
            q_fc2,
            b_fc2,
        )

    return block_hw


def schedule_mx_all(s):
    """One block per cycle in every MX function instance of `s`: unrolled
    block primitives, pipelined block loops."""
    loops = {
        "mx_block_dot": ("unroll", ("j0", "j1")),
        "mx_quantize_block_f32": ("unroll", ("i1",)),
        "mx_matmul": ("pipeline", ("n",)),
        "bias_add": ("pipeline", ("n",)),
        "mx_quantize_rows": ("pipeline", ("b",)),
        "mx_quantize_cols": ("pipeline", ("b",)),
    }
    for op in s.module.body.operations:
        name = op.attributes["sym_name"].value
        base = re.sub(r"_\d+$", "", name)
        if base in loops:
            prim, names = loops[base]
            for loop in names:
                getattr(s, prim)(f"{name}:{loop}")
    return s


# ---- weights and the float64 reference ---------------------------------------

LINEARS = ("w_qkv", "w_o", "w_fc1", "w_fc2")


def init_weights(seed=0):
    rng = np.random.default_rng(seed)
    w = lambda *s: (rng.standard_normal(s) * 0.02).astype(np.float32)
    b = lambda n: (rng.standard_normal(n) * 0.01).astype(np.float32)
    g = lambda n: (1 + rng.standard_normal(n) * 0.1).astype(np.float32)
    layers = [
        {
            "ln1_g": g(D),
            "ln1_b": b(D),
            "w_qkv": w(QKV, D),
            "b_qkv": b(QKV),
            "w_o": w(D, D),
            "b_o": b(D),
            "ln2_g": g(D),
            "ln2_b": b(D),
            "w_fc1": w(F, D),
            "b_fc1": b(F),
            "w_fc2": w(D, F),
            "b_fc2": b(D),
        }
        for _ in range(N_LAYER)
    ]
    # embeddings at GPT-2's scale relative to the block weights
    head = {"wte": w(VOCAB, D) * 5, "wpe": w(L, D) * 5, "lnf_g": g(D), "lnf_b": b(D)}
    return layers, head


def _ln(x, gamma, beta):
    mu = x.mean(-1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(x.var(-1, keepdims=True) + 1e-5) + beta


def _gelu(x):
    return 0.5 * x * (1 + np.tanh(0.797885 * (x + 0.044715 * x**3)))


def block_ref(x, p):
    """float64 numpy block, same math as make_block's."""
    p = {k: v.astype(np.float64) for k, v in p.items()}
    qkv = _ln(x, p["ln1_g"], p["ln1_b"]) @ p["w_qkv"].T + p["b_qkv"]
    a = np.zeros_like(x)
    causal = np.tril(np.ones((L, L), bool))
    for h in range(H):
        cols = slice(h * DH, (h + 1) * DH)
        q, k, v = (qkv[:, i * D :][:, cols] for i in range(3))
        s = np.where(causal, q @ k.T * SCALE, -np.inf)
        e = np.exp(s - s.max(-1, keepdims=True))
        a[:, cols] = e / e.sum(-1, keepdims=True) @ v
    x1 = x + a @ p["w_o"].T + p["b_o"]
    f = _gelu(_ln(x1, p["ln2_g"], p["ln2_b"]) @ p["w_fc1"].T + p["b_fc1"])
    return x1 + f @ p["w_fc2"].T + p["b_fc2"]


# ---- host side: embedding, layer loop, LM head ---------------------------------


def gpt2(tokens, layers, head, run_block):
    """Logits [L, VOCAB]; run_block(x, layer params) runs one block."""
    x = head["wte"][tokens] + head["wpe"]
    for p in layers:
        x = run_block(x, p)
    x = _ln(x.astype(np.float64), head["lnf_g"], head["lnf_b"])
    return x @ head["wte"].T.astype(np.float64)  # tied LM head, as in GPT-2


def mx_block_runner(Ty, attn_mx):
    mod = allo.customize(make_block(Ty, attn_mx)).build(target="llvm")
    # pack each layer's weights once
    cache = {}

    def run(x, p):
        key = id(p)
        if key not in cache:
            cache[key] = {
                k: (mx_pack_np(Ty, v) if k in LINEARS else v) for k, v in p.items()
            }
        q = cache[key]
        return mod(
            np.ascontiguousarray(x, np.float32),
            *(q[k] for k in p),
        )

    return run


FORMATS = [T.MXInt(8, 32), T.MXFP(4, 3, 32), T.MXFP(2, 1, 32)]


def main():
    layers, head = init_weights()
    tokens = np.random.default_rng(1).integers(0, VOCAB, L)
    ref = gpt2(tokens, layers, head, lambda x, p: block_ref(x.astype(np.float64), p))
    top1 = ref.argmax(-1)
    print(f"GPT-2: L={L} D={D} heads={H} MLP={F} layers={N_LAYER} vocab={VOCAB}")
    print(f"{'format':<16}{'attention':<11}{'logits rel err':>16}{'top-1 match':>13}")
    results = {}
    for Ty in FORMATS:
        for attn_mx in (False, True):
            logits = gpt2(tokens, layers, head, mx_block_runner(Ty, attn_mx))
            rel = np.linalg.norm(logits - ref) / np.linalg.norm(ref)
            match = (logits.argmax(-1) == top1).mean()
            results[Ty.name, attn_mx] = rel
            attn = "MX" if attn_mx else "float32"
            print(f"{Ty.name:<16}{attn:<11}{rel:>16.2e}{match:>12.0%}")
    return results


def run_hw(mode, project, Ty=FORMATS[0], attn_mx=False):
    """One block on Vitis HLS (csyn / hw_emu), checked against LLVM."""
    layers, head = init_weights()
    p = layers[0]
    tokens = np.random.default_rng(1).integers(0, VOCAB, L)
    x = np.ascontiguousarray(head["wte"][tokens] + head["wpe"], np.float32)
    args = [x, *(np.ascontiguousarray(v) for v in p.values())]
    top = make_block_hw(Ty, attn_mx)
    ref = allo.customize(top).build(target="llvm")(*args)
    s = schedule_mx_all(allo.customize(top))
    mod = s.build(target="vitis_hls", mode=mode, project=project)
    # issue #603: extern "C" functions returning ap_int<N> (the packed MX word)
    patch_extern_c_for_class_return_types(f"{project}/kernel.cpp")
    if mode == "csyn":
        mod()
        return None
    y = np.zeros((L, D), np.float32)
    mod(*args, y)
    rel = np.linalg.norm(y - ref) / np.linalg.norm(ref)
    print(f"[{mode}] {Ty.name} block: rel diff vs LLVM {rel:.2e}")
    return rel


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["llvm", "csyn", "hw_emu"], default="llvm")
    parser.add_argument("--project", default="mx_gpt2_block.prj")
    parser.add_argument("--attn-mx", action="store_true")
    args = parser.parse_args()
    if args.mode == "llvm":
        main()
    else:
        run_hw(args.mode, args.project, attn_mx=args.attn_mx)
