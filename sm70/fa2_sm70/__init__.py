# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# HOOLIGAN -- a kernel class for inference engines on NVIDIA Volta (sm_70): exact prefill (cutlass
# FMHA + hand-written d=256 mainloop + byte-KV role kernel), a purpose-built decode stack (split-KV
# flash-decoding, deferred rescale, int8/int4/e4m3 KV), the gated-delta linear scan, and quantised
# weight paths. docs/KERNELS.md is the catalogue; README has the math.
import math
import os
from functools import lru_cache as _lru_cache   # обе функции ниже ЧИСТЫЕ (см. их разбор 31.08)
import torch
from ._ext import prefill_ext, decode_ext, backward_ext

# Чанковый скан gated-delta-net (линейные слои Qwen3.5). Импортируется ЗДЕСЬ намеренно: пока
# модуля не было в пакете, `import fa2_sm70` давал НОЛЬ GDN-символов -- и это был самый честный
# признак того, что пять готовых ядер в боевом пути не стоят вообще. Сам подмодуль ничего не
# собирает при импорте (сборка расширения ленивая, в gdn_ext), поэтому цена -- ноль.
from . import gdn  # noqa: F401

# Gauge-centring (V/O common-mode removal) was evaluated and disabled: the ~4e-2 dK/dQ error it
# targeted was a BENCHMARK ARTIFACT (an fp64-reconstructed O/L passed to the fp16 kernel), not the real
# path -- flash_attn_func's own fp16 forward gives consistent O/L and dK/dQ ~3.7e-4, where centring is
# neutral-to-slightly-worse. Kept off.
_GAUGE_CENTRE_BWD = False

__all__ = ["attention", "attention_varlen", "flash_attn_func", "flash_attn_varlen_func", "decode", "decode_paged", "quantize_kv", "decode_quantized", "quantize_kv_prefill", "prefill_bytes", "latent_attention", "mla_attention", "KVCache", "QuantizedKVCache", "SM70_PEAK_TFLOPS", "SM70_SPEC_TFLOPS", "gdn"]
# TWO PEAKS, AND THE ONE TO QUOTE IS THE MEASURED ONE. 125.3 is the data-sheet fp16 tensor peak
# (boost clock, perfect HMMA issue); a dense fp16 GEMM on a V100-SXM2-32GB tops out at 93.6
# (docs/VOLTA_SM70.md, [spec] vs [measured]). Every %roofline in the README and in benchmarks/ is
# against 93.6 -- quoting the spec number deflates a result by 1.34x and flatters the hardware
# rather than the kernel.
SM70_PEAK_TFLOPS = 93.6          # measured achievable HMMA ceiling
SM70_SPEC_TFLOPS = 125.3         # data-sheet number, for reference only


def _prep(Q, K, V):
    """sm_70 has no bf16 tensor cores -> cast bf16 inputs to fp16 (near-lossless for attention-range
    values; overflow only above +/-65504). Returns ((Q,K,V) fp16, original dtype)."""
    orig = Q.dtype
    if orig == torch.bfloat16:
        Q, K, V = Q.to(torch.float16), K.to(torch.float16), V.to(torch.float16)
    return (Q, K, V), orig


def _pad_head(Q, K, V, target):
    """Pad head_dim with zeros so ANY head_dim runs on the aligned kernels (padded dims contribute 0
    to <q,k> and to the output, so the result is unchanged). `target` is an int (round up to that
    multiple, prefill) or a sorted list of supported dims (round up to the smallest >= d, decode).
    Returns (Q,K,V, d_orig, d_pad)."""
    d = Q.shape[-1]
    if isinstance(target, int):
        dp = ((d + target - 1) // target) * target
    else:
        cand = [t for t in target if t >= d]
        if not cand:
            raise ValueError(f"head_dim {d} exceeds max supported {max(target)}")
        dp = min(cand)
    if dp == d:
        return Q, K, V, d, d
    pad = lambda t: torch.nn.functional.pad(t, (0, dp - d))
    return pad(Q).contiguous(), pad(K).contiguous(), pad(V).contiguous(), d, dp


def attention(Q, K, V, causal=False, scale=None, window=0, softcap=0.0, alibi_slopes=None, window_right=-1):
    """Exact prefill attention (cutlass FMHA, sm_70; matches/beats torch SDPA).

    Q [B, H, Sq, D]; K, V [B, Hkv, Sk, D] (GQA/MQA when Hkv < H). fp16 OR bf16 (cast to fp16 on
    sm_70). ANY D <= 1024 (non-multiple-of-8 is zero-padded; D > 128 uses the cutlass
    output-accumulator path, e.g. MLA-scale 256/512). Returns (O [B,H,Sq,D] in the input dtype,
    LSE fp32 [B, H, Sq] = log sum exp(scale*Q.K^T)).
      window: >0 restricts each query i to keys [i-window+1, i] (causal sliding-window, e.g. Gemma-4
              sliding layers); use with causal=True. 0 = full causal / full attention.
      window_right: >=0 makes the window TWO-SIDED (bidirectional local attention): query i attends
              keys [i-(window-1), i+window_right]; use causal=False. -1 (default) = one-sided/causal.
      softcap: >0 applies cap*tanh(logit/cap) to each logit (Gemma-style); 0 = off.
      alibi_slopes: per-head slopes [H] (fp32); adds slope_h*(key_pos - query_pos) to the logit
                    (linear positional bias). None = off.
    """
    if scale is None:
        scale = 1.0 / math.sqrt(Q.shape[-1])
    (Q, K, V), orig = _prep(Q, K, V)
    Q, K, V, d0, dp = _pad_head(Q, K, V, 8)                 # cutlass efficient needs head_dim % 8 == 0
    if alibi_slopes is None:
        al = torch.empty(0, dtype=torch.float32, device=Q.device)
    else:
        al = torch.as_tensor(alibi_slopes, dtype=torch.float32, device=Q.device)
    O, L = prefill_ext().attn_fwd_cutlass(Q, K, V, float(scale), bool(causal), int(window), float(softcap), al, int(window_right))
    if dp != d0:
        O = O[..., :d0].contiguous()
    if orig != O.dtype:
        O = O.to(orig)
    return O, L


def attention_varlen(Q, K, V, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                     causal=False, scale=None):
    """Variable-length / packed attention (no padding for ragged batches).

    Q [Tq, H, D]; K, V [Tk, Hkv, D] are token-packed across all sequences; `cu_seqlens_q`,
    `cu_seqlens_k` are int32 [nseq+1] cumulative offsets (cu[0]=0, cu[-1]=Tq/Tk). Each sequence i
    attends its own Q rows [cu_q[i], cu_q[i+1]) to its own K rows [cu_k[i], cu_k[i+1]); causal is
    per-sequence bottom-right. fp16/bf16, GQA/MQA, head_dim <= 512. Returns (O [Tq, H, D] in the
    input dtype, LSE fp32 [nseq, H, lse_dim] where lse_dim = ceil(max_seqlen_q / 32) * 32)."""
    if scale is None:
        scale = 1.0 / math.sqrt(Q.shape[-1])
    (Q, K, V), orig = _prep(Q, K, V)
    Q, K, V, d0, dp = _pad_head(Q, K, V, 8)
    cu_q = cu_seqlens_q.to(torch.int32); cu_k = cu_seqlens_k.to(torch.int32)
    O, L = prefill_ext().attn_fwd_cutlass_varlen(Q, K, V, cu_q, cu_k, int(max_seqlen_q), int(max_seqlen_k),
                                                 float(scale), bool(causal))
    if dp != d0:
        O = O[..., :d0].contiguous()
    if orig != O.dtype:
        O = O.to(orig)
    return O, L


def _bwd_run(Qp, Kp, Vp, Op, dOp, L, scale, causal, dropout_p=0.0, seed=0, offset=0):
    """Core backward launch. All BHSD & fp16, head padded to `dp`; K/V still [B,Hkv,.,dp].
    Returns dQ, dK, dV [B,H/Hkv,S,dp] (grouped grads reduced back to Hkv). See attn_bwd_cutlass.cu
    for the BSHD layout contract the cutlass backward kernel imposes. When dropout_p>0, the backward
    replays the forward's mask from (seed, offset)."""
    B, H, Sq, dp = Qp.shape
    Hkv, Sk = Kp.shape[1], Kp.shape[2]
    g = H // Hkv
    # The backward kernel is one-KV-head-per-query-head: expand K/V, reduce dK/dV back afterwards.
    Ke = Kp if Hkv == H else Kp.repeat_interleave(g, dim=1)
    Ve = Vp if Hkv == H else Vp.repeat_interleave(g, dim=1)
    # Optional gauge-centring (exact; disabled by default): the
    # softmax makes o_i ~= vbar, so U_ij - D_i = <g_i, v_j - o_i> subtracts the common mode <g_i, vbar> ->
    # catastrophic fp16 cancellation. Subtract vbar (per-(b,h) key-mean of V) from BOTH V and O; the
    # in-kernel U=<g,V-vbar>, D=<g,O-vbar> shrink, T=P(U-D) is unchanged. Verified across shapes/seeds in
    # Toggle: fa2_sm70._GAUGE_CENTRE_BWD.
    if _GAUGE_CENTRE_BWD:
        vbar = Ve.float().mean(dim=2, keepdim=True)           # [B,H,1,dp] per-(b,h) key-mean of V (Ve is H-expanded)
        Ve = (Ve.float() - vbar).to(Ve.dtype)
        Op = (Op.float() - vbar).to(Op.dtype)
    # LSE must have strideH % 8 == 0: pad the query axis up to a multiple of 8 (extra rows unread).
    Sq_pad = (Sq + 7) // 8 * 8
    if L.shape[2] == Sq_pad:
        Lp = L.contiguous()
    else:
        Lp = torch.zeros(B, H, Sq_pad, dtype=torch.float32, device=L.device)
        Lp[..., :Sq] = L
    # delta = (dO * O).sum(-1). For dp<=128 with Sq a multiple of 128 the backward runs the kOutputInRF tile
    # (blk=2, kBlockSizeI=128 >= kMaxK) where kKernelComputesDelta is TRUE on sm_70 -- the kernel computes delta
    # in-kernel and writes it to delta_ptr, so we pass an UNINITIALISED buffer and skip the redundant torch
    # reduction (a full extra read of dO+O + a launch; measurable at short seqlen). Otherwise (dp>128, OR a
    # partial last query block Sq%128!=0 which the C++ dispatch reroutes to the kMaxK>kBlockSizeI lock-RMW tile
    # to dodge R7's partial-block dQ defect) kOutputInRF is FALSE and kKernelComputesDelta is FALSE -> the kernel
    # does NOT compute delta, so we MUST precompute it here (else it reads uninitialised delta -> wrong dQ/dK,
    # dV unaffected). Keep the exact same routing predicate as attn_bwd_cutlass()'s reroute (Sq%128).
    if dp <= 128 and (Sq % 128 == 0):
        delta = torch.empty(B, H, Sq, dtype=torch.float32, device=dOp.device)   # kernel fills it (kOutputInRF path)
    else:
        delta = (dOp.float() * Op.float()).sum(-1).contiguous()   # [B, H, Sq] fp32 (lock-RMW path needs it precomputed)
    # Key-axis split count for the underfilled-B*H case. The C++ dispatch (attn_bwd_cutlass) decides
    # [R7 re-tune] key-split count. Post-R7 the split kernel is no-spill and dominates the no-split kernel at
    # every B*H (the dispatch now always splits), so ns is the FILL knob and should target the finest useful
    # split (one key-block per CTA, matching torch's grid) capped so total CTAs G = ns*B*H stays ~<=1024 (past
    # that the per-CTA operand re-read + fold traffic over-splits). MEASURED clean (clock-locked): ns=max_splits
    # wins for B*H<=~48 (S768/1024/1536 B*H=32: +5-6% over the old ns=8); ns~=1024/(B*H) caps large B*H
    # (B*H=128,S1024 ns=8 -> 1.085x). Old target was 256 (under-split -> parity/loss). Cap at 1 key-block/split.
    if dp > 128:
        # [d>128 re-tune] The 1024-CTA target above was measured for dp<=128, where R7 makes each CTA
        # cheap. At dp>128 R7 is off (its gate is kOutputInRF == kMaxK<=kBlockSizeI), every CTA spills
        # and carries gmem dK/dV accumulators, so over-splitting multiplies that traffic instead of
        # buying fill. Measured optimum is ONE WAVE: ns*B*H ~= 64, i.e. about the 80 SMs.
        #   torch/ours by ns (causal, fp16; >1 = we win):
        #     B1H8  S2048 d256  ns= 1/2/4/6/8/12 -> 0.29/0.50/0.75/0.86/0.93/0.83   best 8  (B*H*ns=64)
        #     B1H8  S4096 d256                   -> 0.26/0.42/0.65/0.78/0.87/0.77   best 8
        #     B2H8  S2048 d256                   -> 0.47/0.70/0.95/0.82/0.86/0.83   best 4  (B*H*ns=64)
        #     B1H16 S2048 d256                   -> 0.46/0.69/0.93/0.81/0.84/0.81   best 4
        #     B1H8  S2048 d512                   -> 0.29/0.47/0.71/0.81/0.86/0.77   best 8
        #     B1H8  S4096 d512                   -> 0.26/0.43/0.67/0.80/0.88/0.77   best 8
        # The old rule picked ns=32..64 here and cost 0.58-0.67 -- a 30-50% loss against this.
        # [wave fill] Target the SM COUNT (80), not 64, once B*H alone is small. The earlier sweep tested
        # ns in {1,2,4,6,8,12} and concluded "one wave = 64 CTAs" -- but at B*H=8 that is ns=8, and ns=10
        # = 80 CTAs (exactly the 80 SMs, still ONE wave) was never in the grid; the next point it did try,
        # ns=12 = 96 CTAs, is already two waves, so the collapse there got read as "64 is the optimum".
        # Filling 80 instead of 64 is strictly finer splits at the same wave count.
        # Measured (clock-locked 1350 MHz, median of 5x40 iters, ratio = torch/ours, >1 = we win):
        #                        tgt64   tgt72   tgt80   tgt88   tgt96
        #   B1H8  S2048 d256     0.850   0.831   0.874   0.751   0.750
        #   B1H8  S4096 d256     0.868   0.839   0.907   0.722   0.762
        #   B1H8  S8192 d256     0.941   0.914   1.014   0.763   0.834   <- first win at B*H=8
        #   B1H8  S2048 d512     0.835   0.807   0.843   0.732   0.735
        #   B1H8  S4096 d512     0.866   0.835   0.909   0.721   0.759
        #   B2H8/B1H16 S2048     0.917   0.917   0.900   0.792   0.796
        #   B2H8/B1H16 S1024     0.925   0.925   0.903   0.804   0.803
        #   B4H8/B2H16 S1024     0.975   0.975   0.975   0.792   0.792
        # So: B*H==8 -> 80 (avg 0.909 vs 0.872); B*H>=16 -> 64 (0.922 vs 0.902; at B*H=32 both round to
        # ns=2 anyway). 88/96 fall off the second-wave cliff everywhere -- do not raise the target further.
        # B*H<=4 keeps 64 as well, MEASURED (my first rule said "<=8" and regressed it):
        #                        tgt64   tgt80   tgt96  tgt128  tgt160  tgt240
        #   B1H4  S2048 d256     0.754   0.713   0.711   0.630   0.625   0.627
        #   B1H4  S4096 d256     0.763   0.773   0.716   0.685   0.644   0.568
        #   B1H4  S2048 d512     0.743   0.708   0.702   0.620   0.618   0.615
        #   B1H2 / B1H1          flat (ns hits the Sk/64 cap, every target is the same config)
        # i.e. 80 CTAs is NOT universally good -- at B*H=4 it means ns=20, and the finer split costs more in
        # fold traffic than the fill buys. The win is specific to B*H=8 (ns=8 -> 10). NB one cell (B1H2 S4096,
        # tgt240 = 1.268 vs tgt160 = 0.566) is a measurement ARTEFACT, not an effect: ns caps at Sk/64=64 for
        # both, so the two configs are byte-identical. Do not build a rule on it.
        _tgt = int(os.environ.get("FA2SM70_BWD_LARGE_TARGET", "0")) or (80 if B * H == 8 else 64)
        ns = max(1, min((Sk + 63) // 64, round(_tgt / max(1, B * H)) or 1))
    else:
        ns = max(1, min((Sk + 63) // 64, round(1024 / max(1, B * H))))
    if dropout_p > 0.0:
        # dropout backward entry has no bhsd path yet -> BSHD transpose glue.
        bshd = lambda t: t.permute(0, 2, 1, 3).contiguous()   # BHSD -> BSHD (packed)
        Qb, Kb, Vb, Ob, dOb = bshd(Qp), bshd(Ke), bshd(Ve), bshd(Op), bshd(dOp)
        dQb, dKb, dVb = backward_ext().attn_bwd_cutlass_dropout(
            Qb, Kb, Vb, Ob, dOb, Lp, delta, float(scale), bool(causal), int(ns),
            float(dropout_p), int(seed), int(offset))
        to_bhsd = lambda t: t.permute(0, 2, 1, 3)             # BSHD -> BHSD
        dQ, dKe, dVe = to_bhsd(dQb).contiguous(), to_bhsd(dKb), to_bhsd(dVb)
    else:
        # BHSD-direct: the kernel reads Q/K/V/O/dO and writes dQ/dK/dV in native [B,H,S,D] via the
        # settable output-stride overrides -- no BSHD<->BHSD permute+contiguous glue (~1.3x of the wall).
        # .contiguous() is a no-op when the tensor is already BHSD-contiguous.
        ctg = lambda t: t.contiguous()
        dQ, dKe, dVe = backward_ext().attn_bwd_cutlass(
            ctg(Qp), ctg(Ke), ctg(Ve), ctg(Op), ctg(dOp), Lp, delta,
            float(scale), bool(causal), int(ns), 2, True)     # bhsd=True
    if Hkv != H:
        dK = dKe.reshape(B, Hkv, g, Sk, dp).sum(2)
        dV = dVe.reshape(B, Hkv, g, Sk, dp).sum(2)
    else:
        dK, dV = dKe.contiguous(), dVe.contiguous()
    return dQ, dK, dV


class _AttnFn(torch.autograd.Function):
    """Differentiable exact attention on sm_70: cutlass FMHA forward + cutlass FMHA backward.
    Head dim padded to a multiple of 8 (<=128); grads sliced/cast back to the input's dtype & dim."""
    @staticmethod
    def forward(ctx, Q, K, V, causal, scale, dropout_p, seed):
        (Qf, Kf, Vf), orig = _prep(Q, K, V)
        Qp, Kp, Vp, d0, dp = _pad_head(Qf, Kf, Vf, 8)
        if dropout_p > 0.0:
            Op, L = prefill_ext().attn_fwd_cutlass_dropout(Qp, Kp, Vp, float(scale), bool(causal),
                                                           float(dropout_p), int(seed), 0)
        else:
            Op, L = prefill_ext().attn_fwd_cutlass(Qp, Kp, Vp, float(scale), bool(causal), 0, 0.0,
                                                   torch.empty(0, dtype=torch.float32, device=Qp.device))
        ctx.save_for_backward(Qp, Kp, Vp, Op, L)
        ctx.causal, ctx.scale, ctx.d0, ctx.dp, ctx.orig = causal, scale, d0, dp, orig
        ctx.dropout_p, ctx.seed = dropout_p, seed
        O = Op[..., :d0] if dp != d0 else Op
        return O.to(orig) if orig != Op.dtype else O

    @staticmethod
    def backward(ctx, dO):
        Qp, Kp, Vp, Op, L = ctx.saved_tensors
        d0, dp = ctx.d0, ctx.dp
        dOp = dO.to(torch.float16).contiguous()
        if dp != d0:
            dOp = torch.nn.functional.pad(dOp, (0, dp - d0)).contiguous()
        dQ, dK, dV = _bwd_run(Qp, Kp, Vp, Op, dOp, L, ctx.scale, ctx.causal,
                              dropout_p=ctx.dropout_p, seed=ctx.seed, offset=0)
        if dp != d0:
            dQ, dK, dV = dQ[..., :d0], dK[..., :d0], dV[..., :d0]
        cast = lambda t: t.to(ctx.orig) if ctx.orig != t.dtype else t
        return cast(dQ.contiguous()), cast(dK.contiguous()), cast(dV.contiguous()), None, None, None, None


class _AttnVarlenFn(torch.autograd.Function):
    """Differentiable variable-length attention: cutlass FMHA forward + backward with cu_seqlens.
    Packed [T, H, D]; grads packed the same way and reduced back to Hkv for GQA/MQA."""
    @staticmethod
    def forward(ctx, Q, K, V, cu_q, cu_k, max_q, max_k, causal, scale):
        (Qf, Kf, Vf), orig = _prep(Q, K, V)
        Qp, Kp, Vp, d0, dp = _pad_head(Qf, Kf, Vf, 8)
        cu_q = cu_q.to(torch.int32); cu_k = cu_k.to(torch.int32)
        Op, L = prefill_ext().attn_fwd_cutlass_varlen(Qp, Kp, Vp, cu_q, cu_k, int(max_q), int(max_k),
                                                      float(scale), bool(causal))
        ctx.save_for_backward(Qp, Kp, Vp, Op, L, cu_q, cu_k)
        ctx.max_q, ctx.max_k, ctx.causal, ctx.scale = int(max_q), int(max_k), causal, scale
        ctx.d0, ctx.dp, ctx.orig = d0, dp, orig
        O = Op[..., :d0] if dp != d0 else Op
        return O.to(orig) if orig != Op.dtype else O

    @staticmethod
    def backward(ctx, dO):
        Qp, Kp, Vp, Op, L, cu_q, cu_k = ctx.saved_tensors
        d0, dp = ctx.d0, ctx.dp
        Tq, H = Qp.shape[0], Qp.shape[1]
        Tk, Hkv = Kp.shape[0], Kp.shape[1]
        g = H // Hkv
        nseq = cu_q.shape[0] - 1
        dOp = dO.to(torch.float16).contiguous()
        if dp != d0:
            dOp = torch.nn.functional.pad(dOp, (0, dp - d0)).contiguous()
        Ke = Kp if Hkv == H else Kp.repeat_interleave(g, dim=1).contiguous()   # [Tk, H, dp]
        Ve = Vp if Hkv == H else Vp.repeat_interleave(g, dim=1).contiguous()
        delta = (dOp.float() * Op.float()).sum(-1).transpose(0, 1).contiguous()  # [H, Tq]
        ns = max(1, min((ctx.max_k + 63) // 64, max((ctx.max_k + 127) // 128, round(256 / max(1, H * nseq)))))
        dQ, dKe, dVe = backward_ext().attn_bwd_cutlass_varlen(
            Qp, Ke, Ve, Op, dOp, L, delta, cu_q, cu_k, ctx.max_q, ctx.max_k,
            float(ctx.scale), bool(ctx.causal), int(ns))
        if Hkv != H:
            dK = dKe.reshape(Tk, Hkv, g, dp).sum(2)
            dV = dVe.reshape(Tk, Hkv, g, dp).sum(2)
        else:
            dK, dV = dKe, dVe
        if dp != d0:
            dQ, dK, dV = dQ[..., :d0], dK[..., :d0], dV[..., :d0]
        cast = lambda t: t.to(ctx.orig) if ctx.orig != t.dtype else t
        return (cast(dQ.contiguous()), cast(dK.contiguous()), cast(dV.contiguous()),
                None, None, None, None, None, None)


def flash_attn_varlen_func(Q, K, V, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                           causal=False, scale=None):
    """Differentiable variable-length / packed attention (forward + backward) for training on sm_70.

    Packed Q [Tq, H, D], K/V [Tk, Hkv, D] with int32 cu_seqlens [nseq+1]. fp16/bf16, GQA/MQA,
    head_dim <= 128. Autograd for dQ, dK, dV. For inference-only varlen (any head_dim <= 512), use
    `attention_varlen`."""
    if scale is None:
        scale = 1.0 / math.sqrt(Q.shape[-1])
    return _AttnVarlenFn.apply(Q, K, V, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal, scale)


def flash_attn_func(Q, K, V, causal=False, scale=None, dropout_p=0.0):
    """Differentiable exact attention (forward + backward) for training on sm_70.

    Q [B, H, Sq, D]; K, V [B, Hkv, Sk, D] (GQA/MQA); fp16/bf16; head_dim <= 512 (D > 128 uses the
    gmem output-accumulator backward path, e.g. Gemma-4 d=256/512 fine-tuning). Returns O [B,H,Sq,D]
    in the input dtype and supports autograd for dQ, dK, dV (cutlass FMHA backward). `dropout_p` > 0
    applies reproducible attention dropout (training, head_dim <= 128). For window / soft-cap / ALiBi,
    use `attention` (inference, no backward)."""
    if scale is None:
        scale = 1.0 / math.sqrt(Q.shape[-1])
    # Seed from torch's RNG (deterministic under manual_seed) so fwd/bwd share one mask and runs
    # reproduce; offset is 0 (the per-(batch,head) offset is added inside the kernel).
    seed = int(torch.randint(0, 2**62, (1,)).item()) if dropout_p > 0.0 else 0
    return _AttnFn.apply(Q, K, V, causal, scale, dropout_p, seed)


def _pick_gf(d, group):
    """Mirror of pick_gf() in split_decode.cu -- how many query heads one CTA fuses.

    Duplicated here on purpose: _auto_splits has to know the grid it is sizing, and the paged
    kernel's grid is (splits, H/gf, B). Without it the fill target would be computed for a grid that
    is not the one being launched, and the paged path would silently get a quarter of the splits it
    was measured to want.
    """
    cap = 2 if d == 512 else (6 if d == 256 else 8)
    for gf in (8, 6, 4, 3, 2):
        if gf <= cap and group % gf == 0:
            return gf
    return 1


# ПОТОЛОК ЧИСЛА СРЕЗОВ -- ОДНО ИМЯ НА ВЕСЬ ПРОЕКТ.
# Он читается ДВУМЯ независимыми местами (здесь и бэкенд vLLM), и пока это были два литерала,
# правка одного молча ничего не меняла: бэкенд обрезал своим. Ровно так ручка и оказалась
# мёртвой. Значение 120 -- ЗАМЕРЕНО (reports/EV_ns_ladder.md), не выбрано.
DEFAULT_MAX_SPLITS = 120


# [ПОВТОРНАЯ РАБОТА 31.08] `_auto_splits` и `_pick_gf` -- чистые функции от небольших целых, но
# зовутся НА КАЖДЫЙ слой внимания каждого шага (16 слоёв x 2 вызова), и внутри `_auto_splits`
# ещё и `os.environ.get` -- обращение к словарю окружения с разбором строки. Функции чистые
# (кроме чтения рычагов, которые в рантайме не меняются), поэтому память по аргументам законна.
def _auto_splits(B, H, Sk, gf=1):
    """How many KV chunks to split a decode step into.

    The split count is purely a parallelism knob -- the deferred-rescale combine is associative, so
    the answer does not depend on it -- but it is the ONLY axis that can fill the GPU when a serving
    request fixes B and H.

    THE TARGET IS A CTA COUNT, NOT A CHUNK SIZE. Measured optimum per shape (int8 cache, V100):

        H/Hkv, Sk        auto(old)      best        CTAs at best
        32/32,  2048      64  65.2us     16  39.0us     512
        32/32,  8192      64 134.6us     16 117.7us     512
        32/8,   8192      64 103.7us     16  79.5us     512
        32/32, 32768      64 409.3us     16 398.8us     512
         8/8,  16384      64  71.5us     64  71.6us     512
        32/32,  4096(d64) 64  58.6us     16  47.5us     512

    Five of six shapes wanted 16 and the sixth wanted 64 -- and in every one of them the product
    B*H*splits is 512, i.e. about 6.4 CTAs per SM on 80 SMs. The old rule aimed at 2048 CTAs (25
    waves) and over-split by 4x, which cost up to 1.67x; its own comment said ">=128 keys per chunk"
    while the code divided by 32, so the floor never bound either.

    `gf` is the number of query heads fused per CTA (the MQA/GQA grouping): the paged kernel's grid
    is (splits, H/gf, B), so its CTA count is B*(H/gf)*splits and it needs proportionally more
    splits. Callers on that path must pass it, or the fill target is computed for a grid that is not
    the one being launched.
    """
    ctas_per_head_group = max(1, B * max(1, H // max(1, gf)))
    # ЦЕЛЬ ПО CTA -- РУЧКА, А НЕ ЛИТЕРАЛ (28.08.2026), НО УМОЛЧАНИЕ 512 ПОДТВЕРЖДЕНО БОЕМ.
    # Микробенч ядра при T=87000 монотонно улучшался с числом делений (32 -> 1.776 мс,
    # 64 -> 1.439, 128 -> 1.282, 256 -> 1.193) и обещал x1.49. СКВОЗНОЙ замер это ОПРОВЕРГ:
    # цель 2048 дала шаг 39.4 мс против 33.2 при 512 (64K, стенд) -- ХУЖЕ на 19 %. Больше
    # делений раздувают рабочее поле A/B и сведение, и в живом пути это перевешивает выигрыш
    # главного ядра. Ручка оставлена для дальнейших опытов; трогать умолчание -- только со
    # СКВОЗНЫМ замером, микробенчу здесь верить нельзя.
    _цель = int(os.environ.get("FA2SM70_SPLIT_FILL", "512"))
    fill = max(1, round(_цель / ctas_per_head_group))
    per_chunk = max(1, Sk // 128)                     # >=128 keys/chunk, as the reduce wants
    # ПОТОЛОК СРЕЗОВ -- РУЧКА, А НЕ ЛИТЕРАЛ (07.08.2026).
    #
    # Здесь стояло глухое `64`, и это делало ВНЕШНЮЮ ручку МЁРТВОЙ: боевой маршрут считает
    # `min(_auto_splits(...), FA2SM70_MAX_SPLITS)`, то есть снаружи можно только ПОНИЗИТЬ то, что
    # внутри уже обрезано до 64. Любой, кто пробовал фальсифицировать потолок этой переменной,
    # получал НОЛЬ изменений и делал вывод «потолок ни при чём» -- вырожденный прибор, вердикт по
    # которому недействителен (правило §3). Ровно такую вырожденную таблицу и нашла ревизия
    # заполнения: при ns>=64 время не зависело от контекста (4096 -> 1.14 мс, 65536 -> 1.13 мс при
    # 16-кратной разнице байт), потому что ns не менялся вовсе.
    #
    # Что связывает на боевой форме: B=1, H=12, gf=6 -> ctas_per_head_group=2, fill=256,
    # per_chunk=63000//128=492. min(492, 256, 64) = 64 -- то есть решает ЛИТЕРАЛ, а два расчётных
    # аргумента не при делах. Грид (64, 2, 1) = 128 CTA; резидентность тела -- НЕ ОДИН блок на SM,
    # значит знаменатель здесь СЛОТЫ, а не 80, и 128 CTA машину не заполняют.
    #
    # Умолчание оставлено 64: поведение НЕ меняется, пока не замерено. Ручка теперь способна
    # ПОДНЯТЬ потолок, и это условие того, чтобы замер вообще был возможен.
    # УМОЛЧАНИЕ 64 -> 120, ЗАМЕРЕНО 07.08.2026 (reports/EV_ns_ladder.md), карта 0, k_absmax вынесен.
    #
    #   B=1 H=12 Hkv=2 S=63000 : ns=64 285.0 мкс -> ns=120 224.9 мкс  x1.267
    #   B=1 H=12 Hkv=2 S=131072: ns=64 478.4     -> ns=120 395.3      x1.210
    #   B=1 H=24 Hkv=4 S=63000 : ns=64 538.7     -> ns=120 419.1      x1.285
    #   B=4 H=12 Hkv=2 S=63000 : ns=64 748.6     -> ns=120 743.0      x1.008 (fill сам даёт 64)
    #   B=1 H=12 Hkv=2 S=8192  : упор в per_chunk=64, ns не меняется   x1.000
    # relL2 между всеми ns: 4e-07..8e-07 -- свод ассоциативен, ответ от числа срезов не зависит.
    #
    # ЧТО ЗДЕСЬ ВАЖНЕЕ САМОГО ЧИСЛА. Оптимум держится за ns, а НЕ за число CTA: лучшие CTA гуляют
    # 240..960 при одном и том же ns=120. Значит «целиться в резидентность» -- НЕ правило, это
    # совпадение на боевой форме. И второе: собственная цель `fill` = 512 CTA ВРЕДНА -- при B=1,H=12
    # она просит ns=256, а это ЗАМЕРЕНО 0.517x. Литерал 64 её случайно ЗАКРЫВАЛ, поэтому снимать
    # потолок, не тронув цель, было бы хуже, чем не трогать ничего.
    #
    # Менять `fill` не стал: при B=4 он даёт 64, и замер там показывает x1.008 -- то есть на батче
    # он не вредит, а форм с батчем я прогнал одну. Правится по замеру, а не заодно.
    cap = int(os.environ.get("FA2SM70_MAX_SPLITS", str(DEFAULT_MAX_SPLITS)))
    # [29.08] ЦЕЛЬ ПО ЧИСЛУ БЛОКОВ НЕ ДОЛЖНА РЕЗАТЬ МАЛЫЙ БАТЧ.
    # `fill` делит цель на число строк, поэтому при спекуляции (B = k+1 = 4 строки ОДНОГО
    # запроса) он даёт ns=64, а замеренный оптимум -- 120. Лесенка на ЗДОРОВОЙ карте
    # (карта 1, микробенч, мкс; ns=64 / ns=120):
    #     Sk=65536  q=1  347.7 / **238.1**   Sk=262144 q=1  1150.7 / **726.5** (46.2 % крыши)
    #     Sk=65536  q=4  991.7 / **816.2**   Sk=262144 q=4  3631.4 / **2856.8**
    # То есть 120 лучше во ВСЕХ четырёх случаях, на 21-58 %. Прежний замер (07.08) считал, что
    # при B=4 цель безвредна (x1.008) -- он снят на ДРУГОЙ карте и другой длине; сегодняшний
    # прогон его отменяет. Соседние значения (160, 192, 256, 512) ХУЖЕ 120 везде -- оптимум
    # держится за само ns, а не за число блоков, как и записано выше.
    # Ограничение по строкам: при большом батче `fill` по-прежнему бережёт рабочую область
    # (она растёт как строки x H x ns x d), поэтому правило меняем ТОЛЬКО для малого батча.
    _строк_мало = B <= 8
    _ns = min(per_chunk, cap) if _строк_мало else min(per_chunk, fill, max(1, cap))
    return max(1, _ns)


def quantize_kv(K, V, bits=8, per_channel=None):
    """Per-token symmetric quantization of a KV cache (do this ONCE when keys are appended).

    per_channel (default: True for int4, False for int8): also divide out a per-channel scale
    before per-token quantization ("2-level"). This removes the outlier-channel penalty (a real-LLM
    tell that otherwise wrecks int4: ~28% -> ~12% relL2). It is applied for free at decode time --
    the per-channel key scale cancels in <q,k> (fold into the query) and the per-channel value scale
    is a post-multiply on the output -- so no kernel change (see README section 4).

    Returns a dict `cache` to pass to decode_quantized(...): {Kq, K_scale, Vq, V_scale, k_absmax,
    k_chan, v_chan, bits}. Kq/Vq are int8 [.,.,Sk,D] (bits=8) or packed uint8 [.,.,Sk,D/2] (bits=4);
    *_scale fp32 [.,.,Sk]; k_absmax fp32 [.,.,Hkv]; k_chan/v_chan fp32 [.,.,D] (or None).
    """
    assert bits in (8, 4)
    if per_channel is None:
        per_channel = (bits == 4)
    k_chan = v_chan = None
    if per_channel:
        k_chan = K.abs().amax(dim=2, keepdim=True).clamp_min(1e-8)   # [B,Hkv,1,D]
        v_chan = V.abs().amax(dim=2, keepdim=True).clamp_min(1e-8)
        K = K / k_chan
        V = V / v_chan
        k_chan = k_chan.squeeze(2).float().contiguous()             # [B,Hkv,D]
        v_chan = v_chan.squeeze(2).float().contiguous()

    def q(X):
        qmax = (1 << (bits - 1)) - 1
        s = (X.abs().amax(-1, keepdim=True) / qmax).clamp_min(1e-8)
        qd = torch.round(X / s).clamp(-qmax, qmax)
        deq = qd * s
        if bits == 8:
            return qd.to(torch.int8).contiguous(), s.squeeze(-1).float().contiguous(), deq
        qn = (qd + 8).to(torch.uint8)
        return (qn[..., 0::2] | (qn[..., 1::2] << 4)).contiguous(), s.squeeze(-1).float().contiguous(), deq

    Kq, K_scale, Kdq = q(K)
    Vq, V_scale, _ = q(V)
    k_absmax = Kdq.float().norm(dim=-1).amax(dim=-1)               # norm of the (pre-channel-scaled) key
    return dict(Kq=Kq, K_scale=K_scale, Vq=Vq, V_scale=V_scale, k_absmax=k_absmax,
                k_chan=k_chan, v_chan=v_chan, bits=bits)


def decode(Q, K, V, scale=None, kv_dtype="fp16", deferred=True, num_splits=None, k_absmax=None, mqa="auto",
           softcap=0.0, alibi_slopes=None):
    """Autoregressive decode step (Sq == 1); split-KV flash-decoding for parallelism at Sq=1.

    Q [B, H, 1, D] fp16; K, V [B, Hkv, Sk, D] fp16. D in {64, 128, 256, 512}.
      kv_dtype: 'fp16' (EXACT, default) | 'int8' (~0.7% relL2) | 'int4' (coarse; see README).
      deferred: True prefers the certified-U associative sum (no online rescale -> more ILP; exact
                for fp16) -- used when k_absmax is supplied (a real cache: KVCache/decode_paged) or
                for MQA (Hkv==1). For a bare, uncached decode with Hkv>=2, computing max||k|| would
                cost a full extra read of K, so decode() transparently takes the exact online path
                instead (up to 3.3x faster). Pass k_absmax (running max of ||k||) to force deferred.
      mqa: 'auto' (default) fuses GF query heads per CTA when Hkv<H so the shared KV is read from HBM
           ONCE instead of once per head (bit-exact; ~1.4-1.6x when the KV is large: d=512 or long
           context, neutral otherwise, never slower). Applies to fp16+deferred. Set False to disable.
    Returns (O fp32 [B, H, 1, D], LSE fp32 [B, H, 1]). For int8/int4, K/V are quantized internally;
    to reuse a quantized cache across steps use quantize_kv + decode_quantized instead.
    """
    B, H = Q.shape[0], Q.shape[1]
    Hkv, Sk = K.shape[1], K.shape[2]
    if scale is None:
        scale = 1.0 / math.sqrt(Q.shape[-1])
    if num_splits is None:
        num_splits = _auto_splits(B, H, Sk)
    (Q, K, V), _ = _prep(Q, K, V)                   # bf16 -> fp16
    Q, K, V, d0, dp = _pad_head(Q, K, V, [64, 128, 256, 512])   # decode kernel supports these head dims
    ext = decode_ext()
    # softcap / ALiBi ride the ONLINE path (running max) -- the deferred path's Cauchy-Schwarz U-bound
    # does not bound modified scores (would risk exp underflow), so route modifiers there exactly.
    has_mod = softcap > 0.0 or alibi_slopes is not None
    if kv_dtype == "fp16":
        if has_mod:
            al = torch.as_tensor(alibi_slopes, dtype=torch.float32, device=Q.device) if alibi_slopes is not None \
                else torch.empty(0, dtype=torch.float32, device=Q.device)
            O, lse = ext.flash_decode(Q, K, V, float(scale), int(num_splits), float(softcap), al)
        elif deferred and (k_absmax is not None or Hkv == 1):
            # Deferred needs the certified-U bound max||k||. Computing it HERE (uncached) is a full
            # extra read of K -- ~as costly as the whole decode -- so it only pays off when either
            #   (a) k_absmax is already maintained (a real cache: KVCache/decode_paged), or
            #   (b) Hkv == 1 (MQA): the bound is one cheap pass over a single KV head, and the fused
            #       kernel then reads the shared KV ONCE instead of g times (net win, measured 1.1-1.5x).
            # For Hkv >= 2 with no cached bound, the online path below is up to 3.3x faster (it needs
            # no bound, and re-reads the shared KV only g/2x vs paying the full-K recompute here).
            if k_absmax is None:
                k_absmax = K.float().norm(dim=-1).amax(dim=-1)
            use_mqa = (mqa is not False) and Hkv < H  # fuse shared-KV heads; kernel self-checks GF-divisibility
            fn = ext.flash_decode_defer_mqa if use_mqa else ext.flash_decode_defer
            O, lse = fn(Q, K, V, k_absmax, float(scale), int(num_splits))
        else:
            # Online path (running max, no U-bound): exact, and the default for a bare (uncached)
            # decode with Hkv >= 2 -- avoids the full-K max||k|| recompute the deferred path would need.
            O, lse = ext.flash_decode(Q, K, V, float(scale), int(num_splits), 0.0,
                                      torch.empty(0, dtype=torch.float32, device=Q.device))
        return (O[..., :d0].contiguous() if dp != d0 else O), lse
    if has_mod:
        raise NotImplementedError("fa2_sm70 decode: softcap/alibi_slopes are supported on the fp16 path only.")
    bits = 8 if kv_dtype == "int8" else 4
    cache = quantize_kv(K, V, bits)                 # int4 -> 2-level (per-channel) by default
    O, lse = decode_quantized(Q, cache, scale=scale, deferred=deferred, num_splits=num_splits)
    return (O[..., :d0].contiguous() if dp != d0 else O), lse


def _paged_kmax(Kcache, block_table, block_size, seqlens):
    """Deferred-U bound per (b, kv-head): max_j ||k_j|| over each sequence's valid keys, gathered
    through the block table. `seqlens` is a per-sequence length list/tensor. [B, Hkv] fp32.
    (A serving engine keeps this incrementally, like KVCache.)"""
    B, Hkv = block_table.shape[0], Kcache.shape[2]
    out = torch.zeros(B, Hkv, device=Kcache.device, dtype=torch.float32)
    for b in range(B):
        Lb = int(seqlens[b])
        if Lb == 0:
            continue
        parts = []
        for lb in range((Lb + block_size - 1) // block_size):
            pb = int(block_table[b, lb]); s = lb * block_size; e = min(s + block_size, Lb)
            parts.append(Kcache[pb, :e - s])                     # [tokens, Hkv, D]
        kb = torch.cat(parts, 0)                                 # [Lb, Hkv, D]
        out[b] = kb.float().norm(dim=-1).amax(dim=0)             # max_j ||k|| per kv-head
    return out


def decode_paged(Q, Kcache, Vcache, block_table, block_size, kv_len, scale=None, k_absmax=None,
                 num_splits=None, seqlens=None):
    """Paged autoregressive decode (Sq == 1): K/V live in a block pool indexed by a block table, so
    physically non-contiguous cache blocks are attended in logical order (serving / vLLM layout).

    Q [B, H, 1, D] fp16/bf16; Kcache, Vcache [num_blocks, block_size, Hkv, D] fp16 (Vcache may be
    Kcache for V==K layers -> the KV is read once); block_table [B, max_blocks] int32 maps each
    sequence's logical block to a physical block. `kv_len` = max valid length (sizes the split grid).
    `seqlens` (optional int32/list [B]) gives PER-SEQUENCE valid lengths for a ragged batch; if omitted
    all sequences use `kv_len`. D in {64, 128, 256, 512}. Returns (O fp32 [B, H, 1, D], LSE [B, H, 1])."""
    B, H = Q.shape[0], Q.shape[1]
    Hkv = Kcache.shape[2]
    if scale is None:
        scale = 1.0 / math.sqrt(Q.shape[-1])
    if num_splits is None:
        num_splits = _auto_splits(B, H, int(kv_len), gf=_pick_gf(Q.shape[-1], max(1, H // Hkv)))
    (Q, _, _), _ = _prep(Q, Kcache[:1], Kcache[:1])              # bf16 Q -> fp16 (cache assumed fp16)
    Q = Q.contiguous()
    if seqlens is None:
        sl = torch.full((B,), int(kv_len), device=Kcache.device, dtype=torch.int32)
        sl_arg = torch.empty(0, device=Kcache.device, dtype=torch.int32)   # empty -> kernel uses kv_len
    else:
        sl = torch.as_tensor(seqlens, device=Kcache.device, dtype=torch.int32)
        sl_arg = sl
    if k_absmax is None:
        k_absmax = _paged_kmax(Kcache, block_table, block_size, sl)
    O, lse = decode_ext().flash_decode_defer_mqa_paged(
        Q, Kcache, Vcache, k_absmax, block_table.to(torch.int32), int(block_size),
        float(scale), int(num_splits), int(kv_len), sl_arg)
    return O, lse


class KVCache:
    """Caller-owned stateful K/V cache for incremental decode (`Sq == 1`).

    Preallocates `[B, Hkv, max_len, D]` once. `append(k, v)` writes the new tokens into the next slots
    and maintains the deferred-decode certified-U scan (a running max of ``||k_j||`` per (B, Hkv))
    incrementally -- so decode never rescans the whole cache to recompute the bound. `decode(q, ...)`
    attends the query over the valid prefix `[:length]`. MQA/GQA needs **no per-step K/V expansion**
    (the decode kernel maps query heads to KV heads natively via `k_absmax`/head grouping).

    This is the fp16 (exact) cache; for a smaller footprint see `QuantizedKVCache` (int8/int4, keeps the
    quantized bytes across steps -- ~2x/~4x less cache memory and HBM read, int8 ~0.7% error).
    """

    def __init__(self, B, Hkv, max_len, D, device, dtype=torch.float16):
        self.K = torch.empty(B, Hkv, max_len, D, device=device, dtype=dtype)
        self.V = torch.empty(B, Hkv, max_len, D, device=device, dtype=dtype)
        self.k_absmax = torch.zeros(B, Hkv, device=device, dtype=torch.float32)   # running max ||k_j||
        self.length = 0
        self.max_len = max_len

    def append(self, k, v):
        """k, v: [B, Hkv, n, D] (fp16/bf16). Writes n new tokens; O(n) cache-update work, not O(length)."""
        n = k.shape[2]
        if self.length + n > self.max_len:
            raise ValueError(f"KVCache overflow: {self.length}+{n} > max_len {self.max_len}")
        k = k.to(self.K.dtype); v = v.to(self.V.dtype)
        self.K[:, :, self.length:self.length + n] = k
        self.V[:, :, self.length:self.length + n] = v
        self.k_absmax = torch.maximum(self.k_absmax, k.float().norm(dim=-1).amax(dim=-1))  # incremental U-scan
        self.length += n
        return self

    def decode(self, q, scale=None, num_splits=None):
        """Attend q [B, H, 1, D] over the valid cache prefix. ZERO-COPY: passes the preallocated cache
        buffer straight to the kernel with `kv_len = self.length` (the kernel loops the valid prefix but
        uses the physical `max_len` stride to index), so no per-step copy/rebuild of the cache. Uses the
        deferred + MQA/GQA-aware path (native head mapping, V==K reuse). Separates cache-update (append)
        from attention (here)."""
        B, H = q.shape[0], q.shape[1]
        if scale is None:
            scale = 1.0 / math.sqrt(q.shape[-1])
        if num_splits is None:
            num_splits = _auto_splits(B, H, self.length)
        qf = q.contiguous().to(torch.float16)
        O, lse = decode_ext().flash_decode_defer_mqa(
            qf, self.K, self.V, self.k_absmax, float(scale), int(num_splits), int(self.length))
        return O, lse.reshape(B, H, 1)


class QuantizedKVCache:
    """Stateful int8/int4 KV cache: keeps the QUANTIZED bytes across decode steps (~2x / ~4x less cache
    memory and HBM read than the fp16 `KVCache`). Per-token symmetric quantization only -- the 2-level
    per-channel scale is a global-over-tokens statistic that cannot be maintained incrementally without
    re-quantizing the whole cache, so it is a prompt-time (`quantize_kv`) optimization, not a streaming
    one. int8 is the practical serving choice (~0.7% relL2); int4 per-token is coarse.

    `append(k, v)` quantizes and stores the new tokens (O(n)) and maintains the deferred-U bound over the
    dequantized keys incrementally; `.decode(q, scale)` attends the valid prefix zero-copy (`kv_len`)."""

    def __init__(self, B, Hkv, max_len, D, device, bits=8):
        assert bits in (8, 4) and D % 2 == 0
        self.bits = bits; self.qmax = (1 << (bits - 1)) - 1
        kd = D if bits == 8 else D // 2
        qdt = torch.int8 if bits == 8 else torch.uint8
        self.Kq = torch.zeros(B, Hkv, max_len, kd, device=device, dtype=qdt)
        self.Vq = torch.zeros(B, Hkv, max_len, kd, device=device, dtype=qdt)
        self.K_scale = torch.zeros(B, Hkv, max_len, device=device, dtype=torch.float32)
        self.V_scale = torch.zeros(B, Hkv, max_len, device=device, dtype=torch.float32)
        self.k_absmax = torch.zeros(B, Hkv, device=device, dtype=torch.float32)   # max ||k_dequant||
        self.length = 0; self.max_len = max_len

    def _quant(self, X):                            # per-token symmetric; matches quantize_kv's q() exactly
        s = (X.abs().amax(-1, keepdim=True) / self.qmax).clamp_min(1e-8)
        qd = torch.round(X / s).clamp(-self.qmax, self.qmax)
        deq = qd * s
        if self.bits == 8:
            return qd.to(torch.int8), s.squeeze(-1).float(), deq
        qn = (qd + 8).to(torch.uint8)               # int4 packed 2/byte (zero-point 8), matches quantize_kv
        return (qn[..., 0::2] | (qn[..., 1::2] << 4)), s.squeeze(-1).float(), deq

    def append(self, k, v):
        """k, v: [B, Hkv, n, D]. Quantizes + stores n new tokens (O(n)); updates the deferred-U bound."""
        n = k.shape[2]
        if self.length + n > self.max_len:
            raise ValueError(f"QuantizedKVCache overflow: {self.length}+{n} > max_len {self.max_len}")
        # Quantize in the input dtype (fp16) so the stored bytes match a one-shot quantize_kv exactly.
        Kqn, Ksn, Kdq = self._quant(k); Vqn, Vsn, _ = self._quant(v)
        sl = slice(self.length, self.length + n)
        self.Kq[:, :, sl] = Kqn; self.Vq[:, :, sl] = Vqn
        self.K_scale[:, :, sl] = Ksn; self.V_scale[:, :, sl] = Vsn
        self.k_absmax = torch.maximum(self.k_absmax,
                                      Kdq.float().norm(dim=-1).amax(dim=-1))   # dequant-key U-scan (matches quantize_kv)
        self.length += n
        return self

    def decode(self, q, scale=None, num_splits=None):
        """Attend q [B, H, 1, D] over the valid quantized prefix (zero-copy via `kv_len`). Returns
        (O fp32 [B, H, 1, D], LSE)."""
        B, H = q.shape[0], q.shape[1]
        if scale is None:
            scale = 1.0 / math.sqrt(q.shape[-1])
        if num_splits is None:
            num_splits = _auto_splits(B, H, self.length)
        cache = dict(Kq=self.Kq, K_scale=self.K_scale, Vq=self.Vq, V_scale=self.V_scale,
                     k_absmax=self.k_absmax, k_chan=None, v_chan=None, bits=self.bits)
        return decode_quantized(q, cache, scale=scale, num_splits=int(num_splits), kv_len=self.length)


def _expand(t, H):                                  # per-channel scale [B,Hkv,D] -> query heads [B,H,1,D]
    Hkv = t.shape[1]
    t = t if Hkv == H else t.repeat_interleave(H // Hkv, 1)
    return t.unsqueeze(2)


def latent_attention(qt, C, causal=False, scale=None, B=None, force_decode=None):
    """Low-rank / MLA attention run entirely in the compressed latent space.

    When keys/values are k_j = A c_j, v_j = B c_j with a shared latent c_j in R^r (r < head_dim d, or
    an MLA KV latent), the exact attention needs only the r-wide latent: score = <A^T q, c_j> and
    O = B (sum_j w_j c_j / sum_j w_j). Pass the ABSORBED query qt = (A^T q) [B,H,S,r] (fold A into your
    Q-projection) and the latent C [B,Hkv,Sk,r]; this runs our attention at head_dim=r (register-
    resident for r<=128 -> NO output-accumulator round-trip) and returns the r-wide weighted latent.
    For prefill keep the output compressed and absorb B into the next projection (else the per-position
    B costs O(n*d) and dominates); pass B=[d,r] only if you really want it decompressed here.
    Exact; wins over full-d attention by ~d/r in FLOPs and traffic when r < d. `decode` (bool) forces
    the Sq==1 path; default: infer from qt.shape[2].
    """
    if scale is None:
        scale = 1.0 / math.sqrt(qt.shape[-1])           # NOTE: scale by r (=qt head dim), fold any beta into qt
    is_dec = (qt.shape[2] == 1) if force_decode is None else force_decode
    O, lse = (decode(qt, C, C, scale=scale) if is_dec
              else attention(qt, C, C, causal=causal, scale=scale))
    if B is not None:
        O = (O.float() @ B.t().float()).to(O.dtype)     # decompress once: O_d = O_r @ B^T  (usually skip -> absorb into W_O)
    return O, lse


def mla_attention(Q, C, Wk, Wv, causal=False, scale=None):
    """MLA / low-rank attention over a shared latent C, with the SAME output either way -- but the
    fast route differs by phase, and picking wrong costs ~10x. Q [B,H,Sq,d_h]
    per-head queries; C [B,1,Sk,r_lat] shared MQA latent; Wk,Wv [H,d_h,r_lat] per-head decompression.

    * PREFILL (Sq>1, compute-bound): DECOMPRESS the latent to per-head K,V at d_h (one cheap batched
      GEMM) and run our fast d_h kernel (14-16% roofline, register-resident O) -- beats attending at
      the large r_lat (6-8%) by the FLOP ratio d_h/r_lat and the higher roofline. Measured 1.4-3.2x
      over the absorbed route.
    * DECODE (Sq==1, memory-bound): ABSORB instead -- score in the latent (qt = Wk^T q, r_lat-wide),
      run our r_lat kernel over C read ONCE, decompress the r_lat-wide result by Wv. Materialising the
      H*Sk*d_h per-head K,V here would move H*d_h/r_lat x more bytes (16*128/512 = 4x) and dominates
      the GEMV -- measured 10x slower. So decode never reconstructs.

    Returns (O [B,H,Sq,d_h], LSE). scale defaults to 1/sqrt(d_h) (applied to the true q.k, which the
    absorption preserves exactly: scale*<q,k_j> = scale*<Wk^T q, c_j>)."""
    dh = Q.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(dh)
    Wk = Wk.to(Q.dtype); Wv = Wv.to(Q.dtype); C = C.to(Q.dtype)            # fp16 einsum -> HMMA (fp32-accum via cublas)
    if Q.shape[2] == 1:                                                     # DECODE: absorb, never materialize
        qt = torch.einsum("bhsd,hdr->bhsr", Q, Wk)                         # A^T q, r_lat-wide
        O_lat, lse = decode(qt, C, C, scale=scale)                         # weighted latent (r_lat-wide)
        O = torch.einsum("bhsr,hdr->bhsd", O_lat.to(Wv.dtype), Wv)         # decompress by Wv
        return O, lse
    K = torch.einsum("bosr,hdr->bhsd", C, Wk)                              # PREFILL: decompress k_j^h = Wk^h c_j
    V = torch.einsum("bosr,hdr->bhsd", C, Wv)
    return attention(Q, K, V, causal=causal, scale=scale)


def decode_quantized(Q, cache, scale=None, deferred=True, num_splits=None, kv_len=0):
    """Decode from a PRE-quantized cache (the serving pattern: quantize once, decode many).

    `cache` is the dict from quantize_kv. 2-level (per-channel) scales are folded here at zero kernel
    cost: the per-channel KEY scale cancels in <q,k> so we pre-scale the query, and the per-channel
    VALUE scale post-multiplies the output. deferred=True recommended. Returns (O fp32 [B,H,1,D], LSE).
    """
    Kq, K_scale, Vq, V_scale = cache["Kq"], cache["K_scale"], cache["Vq"], cache["V_scale"]
    k_absmax, k_chan, v_chan, bits = cache["k_absmax"], cache["k_chan"], cache["v_chan"], cache["bits"]
    B, H = Q.shape[0], Q.shape[1]
    Sk = K_scale.shape[2]
    if scale is None:
        scale = 1.0 / math.sqrt(Q.shape[-1])
    if num_splits is None:
        num_splits = _auto_splits(B, H, Sk)
    Qs = Q if k_chan is None else (Q.float() * _expand(k_chan, H)).to(Q.dtype)   # q' = q * per-channel key scale
    ext = decode_ext()
    if deferred:
        O, lse = ext.flash_decode_defer_q(Qs, Kq, K_scale, Vq, V_scale, k_absmax, float(scale), int(num_splits), int(bits), int(kv_len))
    else:
        O, lse = ext.flash_decode_q(Qs, Kq, K_scale, Vq, V_scale, float(scale), int(num_splits), int(bits), None, None)
    if v_chan is not None:
        O = O * _expand(v_chan, H)                   # O = O_kernel * per-channel value scale
    return O, lse


def quantize_kv_prefill(K, V, fmt=8):
    """Quantise a K/V cache into the BYTE store the `d=256` role kernel consumes fastest (format 8).

    `K`, `V` are `[B, Hkv, Sk, 256]` fp16/bf16. Returns a dict for `prefill_bytes(...)`:
    `Kb`/`Vb` are the **biased byte** (`u = v + 128`, stored as int8) and `scales` is a single
    **interleaved** `[B, Hkv, Sk, 2]` fp32 table holding `(k_scale, v_scale)` side by side per key
    position. Symmetric per-position (per-key-row) quantisation, `|q| <= 127`.

    WHY THESE TWO SHAPES, AND WHY THE QUANTISER DOES THIS RATHER THAN THE KERNEL. Both moves take work
    out of the inner loop permanently, at zero accuracy cost (measured `torch.equal` against the
    plain-int8 path):
      * the +128 bias means the kernel does NOT have to XOR the sign bit per element -- the expansion
        offset (1152) is the same either way, so the only question is who pays it, and the quantiser
        pays it once at write time instead of the mainloop paying it every tile;
      * interleaving the two scale tables means one `.64` load brings both values from ONE address.
        A SASS transcript of the separate-table variant shows the table costs +14 `LDG` but +36 `IMAD`
        and +11 `IADD3` -- the price was never the volume (0.8% of traffic), it was recomputing a
        64-bit address per access.
    Measured effect at `d=256`, causal, `Sk` 8K/32K/128K: **1.061-1.064x** over the shipped int8 path,
    bit-identical output, and at or slightly above the e5m2 byte route while carrying int8 accuracy
    (relL2 to fp32 7.0e-3 vs e5m2's 5.4e-2). See `docs/VOLTA_SM70.md` section 3b.
    """
    if fmt != 8:
        raise ValueError("quantize_kv_prefill: only format 8 is a shipped store (see docs/KERNELS.md)")
    if K.shape[-1] != 256 or V.shape[-1] != 256:
        raise ValueError("quantize_kv_prefill: head_dim must be 256 (the role kernel's instantiation)")

    def q(X):
        s = (X.float().abs().amax(-1, keepdim=True).clamp_min(6.1e-5) / 127.0)
        qd = torch.clamp(torch.round(X.float() / s), -127, 127).to(torch.int8)
        # biased byte: v + 128 in uint8 space, re-viewed as int8 so the kernel's tensor type is unchanged
        b = (qd.to(torch.int16) + 128).to(torch.uint8).view(torch.int8).contiguous()
        return b, s.squeeze(-1).float().contiguous()

    Kb, ks = q(K)
    Vb, vs = q(V)
    return dict(Kb=Kb, Vb=Vb, scales=torch.stack([ks, vs], dim=-1).contiguous(), fmt=8)


def prefill_bytes(Q, cache, causal=False, scale=None):
    """Exact prefill at `head_dim = 256` on a BYTE KV cache (the warp-role kernel).

    `Q` is `[B, Sq, H, 256]` fp16 in **BSHD** layout (note: not the `[B,H,Sq,D]` of `attention()` --
    this kernel is fed straight from the serving path, which already holds Q that way, and a transpose
    in the glue costs more than the kernel saves). `cache` comes from `quantize_kv_prefill(K, V)`.
    GQA/MQA via `Hkv < H`; requires `Sk >= Sq`. Returns `(O [B, Sq, H, 256] fp16, LSE fp32)`.

    The result is the sum a dequantised fp16 kernel would compute -- the kernel does not approximate;
    the quantisation is the caller's decision. `sm_70` has no int8 tensor instruction: the bytes reach
    the tensor cores by exact encoding into an fp16 mantissa, with the constant offset folded into a
    rank-1 correction (README section 3).
    """
    # ПОРЯДОК ПРОВЕРОК ЗНАЧИМ: сперва отвергнуть чужой кэш, и только потом трогать формы Q.
    # Иначе пустой/неверный тензор падает в `1/sqrt(0)` -- ZeroDivisionError вместо внятного отказа,
    # то есть вызывающий узнаёт про НЕ ТУ ошибку. Поймано `tests/test_prefill_bytes.py`.
    if cache.get("fmt") != 8:
        raise ValueError("prefill_bytes: cache must come from quantize_kv_prefill(...)")
    if Q.dim() != 4 or Q.shape[-1] != 256:
        raise ValueError("prefill_bytes: Q must be [B, Sq, H, 256] fp16 (BSHD)")
    if scale is None:
        scale = 1.0 / math.sqrt(Q.shape[-1])
    sc = cache["scales"]
    return prefill_ext().attn_fwd_volta_i8(Q.contiguous(), cache["Kb"], sc, cache["Vb"], sc,
                                           float(scale), bool(causal), 8)


def mxfp4_dequant(blocks, scales, dtype=None):
    """Dequantise MXFP4 (OCP Microscaling v1.0) weights on `sm_70`.

    `blocks` uint8 `[..., nblk, 16]` (16 bytes = 32 nibbles = one MX block of 32 elements),
    `scales` uint8 `[..., nblk]` (E8M0: value = 2^(s-127)). Returns `[..., nblk*32]` in `dtype`
    (fp16 by default, fp32 also supported).

    Why this exists: vLLM only provides an MXFP4 path for device capability 9.0-11.0, so an MXFP4
    checkpoint (gpt-oss) cannot load on Volta at all. Volta has no FP4 hardware, but none is needed
    -- the format is a 16-entry nibble table and a power-of-two scale. Verified BIT-EXACT against the
    reference in `transformers/integrations/mxfp4.py` on all 256 byte values and on real gpt-oss
    weights (`tests/test_mxfp4.py`).
    """
    import torch as _t
    if dtype is None:
        dtype = _t.float16
    return _ext.mxfp4_ext().mxfp4_dequant(blocks, scales, dtype)
