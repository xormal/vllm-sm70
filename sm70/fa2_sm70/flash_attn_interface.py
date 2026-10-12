# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# FlashAttention-compatible convenience API (drop-in-ish for `flash_attn`), mapping the standard
# calling conventions onto this package's sm_70 kernels. The key difference from our core API is
# LAYOUT: flash-attn uses [B, S, H, D] (sequence before heads); our kernels use [B, H, S, D]. These
# wrappers transpose, forward the modifiers, and transpose back -- preserving autograd where the
# backend supports it. They intentionally keep this package's explicit sm_70 limitations rather than
# silently emulating unavailable features.
import math
import torch
from . import attention as _attention, flash_attn_func as _faf, flash_attn_varlen_func as _fav


def _win_map(window_size):
    """Map flash-attn window_size=(left, right) to our (window, window_right). (-1,-1) = no window.
    window = left+1 (past keys incl. self); window_right = right (>=0 = two-sided, -1 = one-sided/causal)."""
    if window_size is None or tuple(window_size) == (-1, -1):
        return 0, -1
    left, right = int(window_size[0]), int(window_size[1])
    window = left + 1 if left >= 0 else 0
    return window, (right if right >= 0 else -1)


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
                    window_size=(-1, -1), softcap=0.0, alibi_slopes=None, return_attn_probs=False):
    """flash-attn-style attention. q [B, Sq, H, D]; k, v [B, Sk, Hkv, D] (GQA/MQA). Returns O
    [B, Sq, H, D] (and LSE [B, H, Sq] if return_attn_probs). window_size=(left, right) supports both
    causal past-windows (right=0/-1) and two-sided bidirectional windows (right>0). Modifiers
    (window/softcap/alibi) are inference-only on sm_70; the plain (causal, dropout) path is differentiable."""
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    window, window_right = _win_map(window_size)
    qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)      # [B,S,H,D] -> [B,H,S,D]
    modifiers = window or (window_right >= 0) or softcap or (alibi_slopes is not None)
    if modifiers:
        # windowed / soft-capped / ALiBi -> exact inference kernel (no backward for these on sm_70)
        O, L = _attention(qt, kt, vt, causal=causal, scale=softmax_scale, window=window,
                          window_right=window_right, softcap=softcap, alibi_slopes=alibi_slopes)
    else:
        O = _faf(qt, kt, vt, causal=causal, scale=softmax_scale, dropout_p=dropout_p)  # differentiable
        L = None
    O = O.transpose(1, 2).contiguous()                                        # -> [B,Sq,H,D]
    if return_attn_probs:
        if L is None:
            _, L = _attention(qt, kt, vt, causal=causal, scale=softmax_scale)
        return O, L, None
    return O


def flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None, causal=False,
                              window_size=(-1, -1), softcap=0.0, alibi_slopes=None, return_attn_probs=False):
    """qkv packed [B, S, 3, H, D] (self-attention). Returns O [B, S, H, D]."""
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    return flash_attn_func(q, k, v, dropout_p, softmax_scale, causal, window_size, softcap,
                           alibi_slopes, return_attn_probs)


def flash_attn_kvpacked_func(q, kv, dropout_p=0.0, softmax_scale=None, causal=False,
                             window_size=(-1, -1), softcap=0.0, alibi_slopes=None, return_attn_probs=False):
    """q [B, Sq, H, D], kv packed [B, Sk, 2, Hkv, D] (cross-attention). Returns O [B, Sq, H, D]."""
    k, v = kv[:, :, 0], kv[:, :, 1]
    return flash_attn_func(q, k, v, dropout_p, softmax_scale, causal, window_size, softcap,
                           alibi_slopes, return_attn_probs)


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                           dropout_p=0.0, softmax_scale=None, causal=False,
                           window_size=(-1, -1), softcap=0.0, alibi_slopes=None, return_attn_probs=False):
    """Variable-length packed attention. q [total_q, H, D]; k, v [total_k, Hkv, D]; int32 cu_seqlens
    [nseq+1]. Differentiable (causal). window/softcap/alibi not supported for varlen on sm_70."""
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    if (window_size is not None and tuple(window_size) != (-1, -1)) or softcap or (alibi_slopes is not None):
        raise NotImplementedError("fa2_sm70 varlen: window/softcap/alibi modifiers are not supported.")
    if dropout_p:
        raise NotImplementedError("fa2_sm70 varlen: dropout is not supported (use the padded path).")
    O = _fav(q, k, v, cu_seqlens_q, cu_seqlens_k, int(max_seqlen_q), int(max_seqlen_k),
             causal=causal, scale=softmax_scale)
    if return_attn_probs:
        return O, None, None
    return O
