# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# ПУБЛИЧНЫЙ БАЙТОВЫЙ ПРЕФИЛЛ (quantize_kv_prefill + prefill_bytes, формат 8).
#
# Проверяется ровно то, ради чего формат 8 отгружен, и НЕ проверяется скорость -- она меряется
# секундомером в benchmarks/bench_slots_ab.py, а не утверждением в тесте:
#   1. ПОБИТОВОЕ равенство сырому пути формата 0 на тех же значениях. Это главное: смещённый байт и
#      чередование таблицы -- перестановка работы между квантователем и ядром, а не другая
#      арифметика. Если равенство сломалось, значит поехало кодирование, а не «немного точность».
#   2. Согласие с fp32-оракулом в пределах ошибки САМОГО КВАНТОВАНИЯ (а не ошибки ядра): порог
#      2e-2 при замеренных ~7e-3 на iid-данных.
#   3. Формы и раскладки: Q в BSHD, K/V в BHSD, GQA, Sk > Sq (чанковый префилл).
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fa2_sm70  # noqa: E402

D = 256
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="нужна CUDA-карта")


def _ref(q_bshd, k, v, sc, causal):
    """fp32-оракул на ДЕКВАНТОВАННЫХ значениях: Q [B,Sq,H,D] -> O [B,Sq,H,D]."""
    B, Sq, H, d = q_bshd.shape
    Hkv, Sk = k.shape[1], k.shape[2]
    r = H // Hkv
    qq = q_bshd.float().permute(0, 2, 1, 3)
    kk = k.float().repeat_interleave(r, dim=1)
    vv = v.float().repeat_interleave(r, dim=1)
    s = torch.matmul(qq, kk.transpose(-1, -2)) * sc
    if causal:
        off = Sk - Sq
        i = torch.arange(Sq, device=q_bshd.device).view(-1, 1) + off
        j = torch.arange(Sk, device=q_bshd.device).view(1, -1)
        s = s.masked_fill(j > i, float("-inf"))
    p = torch.softmax(s, dim=-1)
    return torch.matmul(p, vv).permute(0, 2, 1, 3)


def _rel(a, b):
    a, b = a.float(), b.float()
    return (torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b)).item()


@pytest.mark.parametrize("B,Sq,Sk,H,Hkv,causal", [
    (1, 256, 256, 8, 8, True),        # MHA, квадрат
    (1, 128, 1024, 8, 2, True),       # GQA 4:1, чанк (Sk > Sq)
    (2, 256, 512, 4, 1, True),        # MQA, батч
    (1, 192, 512, 4, 2, False),       # без причинной маски, Sq не кратен плитке
])
def test_bytes_matches_raw_int8_bitwise(B, Sq, Sk, H, Hkv, causal):
    torch.manual_seed(0)
    dev = "cuda"
    q = (torch.randn(B, Sq, H, D, device=dev, dtype=torch.float16) * 0.3).contiguous()
    k = (torch.randn(B, Hkv, Sk, D, device=dev, dtype=torch.float16) * 0.4).contiguous()
    v = (torch.randn(B, Hkv, Sk, D, device=dev, dtype=torch.float16) * 0.4).contiguous()
    sc = 1.0 / math.sqrt(D)

    cache = fa2_sm70.quantize_kv_prefill(k, v)
    O8, lse8 = fa2_sm70.prefill_bytes(q, cache, causal=causal, scale=sc)

    # тот же кэш, но поданный СЫРЫМ путём формата 0: знаковый байт + две раздельные таблицы
    kq = (cache["Kb"].view(torch.uint8).to(torch.int16) - 128).to(torch.int8).contiguous()
    vq = (cache["Vb"].view(torch.uint8).to(torch.int16) - 128).to(torch.int8).contiguous()
    ks = cache["scales"][..., 0].contiguous()
    vs = cache["scales"][..., 1].contiguous()
    O0, lse0 = fa2_sm70._ext.prefill_ext().attn_fwd_volta_i8(q, kq, ks, vq, vs, sc, causal, 0)

    assert torch.equal(O8, O0), "формат 8 обязан быть ПОБИТОВО равен формату 0 на тех же значениях"
    assert torch.equal(lse8, lse0)

    # и согласие с оракулом на деквантованных значениях -- в пределах ошибки квантования
    kdq = (kq.float() * ks.unsqueeze(-1)).half()
    vdq = (vq.float() * vs.unsqueeze(-1)).half()
    err = _rel(O8, _ref(q, kdq, vdq, sc, causal))
    assert err < 2e-2, f"relL2 к деквантованному оракулу {err:.3e}"


def test_bytes_shapes_and_guards():
    dev = "cuda"
    k = torch.randn(1, 2, 128, D, device=dev, dtype=torch.float16)
    cache = fa2_sm70.quantize_kv_prefill(k, k)
    assert cache["Kb"].dtype is torch.int8 and cache["Kb"].shape == (1, 2, 128, D)
    assert cache["scales"].shape == (1, 2, 128, 2) and cache["scales"].dtype is torch.float32
    # чередование -- это ровно (k_scale, v_scale) рядом, а не два независимых тензора
    assert torch.equal(cache["scales"][..., 0], cache["scales"][..., 1])   # здесь K == V
    with pytest.raises(ValueError):
        fa2_sm70.quantize_kv_prefill(k[..., :128], k[..., :128])           # head_dim != 256
    with pytest.raises(ValueError):
        fa2_sm70.prefill_bytes(torch.empty(0), {"fmt": 0})                 # чужой кэш
