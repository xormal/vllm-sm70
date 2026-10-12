# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# Чанковый скан gated-delta-net: наши пять ядер против ЭТАЛОНА ПО ОПРЕДЕЛЕНИЮ.
#
# Эталон -- ПОСЛЕДОВАТЕЛЬНАЯ рекуррентность в fp64 на torch, а НЕ трионовская чанковая цепочка.
# Причина: чанковая схема и наши ядра считают одно и то же ДРУГИМ способом (WY-представление,
# обращение треугольной матрицы, межчанковое состояние), и сверка двух чанковых реализаций
# пропустила бы общую ошибку в понимании самой схемы. Рекуррентная форма -- это определение.
#
#     S_0 = h0
#     для t:  S = a_t * S ;  S = S + beta_t * (v_t - S k_t) k_t^T ;  o_t = scale * S q_t
# где a_t = exp(g_t) -- гейт забывания, k нормирован по l2.
#
# Запуск:  python tests/test_gdn.py
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fa2_sm70 import gdn as G  # noqa: E402


def reference(q, k, v, g, beta, h0, scale):
    """Рекуррентное определение в fp64. B=1, вход [1,T,H*,D]; GQA -- голова ключа = h // (H/Hg)."""
    T, Hg, D = q.shape[1], q.shape[2], q.shape[3]
    H = v.shape[2]
    q, k, v = q.double(), k.double(), v.double()
    beta, g = beta.double(), g.double()
    o = torch.zeros(1, T, H, D, dtype=torch.float64, device=q.device)
    S = h0.double().clone()                                    # [1,H,V,K]
    for h in range(H):
        hk = h // (H // Hg)
        s = S[0, h]                                            # [V,K]
        for t in range(T):
            s = s * torch.exp(g[0, t, h])
            kt = k[0, t, hk]                                   # [K]
            err = v[0, t, h] - s @ kt                          # [V]
            s = s + beta[0, t, h] * torch.outer(err, kt)
            o[0, t, h] = (s @ q[0, t, hk]) * scale
        S[0, h] = s
    return o, S


def l2n(x):
    return x / x.float().pow(2).sum(-1, keepdim=True).sqrt().to(x.dtype).clamp_min(1e-6)


def main():
    from vllm.model_executor.layers.fla.ops.cumsum import chunk_local_cumsum

    ok = True
    # Формы: боевая GQA=3 в миниатюре, вырожденная H==Hg, ХВОСТ не кратный чанку, и varlen из
    # трёх последовательностей разной длины. Каждая закрывает свой класс адресации.
    for T, H, Hg in ((128, 6, 2), (192, 4, 4), (200, 6, 2), (320, 3, 3)):
        D = 128
        torch.manual_seed(T * 31 + H)
        dev = "cuda"
        q = l2n(torch.randn(1, T, Hg, D, device=dev, dtype=torch.float16) * 0.5)
        k = l2n(torch.randn(1, T, Hg, D, device=dev, dtype=torch.float16) * 0.5)
        v = torch.randn(1, T, H, D, device=dev, dtype=torch.float16) * 0.5
        beta = torch.rand(1, T, H, device=dev, dtype=torch.float16) * 0.9 + 0.05
        graw = torch.nn.functional.logsigmoid(
            torch.rand(1, T, H, device=dev, dtype=torch.float32))
        h0 = torch.randn(1, H, D, D, device=dev, dtype=torch.float32) * 0.05
        cu = torch.tensor([0, T], device=dev, dtype=torch.long)
        scale = D ** -0.5

        gc = chunk_local_cumsum(graw, chunk_size=64, cu_seqlens=cu)
        o, ht = G.chunk_gated_delta_rule_volta(
            q.contiguous(), k.contiguous(), v, gc, beta, scale, h0, True, cu, torch.float16)
        o_ref, ht_ref = reference(q, k, v, graw, beta, h0, scale)

        rel_o = ((o.double() - o_ref).norm() / o_ref.norm()).item()
        rel_h = ((ht.double() - ht_ref).norm() / ht_ref.norm()).item()
        nan = int(torch.isnan(o.float()).sum())
        # Порог АНАЛИТИЧЕСКИЙ: цепочка из трёх fp16-округлений (A, состояние, выход) при свёртках
        # длиной 128; RMS одного округления 2^-11/sqrt(3) = 2.8e-04, накопление по sqrt(3) даёт
        # ~5e-04. Берём 2e-03 -- вчетверо, но НЕ подгоняя под результат.
        good = rel_o < 2e-3 and rel_h < 2e-3 and nan == 0
        ok &= good
        print(f"  T={T:4d} H={H} Hg={Hg}: o relL2={rel_o:.2e}  ht relL2={rel_h:.2e}  "
              f"NaN={nan}  {'OK' if good else 'FAIL'}")

    # VARLEN: последовательности разной длины. Сверяется с рекуррентным эталоном ПО КАЖДОЙ
    # отдельно -- иначе ошибка в базе bos прошла бы незамеченной.
    # ДЛИНА 1 ЗДЕСЬ ОБЯЗАТЕЛЬНА, и это не крайний случай ради полноты: в vLLM при СМЕШАННОМ батче
    # (есть и префилл, и декод) через chunk_gated_delta_rule идут ВСЕ не-спекулятивные токены, и
    # декодирующие последовательности приходят сюда длиной 1. Не покрыв её, мы бы поймали отказ
    # только в бою и только под нагрузкой определённой формы.
    lens = [128, 1, 40, 200, 64]
    D, H, Hg = 128, 6, 2
    Tt = sum(lens)
    torch.manual_seed(77)
    dev = "cuda"
    q = l2n(torch.randn(1, Tt, Hg, D, device=dev, dtype=torch.float16) * 0.5)
    k = l2n(torch.randn(1, Tt, Hg, D, device=dev, dtype=torch.float16) * 0.5)
    v = torch.randn(1, Tt, H, D, device=dev, dtype=torch.float16) * 0.5
    beta = torch.rand(1, Tt, H, device=dev, dtype=torch.float16) * 0.9 + 0.05
    graw = torch.nn.functional.logsigmoid(torch.rand(1, Tt, H, device=dev, dtype=torch.float32))
    h0 = torch.randn(len(lens), H, D, D, device=dev, dtype=torch.float32) * 0.05
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), device=dev, dtype=torch.long)
    scale = D ** -0.5
    from vllm.model_executor.layers.fla.ops.cumsum import chunk_local_cumsum
    gc = chunk_local_cumsum(graw, chunk_size=64, cu_seqlens=cu)
    o, ht = G.chunk_gated_delta_rule_volta(
        q.contiguous(), k.contiguous(), v, gc, beta, scale, h0, True, cu, torch.float16)
    worst = 0.0
    for n in range(len(lens)):
        a, b = int(cu[n]), int(cu[n + 1])
        o_ref, ht_ref = reference(q[:, a:b], k[:, a:b], v[:, a:b], graw[:, a:b], beta[:, a:b],
                                  h0[n:n + 1], scale)
        r1 = ((o[:, a:b].double() - o_ref).norm() / o_ref.norm()).item()
        r2 = ((ht[n:n + 1].double() - ht_ref).norm() / ht_ref.norm()).item()
        worst = max(worst, r1, r2)
        print(f"  varlen посл.{n} len={lens[n]:3d}: o relL2={r1:.2e}  ht relL2={r2:.2e}")
    ok &= worst < 2e-3

    print("GDN scan:", "ALL OK" if ok else "FAILED")
    assert ok


if __name__ == "__main__":
    main()
