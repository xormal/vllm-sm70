"""СЖАТИЕ ПЛОТНОГО fp16-ВЕСА В РАСКЛАДКУ GPTQ-8 (нашу, читаемую `w8a16_gemv`).

ЗАЧЕМ. Словарная проекция `lm_head` -- ПОСЛЕДНИЙ крупный fp16-вес в сети, где всё остальное уже
int8: 5120 x 124160 x 2 Б = 1.27 ГБ читается на КАЖДЫЙ токен. Замер по ленте (docs/SPEC_TURBOMIND.md)
показал, что чужое тело стоит на 98.9 % замеренной крыши полосы, то есть более быстрого ядра в
fp16 не существует и единственный рычаг -- УМЕНЬШИТЬ ЧИСЛО БАЙТ.

РАСКЛАДКА -- РОВНО ТА, ЧТО УЖЕ ЧИТАЕТ НАШЕ ЯДРО, чтобы не заводить второй формат:
    qweight [K/4, N] int32   -- байт i слова несёт строку k = 4*k4 + i
    qzeros  [K/G, N/4] int32 -- байт i слова несёт столбец n = 4*n4 + i
    scales  [K/G, N] fp16
    развёртка: w[k, n] = (q[k, n] - z[k/G, n]) * s[k/G, n]

УСЛОВНОСТЬ НУЛЯ ЗДЕСЬ **v2** (хранится сам z, не z-1) и передаётся ядру ЯВНО. Умолчания нет
намеренно: молчаливое умолчание v2 при слепке v1 и было тем тихим дефектом на 1.4 % (см.
[[fa2-sm70-uslovnost-nulya-gptq-defekt]]). Мы этот вес квантуем САМИ, поэтому вольны выбрать v2,
но обязаны сказать это вслух на каждом вызове.
"""
from __future__ import annotations

import torch

__all__ = ["kvantovat_gptq8", "razvernut_gptq8", "bajt_gptq8"]


def kvantovat_gptq8(w: torch.Tensor, G: int = 128, kusok: int = 8192):
    """w [N, K] fp16 (раскладка nn.Linear) -> (qweight [K/4,N], qzeros [K/G,N/4], scales [K/G,N]).

    Асимметричное поблочное квантование по K группами длины G. Идём кусками по N, потому что
    промежуточный int32-тензор на всю матрицу словаря весит 2.5 ГБ -- вчетверо больше результата.

    Возвращаемые тензоры лежат на том же устройстве, что и `w`.
    """
    assert w.dim() == 2, "ждём [N, K]"
    N, K = w.shape
    assert K % G == 0, f"K={K} не кратно группе {G}"
    assert K % 4 == 0 and N % 4 == 0, "упаковка требует K%4==0 и N%4==0"
    ng = K // G
    dev = w.device

    qweight = torch.empty(K // 4, N, dtype=torch.int32, device=dev)
    qzeros = torch.empty(ng, N // 4, dtype=torch.int32, device=dev)
    scales = torch.empty(ng, N, dtype=torch.float16, device=dev)

    for n0 in range(0, N, kusok):
        n1 = min(N, n0 + kusok)
        ws = w[n0:n1].float().view(n1 - n0, ng, G)          # [n, ng, G]
        mn = ws.amin(dim=2)
        mx = ws.amax(dim=2)
        s = (mx - mn) / 255.0
        s = torch.where(s > 1e-12, s, torch.full_like(s, 1e-12))
        z = torch.clamp(torch.round(-mn / s), 0, 255)
        q = torch.clamp(torch.round(ws / s.unsqueeze(2)) + z.unsqueeze(2), 0, 255).to(torch.int32)

        scales[:, n0:n1] = s.to(torch.float16).t()
        # qweight[k4, n] = сумма q[n, k] << 8*(k%4), k = 4*k4 + i
        qk = q.view(n1 - n0, K).t().contiguous()             # [K, n]
        упак = torch.zeros(K // 4, n1 - n0, dtype=torch.int32, device=dev)
        for i in range(4):
            упак |= (qk[i::4] & 0xFF) << (8 * i)
        qweight[:, n0:n1] = упак
        # qzeros[g, n4] = сумма z[n, g] << 8*(n%4)
        zi = z.to(torch.int32).t().contiguous()              # [ng, n]
        assert (n1 - n0) % 4 == 0, "кусок по N обязан быть кратен 4 -- иначе слово нуля разорвётся"
        упакz = torch.zeros(ng, (n1 - n0) // 4, dtype=torch.int32, device=dev)
        for i in range(4):
            упакz |= (zi[:, i::4] & 0xFF) << (8 * i)
        qzeros[:, n0 // 4:n1 // 4] = упакz
        del ws, mn, mx, s, z, q, qk, упак, zi, упакz
    return qweight, qzeros, scales


def razvernut_gptq8(qweight, qzeros, scales, G: int = 128) -> torch.Tensor:
    """ОБРАТНАЯ развёртка в плотный fp32 [N, K] -- НЕЗАВИСИМЫЙ путь для сверки.

    Пишется по ОПИСАНИЮ формата, а не по коду ядра: сверка, читающая ту же реализацию, что и
    проверяемое, соглашается, будучи неверной (см. [[fa2-sm70-falsifier-symmetry]]).
    """
    K = qweight.shape[0] * 4
    N = qweight.shape[1]
    q = torch.empty(K, N, dtype=torch.int32, device=qweight.device)
    for i in range(4):
        q[i::4] = (qweight >> (8 * i)) & 0xFF
    z = torch.empty(K // G, N, dtype=torch.int32, device=qweight.device)
    for i in range(4):
        z[:, i::4] = (qzeros >> (8 * i)) & 0xFF
    w = (q.float() - z.repeat_interleave(G, 0).float()) * scales.float().repeat_interleave(G, 0)
    return w.t().contiguous()


def bajt_gptq8(N: int, K: int, G: int = 128) -> int:
    """Сколько байт читает проход по такому весу -- для сравнения с fp16 (2*N*K)."""
    return K * N + (K // G) * N * 2 + (K // G) * (N // 4) * 4
