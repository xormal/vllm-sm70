# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# Скорость ядра нового фундамента (int8 K/V) против трёх fp16-путей. ОДНА методика на все четыре:
# один процесс, одно расширение, событийный замер, усечённое среднее, частоты зафиксированы.
# Сравнивать новое ядро из отдельной программы со старыми из торча НЕЛЬЗЯ -- разная обвязка даёт
# разный накладной расход, и разница читается как эффект.
#
# ЧЕСТНАЯ ОГОВОРКА: int8-путь считает НЕ ТО ЖЕ САМОЕ -- у него K/V квантованы. Поэтому рядом с
# временем печатается и расхождение с fp16-эталоном: скорость без точности здесь ничего не значит.
import os, sys, time, math, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fa2_sm70
from torch.nn.attention import sdpa_kernel, SDPBackend

D = 256
SHAPES = [(1, 4, 4, 4096, 4096), (1, 16, 4, 4096, 4096), (1, 4, 4, 8192, 8192),
          (1, 2, 2, 16384, 16384), (1, 24, 4, 8192, 16384), (1, 24, 4, 8192, 32768)]
ext = fa2_sm70._ext.prefill_ext()


def bench(fn, it=10):
    for _ in range(4): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(it):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    ts.sort(); return sum(ts[2:-2]) / max(1, len(ts[2:-2])) * 1e3


def quantize(x):                      # симметричное по позиции ключа, как в тесте корректности
    a = x.abs().amax(dim=-1, keepdim=True).float() / 127.0
    q = torch.clamp(torch.round(x.float() / a), -127, 127).to(torch.int8)
    return q.contiguous(), a.squeeze(-1).contiguous(), (q.float() * a).half()


print(f"head_dim={D}, causal, V100. Время в мс; в скобках -- ускорение к нашему fp16-ядру.\n")
print(f"{'форма (B,H,Sq,Sk)':<26}{'наш fp16':>12}{'наш int8':>16}{'cutlass':>12}{'SDPA-eff':>12}{'отн.ошибка i8':>15}")
for B, H, Hkv, Sq, Sk in SHAPES:
    q = (torch.randn(B, Sq, H, D, device='cuda', dtype=torch.float16) * 0.5)
    k = (torch.randn(B, Hkv, Sk, D, device='cuda', dtype=torch.float16) * 0.5)
    v = (torch.randn(B, Hkv, Sk, D, device='cuda', dtype=torch.float16) * 0.5)
    kq, ks, kd = quantize(k); vq, vs, vd = quantize(v)
    sc = D ** -0.5
    t_f16 = bench(lambda: ext.attn_fwd_volta(q, k, v, sc, True))
    t_i8  = bench(lambda: ext.attn_fwd_volta_i8(q, kq, ks, vq, vs, sc, True))
    try:    t_cut = bench(lambda: ext.attn_fwd_qbshd(q, k, v, sc, True))
    except Exception: t_cut = float('nan')
    # SDPA: ему нужен BHSD и Sq==Sk для честной причинности (см. bench_prefill_d256)
    if Sq == Sk:
        qb = q.transpose(1, 2).contiguous()
        kb = k.repeat_interleave(H // Hkv, dim=1); vb = v.repeat_interleave(H // Hkv, dim=1)
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            t_sdpa = bench(lambda: torch.nn.functional.scaled_dot_product_attention(qb, kb, vb, is_causal=True))
    else:
        t_sdpa = float('nan')          # top-left causal -- считает другое, в таблицу не ставим
    # точность int8 против fp16-ядра НА ДЕКВАНТОВАННЫХ входах отдельно от скорости
    o16, _ = ext.attn_fwd_volta(q, kd, vd, sc, True)
    o8, _ = ext.attn_fwd_volta_i8(q, kq, ks, vq, vs, sc, True)
    err = (o8.float() - o16.float()).abs().max().item() / (o16.float().abs().max().item() + 1e-9)
    f = lambda t: f"{t:.3f}" if t == t else "  --  "
    print(f"{str((B,H,Sq,Sk)):<26}{f(t_f16):>12}{f(t_i8)+f' (x{t_f16/t_i8:.2f})':>16}"
          f"{f(t_cut):>12}{f(t_sdpa):>12}{err:>15.2e}")
