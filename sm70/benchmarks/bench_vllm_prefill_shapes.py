# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# Что даст ядро ролей ПРЕФИЛЛУ vLLM -- замер на ТЕХ ЖЕ формах, что реально приходят из сервера.
#
# Путь префилла в обвязке (integrations/vllm_sm70/flash_attn_v100/__init__.py):
#     gather_paged_kv   -- собирает страничный e4m3-пул и ДЕКВАНТУЕТ его в fp16 [Hkv, T, d]
#     attn_fwd_qbshd    -- fp16-внимание по BSHD-запросу
# Ядро ролей заменяет ВТОРОЙ шаг, и -- что важнее -- снимает надобность в деквантовании на первом:
# байты можно подавать на тензорные ядра прямо. Поэтому меряем ОБА шага, а не только внимание.
#
# Формы: TP=2 у Qwen3.6-27B даёт на карту H=12, Hkv=2, d=256; чанк префилла 16384 токенов,
# контекст T растёт от чанка к чанку. Раскладка входов совпадает с той, что вызывает обвязка.
#
# Запуск: CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_vllm_prefill_shapes.py
import json, math, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fa2_sm70  # noqa: E402

D, H, HKV = 256, 12, 2
CASES = [(4096, 4096), (16384, 16384), (16384, 32768), (16384, 65536), (16384, 131072)]
BLK = 16                                        # размер страницы в пуле vLLM


def bench(fn, it=10, warm=4):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(it):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return sum(ts[2:-2]) / len(ts[2:-2]) * 1e3


def qz(x):
    """Симметричное по позиции ключа -- ровно то, что принимает attn_fwd_volta_i8."""
    a = x.abs().amax(-1, keepdim=True).float() / 127.0
    q = torch.clamp(torch.round(x.float() / a), -127, 127).to(torch.int8)
    return q.contiguous(), a.squeeze(-1).contiguous()


ext, dext = fa2_sm70._ext.prefill_ext(), fa2_sm70._ext.decode_ext()
rows = []
for qlen, T in CASES:
    torch.manual_seed(T)
    sc = 1.0 / math.sqrt(D)
    q = (torch.randn(1, qlen, H, D, device="cuda", dtype=torch.float16) * 0.3).contiguous()
    k = (torch.randn(1, HKV, T, D, device="cuda", dtype=torch.float16) * 0.3).contiguous()
    v = (torch.randn(1, HKV, T, D, device="cuda", dtype=torch.float16) * 0.3).contiguous()
    kq, ks = qz(k); vq, vs = qz(v)

    r = {"qlen": qlen, "T": T}
    r["attn_fp16"] = bench(lambda: ext.attn_fwd_qbshd(q, k, v, sc, True))
    os.environ["FA2SM70_WS"] = "1"
    r["attn_roles_i8"] = bench(lambda: ext.attn_fwd_volta_i8(q, kq, ks, vq, vs, sc, True))

    # ШАГ СБОРКИ: то же, что делает обвязка -- деквантование страничного e4m3-пула в fp16.
    nblk = (T + BLK - 1) // BLK
    kc = torch.randint(0, 255, (nblk, BLK, HKV, D), device="cuda", dtype=torch.uint8)
    vc = torch.randint(0, 255, (nblk, BLK, HKV, D), device="cuda", dtype=torch.uint8)
    bt = torch.arange(nblk, device="cuda", dtype=torch.int32)
    kb = torch.empty(HKV, T, D, device="cuda", dtype=torch.float16)
    vb = torch.empty(HKV, T, D, device="cuda", dtype=torch.float16)
    r["gather"] = bench(lambda: dext.gather_paged_kv(kc, vc, bt, T, 1.0, 1.0, kb, vb))

    r["now"] = r["gather"] + r["attn_fp16"]
    r["roles_same_gather"] = r["gather"] + r["attn_roles_i8"]
    # Верхняя оценка при подключённом байтовом кэше: сборка перекладывает БАЙТЫ, а не разворачивает
    # их в fp16, поэтому пишет вдвое меньше. Считаем её пропорционально записанным байтам -- это
    # ОЦЕНКА, а не замер, и помечена как таковая.
    r["roles_byte_gather_est"] = r["gather"] * 0.5 + r["attn_roles_i8"]
    rows.append(r)
    print(f"qlen={qlen:6d} T={T:7d} | сборка {r['gather']:7.2f} | fp16 {r['attn_fp16']:8.2f} | "
          f"роли-i8 {r['attn_roles_i8']:8.2f} ({r['attn_fp16']/r['attn_roles_i8']:.2f}x) | "
          f"слой сейчас {r['now']:8.2f} -> {r['roles_same_gather']:8.2f} "
          f"({r['now']/r['roles_same_gather']:.2f}x), с байтовой сборкой ~{r['roles_byte_gather_est']:8.2f} "
          f"({r['now']/r['roles_byte_gather_est']:.2f}x)")

results_path = os.environ.get("RESULTS_PATH")
if results_path:
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    print(f"\nresults: {results_path}")
else:
    print("\nresults were printed only; set RESULTS_PATH to save JSON outside the repository")
