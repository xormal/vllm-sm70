"""ОЧНАЯ СТАВКА: НАШЕ УМНОЖЕНИЕ НА ВЕСА ПРОТИВ ЧУЖОГО, НА БОЕВЫХ ФОРМАХ Qwen3.8.

ЗАЧЕМ. Опись декода (16.08): чужое ядро `vllm::gptq::gemm_half_q_half_gptq_8bit` держит **50.6 %
шага**, и пока умножение чужое, путь нельзя ни слить, ни оценить. Крыша у него посчитана --
741.7 из 878 ГБ/с, то есть 84.5 %, -- значит цель подмены НЕ в скорости самого тела, а во
ВЛАДЕНИИ: своё тело можно сливать с соседями, перекладывать и мерить фазами. Но подменять вслепую
нельзя: сперва надо знать, теряем мы на этом или нет.

ФОРМЫ -- НЕ КРУГЛЫЕ, А ИЗ СЛЕПКА МОДЕЛИ (`*.safetensors`, поле qweight), с числом слоёв каждого
вида. Умножение на весах идёт при M=1 (декод), поэтому связывает ЧТЕНИЕ ВЕСОВ, и мера здесь --
достигнутая полоса, а не ТФЛОП/с.

ПОРЯДОК КАК ВСЕГДА: сверка ответа ПЕРВОЙ (иначе быстрый неверный вариант выиграет), потом парные
отношения с чередованием внутри раунда.

    python benchmarks/bench_svoy_put.py [--m 1,2,4]
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fa2_sm70  # noqa: E402

# (имя, K, N, сколько таких слоёв в сети) -- снято со слепка Qwen3.8-27B_int8_i12
ФОРМЫ = [
    ("mlp.down_proj",           17408,  5120, 63),
    ("mlp.gate_proj",            5120, 17408, 63),
    ("mlp.up_proj",              5120, 17408, 63),
    ("linear_attn.in_proj_qkv",  5120, 10240, 47),
    ("linear_attn.in_proj_z",    5120,  6144, 47),
    ("linear_attn.out_proj",     6144,  5120, 47),
    ("self_attn.o_proj",         6144,  5120, 16),
]
G = 128            # group_size из quantization_config
ПОЛОСА = 878.0     # ГБ/с, ЗАМЕРЕНО на этой карте (не паспорт)


def эталон_fp32(x, qw, qz, sc, G, v2=False):
    """РАЗВОРОТ ВЕСА В TORCH И УМНОЖЕНИЕ В fp32 -- ТРЕТИЙ, НЕЗАВИСИМЫЙ ПУТЬ.

    Без него сверка «наше против чужого» отвечает только на вопрос «одинаковы ли они», а нужен
    ответ «кто из них прав». Раскладка GPTQ-8: qweight [K/4, N] int32 -- по 4 строки k в слове;
    qzeros [K/G, N/4] int32 -- по 4 столбца n в слове; scales [K/G, N] fp16. В условности v1
    хранится (z-1), поэтому истинный ноль = хранимое + 1 -- перепутать это значит получить
    систематический сдвиг, который на случайных данных читается как «просто другой ответ».
    """
    K = qw.shape[0] * 4
    N = qw.shape[1]
    q = torch.empty(K, N, dtype=torch.int32, device=qw.device)
    for i in range(4):
        q[i::4] = (qw >> (8 * i)) & 0xFF
    z = torch.empty(K // G, N, dtype=torch.int32, device=qw.device)
    for i in range(4):
        z[:, i::4] = (qz >> (8 * i)) & 0xFF
    if not v2:
        z = z + 1
    w = (q.float() - z.repeat_interleave(G, 0).float()) * sc.float().repeat_interleave(G, 0)
    return (x.float() @ w)


def замер(f, n=11):
    for _ in range(5):
        f()
    torch.cuda.synchronize()
    т = []
    for _ in range(n):
        t0 = time.perf_counter(); f(); torch.cuda.synchronize()
        т.append(time.perf_counter() - t0)
    т.sort()
    return sum(т[3:-3]) / len(т[3:-3]) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", default="1")
    a = ap.parse_args()
    try:
        from vllm import _custom_ops as ops
    except Exception as e:  # noqa: BLE001
        print(f"vLLM не импортируется ({str(e)[:60]}) -- чужое тело недоступно, ставки нет")
        return 2
    ext = fa2_sm70._ext.w8a16_ext()
    torch.manual_seed(0)

    for M in [int(x) for x in a.m.split(",")]:
        print(f"\n=== M={M} (декод: M=1), группа {G}, 8 бит ===")
        print(f"{'слой':<26s} {'K':>6s} {'N':>6s} {'relL2':>9s} "
              f"{'чужое мс':>9s} {'наше мс':>9s} {'наша полоса':>12s} {'отношение':>10s}")
        итог_ч, итог_н = 0.0, 0.0
        for имя, K, N, штук in ФОРМЫ:
            x = (torch.randn(M, K, device="cuda") * 0.1).half()
            qw = torch.randint(-2**31, 2**31 - 1, (K // 4, N), dtype=torch.int32, device="cuda")
            qz = torch.randint(-2**31, 2**31 - 1, (K // G, N // 4), dtype=torch.int32, device="cuda")
            sc = (torch.rand(K // G, N, device="cuda") * 0.01 + 0.001).half()
            gi = torch.empty(0, dtype=torch.int32, device="cuda")   # desc_act=False -> g_idx ПУСТ
            # ЧУЖОЙ ПУТЬ ТРЕБУЕТ ПЕРЕСТАНОВКИ ВЕСА ПЕРЕД exllama. Без неё он считает мусор, и
            # сверка бы «прошла» ровно так же тихо, как у нас однажды прошёл формат 8.
            qw_ч = qw.clone()
            ops.gptq_shuffle(qw_ч, gi, 8)
            # use_v2_format=False -- УСЛОВНОСТЬ НУЛЯ v1 (хранится z-1), под неё писано наше ядро.
            # Перепутать её значит получить систематический сдвиг, который на случайных данных
            # выглядит как "просто другой ответ", а не как ошибка. Сверка ниже это ловит.
            y_ч = ops.gptq_gemm(x, qw_ч, qz, sc, gi, True, False, 8)
            y_н = ext.w8a16_gemv(x, qw, qz, sc, G, False)   # False = v1 (слепок `gptq`)
            y_э = эталон_fp32(x, qw, qz, sc, G)
            нрм = lambda u: ((u.float() - y_э).norm() / y_э.norm().clamp_min(1e-30)).item()
            r_ч, r_н = нрм(y_ч), нрм(y_н)
            r = ((y_н.float() - y_ч.float()).norm() / y_ч.float().norm().clamp_min(1e-30)).item()
            t_ч = замер(lambda: ops.gptq_gemm(x, qw_ч, qz, sc, gi, True, False, 8))
            t_н = замер(lambda: ext.w8a16_gemv(x, qw, qz, sc, G, False))
            байт = K * N + (K // G) * N * 2 + (K // G) * (N // 4) * 4     # вес + масштабы + нули
            гбс = байт / (t_н / 1e3) / 1e9
            итог_ч += t_ч * штук; итог_н += t_н * штук
            print(f"{имя:<26s} {K:>6d} {N:>6d} {r:>9.2e} {t_ч:>9.3f} {t_н:>9.3f} "
                  f"{гбс:>9.1f} ГБ/с {t_ч/t_н:>10.3f}   к fp32: чужое {r_ч:.2e} наше {r_н:.2e}")
            del x, qw, qz, sc, qw_ч, y_ч, y_н, y_э
            torch.cuda.empty_cache()
        print(f"{'ВЕСЬ ТОКЕН (сумма по слоям)':<26s} {'':>6s} {'':>6s} {'':>9s} "
              f"{итог_ч:>9.2f} {итог_н:>9.2f} {'':>12s} {итог_ч/итог_н:>10.3f}")
        print(f"  (полная сеть, без разреза по картам; при PP=2 на карту приходится половина)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
