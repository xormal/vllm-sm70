# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# Fused GPTQ-int8 W8A16 GEMV: the packed kernel must equal an explicitly-unpacked fp16 matmul.
# The reference is built by a DIFFERENT route (unpack to a dense tensor, then torch matmul), so a
# shared misreading of the bit layout cannot pass both.  Run:  python tests/test_w8a16.py
#
# [ПОПРАВКА 16.08.2026 -- ТЕСТ УЗАКОНИВАЛ ДЕФЕКТ]
# Эталон строился по СЫРОМУ нулю, то есть по условности GPTQ v2, -- ровно так же, как его читало
# ядро. Два независимых на вид пути соглашались, будучи оба неверны для боевого слепка (формат
# `gptq` = v1, где хранится z-1), и тест зелено проходил при систематической ошибке ~1.4 % на
# каждом весе. Ломается это только внешним третейским путём: сверка с ЧУЖИМ ядром vLLM, где
# условность задаётся флагом use_v2_format (см. benchmarks/bench_svoy_put.py -- там наше тело
# при верной условности даёт 2.05e-4 против 1.18e-3 у чужого). Теперь гоняются ОБЕ условности,
# и эталон строится по ТОЙ ЖЕ, что передана ядру.
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fa2_sm70

ext = fa2_sm70._ext.w8a16_ext()
SH = [(4096, 4096, 128), (17408, 5120, 128), (5120, 17408, 128),  # real Qwen3.6-27B-INT8 projections
      (2048, 1024, 64),                                           # a finer group
      (1152, 512, 128)]                                           # K not a multiple of XTILE
# M=1 is decode-batch-1, 4 is the measured crossover, 16/33 exercise the M-tile chunking and its
# ragged tail (33 = 16+16+1, so all three tile widths run in one call).
MS = [1, 2, 4, 16, 33]

ok = True
for (K, N, G) in SH:
    torch.manual_seed(K + N)
    qw = torch.randint(-2**31, 2**31 - 1, (K // 4, N), device="cuda", dtype=torch.int32)
    qz = torch.randint(-2**31, 2**31 - 1, (K // G, N // 4), device="cuda", dtype=torch.int32)
    sc = (torch.rand(K // G, N, device="cuda", dtype=torch.float16) * 0.02 + 0.001)
    sh = torch.arange(0, 32, 8, device="cuda")
    q = ((qw.unsqueeze(1) >> sh.view(1, 4, 1)) & 0xFF).reshape(K, N).float()
    z = ((qz.unsqueeze(2) >> sh.view(1, 1, 4)) & 0xFF).reshape(K // G, N).float()
    g = torch.arange(K, device="cuda") // G
    for v2 in (False, True):          # False = формат `gptq` (v1, хранится z-1); True = `gptq_v2`
        W = ((q - (z if v2 else z + 1)[g]) * sc.float()[g]).half()
        for M in MS:
            x = torch.randn(M, K, device="cuda", dtype=torch.float16) * 0.05
            y = ext.w8a16_gemv(x, qw, qz, sc, G, v2)
            ref = x.float() @ W.float()
            rel = ((y.float() - ref).norm() / ref.norm()).item()
            good = rel < 2e-3; ok &= good
            print(f"  K{K} N{N} G{G} M{M} v2={int(v2)}: relL2={rel:.2e} {'OK' if good else 'FAIL'}")
        del W
    del q, z; torch.cuda.empty_cache()

print("W8A16:", "ALL OK" if ok else "FAILED"); assert ok
