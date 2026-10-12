# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# Байтовый сборщик страничного KV (gather_paged_kv_i8) плюс маршрут "страницы -> ядро ролей".
#
# ЗАЧЕМ ОТДЕЛЬНЫЙ ТЕСТ: обвязка vLLM теперь может отдать префилл ядру ролей, а для этого пул
# перекантовывается e4m3 -> int8 с масштабом на позицию. Это ВТОРОЕ квантование поверх первого, и
# его ошибку надо ЗАМЕРИТЬ, а не предположить -- иначе "у нас же int8" превратится в отговорку на
# любое расхождение. Поэтому сверяем ДВЕ вещи по отдельности:
#   1) сам сборщик -- против явного эталона на тех же байтах (ловит раскладку, страницы, масштабы);
#   2) внимание целиком -- против деквантованного e4m3, то есть против тех значений, которые
#      лежат в пуле; так видно ровно цену перекантовки, без примеси ошибки самого пула.
#
# Запуск: pytest tests/test_vllm_i8_gather.py
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fa2_sm70  # noqa: E402

D = 256
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="нужен GPU")


def _e4m3_bytes(x):
    """Кодируем fp16 в e4m3 тем же способом, каким пул vLLM его хранит, и возвращаем И байты,
    И деквантованное значение -- второе служит эталоном "что реально лежит в пуле"."""
    q = x.to(torch.float8_e4m3fn)
    return q.view(torch.uint8), q.to(torch.float16)


@pytest.fixture(scope="module")
def ext():
    return fa2_sm70._ext.decode_ext()


@pytest.mark.parametrize("T,Hkv,bs", [(512, 2, 16), (1024, 4, 16), (777, 2, 32), (2048, 2, 16)])
def test_gather_i8_matches_reference(ext, T, Hkv, bs):
    torch.manual_seed(T + Hkv)
    nblk = (T + bs - 1) // bs
    kf = (torch.randn(nblk, bs, Hkv, D, device="cuda", dtype=torch.float16) * 0.4)
    vf = (torch.randn(nblk, bs, Hkv, D, device="cuda", dtype=torch.float16) * 0.4)
    kc, kdq = _e4m3_bytes(kf)
    vc, vdq = _e4m3_bytes(vf)
    bt = torch.arange(nblk, device="cuda", dtype=torch.int32)

    kq = torch.empty(Hkv, T, D, device="cuda", dtype=torch.int8)
    vq = torch.empty(Hkv, T, D, device="cuda", dtype=torch.int8)
    ks = torch.empty(Hkv, T, device="cuda", dtype=torch.float32)
    vs = torch.empty(Hkv, T, device="cuda", dtype=torch.float32)
    ext.gather_paged_kv_i8(kc, vc, bt, T, 1.0, 1.0, kq, ks, vq, vs)

    # Эталон: та же перестановка страниц в [Hkv, T, d] и то же симметричное квантование по строке.
    ref = kdq.reshape(nblk * bs, Hkv, D)[:T].permute(1, 0, 2).contiguous().float()
    a = ref.abs().amax(-1, keepdim=True).clamp_min(6.1e-5) / 127.0
    exp = torch.clamp(torch.round(ref / a), -127, 127)
    assert torch.equal(kq.float(), exp), "байты сборщика разошлись с эталоном квантования"
    assert torch.allclose(ks, a.squeeze(-1), rtol=1e-6, atol=0), "масштабы разошлись"


@pytest.mark.parametrize("Sq,T,H,Hkv", [(512, 2048, 12, 2), (1024, 4096, 8, 2), (256, 1024, 4, 4)])
def test_i8_route_matches_dequantised_pool(Sq, T, H, Hkv):
    """Маршрут целиком: страницы -> байтовый сборщик -> ядро ролей, против внимания на ТОМ, ЧТО
    ЛЕЖИТ В ПУЛЕ (деквантованный e4m3). Разница здесь -- цена перекантовки, и только она."""
    dext, pext = fa2_sm70._ext.decode_ext(), fa2_sm70._ext.prefill_ext()
    torch.manual_seed(Sq + T)
    bs, sc = 16, 1.0 / math.sqrt(D)
    nblk = (T + bs - 1) // bs
    kf = (torch.randn(nblk, bs, Hkv, D, device="cuda", dtype=torch.float16) * 0.4)
    vf = (torch.randn(nblk, bs, Hkv, D, device="cuda", dtype=torch.float16) * 0.4)
    kc, kdq = _e4m3_bytes(kf)
    vc, vdq = _e4m3_bytes(vf)
    bt = torch.arange(nblk, device="cuda", dtype=torch.int32)
    q = (torch.randn(1, Sq, H, D, device="cuda", dtype=torch.float16) * 0.3).contiguous()

    kq = torch.empty(Hkv, T, D, device="cuda", dtype=torch.int8)
    vq = torch.empty(Hkv, T, D, device="cuda", dtype=torch.int8)
    ks = torch.empty(Hkv, T, device="cuda", dtype=torch.float32)
    vs = torch.empty(Hkv, T, device="cuda", dtype=torch.float32)
    dext.gather_paged_kv_i8(kc, vc, bt, T, 1.0, 1.0, kq, ks, vq, vs)
    o8, _ = pext.attn_fwd_volta_i8(q, kq.unsqueeze(0), ks.unsqueeze(0),
                                   vq.unsqueeze(0), vs.unsqueeze(0), sc, True)

    kd = kdq.reshape(nblk * bs, Hkv, D)[:T].permute(1, 0, 2).contiguous().unsqueeze(0)
    vd = vdq.reshape(nblk * bs, Hkv, D)[:T].permute(1, 0, 2).contiguous().unsqueeze(0)
    Q = q.transpose(1, 2).float()
    K = kd.repeat_interleave(H // Hkv, 1).float()
    V = vd.repeat_interleave(H // Hkv, 1).float()
    s = (Q @ K.transpose(-1, -2)) * sc
    i = torch.arange(Sq, device=s.device).view(-1, 1)
    j = torch.arange(T, device=s.device).view(1, -1)
    s = s.masked_fill(j > i + (T - Sq), float("-inf"))
    ref = (torch.softmax(s, -1) @ V).transpose(1, 2)

    assert torch.isfinite(o8.float()).all(), "не-конечные значения в выходе"
    e = (o8.float() - ref).abs()
    rel = (e.max() / (ref.abs().max() + 1e-30)).item()
    # ПОРОГ ВЗЯТ ИЗ ЗАМЕРА, и замер этот -- главный результат всей ветки. Байтовый маршрут против
    # ТЕКУЩЕГО fp16-пути на одних и тех же страницах (relMax к деквантованному пулу):
    #     Sq/T/H/Hkv      fp16-путь   байтовый маршрут   во сколько хуже
    #     512/2048/12/2    5.87e-04       7.41e-03            x12.6
    #     1024/4096/8/2    4.66e-04       8.32e-03            x17.9
    #     256/1024/4/4     4.49e-04       7.53e-03            x16.8
    # ПРИЧИНА НЕ В ЯДРЕ. e4m3 держит ОТНОСИТЕЛЬНУЮ точность на всём размахе (3 бита мантиссы при
    # широком порядке), а int8 с масштабом по строке -- АБСОЛЮТНУЮ: мелкие значения строки, у которой
    # велик максимум, теряют почти все разряды. Перекантовка выбрасывает ровно то, ради чего e4m3 и
    # берут. Отсюда вывод, записанный в README: байтовый маршрут окупается, только если пул СРАЗУ
    # хранится в int8 с масштабом на позицию (одно квантование вместо двух), а поверх e4m3 -- нет.
    # Порог 1.2e-2 = замеренный максимум 8.3e-3 с запасом в 1.4 раза, чтобы тест ловил регрессию
    # ЯДРА, а не пересчитывал заново цену формата.
    assert rel < 1.2e-2, f"Sq={Sq} T={T}: relMax={rel:.2e}"


# --- e5m2: страницы -> перекладка байт -> ядро ролей, БЕЗ ЕДИНОЙ ПОТЕРИ ---------------------------
@pytest.mark.parametrize("Sq,T,H,Hkv", [(512, 2048, 12, 2), (1024, 4096, 8, 2), (256, 1024, 4, 4)])
def test_e5m2_route_is_bit_exact(Sq, T, H, Hkv):
    """Маршрут e5m2 обязан совпасть с fp16-ядром ПОБИТОВО -- в отличие от int8-маршрута выше.

    Разница принципиальная и в этом весь смысл ветки: int8-маршрут поверх e4m3-пула квантует ВТОРОЙ
    раз и платит 13-18x точности, а e5m2 не конвертирует НИЧЕГО -- сборщик перекладывает байты как
    есть, ядро разворачивает их одной перестановкой (fp16_bits == byte << 8). Поэтому здесь допуска
    нет вовсе: любое расхождение означает поломку перекладки или разворота, а не «цену формата»."""
    dext, pext = fa2_sm70._ext.decode_ext(), fa2_sm70._ext.prefill_ext()
    torch.manual_seed(Sq + T)
    bs, sc, nblk = 16, 1.0 / math.sqrt(D), (T + 15) // 16
    kf = (torch.randn(nblk, bs, Hkv, D, device="cuda", dtype=torch.float16) * 0.4)
    vf = (torch.randn(nblk, bs, Hkv, D, device="cuda", dtype=torch.float16) * 0.4)
    k5, v5 = kf.to(torch.float8_e5m2), vf.to(torch.float8_e5m2)
    kc, vc = k5.view(torch.uint8), v5.view(torch.uint8)
    bt = torch.arange(nblk, device="cuda", dtype=torch.int32)
    q = (torch.randn(1, Sq, H, D, device="cuda", dtype=torch.float16) * 0.3).contiguous()

    kb = torch.empty(Hkv, T, D, device="cuda", dtype=torch.uint8)
    vb = torch.empty(Hkv, T, D, device="cuda", dtype=torch.uint8)
    dext.gather_paged_kv_bytes(kc, vc, bt, T, kb, vb)
    exp = kc.reshape(nblk * bs, Hkv, D)[:T].permute(1, 0, 2).contiguous()
    assert torch.equal(kb, exp), "перекладка байт разошлась со страничной раскладкой"

    o5, _ = pext.attn_fwd_volta_e5m2(q, kb.unsqueeze(0), vb.unsqueeze(0), sc, True)
    kd = k5.to(torch.float16).reshape(nblk * bs, Hkv, D)[:T].permute(1, 0, 2).contiguous().unsqueeze(0)
    vd = v5.to(torch.float16).reshape(nblk * bs, Hkv, D)[:T].permute(1, 0, 2).contiguous().unsqueeze(0)
    o16, _ = pext.attn_fwd_volta(q, kd, vd, sc, True)
    assert torch.equal(o5, o16), (
        f"Sq={Sq} T={T}: e5m2-маршрут разошёлся с fp16 "
        f"(max {(o5.float() - o16.float()).abs().max().item():.3e})")
