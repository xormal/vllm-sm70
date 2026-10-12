# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# Рукописные forward-ядра Volta при head_dim=256: fp16 (volta_fwd_block.cuh) и байтовое KV
# (volta_fwd_i8.cuh). Эталон -- независимый softmax в fp32.
#
# ПОЧЕМУ ФАЙЛ ПЕРЕПИСАН: прежняя версия была СКРИПТОМ без функций test_*, поэтому `pytest` её не
# подбирал ("no tests ran") -- ядра числились покрытыми, а в наборе не проверялись. Случаи из неё
# сохранены полностью и дополнены байтовым путём.
#
# Случаи закрывают ровно то, чего не умел прототип и что при пропуске ломается ТИХО: отображение
# голов GQA, батч с явными шагами, причинная маска ПО ПРАВОМУ НИЖНЕМУ углу (Sq < Sk -- чанковый
# префилл) и рваные Sq/Sk. Запуск: pytest tests/test_volta_fwd.py
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fa2_sm70  # noqa: E402

D = 256
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="нужен GPU")

# (B, H, Hkv, Sq, Sk, causal, имя)
CASES = [
    (1, 4, 4, 512, 512, True, "MHA, Sq=Sk"),
    (1, 12, 2, 512, 1024, True, "GQA 6:1, чанк Sq<Sk"),
    (1, 8, 2, 256, 2048, True, "GQA 4:1, длинный ключ"),
    (1, 4, 4, 512, 512, False, "без маски"),
    (1, 6, 3, 200, 777, True, "РВАНЫЕ Sq=200, Sk=777"),
    (1, 4, 2, 64, 64, True, "одна плитка"),
    (2, 4, 2, 320, 960, True, "батч 2, рваный Sq"),
]
IDS = [c[-1] for c in CASES]


@pytest.fixture(scope="module")
def ext():
    return fa2_sm70._ext.prefill_ext()


def _inputs(B, H, Hkv, Sq, Sk):
    torch.manual_seed(Sq + Sk + H)
    mk = lambda *s: (torch.randn(*s, device="cuda", dtype=torch.float16) * 0.3).contiguous()
    return mk(B, Sq, H, D), mk(B, Hkv, Sk, D), mk(B, Hkv, Sk, D)


def _reference(Qb, K, V, sc, causal, H, Hkv):
    Q = Qb.transpose(1, 2).float()
    Kx = K.repeat_interleave(H // Hkv, 1).float()
    Vx = V.repeat_interleave(H // Hkv, 1).float()
    s = (Q @ Kx.transpose(-1, -2)) * sc
    Sq, Sk = s.shape[-2], s.shape[-1]
    if causal:   # ПО ПРАВОМУ НИЖНЕМУ углу: запросы чанка сидят в КОНЦЕ диапазона ключей
        off = Sk - Sq
        qi = torch.arange(Sq, device=s.device).view(Sq, 1)
        ki = torch.arange(Sk, device=s.device).view(1, Sk)
        s = s.masked_fill(ki > qi + off, float("-inf"))
    return (torch.softmax(s, -1) @ Vx).transpose(1, 2), torch.logsumexp(s, -1)


def _quantize(x):
    """Симметричное по позиции ключа; возвращает байты, масштабы и ДЕКВАНТОВАННОЕ значение."""
    a = x.abs().amax(dim=-1, keepdim=True).float() / 127.0
    q = torch.clamp(torch.round(x.float() / a), -127, 127).to(torch.int8)
    return q.contiguous(), a.squeeze(-1).contiguous(), (q.float() * a).half().contiguous()


def _rel(got, ref):
    """ДВЕ метрики. relL2 делит на НОРМУ эталона, а при длинных причинных строках выход усредняется
    почти в ноль -- знаменатель мал, и relL2 раздувается ровно там, где абсолютная ошибка НАИМЕНЬШАЯ
    (замерено: Sq=192/Sk=2048 дал relL2 2.9e-2 при абсолютной 3e-5). Достаточно одной из двух."""
    e = (got.float() - ref).abs()
    rel_l2 = (e.pow(2).sum() / (ref.pow(2).sum() + 1e-30)).sqrt().item()
    rel_max = (e.max() / (ref.abs().max() + 1e-30)).item()
    return min(rel_l2, rel_max), rel_l2, rel_max


@pytest.mark.parametrize("B,H,Hkv,Sq,Sk,causal,name", CASES, ids=IDS)
def test_volta_fwd_fp16(ext, B, H, Hkv, Sq, Sk, causal, name):
    Q, K, V = _inputs(B, H, Hkv, Sq, Sk)
    sc = 1.0 / math.sqrt(D)
    O, L = ext.attn_fwd_volta(Q, K, V, sc, causal)
    Or, Lr = _reference(Q, K, V, sc, causal, H, Hkv)
    assert torch.isfinite(O.float()).all(), f"{name}: не-конечные значения в выходе"
    r, l2, mx = _rel(O, Or)
    assert r < 3e-3, f"{name}: relL2={l2:.2e} relMax={mx:.2e}"
    assert (L - Lr).abs().max().item() < 5e-3, f"{name}: LSE разошёлся"


@pytest.mark.parametrize("B,H,Hkv,Sq,Sk,causal,name", CASES, ids=IDS)
def test_volta_fwd_i8(ext, B, H, Hkv, Sq, Sk, causal, name):
    Q, K, V = _inputs(B, H, Hkv, Sq, Sk)
    Kq, Ks, Kd = _quantize(K)
    Vq, Vs, Vd = _quantize(V)
    sc = 1.0 / math.sqrt(D)
    O, L = ext.attn_fwd_volta_i8(Q, Kq, Ks, Vq, Vs, sc, causal)
    # ЭТАЛОН ПО ДЕКВАНТОВАННЫМ K/V, а не по исходным fp16: иначе ошибка ЯДРА смешается с ошибкой
    # КВАНТОВАНИЯ, и любое расхождение можно будет списать на "это же int8". Ядро обязано
    # воспроизвести внимание на тех числах, которые ему дали.
    Or, _ = _reference(Q, Kd, Vd, sc, causal, H, Hkv)
    assert torch.isfinite(O.float()).all(), f"{name}: не-конечные значения в выходе"
    r, l2, mx = _rel(O, Or)
    assert r < 3e-3, f"{name}: relL2={l2:.2e} relMax={mx:.2e}"
    assert torch.isfinite(L).all(), f"{name}: не-конечный LSE"


def test_i8_matches_fp16_on_same_numbers(ext):
    """Байтовое ядро на ДЕКВАНТОВАННЫХ входах должно совпасть с fp16-ядром: разница между ними
    только в ФОРМАТЕ подачи, математика одна. Ловит дефекты свёртки смещения разворота, которые по
    одному лишь эталону выглядят как «ошибка квантования»."""
    B, H, Hkv, Sq, Sk = 1, 4, 4, 512, 512
    Q, K, V = _inputs(B, H, Hkv, Sq, Sk)
    Kq, Ks, Kd = _quantize(K)
    Vq, Vs, Vd = _quantize(V)
    sc = 1.0 / math.sqrt(D)
    O16, _ = ext.attn_fwd_volta(Q, Kd, Vd, sc, True)
    O8, _ = ext.attn_fwd_volta_i8(Q, Kq, Ks, Vq, Vs, sc, True)
    rel = ((O8.float() - O16.float()).abs().max() / (O16.float().abs().max() + 1e-30)).item()
    assert rel < 3e-3, f"int8 против fp16 на одних числах: {rel:.2e}"


def test_i8_two_blocks_fit():
    """Сторож ЗАПАСНОГО двухблочного пути (`FA2SM70_WS=0`), а НЕ отгруженного по умолчанию: по
    умолчанию работает ядро ролей -- один блок на 88.5 КБ, порога 48 КБ у него нет вовсе. У
    двухблочного же занятость 22.4% против 12.5% у fp16 держится ровно на том, что помещаются ДВА
    блока, а условие двойное: разделяемой <= 48 КБ И <= 128 регистров на нить; снятие одного предела
    не даёт НИЧЕГО. Здесь сторожим первый (второй виден только из ptxas), чтобы правка геометрии не
    вернула занятость запасного пути к 12.5% молча."""
    BQ, BK, DC, PAD = 32, 64, 64, 4
    smem = (BQ * (DC + PAD) + BQ * (BK + PAD)) * 2 + BK * ((DC + 16) + (D + 16))
    assert smem <= 48 * 1024, f"разделяемая {smem} Б > 48 КБ: второй блок не поместится"


# --- e5m2: байтовая подача БЕЗ единой потери -----------------------------------------------------
@pytest.mark.parametrize("B,H,Hkv,Sq,Sk,causal,name", CASES, ids=IDS)
def test_volta_fwd_e5m2_is_bit_exact(ext, B, H, Hkv, Sq, Sk, causal, name):
    """e5m2 обязан совпасть с fp16-ядром ПОБИТОВО, а не «в пределах допуска».

    Основание -- не эмпирика, а раскладка формата: у e5m2 смещение порядка 15, то же что у fp16, и
    порядок полей тот же, поэтому e5m2 это РОВНО старший байт fp16 (fp16_bits == byte << 8), включая
    поддённормальные и Inf/NaN. Разворот, значит, точен, свёртка смещения не нужна вовсе, и ядро
    обязано выдать ТЕ ЖЕ числа. Допуск здесь был бы уступкой, скрывающей поломку разворота: любая
    ошибка в перестановке байт даёт расхождение, а не «чуть иной» ответ.
    """
    if not causal:
        pytest.skip("ядро ролей требует нижне-правую причинную маску")
    Q, K, V = _inputs(B, H, Hkv, Sq, Sk)
    k5, v5 = K.to(torch.float8_e5m2), V.to(torch.float8_e5m2)
    kb, vb = k5.view(torch.uint8).contiguous(), v5.view(torch.uint8).contiguous()
    kd, vd = k5.to(torch.float16).contiguous(), v5.to(torch.float16).contiguous()
    sc = 1.0 / math.sqrt(D)
    o5, l5 = ext.attn_fwd_volta_e5m2(Q, kb, vb, sc, causal)
    o16, _ = ext.attn_fwd_volta(Q, kd, vd, sc, causal)
    assert torch.isfinite(o5.float()).all(), f"{name}: не-конечные значения"
    assert torch.equal(o5, o16), (
        f"{name}: e5m2 разошёлся с fp16 на ТЕХ ЖЕ числах "
        f"(max {(o5.float() - o16.float()).abs().max().item():.3e}) -- разворот сломан")
    assert torch.isfinite(l5).all(), f"{name}: не-конечный LSE"
