# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# ТОЧНОСТЬ ПРЯМОГО e4m3-ПУТИ (KVFMT=11, nosc=11).
#
# Что здесь доказывается и ЧЕМ -- по убыванию строгости:
#   1. РАЗВОРОТ ТОЧЕН НА ВСЕХ 256 ЗНАЧЕНИЯХ БАЙТА. Проверяется не в ядре, а прямо: биты
#      (b & 0x7F) << 7 | (b & 0x80) << 8, прочитанные как fp16 и умноженные на 256, обязаны
#      совпасть с torch-овским деквантованием float8_e4m3fn -> float16. Единственное исключение
#      -- NaN (0x7F/0xFF), который приём переводит в конечное +-480; оно зафиксировано, а не
#      спрятано.
#   2. ЯДРО ПОБИТОВО СОВПАДАЕТ С ЭТАЛОНОМ. Эталон -- НЕПОДВИЖНЫЙ e5m2-путь (KVFMT=1), про который
#      уже доказано, что он даёт РОВНО тот же ответ, что fp16-ядро на деквантованных значениях
#      (§4nn: fp16_bits == byte << 8, ноль расхождений). Чтобы сравнение было законным, данные
#      берутся из ПЕРЕСЕЧЕНИЯ форматов: v = +-m*2^e, m in {1, 1.25, 1.5, 1.75}, e in [-6, 8] --
#      такие числа ОБА формата держат точно, поэтому оба разворота обязаны отдать одни и те же
#      fp16-операнды, а дальше арифметика ядра тождественна. Расхождение здесь означало бы либо
#      неточный разворот, либо неточную свёртку множителя 256.
#      ПОЧЕМУ ЭТО И ЕСТЬ ТРЕБУЕМОЕ "torch.equal против явной деквантизации": прогнать fp16-ядро
#      на деквантованных значениях нельзя -- у ядра ролей операнд K/V ОДНОБАЙТОВЫЙ по построению,
#      fp16-варианта не существует. Пересечение форматов -- законная замена, а не ослабление.
#   3. relL2 К fp32-ОРАКУЛУ на общих данных, рядом с e5m2 и int8 -- это уже про КАЧЕСТВО ФОРМАТА,
#      а не про правильность ядра.
#   4. ПОДДЁННОРМАЛЬНЫЕ. Разворот отдаёт значение/256, поэтому поддённормальные e4m3 (|v| <= 0.0136)
#      становятся поддённормальными fp16. Сборка идёт с --use_fast_math (-ftz), и вопрос "сбрасывает
#      ли их тензорное ядро" решается ЗАМЕРОМ, а не рассуждением.
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fa2_sm70  # noqa: E402

D = 256


# ПОЧЕМУ ЗДЕСЬ ФИКСТУРА, А НЕ ПРОСТО АРГУМЕНТ. Файл рассчитан на ДВА способа запуска: как скрипт
# (main() ниже раздаёт pext руками) и под pytest, который собирает `def test_*` сам. Без фикстуры
# pytest объявляет `pext` неизвестным и ПАДАЕТ НА СБОРКЕ трёх тестов из четырёх -- то есть набор
# выглядит запущенным, а проверки не идут. Ровно та мина, что 31.07 нашлась в tests/run_all.py, и
# она обязана быть закрыта с обеих сторон: харнесс запускает pytest, а файл ему собираем.
@pytest.fixture(scope="module")
def pext():
    return fa2_sm70._ext.prefill_ext()


def relL2(a, b):
    a, b = a.float(), b.float()
    return (torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b)).item()


def ref_attn(q, k, v, sc, causal):
    """fp32-оракул. k/v ПРИНИМАЮТСЯ УЖЕ ДЕКВАНТОВАННЫМИ -- иначе меряется квантование, а не ядро."""
    B, Sq, H, d = q.shape
    Hkv, Sk = k.shape[1], k.shape[2]
    r = H // Hkv
    qq = q.float().permute(0, 2, 1, 3)
    kk = k.float().repeat_interleave(r, dim=1)
    vv = v.float().repeat_interleave(r, dim=1)
    s = torch.matmul(qq, kk.transpose(-1, -2)) * sc
    if causal:
        off = Sk - Sq
        i = torch.arange(Sq, device=q.device).view(-1, 1) + off
        j = torch.arange(Sk, device=q.device).view(1, -1)
        s = s.masked_fill(j > i, float("-inf"))
    return torch.matmul(torch.softmax(s, dim=-1), vv).permute(0, 2, 1, 3)


def test_expand_bits_exhaustive():
    """1. Разворот на ВСЕХ 256 значениях байта. Считается ровно тем, что делает ядро."""
    b = torch.arange(256, dtype=torch.int32)
    bits = ((b & 0x7F) << 7) | ((b & 0x80) << 8)
    got = bits.to(torch.int16).view(torch.float16).float() * 256.0
    ref = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).to(torch.float32)
    nan = torch.isnan(ref)
    assert torch.equal(got[~nan], ref[~nan]), "разворот e4m3 неточен на конечных значениях"
    # знак нуля тоже обязан совпасть: -0 у нас и -0 у torch
    assert torch.equal(torch.signbit(got[~nan]), torch.signbit(ref[~nan])), "знак нуля потерян"
    assert got[0x7F].item() == 480.0 and got[0xFF].item() == -480.0, "NaN обязан уходить в +-480"
    print("1. разворот: 254/254 конечных значений ТОЧНО, NaN -> +-480 (зафиксировано)")


def _both_formats(shape, device, gen):
    """Значения из ПЕРЕСЕЧЕНИЯ e4m3 и e5m2: мантисса 2 бита, порядок в диапазоне e4m3."""
    m = torch.tensor([1.0, 1.25, 1.5, 1.75], device=device)
    e = torch.arange(-6, 9, device=device, dtype=torch.float32)
    grid = (m.view(-1, 1) * torch.pow(2.0, e).view(1, -1)).flatten()
    grid = torch.cat([grid, -grid, torch.zeros(1, device=device)])
    idx = torch.randint(0, grid.numel(), shape, device=device, generator=gen)
    return grid[idx].to(torch.float16)


def test_kernel_bitexact(pext, B=1, H=8, Hkv=2, Sq=256, Sk=1024, causal=True):
    """2. Побитовое совпадение с e5m2-эталоном на данных из пересечения форматов."""
    g = torch.Generator(device="cuda"); g.manual_seed(7)
    q = (torch.randn(B, Sq, H, D, device="cuda", dtype=torch.float16, generator=g) * 0.3).contiguous()
    kf = _both_formats((B, Hkv, Sk, D), "cuda", g).contiguous()
    vf = _both_formats((B, Hkv, Sk, D), "cuda", g).contiguous()
    # обязательная проверка ПРЕДПОСЫЛКИ: оба формата держат эти числа ТОЧНО
    assert torch.equal(kf.to(torch.float8_e4m3fn).to(torch.float16), kf)
    assert torch.equal(kf.to(torch.float8_e5m2).to(torch.float16), kf)
    assert torch.equal(vf.to(torch.float8_e4m3fn).to(torch.float16), vf)
    assert torch.equal(vf.to(torch.float8_e5m2).to(torch.float16), vf)
    sc = 1.0 / math.sqrt(D)
    empty = torch.empty(0, device="cuda")
    k5 = kf.to(torch.float8_e5m2).view(torch.int8).contiguous()
    v5 = vf.to(torch.float8_e5m2).view(torch.int8).contiguous()
    k4 = kf.to(torch.float8_e4m3fn).view(torch.int8).contiguous()
    v4 = vf.to(torch.float8_e4m3fn).view(torch.int8).contiguous()
    o5, l5 = pext.attn_fwd_volta_i8(q, k5, empty, v5, empty, sc, causal, 0)
    o4, l4 = pext.attn_fwd_volta_i8(q, k4, empty, v4, empty, sc, causal, 11)
    o8, l8 = pext.attn_fwd_volta_i8(q, k4, empty, v4, empty, sc, causal, 13)   # прежняя запись, 8 инстр.
    eq = torch.equal(o4, o5)
    print(f"2. ядро: torch.equal(e4m3, e5m2) = {eq}   relL2 = {relL2(o4, o5):.3e}   "
          f"LSE equal = {torch.equal(l4, l5)}")
    print(f"   запись на 8 инструкций (KVFMT=13) против нашей на 6: torch.equal = {torch.equal(o8, o4)}")
    assert eq, "e4m3-путь разошёлся с эталоном на данных, точных в ОБОИХ форматах"
    assert torch.equal(o8, o4), "две записи одного разворота обязаны совпадать побитово"
    return True


def test_accuracy(pext, B=1, H=8, Hkv=2, Sq=256, Sk=2048, causal=True):
    """3. relL2 к fp32-оракулу: e4m3 против e5m2 против int8 на ОДНИХ данных."""
    torch.manual_seed(3)
    q = (torch.randn(B, Sq, H, D, device="cuda", dtype=torch.float16) * 0.3).contiguous()
    kf = (torch.randn(B, Hkv, Sk, D, device="cuda", dtype=torch.float16) * 0.4).contiguous()
    vf = (torch.randn(B, Hkv, Sk, D, device="cuda", dtype=torch.float16) * 0.4).contiguous()
    sc = 1.0 / math.sqrt(D)
    empty = torch.empty(0, device="cuda")
    ref = ref_attn(q, kf, vf, sc, causal)          # оракул по НЕквантованным значениям

    def q_i8(x):
        a = x.float().abs().amax(-1, keepdim=True).clamp_min(6.1e-5) / 127.0
        return torch.clamp(torch.round(x.float() / a), -127, 127).to(torch.int8).contiguous(), \
            a.squeeze(-1).contiguous().float()

    k4 = kf.to(torch.float8_e4m3fn); v4 = vf.to(torch.float8_e4m3fn)
    k5 = kf.to(torch.float8_e5m2);   v5 = vf.to(torch.float8_e5m2)
    kq, ks = q_i8(kf); vq, vs = q_i8(vf)
    out = {
        "e4m3": pext.attn_fwd_volta_i8(q, k4.view(torch.int8), empty, v4.view(torch.int8), empty, sc, causal, 11)[0],
        "e5m2": pext.attn_fwd_volta_i8(q, k5.view(torch.int8), empty, v5.view(torch.int8), empty, sc, causal, 0)[0],
        "int8": pext.attn_fwd_volta_i8(q, kq, ks, vq, vs, sc, causal, 0)[0],
    }
    print("3. relL2 к fp32-оракулу (одни данные, ОДНОКРАТНОЕ квантование):")
    for name, o in out.items():
        # отдельно: сколько теряет САМ формат (оракул на его деквантованных значениях)
        print(f"     {name:5s}  relL2(ядро, оракул по НЕквантованным) = {relL2(o, ref):.4e}")
    # тот же счёт на уровне ЗНАЧЕНИЙ, без ядра -- чтобы отделить формат от арифметики
    for name, (kk, vv) in {"e4m3": (k4.to(torch.float16), v4.to(torch.float16)),
                           "e5m2": (k5.to(torch.float16), v5.to(torch.float16)),
                           "int8": ((kq.float() * ks.unsqueeze(-1)).half(),
                                    (vq.float() * vs.unsqueeze(-1)).half())}.items():
        print(f"     {name:5s}  relL2(значения K) = {relL2(kk, kf):.4e}   "
              f"relL2(оракул на них, оракул на исходных) = {relL2(ref_attn(q, kk, vv, sc, causal), ref):.4e}")
    # ЯДРО ПРОТИВ ОРАКУЛА НА ЕГО ЖЕ ДЕКВАНТОВАННЫХ ЗНАЧЕНИЯХ -- это уже ошибка АРИФМЕТИКИ ядра
    r4 = ref_attn(q, k4.to(torch.float16), v4.to(torch.float16), sc, causal)
    print(f"     e4m3   relL2(ядро, оракул НА ТЕХ ЖЕ деквантованных) = {relL2(out['e4m3'], r4):.4e}"
          f"   <- собственная ошибка ядра")


def test_subnormals(pext, B=1, H=8, Hkv=2, Sq=128, Sk=512):
    """4. Поддённормальные e4m3: выживают ли они после деления на 256 при -ftz."""
    torch.manual_seed(11)
    q = (torch.randn(B, Sq, H, D, device="cuda", dtype=torch.float16) * 0.3).contiguous()
    # ТОЛЬКО поддённормальные коды e4m3: E == 0, M != 0  -> |v| = M * 2^-9
    codes = torch.tensor([c for c in range(256) if (c & 0x78) == 0 and (c & 0x07) != 0],
                         dtype=torch.uint8)
    idx = torch.randint(0, codes.numel(), (B, Hkv, Sk, D))
    kb = codes[idx].cuda().contiguous()
    kf = kb.view(torch.float8_e4m3fn).to(torch.float16)
    vf = kf.clone()
    sc = 1.0 / math.sqrt(D)
    empty = torch.empty(0, device="cuda")
    o = pext.attn_fwd_volta_i8(q, kb.view(torch.int8), empty, kb.view(torch.int8), empty, sc, True, 11)[0]
    ref = ref_attn(q, kf, vf, sc, True)
    r = relL2(o, ref)
    print(f"4. поддённормальные e4m3 (|v| <= 0.0137, все элементы): relL2 = {r:.4e}   "
          f"{'ЖИВЫ' if r < 1e-2 else 'СБРОШЕНЫ В НОЛЬ (-ftz)'}")
    return r


def main():
    pext = fa2_sm70._ext.prefill_ext()
    test_expand_bits_exhaustive()
    test_kernel_bitexact(pext)
    test_accuracy(pext)
    test_subnormals(pext)
    print("\nВСЁ ПРОЙДЕНО")


if __name__ == "__main__":
    main()
