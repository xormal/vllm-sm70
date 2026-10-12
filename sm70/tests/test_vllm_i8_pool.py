# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# ФОРМАТ KV-ПУЛА: int8 + масштаб НА ПОЗИЦИЮ, сквозной (хранение -> префилл -> декод).
#
# ЗАЧЕМ ЭТОТ ФАЙЛ ОТДЕЛЬНО ОТ test_vllm_i8_gather.py. Тот тест мерил ПЕРЕКАНТОВКУ e4m3 -> int8 на
# выходе из пула и показал, что она стоит 13-18x точности: два квантования подряд, причём второе
# выбрасывает ровно то, ради чего берут e4m3 (относительную точность на всём размахе). Вывод оттуда
# был не «int8 плох», а «int8 надо ХРАНИТЬ». Здесь проверяется именно это: пул СРАЗУ int8, квантование
# ОДНО, и тогда весь дальнейший путь -- перекладка байт без единой потери.
#
# Проверяется пять независимых утверждений:
#   1) хранение побитово совпадает с явным эталоном квантования (ловит раскладку страниц и масштабы);
#   2) сборка для ядра ролей ТОЖДЕСТВЕННА содержимому пула (байты равны, масштабы равны);
#   3) точность внимания на int8-пуле ЛУЧШЕ, чем на e4m3-пуле, на ОДНИХ И ТЕХ ЖЕ числах -- это
#      главное утверждение всей правки, и оно мерится, а не предполагается;
#   4) декод читает ТОТ ЖЕ пул и совпадает с эталоном на деквантованном пуле;
#   5) рассогласование форматов ПАДАЕТ ГРОМКО, а не отвечает бессмыслицей.
#
# Запуск: pytest tests/test_vllm_i8_pool.py -v
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fa2_sm70  # noqa: E402

D = 256
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="нужен GPU")


def _pool(nblk, bs, Hkv, d):
    """Пул движка: [блоки, размер_блока, Hkv, d] байт (vLLM держит его именно так)."""
    return (torch.zeros(nblk, bs, Hkv, d, device="cuda", dtype=torch.uint8),
            torch.zeros(nblk, bs, Hkv, d, device="cuda", dtype=torch.uint8))


def _tabs(nblk, bs, Hkv):
    """Побочная таблица масштабов: [блоки, размер_блока, Hkv] fp32 -- раскладка s_off()."""
    return (torch.zeros(nblk * bs * Hkv, device="cuda", dtype=torch.float32),
            torch.zeros(nblk * bs * Hkv, device="cuda", dtype=torch.float32))


def _ref_quant(x):
    """Эталон: симметрично по строке, absmax/127, к ближайшему. Возвращает (байты, масштаб)."""
    a = x.float().abs().amax(-1, keepdim=True).clamp_min(6.1e-5) / 127.0
    q = torch.clamp(torch.round(x.float() / a), -127, 127)
    return q, a.squeeze(-1)


def _rel_l2(o, ref):
    return ((o.float() - ref.float()).norm() / ref.float().norm()).item()


def _sdpa_ref(q, kd, vd, H, Hkv, Sq, T, sc):
    """Эталон в fp32: нижне-правая причинная маска; q [1,Sq,H,D], kd/vd [1,Hkv,T,D]."""
    Q = q.transpose(1, 2).float()
    K = kd.repeat_interleave(H // Hkv, 1).float()
    V = vd.repeat_interleave(H // Hkv, 1).float()
    s = (Q @ K.transpose(-1, -2)) * sc
    i = torch.arange(Sq, device=s.device).view(-1, 1)
    j = torch.arange(T, device=s.device).view(1, -1)
    s = s.masked_fill(j > i + (T - Sq), float("-inf"))
    return (torch.softmax(s, -1) @ V).transpose(1, 2)


@pytest.fixture(scope="module")
def ext():
    return fa2_sm70._ext.decode_ext()


# --- 1. ХРАНЕНИЕ ---------------------------------------------------------------------------------
@pytest.mark.parametrize("n,Hkv,bs", [(512, 2, 16), (1000, 4, 16), (777, 2, 32), (64, 2, 16)])
def test_store_i8_bit_exact(ext, n, Hkv, bs):
    torch.manual_seed(n + Hkv)
    nblk = (n + bs - 1) // bs + 2
    k = torch.randn(n, Hkv, D, device="cuda", dtype=torch.float16) * 0.4
    v = torch.randn(n, Hkv, D, device="cuda", dtype=torch.float16) * 0.7
    kc, vc = _pool(nblk, bs, Hkv, D)
    kst, vst = _tabs(nblk, bs, Hkv)
    # Слоты вразброс, и один ПУСТОЙ (-1): именно так движок отдаёт дополненный батч.
    slots = torch.randperm(nblk * bs, device="cuda")[:n].to(torch.long)
    slots[0] = -1
    ext.reshape_and_cache_i8(k, v, kc, vc, slots, kst, vst)

    qk, sk = _ref_quant(k)
    qv, sv = _ref_quant(v)
    live = slots >= 0
    sl = slots[live]
    blk, off = sl // bs, sl % bs
    assert torch.equal(kc.view(torch.int8)[blk, off].float(), qk[live]), "байты K разошлись с эталоном"
    assert torch.equal(vc.view(torch.int8)[blk, off].float(), qv[live]), "байты V разошлись с эталоном"
    assert torch.equal(kst.view(nblk, bs, Hkv)[blk, off], sk[live]), "масштабы K разошлись"
    assert torch.equal(vst.view(nblk, bs, Hkv)[blk, off], sv[live]), "масштабы V разошлись"
    # Пустой слот не должен был тронуть НИЧЕГО: масштаб там остался нулевым (он всегда > 0 у живых).
    assert int(kst.ne(0).sum()) == int(live.sum()) * Hkv, "затронут лишний слот"


# --- 2. СБОРКА ДЛЯ ЯДРА РОЛЕЙ: ТОЖДЕСТВЕННО СОДЕРЖИМОМУ ПУЛА -------------------------------------
@pytest.mark.parametrize("T,Hkv,bs", [(2048, 2, 16), (1024, 4, 16), (777, 2, 32)])
def test_gather_i8_pool_is_verbatim(ext, T, Hkv, bs):
    torch.manual_seed(T)
    nblk = (T + bs - 1) // bs
    k = torch.randn(T, Hkv, D, device="cuda", dtype=torch.float16) * 0.4
    v = torch.randn(T, Hkv, D, device="cuda", dtype=torch.float16) * 0.4
    kc, vc = _pool(nblk, bs, Hkv, D)
    kst, vst = _tabs(nblk, bs, Hkv)
    # Таблица блоков ПЕРЕМЕШАНА: страничная адресация обязана идти через block_table, а не подряд.
    perm = torch.randperm(nblk, device="cuda").to(torch.int32)
    slots = (perm.long().repeat_interleave(bs) * bs
             + torch.arange(bs, device="cuda").repeat(nblk))[:T]
    ext.reshape_and_cache_i8(k, v, kc, vc, slots, kst, vst)

    kq = torch.empty(Hkv, T, D, device="cuda", dtype=torch.int8)
    vq = torch.empty(Hkv, T, D, device="cuda", dtype=torch.int8)
    ks = torch.empty(Hkv, T, device="cuda", dtype=torch.float32)
    vs = torch.empty(Hkv, T, device="cuda", dtype=torch.float32)
    ext.gather_paged_kv_i8_pool(kc, vc, kst, vst, perm, T, kq, ks, vq, vs)

    qk, sk = _ref_quant(k)
    assert torch.equal(kq.float(), qk.permute(1, 0, 2)), "сборка байт разошлась с пулом"
    assert torch.equal(ks, sk.permute(1, 0).contiguous()), "сборка масштабов разошлась с пулом"
    # Тот же обход в fp16 -- ровно q*s, без повторного квантования.
    fk = torch.empty(Hkv, T, D, device="cuda", dtype=torch.float16)
    fv = torch.empty(Hkv, T, D, device="cuda", dtype=torch.float16)
    ext.gather_paged_kv_i8_pool_f16(kc, vc, kst, vst, perm, T, fk, fv)
    assert torch.equal(fk, (qk * sk.unsqueeze(-1)).permute(1, 0, 2).half()), "деквантование != q*s"


# --- 3. ГЛАВНОЕ: ТОЧНОСТЬ int8-ПУЛА ПРОТИВ e4m3-ПУЛА ---------------------------------------------
@pytest.mark.parametrize("Sq,T,H,Hkv", [(512, 2048, 12, 2), (1024, 4096, 8, 2), (256, 1024, 4, 4)])
def test_i8_pool_beats_e4m3_pool(Sq, T, H, Hkv):
    """Оба формата -- ОДИН байт на элемент, одни и те же числа, один и тот же эталон.

    Эталон -- внимание в fp32 на ИСХОДНЫХ fp16 K/V, то есть на том, что модель ДАЛА бы пулу. Так
    сравниваются именно форматы ХРАНЕНИЯ: у каждого свой законный путь чтения (e4m3 -> fp16-ядро,
    int8 -> ядро ролей), и мерится сквозная ошибка каждого маршрута целиком.

    e4m3 получает свой ЛУЧШИЙ случай: k_scale/v_scale берутся как absmax/448, то есть так, как их
    даёт калибровка развёртывания -- иначе сравнение было бы подтасовкой в нашу пользу.
    """
    dext, pext = fa2_sm70._ext.decode_ext(), fa2_sm70._ext.prefill_ext()
    torch.manual_seed(Sq + T)
    bs, sc, nblk = 16, 1.0 / math.sqrt(D), (T + 15) // 16
    k = torch.randn(T, Hkv, D, device="cuda", dtype=torch.float16) * 0.4
    v = torch.randn(T, Hkv, D, device="cuda", dtype=torch.float16) * 0.4
    q = (torch.randn(1, Sq, H, D, device="cuda", dtype=torch.float16) * 0.3).contiguous()
    bt = torch.arange(nblk, device="cuda", dtype=torch.int32)
    slots = torch.arange(T, device="cuda", dtype=torch.long)
    ref = _sdpa_ref(q, k.permute(1, 0, 2).unsqueeze(0), v.permute(1, 0, 2).unsqueeze(0), H, Hkv, Sq, T, sc)

    # --- маршрут e4m3: наш store -> сборка в fp16 -> fp16-ядро префилла (сегодняшний путь) ---
    kc8, vc8 = _pool(nblk, bs, Hkv, D)
    kb = torch.zeros(Hkv, D, device="cuda", dtype=torch.float32)
    ks_l = float(k.abs().max()) / 448.0
    vs_l = float(v.abs().max()) / 448.0
    dext.reshape_and_cache_e4m3(k, v, kc8, vc8, slots, ks_l, vs_l, kb, 1.15)
    fk = torch.empty(Hkv, T, D, device="cuda", dtype=torch.float16)
    fv = torch.empty(Hkv, T, D, device="cuda", dtype=torch.float16)
    dext.gather_paged_kv(kc8, vc8, bt, T, ks_l, vs_l, fk, fv)
    o_e4m3, _ = pext.attn_fwd_qbshd(q, fk.unsqueeze(0), fv.unsqueeze(0), sc, True)

    # --- маршрут int8: наш store -> перекладка байт -> ядро ролей ---
    kci, vci = _pool(nblk, bs, Hkv, D)
    kst, vst = _tabs(nblk, bs, Hkv)
    dext.reshape_and_cache_i8(k, v, kci, vci, slots, kst, vst)
    kq = torch.empty(Hkv, T, D, device="cuda", dtype=torch.int8)
    vq = torch.empty(Hkv, T, D, device="cuda", dtype=torch.int8)
    kqs = torch.empty(Hkv, T, device="cuda", dtype=torch.float32)
    vqs = torch.empty(Hkv, T, device="cuda", dtype=torch.float32)
    dext.gather_paged_kv_i8_pool(kci, vci, kst, vst, bt, T, kq, kqs, vq, vqs)
    o_i8, _ = pext.attn_fwd_volta_i8(q, kq.unsqueeze(0), kqs.unsqueeze(0),
                                     vq.unsqueeze(0), vqs.unsqueeze(0), sc, True)

    r8, ri = _rel_l2(o_e4m3, ref), _rel_l2(o_i8, ref)
    print(f"\n[формат] Sq={Sq} T={T} H={H} Hkv={Hkv}: e4m3 relL2={r8:.3e}  "
          f"int8+масштаб relL2={ri:.3e}  выигрыш x{r8 / ri:.2f}")
    assert torch.isfinite(o_i8.float()).all(), "не-конечные значения в выходе int8-маршрута"
    # ПОРОГ НЕ ПОДОГНАН ПОД РЕЗУЛЬТАТ: утверждение структурное -- int8 с масштабом на позицию тратит
    # 7 бит на мантиссу против 3 у e4m3, поэтому он ОБЯЗАН быть точнее при том же байте. Требуем
    # выигрыш хотя бы вдвое, чтобы тест ловил регрессию, а не шум (замерено ~2.5-3x).
    assert ri * 2.0 < r8, f"int8-пул не оказался точнее e4m3: {ri:.3e} против {r8:.3e}"


# --- 4. ДЕКОД ИЗ ТОГО ЖЕ ПУЛА --------------------------------------------------------------------
@pytest.mark.parametrize("B,T,H,Hkv", [(1, 2048, 12, 2), (2, 1000, 8, 2), (1, 517, 4, 4)])
def test_decode_i8_pool(ext, B, T, H, Hkv):
    """Декод обязан читать ТОТ ЖЕ формат, что записал store. Эталон -- софтмакс в fp32 на
    ДЕКВАНТОВАННОМ пуле: так измеряется ошибка ЯДРА, а не ошибка формата (её мерит тест 3)."""
    torch.manual_seed(T + B)
    bs, sc = 16, 1.0 / math.sqrt(D)
    nb = (T + bs - 1) // bs
    nblk = nb * B
    kc, vc = _pool(nblk, bs, Hkv, D)
    kst, vst = _tabs(nblk, bs, Hkv)
    bt = torch.arange(nblk, device="cuda", dtype=torch.int32).view(B, nb)
    kd, vd = [], []
    for b in range(B):
        k = torch.randn(T, Hkv, D, device="cuda", dtype=torch.float16) * 0.4
        v = torch.randn(T, Hkv, D, device="cuda", dtype=torch.float16) * 0.4
        slots = (bt[b, 0].long() * bs + torch.arange(T, device="cuda", dtype=torch.long))
        ext.reshape_and_cache_i8(k, v, kc, vc, slots, kst, vst)
        qk, sk = _ref_quant(k)
        qv, sv = _ref_quant(v)
        kd.append((qk * sk.unsqueeze(-1)).permute(1, 0, 2))       # [Hkv,T,D] то, что ЛЕЖИТ в пуле
        vd.append((qv * sv.unsqueeze(-1)).permute(1, 0, 2))
    q = torch.randn(B, H, 1, D, device="cuda", dtype=torch.float16) * 0.3
    ns = 8
    A = torch.empty(B * H * ns * D, device="cuda", dtype=torch.float32)
    Bw = torch.empty(2 * B * H * ns, device="cuda", dtype=torch.float32)
    out = torch.empty(B, H, D, device="cuda", dtype=torch.float16)
    km = torch.zeros(B, Hkv, device="cuda", dtype=torch.float32)
    sl = torch.full((B,), T, device="cuda", dtype=torch.int32)
    ext.flash_decode_defer_mqa_paged_into(q, kc, vc, km, bt, bs, sc, ns, nb * bs, sl,
                                          False, 1.0, 1.0, A, Bw, out, H * D,
                                          kst, vst)
    ref = []
    for b in range(B):
        kr = kd[b].repeat_interleave(H // Hkv, 0).float()      # [H,T,D]
        vr = vd[b].repeat_interleave(H // Hkv, 0).float()
        s = torch.einsum("hd,htd->ht", q[b, :, 0].float(), kr) * sc
        # einsum, а НЕ `@`: matmul [H,T] на [H,T,D] молча транслирует и даёт [H,H,D] -- эталон при
        # этом выглядит правдоподобно и врёт (на этом уже потерян один прогон).
        ref.append(torch.einsum("ht,htd->hd", torch.softmax(s, -1), vr))
    ref = torch.stack(ref)
    rel = _rel_l2(out, ref)
    print(f"\n[декод] B={B} T={T} H={H} Hkv={Hkv}: relL2={rel:.3e}")
    assert torch.isfinite(out.float()).all(), "не-конечные значения в выходе декода"
    # Порог -- цена fp16-выхода и fp32-накопления в ядре против fp32-эталона на ТОМ ЖЕ пуле; формат
    # здесь ни при чём (он уже в эталоне), поэтому допуск узкий.
    assert rel < 2e-3, f"декод из int8-пула разошёлся с эталоном: relL2={rel:.3e}"


# --- 5. РАССОГЛАСОВАНИЕ ФОРМАТОВ ПАДАЕТ ГРОМКО ---------------------------------------------------
def test_format_mismatch_raises(ext):
    """Два формата одновременно -- это не «немного другой масштаб», а другой смысл каждого байта.
    В этом проекте такое уже давало сервер, который поднимался и выдавал бессмыслицу, поэтому
    ядро обязано ОТКАЗАТЬ, а не посчитать."""
    B, H, Hkv, T, bs = 1, 4, 2, 64, 16
    nb = T // bs
    kc, vc = _pool(nb, bs, Hkv, D)
    kst, vst = _tabs(nb, bs, Hkv)
    q = torch.zeros(B, H, 1, D, device="cuda", dtype=torch.float16)
    bt = torch.arange(nb, device="cuda", dtype=torch.int32).view(B, nb)
    A = torch.empty(B * H * 4 * D, device="cuda", dtype=torch.float32)
    Bw = torch.empty(2 * B * H * 4, device="cuda", dtype=torch.float32)
    out = torch.empty(B, H, D, device="cuda", dtype=torch.float16)
    km = torch.zeros(B, Hkv, device="cuda", dtype=torch.float32)
    sl = torch.full((B,), T, device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError, match="DIFFERENT pool formats"):
        ext.flash_decode_defer_mqa_paged_into(q, kc, vc, km, bt, bs, 1.0, 4, T, sl,
                                              True, 1.0, 1.0, A, Bw, out, H * D, kst, vst)
    # Таблица не того размера -- тоже отказ, а не чтение мимо (масштаб идёт ПО СЛОТУ, не по блоку).
    with pytest.raises(RuntimeError, match="scale table"):
        ext.flash_decode_defer_mqa_paged_into(q, kc, vc, km, bt, bs, 1.0, 4, T, sl,
                                              False, 1.0, 1.0, A, Bw, out, H * D,
                                              kst[:8].contiguous(), vst[:8].contiguous())


# --- 6. ОБВЯЗКА: ТРИ ВХОДА ЧИТАЮТ ОДИН ФОРМАТ ----------------------------------------------------
def test_shim_three_entries_agree():
    """Сквозная проверка признака формата в обвязке: store -> prefill_paged -> decode_paged на одном
    пуле. Именно здесь ловится рассогласование, которое в этом проекте уже случалось: сервер
    поднимался, а выдавал бессмыслицу, потому что писали один формат, а читали другой."""
    os.environ["FA2SM70_KV_I8"] = "1"
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "integrations", "vllm_sm70"))
    for m in [m for m in list(sys.modules) if m.startswith("flash_attn_v100")]:
        del sys.modules[m]
    import flash_attn_v100 as shim

    torch.manual_seed(7)
    Hkv, H, T, Sq, bs = 2, 12, 1024, 256, 16
    nblk = T // bs
    sc = 1.0 / math.sqrt(D)
    kc, vc = _pool(nblk, bs, Hkv, D)
    k = torch.randn(T, Hkv, D, device="cuda", dtype=torch.float16) * 0.4
    v = torch.randn(T, Hkv, D, device="cuda", dtype=torch.float16) * 0.4
    slots = torch.arange(T, device="cuda", dtype=torch.long)
    shim.reshape_and_cache_fp8(k, v, kc, vc, slots, "fp8_e4m3", 1.0, 1.0)
    assert shim._kv_fmt(kc) == "i8", "обвязка не отметила формат пула"

    q = (torch.randn(1, Sq, H, D, device="cuda", dtype=torch.float16) * 0.3).contiguous()
    bt = torch.arange(nblk, device="cuda", dtype=torch.int32).view(1, nblk)
    sl = torch.full((1,), T, device="cuda", dtype=torch.int32)
    o = shim.flash_attn_prefill_paged(q, kc, vc, bt, sl, softmax_scale=sc,
                                      kv_cache_dtype="fp8_e4m3", k_scale=1.0, v_scale=1.0)
    qk, sk = _ref_quant(k)
    qv, sv = _ref_quant(v)
    kd = (qk * sk.unsqueeze(-1)).permute(1, 0, 2).unsqueeze(0)
    vd = (qv * sv.unsqueeze(-1)).permute(1, 0, 2).unsqueeze(0)
    ref = _sdpa_ref(q, kd, vd, H, Hkv, Sq, T, sc)
    rel = _rel_l2(o, ref)
    print(f"\n[обвязка префилл] relL2={rel:.3e}")
    assert rel < 2e-3, f"префилл обвязки разошёлся с пулом: {rel:.3e}"

    qd = torch.randn(1, H, D, device="cuda", dtype=torch.float16) * 0.3
    out = torch.empty(1, H, D, device="cuda", dtype=torch.float16)
    shim.flash_attn_decode_paged(qd, kc, vc, bt, sl, softmax_scale=sc, out=out,
                                 kv_cache_dtype="fp8_e4m3", k_scale=1.0, v_scale=1.0)
    kr = kd[0].repeat_interleave(H // Hkv, 0).float()
    vr = vd[0].repeat_interleave(H // Hkv, 0).float()
    refd = torch.einsum("ht,htd->hd",
                        torch.softmax(torch.einsum("hd,htd->ht", qd[0].float(), kr) * sc, -1), vr)
    reld = _rel_l2(out[0], refd)
    print(f"[обвязка декод] relL2={reld:.3e}")
    assert reld < 2e-3, f"декод обвязки разошёлся с пулом: {reld:.3e}"

    # --- ПЛОМБИРОВКА ЗАГРУЗКИ: то, что ломается не в математике, а на подъёме сервера ---------
    # 1) ПРОФИЛИРОВОЧНЫЙ ПРОГОН движка идёт по тем же входам, но со слотами -1 (заполнитель). Он не
    #    должен ни писать в пул, ни ронять процесс: вся раскладка ещё фиктивная.
    dummy_k = torch.randn(8, Hkv, D, device="cuda", dtype=torch.float16)
    dummy_v = torch.randn(8, Hkv, D, device="cuda", dtype=torch.float16)
    kc2, vc2 = _pool(nblk, bs, Hkv, D)
    shim.reshape_and_cache_fp8(dummy_k, dummy_v, kc2, vc2,
                               torch.full((8,), -1, device="cuda", dtype=torch.long),
                               "fp8_e4m3", 1.0, 1.0)
    assert int(kc2.ne(0).sum()) == 0, "профилировочный прогон записал в пул"
    # 2) ДВА СЛОЯ = ДВА ПУЛА: реестр формата и таблицы масштабов ведутся ПО ПУЛУ, а не глобально.
    #    Если бы они были общими, второй слой читал бы масштабы первого -- и это НЕ упало бы, а
    #    тихо испортило выход, то есть худший из возможных отказов.
    k2 = torch.randn(T, Hkv, D, device="cuda", dtype=torch.float16) * 0.9
    v2 = torch.randn(T, Hkv, D, device="cuda", dtype=torch.float16) * 0.9
    shim.reshape_and_cache_fp8(k2, v2, kc2, vc2, slots, "fp8_e4m3", 1.0, 1.0)
    o2 = shim.flash_attn_prefill_paged(q, kc2, vc2, bt, sl, softmax_scale=sc,
                                       kv_cache_dtype="fp8_e4m3", k_scale=1.0, v_scale=1.0)
    qk2, sk2 = _ref_quant(k2)
    qv2, sv2 = _ref_quant(v2)
    ref2 = _sdpa_ref(q, (qk2 * sk2.unsqueeze(-1)).permute(1, 0, 2).unsqueeze(0),
                     (qv2 * sv2.unsqueeze(-1)).permute(1, 0, 2).unsqueeze(0), H, Hkv, Sq, T, sc)
    rel2 = _rel_l2(o2, ref2)
    print(f"[обвязка, второй пул] relL2={rel2:.3e}")
    assert rel2 < 2e-3, f"второй пул прочитан с чужими масштабами: {rel2:.3e}"
    assert shim._cache_key(kc) != shim._cache_key(kc2), "два пула получили один ключ"

    # А теперь -- РАССОГЛАСОВАНИЕ. Признак формата устроен так, что читатели СЛЕДУЮТ за store'ом
    # (запись помечает пул, чтение сверяется), поэтому «настроили одно, записали другое» не даёт
    # неверного чтения по построению. Остаётся один опасный случай: чтение int8-пула, в который наш
    # store НЕ ПИСАЛ -- тогда в байтах лежит чужой формат. Он обязан упасть, а не посчитать.
    fresh_k, fresh_v = _pool(nblk, bs, Hkv, D)
    with pytest.raises(RuntimeError, match="масштабов"):
        shim.flash_attn_prefill_paged(q, fresh_k, fresh_v, bt, sl, softmax_scale=sc,
                                      kv_cache_dtype="fp8_e4m3", k_scale=1.0, v_scale=1.0)
    with pytest.raises(RuntimeError, match="масштабов"):
        shim.flash_attn_decode_paged(qd, fresh_k, fresh_v, bt, sl, softmax_scale=sc, out=out,
                                     kv_cache_dtype="fp8_e4m3", k_scale=1.0, v_scale=1.0)


# --- 7. ГЕЙТ ДВУСТОРОННИЙ: ПРИ ВКЛЮЧЕНИИ РЕЗУЛЬТАТ ДРУГОЙ, А НЕ ТОЛЬКО «СОВПАЛ ПРИ ВЫКЛЮЧЕНИИ» ---
def test_switch_is_two_sided():
    """«Совпало при выключении» без «разошлось при включении» не отличает работающую правку от
    неработающей: ровно так однажды был принят A/B, сдвинувший сквозную метрику на 0.03% просто
    потому, что сервер в новый код не заходил.

    Здесь ОДНИ И ТЕ ЖЕ K/V кладутся в ОДИНАКОВЫЕ пулы двумя режимами обвязки, и проверяется, что
    (а) байты РАЗНЫЕ (формат действительно сменился), (б) отметка формата разная, (в) чтение
    e4m3-пула int8-маршрутом ПАДАЕТ, а не считает."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "integrations", "vllm_sm70"))
    for m in [m for m in list(sys.modules) if m.startswith("flash_attn_v100")]:
        del sys.modules[m]
    os.environ["FA2SM70_KV_I8"] = "1"
    import flash_attn_v100 as shim

    torch.manual_seed(11)
    Hkv, T, bs = 2, 512, 16
    nblk = T // bs
    k = torch.randn(T, Hkv, D, device="cuda", dtype=torch.float16) * 0.4
    v = torch.randn(T, Hkv, D, device="cuda", dtype=torch.float16) * 0.4
    slots = torch.arange(T, device="cuda", dtype=torch.long)

    kc_on, vc_on = _pool(nblk, bs, Hkv, D)
    shim._KV_I8 = True
    shim.reshape_and_cache_fp8(k, v, kc_on, vc_on, slots, "fp8_e4m3", 1.0, 1.0)

    kc_off, vc_off = _pool(nblk, bs, Hkv, D)
    shim._KV_I8 = False
    shim.reshape_and_cache_fp8(k, v, kc_off, vc_off, slots, "fp8_e4m3", 1.0, 1.0)

    assert shim._kv_fmt(kc_on) == "i8" and shim._kv_fmt(kc_off) == "e4m3", "отметка формата не разошлась"
    assert not torch.equal(kc_on, kc_off), "включение НИЧЕГО не изменило в байтах -- правка не работает"
    # e4m3-пул int8-маршрутом читать нельзя: масштабов у него нет, и это обязано быть ОТКАЗОМ.
    with pytest.raises(RuntimeError, match="формат пула|масштабов"):
        shim._require_i8(kc_off, "gate")
    # А int8-пул -- можно, и таблицы у него заполнены (не нули).
    ks, vs = shim._require_i8(kc_on, "gate")
    assert float(ks.max()) > 0 and float(vs.max()) > 0, "таблицы масштабов пусты после записи"
    shim._KV_I8 = True


# --- 8. ПРЕФИЛЛ ИДЁТ РОЛЕВЫМ ЯДРОМ, А НЕ ЧЕРЕЗ ДЕКВАНТОВАНИЕ -------------------------------------
def test_prefill_takes_role_kernel():
    """Пункт 4 задачи: при int8-пуле и d=256 префилл обязан идти ЯДРОМ РОЛЕЙ (volta_fwd_ws.cuh),
    а не разворачивать пул в fp16. Проверяется ПОВЕДЕНИЕМ -- по маршруту, который обвязка
    объявляет один раз на процесс, -- и ДВУСТОРОННЕ: при FA2SM70_KV_I8_BYTES=0 маршрут другой,
    а ответ обязан остаться тем же (разворот q*s точен)."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "integrations", "vllm_sm70"))
    for m in [m for m in list(sys.modules) if m.startswith("flash_attn_v100")]:
        del sys.modules[m]
    os.environ["FA2SM70_KV_I8"] = "1"
    import flash_attn_v100 as shim

    torch.manual_seed(13)
    Hkv, H, T, Sq, bs = 2, 8, 1024, 256, 16
    nblk = T // bs
    sc = 1.0 / math.sqrt(D)
    kc, vc = _pool(nblk, bs, Hkv, D)
    k = torch.randn(T, Hkv, D, device="cuda", dtype=torch.float16) * 0.4
    v = torch.randn(T, Hkv, D, device="cuda", dtype=torch.float16) * 0.4
    slots = torch.arange(T, device="cuda", dtype=torch.long)
    shim.reshape_and_cache_fp8(k, v, kc, vc, slots, "fp8_e4m3", 1.0, 1.0)
    q = (torch.randn(1, Sq, H, D, device="cuda", dtype=torch.float16) * 0.3).contiguous()
    bt = torch.arange(nblk, device="cuda", dtype=torch.int32).view(1, nblk)
    sl = torch.full((1,), T, device="cuda", dtype=torch.int32)

    shim._SAID.clear()
    shim._KV_I8_BYTES = True
    o_role = shim.flash_attn_prefill_paged(q, kc, vc, bt, sl, softmax_scale=sc,
                                           kv_cache_dtype="fp8_e4m3", k_scale=1.0, v_scale=1.0)
    assert any("role kernel" in s for s in shim._SAID), f"пошёл НЕ ролевым ядром: {shim._SAID}"

    shim._SAID.clear()
    shim._KV_I8_BYTES = False
    o_f16 = shim.flash_attn_prefill_paged(q, kc, vc, bt, sl, softmax_scale=sc,
                                          kv_cache_dtype="fp8_e4m3", k_scale=1.0, v_scale=1.0)
    assert any("fp16 dequant" in s for s in shim._SAID), f"выключатель не переключил маршрут: {shim._SAID}"
    shim._KV_I8_BYTES = True

    # ОДИН ФОРМАТ -- ОДИН ОТВЕТ: два маршрута чтения одного пула обязаны совпасть с точностью до
    # порядка сложений в разных ядрах (разворот q*s точен, поэтому расхождение здесь -- только fp16).
    rel = _rel_l2(o_role, o_f16)
    print(f"\n[маршруты] ядро ролей против fp16-разворота на ОДНОМ пуле: relL2={rel:.3e}")
    assert rel < 5e-3, f"два маршрута чтения ОДНОГО формата разошлись: {rel:.3e}"


# --- 9. ОБРАТНАЯ СОВМЕСТИМОСТЬ ВХОДА ДЕКОДА -----------------------------------------------------
def test_decode_into_back_compat(ext):
    """Вход декода получил ДВА новых аргумента (таблицы масштабов), и прежние вызовы на 17 аргументов
    обязаны работать без изменений -- ими ходит ВЕСЬ e4m3-декод обвязки.

    Тест не теоретический: первая версия объявила их как `torch::Tensor = torch::Tensor()`, что в
    Python даёт None, а None в этой сборке torch НЕ приводится к torch::Tensor -- то есть e4m3-декод
    падал бы с TypeError на первом же токене после перезапуска сервера. Поймано замером."""
    B, H, Hkv, T, bs, ns = 1, 12, 2, 1024, 16, 8
    nblk = T // bs
    sc = 1.0 / math.sqrt(D)
    kc, vc = _pool(nblk, bs, Hkv, D)
    q = torch.randn(B, H, 1, D, device="cuda", dtype=torch.float16)
    km = torch.zeros(B, Hkv, device="cuda", dtype=torch.float32)
    bt = torch.arange(nblk, device="cuda", dtype=torch.int32).view(B, nblk)
    A = torch.empty(B * H * ns * D, device="cuda", dtype=torch.float32)
    Bw = torch.empty(2 * B * H * ns, device="cuda", dtype=torch.float32)
    out = torch.empty(B, H, D, device="cuda", dtype=torch.float16)
    sl = torch.full((B,), T, device="cuda", dtype=torch.int32)
    # 17 аргументов, e4m3 -- ровно та форма вызова, что стоит в _decode_e4m3.
    ext.flash_decode_defer_mqa_paged_into(q, kc, vc, km, bt, bs, sc, ns, nblk * bs, sl,
                                          True, 1.0, 1.0, A, Bw, out, H * D)
    # и явный None -- то же самое.
    ext.flash_decode_defer_mqa_paged_into(q, kc, vc, km, bt, bs, sc, ns, nblk * bs, sl,
                                          True, 1.0, 1.0, A, Bw, out, H * D, None, None)
    assert torch.isfinite(out.float()).all()


# --- 10. ВЫТЕСНЕНИЕ И ПЕРЕИСПОЛЬЗОВАНИЕ БЛОКА ----------------------------------------------------
@pytest.mark.parametrize("Hkv,bs", [(2, 16), (4, 32)])
def test_slot_reuse_keeps_bytes_and_scale_together(ext, Hkv, bs):
    """Требование задачи: «те же гарантии при вытеснении блоков», что и у таблицы границы ||k||.

    У масштаба гарантия ДАЖЕ СИЛЬНЕЕ, и это надо проверить, а не заявить: граница ведётся по блоку и
    только растёт (переиспользованный блок оставляет её завышенной -- законно, нужна лишь верхняя
    оценка), а масштаб пишется ПО СЛОТУ тем же вызовом, что и байты слота. Значит после
    переиспользования слота под ДРУГИЕ данные и байты, и масштаб обязаны быть новыми -- никакого
    следа прежнего постояльца, иначе чтение дало бы правдоподобный мусор."""
    torch.manual_seed(int(Hkv) * 97 + bs)
    n, nblk = 256, 256 // bs + 1
    kc, vc = _pool(nblk, bs, Hkv, D)
    kst, vst = _tabs(nblk, bs, Hkv)
    slots = torch.arange(n, device="cuda", dtype=torch.long)

    k1 = torch.randn(n, Hkv, D, device="cuda", dtype=torch.float16) * 0.05   # мелкий постоялец
    v1 = torch.randn(n, Hkv, D, device="cuda", dtype=torch.float16) * 0.05
    ext.reshape_and_cache_i8(k1, v1, kc, vc, slots, kst, vst)
    s1 = kst.clone()

    k2 = torch.randn(n, Hkv, D, device="cuda", dtype=torch.float16) * 4.0    # крупный -- те же слоты
    v2 = torch.randn(n, Hkv, D, device="cuda", dtype=torch.float16) * 4.0
    ext.reshape_and_cache_i8(k2, v2, kc, vc, slots, kst, vst)

    qk2, sk2 = _ref_quant(k2)
    qv2, sv2 = _ref_quant(v2)
    blk, off = slots // bs, slots % bs
    assert torch.equal(kc.view(torch.int8)[blk, off].float(), qk2), "байты остались от прежнего постояльца"
    assert torch.equal(kst.view(nblk, bs, Hkv)[blk, off], sk2), "масштаб остался от прежнего постояльца"
    assert torch.equal(vst.view(nblk, bs, Hkv)[blk, off], sv2), "масштаб V остался от прежнего постояльца"
    assert not torch.equal(s1, kst), "переиспользование слота НЕ обновило масштаб (тест бесполезен)"
    # И чтение через страничный сборщик даёт НОВЫЕ значения, а не смесь старого масштаба с новыми байтами.
    perm = torch.arange(nblk, device="cuda", dtype=torch.int32)
    fk = torch.empty(Hkv, n, D, device="cuda", dtype=torch.float16)
    fv = torch.empty(Hkv, n, D, device="cuda", dtype=torch.float16)
    ext.gather_paged_kv_i8_pool_f16(kc, vc, kst, vst, perm, n, fk, fv)
    assert torch.equal(fk, (qk2 * sk2.unsqueeze(-1)).permute(1, 0, 2).half()), "сборка после переиспользования врёт"


# --- 11. ЗАХВАТ ГРАФА: УКАЗАТЕЛИ ТАБЛИЦ ЗАПЕКАЮТСЯ, ЗНАЧЕНИЯ ЧИТАЮТСЯ ПРИ ПОВТОРЕ -----------------
def test_decode_i8_under_cuda_graph(ext):
    """Декод сервера исполняется ИЗ ЗАХВАЧЕННОГО ГРАФА: наш Python отработает один раз при захвате,
    а дальше граф повторяется. Значит таблицы масштабов обязаны вести себя как _bound8 -- постоянный
    адрес, значения читаются ПРИ КАЖДОМ ПОВТОРЕ. Если бы таблица создавалась заново на вызов, граф
    запёк бы указатель на мусор: сервер бы работал и врал.

    Проверяется ровно это: граф захвачен на ОДНИХ данных, затем пул и таблицы переписаны IN PLACE
    под ДРУГИЕ данные, и повтор графа обязан дать НОВЫЙ ответ, совпадающий с эталоном по новым."""
    B, H, Hkv, T, bs, ns = 1, 12, 2, 1024, 16, 8
    nblk = T // bs
    sc = 1.0 / math.sqrt(D)
    kc, vc = _pool(nblk, bs, Hkv, D)
    kst, vst = _tabs(nblk, bs, Hkv)
    slots = torch.arange(T, device="cuda", dtype=torch.long)
    bt = torch.arange(nblk, device="cuda", dtype=torch.int32).view(B, nblk)
    q = torch.randn(B, H, 1, D, device="cuda", dtype=torch.float16) * 0.3
    km = torch.zeros(B, Hkv, device="cuda", dtype=torch.float32)
    A = torch.empty(B * H * ns * D, device="cuda", dtype=torch.float32)
    Bw = torch.empty(2 * B * H * ns, device="cuda", dtype=torch.float32)
    out = torch.empty(B, H, D, device="cuda", dtype=torch.float16)
    sl = torch.full((B,), T, device="cuda", dtype=torch.int32)

    def call():
        ext.flash_decode_defer_mqa_paged_into(q, kc, vc, km, bt, bs, sc, ns, nblk * bs, sl,
                                              False, 1.0, 1.0, A, Bw, out, H * D, kst, vst)

    k1 = torch.randn(T, Hkv, D, device="cuda", dtype=torch.float16) * 0.4
    v1 = torch.randn(T, Hkv, D, device="cuda", dtype=torch.float16) * 0.4
    ext.reshape_and_cache_i8(k1, v1, kc, vc, slots, kst, vst)

    s = torch.cuda.Stream()                       # прогрев на боковом потоке -- как делает vLLM
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call()
    g.replay()
    torch.cuda.synchronize()
    o_first = out.clone()

    # ТЕ ЖЕ буферы, ДРУГИЕ данные: пул и таблицы переписываются на месте (как это делает store).
    k2 = torch.randn(T, Hkv, D, device="cuda", dtype=torch.float16) * 2.5
    v2 = torch.randn(T, Hkv, D, device="cuda", dtype=torch.float16) * 2.5
    ext.reshape_and_cache_i8(k2, v2, kc, vc, slots, kst, vst)
    g.replay()
    torch.cuda.synchronize()
    o_second = out.clone()

    assert not torch.allclose(o_first, o_second), "повтор графа НЕ увидел новых данных пула"
    qk, sk = _ref_quant(k2)
    qv, sv = _ref_quant(v2)
    kr = (qk * sk.unsqueeze(-1)).permute(1, 0, 2).repeat_interleave(H // Hkv, 0).float()
    vr = (qv * sv.unsqueeze(-1)).permute(1, 0, 2).repeat_interleave(H // Hkv, 0).float()
    p = torch.softmax(torch.einsum("hd,htd->ht", q[0, :, 0].float(), kr) * sc, -1)
    ref = torch.einsum("ht,htd->hd", p, vr)
    r = _rel_l2(o_second[0], ref)
    print(f"\n[граф] повтор после перезаписи пула: relL2={r:.3e}")
    assert r < 2e-3, f"повтор графа разошёлся с эталоном: {r:.3e}"
