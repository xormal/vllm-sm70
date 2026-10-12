"""ЭТАЛОННОЕ блочно-разреженное внимание с отбором по ИСТИННОЙ МАССЕ (оракул).

Назначение -- ИЗМЕРЕНИЕ, не скорость. Это тот же отбор, который считает tools/attn_oracle.py
(потолок любого отборщика: блоки ключей ранжируются по настоящей softmax-массе), но применённый
внутри работающей модели во всех 16 слоях полного внимания. Так меряется НАКОПЛЕНИЕ ошибки по
глубине, которого послойная таблица оракула не видит.

СЕМАНТИКА (обязана совпадать с оракулом, иначе сравнение сквозной ошибки с послойной бессмысленно):
  * плитка запросов BQ=64, плитка ключей BK=64;
  * масса блока ключей = сумма НОРМИРОВАННЫХ вероятностей блока, усреднённая по 64 запросам плитки;
  * оставляем nk = max(1, round(dens * nb)) блоков с наибольшей массой, где nb -- число блоков в
    ПРИЧИННОМ префиксе ИМЕННО ЭТОЙ плитки запросов (то есть плотность считается от префикса);
  * софтмакс ПЕРЕНОРМИРУЕТСЯ по оставшимся блокам (ровно то, что сделало бы ядро, пропустив блок).

АРИФМЕТИКА. Счёт идёт в fp16 на тензорных ядрах (иначе прогон модели не помещается в разумное
время), суммы -- в fp32. Контрольный прогон (dens=1.0) идёт ТЕМ ЖЕ путём, поэтому в разнице
"разреженный минус плотный" арифметика сокращается, а не подмешивается. Точный fp32-режим
(exact=True) оставлен для сверки шума этого решения -- он и меряет пол.
"""
import torch

BQ_DEF = 64
BK_DEF = 64
QT_DEF = 256


def _neg_inf(dtype):
    return torch.finfo(dtype).min / 2


def sparse_attn(q, kb, vb, scale, dens, p0, BQ=BQ_DEF, BK=BK_DEF, QT=QT_DEF,
                exact=False, out=None, stats=None):
    """Разреженное внимание одной последовательности.

    q  -- [Sq, H, d]   запросы (абсолютные позиции p0 .. p0+Sq-1)
    kb -- [Hkv, T, d]  ключи всего префикса (может быть нераскладной вид -- это допустимо)
    vb -- [Hkv, T, d]
    dens -- доля оставляемых блоков ключей; >=1.0 даёт плотное внимание ТЕМ ЖЕ путём.
    Возвращает [Sq, H, d] в dtype запросов.
    """
    Sq, H, d = q.shape
    Hkv, T, d2 = kb.shape
    assert d == d2, (q.shape, kb.shape)
    assert H % Hkv == 0, (H, Hkv)
    assert p0 + Sq <= T, (p0, Sq, T)
    G = H // Hkv
    comp = torch.float32 if exact else torch.float16
    dev = q.device
    if out is None:
        out = torch.empty_like(q)
    QT = max(BQ, (QT // BQ) * BQ)
    ninf = _neg_inf(comp)
    # ВЫРАВНИВАНИЕ ПО БЛОКУ КЛЮЧЕЙ. Все окна ключей делаются кратными BK, а добавленные столбцы
    # глушатся в -inf. Без этого хвостовой НЕполный блок ломает раскладку: срез последней оси
    # перестаёт быть непрерывным, и `view` на нём либо падает, либо (через reshape) молча уходит в
    # копию -- тогда вычёркивание блоков применилось бы к копии, а не к самим весам.
    if T % BK:
        Tp = ((T + BK - 1) // BK) * BK
        kb = torch.nn.functional.pad(kb, (0, 0, 0, Tp - T))
        vb = torch.nn.functional.pad(vb, (0, 0, 0, Tp - T))
        T = Tp

    for g in range(Hkv):
        kg = kb[g]                                   # [T, d]
        vg = vb[g]
        if kg.dtype != comp:
            kg = kg.to(comp)
            vg = vg.to(comp)
        for s in range(0, Sq, QT):
            e = min(s + QT, Sq)
            nq = e - s
            c0 = p0 + s                              # абсолютная позиция первой строки плитки
            end = p0 + e                             # причинный предел последней строки (искл.)
            nb = (end + BK - 1) // BK
            endp = nb * BK                           # окно ключей, кратное блоку
            qt = q[s:e, g * G:(g + 1) * G, :].permute(1, 0, 2).to(comp)   # [G, nq, d]
            sco = (qt.reshape(G * nq, d) * scale) @ kg[:endp].t()          # [G*nq, endp]
            sco = sco.view(G, nq, endp)
            if endp > end:                           # добивка окна -- вне причинного предела
                sco[:, :, end:endp] = ninf
            # причинная маска нужна ТОЛЬКО в диагональной полосе: столбцы < c0 видны всем строкам
            band = sco[:, :, c0:end]
            ii = torch.arange(nq, device=dev)
            band.masked_fill_((ii.view(1, 1, nq) > ii.view(1, nq, 1)), ninf)
            m = sco.amax(-1, keepdim=True)
            w = sco.sub_(m).exp_()                   # [G, nq, endp], значения в (0,1]
            # --- массы блоков ключей -------------------------------------------------
            bs = w.view(G, nq, nb, BK).sum(-1, dtype=torch.float32)
            den = bs.sum(-1)                         # [G, nq] знаменатель плотного софтмакса
            qb = (nq + BQ - 1) // BQ
            pn = bs / den.unsqueeze(-1)
            if qb * BQ != nq:                        # хвостовая неполная плитка запросов
                pad = qb * BQ - nq
                cnt = torch.full((qb,), float(BQ), device=dev)
                cnt[-1] = BQ - pad
                pn = torch.nn.functional.pad(pn, (0, 0, 0, pad))
                mass = pn.view(G, qb, BQ, nb).sum(2) / cnt.view(1, qb, 1)
            else:
                mass = pn.view(G, qb, BQ, nb).mean(2)                     # [G, qb, nb]
            # nb и nk СВОИ у каждой плитки запросов (плотность от её причинного префикса)
            endq = torch.arange(1, qb + 1, device=dev) * BQ + c0
            endq = endq.clamp(max=end)
            nbq = torch.ceil(endq.float() / BK).long()
            if dens >= 1.0:
                nkq = nbq
            else:
                nkq = torch.clamp(torch.round(nbq.float() * dens).long(), min=1)
            rank = mass.argsort(-1, descending=True).argsort(-1)
            keep = rank < nkq.view(1, qb, 1)                              # [G, qb, nb]
            keepf = keep.to(comp)
            if stats is not None:
                stats["kept"] += float(nkq.sum().item()) * G
                stats["tot"] += float(nbq.sum().item()) * G
            # --- вычёркивание блоков и перенормировка --------------------------------
            for qi in range(qb):
                r0 = qi * BQ
                r1 = min(r0 + BQ, nq)
                wb = w[:, r0:r1, :]                  # ВИД, не копия: mul_ пишет в сами веса
                rows = r1 - r0
                wb.view(G, rows, nb, BK).mul_(keepf[:, qi, :].view(G, 1, nb, 1))
                kden = (bs[:, r0:r1, :] * keepf[:, qi, :].view(G, 1, nb)).sum(-1)  # [G,rows]
                ob = (wb.reshape(G * rows, endp) @ vg[:endp]).float()
                ob = ob / kden.reshape(G * rows, 1).clamp_min(1e-30)
                out[s + r0:s + r1, g * G:(g + 1) * G, :] = \
                    ob.view(G, rows, d).permute(1, 0, 2).to(out.dtype)
            del w, sco, bs, pn, mass, keep, keepf
    return out


def perturb_kv(kb, vb, sigma, seed):
    """МУЛЬТИПЛИКАТИВНЫЙ несмещённый шум на K/V -- модель ошибки ФОРМАТА хранения.

    Нужен для сравнения ХАРАКТЕРА накопления: ошибка квантования поэлементная и несмещённая, ошибка
    разреженности структурная. Если сквозной множитель у них разный, то послойный бюджет нельзя
    переносить один в один -- именно это и надо померить, а не предположить.
    Возвращает КОПИИ (буфер сборки переиспользуется дальше по вызовам)."""
    g = torch.Generator(device=kb.device)
    g.manual_seed(int(seed) & 0x7FFFFFFF)
    ko = kb.clone()
    vo = vb.clone()
    n = torch.empty_like(ko, dtype=torch.float16)
    n.normal_(0.0, sigma, generator=g)
    ko.mul_(n.add_(1.0))
    n.normal_(0.0, sigma, generator=g)
    vo.mul_(n.add_(1.0))
    return ko, vo


def coarsen_kv(kb, vb, keep_bits: int):
    """ДЕТЕРМИНИРОВАННОЕ огрубление мантиссы K/V -- ошибка НАСТОЯЩЕГО формата, не шум.

    Зачем отдельно от perturb_kv: случайный множитель перерисовывается на каждом вызове, поэтому
    один и тот же элемент K/V получает РАЗНУЮ ошибку в разных куска́х запроса, и накопление частично
    усредняется. Ошибка формата, наоборот, ЗАМОРОЖЕНА в элементе: все запросы видят одно и то же
    отклонение, и оно складывается согласованно. Раз вердикт держится на отношении множителей
    накопления, занижать множитель квантования нельзя -- иначе разреженность осуждается зря.

    K/V уже лежат на сетке e4m3 (3 бита мантиссы). Оставляя keep_bits<3, получаем сетку РОВНО ВДВОЕ
    (вчетверо) грубее -- ошибка того же характера и происхождения, только измеримой амплитуды;
    к оплаченной она приводится по линейности отклика (проверяется парой точек).
    Округление к ближайшему, связи -- к чётному. Возвращает КОПИИ."""
    if keep_bits >= 10:
        return kb, vb
    k = 10 - int(keep_bits)                 # столько младших бит мантиссы fp16 обнуляем
    half = 1 << (k - 1)
    mask = ~((1 << k) - 1) & 0xFFFF
    out = []
    for t in (kb, vb):
        i = t.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
        lsb = (i >> k) & 1                  # к чётному: +half-1 если младший бит остатка 0
        i = i + (half - 1 + lsb)
        i = i & mask
        # переполнение мантиссы корректно переносится в экспоненту: биты идут подряд
        out.append((i & 0xFFFF).to(torch.int16).view(torch.float16).view(t.shape).clone())
    return out[0], out[1]


def dense_attn(q, kb, vb, scale, p0, exact=True):
    """Плотный эталон в fp32 -- для сверки самого этого решения (не для прогона модели)."""
    Sq, H, d = q.shape
    Hkv, T, _ = kb.shape
    G = H // Hkv
    comp = torch.float32 if exact else torch.float16
    out = torch.empty_like(q)
    for g in range(Hkv):
        kg = kb[g].to(comp)
        vg = vb[g].to(comp)
        for s in range(0, Sq, QT_DEF):
            e = min(s + QT_DEF, Sq)
            nq = e - s
            c0, end = p0 + s, p0 + e
            qt = q[s:e, g * G:(g + 1) * G, :].permute(1, 0, 2).to(comp)
            sco = ((qt.reshape(G * nq, d) * scale) @ kg[:end].t()).view(G, nq, end)
            ii = torch.arange(nq, device=q.device)
            sco[:, :, c0:end].masked_fill_(ii.view(1, 1, nq) > ii.view(1, nq, 1),
                                           _neg_inf(comp))
            p = torch.softmax(sco.float(), -1)
            ob = p.reshape(G * nq, end).to(comp) @ vg[:end]
            out[s:e, g * G:(g + 1) * G, :] = ob.view(G, nq, d).permute(1, 0, 2).to(out.dtype)
    return out
