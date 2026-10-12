# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# ================================================================================================
# ПОДМЕНА ЧАНКОВОГО СКАНА GATED-DELTA-NET В vLLM -- БЕЗ ПРАВКИ ИСХОДНИКОВ vLLM.
#
# ЧТО ПОДМЕНЯЕТСЯ. vllm/model_executor/models/qwen3_next.py делает на импорте
#     from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule
# и вызывает это имя на префилле (строка ~815). Класс Qwen3_5GatedDeltaNet наследует _forward_core
# оттуда же, поэтому у Qwen3.5 работает РОВНО этот вызов.
#
# ПОЧЕМУ ХУК, А НЕ ПРОСТО setattr ПОСЛЕ ИМПОРТА. `from X import f` копирует ССЫЛКУ в globals
# модуля-потребителя в момент импорта. Значит подменить X.f ПОСЛЕ того, как qwen3_next
# импортирован, БЕСПОЛЕЗНО -- у него уже лежит старая ссылка. Подменять надо в окне между
# "пакет fla.ops исполнен" и "qwen3_next связал имя", а это окно существует только внутри
# исполнения самого оператора import. Отсюда finder в sys.meta_path, который патчит атрибут
# ПАКЕТА сразу после exec_module.
#
# ПОЧЕМУ ЭТО ЗАПУСКАЕТСЯ ИЗ sitecustomize. Воркеры vLLM поднимаются spawn'ом, и их окружение НЕ
# несёт ничего из окружения подъёмника, кроме того, что наследуется процессом. PYTHONPATH
# наследуется, а site.py импортирует sitecustomize из любого каталога sys.path -- значит хук
# встаёт в КАЖДОМ воркере сам, без единой правки vLLM и без надежды на то, что кто-то вызовет
# нашу функцию инициализации. Проверка через /proc/<pid>/environ у воркеров НЕ ГОДИТСЯ (их
# spawn-блоб не содержит даже CUDA_VISIBLE_DEVICES) -- поэтому и факт врезки, и её эффект
# проверяются ПОВЕДЕНИЕМ, а не окружением.
#
# ВЫКЛЮЧАТЕЛЬ: FA2SM70_GDN=0 -> хук не ставится вообще (контроль для A/B).
# ЖУРНАЛ:      FA2SM70_GDN_LOG=1 -> одна строка на процесс о том, что подмена встала,
#              и одна строка при КАЖДОМ откате к Triton с причиной (ловит "тихо не работает").
# ================================================================================================
from __future__ import annotations

import os
import sys

_ENV = "FA2SM70_GDN"
_LOG = os.environ.get("FA2SM70_GDN_LOG", "0") == "1"
_TARGET = "vllm.model_executor.layers.fla.ops"

# Счётчики -- ЕДИНСТВЕННЫЙ честный признак того, что подмена работает в бою. Импортировать их
# может кто угодно (гейт читает их через отдельный вызов внутри воркера).
_ГОЛОВА = os.environ.get("FA2SM70_GDN_HEAD", "0") == "1"

STATS = {"calls": 0, "ours": 0, "fallback": 0, "reason": "", "off": False}

# ------------------------------------------------------------------------------------------------
# ПЕРЕКЛЮЧАТЕЛЬ ДЛЯ A/B В ОДНОМ ПОДЪЁМЕ. Путь к файлу-флажку; "0" внутри = идти образцом.
#
# ЗАЧЕМ. Дисциплина замера требует чередовать контроль и кандидата В ОДНОМ ПОДЪЁМЕ: два разных
# подъёма отличаются не только осью A/B, но и состоянием распределителя, префикс-кэшем, прогревом
# и дрейфом частот, а разница, которую мы ищем, -- единицы процентов сквозного TTFT. Переменная
# окружения на эту роль НЕ ГОДИТСЯ: воркеры vLLM поднимаются spawn'ом, и снаружи их окружение уже
# не изменить. Файл -- единственный канал, который виден и подъёмщику, и всем воркерам сразу.
#
# ЧТО ИМЕННО ПЕРЕКЛЮЧАЕТСЯ. Ровно то же, что и FA2SM70_GDN=0: вызов уходит в orig(...) с теми же
# позиционными аргументами, то есть в НАСТОЯЩИЙ trion-овский chunk_gated_delta_rule. Разница с
# честным подъёмом при FA2SM70_GDN=0 -- один вызов python-обёртки и одно чтение флажка на вызов
# (единицы микросекунд против 20.66 мс скана), и эта цена ОДИНАКОВА в обеих ветках, поэтому на
# отношение она не влияет. Строгий гейт "при FA2SM70_GDN=0 строки НЕТ" всё равно проверяется
# отдельным подъёмом -- флажок его не заменяет, а дополняет.
#
# ПО УМОЛЧАНИЮ ВЫКЛЮЧЕН: без FA2SM70_GDN_SWITCH ни одного лишнего обращения к диску не делается.
# ------------------------------------------------------------------------------------------------
_SWITCH = os.environ.get("FA2SM70_GDN_SWITCH", "")


def _log(msg: str) -> None:
    if _LOG:
        print(f"[fa2_sm70 gdn] {msg}", file=sys.stderr, flush=True)


# [ПЕРЕМЕННАЯ -- ОДИН РАЗ, ФАЙЛ -- ДИНАМИЧЕСКИ, 31.08]
# `_off_now` зовётся на КАЖДЫЙ `chunk_gated_delta_rule`, то есть на каждый из 48 слоёв GDN в
# префилле, и читал окружение каждый раз. Динамика нужна только ФАЙЛУ-флажку (он и заводился
# ради чередования режимов ВНУТРИ одного прогона -- см. его разбор ниже); переменная окружения
# в рантайме не меняется, поэтому её читаем при импорте.
_ВЫКЛ_ОКРУЖЕНИЕМ = os.environ.get(_ENV, "1") == "0"


def _off_now() -> bool:
    """Идти ли образцом ПРЯМО СЕЙЧАС. Нечитаемый/отсутствующий флажок = НЕ выключено: сбой
    файловой системы не должен молча превращать замер кандидата в замер контроля."""
    if _ВЫКЛ_ОКРУЖЕНИЕМ:
        return True
    if _SWITCH:
        try:
            with open(_SWITCH) as f:
                return f.read(8).strip() == "0"
        except OSError:
            return False
    return False


def _make_replacement(orig):
    """Наша обёртка с ОТКАТОМ. Откат обязателен: у vLLM есть режимы (профилировочный прогон,
    спекулятивные ветки, другие модели с той же fla), где форма не та, под которую ядра
    инстанцированы. Падать там нельзя -- надо считать образцом."""
    import torch

    def chunk_gated_delta_rule(
        q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False,
        cu_seqlens=None, head_first=False, use_qk_l2norm_in_kernel=False,
    ):
        STATS["calls"] += 1
        off = _off_now()
        if off != STATS["off"]:
            # ПЕРЕХОД РЕЖИМА -- в журнал. Без этой строки чередование внутри одного подъёма
            # неразличимо снаружи, и отчёт пришлось бы строить на вере в то, что флажок доехал.
            STATS["off"] = off
            _log(f"режим -> {'TRITON (контроль)' if off else 'НАШ (кандидат)'}"
                 f" на вызове {STATS['calls']}, наших {STATS['ours']}")
        if off:
            STATS["fallback"] += 1
            STATS["reason"] = "выключено переменной"
            return orig(q, k, v, g, beta, scale, initial_state, output_final_state,
                        cu_seqlens, head_first, use_qk_l2norm_in_kernel)
        # СЪЁМ БОЕВОГО ВИДА. Отладка слитого тела идёт на ТЕХ САМЫХ тензорах, что даёт модель, а
        # не на синтетике: форма, раскладка, диапазоны и cu_seqlens у боевого куска свои, и именно
        # на них ловятся тихие расхождения. Пишем ОДИН вызов (по номеру), со всеми входами.
        _дамп = os.environ.get("FA2SM70_GDN_ДАМП")
        if _дамп and STATS["calls"] == int(os.environ.get("FA2SM70_GDN_ДАМП_ВЫЗОВ", "8")):
            try:
                import torch as _t
                _t.save({"q": q.cpu(), "k": k.cpu(), "v": v.cpu(), "g": g.cpu(),
                         "beta": beta.cpu(), "scale": scale,
                         "initial_state": None if initial_state is None else initial_state.cpu(),
                         "cu_seqlens": None if cu_seqlens is None else cu_seqlens.cpu(),
                         "output_final_state": bool(output_final_state),
                         "head_first": bool(head_first),
                         "use_qk_l2norm_in_kernel": bool(use_qk_l2norm_in_kernel),
                         "вызов": STATS["calls"]}, _дамп)
                _log(f"СНЯТ БОЕВОЙ ВИД в {_дамп}: вызов {STATS['calls']}, "
                     f"q={tuple(q.shape)}/{q.dtype} v={tuple(v.shape)} "
                     f"cu={None if cu_seqlens is None else cu_seqlens.tolist()[:6]}")
            except Exception as _e:  # noqa: BLE001
                _log(f"съём не удался: {type(_e).__name__}: {_e}")
        try:
            from fa2_sm70 import gdn as G
            from vllm.model_executor.layers.fla.ops.cumsum import chunk_local_cumsum
            from vllm.model_executor.layers.fla.ops.l2norm import l2norm_fwd

            if scale is None:
                scale = k.shape[-1] ** -0.5
            if head_first or not G.supported(q, k, v, g, beta, initial_state, cu_seqlens):
                raise NotImplementedError("форма вне инстанцированных")
            # Голова цепочки ОСТАЁТСЯ трионовской (см. fa2_sm70/gdn.py): 0.8% скана и ни одного
            # матричного умножения. q,k сразу переводятся в fp16 -- на bf16-развёртке это и есть
            # та самая структурная конверсия, и она делается ОДИН раз здесь, а не внутри ядер.
            #
            # ИМЕНА q,k НЕ ПЕРЕПРИСВАИВАЮТСЯ. Иначе отказ, случившийся ПОСЛЕ нормировки, ушёл бы
            # в откат с уже нормированными q,k И с use_qk_l2norm_in_kernel=True, то есть образец
            # нормировал бы их ВТОРОЙ раз. Численно это ничего не меняет (нормировка идемпотентна),
            # но полагаться на идемпотентность в аварийном пути -- значит держать корректность
            # отката на свойстве, которое никто не проверяет.
            # [03.09] ГОЛОВА ЦЕПОЧКИ ПЕРЕВЕДЕНА НА СВОЁ ЯДРО (записка 25 §108).
            # Комментарий выше утверждал «0.8 % скана», и это оказалось неверно: замер
            # `tools/gdn_vs_triton.py` на боевых формах (T=4) дал l2norm+cumsum 0.273 мс из
            # 0.418 мс всего нашего GDN -- 65 %, то есть 13 мс на шаг декода из 58.8. Причина
            # не в работе (её тут на микросекунды), а в ТРЁХ отдельных пусках Triton с
            # подготовкой. Слитое ядро: q,k побайтово (relL2 0.000e+00), g 3.4e-08, x10-16.
            # Условия применимости проверяются числом; иначе -- прежний трионовский путь.
            _голова = None
            if (_ГОЛОВА and use_qk_l2norm_in_kernel and q.shape[-1] == 128
                    and (cu_seqlens is None or cu_seqlens.numel() == 2)):
                try:
                    from fa2_sm70 import _ext as _E
                    _qh = q.to(torch.float16).contiguous()
                    _kh = k.to(torch.float16).contiguous()
                    _gh = g.to(torch.float32).contiguous()
                    qn, kn, gg = _E.gdn_head_ext().gdn_head(_qh, _kh, _gh, 64, 1e-6)
                    _голова = True
                except Exception as _e:  # noqa: BLE001
                    _log(f"голова GDN: своё ядро отказало ({type(_e).__name__}), иду прежним путём")
                    _голова = None
            if _голова is None:
                qn, kn = (l2norm_fwd(q), l2norm_fwd(k)) if use_qk_l2norm_in_kernel else (q, k)
                gg = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens)
            o, ht = G.chunk_gated_delta_rule_volta(
                qn.to(torch.float16).contiguous(), kn.to(torch.float16).contiguous(),
                v, gg, beta, float(scale), initial_state, bool(output_final_state),
                cu_seqlens, v.dtype)
            STATS["ours"] += 1
            # ПРИЗНАК УСПЕХА -- симметрично строке про откат, и он ОБЯЗАТЕЛЕН. Без него снаружи
            # НЕРАЗЛИЧИМЫ «наш путь отработал» и «молча откатились на Triton»: откат тотальный
            # (except Exception), STATS живут внутри spawn-воркера, а в логе при успехе не было
            # ни строки. Сквозной A/B без этого признака измеряет неизвестно что -- ровно та
            # ошибка, из-за которой пять готовых ядер полдня считались отгруженными.
            # Печатаем на СТЕПЕНЯХ ДВОЙКИ (1, 2, 4, 8, ...), а не только на первом вызове: гейт
            # врезки требует не факта "сработало один раз", а РОСТА счётчика -- иначе один
            # успешный вызов на профилировочном прогоне выглядел бы точно так же, как работающий
            # боевой путь. Степеней двойки на весь замер около двух десятков, потоком это не
            # становится.
            if STATS["ours"] & (STATS["ours"] - 1) == 0:
                _log(f"наш скан отработал (вызовов {STATS['calls']}, наших {STATS['ours']})")
            return o, ht
        except torch.OutOfMemoryError:
            # ОТКАТ НЕ ПРИМЕНЯЕТСЯ К НЕХВАТКЕ ПАМЯТИ. Замерено 11.08.2026: на вызове 496 наш скан
            # попросил 92 МиБ при 63 свободных, тотальный except увёл управление в Triton, тот на
            # новой форме пошёл в АВТОТЮНЕР (он бенчит все конфигурации), запросил ЕЩЁ БОЛЬШЕ -- и
            # движок умер. То есть запасной путь ТЯЖЕЛЕЕ основного и включается ровно тогда, когда
            # памяти нет. Восстановимый отказ ОДНОГО запроса превращался в смерть сервера.
            #
            # ПРАВИЛО: запасной путь применим только к отказам ФОРМЫ (мы не умеем считать это), а
            # не к отказам РЕСУРСА (никто не умеет посчитать это сейчас). Ресурсный отказ отдаём
            # движку -- у него есть вытеснение и пересборка пакета.
            STATS["fallback"] += 1
            STATS["reason"] = f"вызов {STATS['calls']}: нехватка памяти -- ОТДАНО ДВИЖКУ, без отката"
            _log(f"НЕХВАТКА ПАМЯТИ на вызове {STATS['calls']}: отката НЕТ (он тяжелее), отдаём движку")
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
            raise
        except Exception as e:                      # noqa: BLE001 -- откат по ФОРМЕ обязан быть тотальным
            STATS["fallback"] += 1
            # номер вызова в причине -- иначе один давний откат оставляет строку НАВСЕГДА и
            # выглядит как текущее состояние (reason при успехе не очищается намеренно: стирать
            # след отказа нельзя, но и выдавать его за настоящее положение тоже).
            STATS["reason"] = f"вызов {STATS['calls']}: {type(e).__name__}: {e}"
            _log(f"откат к Triton: {STATS['reason']}")
            return orig(q, k, v, g, beta, scale, initial_state, output_final_state,
                        cu_seqlens, head_first, use_qk_l2norm_in_kernel)

    # У ЗАМЕНЯЕМОЙ функции стоит @torch.compiler.disable, и снимать его НЕЛЬЗЯ. Скан вызывается
    # из custom-op qwen3_next (direct_register_custom_op), то есть сегодня dynamo внутрь не
    # заходит -- но подмена не должна ЗАВИСЕТЬ от этого: стоит вызову оказаться в
    # компилируемой области, и наш путь (вызов расширения, обращения к .shape/.dtype, построение
    # индексов на хосте) начал бы либо трассироваться, либо рвать граф на каждом из 48 слоёв.
    # Ставим тот же барьер, что у оригинала: свойство сохраняется вместе с именем.
    try:
        import torch as _t

        wrapped = _t.compiler.disable(chunk_gated_delta_rule)
    except Exception:  # noqa: BLE001
        wrapped = chunk_gated_delta_rule
    wrapped.__fa2sm70__ = True                      # признак для гейта: имя ПОДМЕНЕНО
    wrapped.__wrapped_orig__ = orig
    return wrapped


# Модули, которые держат имя У СЕБЯ (через `from ... import`). Их globals надо переписать
# отдельно, если они уже импортированы: ссылка скопирована и на пакет больше не смотрит.
_CONSUMERS = (
    "vllm.model_executor.models.qwen3_next",
    "vllm.model_executor.models.qwen3_5",
    "vllm.model_executor.layers.fla.ops.chunk",
)


def _make_rec_replacement(orig):
    """[ЗАДАЧА 180] Подмена РЕКУРРЕНТНОГО шага декода на наше ядро gdn_recurrent.
    Включается ОТДЕЛЬНЫМ флагом FA2SM70_GDN_REC=1 (умолчание -- выкл до обкатки).
    Сверено с Triton на 7 случаях (T=1 партии 1/8/48, fp16/bf16, мультитокен, PAD_SLOT_ID,
    спекуляция): o relL2 <= 2.3e-05, пул состояний ~5e-08; замер x4.87 (0.152 -> 0.031 мс,
    -5.8 мс/токен на 48 слоях). scratchpad/rec_test2.out, 19.08.2026."""
    import torch

    def fused_recurrent_gated_delta_rule(
        q, k, v, g, beta=None, scale=None, initial_state=None, inplace_final_state=True,
        cu_seqlens=None, ssm_state_indices=None, num_accepted_tokens=None,
        use_qk_l2norm_in_kernel=False,
    ):
        STATS["rec_calls"] = STATS.get("rec_calls", 0) + 1
        try:
            # Откат ТОЛЬКО по форме: всё, что наше ядро не покрывает, уходит образцу.
            if (initial_state is None or not inplace_final_state or beta is None
                    or q.shape[-1] != 128 or v.shape[-1] != 128
                    or g.dim() != 3 or beta.dim() != 3
                    or v.shape[2] % q.shape[2] != 0):
                raise NotImplementedError("форма вне нашего рекуррента")
            if scale is None:
                scale = q.shape[-1] ** -0.5
            cu = cu_seqlens
            if cu is not None and ssm_state_indices is not None \
                    and cu.dtype != ssm_state_indices.dtype:
                cu = cu.to(ssm_state_indices.dtype)
            from fa2_sm70 import _ext as _fe
            o = _fe.gdn_ext().gdn_recurrent(
                q.contiguous(), k.contiguous(), v.contiguous(), g.contiguous(),
                beta.contiguous(), float(scale), initial_state,
                ssm_state_indices, num_accepted_tokens, cu,
                bool(use_qk_l2norm_in_kernel))
            STATS["rec_ours"] = STATS.get("rec_ours", 0) + 1
            n = STATS["rec_ours"]
            if n & (n - 1) == 0:
                _log(f"наш РЕКУРРЕНТ декода отработал (вызовов {STATS['rec_calls']}, наших {n})")
            return o, initial_state
        except torch.OutOfMemoryError:
            raise                      # ресурсный отказ -- движку, не образцу (урок 11.08)
        except Exception as e:         # noqa: BLE001 -- отказ формы: путь образца
            STATS["rec_fallback"] = STATS.get("rec_fallback", 0) + 1
            if STATS["rec_fallback"] == 1:
                _log(f"рекуррент: откат на образец: {type(e).__name__}: {e}")
            return orig(q, k, v, g, beta, scale, initial_state, inplace_final_state,
                        cu_seqlens, ssm_state_indices, num_accepted_tokens,
                        use_qk_l2norm_in_kernel)

    fused_recurrent_gated_delta_rule.__fa2sm70__ = True
    return fused_recurrent_gated_delta_rule


def _make_gating_replacement(orig):
    """[ЗАДАЧА 180, боевой быстрый путь] Подмена fused_sigmoid_gating_delta_rule_update --
    ровно её зовёт _forward_core декода (fused_recurrent -- только запасная ветка). Гейтинг
    (softplus/sigmoid из a,b,A_log,dt_bias) слит В ЯДРО: отдельного пуска гейтинга нет."""
    import torch

    def fused_sigmoid_gating_delta_rule_update(
        A_log, a, b, dt_bias, q, k, v, beta=1.0, threshold=20.0, scale=None,
        initial_state=None, inplace_final_state=True, cu_seqlens=None,
        ssm_state_indices=None, num_accepted_tokens=None,
        use_qk_l2norm_in_kernel=False, is_kda=False,
    ):
        STATS["rec_calls"] = STATS.get("rec_calls", 0) + 1
        try:
            # Боевые формы (снято логом отката 19.08): a, b приходят ДВУМЕРНЫМИ [tok, HV] --
            # батч-ось наращивается вьюшкой (бесплатно); A_log/dt_bias -- fp16-параметрами.
            HV = v.shape[2]
            a3 = a if a.dim() == 3 else a.view(1, -1, HV)
            b3 = b if b.dim() == 3 else b.view(1, -1, HV)
            if (initial_state is None or not inplace_final_state or is_kda
                    or float(beta) != 1.0 or float(threshold) != 20.0
                    or q.shape[-1] != 128 or v.shape[-1] != 128
                    or a3.dim() != 3 or v.shape[2] % q.shape[2] != 0):
                raise NotImplementedError(
                    f"форма вне нашего слитого гейтинга: init={initial_state is not None} "
                    f"inpl={inplace_final_state} kda={is_kda} beta={beta} thr={threshold} "
                    f"q={tuple(q.shape)}/{q.dtype} v={tuple(v.shape)} a={tuple(a.shape)} "
                    f"b={tuple(b.shape)} Alog={tuple(A_log.shape)}/{A_log.dtype} "
                    f"cu={None if cu_seqlens is None else cu_seqlens.dtype} "
                    f"idx={None if ssm_state_indices is None else (tuple(ssm_state_indices.shape), ssm_state_indices.dtype)} "
                    f"pool={tuple(initial_state.shape) if initial_state is not None else None}")
            if scale is None:
                scale = q.shape[-1] ** -0.5
            cu = cu_seqlens
            if cu is not None and ssm_state_indices is not None \
                    and cu.dtype != ssm_state_indices.dtype:
                cu = cu.to(ssm_state_indices.dtype)
            from fa2_sm70 import _ext as _fe
            o = _fe.gdn_ext().gdn_rec_gating(
                A_log.float(), a3.contiguous(), b3.contiguous(), dt_bias.float(),
                q.contiguous(), k.contiguous(), v.contiguous(), float(scale),
                initial_state, ssm_state_indices, num_accepted_tokens, cu,
                bool(use_qk_l2norm_in_kernel),
                # сигнатура ядра выросла (ssm_idx_out, масштабы int16, строка ветвления):
                # у m.def нет умолчаний, все аргументы позиционные
                None, None, 0)
            STATS["rec_ours"] = STATS.get("rec_ours", 0) + 1
            n = STATS["rec_ours"]
            if n & (n - 1) == 0:
                _log(f"наш РЕКУРРЕНТ декода отработал (вызовов {STATS['rec_calls']}, наших {n})")
            return o, initial_state
        except torch.OutOfMemoryError:
            raise
        except Exception as e:  # noqa: BLE001
            STATS["rec_fallback"] = STATS.get("rec_fallback", 0) + 1
            if STATS["rec_fallback"] == 1:
                _log(f"рекуррент(гейтинг): откат на образец: {type(e).__name__}: {e}")
            return orig(A_log, a, b, dt_bias, q, k, v, beta, threshold, scale,
                        initial_state, inplace_final_state, cu_seqlens,
                        ssm_state_indices, num_accepted_tokens,
                        use_qk_l2norm_in_kernel, is_kda)

    fused_sigmoid_gating_delta_rule_update.__fa2sm70__ = True
    return fused_sigmoid_gating_delta_rule_update


def _patch_package(mod) -> None:
    orig = getattr(mod, "chunk_gated_delta_rule", None)
    if orig is None or getattr(orig, "__fa2sm70__", False):
        return
    repl = _make_replacement(orig)
    setattr(mod, "chunk_gated_delta_rule", repl)
    _rebind_consumers(repl)
    if os.environ.get("FA2SM70_GDN_REC", "0") == "1":
        gat_orig = getattr(mod, "fused_sigmoid_gating_delta_rule_update", None)
        if gat_orig is not None and not getattr(gat_orig, "__fa2sm70__", False):
            gat_repl = _make_gating_replacement(gat_orig)
            setattr(mod, "fused_sigmoid_gating_delta_rule_update", gat_repl)
            ng = 0
            for name in _CONSUMERS:
                m = sys.modules.get(name)
                if m is None:
                    continue
                cur = getattr(m, "fused_sigmoid_gating_delta_rule_update", None)
                if cur is not None and not getattr(cur, "__fa2sm70__", False):
                    setattr(m, "fused_sigmoid_gating_delta_rule_update", gat_repl)
                    ng += 1
            _log(f"подмена СЛИТОГО ГЕЙТИНГА встала (FA2SM70_GDN_REC=1), перевязано {ng}")
        rec_orig = getattr(mod, "fused_recurrent_gated_delta_rule", None)
        if rec_orig is not None and not getattr(rec_orig, "__fa2sm70__", False):
            rec_repl = _make_rec_replacement(rec_orig)
            setattr(mod, "fused_recurrent_gated_delta_rule", rec_repl)
            n = 0
            for name in _CONSUMERS:
                m = sys.modules.get(name)
                if m is None:
                    continue
                cur = getattr(m, "fused_recurrent_gated_delta_rule", None)
                if cur is not None and not getattr(cur, "__fa2sm70__", False):
                    setattr(m, "fused_recurrent_gated_delta_rule", rec_repl)
                    n += 1
            _log(f"подмена РЕКУРРЕНТА встала (FA2SM70_GDN_REC=1), перевязано {n}")
    _log(f"подмена встала в pid={os.getpid()}")


def _rebind_consumers(repl) -> int:
    """Переписать уже связанные имена. Возвращает число переписанных модулей.

    ЗАЧЕМ ОТДЕЛЬНО. Хук в meta_path успевает до связывания имени только у тех потребителей,
    которые импортируются ПОЗЖЕ. Если по какой-то причине (иной порядок импорта, повторная
    установка, тест) потребитель уже импортирован -- его globals надо переписать руками, иначе
    подмена "встанет" и не сделает НИЧЕГО. Именно так и выглядит правка, прошедшая ревью и не
    работающая в бою.
    """
    n = 0
    for name in _CONSUMERS:
        m = sys.modules.get(name)
        if m is None:
            continue
        cur = getattr(m, "chunk_gated_delta_rule", None)
        if cur is not None and not getattr(cur, "__fa2sm70__", False):
            setattr(m, "chunk_gated_delta_rule", repl)
            n += 1
    return n


class _Finder:
    """Оборачивает загрузчик ЦЕЛЕВОГО пакета и патчит его сразу после исполнения."""

    def find_module(self, fullname, path=None):      # для старых путей импорта
        return None

    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET:
            return None
        # Снимаем себя, ищем настоящий spec, возвращаем его с обёрнутым exec_module.
        sys.meta_path.remove(self)
        try:
            import importlib.util
            spec = importlib.util.find_spec(fullname)
        finally:
            if self not in sys.meta_path:
                sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        loader = spec.loader
        orig_exec = loader.exec_module

        def exec_module(mod, _orig=orig_exec):
            _orig(mod)
            try:
                _patch_package(mod)
            except Exception as e:                   # noqa: BLE001
                _log(f"НЕ УДАЛОСЬ подменить: {e}")

        loader.exec_module = exec_module
        return spec


def install() -> bool:
    # [КУРС «БЕЗ ПРОКЛАДОК», 19.08.2026] Скан и рекуррент живут НАТИВНЫМИ ветками в самих
    # vllm-модулях (fla/ops/chunk.py и fla/ops/fused_sigmoid_gating.py, флаги FA2SM70_GDN и
    # FA2SM70_GDN_REC) -- шим-подмена поверх импортов больше не боевой путь. Оставлен как
    # инструмент A/B и дампов: включается ЯВНО FA2SM70_GDN_SHIM=1.
    if os.environ.get("FA2SM70_GDN_SHIM", "0") != "1":
        return False
    if os.environ.get(_ENV, "1") == "0":
        _log("FA2SM70_GDN=0 -> хук НЕ ставится")
        return False
    if any(isinstance(f, _Finder) for f in sys.meta_path):
        return True
    # Уже импортирован? Тогда патчим прямо (это путь для тестов; в бою хук встаёт раньше).
    if _TARGET in sys.modules:
        _patch_package(sys.modules[_TARGET])
        return True
    sys.meta_path.insert(0, _Finder())
    return True


def repatch() -> int:
    """Догнать уже импортированных потребителей. Вызывается из flash_attn_v100 -- ВТОРЫМ рубежом
    на случай, если sitecustomize по какой-то причине не отработал (иной запуск, урезанный
    site, чужой PYTHONPATH). Дважды подменить нельзя: признак __fa2sm70__ это ловит."""
    if os.environ.get(_ENV, "1") == "0":
        return 0
    pkg = sys.modules.get(_TARGET)
    if pkg is None:
        return 0
    cur = getattr(pkg, "chunk_gated_delta_rule", None)
    if cur is None:
        return 0
    if not getattr(cur, "__fa2sm70__", False):
        _patch_package(pkg)
        return 1
    return _rebind_consumers(cur)


def stats() -> dict:
    """Читается ИЗНУТРИ воркера (через torch-op или отладочный вызов) -- поведенческий признак."""
    return dict(STATS)
