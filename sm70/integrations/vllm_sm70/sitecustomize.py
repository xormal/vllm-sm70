# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# ТОЧКА, ГДЕ ВСТАЁТ ПОДМЕНА СКАНА GDN. Этот файл лежит в том же каталоге, что и пакет
# flash_attn_v100, и подъёмник кладёт этот каталог первым в PYTHONPATH. Значит:
#   * site.py импортирует sitecustomize из ЛЮБОГО каталога sys.path -> хук встаёт автоматически;
#   * PYTHONPATH наследуется spawn-воркерами vLLM -> хук встаёт и в них, а это принципиально:
#     скан считается ВНУТРИ воркера, и подмена только в процессе API-сервера не сделала бы ничего.
#
# ПОЧЕМУ НЕ ИЗ flash_attn_v100/__init__.py. Тот модуль импортируется ЛЕНИВО, когда vLLM выбирает
# бэкенд внимания -- то есть УЖЕ ПОСЛЕ того, как построена модель и qwen3_next связал у себя имя
# chunk_gated_delta_rule. Подмена оттуда опоздала бы ровно на один импорт и молча ничего не дала.
# Это и есть тот класс ошибки, ради которого гейт врезки обязан быть поведенческим.
#
# ОСТОРОЖНО И ТИХО. Ошибка здесь роняет ЛЮБОЙ python-процесс с этим PYTHONPATH, включая утилиты,
# не имеющие отношения к vLLM. Поэтому всё завёрнуто в широкий except и по умолчанию молчит.
import os

# ---------------------------------------------------------------------------------------------
# ASCII-ВОРОТА К КИРИЛЛИЧЕСКИМ РЫЧАГАМ. bash ОТВЕРГАЕТ имена переменных не из [A-Za-z0-9_]:
#     export FA2SM70_БЕЗ_ОБМЕНА=1  ->  "not a valid identifier", И КОД ВОЗВРАТА 0.
# То есть пускач молча продолжает, рычаг не включается, фаза не снимается -- а замер выглядит
# состоявшимся и говорит "эта фаза стоит ноль". Прибор, врущий В ПОЛЬЗУ ВЫВОДА. В дереве 29 таких
# рычагов (все фальсификаторы снятием фазы), и до этой врезки НИ ОДИН не включался из оболочки.
# Теперь есть один ASCII-вход, через который проходит любое имя:
#     FA2SM70_GATES='FA2SM70_БЕЗ_ОБМЕНА=1;FA2SM70_ТОЛЬКО_MLP=1'
# (os.environ в Python принимает произвольные строки, ограничение чисто башевское).
_ворота = os.environ.get("FA2SM70_GATES", "")
if _ворота:
    import sys as _sys
    for _пара in _ворота.split(";"):
        _пара = _пара.strip()
        if not _пара or "=" not in _пара:
            continue
        _и, _з = _пара.split("=", 1)
        os.environ[_и.strip()] = _з.strip()
        print(f"[fa2_sm70] рычаг через ASCII-ворота: {_и.strip()}={_з.strip()}",
              file=_sys.stderr, flush=True)


if os.environ.get("FA2SM70_GDN", "1") != "0":
    try:
        import fa2_gdn_shim

        fa2_gdn_shim.install()
    except Exception:  # noqa: BLE001
        if os.environ.get("FA2SM70_GDN_LOG", "0") == "1":
            import traceback

            traceback.print_exc()

# Тихий лог: клиентские ошибки валидации (длина промпта > контекста) -- одной строкой WARNING
# вместо 25-строчного ERROR-трейсбека. Ставится ЗДЕСЬ, потому что обёртка logging.config.dictConfig
# обязана появиться ДО импорта vllm.logger, который этот dictConfig и зовёт. Подробности и границы
# (что именно глушится, а что нет) -- в fa2sm70_logquiet.py.
if os.environ.get("FA2SM70_QUIET_CLIENT_ERR", "1") != "0":
    try:
        import fa2sm70_logquiet

        fa2sm70_logquiet.install()
    except Exception:  # noqa: BLE001
        pass

# Ловушка NaN (FA2SM70_NAN_TRAP=1): показывает ПЕРВЫЙ модуль, чей выход потерял конечность, и
# конечен ли при этом его вход. Нужна потому, что "забор из восклицаний" -- симптом на выходе, а
# источников у него за одну сессию нашлось три разных. По умолчанию выключена.
if os.environ.get("FA2SM70_NAN_TRAP", "0") == "1":
    try:
        import fa2sm70_nantrap

        fa2sm70_nantrap.install()
    except Exception:  # noqa: BLE001
        pass

# Сторож логитов (FA2SM70_LOGIT_WATCH=1): то же разделение "тело или голова", но БЕЗ хуков на
# модулях, то есть С ГРАФАМИ -- в том самом режиме, в котором сервер работает в бою. Ловушка выше
# требует --enforce-eager и потому меряет другой режим. По умолчанию выключен.
if os.environ.get("FA2SM70_LOGIT_WATCH", "0") == "1":
    try:
        import fa2sm70_logitwatch

        fa2sm70_logitwatch.install()
    except Exception:  # noqa: BLE001
        pass


# ФАЛЬСИФИКАТОР СНЯТИЕМ ФАЗЫ (FA2SM70_БЕЗ_ОБМЕНА=1): обмен TP -> тождество. Ответы НЕВЕРНЫ по
# построению; включать только на тестовом порту, чтобы измерить ДОЛЮ обмена во времени шага.
if os.environ.get("FA2SM70_БЕЗ_ОБМЕНА", "0") == "1":
    try:
        import fa2sm70_noexch  # noqa: F401
    except Exception:
        import traceback
        traceback.print_exc()


# ИЗМЕРИТЕЛЬ ОБМЕНА (FA2SM70_ЗАМЕР_ОБМЕНА=1): делит цену свёртки на ПЕРЕКОС рангов и ПЕРЕВОЗКУ.
if os.environ.get("FA2SM70_ЗАМЕР_ОБМЕНА", "0") == "1":
    try:
        import fa2sm70_exchmeter  # noqa: F401
    except Exception:
        import traceback
        traceback.print_exc()


# ОБМЕН TP В 12 РАЗРЯДАХ (FA2SM70_ОБМЕН_I12=1): везём меньше байт по узкому мосту.
if os.environ.get("FA2SM70_EXCH12", os.environ.get("FA2SM70_ОБМЕН_I12", "0")) == "1":
    try:
        import fa2sm70_exch_i12  # noqa: F401
    except Exception:
        import traceback
        traceback.print_exc()


# ВЕСА ВНИМАНИЯ В 12 РАЗРЯДАХ (FA2SM70_W12=1). Единственные веса сети, оставшиеся в BF16 -- слепок
# исключает self_attn.{q,k,v}_proj из GPTQ намеренно. int12 представляет их ВТРОЕ ТОЧНЕЕ, чем BF16,
# при 0.75 байт: размена нет. См. шапку fa2sm70_w12.py.
if os.environ.get("FA2SM70_W12", "0") == "1":
    try:
        import fa2sm70_w12  # noqa: F401
    except Exception:
        import traceback
        traceback.print_exc()


# УМНОЖЕНИЕ НА GPTQ-ВЕСА НАШИМ ЯДРОМ (FA2SM70_W8=1). Чужое тело держит 50.6 % шага декода, и пока
# оно чужое, путь нельзя ни слить, ни разложить по фазам. Наше при верной условности нуля ТОЧНЕЕ
# чужого в 5.7-9 раз при паритете скорости (замер: benchmarks/bench_svoy_put.py). По умолчанию
# ВЫКЛЮЧЕНО -- до сквозного гейта по NLL на боевом сервере. См. шапку fa2sm70_w8.py.
# СВОЙ ALL-REDUCE ВМЕСТЕ С ПОЛНЫМ ГРАФОМ (FA2SM70_CUSTOM_AR=1). Обмен NCCL -- 6.65 мс из 45.2 мс
# шага (15 %), и это чистая латентность. Прежний запрет («либо граф, либо свой обмен») снят: у
# vLLM есть путь registered=False, которому регистрация графовых буферов не нужна вовсе.
if os.environ.get("FA2SM70_CUSTOM_AR", "0") == "1":
    try:
        import fa2sm70_custom_ar
        fa2sm70_custom_ar.установить()

        # СВОЁ СВЕДЕНИЕ TP вместо NCCL на мелких тензорах (задача 201). Приз замерен:
        # 26.7 мкс против 56.7-91.7 у NCCL при железном поле 24.0 (цена прямой копии),
        # то есть 3.8-8.3 мс на шаг декода. Графобезопасно: счётчик в памяти устройства.
        import fa2sm70_ar2
        fa2sm70_ar2.установить()
    except Exception as _ex:  # noqa: BLE001
        print(f"[fa2_sm70] свой all-reduce НЕ встал: {_ex}", file=_sys.stderr, flush=True)

# ЧЕРНОВИК DFlash2 НА ТЕНЗОРНОМ ТЕЛЕ (FA2SM70_DRAFT_TM8=1). Тело черновика 4.9 мс из 45.2 мс шага
# при поле чтения 2.05 -- плотный GEMM при M=k+1 связан занятостью, а не полосой. См. шапку
# fa2sm70_draft_tm8.py. ВЫКЛЮЧЕНО по умолчанию: гейт -- ПРИЁМКА, а не время.
if os.environ.get("FA2SM70_DRAFT_TM8", "0") == "1":
    try:
        import fa2sm70_draft_tm8
        fa2sm70_draft_tm8.установить()
    except Exception as _ex:  # noqa: BLE001
        print(f"[fa2_sm70] черновик на tm8 НЕ встал: {_ex}", file=_sys.stderr, flush=True)

if os.environ.get("FA2SM70_W8", "0") == "1":
    try:
        import fa2sm70_w8
        fa2sm70_w8.установить()
    except Exception:
        import traceback
        traceback.print_exc()

# ТЕЛО ГОЛОВЫ MTP В int8 (FA2SM70_MTP_W8=1). Голова MTP лежит в слепке НЕ квантованной, а
# черновик проходит её трижды за шаг: 3.94 мс из 10.59 мс прохода (записка 25 §249). Замер на
# настоящих весах -- x2.74 при M=1 и 202 МиБ освобождённой памяти на ранг (§251). Требует
# FA2SM70_W8=1: пользуется его зарегистрированной операцией. По умолчанию ВЫКЛЮЧЕНО.
if os.environ.get("FA2SM70_MTP_W8", "0") == "1":
    try:
        import fa2sm70_mtp_w8
        fa2sm70_mtp_w8.установить()
    except Exception:
        import traceback
        traceback.print_exc()


# РАССЫЛКА-СО-СВЕДЕНИЕМ, ПЕРЕКРЫТАЯ С УМНОЖЕНИЕМ (FA2SM70_AR2=1). Заменяет пару «умножить всё,
# затем свести всё» в RowParallelLinear: чужая половина строк считается первой и уезжает соседу
# кусками, своя половина служит ей укрытием. Потолок известен замером -- обмен ДЛИННЕЕ своего
# умножения (o_proj 15.02 против 3.01 мс), поэтому прячется 20-52 %; на боевом теле замерено
# x1.391 / x1.430 при разности РОВНО 0. По умолчанию ВЫКЛЮЧЕНО -- до сквозного гейта на стенде.
if os.environ.get("FA2SM70_AR2", "0") == "1":
    try:
        import fa2sm70_ar2
        fa2sm70_ar2.установить()
    except Exception:
        import traceback
        traceback.print_exc()


# ГЕЙТОВАННАЯ НОРМА GDN ОДНИМ ЯДРОМ (FA2SM70_GNORM=1): native-лесенка ~6 aten-пусков
# после нашего сшитого GDN-тела -> одно fla-ядро (x2.84 и точнее на боевой форме).
# Глобальное включение '+rms_norm_gated' через compilation-config ЗАПРЕЩЕНО опытом:
# dynamo падает на staticmethod.__call__ (18.08). См. шапку fa2sm70_gnorm.py.
if os.environ.get("FA2SM70_GNORM", "0") == "1":
    try:
        import fa2sm70_gnorm
        fa2sm70_gnorm.установить()
    except Exception:
        import traceback
        traceback.print_exc()


# СРОК ЖИЗНИ ЗАЯВКИ (FA2SM70_REQ_TTL, секунды; 0 = выключено). Вторая линия к потолку
# генерации: клиент может отвалиться по своему таймауту, НЕ закрыв сокет (наблюдение на
# боевом 20.08) -- тогда обрыва нет, отменять нечему, и сервер греет карты впустую.
try:
    import fa2sm70_ttl
    fa2sm70_ttl.установить()
except Exception:
    import traceback
    traceback.print_exc()


# ОТМЕНА ПО ОБРЫВУ СОЕДИНЕНИЯ: диагностика (FA2SM70_ABORT_LOG=1). Молчит по умолчанию;
# нужна, чтобы отличить «событие обрыва не доходит» от «доходит, но abort не срабатывает».
try:
    import fa2sm70_abort
    fa2sm70_abort.установить()
except Exception:
    import traceback
    traceback.print_exc()


# СЛОВАРНАЯ ПРОЕКЦИЯ НА НАШЕМ ЯДРЕ (FA2SM70_LMH=1 или 2). Это единственное крупное чужое тело,
# оставшееся в шаге декода: turbomind, 1463 мкс/токен = 4.79 % ядерного времени, и оно стоит на
# 98.9 % крыши полосы -- быстрее в fp16 не бывает, поэтому ход один: меньше байт (int8 вместо
# fp16, x1.881 замерено). Режим 2 добавляет точный доводчик верхушки списка.
# По умолчанию ВЫКЛЮЧЕНО -- до сквозного гейта по совпадению вывода. См. шапку fa2sm70_lm_head.py.
# СЛИЯНИЕ ХВОСТА СЛОЯ (FA2SM70_FMLP=1): норма + gate_up + silu одним пуском вместо трёх тел
# Triton. Снимает 72.8 мс = 1.4 % шага декода и два пуска на слой. По умолчанию ВЫКЛЮЧЕНО.
# См. шапку fa2sm70_fused_mlp.py: перед привязкой в слитом теле был найден и закрыт дефект.
if os.environ.get("FA2SM70_FMLP", "0") == "1":
    try:
        import fa2sm70_fused_mlp
        fa2sm70_fused_mlp.установить()
    except Exception:
        import traceback
        traceback.print_exc()


if os.environ.get("FA2SM70_LMH", "0") != "0":
    try:
        import fa2sm70_lm_head
        fa2sm70_lm_head.установить()
    except Exception:
        import traceback
        traceback.print_exc()


# ДУМАЮЩИЕ СЕТИ: ответ без блока размышления терялся ЦЕЛИКОМ (235 токенов -> пустая строка).
# FA2SM70_QWEN3_THINK=0 -- отключить. См. шапку fa2sm70_qwen3_think.py.
if os.environ.get("FA2SM70_QWEN3_THINK", "1") == "1":
    try:
        import fa2sm70_qwen3_think  # noqa: F401
    except Exception:
        import traceback
        traceback.print_exc()


# ФАЛЬСИФИКАТОР доли рекуррентного ядра GDN на декоде (FA2SM70_GDN_DEC_NULL=1).
# ОТВЕТЫ СТАНОВЯТСЯ НЕВЕРНЫМИ -- это прибор, а не режим.
if (os.environ.get("FA2SM70_GDN_DEC_NULL", "0") == "1"
        or os.environ.get("FA2SM70_GDN_PRE_NULL", "0") == "1"):
    try:
        import fa2sm70_gdn_dec_null  # noqa: F401
    except Exception:
        import traceback
        traceback.print_exc()

# [ШУМ В ЖУРНАЛЕ, 06.09] fla предупреждает «seq_len < num_heads -- возможно, head-first»
# (fla/ops/cumsum.py: chunk_local_cumsum). У нас это ЛОЖНАЯ тревога по построению: чанковый
# путь GDN зовётся на СЕГМЕНТАХ префилла (qwen3_next: `query_non_spec[:, a0:a1]`), и хвостовой
# сегмент бывает короче числа голов (13 против 24). Раскладка при этом верна -- [B, T, H, D],
# ось токенов вторая, ровно как требует head_first=False; подтверждение -- ответы модели
# (гейт 391, три иглы из трёх на 238K). Гасим ТОЧЕЧНО по тексту, чтобы журнал не уводил
# следующего читателя в погоню за призраком; чужой код при этом не трогаем.
import warnings as _w
_w.filterwarnings("ignore", message=r".*seq_len \(\d+\) < num_heads.*",
                  category=UserWarning)

# CLEF В БОЕВОЙ СЕТИ (FA2SM70_CLEF=1): эндпоинт /v1/systemone + поправка весов и голова классификатора
# по строкам запросов Clef (fa2sm70_clef.py, расширение fa2sm70_clef). Умолчание 0 -- путь не меняется.
if os.environ.get("FA2SM70_CLEF", "0") == "1":
    try:
        import fa2sm70_clef  # noqa: F401
    except Exception:
        import traceback
        traceback.print_exc()
