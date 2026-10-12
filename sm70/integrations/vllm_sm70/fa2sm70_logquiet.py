# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# ТИХАЯ ОБРАБОТКА КЛИЕНТСКИХ ОШИБОК В ЛОГЕ vLLM.
#
# ЧТО ЛЕЧИМ. Когда заявка не проходит валидацию (самый частый случай -- длина промпта больше
# контекста), vllm/entrypoints/openai/completion/serving.py делает logger.exception(...) и лишь
# ПОТОМ возвращает корректный HTTP 400. То есть по протоколу всё правильно, а в лог при этом
# падает 25-строчный ERROR-трейсбек на ошибку, в которой сервер не виноват вообще.
#
# ПОЧЕМУ ЭТО НЕ КОСМЕТИКА. В этом проекте лог сервера -- это ДОКАЗАТЕЛЬНАЯ БАЗА: приёмка
# (verify_all.sh) грепает его на "Error|Traceback|not supported", чтобы отличить "наши ядра
# встали" от "молча откатились на штатный путь". Клиентские трейсбеки дают ложные красные и
# приучают глаз пролистывать ERROR -- ровно та привычка, из-за которой в этой сессии трижды
# принимали заявленное за исполненное. Лог обязан быть тихим, чтобы шум в нём что-то значил.
#
# ЧТО ДЕЛАЕМ. Гасим ТОЛЬКО трейсбек и ТОЛЬКО для ошибок валидации заявки (VLLMValidationError,
# он же подкласс ValueError). Само сообщение сохраняем целиком -- в нём уже есть все числа
# ("You passed 188058 input tokens ... context length is only 131072"). Уровень понижаем до
# WARNING: это не отказ сервера, это отказ заявке. Любая ДРУГАЯ ошибка проходит нетронутой
# вместе с трейсбеком -- глушить серверные сбои нельзя ни при каких условиях.
#
# КАК ВСТАЁТ. Не патчим vLLM и не подменяем VLLM_LOGGING_CONFIG_PATH (тот ЗАМЕНЯЕТ конфиг
# целиком -- пришлось бы копировать DEFAULT_LOGGING_CONFIG и он бы разъехался при обновлении).
# Вместо этого из sitecustomize оборачиваем logging.config.dictConfig: vLLM зовёт его при
# импорте vllm.logger, и сразу после его вызова мы навешиваем фильтр на ОБРАБОТЧИК логгера
# "vllm". Фильтр на обработчике видит все записи, дошедшие до него, включая записи дочерних
# логгеров -- в отличие от фильтра на логгере, который на проброшенные записи не срабатывает.
#
# ВЫКЛЮЧАТЕЛЬ: FA2SM70_QUIET_CLIENT_ERR=0.
import logging
import os

_STATE = {"wrapped": False, "quieted": 0}


def _is_client_error(exc: BaseException) -> bool:
    """Ошибка ЗАЯВКИ, а не сервера. Список намеренно узкий."""
    try:
        from vllm.exceptions import VLLMValidationError

        if isinstance(exc, VLLMValidationError):
            return True
    except Exception:  # noqa: BLE001 -- старая vLLM без этого класса
        pass
    # запасной признак по имени: класс мог переехать между версиями
    return type(exc).__name__ in ("VLLMValidationError", "RequestValidationError")


class QuietClientErrors(logging.Filter):
    """Схлопывает трейсбек клиентской ошибки в одну строку WARNING. Ничего не выбрасывает."""

    def filter(self, record: logging.LogRecord) -> bool:
        exc = record.exc_info[1] if record.exc_info else None
        if exc is None or not _is_client_error(exc):
            return True
        try:
            head = record.getMessage()
        except Exception:  # noqa: BLE001
            head = str(record.msg)
        record.msg = f"{head}: {type(exc).__name__}: {exc}"
        record.args = ()
        record.exc_info = None
        record.exc_text = None
        record.levelno = logging.WARNING
        record.levelname = "WARNING"
        _STATE["quieted"] += 1
        return True


def _attach(logger_name: str = "vllm") -> int:
    """Навесить фильтр на обработчики логгера. Идемпотентно."""
    n = 0
    lg = logging.getLogger(logger_name)
    for h in lg.handlers:
        if not any(isinstance(f, QuietClientErrors) for f in h.filters):
            h.addFilter(QuietClientErrors())
            n += 1
    return n


def install() -> None:
    if _STATE["wrapped"] or os.environ.get("FA2SM70_QUIET_CLIENT_ERR", "1") == "0":
        return
    import logging.config as _lc

    _orig = _lc.dictConfig

    def _patched(config, *a, **kw):
        r = _orig(config, *a, **kw)
        try:
            _attach("vllm")
        except Exception:  # noqa: BLE001
            pass
        return r

    _lc.dictConfig = _patched
    _STATE["wrapped"] = True
    # если логирование уже настроено (импорт после vllm.logger) -- навесить сразу
    try:
        _attach("vllm")
    except Exception:  # noqa: BLE001
        pass
