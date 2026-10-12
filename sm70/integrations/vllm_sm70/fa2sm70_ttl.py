# СРОК ЖИЗНИ ЗАЯВКИ (FA2SM70_REQ_TTL, секунды; 0 = выключено).
#
# ПОЧЕМУ ЭТОГО НЕ ХВАТАЛО. У vLLM отмена висит на РАЗРЫВЕ соединения (`with_cancellation`
# -> http.disconnect). Замер 20.08: при настоящем разрыве это работает -- карты свободны
# через 4 с. Но боевой случай ДРУГОЙ: клиент отваливается ПО СВОЕМУ ТАЙМАУТУ и сокет НЕ
# закрывает (наблюдение на боевом: соединение 127.0.0.1:58326 висит ESTABLISHED, running=1,
# карты на 90 %). Разрыва нет -- значит и отменять нечему; сервер честно молотит заявку,
# которую уже никто не ждёт.
#
# ЧТО ДЕЛАЕМ. Оборачиваем обработчик в asyncio.wait_for со СРОКОМ. По сроку задача
# отменяется -- дальше отрабатывает штатный путь vLLM (CancelledError -> abort запроса в
# движке), тот же самый, что и при разрыве. Ничего в движке не патчим.
#
# КАК ВЫБИРАТЬ СРОК. Он обязан быть БОЛЬШЕ самой длинной ЗАКОННОЙ заявки и МЕНЬШЕ
# клиентского таймаута, иначе либо режем своих, либо продолжаем греть карты впустую.
# Замер 20.08 на стенде: 250K = 406 с, 200K = 300 с. При клиентском таймауте 600 с
# разумный срок -- 540..660 с.
import asyncio
import os
import sys
import time

_СОСТ = {"поставлено": False, "снято": 0}


def _срок() -> float:
    try:
        return float(os.environ.get("FA2SM70_REQ_TTL", "0"))
    except ValueError:
        return 0.0


def установить() -> None:
    срок = _срок()
    if срок <= 0 or _СОСТ["поставлено"]:
        return
    поставлено = []
    for имя in ("vllm.entrypoints.openai.chat_completion.api_router",
                "vllm.entrypoints.openai.completion.api_router"):
        try:
            мод = __import__(имя, fromlist=["*"])
        except Exception:  # noqa: BLE001 -- маршрута может не быть в сборке
            continue
        for точка in ("create_chat_completion", "create_completion"):
            исходная = getattr(мод, точка, None)
            if исходная is None or getattr(исходная, "_fa2sm70_ttl", False):
                continue

            def обёртка(*a, __и=исходная, __т=точка, **kw):
                async def _run():
                    t0 = time.time()
                    try:
                        return await asyncio.wait_for(__и(*a, **kw), timeout=_срок())
                    except asyncio.TimeoutError:
                        _СОСТ["снято"] += 1
                        print(f"[fa2_sm70 ttl] заявка снята по сроку {_срок():.0f} с "
                              f"({__т}, прошло {time.time() - t0:.0f} с); всего снято "
                              f"{_СОСТ['снято']}", file=sys.stderr, flush=True)
                        raise
                return _run()

            обёртка._fa2sm70_ttl = True
            # FastAPI уже держит ССЫЛКУ на исходную функцию в маршруте -- меняем ЕЁ,
            # а не только имя в модуле, иначе подмена не участвует (урок «написано
            # != работает»).
            setattr(мод, точка, обёртка)
            поставлено.append(f"{имя.rsplit('.', 2)[-2]}.{точка}")
    _СОСТ["поставлено"] = True
    print(f"[fa2_sm70 ttl] срок жизни заявки {срок:.0f} с; обёрнуто: "
          f"{', '.join(поставлено) or 'НИЧЕГО (маршруты не найдены)'}",
          file=sys.stderr, flush=True)
