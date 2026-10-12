# ОТМЕНА ПО ОБРЫВУ СОЕДИНЕНИЯ: диагностика + страховка (задача 195).
#
# ЗАЧЕМ. Замер 20.08: клиента снимают на 8-й секунде запроса длиной 33 с -- сервер
# ДОСЧИТЫВАЕТ его до конца и держит карты. На 250K это 400 с занятой машины за никому не
# нужный ответ; именно так родилась «очередь брошенного запроса», из-за которой замер
# 250K читался как «зависание на 30 минут».
#
# ЧТО ЗДЕСЬ. Обёртка vllm.entrypoints.utils.listen_for_disconnect, которая ПЕЧАТАЕТ факт
# прихода http.disconnect (и время от начала ожидания). Это различитель между двумя
# причинами, которые лечатся по-разному:
#   * событие НЕ приходит  -> ломается доставка (ASGI/uvicorn), лечится своим сторожем;
#   * событие приходит     -> ломается путь abort до движка, лечится там.
# Диагностика включается FA2SM70_ABORT_LOG=1 и по умолчанию МОЛЧИТ.
import os
import sys
import time


def установить() -> None:
    if os.environ.get("FA2SM70_ABORT_LOG", "0") != "1":
        return
    try:
        from vllm.entrypoints import utils as _u
    except Exception:  # noqa: BLE001
        return
    исходная = getattr(_u, "listen_for_disconnect", None)
    if исходная is None or getattr(исходная, "_fa2sm70", False):
        return

    async def обёртка(request):
        t0 = time.time()
        print("[fa2_sm70 abort] жду обрыва соединения", file=sys.stderr, flush=True)
        try:
            await исходная(request)
            print(f"[fa2_sm70 abort] ОБРЫВ ПОЛУЧЕН через {time.time() - t0:.1f} с",
                  file=sys.stderr, flush=True)
        except BaseException as e:  # noqa: BLE001
            print(f"[fa2_sm70 abort] ожидание прервано: {type(e).__name__}",
                  file=sys.stderr, flush=True)
            raise

    обёртка._fa2sm70 = True
    _u.listen_for_disconnect = обёртка
    # маршруты импортировали имя НАПРЯМУЮ -- подменяем и у них
    for мод in ("vllm.entrypoints.openai.chat_completion.api_router",
                "vllm.entrypoints.openai.completion.api_router"):
        try:
            m = __import__(мод, fromlist=["*"])
            if getattr(m, "listen_for_disconnect", None) is исходная:
                m.listen_for_disconnect = обёртка
        except Exception:  # noqa: BLE001
            pass
    print("[fa2_sm70 abort] диагностика обрыва включена", file=sys.stderr, flush=True)
