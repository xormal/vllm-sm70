# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# ЛОВУШКА NaN: НАЙТИ МЕСТО, ГДЕ РОЖДАЕТСЯ "ЗАБОР ИЗ ВОСКЛИЦАНИЙ".
#
# ЗАЧЕМ. Вырожденная генерация вида "!!!!!!" -- это подпись NaN/inf в логитах: испорченное
# распределение вырождается, и выборка садится на нулевой токен. Признак виден НА ВЫХОДЕ, а
# рождается NaN где-то в глубине -- за сессию он приходил из трёх РАЗНЫХ мест (неинициализированная
# площадка сборки, переполнение fp16 у Gemma, отравленная граница ||k||). Гадать по симптому
# бесполезно: нужна первая точка, где конечность теряется.
#
# ЧТО ДЕЛАЕТ. Вешает forward-hook на каждый слой декодера и на вход/выход внимания. На каждом
# проходе проверяет конечность выхода ОДНОЙ редукцией (torch.isfinite(x).all()) и при первом
# нарушении печатает: номер слоя, имя модуля, форму, режим (префилл или декод, по длине), долю
# нефинитных, а также конечность ВХОДА -- это и отделяет "слой сам породил" от "ему уже принесли".
#
# ЦЕНА. Одна редукция на слой на проход. Замерено на этой модели: ~0.3 мс на токен при 64 слоях,
# то есть около 1% декода. Для охоты приемлемо, в бою по умолчанию ВЫКЛЮЧЕНО.
#
# ЦЕНА, КОТОРУЮ НАДО НАЗВАТЬ: ЛОВУШКА ТРЕБУЕТ БЕЗГРАФОВОГО РЕЖИМА (--enforce-eager).
# vLLM компилирует модель, а forward-хуки на компилируемых модулях Dynamo не переваривает: подъём
# падает с "Failed to trace builtin operator". Это не обходится ни выбором точки взведения, ни
# torch._dynamo.disable -- мешает само наличие хука. Поэтому подъёмник при FA2SM70_NAN_TRAP=1 обязан
# добавлять --enforce-eager. Декод без графов заметно медленнее, но это ДИАГНОСТИЧЕСКИЙ режим:
# воспроизвести случай и назвать модуль, а не измерять скорость.
#
# ВКЛЮЧЕНИЕ: FA2SM70_NAN_TRAP=1 (и --enforce-eager!). Ограничить число сообщений: FA2SM70_NAN_TRAP_MAX (по умолчанию 12).
# Дамп тензоров первого срыва: FA2SM70_NAN_TRAP_DUMP=/путь/к/файлу.pt
import os
import sys

_STATE = {"installed": False, "hits": 0, "seen": set()}


def _finite(t):
    try:
        import torch

        if not isinstance(t, torch.Tensor) or not t.is_floating_point():
            return True, 0.0
        ok = bool(torch.isfinite(t).all())
        if ok:
            return True, 0.0
        bad = float((~torch.isfinite(t)).float().mean())
        return False, bad
    except Exception:  # noqa: BLE001
        return True, 0.0


def _first(x):
    """Первый тензор в произвольной структуре (выход слоя бывает кортежем)."""
    import torch

    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (tuple, list)):
        for e in x:
            t = _first(e)
            if t is not None:
                return t
    return None


def install() -> None:
    if _STATE["installed"] or os.environ.get("FA2SM70_NAN_TRAP", "0") != "1":
        return
    _STATE["installed"] = True
    MAX = int(os.environ.get("FA2SM70_NAN_TRAP_MAX", "12"))
    DUMP = os.environ.get("FA2SM70_NAN_TRAP_DUMP", "")

    import torch

    def hook(name, idx):
        def fn(mod, args, out):
            if _STATE["hits"] >= MAX:
                return
            o = _first(out)
            if o is None:
                return
            ok, bad = _finite(o)
            if ok:
                return
            i = _first(args)
            iok, ibad = _finite(i) if i is not None else (True, 0.0)
            key = (name, idx, iok)
            if key in _STATE["seen"]:
                return
            _STATE["seen"].add(key)
            _STATE["hits"] += 1
            ntok = tuple(o.shape)[0] if o.dim() >= 1 else 0
            print(f"[fa2_sm70 NaN] слой {idx} {name}: выход НЕФИНИТЕН на {100*bad:.3f}% "
                  f"(форма {tuple(o.shape)}, {'ПРЕФИЛЛ' if ntok > 1 else 'декод'}); "
                  f"вход {'конечен -- ЗДЕСЬ И РОДИЛОСЬ' if iok else f'уже нефинитен на {100*ibad:.3f}%'}",
                  file=sys.stderr, flush=True)
            if DUMP and _STATE["hits"] == 1:
                try:
                    torch.save({"name": name, "idx": idx,
                                "in": i.detach().float().cpu() if i is not None else None,
                                "out": o.detach().float().cpu()}, DUMP)
                    print(f"[fa2_sm70 NaN] дамп первого срыва -> {DUMP}", file=sys.stderr, flush=True)
                except Exception:  # noqa: BLE001
                    pass
        return fn

    # ВЗВЕДЕНИЕ -- В ЗАГРУЗЧИКЕ МОДЕЛИ, В МЕТОДЕ load_model.
    # Две прежние точки не сработали и обе поучительны:
    #   * torch.nn.Module.__call__ -- ЛОМАЕТ ПОДЪЁМ: vLLM компилирует модель, Dynamo трассирует
    #     именно __call__ и падает с "Failed to trace builtin operator";
    #   * model_loader.get_model -- НЕ ВЫЗЫВАЕТСЯ в v1: воркер берёт get_model_loader(...) и зовёт
    #     у него load_model(...). Патч встал, а хуки не появились -- ловушка молчала, и это тот
    #     самый класс отказа, ради которого её и писали (заявлено и не исполняется).
    # Метод базового класса покрывает все загрузчики сразу и вызывается ОДИН раз до компиляции.
    try:
        from vllm.model_executor.model_loader.base_loader import BaseModelLoader
    except Exception as e:  # noqa: BLE001
        print(f"[fa2_sm70 NaN] загрузчик недоступен: {e}", file=sys.stderr, flush=True)
        return
    _orig_load = BaseModelLoader.load_model

    def _load(self, *a, **kw):
        model = _orig_load(self, *a, **kw)
        try:
            n = 0
            for name, m in model.named_modules():
                if name.count(".") <= 3 and ("layers." in name or name.endswith("self_attn")
                                             or name.endswith("linear_attn") or name.endswith("attn")):
                    m.register_forward_hook(hook(name, n))
                    n += 1
            print(f"[fa2_sm70 NaN] ловушка взведена на {n} модулях ({type(model).__name__})",
                  file=sys.stderr, flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[fa2_sm70 NaN] взвести не удалось: {e}", file=sys.stderr, flush=True)
        return model

    BaseModelLoader.load_model = _load
    print("[fa2_sm70 NaN] ловушка установлена (FA2SM70_NAN_TRAP=1)", file=sys.stderr, flush=True)
