# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# СТОРОЖ ЛОГИТОВ: ловит "забор из восклицаний" В БОЕВОМ РЕЖИМЕ, С ГРАФАМИ.
#
# ЗАЧЕМ ОТДЕЛЬНО ОТ fa2sm70_nantrap. Та ловушка вешает forward-хуки на 91 модуль и поэтому требует
# --enforce-eager: Dynamo не переваривает хуки на компилируемых модулях. Но режим без графов -- это
# ДРУГОЙ режим (иной путь gptq-свёртки C, иной планировщик), и отказ, который ищем, может в нём
# просто не воспроизвестись. Сторож нужен там, где сервер работает КАК В БОЮ.
#
# ПОЧЕМУ ЭТО БЕЗОПАСНО ПРИ ГРАФАХ. Обе точки врезки -- ОБЫЧНЫЕ МЕТОДЫ, вызываемые ВНЕ
# компилируемой области:
#   * model.compute_logits(hidden)  -- зовётся из gpu_model_runner уже ПОСЛЕ прохода модели;
#   * GPUModelRunner._get_nans_in_logits -- штатный счётчик vLLM (VLLM_COMPUTE_NANS_IN_LOGITS=1).
# Никаких register_forward_hook, поэтому трассировка не ломается.
#
# ЧТО ДАЁТ. На первом же срыве -- строка с РАЗДЕЛЕНИЕМ ПРИЧИНЫ:
#   вход compute_logits (скрытое состояние последнего слоя) конечен?  нет -> родилось В ТЕЛЕ модели;
#                                                                     да  -> родилось В ГОЛОВЕ (lm_head).
# Плюс форма (сколько позиций сразу) -- это отличает ПРЕФИЛЛ (много) от ДЕКОДА (по одной на заявку),
# и номер шага, чтобы понять "с первого токена" или "посреди генерации".
#
# ВКЛЮЧЕНИЕ: FA2SM70_LOGIT_WATCH=1. Цена -- две редукции isfinite на шаг (тензор [B, vocab]),
# единицы десятков микросекунд. По умолчанию ВЫКЛЮЧЕН.
import os
import sys

_STATE = {"installed": False, "hits": 0, "step": 0, "seen": set()}
_MAX = int(os.environ.get("FA2SM70_LOGIT_WATCH_MAX", "40"))


def _say(msg: str) -> None:
    print(f"[fa2_sm70 сторож] {msg}", file=sys.stderr, flush=True)


def install() -> None:
    if _STATE["installed"] or os.environ.get("FA2SM70_LOGIT_WATCH", "0") != "1":
        return
    _STATE["installed"] = True

    import torch

    # --- 1. ТЕЛО ПРОТИВ ГОЛОВЫ -------------------------------------------------------------
    # Оборачиваем compute_logits У ЭКЗЕМПЛЯРА МОДЕЛИ, в load_model -- то же место, где взводится
    # nantrap, и по той же причине: get_model в v1 не вызывается, а метод базового загрузчика
    # вызывается ровно один раз до компиляции.
    try:
        from vllm.model_executor.model_loader.base_loader import BaseModelLoader
    except Exception as e:  # noqa: BLE001
        _say(f"загрузчик недоступен: {e}")
        return

    _orig_load = BaseModelLoader.load_model

    def _load(self, *a, **kw):
        model = _orig_load(self, *a, **kw)
        try:
            _orig_cl = model.compute_logits

            def compute_logits(hidden_states, *args, **kwargs):
                out = _orig_cl(hidden_states, *args, **kwargs)
                try:
                    _STATE["step"] += 1
                    if _STATE["hits"] >= _MAX or out is None:
                        return out
                    if bool(torch.isfinite(out).all()):
                        return out
                    hs_ok = bool(torch.isfinite(hidden_states).all())
                    nrow = int(out.shape[0]) if out.dim() >= 1 else 0
                    bad = float((~torch.isfinite(out)).float().mean())
                    _STATE["hits"] += 1
                    _say(
                        f"ЛОГИТЫ НЕФИНИТНЫ шаг={_STATE['step']} строк={nrow} "
                        f"доля={100*bad:.3f}% форма={tuple(out.shape)}; скрытое состояние "
                        f"{'КОНЕЧНО -> родилось в ГОЛОВЕ (lm_head/gptq)' if hs_ok else 'УЖЕ НЕФИНИТНО -> родилось в ТЕЛЕ модели'}"
                        f" (hidden {tuple(hidden_states.shape)})")
                except Exception:  # noqa: BLE001
                    pass
                return out

            model.compute_logits = compute_logits
            _say(f"взведён на compute_logits ({type(model).__name__})")
        except Exception as e:  # noqa: BLE001
            _say(f"взвести на compute_logits не удалось: {e}")
        return model

    BaseModelLoader.load_model = _load

    # --- 2. ШТАТНЫЙ СЧЁТЧИК vLLM: ИМЯ ЗАЯВКИ -----------------------------------------------
    # compute_logits знает форму, но не знает, ЧЬЯ строка испортилась. Штатный счётчик знает.
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        _orig_nans = GPUModelRunner._get_nans_in_logits

        def _nans(self, logits):
            d = _orig_nans(self, logits)
            try:
                for rid, n in (d or {}).items():
                    if n > 0 and rid not in _STATE["seen"] and len(_STATE["seen"]) < _MAX:
                        _STATE["seen"].add(rid)
                        _say(f"NaN в логитах заявки {rid}: {n} позиций словаря")
            except Exception:  # noqa: BLE001
                pass
            return d

        GPUModelRunner._get_nans_in_logits = _nans
        _say("взведён на штатный счётчик (нужен VLLM_COMPUTE_NANS_IN_LOGITS=1)")
    except Exception as e:  # noqa: BLE001
        _say(f"штатный счётчик недоступен: {e}")
