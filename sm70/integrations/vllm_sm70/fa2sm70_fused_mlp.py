"""СЛИЯНИЕ ХВОСТА СЛОЯ: норма + gate_up + silu*mul ОДНИМ пуском вместо трёх тел Triton.

ЧТО ЭТО СНИМАЕТ. Опись установившегося декода (16.08, лента, Qwen3.8-27B, TP=2) назвала Triton
третьим по величине чужим владением -- 7.3 % шага, восемь тел порождения inductor. Два из них
лежат ровно на этом стыке и снимаются целиком:

    triton_red_fused__to_copy_add_fa2sm70_w8_mm_mean_mul_pow_rsqrt_3   44.0 мс   47 пусков/шаг
    triton_poi_fused_fa2sm70_w8_mm_mul_silu_slice_4                    28.8 мс   47 пусков/шаг

плюс исчезает отдельный пуск умножения gate_up. Итого 72.8 мс = 1.4 % шага и два пуска на слой.

ПОЧЕМУ ПОДМЕНА ЯДРА НЕ РАБОТАЕТ, А ПАТЧ СЛОЯ -- РАБОТАЕТ. Inductor уже слил норму, наше умножение
и вентиль в ОДИН граф: заменить внутри него нечего. Убрать Triton можно только сделав пару
НЕПРОЗРАЧНОЙ -- зарегистрировать свою операцию и звать её из модели. Ровно так уже сделано для
`fa2sm70_w8_mm`.

ТОЖДЕСТВО, КОТОРОЕ ЗДЕСЬ ИСПОЛЬЗУЕТСЯ. `RMSNorm.forward(x, residual)` у vLLM делает
`residual = residual + x; out = rmsnorm(residual)` и возвращает пару. Значит хвост слоя
    hidden, residual = post_attention_layernorm(hidden, residual)
    hidden = mlp(hidden)                     # gate_up -> silu*mul -> down
тождественно равен
    residual = residual + hidden
    hidden = down_proj( rmsnorm_gateup_silu(residual, w_norm, eps, gate.., up..) )
Сложение невязки остаётся снаружи: оно стоит 5120 элементов, то есть ничего.

РАЗРЕЗ ВЕСА gate_up. `MergedColumnParallelLinear` кладёт [gate_local, up_local] подряд по выходной
оси. Порядок ПОДТВЕРЖДЁН машинным кодом inductor, а не документацией: снятое тело читает
`in_ptr0 + x0 + 17408*x1` и `in_ptr0 + 8704 + x0 + 17408*x1`, то есть gate со смещением 0, up --
со смещением I. Разрез делается ОДИН РАЗ при загрузке.

ГЕЙТ ПЕРЕД ПРИВЯЗКОЙ БЫЛ ОБЯЗАТЕЛЕН И СРАБОТАЛ: тело `fused_norm_gemv.cu` числилось «написано и
готово» полгода и НИ РАЗУ не исполнялось -- первый запуск дал relL2 3.0..39 (ядро игнорировало
разрез по k, который пускач ему передавал). Починено, приёмка `tests/test_fused_norm_gemv.py`.

РЫЧАГ `FA2SM70_FMLP=1`, по умолчанию ВЫКЛЮЧЕНО. Откат -- снять переменную.
"""
from __future__ import annotations

import hashlib
import inspect
import os
import sys

import torch

_ПУТЬ = os.environ.get("FA2SM70_PATH", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
if _ПУТЬ not in sys.path:
    sys.path.insert(0, _ПУТЬ)

ВКЛ = os.environ.get("FA2SM70_FMLP", "0") == "1"
ГРУППА = int(os.environ.get("FA2SM70_FMLP_G", "128"))
# Порог запасной ветви -- ТОТ ЖЕ, что у отгруженного пути W8 (16). Первая редакция ставила сюда
# 1<<30, то есть гнала ПРЕФИЛЛ через наше скалярное ядро: оно там и медленнее, и сводит
# k-расщепление атомарными, отчего NLL перестал быть устойчивым (2.3267 против 2.3282 на двух
# прогонах, тогда как до правки он повторялся до шестого знака). Прибор сломала сама правка.
ММАКС = int(os.environ.get("FA2SM70_FMLP_MMAX", "16"))
_расш = [None]
_счёт = {"наше": 0, "чужое": 0}

# Отпечаток ЧУЖОГО forward, поверх которого писан патч. Если движок обновят и тело изменится,
# патч обязан ОТКАЗАТЬСЯ, а не молча исполнять устаревшую копию: тихая неверность здесь страшнее
# отказа, потому что она уедет в вывод.
_ОТПЕЧАТОК = "post_attention_layernorm(hidden_states, residual)"


def _ext():
    if _расш[0] is None:
        import fa2_sm70
        _расш[0] = fa2_sm70._ext.fused_norm_gemv_ext()
    return _расш[0]


def _разрезать(слой, G):
    """gate_up [K/4, 2I] -> два ПЛОТНЫХ среза. Делается один раз, при первом обращении к слою."""
    qw, qz, sc = слой.qweight, слой.qzeros, слой.scales
    двa_i = qw.shape[1]
    if двa_i % 8:                       # I обязано быть кратно 4: qzeros пакует 4 столбца в слово
        return None
    i = двa_i // 2
    if i % 4:
        return None
    части = (qw[:, :i].contiguous(), qz[:, : i // 4].contiguous(), sc[:, :i].contiguous(),
             qw[:, i:].contiguous(), qz[:, i // 4:].contiguous(), sc[:, i:].contiguous())
    # [ИНАЧЕ ПАМЯТЬ УДВАИВАЕТСЯ И ПОДЪЁМ ПАДАЕТ ПО OOM -- второй отказ этого патча]
    # Срез через `.contiguous()` -- это КОПИЯ. Оставив исходный gate_up рядом, мы держим вес
    # ДВАЖДЫ: 89 МБ на слой x 47 слоёв = 4.2 ГБ, и подъём падает на 44 МиБ. Половинки покрывают
    # исходный ЦЕЛИКОМ и он больше не читается никем (запасной путь при M>1 берёт их же),
    # поэтому оригинал освобождается здесь же -- суммарный объём не меняется.
    слой.qweight.data = torch.empty(0, dtype=qw.dtype, device=qw.device)
    слой.qzeros.data = torch.empty(0, dtype=qz.dtype, device=qz.device)
    слой.scales.data = torch.empty(0, dtype=sc.dtype, device=sc.device)
    return части


def _годен(слой) -> bool:
    """Только 8-разрядный GPTQ без перестановки и с нашей группой. Условия СТАТИЧЕСКИЕ."""
    for имя in ("qweight", "qzeros", "scales"):
        if not hasattr(слой, имя):
            return False
    g = getattr(слой, "g_idx", None)
    if g is not None and g.numel() != 0:
        return False
    кв = getattr(слой, "quant_method", None)
    биты = getattr(getattr(кв, "quant_config", None), "weight_bits", None)
    if биты is not None and int(биты) != 8:
        return False
    return True


def установить() -> bool:
    if not ВКЛ:
        return False
    from vllm.model_executor.models import qwen3_next as qn
    from vllm.utils.torch_utils import direct_register_custom_op

    Л = qn.Qwen3NextDecoderLayer
    if getattr(Л, "_fa2sm70_fmlp", False):
        return True

    # [РАЗРЕЗ ВЕСА -- В ЗАГРУЗКУ, А НЕ В forward. ТРЕТИЙ ОТКАЗ ПОДЪЁМА БЫЛ ИМЕННО ЗДЕСЬ]
    # Первая редакция резала вес ЛЕНИВО, при первом проходе. Но forward трассируется Dynamo, а
    # разрез ещё и МУТИРУЕТ модуль (освобождает оригинал) -- то есть состояние менялось под
    # трассировкой, повторный проход видел уже пустой тензор, и падало ВНУТРИ графа
    # (`IndexError: too many indices for tensor of dimension 1`, `<eval_with_key>`).
    # Это тот же класс, что уже стоил захода на W8: питоновское состояние в теле, которое движок
    # компилирует. Единственное законное окно -- `process_weights_after_loading`: веса уже на
    # карте, снимок памяти сделан, компиляции ещё нет.
    try:
        from vllm.model_executor.layers.quantization.gptq import GPTQLinearMethod
    except Exception as e:  # noqa: BLE001
        print(f"[fa2_sm70 FMLP] нет GPTQLinearMethod ({str(e)[:50]}) -- патч не встал")
        return False

    _чужая_подготовка = GPTQLinearMethod.process_weights_after_loading

    def подготовить(self, layer):
        _чужая_подготовка(self, layer)
        имя = getattr(layer, "prefix", "")
        if not имя or имя.rsplit(".", 1)[-1] != "gate_up_proj":
            return
        if not _годен(layer):
            return
        части = _разрезать(layer, ГРУППА)
        if части is not None:
            layer._fa2_gu = части
            print(f"[fa2_sm70 FMLP] {имя}: вес разрезан на gate/up при загрузке", flush=True)

    GPTQLinearMethod.process_weights_after_loading = подготовить

    исходник = inspect.getsource(Л.forward)
    if _ОТПЕЧАТОК not in исходник:
        print("[fa2_sm70 FMLP] ОТКАЗ: чужой forward изменился, патч писан не под него "
              "(ищем строку post_attention_layernorm(hidden_states, residual))")
        return False
    отп = hashlib.md5(исходник.encode()).hexdigest()[:10]

    # ------------------------------------------------------------------ НАША ОПЕРАЦИЯ
    # Ветвь ПО ФОРМЕ стоит ВНУТРИ операции: движок компилирует ОДИН граф на диапазон (1, 16384), и
    # питоновский `if` снаружи разрешился бы на трассировке по ПРЕФИЛЛЬНОЙ форме -- чужая ветвь
    # запеклась бы навсегда. Это уже стоило нам одного захода.
    def _fmlp(x: torch.Tensor, w: torch.Tensor, eps: float,
              qwg: torch.Tensor, qzg: torch.Tensor, scg: torch.Tensor,
              qwu: torch.Tensor, qzu: torch.Tensor, scu: torch.Tensor,
              group: int, v2: bool) -> torch.Tensor:
        x2 = x.reshape(-1, x.shape[-1])
        if x2.shape[0] != 1:                       # скелет слитого тела собран под декодный токен
            # w здесь СЫРОЙ вес нормы: (1+w) сворачиваем и в запасном пути тоже.
            xf = x2.float()
            n = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
                 * (1.0 + w.float())).to(scg.dtype)
            g = torch.ops.vllm.fa2sm70_w8_mm(n, qwg, qzg, scg,
                                             torch.empty(0, dtype=torch.int32, device=x.device),
                                             group, v2, ММАКС)
            u = torch.ops.vllm.fa2sm70_w8_mm(n, qwu, qzu, scu,
                                             torch.empty(0, dtype=torch.int32, device=x.device),
                                             group, v2, ММАКС)
            _счёт["чужое"] += 1
            return (torch.nn.functional.silu(g) * u).to(scg.dtype).reshape(*x.shape[:-1], -1)
        y = _ext().rmsnorm_gateup_silu(x2.contiguous(), w, eps, True,
                                       qwg, qzg, scg, qwu, qzu, scu, group, v2)
        _счёт["наше"] += 1
        if _счёт["наше"] == 1 or (_счёт["наше"] & (_счёт["наше"] - 1)) == 0:
            print(f"[fa2_sm70 FMLP] СЛИТЫЙ хвост слоя сработал, всего {_счёт['наше']}", flush=True)
        return y.reshape(*x.shape[:-1], -1)

    def _fmlp_fake(x, w, eps, qwg, qzg, scg, qwu, qzu, scu, group, v2):
        # ТИП ВЫХОДА -- ПО ВЕСУ, А НЕ ПО ВХОДУ. Вход здесь fp32 (невязка Gemma), а тело возвращает
        # fp16. Ложная реализация задаёт форму И ТИП буфера, который движок выделит под графом:
        # соврав тип, она даёт `Expected out tensor to have dtype Half, but got float` уже на
        # захвате -- то есть падение, а не тихую порчу. Это первый отказ подъёма этого патча.
        return torch.empty((*x.shape[:-1], scg.shape[1]), dtype=scg.dtype, device=x.device)

    try:
        direct_register_custom_op(op_name="fa2sm70_fused_mlp", op_func=_fmlp, mutates_args=[],
                                  fake_impl=_fmlp_fake)
    except Exception as e:  # noqa: BLE001
        print(f"[fa2_sm70 FMLP] операция не зарегистрирована ({str(e)[:80]}) -- патч не встал")
        return False

    старый = Л.forward

    def forward(self, hidden_states, residual=None, positions=None, **kwargs):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        self_attention_output = torch.empty_like(hidden_states)
        if self.layer_type == "linear_attention":
            self.linear_attn(hidden_states=hidden_states, output=self_attention_output)
        elif self.layer_type == "full_attention":
            self.self_attn(hidden_states=hidden_states, output=self_attention_output,
                           positions=positions)
        else:
            raise ValueError("Invalid layer_type")
        hidden_states = self_attention_output

        if self.layer_scale:
            hidden_states = (hidden_states * self._attn_scale_p1 if len(hidden_states.shape) == 2
                             else hidden_states * self._attn_scale_p1.unsqueeze(0))

        # ------- ХВОСТ: норма + gate_up + silu слиты в ОДНО тело -------
        # forward НИЧЕГО не заводит и не мутирует: и разрез веса, и свёртка (1+w) сделаны
        # при загрузке. Здесь только чтение готового -- иначе Dynamo увидит мутацию модуля.
        млп = getattr(self, "mlp", None)
        слой = getattr(млп, "gate_up_proj", None)
        части = getattr(слой, "_fa2_gu", None) if слой is not None else None
        if части is not None and getattr(млп, "is_sequence_parallel", False):
            части = None

        if части is None:
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
            hidden_states = self.mlp(hidden_states)
        else:
            # ТОЖДЕСТВО GemmaRMSNorm-с-невязкой, воспроизведённое ДОСЛОВНО (см. её
            # `_forward_static_with_residual`): при fp16 сумма и новая невязка идут в fp32.
            residual = (hidden_states.float() + residual.float()
                        if hidden_states.dtype == torch.float16
                        else hidden_states + residual)
            # Вес нормы идёт СЫРЫМ: (1+w) у Gemma сворачивается ВНУТРИ ядра, поэтому здесь нет
            # ни лишнего тела, ни мутации модуля под трассировкой.
            h = torch.ops.vllm.fa2sm70_fused_mlp(
                residual, self.post_attention_layernorm.weight,
                float(self.post_attention_layernorm.variance_epsilon),
                *части, ГРУППА, False)
            hidden_states = self.mlp.down_proj(h)[0]

        if self.layer_scale:
            hidden_states = (hidden_states * self._ffn_scale_p1 if len(hidden_states.shape) == 2
                             else hidden_states * self._ffn_scale_p1.unsqueeze(0))
        return hidden_states, residual

    Л.forward = forward
    Л._fa2sm70_fmlp = True
    Л._fa2sm70_fmlp_src = отп
    print(f"[fa2_sm70 FMLP] хвост слоя слит: норма + gate_up + silu одним пуском "
          f"(отпечаток чужого forward {отп})")
    return True
