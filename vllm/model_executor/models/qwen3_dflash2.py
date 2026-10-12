# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# ==================================================================================================
# [FA2/SM70 24.08] DFlash2 -- ПОЛНАЯ ВТОРАЯ ВЕРСИЯ, а не чекпойнт, сведённый к первой.
# ==================================================================================================
# ЧЕМ V2 ОТЛИЧАЕТСЯ ОТ V1 (и почему сведение к V1 -- потеря, а не упрощение):
#
#   1. ДИНАМИЧЕСКАЯ ГРУППОВАЯ СВЁРТКА вокруг внимания и MLP. Ядро не лежит в весах -- оно
#      ПРЕДСКАЗЫВАЕТСЯ из самого входа (`kernel_projection`), а применяется двумя тактами внутри
#      блока. Это даёт позициям блока связь ПО ВРЕМЕНИ, которой у параллельной выдачи нет.
#      Блок свёртки = 1 + k (бонусный токен и маски), а НЕ block_size из конфига: свёртка живёт в
#      координатах запроса, а не в координатах обучения.
#
#   2. СЕЛЕКТОР КАНДИДАТОВ. Позиции блока считаются параллельно и о выборе соседа не знают;
#      поточечный argmax склеивает k несвязанных догадок. Селектор берёт top_k кандидатов на
#      позицию и выбирает СВЯЗНЫЙ путь: ребро (предшественник p -> преемник c) оценивается
#      низкоранговой формой из двух кодбуков ранга 256, а не таблицей V*V (это 248320^2 -- нереально):
#
#          ребро[l,p,c] = унарная[l,c] + < кодбук_пред[id_p] * (W_h h_l), кодбук_прее[id_c] >
#
#      Путь ищется ТОЧНО (Виттерби по цепи), позиция 0 пришита к уже принятому токену-якорю.
#      Именно это поднимает приёмку: черновик перестаёт предлагать несогласованные хвосты.
#
# ГЕЙТЫ (пройдены до написания этого файла, на НАСТОЯЩИХ весах чекпойнта):
#   свёртка -- ПОБИТОВО равна эталону (0.000e+00), блоки независимы;
#   форма рёбер -- совпала с прямым двойным циклом (9.5e-07 в fp32);
#   Виттерби -- ТОЧНЫЙ максимум, сверено ПОЛНЫМ перебором K^L;
#   отбор != поточечный argmax -- то есть связывание реально работает.
#   Разбор и гейты: solutions/fa2_sm70_cutlass_grade/integrations/vllm_sm70/dflash2/.
#
# ЧЕГО В НАШЕМ ФОРКЕ НЕТ И ЧТО НАПИСАНО ЗДЕСЬ СВОИМ: `LogitsProcessor.get_top_k_tokens` (upstream
# добавил его позже). Правило владельца -- не менять встроенные методы, а дописывать своё, поэтому
# верхушка берётся своей функцией `_верхушка_словаря` НИЖЕ, а встроенный класс не тронут.
# ==================================================================================================

import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.backends import set_model_tag
from vllm.logger import init_logger
from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.triton_utils import tl, triton
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from .qwen3_dflash import (
    DFlashQwen3DecoderLayer,
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)
from .utils import maybe_prefix

logger = init_logger(__name__)


def _grouped_conv(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    block_size: int,
    num_groups: int,
    group_size: int,
    taps: int,
) -> torch.Tensor:
    """Двухтактная свёртка с ПРЕДСКАЗАННЫМ ядром, обрывающаяся на границе блока.

    Обрыв на границе -- не деталь, а условие корректности: блоки разных запросов лежат в одном
    буфере подряд, и такт, перешагнувший границу, потянул бы чужой запрос. Позиция внутри блока
    берётся маской по степени двойки, когда это возможно (блок 1+k почти всегда степень двойки).
    """
    blocks = hidden_states.unflatten(-1, (num_groups, group_size))
    coefficients = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)
    output = coefficients[:, 0] * blocks
    position = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    if block_size & (block_size - 1) == 0:
        position = position & (block_size - 1)
    else:
        position = position % block_size
    for tap in range(1, taps):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        output += coefficients[:, tap] * shifted * (position >= tap).view(-1, 1, 1)
    return output.flatten(-2)


class DFlashGroupedConv(nn.Module):
    """Пара «подготовить/завершить»: обе стороны считаются ОДНИМ умножением проекции.

    Коэффициенты обеих сторон приезжают вместе (`2 * taps * groups`), поэтому вторая свёртка не
    стоит второго прохода по входу -- она берёт уже посчитанное. На нашей машине это ровно тот
    случай, где склейка по выходу бесплатна: слой связан ЧТЕНИЕМ ниже M=128, а блок здесь k+1.
    """

    def __init__(
        self,
        hidden_size: int,
        taps: int,
        group_size: int,
        block_size: int,
        params_dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__()
        if hidden_size % group_size:
            raise ValueError(
                f"conv_group_size={group_size} must divide hidden_size={hidden_size}."
            )
        self.block_size = block_size
        self.taps = taps
        self.group_size = group_size
        self.num_groups = hidden_size // group_size
        self.base_kernel = nn.Parameter(
            torch.empty(2, taps, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        self.kernel_projection = ReplicatedLinear(
            hidden_size,
            2 * taps * self.num_groups,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "kernel_projection"),
            return_bias=False,
        )

    def _convolve(self, hidden_states, delta, side: int) -> torch.Tensor:
        return _grouped_conv(
            hidden_states,
            delta,
            self.base_kernel[side],
            self.block_size,
            self.num_groups,
            self.group_size,
            self.taps,
        )

    def prepare(self, hidden_states: torch.Tensor):
        coefficients = self.kernel_projection(hidden_states).reshape(
            hidden_states.shape[0], 2, self.taps, self.num_groups
        )
        return self._convolve(hidden_states, coefficients[:, 0], 0), coefficients[:, 1]

    def finish(self, hidden_states: torch.Tensor, coefficients: torch.Tensor):
        return self._convolve(hidden_states, coefficients, 1)


# [FA2/SM70 25.08] ДИНАМИКА ПРЕФИКСА ОДНИМ ПУСКОМ.
# Фазомер: `ОТБОР пути` = 1.06 мс из 9.4 мс всего propose, при том что арифметики там
# B*L*K*K = 1*4*16*16 = 1024 умножения, то есть МИКРОсекунды. Всё это -- накладные запуска:
# питоновский цикл по L с четырьмя-пятью тензорными операциями на итерацию плюс обратный
# проход, около пятидесяти запусков ядер на шаг. Ровно тот случай, который правило
# «питон в пути запрещён» и описывает.
#
# Ядро повторяет `динамика_префикса` ОДИН В ОДИН, включая нормировку по предшественнику:
#   q = softmax(рёбра, dim=-1);  G_l(p) = max_c q_l(p,c) * (1 + G_{l+1}(c))
# и обратную протяжку от якоря (позиция 0 -- только предшественник 0).
@triton.jit
def _dp_prefix_kernel(
    edges_ptr, choice_ptr, path_ptr,
    sb, sl, sp,                          # шаги рёбер: батч, позиция, предшественник
    L: tl.constexpr, K: tl.constexpr,
):
    # ИМЕНА ЗДЕСЬ ЛАТИНСКИЕ ВЫНУЖДЕННО: Triton разбирает исходник как AST и отвергает
    # нелатинские идентификаторы («invalid function identifier»). Комментарии он не трогает.
    b = tl.program_id(0)
    k = tl.arange(0, K)
    grid = k[:, None] * sp + k[None, :]            # [K пред, K прее]
    G = tl.zeros([K], dtype=tl.float32)
    # НАЗАД: ожидаемая длина принятого начала, с запоминанием выбора.
    for i in range(L):
        l = L - 1 - i
        e = tl.load(edges_ptr + b * sb + l * sl + grid).to(tl.float32)
        m = tl.max(e, axis=1)
        pe = tl.exp(e - m[:, None])
        q = pe / tl.sum(pe, axis=1)[:, None]       # нормировка ПО ПРЕДШЕСТВЕННИКУ
        val = q * (1.0 + G)[None, :]
        G = tl.max(val, axis=1)
        tl.store(choice_ptr + b * L * K + l * K + k, tl.argmax(val, axis=1).to(tl.int32))
    # ВПЕРЁД: протяжка от якоря. У позиции 0 единственный предшественник -- якорь (0).
    # Предшественник держим ОДНОГОРЯЧИМ вектором: у Triton скаляр нельзя получить
    # индексацией блока, но можно свёрткой (tl.sum по блоку даёт скаляр).
    sel = (k == 0).to(tl.int32)
    for l in range(L):
        ch = tl.load(choice_ptr + b * L * K + l * K + k)
        c = tl.sum(ch * sel)
        tl.store(path_ptr + b * L + l, c.to(tl.int64))
        sel = (k == c).to(tl.int32)


def динамика_префикса_ядром(рёбра: torch.Tensor) -> torch.Tensor:
    """[B,L,K,K] -> [B,L] индексов кандидатов. Один пуск вместо ~50."""
    B, L, K, _ = рёбра.shape
    рёбра = рёбра.contiguous()
    выбор = torch.empty(B, L, K, dtype=torch.int32, device=рёбра.device)
    путь = torch.empty(B, L, dtype=torch.int64, device=рёбра.device)
    _dp_prefix_kernel[(B,)](
        рёбра, выбор, путь,
        рёбра.stride(0), рёбра.stride(1), рёбра.stride(2),
        L=L, K=K, num_warps=1,
    )
    return путь


class CandidateSelector(nn.Module):
    """Отбор СВЯЗНОГО пути по кандидатам. Имена полей = имена в чекпойнте, менять нельзя."""

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        rank: int,
        top_k: int,
        params_dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        # Рычаг честного сравнения: 1 -- прежний отбор эталона (сумма рёбер, Виттерби),
        # 0 (умолчание) -- точный максимум ожидаемой длины принятого начала.
        import os as _os

        self._виттерби = _os.environ.get("FA2SM70_DFLASH2_VITERBI", "0") == "1"
        # Одно ядро вместо ~50 запусков (см. `_ядро_динамики_префикса`). Откат: =0.
        self._ядром = _os.environ.get("FA2SM70_DFLASH2_DP_KERNEL", "1") == "1"
        self._темп = float(_os.environ.get("FA2SM70_DFLASH2_DP_T", "1.0"))
        self._голова = _os.environ.get("FA2SM70_DFLASH2_DP_HEAD", "0") == "1"
        self.predecessor_codebook = nn.Parameter(
            torch.empty(vocab_size, rank, dtype=params_dtype), requires_grad=False
        )
        self.successor_codebook = nn.Parameter(
            torch.empty(vocab_size, rank, dtype=params_dtype), requires_grad=False
        )
        self.hidden_projection = ReplicatedLinear(
            hidden_size,
            rank,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "hidden_projection"),
            return_bias=False,
        )

    def оценить_рёбра(
        self,
        candidate_ids: torch.Tensor,   # [B, L, K]
        unary_logits: torch.Tensor,    # [B, L, K]
        hidden_states: torch.Tensor,   # [B, L, H]
        anchor_token_ids: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        """[B, L, K_пред, K_прее]. Гейт формы пройден против прямого двойного цикла."""
        hidden = self.hidden_projection(hidden_states)
        K = candidate_ids.shape[-1]
        successors = self.successor_codebook[candidate_ids]
        predecessor_ids = torch.cat(
            (anchor_token_ids[:, None, None].expand(-1, 1, K), candidate_ids[:, :-1]),
            dim=1,
        )
        predecessors = self.predecessor_codebook[predecessor_ids]
        return unary_logits[:, :, None] + torch.einsum(
            "blpr,blcr->blpc", predecessors * hidden[:, :, None], successors
        )

    @staticmethod
    def витерби(рёбра: torch.Tensor) -> torch.Tensor:
        """Точный лучший путь по цепи: [B,L,K,K] -> [B,L] индексов кандидатов.

        Перебора K^L не нужно и приближения тоже: граф -- цепь, значит динамика точна. Цикл идёт
        по L (=k+1, единицы), тензоры крошечные; это кандидат на СВОЁ ядро одним пуском, но даже
        так он стоит десятки микросекунд против миллисекунд тела.
        """
        B, L, K, _ = рёбра.shape
        цена = рёбра[:, 0, 0, :]
        откуда = torch.zeros(B, L, K, dtype=torch.long, device=рёбра.device)
        for l in range(1, L):
            цена, лучший = (цена[:, :, None] + рёбра[:, l]).max(dim=1)
            откуда[:, l] = лучший
        путь = torch.zeros(B, L, dtype=torch.long, device=рёбра.device)
        текущий = цена.argmax(dim=1)
        for l in range(L - 1, -1, -1):
            путь[:, l] = текущий
            текущий = откуда[:, l].gather(1, текущий[:, None]).squeeze(1)
        return путь

    @staticmethod
    def динамика_префикса(рёбра: torch.Tensor) -> torch.Tensor:
        """Точный максимум ОЖИДАЕМОЙ ДЛИНЫ ПРИНЯТОГО НАЧАЛА. [B,L,K,K] -> [B,L].

        ПОЧЕМУ НЕ ВИТТЕРБИ (разбор Fable, 25.08). Сумма рёбер раскладывается точно:

            W(s) = log П q_l(s_l|s_{l-1}) + Σ_l log Z_l(s_{l-1}),

        то есть Виттерби максимизирует вероятность совпадения ВСЕХ L позиций СРАЗУ, да ещё с
        паразитным членом Σ log Z (он вознаграждает предшественников с большой массой следующей
        позиции -- к согласию это отношения не имеет). Платят же нам за ДЛИНУ СОВПАВШЕГО НАЧАЛА:
        проверка обрывается на первом расхождении. Максимумы различаются уже при L=K=2, и разрыв
        неограничен.

        Правильная величина сворачивается по Горнеру: E[R] = q_1(1 + q_2(1 + q_3(...))), откуда
        обратная динамика ОДНИМ числом на узел. Фронт Парето не нужен: хвост входит одним
        множителем, а q>0 делает выбор хвоста масштабно-инвариантным. Стоимость O(L*K^2) --
        ТА ЖЕ, что у Виттерби, то есть улучшение ДАРОМ.

        Гейт: сверено с ПОЛНЫМ перебором K^L (integrations/vllm_sm70/dflash2/otbor.py) --
        совпадение точное, Виттерби на тех же данных давал E[R] 0.34 против 2.04.
        """
        B, L, K, _ = рёбра.shape
        # Нормировка ПО ПРЕДШЕСТВЕННИКУ убирает паразитный член: тем же проходом, без лишних чтений.
        q = (рёбра - рёбра.logsumexp(dim=-1, keepdim=True)).exp()
        G = torch.zeros(B, K, dtype=q.dtype, device=q.device)
        выбор = torch.zeros(B, L, K, dtype=torch.long, device=q.device)
        for l in range(L - 1, -1, -1):
            G, лучший = (q[:, l] * (1.0 + G[:, None, :])).max(dim=-1)
            выбор[:, l] = лучший
        путь = torch.zeros(B, L, dtype=torch.long, device=q.device)
        предш = torch.zeros(B, 1, dtype=torch.long, device=q.device)  # позиция 0: только якорь
        for l in range(L):
            c = выбор[:, l].gather(1, предш)
            путь[:, l] = c.squeeze(1)
            предш = c
        return путь

    def forward(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Готовые токены черновика: [B, L]."""
        рёбра = self.оценить_рёбра(
            candidate_ids, unary_logits, hidden_states, anchor_token_ids
        )
        # [КАЛИБРОВКА ДИНАМИКИ 25.08] Динамика максимизирует ОЖИДАЕМУЮ длину принятого начала,
        # считая вероятность приёмки равной softmax(рёбра). Если эта вероятность смещена, обмен
        # «позиция 0 против продолжения» делается по неверному курсу. Замер приёмки по позициям:
        #   поточечный максимум: 0.736 0.368 0.132 ... (tau 2.30)
        #   динамика:            0.664 0.390 0.192 ... (tau 2.40)
        # То есть динамика ПРОДАЁТ первую позицию за хвост. Два вентиля, чтобы курс проверить:
        #   T -- температура рёбер (T>1 сглаживает, уменьшая ценность длинного хвоста);
        #   ГОЛОВА -- запретить всё, кроме поточечного максимума, НА ПЕРВОЙ позиции (остальные
        #   позиции динамика по-прежнему выбирает условно, то есть это не «argmax везде»).
        if self._темп != 1.0:
            рёбра = рёбра / self._темп
        if self._голова:
            рёбра = рёбра.clone()
            рёбра[:, 0, :, 1:] = float("-inf")
        if self._виттерби:
            путь = self.витерби(рёбра)
        elif self._ядром:
            путь = динамика_префикса_ядром(рёбра)
        else:
            путь = self.динамика_префикса(рёбра)
        return candidate_ids.gather(2, путь[:, :, None]).squeeze(2)


def _норма_с_потоком_fp32(norm, hidden_states, residual):
    """RMS-норма, где ПОТОК ОСТАТКА живёт в fp32, а дельты и веса остаются в рабочем типе.

    ПОЧЕМУ ЭТО ОБЯЗАТЕЛЬНО НА VOLTA (замерено, а не предположено). Чекпойнт DFlash2 обучен в
    bf16, и поток остатка у него доходит до **4.8e5** уже после НУЛЕВОГО слоя. У bf16 диапазон
    +-3e38, у fp16 потолок **65504**, а bf16 на sm_70 НЕТ. Штатная `RMSNorm` считает в fp32
    внутри, но возвращает остаток обратно в fp16 -- и он переполняется, давая NaN на всех
    строках маски (строка бонусного токена выживала, потому что у неё нет второго такта свёртки).
    Диагноз снят построчно: вложения конечны, выход -- нет.

    Дельты (свёртка ~3e2, MLP ~2.3e4, внимание ~1e2) в fp16 помещаются, поэтому в fp32 держим
    ТОЛЬКО накопитель: память та же, тензорные ядра fp16 работают как работали.
    """
    residual = hidden_states.float() if residual is None else residual + hidden_states.float()
    x = residual * torch.rsqrt(
        residual.pow(2).mean(dim=-1, keepdim=True) + norm.variance_epsilon
    )
    return x.to(hidden_states.dtype) * norm.weight, residual


class DFlash2Qwen3DecoderLayer(DFlashQwen3DecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        config,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config,
            config=config,
            layer_idx=layer_idx,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )
        draft_config = config.dflash_config
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        conv_args = dict(
            hidden_size=config.hidden_size,
            taps=int(draft_config["conv_kernel_size"]),
            group_size=int(draft_config["conv_group_size"]),
            # Блок = бонусный токен + маски. НЕ block_size из конфига.
            block_size=1 + speculative_config.num_speculative_tokens,
            params_dtype=vllm_config.model_config.dtype,
        )
        self.attention_conv = DFlashGroupedConv(
            **conv_args, prefix=maybe_prefix(prefix, "attention_conv")
        )
        self.mlp_conv = DFlashGroupedConv(
            **conv_args, prefix=maybe_prefix(prefix, "mlp_conv")
        )
        # ФАЛЬСИФИКАТОРЫ СНЯТИЕМ ФАЗЫ. Ответ заведомо неверен -- читается ТОЛЬКО конечность чисел.
        # Флаг снимается в конструкторе: чтение окружения внутри forward рвёт torch.compile.
        import os as _os
        self._без_внимания = _os.environ.get("FA2SM70_DFLASH2_NOATTN", "0") == "1"
        self._без_свёрток = _os.environ.get("FA2SM70_DFLASH2_NOCONV", "0") == "1"
        # Поток в fp32 -- УМОЛЧАНИЕ на этой машине: см. пояснение у `_норма_с_потоком_fp32`.
        self._поток_fp32 = _os.environ.get("FA2SM70_DFLASH2_FP32RES", "1") == "1"

    def forward(self, positions, hidden_states, residual):
        if self._поток_fp32:
            hidden_states, residual = _норма_с_потоком_fp32(
                self.input_layernorm, hidden_states, residual
            )
        elif residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if self._без_свёрток:
            if not self._без_внимания:
                hidden_states = self.self_attn(
                    positions=positions, hidden_states=hidden_states
                )
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )
            hidden_states = self.mlp(hidden_states)
            return hidden_states, residual

        hidden_states, coefficients = self.attention_conv.prepare(hidden_states)
        if not self._без_внимания:
            hidden_states = self.self_attn(
                positions=positions, hidden_states=hidden_states
            )
        hidden_states = self.attention_conv.finish(hidden_states, coefficients)

        if self._поток_fp32:
            hidden_states, residual = _норма_с_потоком_fp32(
                self.post_attention_layernorm, hidden_states, residual
            )
        else:
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )
        hidden_states, coefficients = self.mlp_conv.prepare(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_conv.finish(hidden_states, coefficients)
        return hidden_states, residual


class DFlash2Qwen3Model(DFlashQwen3Model):
    decoder_layer_cls = DFlash2Qwen3DecoderLayer

    def __init__(self, *, vllm_config: VllmConfig, start_layer_id: int = 0, prefix: str = ""):
        super().__init__(
            vllm_config=vllm_config, start_layer_id=start_layer_id, prefix=prefix
        )
        draft_config = self.config.dflash_config
        self.input_embedding_scale = float(
            draft_config.get("input_embedding_scale", 1.0)
        )
        # Своя метка компиляции: без неё селектор делит кэш с телом черновика и первый же
        # прогон ловит чужую форму (мина компил-кэша у нас уже повторялась дважды).
        with set_model_tag("dflash2_candidate_selector"):
            self.candidate_selector = CandidateSelector(
                hidden_size=self.config.hidden_size,
                vocab_size=self.config.vocab_size,
                rank=int(draft_config["selector_rank"]),
                top_k=int(draft_config["selector_top_k"]),
                params_dtype=vllm_config.model_config.dtype,
                prefix=maybe_prefix(prefix, "candidate_selector"),
            )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().embed_input_ids(input_ids) * self.input_embedding_scale

    def forward(self, input_ids, positions, input_embeds=None):
        """Свой проход: тот же порядок, но ПОТОК ОСТАТКА в fp32 (иначе NaN, см. пояснение выше)."""
        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)
        hidden_states = input_embeds
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions, hidden_states=hidden_states, residual=residual
            )
        if getattr(self.layers[0], "_поток_fp32", False):
            hidden_states, _ = _норма_с_потоком_fp32(self.norm, hidden_states, residual)
            return hidden_states
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


# Путь файла-переключателя для ЧЕРЕДОВАНИЯ ветвей внутри одного экземпляра (см. ниже).
_ЧЕРЕД_ФАЙЛ = __import__("os").environ.get("FA2SM70_VOC8_FILE", "")


class DFlash2Qwen3ForCausalLM(DFlashQwen3ForCausalLM):
    model_cls = DFlash2Qwen3Model

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        draft_config = self.config.dflash_config
        softcap = float(draft_config.get("final_logit_softcapping") or 0.0)
        self._описание_кэша_полным()
        import os as _os2
        self._словарь_tm8 = None
        self._словарь_int8 = _os2.environ.get("FA2SM70_DRAFT_VOCAB_I8", "0") == "1"
        self.candidate_logits_processor = LogitsProcessor(
            vllm_config.model_config.get_vocab_size(),
            scale=float(draft_config.get("output_multiplier", 1.0)),
            soft_cap=softcap if softcap > 0 else None,
        )

    def _описание_кэша_полным(self) -> None:
        """Черновику окно нужно НА ВЫЧИСЛЕНИИ, а не на хранении -- меняем ОПИСАНИЕ, не ядро.

        ЗАЧЕМ (замерено дозором кэша на 8K, второй запрос тем же промптом):

            FullAttentionSpec (блок 2048, 4 группы): попадание 4096
            MambaSpec (10 групп):                    попадание 4096
            SlidingWindowSpec (черновик, 1 группа):  попадание 2048   <- режет ВСЕХ
            ИТОГ: 2048 из 8063

        У групп со скользящим окном движок хранит ТОЛЬКО последнее окно, поэтому попадание для них
        возможно лишь у самого конца прошлой последовательности; при блоке кэша 2048 совпасть почти
        невозможно. А минимум берётся по ВСЕМ группам -- и окно черновика обнуляет попадание цели.

        При этом СЕМАНТИКА не меняется: окно 2048 остаётся у ядра (`impl.sliding_window`), черновик
        по-прежнему смотрит не дальше него. Меняется только СРОК ХРАНЕНИЯ блоков: они переживают
        запрос и участвуют в префикс-кэше. Цена -- память: 5 слоёв x контекст x 8 голов x 128 x 2
        байта int8 = 640 МБ на 64K.

        Выключатель: FA2SM70_DFLASH2_FULLSPEC=0.
        """
        import os as _os

        if _os.environ.get("FA2SM70_DFLASH2_FULLSPEC", "1") != "1":
            return
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        сколько = 0
        for сл in self.model.layers:
            вним = getattr(getattr(сл, "self_attn", None), "attn", None)
            if вним is None or getattr(вним, "sliding_window", None) is None:
                continue

            def _полное(self_layer=вним, vllm_config=None, **_):
                block_size = self_layer._fa2sm70_block_size
                return FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=self_layer.num_kv_heads,
                    head_size=self_layer.head_size,
                    head_size_v=getattr(self_layer, "head_size_v", self_layer.head_size),
                    dtype=self_layer.kv_cache_torch_dtype,
                    cache_dtype_str=self_layer.kv_cache_dtype,
                )

            исходный = вним.get_kv_cache_spec

            def обёртка(vllm_config, _вним=вним, _исх=исходный):
                _вним._fa2sm70_block_size = vllm_config.cache_config.block_size
                return _полное(_вним)

            вним.get_kv_cache_spec = обёртка
            сколько += 1
        if сколько:
            logger.info(
                "[FA2/SM70] DFlash2: %d слоёв черновика описаны как ПОЛНОЕ внимание для кэша "
                "(окно %s остаётся у ядра) -- иначе окно обнуляет попадание префикс-кэша всей сети",
                сколько,
                getattr(self.model.layers[0].self_attn.attn, "sliding_window", None),
            )

    def load_weights(self, weights):
        загружено = super().load_weights(weights)
        self._перешкалировать_поток()
        return загружено

    def _перешкалировать_поток(self) -> None:
        """ТОЧНАЯ перепараметризация: весь поток остатка делится на S. Выход НЕ МЕНЯЕТСЯ.

        ЗАЧЕМ. Замер профиля черновика (эталон fp32) на настоящих весах:

            слой | норма | свёртка.подг | gate*up | down     | свёртка.зав | ПОТОК
              0  | 12.9  | 23.4         | 846.8   | 140436.6 | 484810.1    | 301.2
              1  |  9.2  |  7.4         | 750.1   |  13915.0 | 320799.5    | 484836.3
              4  | 29.6  | 11.6         | 1467.0  |   9623.2 |  55431.2    | 457213.0

        Всё, что идёт ЧЕРЕЗ НОРМУ, мало (<40); за потолок fp16 (65504) вылезают только ВЫХОДНЫЕ
        величины: `down`, свёртка-завершить и сам поток -- до 4.9e5. Чекпойнт обучен в bf16
        (диапазон +-3e38), а bf16 на sm_70 НЕТ. Отсюда NaN во всех строках маски.

        ПОЧЕМУ ДЕЛЕНИЕ ТОЧНОЕ. RMS-норма масштабно-инвариантна, а КАЖДАЯ дельта потока проходит
        через выходную проекцию (`o_proj` у внимания, `down_proj` у MLP). Значит деление этих
        двух проекций и вложений на S делит ВЕСЬ поток на S, а вход всех норм не меняется --
        кроме одного места: **eps**. У первого слоя поток -- это одни вложения (rms 3e-3), и
        при делении на S член eps=1e-6 начинает доминировать. Поэтому eps делится на S^2:
        RMS(x/S; eps/S^2) == RMS(x; eps) ТОЖДЕСТВЕННО.

        ГЕЙТ (эталон fp32, настоящие веса): relL2 = **0.000e+00** при S=32 в fp32 и 2.8e-3 в
        fp16 (уровень округления), тогда как при S=1 в fp16 -- inf. Проверено до внедрения.

        Цена: НОЛЬ. Ни памяти, ни времени -- умножения остаются fp16 на тензорных ядрах.
        """
        import os as _os

        S = float(_os.environ.get("FA2SM70_DFLASH2_STREAM_SCALE", "32"))
        if S == 1.0:
            return
        обратно = 1.0 / S
        for сл in self.model.layers:
            сл.self_attn.o_proj.weight.data.mul_(обратно)
            if getattr(сл.self_attn.o_proj, "bias", None) is not None:
                сл.self_attn.o_proj.bias.data.mul_(обратно)
            сл.mlp.down_proj.weight.data.mul_(обратно)
            сл.input_layernorm.variance_epsilon /= S * S
            сл.post_attention_layernorm.variance_epsilon /= S * S
        self.model.norm.variance_epsilon /= S * S
        self.model.input_embedding_scale *= обратно
        logger.info(
            "[FA2/SM70] DFlash2: поток остатка поделён на %g (eps на %g) -- точная "
            "перепараметризация под fp16, выход не меняется",
            S,
            S * S,
        )

    def _верхушка_словаря(self, hidden_states: torch.Tensor, k: int):
        """Свой аналог `get_top_k_tokens` (в нашем форке его нет).

        Встроенный `LogitsProcessor` не трогаем -- правило владельца: дописывать своё, а не менять
        встроенные методы. Логиты берём его же forward-ом (он сам собирает шарды словаря при TP),
        а верхушку снимаем сами.
        """
        # [СВЁРНУТЫЙ ОБМЕН 25.08] Верхушку снимаем ДО сборки шардов, а не после.
        #
        # Было: `LogitsProcessor.forward` собирает ПОЛНЫЕ логиты со всех рангов
        # (248320 столбцов x k строк = 6.9 МБ за шаг через шину) и только потом берёт top-16.
        # Стало: каждый ранг берёт top-16 у СВОЕГО куска словаря и отдаёт 16 пар
        # (значение, глобальный id) -- 896 байт вместо 6.9 МБ, в 7700 раз меньше. На стенде
        # (PCIe ~5 ГБ/с) одна эта сборка стоила порядка миллисекунды на шаг.
        #
        # ТОЧНО, А НЕ ПРИБЛИЖЁННО: глобальные top-k обязаны лежать в объединении поранговых
        # top-k, потому что каждый кандидат целиком принадлежит ровно одному шарду словаря.
        # Идентичность id через float32 законна: словарь 248320 < 2^24, представление точное
        # (тот же приём в штатном `get_top_tokens`).
        лп = self.candidate_logits_processor
        # [СЛОВАРЬ ЧЕРНОВИКА -- В int8, И ЭТО БЕЗОПАСНО ПО ПОСТРОЕНИЮ]
        # Замер: `верхушка словаря` = 2.47 мс из 7.9 мс всего propose, при поле ЧТЕНИЯ 1.63 мс
        # (124160 x 5120 fp16 на ранг). Сжатие вдвое даёт пол 0.82.
        #
        # ПОЧЕМУ РИСКА ДЛЯ ОТВЕТА НЕТ. Словарная голова ОБЩАЯ с целью, и квантовать её НА МЕСТЕ
        # нельзя -- это меняло бы выход сети. Здесь строится ОТДЕЛЬНАЯ int8-копия, которой
        # пользуется ТОЛЬКО черновик. Его кандидаты -- эвристика: цель проверяет каждый
        # предложенный токен, поэтому ошибка кандидата стоит ПРИЁМКИ, а не правильности.
        # Гейт соответственно -- приёмка (tau), а не «17*23».
        # [КОГДА СТРОИТЬ int8-СЛОВАРЬ -- ДВЕ ОШИБКИ ПОДРЯД, ОБЕ ПОЙМАНЫ ЗАМЕРОМ]
        # (1) На `load_weights` строить НЕЛЬЗЯ: словарная голова у черновика ОБЩАЯ с целью и в
        #     этот момент ещё не подключена -- квантуется мусор. Признак: приёмка 0.01 при
        #     идеальных фазах (то есть «быстро и неверно»).
        # (2) Лениво, без оговорок, тоже нельзя: первый вызов может прийтись на ЗАХВАТ ГРАФА, и
        #     выделение 606 МиБ с квантованием попадут внутрь графа -- сквозняк 46.9 -> 37.8 при
        #     улучшившейся фазе словаря.
        # Верно: строить при первом вызове, но ТОЛЬКО когда поток не захватывается. Первый такой
        # вызов -- прогонный (профилирование), он идёт ДО захвата, и графу достаётся готовое.
        уп = getattr(self, "_словарь_tm8", None)
        if уп is None and self._словарь_int8 and not torch.cuda.is_current_stream_capturing():
            уп = self._собрать_словарь_int8()
        # [ЧЕРЕДОВАНИЕ БЕЗ ПЕРЕЗАПУСКА -- 08.09, записка 25 §211]
        # Разброс шага МЕЖДУ экземплярами сервера ~±1.2 мс, а цена этой фазы ~1.1 мс: сравнить
        # два перезапуска нельзя в принципе. Здесь ветвь выбирается ПО ФАЙЛУ, значит обе
        # ветви меряются на ОДНОМ экземпляре, с одним пулом, одними графами и одной картой.
        # Проверка файла -- раз в 64 вызова, то есть ничто (пара микросекунд на шаг).
        if уп is not None and _ЧЕРЕД_ФАЙЛ:
            self._черед_счёт = getattr(self, "_черед_счёт", 0) + 1
            if self._черед_счёт % 64 == 1:
                try:
                    with open(_ЧЕРЕД_ФАЙЛ) as _ф:
                        self._черед_вкл = _ф.read(1) == "1"
                except OSError:
                    self._черед_вкл = True
            if not getattr(self, "_черед_вкл", True):
                уп = None
        if уп is not None:
            xr = hidden_states.reshape(-1, hidden_states.shape[-1])
            логиты = torch.ops.vllm.fa2sm70_draft_tm8_mm(xr, уп[0], уп[1], уп[2], уп[3])
        else:
            логиты = self.lm_head.quant_method.apply(self.lm_head, hidden_states, bias=None)
        if лп.soft_cap is not None:
            логиты = torch.tanh(логиты / лп.soft_cap) * лп.soft_cap
        if лп.scale != 1.0:
            логиты = логиты * лп.scale
        # [СРЕЗ ДЕЙСТВУЕТ ТОЛЬКО ВМЕСТЕ С int8-ПУТЁМ -- 08.09]
        # Логиты со срезом даёт ТОЛЬКО ветвь `уп`; на полной fp16-голове `topk` возвращает
        # позиции во ВСЁМ шарде (124160), а таблица среза короче (22500) -- индексация за
        # границей даёт device-side assert и смерть воркера. Раньше это было неразличимо,
        # потому что ветвь выбиралась один раз при подъёме; с переключателем ветвей (§212)
        # обе ветви живут в одном процессе, и связку пришлось выразить ЯВНО.
        _ср_взят = уп is not None
        доп = self.lm_head.shard_indices.num_org_vocab_padding
        if доп > 0 and not (_ср_взят and getattr(self, "_срез_локальный", None) is not None):
            логиты[..., -доп:] = -float("inf")
        унарные, кандидаты = логиты.topk(k, dim=-1)
        # При срезе topk даёт позиции ВНУТРИ среза -- их надо перевести в локальные позиции
        # шарда и только потом сдвигать на начало шарда.
        _ср = getattr(self, "_срез_локальный", None) if _ср_взят else None
        if _ср is not None:
            кандидаты = _ср[кандидаты]
        кандидаты = кандидаты + self.lm_head.shard_indices.org_vocab_start_index
        if get_tensor_model_parallel_world_size() == 1:
            return унарные, кандидаты
        T = унарные.shape[0]
        пара = torch.stack([унарные.float(), кандидаты.float()], dim=-1).view(T, 2 * k)
        собрано = tensor_model_parallel_all_gather(пара, dim=-1).view(T, -1, 2)
        зн, поз = собрано[:, :, 0].topk(k, dim=-1)
        return зн.to(унарные.dtype), собрано[:, :, 1].gather(1, поз).to(torch.int64)

    def _собрать_словарь_int8(self):
        """Отдельная int8-раскладка словарной головы ДЛЯ ЧЕРНОВИКА (цель не затронута)."""
        try:
            import fa2sm70_draft_tm8 as _д
            import fa2_sm70
        except Exception as ex:  # noqa: BLE001
            logger.warning("[FA2/SM70] словарь черновика в int8 не собран: %s", ex)
            self._словарь_int8 = False
            return None
        w = self.lm_head.weight.data
        if w.dim() != 2 or w.shape[1] % 128 or w.shape[0] % 4:
            self._словарь_int8 = False
            return None
        # [СРЕЗ ЧАСТЫХ ТОКЕНОВ ПОВЕРХ int8 -- записка 25 §113e]
        # По отдельности обе правки не годятся: срез БЕЗ int8 не даёт приза (голова уже около
        # полосы), int8 БЕЗ среза не влезает (1.19 ГиБ при запасе 0.32 при max_model_len=262144).
        # Вместе -- 32K строк в int8 = 0.16 ГиБ, влезает с запасом и даёт ~1.3 мс из 10.4.
        # Срез применяется к ЛОКАЛЬНОЙ части шарда: индекс частот глобальный, а голова при TP=2
        # разрезана (см. разбор в `_верхушка_словаря`), и глобальными id её индексировать нельзя
        # -- ровно это дало device-side assert в первой попытке.
        import os as _osl                       # _os2 живёт в __init__, здесь он не виден
        self._срез_локальный = None
        _сп = _osl.environ.get("FA2SM70_DRAFT_VOCAB", "")
        if _сп:
            try:
                _гл = torch.load(_сп, map_location="cpu").to(torch.long)
                _s = int(self.lm_head.shard_indices.org_vocab_start_index)
                _e = _s + int(w.shape[0])
                _лок = (_гл[(_гл >= _s) & (_гл < _e)] - _s).to(w.device)
                _лок = _лок[: (_лок.numel() // 4) * 4]          # prepare требует кратности 4
                if _лок.numel() >= 1024:
                    w = w.index_select(0, _лок).contiguous()
                    self._срез_локальный = _лок
                    logger.info("[FA2/SM70] срез словаря черновика: %d строк из %d на ранге",
                                _лок.numel(), _e - _s)
            except Exception as ex:  # noqa: BLE001
                logger.warning("[FA2/SM70] срез словаря не применён: %s", ex)
        коды, масштабы, нули = _д._квантовать(w)
        tm_w, tm_s, meta = fa2_sm70._ext.tm8_ext().prepare(коды, масштабы, нули, 128)
        del коды, масштабы, нули
        torch.cuda.empty_cache()
        self._словарь_tm8 = (tm_w, tm_s, int(meta[0]), int(meta[1]))
        logger.info("[FA2/SM70] словарь черновика в int8: [%d, %d], %.0f МиБ",
                    w.shape[0], w.shape[1], w.numel() / (1 << 20))
        return self._словарь_tm8

    def compute_candidates(self, hidden_states: torch.Tensor):
        """[T, H] -> ([T, K] логиты, [T, K] id). K = selector_top_k."""
        return self._верхушка_словаря(hidden_states, self.model.candidate_selector.top_k)

    def выбрать_путь(
        self,
        hidden_states: torch.Tensor,   # [B, L, H]
        candidate_ids: torch.Tensor,   # [B, L, K]
        unary_logits: torch.Tensor,    # [B, L, K]
        anchor_token_ids: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        """Связный путь -> [B, L] токенов черновика."""
        return self.model.candidate_selector(
            candidate_ids, unary_logits, hidden_states, anchor_token_ids
        )


EntryClass = DFlash2Qwen3ForCausalLM
