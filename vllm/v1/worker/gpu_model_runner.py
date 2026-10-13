# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import gc
import os as _os
import itertools
import threading
import os
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from copy import copy, deepcopy
from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias, cast

import numpy as np
import torch
import torch.distributed
import torch.nn as nn
from tqdm import tqdm

import vllm.envs as envs
from vllm.compilation.counter import compilation_counter
from vllm.compilation.cuda_graph import CUDAGraphStat, CUDAGraphWrapper
from vllm.compilation.monitor import set_cudagraph_capturing_enabled
from vllm.config import (
    CompilationMode,
    CUDAGraphMode,
    VllmConfig,
    get_layers_from_vllm_config,
    update_config,
)
from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.distributed.eplb.eplb_state import EplbState
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.kv_transfer.kv_connector.utils import copy_kv_blocks
from vllm.distributed.parallel_state import (
    get_dcp_group,
    get_pp_group,
    get_tp_group,
    graph_capture,
    is_global_first_rank,
    prepare_communication_buffer_for_model,
)
from vllm.forward_context import (
    BatchDescriptor,
    set_forward_context,
)
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping, LoRAMappingType
from vllm.model_executor.layers.attention import Attention, MLAAttention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
)
from vllm.model_executor.layers.rotary_embedding import (
    MRotaryEmbedding,
    XDRotaryEmbedding,
)
from vllm.model_executor.model_loader import TensorizerLoader, get_model_loader
from vllm.model_executor.model_loader.reload import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
)
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsXDRoPE,
    is_mixture_of_experts,
    supports_eagle3,
    supports_mrope,
    supports_multimodal_pruning,
    supports_realtime,
    supports_transcription,
    supports_xdrope,
)
from vllm.model_executor.models.interfaces_base import (
    VllmModelForPooling,
    is_pooling_model,
    is_text_generation_model,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.budget import MultiModalBudget
from vllm.multimodal.inputs import (
    BatchedTensorInputs,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.multimodal.utils import group_mm_kwargs_by_modality
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingType
from vllm.sequence import IntermediateTensors
from vllm.tasks import GenerationTask, PoolingTask, SupportedTask
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.utils.jsontree import json_map_leaves
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.mem_utils import DeviceMemoryProfiler, format_gib
from vllm.utils.nvtx_pytorch_hooks import PytHooks
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.torch_utils import (
    get_dtype_size,
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.utils import (
    create_fast_prefill_custom_backend,
    get_dcp_local_seq_lens,
    reorder_batch_to_split_decodes_and_prefills,
)
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    ChunkedLocalAttentionSpec,
    CrossAttentionSpec,
    EncoderOnlyAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncModelRunnerOutput,
    DraftTokenIds,
    ECConnectorOutput,
    KVConnectorOutput,
    LogprobsLists,
    LogprobsTensors,
    ModelRunnerOutput,
    PoolerOutput,
    SamplerOutput,
    make_empty_encoder_model_runner_output,
)
from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates
from vllm.v1.sample.logits_processor import LogitsProcessors, build_logitsprocs
from vllm.v1.sample.logits_processor.interface import LogitsProcessor
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.draft_model import DraftModelProposer
from vllm.v1.spec_decode.eagle import EagleProposer
# [FA2/SM70 25.08] Свой предлагатель DFlash2: ОДИН проход на блок (у eagle -- цикл по k) и
# отбор СВЯЗНОГО пути вместо поточечного argmax. Наследование от EagleProposer было бы
# наследованием ровно того цикла, ради снятия которого метод и берётся.
from vllm.v1.spec_decode.dflash2 import DFlash2Proposer
from vllm.v1.spec_decode.medusa import MedusaProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.suffix_decoding import SuffixDecodingProposer
from vllm.v1.structured_output.utils import apply_grammar_bitmask
from vllm.v1.utils import CpuGpuBuffer, record_function_or_nullcontext
from vllm.v1.worker import mamba_utils
from vllm.v1.worker.cp_utils import (
    check_attention_cp_compatibility,
    get_total_cp_world_size,
)
from vllm.v1.worker.dp_utils import coordinate_batch_across_dp
from vllm.v1.worker.ec_connector_model_runner_mixin import ECConnectorModelRunnerMixin
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper
from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorModelRunnerMixin
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin
from vllm.v1.worker.ubatch_utils import (
    UBatchSlices,
    check_ubatch_thresholds,
    maybe_create_ubatch_slices,
    split_attn_metadata,
)
from vllm.v1.worker.utils import is_residual_scattered_for_sp
from vllm.v1.worker.workspace import lock_workspace

from .utils import (
    AttentionGroup,
    add_kv_sharing_layers_to_kv_cache_groups,
    bind_kv_cache,
    sanity_check_mm_encoder_outputs,
)

if TYPE_CHECKING:
    from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.spec_decode.ngram_proposer import NgramProposer

# Флаг прибора слоёв читается ОДИН раз при загрузке: шаг декода -- горячий путь.
_ДБГ_СЛОИ_ВКЛ = _os.environ.get("FA2SM70_DBG_LAYERS", "0") == "1"

logger = init_logger(__name__)

# Пропуск построения метаданных для группы черновика (см. _build_attn_group_metadata).
_ПРОПУСК_МЕТЫ_ЧЕРНОВИКА = os.environ.get("FA2SM70_SKIP_DRAFT_META", "0") == "1"
# [ПРИБОР ФАЗ -- КОНСТАНТА, 31.08] `int(os.environ.get("FA2SM70_STEP_PHASE"))` стояло в ДВАДЦАТИ
# ОДНОМ месте горячего шага плюс внутри самого `_фаза_шага`: обращение к словарю окружения и
# разбор строки на каждом, и всё это при ВЫКЛЮЧЕННОМ приборе. Окружение в рантайме не меняется.
# [01.09] ВКЛЮЧЕНО ПО УМОЛЧАНИЮ. Это лечение обрыва на границе блока состояния,
# который отчёт reports/vllm-granica-ne-ushla-2026-08-31.md фиксирует как НЕ ушедший
# (5 событий на 23 генерациях, две точные подписи 4095 и 4097). Механизм: на шаге,
# где k+1 черновых токенов переходят границу, спекуляция снимается и шаг идёт
# обычным декодом, который границу проходит верно. Цена -- 5 шагов из 4096 (0.12 %).
# Откат -- FA2SM70_GDN_GRAN_SKIP=0.
_ГРАН_ПРОПУСК = os.environ.get("FA2SM70_GDN_GRAN_SKIP", "1") == "1"
_ГРАН_ЛОГ = os.environ.get("FA2SM70_GDN_GRAN_LOG", "0") == "1"
# Отброс бонусного токена у границы блока (записка 25 §123). Рычаг отката.
# ОТБРОС БОНУСНОГО ТОКЕНА У ГРАНИЦЫ -- ГИПОТЕЗА ОПРОВЕРГНУТА ЗАМЕРОМ (04.09).
# Обрыв объяснялся неверным бонусным токеном; правка исполнялась (пометка PLACEHOLDER
# печаталась прибором), но обрывов осталось 7 из 20 против 9 из 20 -- шум. Настоящая
# причина найдена в адресации состояния GDN у границы блока (FA2SM70_GRAN_SRC).
# Рычаг оставлен выключенным: код рабочий, но лишний.
_ГРАН_БОНУС = os.environ.get("FA2SM70_GRAN_BONUS", "0") == "1"
# Контроль исполнимости: снимать черновик ВСЕГДА. Если при нём шаги всё равно идут с
# q=k+1, значит правка черновика в этом месте на планирование не влияет вовсе.
_ГРАН_ВСЕГДА = os.environ.get("FA2SM70_GRAN_ALL_OFF", "0") == "1"
# Прибор границы: сверяет, что для КАЖДОЙ позиции шага блок уже выделен в таблице.
# Если нет -- compute_slot_mapping возьмёт block_table[req, k] == 0, то есть KV уедет
# в ЧУЖОЙ блок 0. Только стенд: печать в горячем пути.
_ГРАН_СЛОТ = os.environ.get("FA2SM70_GRAN_SLOT", "0") == "1"
# Ниже этого числа посчитанных токенов спекуляция не запускается: на вырожденном
# контексте она даёт пустые ответы (отчёт 31.08, раздел 5). 0 -- выключить порог.
_МИН_КОНТЕКСТ = int(os.environ.get("FA2SM70_SPEC_MINCTX", "32"))
_ДЕРЕВО_W = int(os.environ.get("FA2SM70_TREE_W", "0"))
# Фальсификатор: 1 = дерево строится и маскируется, но приёмке ВСЕГДА отдаётся
# ветвь A. Отделяет вину маски/черновика от вины ВЫБОРА ветви.
_ДЕРЕВО_ТОЛЬКО_A = os.environ.get("FA2SM70_TREE_ONLY_A", "0") == "1"
# [ПРОСТОЙ КОНВЕЙЕРА, 07.09] Замерено: со спекуляцией GPU делает МЕНЬШЕ работы
# (18-19 мс/шаг против 24.9 без неё), а шаг ВТРОЕ длиннее (43-60 против 22.2) --
# значит 20 мс постоянной платы это ПРОСТОЙ, а не работа. Единственная блокирующая
# синхронизация спекулятивной ветки -- снятие приёмки на хост в конце шага
# (.cpu().numpy()), тогда как значение нужно только в НАЧАЛЕ следующего шага.
# Рычаг убирает круг "GPU -> хост -> GPU": значение остаётся на устройстве.
# ЗАЩИТА: путь применяется ТОЛЬКО когда состав батча не менялся (нет новых и нет
# завершённых запросов) -- иначе индексы хозяйского массива и строк вывода
# расходятся, и приёмка уедет не тому запросу.
_БЕЗ_СИНХРО_ПРИЁМКИ = os.environ.get("FA2SM70_NOSYNC_ACC", "0") == "1"
# [ВЫБОРКА ДЛЯ ПЕРЕРАНЖИРОВЩИКА, 07.09] Метка -- argmax ЦЕЛИ на позиции 0; вход --
# скрытое состояние черновика и его 16 кандидатов. Всё это уже посчитано, платы нет.
_ВЫБОРКА_ФАЙЛ = os.environ.get("FA2SM70_RERANK_DUMP", "")
_ВЫБОРКА_ПРЕДЕЛ = int(os.environ.get("FA2SM70_RERANK_MAX", "60000"))
# [БИСЕКЦИЯ 07.09] Биты режима покрывают conv, KV и off-сдвиг, но НЕ покрывают
# подмену позиций и branch_at. Два недостающих рычага -- ровно для того, чтобы
# отделить, что именно портит выход при включённом дереве.
_ДЕРЕВО_БЕЗ_ПОЗ = os.environ.get("FA2SM70_TREE_NOPOS", "0") == "1"
_ДЕРЕВО_СЧЁТ = int(os.environ.get("FA2SM70_TREE_COUNT", "0"))
_ДЕРЕВО_ФАЙЛ = os.environ.get("FA2SM70_TREE_FILE", "")
_ДЕРЕВО_ТАУ = int(os.environ.get("FA2SM70_TREE_TAU", "0"))
# [ДЕРЕВО, 05.09] Принятие ветви B: сдвиг чтения состояний (через off-буфер qwen3_next)
# и слот-копия KV внимания. Эксперимент: порядок строк батча между соседними шагами
# предполагается неизменным (чистый спекулятивный декод).
_ДЕРЕВО_КВ = os.environ.get("FA2SM70_TREE_KV", "0") == "1"
_ДЕРЕВО_КВ_КОПИЯ = os.environ.get("FA2SM70_TREE_KV_COPY", "1") == "1"   # под-рычаг для бинарного поиска
_ДЕРЕВО_OFF_ВКЛ = os.environ.get("FA2SM70_TREE_OFF", "1") == "1"        # off-сдвиг чтения SSM
# [05.09-3] Умолчание 0: обе правки окна (полный своп из чернового и точечная копия
# столбцов) в бою ДОБАВЛЯЮТ порчу (136 и 206 «!» против 8 без них), хотя по содержимому
# верны (дозор окон) и на офлайн-тесте дают 6e-08. Взаимодействие 5- и 7-элементной
# семантик окна между шагами не понято до конца -- чинить только после расширенного
# офлайн-теста, воспроизводящего боевую пару вызовов на одном пуле.
_ДЕРЕВО_CONV_SWAP = os.environ.get("FA2SM70_TREE_CONV_SWAP", "0") == "1" # подмена conv-окна
# Сколько верхних логитов цели складывать для дистилляции черновика (0 = не собирать).
_ЛОГИТЫ_ТОП = int(os.environ.get("FA2SM70_DFLASH2_TOPK", "0"))
_ФАЗА_ШАГА_N = int(os.environ.get("FA2SM70_STEP_PHASE", "0"))
# КОНСТАНТА, а не чтение окружения на каждый вызов -- см. пояснение к _ФАЗА_ШАГА_N выше.
_ФАЗА_ДЕВ = int(os.environ.get("FA2SM70_STEP_PHASE_DEV", "0"))
# [БЕЗ СИНХРОНИЗАЦИИ -- 09.09, записка 25 §255] Фаза «прямой проход цели» меряется с
# `torch.cuda.synchronize()`, то есть под фазомером шаг СЕРИАЛИЗОВАН и щель между работой
# карты и шагом не видна вовсе. Для разбора щели нужен прибор, который не перестраивает
# конвейер: окна на событиях (`_фаза_карты`) его и не трогают, а вот эта синхронизация
# трогает. Рычаг снимает её, оставляя окна.
_ФАЗА_БЕЗ_СИНХР = int(os.environ.get("FA2SM70_STEP_PHASE_NOSYNC", "0"))
# [СЪЁМ ДАННЫХ ДЛЯ ОБУЧЕНИЯ ГОЛОВЫ MTP -- 11.09] Признаки цели по ВСЕМ позициям префилла плюс
# их токены. Почему здесь: голова MTP получает РОВНО этот тензор (`target_hidden_states`), то
# есть собранное обучающее множество совпадает с боевым входом побайтово -- именно тот гейт,
# которого не было у четырёх прежних заходов обучения черновика (записка 25: прибор двигали, а
# вход с боем не совпадал НИ ОДНИМ звеном). Префилл даёт 3000 позиций за запрос против одной
# за шаг декода, поэтому корпус собирается минутами, а не днями.
# Умолчание ПУСТО: venv общий с боевым, и в выключенном виде это одна проверка строки на шаг.
_ДАМП_MTP = os.environ.get("FA2SM70_MTP_DUMP", "")
_ДАМП_MTP_МИН = int(os.environ.get("FA2SM70_MTP_DUMP_MINTOK", "512"))
_ДАМП_MTP_N = [0]
# [ОПИСЬ ПАМЯТИ -- 09.09] Между весами (13.5 ГиБ на ранг) и пулом KV (9.9) с одной стороны и
# занятыми 31.4 ГиБ с другой -- восемь гигабайт неучтённых, а нам не хватает 303 МиБ под
# int4-словарь черновика (§241). Печатаем разложение РАЗ, после первого шага: сколько торч
# ДЕРЖИТ (reserved) против того, что реально занято (allocated) -- разница и есть то, что
# распределитель не отдал драйверу. 0 = ветка мертва.
_ОПИСЬ_ПАМЯТИ = int(os.environ.get("FA2SM70_MEM_DUMP", "0"))
# [ВОЗВРАТ ДЕРЖАННОЙ ПАМЯТИ -- 09.09] Опись показала: распределитель торча ДЕРЖИТ 30.75 ГиБ,
# а занято 29.73 -- гигабайт свободных блоков, которые он не отдаёт драйверу, при том что у
# драйвера свободно 0.34. Из-за этого не помещался int4-словарь черновика (303 МиБ, +9 %
# декода, §241) и падала игла на 128K. Разовый `empty_cache` ПОСЛЕ захвата графов возвращает
# эти блоки: графы уже сняты, пул KV выделен, а транзиентные буферы прогрева больше не нужны.
# Делается ОДИН раз на первом шаге и печатает до/после -- чтобы приз был виден, а не заявлен.
_ВОЗВРАТ_ПАМЯТИ = os.environ.get("FA2SM70_MEM_TRIM", "0") == "1"
# [ПОДРЕЗКА ПО ПОРОГУ -- 09.09, записка 25 §252] Разовый возврат после захвата не помогает:
# дробление КОПИТСЯ во время длинного префилла (чанки по 4096 дают буферы разного размера).
# Отказ иглы на 128K при включённом int4-словаре был именно такой: «не хватило 52 МиБ, при
# 809 МиБ держанных и НЕиспользованных». Здесь возврат делается ПЕРЕД шагом префилла, когда
# держанного-но-свободного больше порога. Обе величины берутся у распределителя БЕЗ
# синхронизации с картой, поэтому проверка стоит наносекунды; сам `empty_cache` зовётся
# редко и только на префилле, где он тонет в сотнях миллисекунд работы.
_ПОДРЕЗ_МиБ = int(os.environ.get("FA2SM70_MEM_TRIM_MB", "0"))
# Порог по числу токенов шага: ниже него распределитель не опрашивается вовсе. Декодный шаг со
# спекуляцией это (k+1)*запросов, то есть десятки; префилльный чанк -- тысячи.
_ПОДРЕЗ_ТОКЕНОВ = int(os.environ.get("FA2SM70_MEM_TRIM_MINTOK", "512"))

AttnMetadataDict: TypeAlias = dict[str, AttentionMetadata]
# list when ubatching is enabled
PerLayerAttnMetadata: TypeAlias = list[AttnMetadataDict] | AttnMetadataDict


# Wrapper for ModelRunnerOutput to support overlapped execution.
class AsyncGPUModelRunnerOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        sampled_token_ids: torch.Tensor,
        logprobs_tensors: LogprobsTensors | None,
        invalid_req_indices: list[int],
        async_output_copy_stream: torch.cuda.Stream,
        vocab_size: int,
        # [ГРАНИЦА БЛОКА, 04.09] Разбор выдачи спекуляции при асинхронном планировании идёт
        # ЗДЕСЬ, а не в бегунке, и объекту вывода недоступны ни контекст запросов, ни размер
        # блока -- их приходится передать снимком на момент шага.
        гран_ctx: list[int] | None = None,
        гран_B: int = 0,
    ):
        self._model_runner_output = model_runner_output
        self._invalid_req_indices = invalid_req_indices
        self._гран_ctx = гран_ctx
        self._гран_B = гран_B
        self._гран_снят = 0

        # Event on the copy stream so we can synchronize the non-blocking copy.
        self.async_copy_ready_event = torch.Event()

        # Keep a reference to the device tensor to avoid it being
        # deallocated until we finish copying it to the host.
        self._sampled_token_ids = sampled_token_ids
        self.vocab_size = vocab_size
        self._logprobs_tensors = logprobs_tensors

        # Initiate the copy on a separate stream, but do not synchronize it.
        default_stream = torch.cuda.current_stream()
        with torch.cuda.stream(async_output_copy_stream):
            async_output_copy_stream.wait_stream(default_stream)
            self.sampled_token_ids_cpu = self._sampled_token_ids.to(
                "cpu", non_blocking=True
            )
            self._logprobs_tensors_cpu = (
                self._logprobs_tensors.to_cpu_nonblocking()
                if self._logprobs_tensors
                else None
            )
            self.async_copy_ready_event.record()

    def _гран_бонус_пометить(self, тензор):
        """Помечает БОНУСНЫЙ токен отвергнутым у строк, чьи позиции пересекли границу блока.

        Бонусный -- последний НЕотвергнутый в строке: целевая модель считает его на позиции
        последнего принятого, и когда та легла по другую сторону границы, он приходит неверным
        (обычно EOS, отсюда `finish_reason=stop` посреди слова). Строка, где принят только он
        сам, не трогается -- иначе шаг не выдал бы ни одного токена.

        Тензор выдачи -- INFERENCE-тензор: запись в него на месте запрещена (RuntimeError),
        поэтому строки сперва отбираются, и лишь при непустом отборе делается клон. На шагах
        без границы (99.98 %) не копируется ничего.
        """
        # PLACEHOLDER_TOKEN_ID живёт в rejection_sampler (значение -1), не в metadata.
        from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID as _PH
        B = self._гран_B
        ctx_все = self._гран_ctx
        n = min(int(тензор.shape[0]), len(ctx_все))
        отбор = []
        for i in range(n):
            годных = int((тензор[i] != _PH).sum())
            if годных < 2:
                continue
            ctx = int(ctx_все[i])
            if ctx <= 0:
                continue
            # Позиции шага: ctx .. ctx+годных-1. Бонусный посчитан на ctx+годных-1.
            if ctx // B != (ctx + годных - 1) // B:
                отбор.append((i, годных))
        if not отбор:
            return тензор
        тензор = тензор.clone()
        for i, годных in отбор:
            тензор[i, годных - 1] = _PH
        self._гран_снят += len(отбор)
        if _ГРАН_ЛОГ:
            import sys as _s3
            print(f"[ГРАНИЦА] бонусный токен отброшен у {len(отбор)} строк: "
                  f"{отбор} B={B} всего={self._гран_снят}", file=_s3.stderr, flush=True)
        return тензор

    def get_output(self) -> ModelRunnerOutput:
        """Copy the device tensors to the host and return a ModelRunnerOutput.

        This function blocks until the copy is finished.
        """
        max_gen_len = self.sampled_token_ids_cpu.shape[-1]
        self.async_copy_ready_event.synchronize()

        # Release the device tensors once the copy has completed.
        del self._logprobs_tensors
        del self._sampled_token_ids
        if max_gen_len == 1:
            valid_sampled_token_ids = self.sampled_token_ids_cpu.tolist()
            for i in self._invalid_req_indices:
                valid_sampled_token_ids[i].clear()
            logprobs_lists = None
            if self._logprobs_tensors_cpu is not None:
                logprobs_lists = self._logprobs_tensors_cpu.tolists()
        else:
            # [ГРАНИЦА БЛОКА -- ОТБРОС БОНУСНОГО ТОКЕНА, 04.09] Пометка делается В ТЕНЗОРЕ,
            # до разбора: `parse_output` отфильтрует PLACEHOLDER штатно, и счётчики принятых
            # (они считаются из этого же тензора) остаются согласованными. Первая редакция
            # удаляла токен из готового списка -- движок вис на рассинхроне счётчиков.
            _выдача = self.sampled_token_ids_cpu
            if _ГРАН_БОНУС and self._гран_ctx and self._гран_B:
                _выдача = self._гран_бонус_пометить(_выдача)
            valid_sampled_token_ids, logprobs_lists = RejectionSampler.parse_output(
                _выдача,
                self.vocab_size,
                self._invalid_req_indices,
                logprobs_tensors=self._logprobs_tensors_cpu,
            )

        output = self._model_runner_output
        # [ГРАНИЦА БЛОКА -- ОТБРОС БОНУСНОГО ТОКЕНА, 04.09]
        # Диагноз (записка 25 §121b): спекуляция даёт ТОТ ЖЕ текст, что обычный декод, но
        # преждевременно завершает его -- приходит лишний EOS, и `check_stop` обрывает запрос
        # (`finish_reason=stop`, текст оборван посреди слова, сумма кратна block_size).
        # Последний столбец выдачи rejection sampler -- БОНУСНЫЙ токен: его целевая модель
        # считает на позиции последнего ПРИНЯТОГО. Когда k+1 позиций легли по обе стороны
        # границы блока, KV этой позиции читается из блока, который на этом шаге ещё не
        # согласован, и бонусный токен получается неверным -- чаще всего EOS.
        # Лечение точечное: на таком шаге бонусный токен отбрасывается, принятые остаются.
        # Следующий шаг пересчитает его уже внутри нового блока. Цена -- один токен на
        # block_size, то есть 0.02 %, и никакого влияния на скорость: работа уже сделана.
        output.sampled_token_ids = valid_sampled_token_ids
        output.logprobs = logprobs_lists
        return output


class AsyncGPUPoolingModelRunnerOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        raw_pooler_output: PoolerOutput,
        finished_mask: list[bool],
        async_output_copy_stream: torch.cuda.Stream,
    ):
        self._model_runner_output = model_runner_output

        # Event on the copy stream so we can synchronize the non-blocking copy.
        self.async_copy_ready_event = torch.Event()

        # Keep a reference to the device tensors to avoid them being
        # deallocated until we finish copying it to the host.
        self._raw_pooler_output = raw_pooler_output

        # Initiate the copy on a separate stream, but do not synchronize it.
        default_stream = torch.cuda.current_stream()
        with torch.cuda.stream(async_output_copy_stream):
            async_output_copy_stream.wait_stream(default_stream)
            raw_pooler_output_cpu = json_map_leaves(
                lambda x: None if x is None else x.to("cpu", non_blocking=True),
                self._raw_pooler_output,
            )
            self.async_copy_ready_event.record()
            self._model_runner_output.pooler_output = [
                out if include else None
                for out, include in zip(raw_pooler_output_cpu, finished_mask)
            ]

    def get_output(self) -> ModelRunnerOutput:
        """Copy the device tensors to the host and return a ModelRunnerOutput.
        This function blocks until the copy is finished.
        """
        self.async_copy_ready_event.synchronize()

        # Release the device tensors once the copy has completed.
        del self._raw_pooler_output
        return self._model_runner_output


class ExecuteModelState(NamedTuple):
    """Ephemeral cached state transferred between execute_model() and
    sample_tokens(), after execute_model() returns None."""

    scheduler_output: "SchedulerOutput"
    logits: torch.Tensor
    spec_decode_metadata: SpecDecodeMetadata | None
    spec_decode_common_attn_metadata: CommonAttentionMetadata | None
    hidden_states: torch.Tensor
    sample_hidden_states: torch.Tensor
    aux_hidden_states: list[torch.Tensor] | None
    ec_connector_output: ECConnectorOutput | None
    cudagraph_stats: CUDAGraphStat | None
    slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None


class GPUModelRunner(
    LoRAModelRunnerMixin, KVConnectorModelRunnerMixin, ECConnectorModelRunnerMixin
):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config

        from vllm.model_executor.models.utils import set_cpu_offload_max_bytes

        set_cpu_offload_max_bytes(int(self.cache_config.cpu_offload_gb * 1024**3))

        model_config = self.model_config
        cache_config = self.cache_config
        scheduler_config = self.scheduler_config
        parallel_config = self.parallel_config
        self.device = device
        self.pin_memory = is_pin_memory_available()
        self.dtype = self.model_config.dtype

        self.kv_cache_dtype = kv_cache_dtype_str_to_dtype(
            cache_config.cache_dtype, self.model_config
        )

        self.is_pooling_model = model_config.runner_type == "pooling"
        self.enable_prompt_embeds = model_config.enable_prompt_embeds
        self.is_multimodal_raw_input_only_model = (
            model_config.is_multimodal_raw_input_only_model
        )
        # This will be overridden in load_model()
        self.is_multimodal_pruning_enabled = False
        self.max_model_len = model_config.max_model_len

        # Always set to false after the first forward pass
        self.calculate_kv_scales = self.cache_config.calculate_kv_scales
        self.dcp_world_size = self.parallel_config.decode_context_parallel_size
        self.dcp_rank = 0 if self.dcp_world_size <= 1 else get_dcp_group().rank_in_group
        self.max_num_tokens = scheduler_config.max_num_batched_tokens
        self.max_num_reqs = scheduler_config.max_num_seqs

        # Broadcast PP output for external_launcher (torchrun)
        # to make sure we are synced across pp ranks
        # TODO: Support overlapping mirco-batches
        # https://github.com/vllm-project/vllm/issues/18019
        self.broadcast_pp_output = (
            self.parallel_config.distributed_executor_backend == "external_launcher"
            and len(get_pp_group().ranks) > 1
        )

        # Model-related.
        self.num_query_heads = model_config.get_num_attention_heads(parallel_config)
        self.inputs_embeds_size = model_config.get_inputs_embeds_size()
        self.attention_chunk_size = model_config.attention_chunk_size
        # Only relevant for models using ALiBi (e.g, MPT)
        self.use_alibi = model_config.uses_alibi

        self.cascade_attn_enabled = not self.model_config.disable_cascade_attn
        self.is_mm_prefix_lm = self.model_config.is_mm_prefix_lm

        # Multi-modal data support
        self.mm_registry = MULTIMODAL_REGISTRY
        self.uses_mrope = model_config.uses_mrope
        self.uses_xdrope_dim = model_config.uses_xdrope_dim
        self.supports_mm_inputs = self.mm_registry.supports_multimodal_inputs(
            model_config
        )

        if self.model_config.is_encoder_decoder:
            # Maximum length of the encoder input, only for encoder-decoder
            # models.
            self.max_encoder_len = scheduler_config.max_num_encoder_input_tokens
        else:
            self.max_encoder_len = 0

        # Async scheduling
        self.use_async_scheduling = self.scheduler_config.async_scheduling

        # Sampler
        self.sampler = Sampler(logprobs_mode=self.model_config.logprobs_mode)

        self.eplb_state: EplbState | None = None
        """
        State of the expert parallelism load balancer.

        Will be lazily initialized when the model is loaded.
        """

        # Lazy initializations
        # self.model: nn.Module  # Set after load_model
        # Initialize in initialize_kv_cache
        self.kv_caches: list[torch.Tensor] = []
        # Initialize in initialize_kv_cache_tensors
        self.cross_layers_kv_cache: torch.Tensor | None = None
        self.cross_layers_attn_backend: type[AttentionBackend] | None = None
        # indexes: [kv_cache_group_id][attn_group]
        self.attn_groups: list[list[AttentionGroup]] = []
        # self.kv_cache_config: KVCacheConfig

        # mm_hash ->  encoder_output
        self.encoder_cache: dict[str, torch.Tensor] = {}

        self.use_aux_hidden_state_outputs = False
        # Set up speculative decoding.
        # NOTE(Jiayi): currently we put the entire draft model on
        # the last PP rank. This is not ideal if there are many
        # layers in the draft model.
        if self.speculative_config and get_pp_group().is_last_rank:
            self.drafter: (
                NgramProposer  # noqa: F823
                | SuffixDecodingProposer
                | EagleProposer
                | DFlash2Proposer
                | DraftModelProposer
                | MedusaProposer
            )
            if self.speculative_config.method == "dflash2":
                # [FA2/SM70 25.08] ПОЛНАЯ вторая версия: блок за один проход + связывание позиций
                # селектором (точный путь по цепи). Сведение чекпойнта к V1 теряло селектор.
                self.drafter = DFlash2Proposer(self.vllm_config, self.device, self)
                self.use_aux_hidden_state_outputs = getattr(
                    self.drafter, "eagle3_use_aux_hidden_state", True
                )
            elif self.speculative_config.method == "dflash":
                # [FA2/SM70 24.08] Блочно-диффузионный черновик: k токенов за ОДИН проход, вместо
                # k последовательных. Наш замер показал, что именно последовательность проходов
                # съедает выигрыш спекуляции (14.3 мс НА ПРОХОД), а у этого черновика ещё и
                # скользящее окно 2048 -- значит его цена НЕ РАСТЁТ с длиной контекста.
                from vllm.v1.spec_decode.dflash import DFlashProposer

                self.drafter = DFlashProposer(self.vllm_config, self.device, self)
                self.use_aux_hidden_state_outputs = getattr(
                    self.drafter, "eagle3_use_aux_hidden_state", True
                )
            elif self.speculative_config.method == "ngram":
                from vllm.v1.spec_decode.ngram_proposer import NgramProposer

                self.drafter = NgramProposer(self.vllm_config)
            elif self.speculative_config.uses_draft_model():
                self.drafter = DraftModelProposer(
                    vllm_config=self.vllm_config,
                    device=self.device,
                    runner=self,
                )
            elif self.speculative_config.method == "suffix":
                self.drafter = SuffixDecodingProposer(self.vllm_config)
            elif self.speculative_config.use_eagle():
                self.drafter = EagleProposer(self.vllm_config, self.device, self)
                if self.speculative_config.method == "eagle3":
                    self.use_aux_hidden_state_outputs = (
                        self.drafter.eagle3_use_aux_hidden_state
                    )
            elif self.speculative_config.method == "medusa":
                self.drafter = MedusaProposer(
                    vllm_config=self.vllm_config, device=self.device
                )
            else:
                raise ValueError(
                    "Unknown speculative decoding method: "
                    f"{self.speculative_config.method}"
                )
            self.rejection_sampler = RejectionSampler(self.sampler)

        self.num_spec_tokens = 0
        if self.speculative_config:
            self.num_spec_tokens = self.speculative_config.num_speculative_tokens
            draft_config = self.speculative_config.draft_model_config
            if draft_config is not None and draft_config.max_model_len is not None:
                self.effective_drafter_max_model_len = draft_config.max_model_len
            else:
                self.effective_drafter_max_model_len = self.max_model_len

        # Request states.
        self.requests: dict[str, CachedRequestState] = {}
        # NOTE(rob): num_prompt_logprobs only includes reqs
        # that are currently in the prefill phase.
        self.num_prompt_logprobs: dict[str, int] = {}
        self.comm_stream = torch.cuda.Stream()

        # Input Batch
        # NOTE(Chen): Ideally, we should initialize the input batch inside
        # `initialize_kv_cache` based on the kv cache config. However, as in
        # https://github.com/vllm-project/vllm/pull/18298, due to some unknown
        # reasons, we have to initialize the input batch before `load_model`,
        # quantization + weight offloading will fail otherwise. As a temporary
        # solution, we initialize the input batch here, and re-initialize it
        # in `initialize_kv_cache` if the block_sizes here is different from
        # the block_sizes in the kv cache config.
        logits_processors = model_config.logits_processors
        custom_logitsprocs: Sequence[str | type[LogitsProcessor]] = (
            tuple(logits_processors) if logits_processors is not None else ()
        )
        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            # We need to use the encoder length for encoder-decoer
            # because of KV cache for cross-attention.
            max_model_len=max(self.max_model_len, self.max_encoder_len),
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=[self.cache_config.block_size],
            kernel_block_sizes=[self.cache_config.block_size],
            is_spec_decode=bool(self.vllm_config.speculative_config),
            logitsprocs=build_logitsprocs(
                self.vllm_config,
                self.device,
                self.pin_memory,
                self.is_pooling_model,
                custom_logitsprocs,
            ),
            # We currently don't know whether a particular custom logits processor
            # uses output token ids so we set this conservatively.
            logitsprocs_need_output_token_ids=bool(custom_logitsprocs),
            is_pooling_model=self.is_pooling_model,
            cp_kv_cache_interleave_size=self.parallel_config.cp_kv_cache_interleave_size,
        )

        # Separate cuda stream for overlapping transfer of sampled token ids from
        # GPU to CPU when async scheduling is enabled.
        self.async_output_copy_stream: torch.cuda.Stream | None = None
        # cuda event to synchronize use of reused CPU tensors between steps
        # when async scheduling is enabled.
        self.prepare_inputs_event: torch.Event | None = None
        if self.use_async_scheduling:
            self.async_output_copy_stream = torch.cuda.Stream()
            self.prepare_inputs_event = torch.Event()

        # self.cudagraph_batch_sizes sorts in ascending order.
        if (
            self.compilation_config.cudagraph_capture_sizes
            and self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        ):
            self.cudagraph_batch_sizes = sorted(
                self.compilation_config.cudagraph_capture_sizes
            )

        # Cache the device properties.
        self._init_device_properties()

        # Encoder timing registry for observability
        self.encoder_timing_registry: dict[str, EncoderTimingStats] = {}
        self._encoder_timing_lock = threading.Lock()

        # Persistent buffers for CUDA graphs.
        self.input_ids = self._make_buffer(self.max_num_tokens, dtype=torch.int32)
        self.positions = self._make_buffer(self.max_num_tokens, dtype=torch.int64)
        self.query_start_loc = self._make_buffer(
            self.max_num_reqs + 1, dtype=torch.int32
        )
        self.seq_lens = self._make_buffer(self.max_num_reqs, dtype=torch.int32)
        self.encoder_seq_lens = self._make_buffer(self.max_num_reqs, dtype=torch.int32)
        if self.dcp_world_size > 1:
            self.dcp_local_seq_lens = self._make_buffer(
                self.max_num_reqs, dtype=torch.int32
            )
        # Because inputs_embeds may be bfloat16 and we don't need a numpy
        # version of this tensor, avoid a RuntimeError by not creating a
        # numpy buffer.
        self.inputs_embeds = self._make_buffer(
            self.max_num_tokens, self.inputs_embeds_size, dtype=self.dtype, numpy=False
        )
        self.is_token_ids = self._make_buffer(self.max_num_tokens, dtype=torch.bool)
        self.discard_request_mask = self._make_buffer(
            self.max_num_reqs, dtype=torch.bool
        )
        self.num_decode_draft_tokens = self._make_buffer(
            self.max_num_reqs, dtype=torch.int32
        )
        self.num_accepted_tokens = self._make_buffer(
            self.max_num_reqs, dtype=torch.int64
        )

        # Only relevant for multimodal models
        if self.supports_mm_inputs:
            # Double buffer to avoid race condition: previous iteration's async
            # copy may still be reading from CPU while current iteration writes.
            self.is_mm_embed_buffers = [
                self._make_buffer(self.max_num_tokens, dtype=torch.bool),
                self._make_buffer(self.max_num_tokens, dtype=torch.bool),
            ]
            self.is_mm_embed_idx = 0

        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            # NOTE: `mrope_positions` is implemented with one additional dummy
            # position on purpose to make it non-contiguous so that it can work
            # with torch compile.
            # See detailed explanation in https://github.com/vllm-project/vllm/pull/12128#discussion_r1926431923

            # NOTE: When M-RoPE is enabled, position ids are 3D regardless of
            # the modality of inputs. For text-only inputs, each dimension has
            # identical position IDs, making M-RoPE functionally equivalent to
            # 1D-RoPE.
            # See page 5 of https://arxiv.org/abs/2409.12191
            self.mrope_positions = self._make_buffer(
                (3, self.max_num_tokens + 1), dtype=torch.int64
            )

        # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)
        if self.uses_xdrope_dim > 0:
            # Similar to mrope but use assigned dimension number for RoPE, 4 as default.
            self.xdrope_positions = self._make_buffer(
                (self.uses_xdrope_dim, self.max_num_tokens + 1), dtype=torch.int64
            )

        # None in the first PP rank. The rest are set after load_model.
        self.intermediate_tensors: IntermediateTensors | None = None

        # OPTIMIZATION: Cache the tensors rather than creating them every step.
        # Keep in int64 to avoid overflow with long context
        self.arange_np = np.arange(
            max(self.max_num_reqs + 1, self.max_model_len, self.max_num_tokens),
            dtype=np.int64,
        )

        # Layer pairings for cross-layer KV sharing.
        # If an Attention layer `layer_name` is in the keys of this dict, it
        # means this layer will perform attention using the keys and values
        # from the KV cache of `shared_kv_cache_layers[layer_name]`.
        self.shared_kv_cache_layers: dict[str, str] = {}
        self.kv_sharing_fast_prefill_eligible_layers: set[str] = set()

        self.kv_sharing_fast_prefill_logits_indices = None
        if self.cache_config.kv_sharing_fast_prefill:
            self.kv_sharing_fast_prefill_logits_indices = torch.zeros(
                self.max_num_tokens, dtype=torch.int32, device=self.device
            )

        self.uniform_decode_query_len = 1 + self.num_spec_tokens

        # Cudagraph dispatcher for runtime cudagraph dispatching.
        self.cudagraph_dispatcher = CudagraphDispatcher(self.vllm_config)

        self.mm_budget = (
            MultiModalBudget(self.vllm_config, self.mm_registry)
            if self.supports_mm_inputs
            else None
        )

        self.reorder_batch_threshold: int | None = None

        # Attention layers that are only in the KVCacheConfig of the runner
        # (e.g., KV sharing, encoder-only attention), but not in the
        # KVCacheConfig of the scheduler.
        self.runner_only_attn_layers: set[str] = set()

        # Cached outputs.
        self._draft_token_ids: list[list[int]] | torch.Tensor | None = None
        self._draft_token_req_ids: list[str] | None = None
        self.transfer_event = torch.Event()
        self.sampled_token_ids_pinned_cpu = torch.empty(
            (self.max_num_reqs, 1),
            dtype=torch.int64,
            device="cpu",
            pin_memory=self.pin_memory,
        )

        # Pre-allocated tensor for copying valid sampled token counts to CPU,
        # with dedicated stream for overlapping and event for coordination.
        self.valid_sampled_token_count_event: torch.Event | None = None
        self.valid_sampled_token_count_copy_stream: torch.cuda.Stream | None = None
        # We also copy the drafted tokens to the CPU asynchronously,
        # in case we need them for structured outputs.
        self.draft_token_ids_event: torch.Event | None = None
        self.draft_token_ids_copy_stream: torch.cuda.Stream | None = None
        self.valid_sampled_token_count_cpu: torch.Tensor | None = None
        self.draft_token_ids_cpu: torch.Tensor | None = None
        if self.num_spec_tokens:
            self.draft_token_ids_event = torch.Event()
            self.draft_token_ids_copy_stream = torch.cuda.Stream()
            self.draft_token_ids_cpu = torch.empty(
                (self.max_num_reqs, self.num_spec_tokens),
                dtype=torch.int64,
                device="cpu",
                pin_memory=self.pin_memory,
            )
            if self.use_async_scheduling:
                self.valid_sampled_token_count_event = torch.Event()
                self.valid_sampled_token_count_copy_stream = torch.cuda.Stream()
                self.valid_sampled_token_count_cpu = torch.empty(
                    self.max_num_reqs,
                    dtype=torch.int64,
                    device="cpu",
                    pin_memory=self.pin_memory,
                )

        # Ephemeral state transferred between execute_model() and sample_tokens().
        self.execute_model_state: ExecuteModelState | None = None
        self.kv_connector_output: KVConnectorOutput | None = None
        self.mamba_state_idx: dict[str, int] = {}
        self.layerwise_nvtx_hooks_registered = False

    def update_max_model_len(self, max_model_len: int) -> None:
        self.max_model_len = max_model_len
        if self.speculative_config:
            draft_config = self.speculative_config.draft_model_config
            if draft_config is None or draft_config.max_model_len is None:
                self.effective_drafter_max_model_len = self.max_model_len

    def reset_mm_cache(self) -> None:
        """
        Clear the multi-modal cache that was used during profiling,
        but no longer needed during inference.
        """
        if self.mm_budget:
            self.mm_budget.reset_cache()

    def reset_encoder_cache(self) -> None:
        """Clear the GPU-side encoder cache storing vision embeddings.

        This should be called when model weights are updated to ensure
        stale embeddings computed with old weights are not reused.
        """
        self.encoder_cache.clear()

    @torch.inference_mode()
    def init_fp8_kv_scales(self) -> None:
        """
        Re-initialize the KV cache and FP8 scales after waking from sleep.
        1. Zero out the KV cache tensors to remove garbage data from re-allocation.
        2. Reset Attention layer scaling factors (_k_scale, _v_scale) to 1.0.
          If these are left at 0.0 (default after wake_up), all KV cache values
          become effectively zero, causing gibberish output.
        """
        if not self.cache_config.cache_dtype.startswith("fp8"):
            return

        kv_caches = getattr(self, "kv_caches", [])
        for cache_tensor in kv_caches:
            if cache_tensor is not None:
                cache_tensor.zero_()

        k_attr_names = ("_k_scale", "k_scale")
        v_attr_names = ("_v_scale", "v_scale")

        attn_layers = self.compilation_config.static_forward_context
        for name, module in attn_layers.items():
            if isinstance(module, (Attention, MLAAttention)):
                # TODO: Generally, scale is 1.0 if user uses on-the-fly fp8
                # kvcache quant. However, to get better accuracy, compression
                # frameworks like llm-compressors allow users to tune the
                # scale. We may need to restore the specific calibrated scales
                # here in the future.
                k_scale_val, v_scale_val = 1.0, 1.0

                # Processing K Scale
                for attr in k_attr_names:
                    if hasattr(module, attr):
                        param = getattr(module, attr)
                        if isinstance(param, torch.Tensor):
                            param.fill_(k_scale_val)

                # Processing V Scale
                for attr in v_attr_names:
                    if hasattr(module, attr):
                        param = getattr(module, attr)
                        if isinstance(param, torch.Tensor):
                            param.fill_(v_scale_val)

    def _get_positions(self, num_tokens: Any):
        if isinstance(num_tokens, int):
            if self.uses_mrope:
                return self.mrope_positions.gpu[:, :num_tokens]
            if self.uses_xdrope_dim > 0:
                return self.xdrope_positions.gpu[:, :num_tokens]
            return self.positions.gpu[:num_tokens]
        else:
            if self.uses_mrope:
                return self.mrope_positions.gpu[:, num_tokens]
            if self.uses_xdrope_dim > 0:
                return self.xdrope_positions.gpu[:, num_tokens]
            return self.positions.gpu[num_tokens]

    def _make_buffer(
        self, *size: int | torch.SymInt, dtype: torch.dtype, numpy: bool = True
    ) -> CpuGpuBuffer:
        return CpuGpuBuffer(
            *size,
            dtype=dtype,
            device=self.device,
            pin_memory=self.pin_memory,
            with_numpy=numpy,
        )

    def _init_model_kwargs(self):
        model_kwargs = dict[str, Any]()

        if not self.is_pooling_model:
            return model_kwargs

        num_reqs = self.input_batch.num_reqs
        pooling_params = self.input_batch.get_pooling_params()

        token_type_id_requests = dict[int, Any]()
        for i, param in enumerate(pooling_params):
            if (
                param.extra_kwargs is not None
                and (token_types := param.extra_kwargs.get("compressed_token_type_ids"))
                is not None
            ):
                token_type_id_requests[i] = token_types

        if len(token_type_id_requests) == 0:
            return model_kwargs

        seq_lens = self.seq_lens.gpu[:num_reqs]
        token_type_ids = []

        for i in range(num_reqs):
            pos = token_type_id_requests.get(i, seq_lens[i])
            ids = (torch.arange(seq_lens[i]) >= pos).int()
            token_type_ids.append(ids)

        model_kwargs["token_type_ids"] = torch.concat(token_type_ids).to(
            device=self.device
        )
        return model_kwargs

    def _may_reorder_batch(self, scheduler_output: "SchedulerOutput") -> None:
        """
        Update the order of requests in the batch based on the attention
        backend's needs. For example, some attention backends (namely MLA) may
        want to separate requests based on if the attention computation will be
        compute-bound or memory-bound.

        Args:
            scheduler_output: The scheduler output.
        """
        # Attention free models have zero kv_cache_goups, however models
        # like Mamba are also attention free but use the kv_cache for
        # keeping its internal state. This is why we check the number
        # of kv_cache groups instead of solely checking
        # for self.model_config.is_attention_free.
        if len(self.kv_cache_config.kv_cache_groups) == 0:
            return

        if self.reorder_batch_threshold is not None:
            reorder_batch_to_split_decodes_and_prefills(
                self.input_batch,
                scheduler_output,
                decode_threshold=self.reorder_batch_threshold,
            )

    # Note: used for model runner override.
    def _init_device_properties(self) -> None:
        """Initialize attributes from torch.cuda.get_device_properties"""
        self.device_properties = torch.cuda.get_device_properties(self.device)
        self.num_sms = self.device_properties.multi_processor_count

    # Note: used for model runner override.
    def _sync_device(self) -> None:
        torch.cuda.synchronize()

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        """Update the cached states and the persistent batch with the scheduler
        output.

        The updated states are used by the `_prepare_inputs` function to create
        the input GPU tensors for the model.

        The SamplingMetadata is updated and copied to the GPU if there is a
        new/resumed/paused/finished request in the batch.
        """
        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
            self.num_prompt_logprobs.pop(req_id, None)
        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
        for req_id in scheduler_output.finished_req_ids:
            self.input_batch.remove_request(req_id)

        # Free the cached encoder outputs.
        for mm_hash in scheduler_output.free_encoder_mm_hashes:
            self.encoder_cache.pop(mm_hash, None)

        # Remove the unscheduled requests from the persistent batch.
        # NOTE(woosuk): The unscheduled requests are either preempted requests
        # or running requests that are not scheduled in this step. We remove
        # them from the persistent batch but keep their cached states since
        # they will be scheduled again sometime in the future.
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
        # NOTE(zhuohan): cached_req_ids and resumed_req_ids are usually disjoint,
        # so `(scheduled_req_ids - resumed_req_ids) == scheduled_req_ids` holds
        # apart from the forced-preemption case in reset_prefix_cache. And in
        # that case we include the resumed_req_ids in the unscheduled set so
        # that they get cleared from the persistent batch before being re-scheduled
        # in the normal resumed request path.
        unscheduled_req_ids = cached_req_ids - (scheduled_req_ids - resumed_req_ids)
        # NOTE(woosuk): The persistent batch optimization assumes that
        # consecutive batches contain mostly the same requests. If batches
        # have low request overlap (e.g., alternating between two distinct
        # sets of requests), this optimization becomes very inefficient.
        for req_id in unscheduled_req_ids:
            self.input_batch.remove_request(req_id)

        reqs_to_add: list[CachedRequestState] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            if req_id in self.requests:
                # For streaming case only.
                req_state = self._update_streaming_request(req_id, new_req_data)
                reqs_to_add.append(req_state)
                continue

            sampling_params = new_req_data.sampling_params
            pooling_params = new_req_data.pooling_params

            if (
                sampling_params
                and sampling_params.sampling_type == SamplingType.RANDOM_SEED
            ):
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            if self.is_pooling_model:
                assert pooling_params is not None
                task = pooling_params.task
                assert task is not None, "You did not set `task` in the API"

                model = cast(VllmModelForPooling, self.get_model())
                to_update = model.pooler.get_pooling_updates(task)
                to_update.apply(pooling_params)

            req_state = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                prompt_embeds=new_req_data.prompt_embeds,
                mm_features=new_req_data.mm_features,
                sampling_params=sampling_params,
                pooling_params=pooling_params,
                generator=generator,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
            )
            self.requests[req_id] = req_state

            if sampling_params and sampling_params.prompt_logprobs is not None:
                self.num_prompt_logprobs[req_id] = (
                    self.input_batch.vocab_size
                    if sampling_params.prompt_logprobs == -1
                    else sampling_params.prompt_logprobs
                )

            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            if self.uses_mrope:
                self._init_mrope_positions(req_state)

            # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)
            if self.uses_xdrope_dim > 0:
                self._init_xdrope_positions(req_state)

            reqs_to_add.append(req_state)

        # Update the states of the running/resumed requests.
        is_last_rank = get_pp_group().is_last_rank
        req_data = scheduler_output.scheduled_cached_reqs
        scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens

        # Wait until valid_sampled_tokens_count is copied to cpu,
        # then use it to update actual num_computed_tokens of each request.
        valid_sampled_token_count = self._get_valid_sampled_token_count()

        for i, req_id in enumerate(req_data.req_ids):
            req_state = self.requests[req_id]
            num_computed_tokens = req_data.num_computed_tokens[i]
            new_block_ids = req_data.new_block_ids[i]
            resumed_from_preemption = req_id in req_data.resumed_req_ids
            num_output_tokens = req_data.num_output_tokens[i]
            req_index = self.input_batch.req_id_to_index.get(req_id)

            if req_state.prev_num_draft_len and self.use_async_scheduling:
                # prev_num_draft_len is used in async scheduling mode with
                # spec decode. it indicates if need to update num_computed_tokens
                # of the request. for example:
                # fist step: num_computed_tokens = 0, spec_tokens = [],
                # prev_num_draft_len = 0.
                # second step: num_computed_tokens = 100(prompt lenth),
                # spec_tokens = [a,b], prev_num_draft_len = 0.
                # third step: num_computed_tokens = 100 + 2, spec_tokens = [c,d],
                # prev_num_draft_len = 2.
                # num_computed_tokens in first step and second step does't contain
                # the spec tokens length, but in third step it contains the
                # spec tokens length. we only need to update num_computed_tokens
                # when prev_num_draft_len > 0.
                if req_index is None:
                    req_state.prev_num_draft_len = 0
                else:
                    assert self.input_batch.prev_req_id_to_index is not None
                    prev_req_index = self.input_batch.prev_req_id_to_index[req_id]
                    num_accepted = valid_sampled_token_count[prev_req_index] - 1
                    num_rejected = req_state.prev_num_draft_len - num_accepted
                    num_computed_tokens -= num_rejected
                    req_state.output_token_ids.extend([-1] * num_accepted)

            # Update the cached states.
            req_state.num_computed_tokens = num_computed_tokens

            if not is_last_rank:
                if not req_data.new_token_ids:
                    # Async scheduled PP: Sampled tokens propagated via GPU broadcast.
                    new_token_ids: list[int] = []
                else:
                    # Non-async scheduling with PP: The scheduler sends
                    # sampled token ids back because there's no direct communication
                    # between the first-stage worker and the last-stage worker.
                    new_token_ids = req_data.new_token_ids[i]
                    # Add the sampled token(s) from the previous step (if any).
                    # This doesn't include "unverified" tokens like spec tokens.
                    num_new_tokens = (
                        num_computed_tokens + len(new_token_ids) - req_state.num_tokens
                    )
                    if num_new_tokens == 1:
                        # Avoid slicing list in most common case.
                        req_state.output_token_ids.append(new_token_ids[-1])
                    elif num_new_tokens > 0:
                        req_state.output_token_ids.extend(
                            new_token_ids[-num_new_tokens:]
                        )
            elif num_output_tokens < len(req_state.output_token_ids):
                # Some output tokens were discarded due to a sync-KV-load
                # failure. Align the cached state.
                del req_state.output_token_ids[num_output_tokens:]
                if req_index is not None:
                    end_idx = (
                        self.input_batch.num_prompt_tokens[req_index]
                        + num_output_tokens
                    )
                    self.input_batch.num_tokens_no_spec[req_index] = end_idx

            # Update the block IDs.
            if not resumed_from_preemption:
                if new_block_ids is not None:
                    # Append the new blocks to the existing block IDs.
                    for block_ids, new_ids in zip(req_state.block_ids, new_block_ids):
                        block_ids.extend(new_ids)
            else:
                assert req_index is None
                assert new_block_ids is not None
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = new_block_ids

            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.

                if self.use_async_scheduling and num_output_tokens > 0:
                    # We must recover the output token ids for resumed requests in the
                    # async scheduling case, so that correct input_ids are obtained.
                    resumed_token_ids = req_data.all_token_ids[req_id]
                    req_state.output_token_ids = resumed_token_ids[-num_output_tokens:]

                reqs_to_add.append(req_state)
                continue

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[req_index] = num_computed_tokens
            if new_block_ids is not None:
                self.input_batch.block_table.append_row(new_block_ids, req_index)

            # For the last rank, we don't need to update the token_ids_cpu
            # because the sampled tokens are already cached.
            if not is_last_rank:
                # Add new_token_ids to token_ids_cpu.
                start_token_index = num_computed_tokens
                end_token_index = num_computed_tokens + len(new_token_ids)
                self.input_batch.token_ids_cpu[
                    req_index, start_token_index:end_token_index
                ] = new_token_ids
                self.input_batch.num_tokens_no_spec[req_index] = end_token_index

            # Add spec_token_ids to token_ids_cpu.
            self.input_batch.update_req_spec_token_ids(req_state, scheduled_spec_tokens)

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        for request in reqs_to_add:
            self.input_batch.add_request(request)
            self.input_batch.update_req_spec_token_ids(request, scheduled_spec_tokens)

        # Condense the batched states if there are gaps left by removed requests
        self.input_batch.condense()
        # Allow attention backend to reorder the batch, potentially
        self._may_reorder_batch(scheduler_output)
        # Refresh batch metadata with any pending updates.
        self.input_batch.refresh_metadata()

    def _update_states_after_model_execute(
        self, output_token_ids: torch.Tensor, scheduler_output: "SchedulerOutput"
    ) -> None:
        """Update the cached states after model execution.

        This is used for MTP/EAGLE for hybrid models, as in linear attention,
        only the last token's state is kept. In MTP/EAGLE, for draft tokens
        the state are kept util we decide how many tokens are accepted for
        each sequence, and a shifting is done during the next iteration
        based on the number of accepted tokens.
        """
        if not self.speculative_config or not self.model_config.is_hybrid:
            return

        # Find the number of accepted tokens for each sequence.
        num_accepted_tokens = (
            (
                torch.cat(
                    [
                        output_token_ids,
                        torch.full(
                            (output_token_ids.size(0), 1),
                            -1,
                            device=output_token_ids.device,
                        ),
                    ],
                    dim=1,
                )
                == -1
            )
            .int()
            .argmax(-1)
        )
        _стабилен = (not getattr(scheduler_output, "scheduled_new_reqs", None)
                     and not getattr(scheduler_output, "finished_req_ids", None))
        if _БЕЗ_СИНХРО_ПРИЁМКИ and _стабилен:
            self._приёмка_gpu = num_accepted_tokens
            num_accepted_tokens = None
        else:
            self._приёмка_gpu = None
            num_accepted_tokens = num_accepted_tokens.cpu().numpy()
            for i, num_tokens in enumerate(num_accepted_tokens):
                self.input_batch.num_accepted_tokens_cpu[i] = num_tokens
        # [СЧЁТЧИК ОБМЕНА, 07.09] Читается ЗДЕСЬ, потому что питон движка исполняется
        # каждый шаг (в граф уходит только проход сети). Счётчик живёт В ЯДРЕ и потому
        # считает и повторы графа -- со стороны питона это единственный способ.
        if os.environ.get("FA2SM70_GDN_GUARD", "0") == "1":
            # [ПРОВЕРКА ГРАНИЦ СОСТОЯНИЯ] Ядро-проверка идёт В ГРАФЕ и считает
            # нарушения на каждом повторе; здесь их только читают. Первое нарушение
            # запоминается целиком -- оно и есть улика к скрытому падению.
            _ш2 = getattr(self, "_гр_шагов", 0) + 1
            self._гр_шагов = _ш2
            if _ш2 % 100 == 0:
                try:
                    import fa2_sm70._ext as _е2
                    _н, _зн, _поз, _пр, _вид = _е2.gdn_ext().gdn_check_read(False)
                    # ПЕЧАТЬ БЕЗУСЛОВНАЯ. Прошлый прогон промолчал целиком, и его
                    # молчание нельзя было прочесть как "нарушений нет": живость
                    # прибора обязана доказываться числом, а не предполагаться.
                    print(f"[GDN ГРАНИЦЫ] шаг {_ш2}: нарушений {_н}"
                          + (f" ПЕРВОЕ значение={_зн} позиция={_поз} предел={_пр}"
                             if _н else " (чисто)"),
                          file=__import__("sys").stderr, flush=True)
                except Exception as _e3:
                    print(f"[GDN ГРАНИЦЫ] чтение не удалось: {_e3}",
                          file=__import__("sys").stderr, flush=True)
        if os.environ.get("FA2SM70_AR2_COUNT", "0") == "1":
            _ш = getattr(self, "_ar2_шагов", 0) + 1
            self._ar2_шагов = _ш
            if _ш % 200 == 0:
                try:
                    import fa2_sm70._ext as _е
                    _c, _e2 = _е.exch_ext().ar2_counter_read(False)
                    print(f"[AR2 ЯДРО] шагов {_ш} запусков {_c} = {_c/_ш:.1f} на шаг",
                          file=__import__("sys").stderr, flush=True)
                except Exception as _ex:
                    print(f"[AR2 ЯДРО] чтение не удалось: {_ex}",
                          file=__import__("sys").stderr, flush=True)
        # [АВТОСПЕКУЛЯЦИЯ] Скользящее среднее tau. Значение УЖЕ на процессоре
        # (строкой выше его и так снимают), поэтому лишней синхронизации нет.
        try:
            if len(num_accepted_tokens):
                _т = float(sum(num_accepted_tokens)) / len(num_accepted_tokens)
                from vllm.v1.spec_decode import dflash2 as _дф
                _дф.ПРИЁМКА_СРЕДН[0] = (0.95 * _дф.ПРИЁМКА_СРЕДН[0] + 0.05 * _т
                                        if _дф.ПРИЁМКА_СРЕДН[0] > 0 else _т)
        except Exception:
            pass
        if self.cache_config.mamba_cache_mode == "align":
            mamba_utils.postprocess_mamba(
                scheduler_output,
                self.kv_cache_config,
                self.input_batch,
                self.requests,
                self.mamba_state_idx,
                self.compilation_config.static_forward_context,
                self.model.get_mamba_state_copy_func(),
            )

    def _update_streaming_request(
        self, req_id: str, new_req_data: NewRequestData
    ) -> CachedRequestState:
        """Updates streaming session request from `scheduled_new_reqs`.

        Removes the request from InputBatch (if present), updates the cached
        state, and prepares it for re-addition to the batch.

        NOTE: prompt_token_ids includes intermediate output tokens - tokens
        previously generated but now are input context (part of the prompt).
        """
        self.input_batch.remove_request(req_id)
        req_state = self.requests[req_id]

        req_state.prompt_token_ids = new_req_data.prompt_token_ids
        req_state.mm_features = new_req_data.mm_features
        req_state.prompt_embeds = new_req_data.prompt_embeds
        req_state.sampling_params = new_req_data.sampling_params
        req_state.pooling_params = new_req_data.pooling_params
        req_state.block_ids = new_req_data.block_ids
        req_state.num_computed_tokens = new_req_data.num_computed_tokens
        req_state.num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
            req_state.prompt_token_ids, req_state.prompt_embeds
        )

        # Clear `output_token_ids` as previous output tokens are now part of
        # `prompt_token_ids`.
        req_state.output_token_ids.clear()

        if self.uses_mrope:
            self._init_mrope_positions(req_state)

        return req_state

    def _init_mrope_positions(self, req_state: CachedRequestState):
        model = self.get_model()
        assert supports_mrope(model), "M-RoPE support is not implemented."
        assert req_state.prompt_token_ids is not None, (
            "M-RoPE requires prompt_token_ids to be available."
        )
        mrope_model = cast(SupportsMRoPE, model)

        req_state.mrope_positions, req_state.mrope_position_delta = (
            mrope_model.get_mrope_input_positions(
                req_state.prompt_token_ids,
                req_state.mm_features,
            )
        )

    def _init_xdrope_positions(self, req_state: CachedRequestState):
        model = self.get_model()
        xdrope_model = cast(SupportsXDRoPE, model)
        assert req_state.prompt_token_ids is not None, (
            "XD-RoPE requires prompt_token_ids to be available."
        )
        assert supports_xdrope(model), "XD-RoPE support is not implemented."

        req_state.xdrope_positions = xdrope_model.get_xdrope_input_positions(
            req_state.prompt_token_ids,
            req_state.mm_features,
        )

    def _extract_mm_kwargs(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> BatchedTensorInputs:
        if not scheduler_output or not self.is_multimodal_raw_input_only_model:
            return {}

        mm_kwargs = list[tuple[str, MultiModalKwargsItem]]()
        for req in scheduler_output.scheduled_new_reqs:
            for feature in req.mm_features:
                if feature.data is not None:
                    mm_kwargs.append((feature.modality, feature.data))

        # Input all modalities at once
        mm_kwargs_combined: BatchedTensorInputs = {}
        for _, _, mm_kwargs_group in group_mm_kwargs_by_modality(
            mm_kwargs,
            device=self.device,
            pin_memory=self.pin_memory,
        ):
            mm_kwargs_combined.update(mm_kwargs_group)

        return mm_kwargs_combined

    def _dummy_mm_kwargs(self, num_seqs: int) -> BatchedTensorInputs:
        if not self.is_multimodal_raw_input_only_model:
            return {}

        mm_budget = self.mm_budget
        assert mm_budget is not None

        dummy_modality = mm_budget.get_modality_with_max_tokens()
        return self._get_mm_dummy_batch(dummy_modality, num_seqs)

    def _get_cumsum_and_arange(
        self,
        num_tokens: np.ndarray,
        cumsum_dtype: np.dtype | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the cumulative sum and batched arange of the given array.
        # E.g., [2, 5, 3] -> ([2, 7, 10], [0, 1, 0, 1, 2, 3, 4, 0, 1, 2])
        # Equivalent to but faster than:
        # np.concatenate([np.arange(n) for n in num_tokens])
        """
        # Step 1. [2, 5, 3] -> [2, 7, 10]
        cu_num_tokens = np.cumsum(num_tokens, dtype=cumsum_dtype)
        total_num_tokens = cu_num_tokens[-1]
        # Step 2. [2, 7, 10] -> [0, 0, 2, 2, 2, 2, 2, 7, 7, 7]
        cumsums_offsets = np.repeat(cu_num_tokens - num_tokens, num_tokens)
        # Step 3. [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        arange = self.arange_np[:total_num_tokens] - cumsums_offsets

        return cu_num_tokens, arange

    def _prepare_input_ids(
        self,
        scheduler_output: "SchedulerOutput",
        total_num_scheduled_tokens: int,
        cu_num_tokens: np.ndarray,
    ) -> None:
        """Prepare the input IDs for the current batch.

        Carefully handles the `prev_sampled_token_ids` which can be cached
        from the previous engine iteration, in which case those tokens on the
        GPU need to be copied into the corresponding slots into input_ids."""

        if self.input_batch.prev_sampled_token_ids is None:
            # Normal scheduling case
            self.input_ids.copy_to_gpu(total_num_scheduled_tokens)
            if self.enable_prompt_embeds:
                self.inputs_embeds.copy_to_gpu(total_num_scheduled_tokens)
                self.is_token_ids.copy_to_gpu(total_num_scheduled_tokens)
            return

        # Async scheduling case, where some decode requests from the previous
        # iteration won't have entries in input_ids_cpu and need to be copied
        # on the GPU from prev_sampled_token_ids.
        prev_req_id_to_index = self.input_batch.prev_req_id_to_index
        assert prev_req_id_to_index is not None
        sample_flattened_indices: list[int] = []
        spec_flattened_indices: list[int] = []
        prev_common_req_indices: list[int] = []
        prev_draft_token_indices: list[int] = []
        indices_match = True
        max_flattened_index = -1
        total_num_spec_tokens = 0
        scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens

        for req_id, cur_index in self.input_batch.req_id_to_index.items():
            if (prev_index := prev_req_id_to_index.get(req_id)) is not None:
                prev_common_req_indices.append(prev_index)
                # We need to compute the flattened input_ids index of the
                # last token in each common request.
                draft_len = len(scheduled_spec_tokens.get(req_id, ()))
                total_num_spec_tokens += draft_len
                flattened_index = cu_num_tokens[cur_index].item() - 1
                # example: cu_num_tokens = [2, 5, 8], draft_tokens = [1, 2, 2]
                # sample_flattened_indices = [0, 2, 5]
                # spec_flattened_indices = [1,   3, 4,    6, 7]
                sample_flattened_indices.append(flattened_index - draft_len)
                spec_flattened_indices.extend(
                    range(flattened_index - draft_len + 1, flattened_index + 1)
                )
                start = prev_index * self.num_spec_tokens
                # prev_draft_token_indices is used to find which draft_tokens_id
                # should be copied to input_ids
                # example: prev draft_tokens_id [[1,2], [3,4], [5, 6]]
                # flatten draft_tokens_id [1,2,3,4,5,6]
                # draft_len of each request [1, 2, 1]
                # then prev_draft_token_indices is [0,   2, 3,   4]
                prev_draft_token_indices.extend(range(start, start + draft_len))
                indices_match &= prev_index == flattened_index
                max_flattened_index = max(max_flattened_index, flattened_index)
        num_commmon_tokens = len(sample_flattened_indices)
        total_without_spec = total_num_scheduled_tokens - total_num_spec_tokens
        if num_commmon_tokens < total_without_spec:
            # If not all requests are decodes from the last iteration,
            # We need to copy the input_ids_cpu to the GPU first.
            self.input_ids.copy_to_gpu(total_num_scheduled_tokens)
            if self.enable_prompt_embeds:
                self.inputs_embeds.copy_to_gpu(total_num_scheduled_tokens)
                self.is_token_ids.copy_to_gpu(total_num_scheduled_tokens)
        if num_commmon_tokens == 0:
            # No requests in common with the previous iteration
            # So input_ids.cpu will have all the input ids.
            return
        if indices_match and max_flattened_index == (num_commmon_tokens - 1):
            # Common-case optimization: the batch is unchanged
            # and no reordering happened.
            # The indices are both the same permutation of 0..N-1 so
            # we can copy directly using a single slice.
            self.input_ids.gpu[:num_commmon_tokens].copy_(
                self.input_batch.prev_sampled_token_ids[:num_commmon_tokens, 0],
                non_blocking=True,
            )
            if self.enable_prompt_embeds:
                self.is_token_ids.gpu[:num_commmon_tokens] = True
            return
        # Upload the index tensors asynchronously so the scatter can be non-blocking.
        sampled_tokens_index_tensor = torch.tensor(
            sample_flattened_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)
        prev_common_req_indices_tensor = torch.tensor(
            prev_common_req_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)
        self.input_ids.gpu.scatter_(
            dim=0,
            index=sampled_tokens_index_tensor,
            src=self.input_batch.prev_sampled_token_ids[
                prev_common_req_indices_tensor, 0
            ],
        )

        # Scatter the draft tokens after the sampled tokens are scattered.
        if self._draft_token_ids is None or not spec_flattened_indices:
            return
        # [ПЕРЕХОД НА БЕЗ-СПЕКУЛЯЦИИ, 07.09] Автопереключатель приёмки может отдать
        # ПУСТОЙ черновик, а планировщик на этом шаге ещё держит индексы прошлого.
        # Индексация пустого тензора роняет воркер (IndexError: size 0). Переход
        # между режимами обязан быть безболезненным: нечего рассыпать -- выходим.
        if self._draft_token_ids.numel() == 0:
            return

        assert isinstance(self._draft_token_ids, torch.Tensor)
        draft_tokens_index_tensor = torch.tensor(
            spec_flattened_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)
        prev_draft_token_indices_tensor = torch.tensor(
            prev_draft_token_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)

        # because input_ids dtype is torch.int32,
        # so convert draft_token_ids to torch.int32 here.
        draft_token_ids = self._draft_token_ids.to(dtype=torch.int32)

        self.input_ids.gpu.scatter_(
            dim=0,
            index=draft_tokens_index_tensor,
            src=draft_token_ids.flatten()[prev_draft_token_indices_tensor],
        )

    def _get_encoder_seq_lens(
        self,
        num_scheduled_tokens: dict[str, int],
        kv_cache_spec: KVCacheSpec,
        num_reqs: int,
        for_cudagraph_capture: bool = False,
    ) -> tuple[torch.Tensor | None, np.ndarray | None]:
        if not isinstance(kv_cache_spec, CrossAttentionSpec):
            return None, None

        # Zero out buffer for padding requests that are not actually scheduled (CGs)
        self.encoder_seq_lens.np[:num_reqs] = 0

        # Build encoder_seq_lens array mapping request indices to
        # encoder lengths for inputs scheduled in this batch
        for req_id in num_scheduled_tokens:
            req_index = self.input_batch.req_id_to_index[req_id]
            req_state = self.requests[req_id]
            if req_state.mm_features is None:
                self.encoder_seq_lens.np[req_index] = 0
                continue

            # Get the total number of encoder input tokens for running encoder requests
            # whether encoding is finished or not so that cross-attention knows how
            # many encoder tokens to attend to.
            encoder_input_tokens = sum(
                feature.mm_position.length for feature in req_state.mm_features
            )
            self.encoder_seq_lens.np[req_index] = encoder_input_tokens
        if for_cudagraph_capture:
            # During CUDA graph capture, we need to use realistic encoder lengths
            # so that max_seqlen_k is captured with the correct value.
            max_encoder_len = getattr(
                self.model_config.hf_config,
                "max_source_positions",
                self.max_encoder_len,
            )
            self.encoder_seq_lens.np[:num_reqs] = max_encoder_len

        self.encoder_seq_lens.copy_to_gpu(num_reqs)
        encoder_seq_lens = self.encoder_seq_lens.gpu[:num_reqs]
        encoder_seq_lens_cpu = self.encoder_seq_lens.np[:num_reqs]

        return encoder_seq_lens, encoder_seq_lens_cpu

    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
        num_scheduled_tokens: np.ndarray,
    ) -> tuple[
        torch.Tensor,
        SpecDecodeMetadata | None,
    ]:
        """
        :return: tuple[
            logits_indices, spec_decode_metadata,
        ]
        """
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        self.input_batch.block_table.commit_block_table(num_reqs)

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)

        # cu_num_tokens: [2, 5, 3] -> [2, 7, 10]
        # arange: [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        cu_num_tokens, arange = self._get_cumsum_and_arange(num_scheduled_tokens)

        # Get positions.
        positions_np = self.positions.np[:total_num_scheduled_tokens]
        np.add(
            self.input_batch.num_computed_tokens_cpu[req_indices],
            arange,
            out=positions_np,
        )

        # Calculate M-RoPE positions.
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            self._calc_mrope_positions(scheduler_output)

        # Calculate XD-RoPE positions.
        # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)
        if self.uses_xdrope_dim > 0:
            self._calc_xdrope_positions(scheduler_output)

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = (
            positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]
        )
        token_indices_tensor = torch.from_numpy(token_indices)

        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        torch.index_select(
            self.input_batch.token_ids_cpu_tensor.flatten(),
            0,
            token_indices_tensor,
            out=self.input_ids.cpu[:total_num_scheduled_tokens],
        )
        if self.enable_prompt_embeds:
            is_token_ids = self.input_batch.is_token_ids_tensor.flatten()
            torch.index_select(
                is_token_ids,
                0,
                token_indices_tensor,
                out=self.is_token_ids.cpu[:total_num_scheduled_tokens],
            )

        # Because we did not pre-allocate a massive prompt_embeds CPU tensor on
        # the InputBatch, we need to fill in the prompt embeds into the expected
        # spots in the GpuModelRunner's pre-allocated prompt_embeds tensor.
        if self.input_batch.req_prompt_embeds:
            output_idx = 0
            for req_idx in range(num_reqs):
                num_sched = num_scheduled_tokens[req_idx]

                # Skip if this request doesn't have embeddings
                if req_idx not in self.input_batch.req_prompt_embeds:
                    output_idx += num_sched
                    continue

                # Skip if no tokens scheduled
                if num_sched <= 0:
                    output_idx += num_sched
                    continue

                req_embeds = self.input_batch.req_prompt_embeds[req_idx]
                start_pos = self.input_batch.num_computed_tokens_cpu[req_idx]

                # Skip if trying to read beyond available embeddings
                if start_pos >= req_embeds.shape[0]:
                    output_idx += num_sched
                    continue

                # Copy available embeddings
                end_pos = start_pos + num_sched
                actual_end = min(end_pos, req_embeds.shape[0])
                actual_num_sched = actual_end - start_pos

                if actual_num_sched > 0:
                    self.inputs_embeds.cpu[
                        output_idx : output_idx + actual_num_sched
                    ].copy_(req_embeds[start_pos:actual_end])

                output_idx += num_sched

        self.input_batch.block_table.compute_slot_mapping(req_indices, positions_np)
        self.input_batch.block_table.commit_slot_mapping(total_num_scheduled_tokens)
        if _ГРАН_СЛОТ:
            self._гран_слот_проверить(req_indices, positions_np, num_reqs)

        # [ДЕРЕВО, ШАГ 4] ПОЗИЦИИ ВЕТВИ B -- ПОСЛЕ РАСЧЁТА СЛОТОВ, И ЭТО СУЩЕСТВЕННО.
        # Строка W+1+j физически стоит на позиции база+W+1+j, а логически обязана стоять там
        # же, где её двойник a(j) -- на база+1+j: обе ветви расходятся ИЗ ОДНОЙ точки. RoPE
        # берёт позицию, поэтому без правки b0 считается не там, где a0, и предсказания цели в
        # строках ветви B неверны -- приёмка сверяется с испорченным argmax.
        # Правка идёт ПОСЛЕ compute_slot_mapping намеренно: слоты обязаны остаться
        # ФИЗИЧЕСКИМИ (иначе обе ветви пишут KV в одни слоты и побеждает случайная), а RoPE --
        # получить логические. Разведение этих двух ролей позиции и есть весь приём.
        if _ДЕРЕВО_W > 0 and not _ДЕРЕВО_БЕЗ_ПОЗ:
            _qд = 2 * _ДЕРЕВО_W + 1
            if (total_num_scheduled_tokens % _qд == 0
                    and int(num_scheduled_tokens.max()) == _qд
                    and int(num_scheduled_tokens.min()) == _qд):
                _pv = positions_np.reshape(-1, _qд)
                _pv[:, _ДЕРЕВО_W + 1:] = _pv[:, 1:_ДЕРЕВО_W + 1]

        # Prepare the attention metadata.
        self.query_start_loc.np[0] = 0
        self.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
        # Note: pad query_start_loc to be non-decreasing, as kernels
        # like FlashAttention requires that
        self.query_start_loc.np[num_reqs + 1 :].fill(cu_num_tokens[-1])
        self.query_start_loc.copy_to_gpu()
        query_start_loc = self.query_start_loc.gpu[: num_reqs + 1]

        self.seq_lens.np[:num_reqs] = (
            self.input_batch.num_computed_tokens_cpu[:num_reqs] + num_scheduled_tokens
        )
        # Fill unused with 0 for full cuda graph mode.
        self.seq_lens.np[num_reqs:].fill(0)
        self.seq_lens.copy_to_gpu()

        num_tokens = [self.requests[r].num_tokens for r in self.input_batch.req_ids]
        num_tokens_np = np.array(num_tokens, dtype=np.int32)

        # Record which requests should not be sampled,
        # so that we could clear the sampled tokens before returning
        self.discard_request_mask.np[:num_reqs] = (
            self.seq_lens.np[:num_reqs] < num_tokens_np
        )
        self.discard_request_mask.copy_to_gpu(num_reqs)

        # Copy the tensors to the GPU.
        self._prepare_input_ids(
            scheduler_output,
            total_num_scheduled_tokens,
            cu_num_tokens,
        )

        if self.uses_mrope:
            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            self.mrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
                self.mrope_positions.cpu[:, :total_num_scheduled_tokens],
                non_blocking=True,
            )
        elif self.uses_xdrope_dim > 0:
            # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)
            self.xdrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
                self.xdrope_positions.cpu[:, :total_num_scheduled_tokens],
                non_blocking=True,
            )
        else:
            # Common case (1D positions)
            self.positions.copy_to_gpu(total_num_scheduled_tokens)

        use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
            # NOTE(woosuk): Due to chunked prefills, the batch may contain
            # partial requests. While we should not sample any token
            # from these partial requests, we do so for simplicity.
            # We will ignore the sampled tokens from the partial requests.
            # TODO: Support prompt logprobs.
            logits_indices = query_start_loc[1:] - 1
            spec_decode_metadata = None
            num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)
        else:
            # Get the number of draft tokens for each request.
            # Iterate over the dictionary rather than all requests since not all
            # requests have draft tokens.
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            # For chunked prefills, use -1 as mask rather than 0, as guided
            # decoding may rollback speculative tokens.
            num_decode_draft_tokens = np.full(num_reqs, -1, dtype=np.int32)
            for (
                req_id,
                draft_token_ids,
            ) in scheduler_output.scheduled_spec_decode_tokens.items():
                req_idx = self.input_batch.req_id_to_index[req_id]
                num_draft_tokens[req_idx] = len(draft_token_ids)
                if (
                    self.input_batch.num_computed_tokens_cpu[req_idx]
                    >= self.input_batch.num_prompt_tokens[req_idx]
                ):
                    num_decode_draft_tokens[req_idx] = len(draft_token_ids)
            spec_decode_metadata = self._calc_spec_decode_metadata(
                num_draft_tokens, cu_num_tokens
            )
            logits_indices = spec_decode_metadata.logits_indices
            num_sampled_tokens = num_draft_tokens + 1
            # For DECODE only cuda graph of some attention backends (e.g., GDN).
            self.num_decode_draft_tokens.np[:num_reqs] = num_decode_draft_tokens
            self.num_decode_draft_tokens.np[num_reqs:].fill(-1)
            self.num_decode_draft_tokens.copy_to_gpu()

        # Hot-Swap lora model
        if self.lora_config:
            assert (
                np.sum(num_sampled_tokens)
                <= self.vllm_config.scheduler_config.max_num_batched_tokens
            )
            self.set_active_loras(
                self.input_batch, num_scheduled_tokens, num_sampled_tokens
            )

        return (
            logits_indices,
            spec_decode_metadata,
        )

    def _build_attention_metadata(
        self,
        num_tokens: int,
        num_reqs: int,
        max_query_len: int,
        num_tokens_padded: int | None = None,
        num_reqs_padded: int | None = None,
        ubatch_slices: UBatchSlices | None = None,
        logits_indices: torch.Tensor | None = None,
        use_spec_decode: bool = False,
        for_cudagraph_capture: bool = False,
        num_scheduled_tokens: dict[str, int] | None = None,
        cascade_attn_prefix_lens: list[list[int]] | None = None,
        slot_mappings: dict[int, torch.Tensor] | None = None,
    ) -> tuple[PerLayerAttnMetadata, CommonAttentionMetadata | None]:
        """
        :return: tuple[attn_metadata, spec_decode_common_attn_metadata]
        """
        # Attention metadata is not needed for attention free models
        if len(self.kv_cache_config.kv_cache_groups) == 0:
            return {}, None

        num_tokens_padded = num_tokens_padded or num_tokens
        num_reqs_padded = num_reqs_padded or num_reqs
        assert num_reqs_padded is not None and num_tokens_padded is not None

        attn_metadata: PerLayerAttnMetadata = {}
        if ubatch_slices is not None:
            attn_metadata = [dict() for _ in range(len(ubatch_slices))]

        if for_cudagraph_capture:
            # For some attention backends (e.g. FA) with sliding window models we need
            # to make sure the backend see a max_seq_len that is larger to the sliding
            # window size when capturing to make sure the correct kernel is selected.
            max_seq_len = self.max_model_len
        else:
            max_seq_len = self.seq_lens.np[:num_reqs].max().item()

        _тм1 = time.perf_counter()
        if use_spec_decode:
            _пг = getattr(self, "_приёмка_gpu", None)
            if _пг is not None and _пг.numel() >= num_reqs:
                _цель = self.num_accepted_tokens.gpu
                _цель[:num_reqs].copy_(_пг[:num_reqs].to(_цель.dtype))
                _цель[num_reqs:].fill_(1)
                self._приёмка_gpu = None
            else:
                self.num_accepted_tokens.np[:num_reqs] = (
                    self.input_batch.num_accepted_tokens_cpu[:num_reqs]
                )
                self.num_accepted_tokens.np[num_reqs:].fill(1)
                self.num_accepted_tokens.copy_to_gpu()
        if _ФАЗА_ШАГА_N:
            self._фаза_шага("    мета: приёмка+H2D", time.perf_counter() - _тм1)
        _тм2 = time.perf_counter()

        kv_cache_groups = self.kv_cache_config.kv_cache_groups

        def _get_block_table(kv_cache_gid: int):
            assert num_reqs_padded is not None and num_tokens_padded is not None
            kv_cache_spec = kv_cache_groups[kv_cache_gid].kv_cache_spec
            if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
                blk_table_tensor = torch.zeros(
                    (num_reqs_padded, 1),
                    dtype=torch.int32,
                    device=self.device,
                )
            else:
                blk_table = self.input_batch.block_table[kv_cache_gid]
                blk_table_tensor = blk_table.get_device_tensor(num_reqs_padded)

            # Fill unused with -1. Needed for reshape_and_cache in full cuda
            # graph mode. `blk_table_tensor` -1 to match mamba PAD_SLOT_ID
            blk_table_tensor[num_reqs:num_reqs_padded].fill_(-1)
            return blk_table_tensor

        assert slot_mappings is not None
        block_table_gid_0 = _get_block_table(0)
        slot_mapping_gid_0 = slot_mappings[0]

        if self.model_config.enable_return_routed_experts:
            self.slot_mapping = slot_mapping_gid_0[:num_tokens].cpu().numpy()
        cm_base = CommonAttentionMetadata(
            query_start_loc=self.query_start_loc.gpu[: num_reqs_padded + 1],
            query_start_loc_cpu=self.query_start_loc.cpu[: num_reqs_padded + 1],
            seq_lens=self.seq_lens.gpu[:num_reqs_padded],
            _seq_lens_cpu=self.seq_lens.cpu[:num_reqs_padded],
            _num_computed_tokens_cpu=self.input_batch.num_computed_tokens_cpu_tensor[
                :num_reqs_padded
            ],
            num_reqs=num_reqs_padded,
            num_actual_tokens=num_tokens_padded,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            block_table_tensor=block_table_gid_0,
            slot_mapping=slot_mapping_gid_0,
            causal=True,
        )

        if self.dcp_world_size > 1:
            self.dcp_local_seq_lens.cpu[:num_reqs] = get_dcp_local_seq_lens(
                self.seq_lens.cpu[:num_reqs],
                self.dcp_world_size,
                self.dcp_rank,
                self.parallel_config.cp_kv_cache_interleave_size,
            )
            self.dcp_local_seq_lens.cpu[num_reqs:].fill_(0)
            self.dcp_local_seq_lens.copy_to_gpu(num_reqs_padded)

            cm_base.dcp_local_seq_lens = self.dcp_local_seq_lens.gpu[:num_reqs_padded]
            cm_base.dcp_local_seq_lens_cpu = self.dcp_local_seq_lens.cpu[
                :num_reqs_padded
            ]

        if logits_indices is not None and self.cache_config.kv_sharing_fast_prefill:
            cm_base.num_logits_indices = logits_indices.size(0)
            cm_base.logits_indices_padded = self._prepare_kv_sharing_fast_prefill(
                logits_indices
            )

        # Cache attention metadata builds across hybrid KV-cache groups
        # The only thing that changes between different hybrid KV-cache groups when the
        # same metadata builder and KVCacheSpec is the same is the block table, so we
        # can cache the attention metadata builds and just update the block table using
        # `builder.update_block_table` if the builder supports it.
        cached_attn_metadata: dict[
            tuple[KVCacheSpec, type[AttentionMetadataBuilder]], AttentionMetadata
        ] = {}

        def _build_attn_group_metadata(
            kv_cache_gid: int,
            attn_gid: int,
            common_attn_metadata: CommonAttentionMetadata,
            ubid: int | None = None,
        ) -> None:
            attn_group = self.attn_groups[kv_cache_gid][attn_gid]
            # [ДУБЛЬ МЕТАДАННЫХ ЧЕРНОВИКА -- ПРОПУСК ПОД РЫЧАГОМ, 31.08]
            # Прибор формы шага (с именем группы) показал: за шаг цель строит метаданные ОДИН
            # раз, черновик -- ТРИ, из них два вызова с идентичными параметрами. Разбор: от
            # черновиковой группы раннеру нужен только `cm` (из него берётся
            # `spec_decode_common_attn_metadata`), а результат `builder.build` кладётся в
            # словарь по слоям, которые в проходе ЦЕЛИ не вызываются: черновик строит свои
            # метаданные (`build_for_drafting`) и разворачивает СВОЙ set_forward_context.
            # Умолчание ВЫКЛЮЧЕНО: цена 0.15 мс при шаге 43 мс, а место горячее -- включать
            # только после гейта тождества выдачи И проверки, что приёмка не просела.
            if _ПРОПУСК_МЕТЫ_ЧЕРНОВИКА and getattr(self, "drafter", None) is not None:
                _имена = getattr(self.drafter, "attn_layer_names", None)
                if _имена and _имена[0] in attn_group.layer_names:
                    return
            builder = attn_group.get_metadata_builder(ubid or 0)
            kv_cache_spec = kv_cache_groups[kv_cache_gid].kv_cache_spec
            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
                kv_cache_spec = kv_cache_spec.kv_cache_specs[attn_group.layer_names[0]]
            cache_key = (kv_cache_spec, type(builder))

            cascade_attn_prefix_len = (
                cascade_attn_prefix_lens[kv_cache_gid][attn_gid]
                if cascade_attn_prefix_lens
                else 0
            )

            extra_attn_metadata_args = {}
            if use_spec_decode and isinstance(builder, GDNAttentionMetadataBuilder):
                assert ubid is None, "UBatching not supported with GDN yet"
                extra_attn_metadata_args = dict(
                    num_accepted_tokens=self.num_accepted_tokens.gpu[:num_reqs_padded],
                    num_decode_draft_tokens_cpu=self.num_decode_draft_tokens.cpu[
                        :num_reqs_padded
                    ],
                )

            if for_cudagraph_capture:
                attn_metadata_i = builder.build_for_cudagraph_capture(
                    common_attn_metadata
                )
            elif (
                cache_key in cached_attn_metadata
                and builder.supports_update_block_table
            ):
                _туб = time.perf_counter()
                attn_metadata_i = builder.update_block_table(
                    cached_attn_metadata[cache_key],
                    common_attn_metadata.block_table_tensor,
                    common_attn_metadata.slot_mapping,
                )
                if _ФАЗА_ШАГА_N:
                    self._фаза_шага(f"    upd:{type(builder).__name__}",
                                    time.perf_counter() - _туб)
                # [FA2/SM70 25.08] ПОБИТОВАЯ СВЕРКА ДВУХ ДОРОГ (FA2SM70_META_CHECK=N).
                # Быстрый путь `update_block_table` уже РАЗ провалил гейт («39» вместо «391»), и
                # формой это не ловилось. Здесь строим ОБА варианта и сравниваем ПОЛЕ ЗА ПОЛЕМ,
                # а в работу отдаём медленный (заведомо верный). Так расхождение НАЗЫВАЕТСЯ, а не
                # угадывается.
                _пров = int(os.environ.get("FA2SM70_META_CHECK", "0"))
                if _пров and getattr(self, "_мета_сверок", 0) < _пров:
                    self._мета_сверок = getattr(self, "_мета_сверок", 0) + 1
                    _эталон = builder.build(
                        common_prefix_len=cascade_attn_prefix_len,
                        common_attn_metadata=common_attn_metadata,
                        **extra_attn_metadata_args,
                    )
                    _разн = []
                    for _поле in getattr(_эталон, "__dataclass_fields__", {}):
                        _a = getattr(attn_metadata_i, _поле, None)
                        _b = getattr(_эталон, _поле, None)
                        if isinstance(_b, torch.Tensor) or isinstance(_a, torch.Tensor):
                            if _a is None or _b is None:
                                _разн.append(f"{_поле}: {'нет' if _a is None else 'есть'} против "
                                             f"{'нет' if _b is None else 'есть'}")
                            elif _a.shape != _b.shape:
                                _разн.append(f"{_поле}: форма {tuple(_a.shape)} != {tuple(_b.shape)}")
                            elif not torch.equal(_a, _b):
                                _разн.append(f"{_поле}: ЗНАЧЕНИЯ различаются "
                                             f"({int((_a != _b).sum())} из {_a.numel()})")
                        else:
                            try:
                                _равно = bool(_a == _b)
                            except Exception:  # noqa: BLE001 - словари/списки тензоров
                                _равно = (repr(_a) == repr(_b))
                            if not _равно:
                                _разн.append(f"{_поле}: {str(_a)[:40]} != {str(_b)[:40]}")
                    logger.info("[fa2_sm70 СВЕРКА МЕТА %d] %s", self._мета_сверок,
                                "; ".join(_разн) if _разн else "СОВПАЛО ПОЛЕ В ПОЛЕ")
                    attn_metadata_i = _эталон  # в работу -- заведомо верный
            else:
                _тб2 = time.perf_counter()
                attn_metadata_i = builder.build(
                    common_prefix_len=cascade_attn_prefix_len,
                    common_attn_metadata=common_attn_metadata,
                    **extra_attn_metadata_args,
                )
                if _ФАЗА_ШАГА_N:
                    self._фаза_шага(
                        f"    build:{type(builder).__name__}",
                        time.perf_counter() - _тб2,
                    )
                if builder.supports_update_block_table:
                    cached_attn_metadata[cache_key] = attn_metadata_i

            if ubid is None:
                assert isinstance(attn_metadata, dict)
                attn_metadata_dict = attn_metadata
            else:
                assert isinstance(attn_metadata, list)
                attn_metadata_dict = attn_metadata[ubid]

            for layer_name in attn_group.layer_names:
                attn_metadata_dict[layer_name] = attn_metadata_i

        if _ФАЗА_ШАГА_N:
            self._фаза_шага("    мета: cm_base+таблицы", time.perf_counter() - _тм2)
        _тм3 = time.perf_counter()

        # Prepare the attention metadata for each KV cache group and make layers
        # in the same group share the same metadata.
        spec_decode_common_attn_metadata = None
        for kv_cache_gid, kv_cache_group in enumerate(kv_cache_groups):
            cm = copy(cm_base)  # shallow copy

            # Basically only the encoder seq_lens, block_table and slot_mapping change
            # for each kv_cache_group.
            _тгр = time.perf_counter()
            cm.encoder_seq_lens, cm.encoder_seq_lens_cpu = self._get_encoder_seq_lens(
                num_scheduled_tokens or {},
                kv_cache_group.kv_cache_spec,
                num_reqs_padded,
                for_cudagraph_capture=for_cudagraph_capture,
            )
            if kv_cache_gid > 0:
                cm.block_table_tensor = _get_block_table(kv_cache_gid)
                cm.slot_mapping = slot_mappings[kv_cache_gid]
            if _ФАЗА_ШАГА_N:
                self._фаза_шага("    мета: таблица+encoder", time.perf_counter() - _тгр)

            if self.speculative_config and spec_decode_common_attn_metadata is None:
                if isinstance(self.drafter, EagleProposer | DFlash2Proposer):
                    if self.drafter.attn_layer_names[0] in kv_cache_group.layer_names:
                        spec_decode_common_attn_metadata = cm
                else:
                    spec_decode_common_attn_metadata = cm

            if _ФАЗА_ШАГА_N and not getattr(self, "_описано_групп", False):
                self._описано_групп = True
                for _g, _гр in enumerate(self.attn_groups):
                    for _a, _агр in enumerate(_гр):
                        _b = _агр.get_metadata_builder(0)
                        logger.info("[fa2_sm70 ГРУППЫ] kv-группа %d, подгруппа %d: %s, слоёв %d",
                                    _g, _a, type(_b).__name__, len(_агр.layer_names))
            for attn_gid in range(len(self.attn_groups[kv_cache_gid])):
                if ubatch_slices is not None:
                    for ubid, _cm in enumerate(split_attn_metadata(ubatch_slices, cm)):
                        _build_attn_group_metadata(kv_cache_gid, attn_gid, _cm, ubid)

                else:
                    _build_attn_group_metadata(kv_cache_gid, attn_gid, cm)

        if _ФАЗА_ШАГА_N:
            self._фаза_шага("    мета: цикл групп", time.perf_counter() - _тм3)

        if self.is_mm_prefix_lm:
            req_doc_ranges = {}
            for req_id in self.input_batch.req_ids:
                image_doc_ranges = []
                req_state = self.requests[req_id]
                for mm_feature in req_state.mm_features:
                    pos_info = mm_feature.mm_position
                    img_doc_range = pos_info.extract_embeds_range()
                    image_doc_ranges.extend(img_doc_range)
                req_idx = self.input_batch.req_id_to_index[req_id]
                req_doc_ranges[req_idx] = image_doc_ranges

            if isinstance(attn_metadata, list):
                for ub_metadata in attn_metadata:
                    for _metadata in ub_metadata.values():
                        _metadata.mm_prefix_range = req_doc_ranges  # type: ignore[attr-defined]
            else:
                for _metadata in attn_metadata.values():
                    _metadata.mm_prefix_range = req_doc_ranges  # type: ignore[attr-defined]

        if spec_decode_common_attn_metadata is not None and (
            num_reqs != num_reqs_padded or num_tokens != num_tokens_padded
        ):
            # Currently the drafter still only uses piecewise cudagraphs (and modifies
            # the attention metadata in directly), and therefore does not want to use
            # padded attention metadata.
            spec_decode_common_attn_metadata = (
                spec_decode_common_attn_metadata.unpadded(num_tokens, num_reqs)
            )

        return attn_metadata, spec_decode_common_attn_metadata

    def _compute_cascade_attn_prefix_lens(
        self,
        num_scheduled_tokens: np.ndarray,
        num_computed_tokens: np.ndarray,
        num_common_prefix_blocks: list[int],
    ) -> list[list[int]] | None:
        """
        :return: Optional[cascade_attn_prefix_lens]
            cascade_attn_prefix_lens is 2D: ``[kv_cache_group_id][attn_group_idx]``,
            None if we should not use cascade attention
        """

        use_cascade_attn = False
        num_kv_cache_groups = len(self.kv_cache_config.kv_cache_groups)
        cascade_attn_prefix_lens: list[list[int]] = [
            [] for _ in range(num_kv_cache_groups)
        ]

        for kv_cache_gid in range(num_kv_cache_groups):
            for attn_group in self.attn_groups[kv_cache_gid]:
                if isinstance(attn_group.kv_cache_spec, EncoderOnlyAttentionSpec):
                    cascade_attn_prefix_len = 0
                else:
                    # 0 if cascade attention should not be used
                    cascade_attn_prefix_len = self._compute_cascade_attn_prefix_len(
                        num_scheduled_tokens,
                        num_computed_tokens,
                        num_common_prefix_blocks[kv_cache_gid],
                        attn_group.kv_cache_spec,
                        attn_group.get_metadata_builder(),
                    )
                cascade_attn_prefix_lens[kv_cache_gid].append(cascade_attn_prefix_len)
                use_cascade_attn |= cascade_attn_prefix_len > 0

        return cascade_attn_prefix_lens if use_cascade_attn else None

    def _compute_cascade_attn_prefix_len(
        self,
        num_scheduled_tokens: np.ndarray,
        num_computed_tokens: np.ndarray,
        num_common_prefix_blocks: int,
        kv_cache_spec: KVCacheSpec,
        attn_metadata_builder: AttentionMetadataBuilder,
    ) -> int:
        """Compute the length of the common prefix for cascade attention.

        NOTE(woosuk): The common prefix length returned by this function
        represents the length used specifically for cascade attention, not the
        actual number of tokens shared between requests. When cascade attention
        is disabled (use_cascade=False), this function returns 0 even if
        requests share common tokens. Additionally, the common prefix length is
        truncated to a multiple of the block size and may be further truncated
        due to implementation details explained below.

        Args:
            num_scheduled_tokens: Number of tokens scheduled per request.
            num_common_prefix_blocks: Number of shared KV cache blocks.

        Returns:
            int: Length of common prefix in tokens.
        """

        common_prefix_len = num_common_prefix_blocks * kv_cache_spec.block_size
        if common_prefix_len == 0:
            # Common case.
            return 0

        # NOTE(woosuk): Cascade attention uses two attention kernels: one
        # for the common prefix and the other for the rest. For the first
        # kernel, we concatenate all the query tokens (possibly from
        # different requests) and treat them as if they are from the same
        # request. Then, we use bi-directional attention to process the
        # common prefix in the KV cache. Importantly, this means that the
        # first kernel does not do any masking.

        # Consider the following example:
        # Request 1's input query: [D, E, X]
        # Request 1's kv cache: [A, B, C, D, E, X]
        # Request 1's num_computed_tokens: 3 (i.e., [A, B, C])
        # Request 2's input query: [E, Y]
        # Request 2's kv cache: [A, B, C, D, E, Y]
        # Request 2's num_computed_tokens: 4 (i.e., [A, B, C, D])

        # If we use [A, B, C, D, E] as the common prefix, then the
        # first kernel will compute the bi-directional attention between
        # input query [D, E, X, E, Y] and common prefix [A, B, C, D, E].
        # However, this is wrong because D in Request 1 should not attend to
        # E in the common prefix (i.e., we need masking).
        # To avoid this, [A, B, C, D] should be the common prefix.
        # That is, the common prefix should be capped by the minimum
        # num_computed_tokens among the requests, and plus one to include
        # the first token of the query.

        # In practice, we use [A, B, C] as the common prefix, instead of
        # [A, B, C, D] (i.e., the common prefix is capped by the minimum
        # num_computed_tokens, without plus one).
        # This is because of an implementation detail: We want to always
        # use two kernels for cascade attention. Let's imagine:
        # Request 3's input query: [D]
        # Request 3's kv cache: [A, B, C, D]
        # Request 3's num_computed_tokens: 3 (i.e., [A, B, C])
        # If we use [A, B, C, D] as the common prefix for Request 1-3,
        # then Request 3 will be processed only by the first kernel,
        # and the second kernel will get an empty input. While this is not
        # a fundamental problem, our current implementation does not support
        # this case.
        common_prefix_len = min(common_prefix_len, num_computed_tokens.min())
        # common_prefix_len should be a multiple of the block size.
        common_prefix_len = (
            common_prefix_len // kv_cache_spec.block_size * kv_cache_spec.block_size
        )
        use_sliding_window = isinstance(kv_cache_spec, SlidingWindowSpec) or (
            isinstance(kv_cache_spec, FullAttentionSpec)
            and kv_cache_spec.sliding_window is not None
        )
        use_local_attention = isinstance(kv_cache_spec, ChunkedLocalAttentionSpec) or (
            isinstance(kv_cache_spec, FullAttentionSpec)
            and kv_cache_spec.attention_chunk_size is not None
        )
        assert isinstance(kv_cache_spec, AttentionSpec)
        use_cascade = attn_metadata_builder.use_cascade_attention(
            common_prefix_len=common_prefix_len,
            query_lens=num_scheduled_tokens,
            num_query_heads=self.num_query_heads,
            num_kv_heads=kv_cache_spec.num_kv_heads,
            use_alibi=self.use_alibi,
            use_sliding_window=use_sliding_window,
            use_local_attention=use_local_attention,
            num_sms=self.num_sms,
            dcp_world_size=self.dcp_world_size,
        )
        return common_prefix_len if use_cascade else 0

    def _calc_mrope_positions(self, scheduler_output: "SchedulerOutput"):
        mrope_pos_ptr = 0
        for index, req_id in enumerate(self.input_batch.req_ids):
            req = self.requests[req_id]
            assert req.mrope_positions is not None

            num_computed_tokens = self.input_batch.num_computed_tokens_cpu[index]
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
                req.prompt_token_ids, req.prompt_embeds
            )

            if num_computed_tokens + num_scheduled_tokens > num_prompt_tokens:
                prompt_part_len = max(0, num_prompt_tokens - num_computed_tokens)
                completion_part_len = max(0, num_scheduled_tokens - prompt_part_len)
            else:
                prompt_part_len = num_scheduled_tokens
                completion_part_len = 0

            assert num_scheduled_tokens == prompt_part_len + completion_part_len

            if prompt_part_len > 0:
                # prompt's mrope_positions are pre-computed
                dst_start = mrope_pos_ptr
                dst_end = mrope_pos_ptr + prompt_part_len
                src_start = num_computed_tokens
                src_end = num_computed_tokens + prompt_part_len

                self.mrope_positions.cpu[:, dst_start:dst_end] = req.mrope_positions[
                    :, src_start:src_end
                ]
                mrope_pos_ptr += prompt_part_len

            if completion_part_len > 0:
                # compute completion's mrope_positions on-the-fly
                dst_start = mrope_pos_ptr
                dst_end = mrope_pos_ptr + completion_part_len

                assert req.mrope_position_delta is not None
                MRotaryEmbedding.get_next_input_positions_tensor(
                    out=self.mrope_positions.np,
                    out_offset=dst_start,
                    mrope_position_delta=req.mrope_position_delta,
                    context_len=num_computed_tokens + prompt_part_len,
                    num_new_tokens=completion_part_len,
                )

                mrope_pos_ptr += completion_part_len

    def _calc_xdrope_positions(self, scheduler_output: "SchedulerOutput"):
        xdrope_pos_ptr = 0
        for index, req_id in enumerate(self.input_batch.req_ids):
            req = self.requests[req_id]
            assert req.xdrope_positions is not None

            num_computed_tokens = self.input_batch.num_computed_tokens_cpu[index]
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
                req.prompt_token_ids, req.prompt_embeds
            )

            if num_computed_tokens + num_scheduled_tokens > num_prompt_tokens:
                prompt_part_len = max(0, num_prompt_tokens - num_computed_tokens)
                completion_part_len = max(0, num_scheduled_tokens - prompt_part_len)
            else:
                prompt_part_len = num_scheduled_tokens
                completion_part_len = 0

            assert num_scheduled_tokens == prompt_part_len + completion_part_len

            if prompt_part_len > 0:
                # prompt's xdrope_positions are pre-computed
                dst_start = xdrope_pos_ptr
                dst_end = xdrope_pos_ptr + prompt_part_len
                src_start = num_computed_tokens
                src_end = num_computed_tokens + prompt_part_len

                self.xdrope_positions.cpu[:, dst_start:dst_end] = req.xdrope_positions[
                    :, src_start:src_end
                ]
                xdrope_pos_ptr += prompt_part_len

            if completion_part_len > 0:
                # compute completion's xdrope_positions on-the-fly
                dst_start = xdrope_pos_ptr
                dst_end = xdrope_pos_ptr + completion_part_len

                XDRotaryEmbedding.get_next_input_positions_tensor(
                    out=self.xdrope_positions.np,
                    out_offset=dst_start,
                    context_len=num_computed_tokens + prompt_part_len,
                    num_new_tokens=completion_part_len,
                )

                xdrope_pos_ptr += completion_part_len

    def _calc_spec_decode_metadata(
        self,
        num_draft_tokens: np.ndarray,
        cu_num_scheduled_tokens: np.ndarray,
    ) -> SpecDecodeMetadata:
        # Inputs:
        # cu_num_scheduled_tokens:  [  4, 104, 107, 207, 209]
        # num_draft_tokens:         [  3,   0,   2,   0,   1]
        # Outputs:
        # cu_num_draft_tokens:      [  3,   3,   5,   5,   6]
        # logits_indices:           [  0,   1,   2,   3, 103, 104, 105, 106,
        #                            206, 207, 208]
        # target_logits_indices:    [  0,   1,   2,   5,   6,   9]
        # bonus_logits_indices:     [  3,   4,   7,   8,  10]

        # Compute the logits indices.
        # [4, 1, 3, 1, 2]
        num_sampled_tokens = num_draft_tokens + 1

        # Step 1. cu_num_sampled_tokens: [4, 5, 8, 9, 11]
        # arange: [0, 1, 2, 3, 0, 0, 1, 2, 0, 0, 1]
        cu_num_sampled_tokens, arange = self._get_cumsum_and_arange(
            num_sampled_tokens, cumsum_dtype=np.int32
        )
        # Step 2. [0, 0, 0, 0, 103, 104, 104, 104, 206, 207, 207]
        logits_indices = np.repeat(
            cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens
        )
        # Step 3. [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208]
        logits_indices += arange

        # Compute the bonus logits indices.
        bonus_logits_indices = cu_num_sampled_tokens - 1

        # Compute the draft logits indices.
        # cu_num_draft_tokens: [3, 3, 5, 5, 6]
        # arange: [0, 1, 2, 0, 1, 0]
        cu_num_draft_tokens, arange = self._get_cumsum_and_arange(
            num_draft_tokens, cumsum_dtype=np.int32
        )
        # [0, 0, 0, 5, 5, 9]
        target_logits_indices = np.repeat(
            cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens
        )
        # [0, 1, 2, 5, 6, 9]
        target_logits_indices += arange

        # TODO: Optimize the CPU -> GPU copy.
        cu_num_draft_tokens = torch.from_numpy(cu_num_draft_tokens).to(
            self.device, non_blocking=True
        )
        cu_num_sampled_tokens = torch.from_numpy(cu_num_sampled_tokens).to(
            self.device, non_blocking=True
        )
        logits_indices = torch.from_numpy(logits_indices).to(
            self.device, non_blocking=True
        )
        target_logits_indices = torch.from_numpy(target_logits_indices).to(
            self.device, non_blocking=True
        )
        bonus_logits_indices = torch.from_numpy(bonus_logits_indices).to(
            self.device, non_blocking=True
        )

        # Compute the draft token ids.
        # draft_token_indices:      [  1,   2,   3, 105, 106, 208]
        draft_token_ids = self.input_ids.gpu[logits_indices]
        draft_token_ids = draft_token_ids[target_logits_indices + 1]

        return SpecDecodeMetadata(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens.tolist(),
            cu_num_draft_tokens=cu_num_draft_tokens,
            cu_num_sampled_tokens=cu_num_sampled_tokens,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
        )

    def _prepare_kv_sharing_fast_prefill(
        self,
        logits_indices: torch.Tensor,
    ) -> torch.Tensor:
        assert self.kv_sharing_fast_prefill_logits_indices is not None
        num_logits = logits_indices.shape[0]
        assert num_logits > 0
        self.kv_sharing_fast_prefill_logits_indices[:num_logits].copy_(logits_indices)
        # There might have leftover indices in logits_indices[num_logits:]
        # from previous iterations, whose values may be greater than the
        # batch size in the current iteration. To ensure indices are always
        # valid, we fill the padded indices with the last index.
        self.kv_sharing_fast_prefill_logits_indices[num_logits:].fill_(
            logits_indices[-1].item()
        )
        # Dispatch for the decoder portion of the model.
        _, batch_desc = self.cudagraph_dispatcher.dispatch(
            num_logits, disable_full=True
        )
        num_logits_padded = batch_desc.num_tokens
        logits_indices_padded = self.kv_sharing_fast_prefill_logits_indices[
            :num_logits_padded
        ]
        return logits_indices_padded

    def _batch_mm_inputs_from_scheduler(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> tuple[
        list[str],
        list[tuple[str, MultiModalKwargsItem]],
        list[tuple[str, PlaceholderRange]],
    ]:
        """Batch multimodal inputs from scheduled encoder inputs.

        Args:
            scheduler_output: The scheduler output containing scheduled encoder
                inputs.

        Returns:
            A tuple of (mm_hashes, mm_kwargs, mm_lora_refs) where:
            - mm_hashes: List of multimodal hashes for each item
            - mm_kwargs: List of multimodal kwargs for each item
            - mm_lora_refs: List of (req_id, placeholder_range) for each item
        """
        scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
        if not scheduled_encoder_inputs:
            return [], [], []

        mm_hashes = list[str]()
        mm_kwargs = list[tuple[str, MultiModalKwargsItem]]()
        # Multimodal LoRA reference info to map each multimodal item
        # back to its request & position
        mm_lora_refs = list[tuple[str, PlaceholderRange]]()
        for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
            req_state = self.requests[req_id]

            for mm_input_id in encoder_input_ids:
                mm_feature = req_state.mm_features[mm_input_id]
                if mm_feature.data is None:
                    continue

                mm_hashes.append(mm_feature.identifier)
                mm_kwargs.append((mm_feature.modality, mm_feature.data))
                mm_lora_refs.append((req_id, mm_feature.mm_position))

        return mm_hashes, mm_kwargs, mm_lora_refs

    def _execute_mm_encoder(
        self, scheduler_output: "SchedulerOutput"
    ) -> list[torch.Tensor]:
        mm_hashes, mm_kwargs, mm_lora_refs = self._batch_mm_inputs_from_scheduler(
            scheduler_output
        )

        if not mm_kwargs:
            return []

        should_time = bool(
            self.observability_config
            and self.observability_config.enable_mm_processor_stats
            and scheduler_output.scheduled_encoder_inputs
        )

        # Batch mm inputs as much as we can: if a request in the batch has
        # multiple modalities or a different modality than the previous one,
        # we process it separately to preserve item order.
        # FIXME(ywang96): This is a hacky way to deal with multiple modalities
        # in the same batch while still being able to benefit from batching
        # multimodal inputs. The proper solution should be reordering the
        # encoder outputs.
        model = cast(SupportsMultiModal, self.model)

        if self.lora_config and self.lora_manager.supports_tower_connector_lora():
            # Build LoRA mappings independently for encoder inputs
            # (encoder batch structure is different from main batch)
            prompt_lora_mapping = []
            token_lora_mapping = []
            lora_requests = set()
            encoder_token_counts = []

            for req_id, pos_info in mm_lora_refs:
                req_idx = self.input_batch.req_id_to_index[req_id]
                lora_id = int(self.input_batch.request_lora_mapping[req_idx])

                # Prefer pos_info.get_num_embeds to count precise MM embedding tokens.
                num_tokens = self.model.get_num_mm_encoder_tokens(  # type: ignore[attr-defined]
                    pos_info.get_num_embeds
                )
                prompt_lora_mapping.append(lora_id)
                token_lora_mapping.extend([lora_id] * num_tokens)
                encoder_token_counts.append(num_tokens)

                if lora_id > 0:
                    lora_request = self.input_batch.lora_id_to_lora_request.get(lora_id)
                    if lora_request is not None:
                        lora_requests.add(lora_request)

            # Set tower adapter mapping
            tower_mapping = LoRAMapping(
                tuple(token_lora_mapping),
                tuple(prompt_lora_mapping),
                is_prefill=True,
                type=LoRAMappingType.TOWER,
            )
            self.lora_manager.set_active_adapters(lora_requests, tower_mapping)

            if hasattr(self.model, "get_num_mm_connector_tokens"):
                post_op_counts = [
                    self.model.get_num_mm_connector_tokens(num_tokens)  # type: ignore[attr-defined]
                    for num_tokens in encoder_token_counts
                ]

                connector_token_mapping = np.repeat(
                    np.array(prompt_lora_mapping, dtype=np.int32),
                    np.array(post_op_counts, dtype=np.int32),
                )
                connector_mapping = LoRAMapping(
                    index_mapping=tuple(connector_token_mapping.tolist()),
                    prompt_mapping=tuple(prompt_lora_mapping),
                    is_prefill=True,
                    type=LoRAMappingType.CONNECTOR,
                )

                self.lora_manager.set_active_adapters(
                    lora_requests,
                    connector_mapping,
                )

        encoder_outputs: list[torch.Tensor] = []
        # Track the current index in mm_kwargs/mm_lora_refs to map groups to request IDs
        current_item_idx = 0
        for modality, num_items, mm_kwargs_group in group_mm_kwargs_by_modality(
            mm_kwargs,
            device=self.device,
            pin_memory=self.pin_memory,
        ):
            curr_group_outputs: MultiModalEmbeddings

            # EVS-related change.
            # (ekhvedchenia): Temporary hack to limit peak memory usage when
            # processing multimodal data. This solves the issue with scheduler
            # putting too many video samples into a single batch. Scheduler
            # uses pruned vision tokens count to compare it versus compute
            # budget which is incorrect (Either input media size or non-pruned
            # output vision tokens count should be considered)
            # TODO(ywang96): Fix memory profiling to take EVS into account and
            # remove this hack.
            if (
                self.is_multimodal_pruning_enabled
                and modality == "video"
                and num_items > 1
            ):
                curr_group_outputs_lst = list[torch.Tensor]()
                for video_idx in range(num_items):
                    video_mm_kwargs_item = mm_kwargs[current_item_idx + video_idx]
                    with self.timed_encoder_operation(
                        should_time, mm_lora_refs, current_item_idx + video_idx, 1
                    ):
                        _, _, micro_batch_mm_inputs = next(
                            group_mm_kwargs_by_modality(
                                [video_mm_kwargs_item],
                                device=self.device,
                                pin_memory=self.pin_memory,
                            )
                        )

                        micro_batch_outputs = model.embed_multimodal(
                            **micro_batch_mm_inputs
                        )

                        curr_group_outputs_lst.extend(micro_batch_outputs)

                curr_group_outputs = curr_group_outputs_lst
            else:
                # Run the encoder.
                # `curr_group_outputs` is either of the following:
                # 1. A tensor of shape (num_items, feature_size, hidden_size)
                # in case feature_size is fixed across all multimodal items.
                # 2. A list or tuple (length: num_items) of tensors,
                # each of shape (feature_size, hidden_size) in case the feature
                # size is dynamic depending on the input multimodal items.

                with self.timed_encoder_operation(
                    should_time, mm_lora_refs, current_item_idx, num_items
                ):
                    curr_group_outputs = model.embed_multimodal(**mm_kwargs_group)

            sanity_check_mm_encoder_outputs(
                curr_group_outputs,
                expected_num_items=num_items,
            )
            encoder_outputs.extend(curr_group_outputs)

            current_item_idx += num_items

        # Cache the encoder outputs by mm_hash
        for mm_hash, output in zip(mm_hashes, encoder_outputs):
            self.encoder_cache[mm_hash] = output
            logger.debug("Finish execute for mm hash %s", mm_hash)
            self.maybe_save_ec_to_connector(self.encoder_cache, mm_hash)

        return encoder_outputs

    def _gather_mm_embeddings(
        self,
        scheduler_output: "SchedulerOutput",
        shift_computed_tokens: int = 0,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens

        # Swap to the other buffer to avoid race condition with previous
        # iteration's async copy that may still be reading from CPU.
        self.is_mm_embed_idx = 1 - self.is_mm_embed_idx
        is_mm_embed_buf = self.is_mm_embed_buffers[self.is_mm_embed_idx]

        mm_embeds = list[torch.Tensor]()
        is_mm_embed = is_mm_embed_buf.cpu
        is_mm_embed[:total_num_scheduled_tokens] = False

        req_start_idx = 0
        should_sync_mrope_positions = False
        should_sync_xdrope_positions = False

        for req_id in self.input_batch.req_ids:
            mm_embeds_req: list[torch.Tensor] = []

            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            req_state = self.requests[req_id]
            num_computed_tokens = req_state.num_computed_tokens + shift_computed_tokens

            for mm_feature in req_state.mm_features:
                pos_info = mm_feature.mm_position
                start_pos = pos_info.offset
                num_encoder_tokens = pos_info.length

                # The encoder output is needed if the two ranges overlap:
                # [num_computed_tokens,
                #  num_computed_tokens + num_scheduled_tokens) and
                # [start_pos, start_pos + num_encoder_tokens)
                if start_pos >= num_computed_tokens + num_scheduled_tokens:
                    # The encoder output is not needed in this step.
                    break
                if start_pos + num_encoder_tokens <= num_computed_tokens:
                    # The encoder output is already processed and stored
                    # in the decoder's KV cache.
                    continue

                start_idx = max(num_computed_tokens - start_pos, 0)
                end_idx = min(
                    num_computed_tokens - start_pos + num_scheduled_tokens,
                    num_encoder_tokens,
                )
                assert start_idx < end_idx
                curr_embeds_start, curr_embeds_end = (
                    pos_info.get_embeds_indices_in_range(start_idx, end_idx)
                )
                # If there are no embeddings in the current range, we skip
                # gathering the embeddings.
                if curr_embeds_start == curr_embeds_end:
                    continue

                mm_hash = mm_feature.identifier
                encoder_output = self.encoder_cache.get(mm_hash, None)
                if (encoder_output is None and shift_computed_tokens > 0
                        and _os.environ.get("FA2SM70_MM_DRAFT_SHIFT", "0") == "1"):
                    # [FA2/SM70 13.10] Черновик -- только ускорение: его токены проверяет цель. Нет
                    # выхода энкодера для окна черновика -- позиции остаются эмбеддингами токенов
                    # (черновик чуть хуже на шаг), а не assert, который роняет движок.
                    # Планировщик с тем же рычагом этот случай уже закрывает; здесь страховка.
                    logger.warning_once("[fa2_sm70] черновик: нет выхода энкодера %s в окне черновика, "
                                        "позиции картинки идут без эмбеддингов", mm_hash)
                    continue
                assert encoder_output is not None, f"Encoder cache miss for {mm_hash}."

                if (is_embed := pos_info.is_embed) is not None:
                    is_embed = is_embed[start_idx:end_idx]
                    mm_embeds_item = encoder_output[curr_embeds_start:curr_embeds_end]
                else:
                    mm_embeds_item = encoder_output[start_idx:end_idx]

                req_start_pos = req_start_idx + start_pos - num_computed_tokens
                # OR mask for overlapping mm_features (use_audio_in_video)
                if is_embed is None:
                    is_mm_embed[req_start_pos + start_idx : req_start_pos + end_idx] = (
                        True
                    )
                else:
                    is_mm_embed[
                        req_start_pos + start_idx : req_start_pos + end_idx
                    ] |= is_embed
                mm_embeds_req.append(mm_embeds_item)

            if self.is_multimodal_pruning_enabled and self.uses_mrope:
                assert req_state.mrope_positions is not None
                should_sync_mrope_positions = True
                mm_embeds_req, new_mrope_positions, new_delta = (
                    self.model.recompute_mrope_positions(
                        input_ids=req_state.prompt_token_ids,
                        multimodal_embeddings=mm_embeds_req,
                        mrope_positions=req_state.mrope_positions,
                        num_computed_tokens=req_state.num_computed_tokens,
                    )
                )
                req_state.mrope_positions.copy_(new_mrope_positions)
                req_state.mrope_position_delta = new_delta

            mm_embeds.extend(mm_embeds_req)
            req_start_idx += num_scheduled_tokens

        is_mm_embed = is_mm_embed_buf.copy_to_gpu(total_num_scheduled_tokens)

        if should_sync_mrope_positions:
            self._calc_mrope_positions(scheduler_output)
            self.mrope_positions.copy_to_gpu(total_num_scheduled_tokens)

        if should_sync_xdrope_positions:
            self._calc_xdrope_positions(scheduler_output)
            self.xdrope_positions.copy_to_gpu(total_num_scheduled_tokens)

        return mm_embeds, is_mm_embed

    def get_model(self) -> nn.Module:
        if not hasattr(self, "model"):
            raise ValueError("Cannot get model before model has been initialized")
        if isinstance(self.model, (CUDAGraphWrapper, UBatchWrapper)):
            # get raw model out of the cudagraph wrapper.
            return self.model.unwrap()
        return self.model

    def get_supported_generation_tasks(self) -> list[GenerationTask]:
        model = self.get_model()
        supported_tasks = list[GenerationTask]()

        if is_text_generation_model(model):
            supported_tasks.append("generate")

        if supports_transcription(model):
            if model.supports_transcription_only:
                return ["transcription"]

            supported_tasks.append("transcription")

        if supports_realtime(model):
            supported_tasks.append("realtime")

        return supported_tasks

    def get_supported_pooling_tasks(self) -> list[PoolingTask]:
        model = self.get_model()
        if not is_pooling_model(model):
            return []

        supported_tasks = list(model.pooler.get_supported_tasks())

        if "score" in supported_tasks:
            num_labels = getattr(self.model_config.hf_config, "num_labels", 0)
            if num_labels != 1:
                supported_tasks.remove("score")
                logger.debug_once("Score API is only enabled for num_labels == 1.")

        return supported_tasks

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        tasks = list[SupportedTask]()

        if self.model_config.runner_type == "generate":
            tasks.extend(self.get_supported_generation_tasks())
        if self.model_config.runner_type == "pooling":
            tasks.extend(self.get_supported_pooling_tasks())

        return tuple(tasks)

    def sync_and_slice_intermediate_tensors(
        self,
        num_tokens: int,
        intermediate_tensors: IntermediateTensors | None,
        sync_self: bool,
    ) -> IntermediateTensors:
        assert self.intermediate_tensors is not None

        tp = self.vllm_config.parallel_config.tensor_parallel_size
        is_rs = is_residual_scattered_for_sp(self.vllm_config, num_tokens)

        # When sequence parallelism is enabled, the "residual" tensor is sharded
        # across tensor parallel ranks, so each rank only needs its own slice.
        if sync_self:
            assert intermediate_tensors is not None
            for k, v in intermediate_tensors.items():
                is_scattered = k == "residual" and is_rs
                copy_len = num_tokens // tp if is_scattered else num_tokens
                self.intermediate_tensors[k][:copy_len].copy_(
                    v[:copy_len], non_blocking=True
                )

        return IntermediateTensors(
            {
                k: v[: num_tokens // tp]
                if k == "residual" and is_rs
                else v[:num_tokens]
                for k, v in self.intermediate_tensors.items()
            }
        )

    def eplb_step(self, is_dummy: bool = False, is_profile: bool = False) -> None:
        """
        Step for the EPLB (Expert Parallelism Load Balancing) state.
        """
        if not self.parallel_config.enable_eplb:
            return

        assert self.eplb_state is not None
        model = self.get_model()
        assert is_mixture_of_experts(model)
        self.eplb_state.step(
            is_dummy,
            is_profile,
            log_stats=self.parallel_config.eplb_config.log_balancedness,
        )

    def _pool(
        self,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: int,
        num_scheduled_tokens_np: np.ndarray,
        kv_connector_output: KVConnectorOutput | None,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        num_reqs = self.input_batch.num_reqs
        assert num_reqs == len(self.input_batch.pooling_params), (
            "Either all or none of the requests in a batch must be pooling request"
        )

        hidden_states = hidden_states[:num_scheduled_tokens]
        seq_lens_cpu = self.seq_lens.cpu[:num_reqs]

        pooling_metadata = self.input_batch.get_pooling_metadata()
        pooling_metadata.build_pooling_cursor(
            num_scheduled_tokens_np, seq_lens_cpu, device=hidden_states.device
        )

        model = cast(VllmModelForPooling, self.model)
        raw_pooler_output: PoolerOutput = model.pooler(
            hidden_states=hidden_states, pooling_metadata=pooling_metadata
        )

        finished_mask = [
            seq_len == prompt_len
            for seq_len, prompt_len in zip(seq_lens_cpu, pooling_metadata.prompt_lens)
        ]

        model_runner_output = ModelRunnerOutput(
            req_ids=self.input_batch.req_ids.copy(),
            req_id_to_index=self.input_batch.req_id_to_index.copy(),
            kv_connector_output=kv_connector_output,
        )

        if raw_pooler_output is None or not any(finished_mask):
            model_runner_output.pooler_output = [None] * num_reqs
            return model_runner_output

        if self.use_async_scheduling:
            return AsyncGPUPoolingModelRunnerOutput(
                model_runner_output=model_runner_output,
                raw_pooler_output=raw_pooler_output,
                finished_mask=finished_mask,
                async_output_copy_stream=self.async_output_copy_stream,
            )

        raw_pooler_output = json_map_leaves(
            lambda x: None if x is None else x.to("cpu", non_blocking=True),
            raw_pooler_output,
        )
        model_runner_output.pooler_output = [
            out if include else None
            for out, include in zip(raw_pooler_output, finished_mask)
        ]
        self._sync_device()

        return model_runner_output

    def _pad_for_sequence_parallelism(self, num_scheduled_tokens: int) -> int:
        # Pad tokens to multiple of tensor_parallel_size when
        # enabled collective fusion for SP
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        if self.compilation_config.pass_config.enable_sp and tp_size > 1:
            return round_up(num_scheduled_tokens, tp_size)
        return num_scheduled_tokens

    def _prepare_mm_inputs(
        self, num_tokens: int
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        if self.model.requires_raw_input_tokens:
            input_ids = self.input_ids.gpu[:num_tokens]
        else:
            input_ids = None

        inputs_embeds = self.inputs_embeds.gpu[:num_tokens]
        return input_ids, inputs_embeds

    def _preprocess(
        self,
        scheduler_output: "SchedulerOutput",
        num_input_tokens: int,  # Padded
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
        IntermediateTensors | None,
        dict[str, Any],
        ECConnectorOutput | None,
    ]:
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        is_first_rank = get_pp_group().is_first_rank
        is_encoder_decoder = self.model_config.is_encoder_decoder

        # _prepare_inputs may reorder the batch, so we must gather multi
        # modal outputs after that to ensure the correct order
        ec_connector_output = None

        if self.supports_mm_inputs and is_first_rank and not is_encoder_decoder:
            # Run the multimodal encoder if any.
            with self.maybe_get_ec_connector_output(
                scheduler_output,
                encoder_cache=self.encoder_cache,
            ) as ec_connector_output:
                self._execute_mm_encoder(scheduler_output)
                mm_embeds, is_mm_embed = self._gather_mm_embeddings(scheduler_output)

            # NOTE(woosuk): To unify token ids and soft tokens (vision
            # embeddings), we always use embeddings (rather than token ids)
            # as input to the multimodal model, even when the input is text.
            inputs_embeds_scheduled = self.model.embed_input_ids(
                self.input_ids.gpu[:num_scheduled_tokens],
                multimodal_embeddings=mm_embeds,
                is_multimodal=is_mm_embed,
            )

            # TODO(woosuk): Avoid the copy. Optimize.
            self.inputs_embeds.gpu[:num_scheduled_tokens].copy_(inputs_embeds_scheduled)

            input_ids, inputs_embeds = self._prepare_mm_inputs(num_input_tokens)
            model_kwargs = {
                **self._init_model_kwargs(),
                **self._extract_mm_kwargs(scheduler_output),
            }
        elif self.enable_prompt_embeds and is_first_rank:
            # Get the input embeddings for the tokens that are not input embeds,
            # then put them into the appropriate positions.
            # TODO(qthequartermasterman): Since even when prompt embeds are
            # enabled, (a) not all requests will use prompt embeds, and (b)
            # after the initial prompt is processed, the rest of the generated
            # tokens will be token ids, it is not desirable to have the
            # embedding layer outside of the CUDA graph all the time. The v0
            # engine avoids this by "double compiling" the CUDA graph, once
            # with input_ids and again with inputs_embeds, for all num_tokens.
            # If a batch only has token ids, then including the embedding layer
            # in the CUDA graph will be more performant (like in the else case
            # below).
            token_ids_idx = (
                self.is_token_ids.gpu[:num_scheduled_tokens]
                .nonzero(as_tuple=False)
                .squeeze(1)
            )
            # Some tokens ids may need to become embeds
            if token_ids_idx.numel() > 0:
                token_ids = self.input_ids.gpu[token_ids_idx]
                tokens_to_embeds = self.model.embed_input_ids(input_ids=token_ids)
                self.inputs_embeds.gpu[token_ids_idx] = tokens_to_embeds

            inputs_embeds = self.inputs_embeds.gpu[:num_input_tokens]
            model_kwargs = self._init_model_kwargs()
            input_ids = None
        else:
            # For text-only models, we use token ids as input.
            # While it is possible to use embeddings as input just like the
            # multimodal models, it is not desirable for performance since
            # then the embedding layer is not included in the CUDA graph.
            input_ids = self.input_ids.gpu[:num_input_tokens]
            inputs_embeds = None
            model_kwargs = self._init_model_kwargs()

        if self.uses_mrope:
            positions = self.mrope_positions.gpu[:, :num_input_tokens]
        elif self.uses_xdrope_dim > 0:
            positions = self.xdrope_positions.gpu[:, :num_input_tokens]
        else:
            positions = self.positions.gpu[:num_input_tokens]

        if is_first_rank:
            intermediate_tensors = None
        else:
            assert intermediate_tensors is not None
            intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                num_input_tokens, intermediate_tensors, True
            )

        if is_encoder_decoder and scheduler_output.scheduled_encoder_inputs:
            # Run the encoder, just like we do with other multimodal inputs.
            # For an encoder-decoder model, our processing here is a bit
            # simpler, because the outputs are just passed to the decoder.
            # We are not doing any prompt replacement. We also will only
            # ever have a single encoder input.
            encoder_outputs = self._execute_mm_encoder(scheduler_output)
            model_kwargs.update({"encoder_outputs": encoder_outputs})

        return (
            input_ids,
            inputs_embeds,
            positions,
            intermediate_tensors,
            model_kwargs,
            ec_connector_output,
        )

    # [FA2/SM70 25.08] ФАЗОМЕР ШАГА (FA2SM70_STEP_PHASE=N -- печать каждые N шагов).
    # Замер снятием фазы показал: включение спекуляции стоит +16.7 мс НА ШАГ ещё до всякой
    # полезной работы (пустой черновик, q=2: 46.6 мс против 29.9 без спекуляции). GDN (1.3 мс) и
    # линейные тела (x1.20 при M=4) этого не объясняют -- значит статья в самом шаге движка,
    # и её надо НАЗВАТЬ, а не угадать.
    def _снять_для_головы(self, признаки: torch.Tensor, токены: torch.Tensor) -> None:
        """Кладёт (признаки, токены) префилла на диск для обучения головы MTP.

        Пишет ТОЛЬКО ранг 0: при TP=2 признаки уже сведены и на рангах одинаковы, вторая копия
        была бы мусором на 10 ГБ. Отказ записи НЕ роняет шаг -- съём это прибор, а не условие.
        """
        try:
            from vllm.distributed.parallel_state import get_tensor_model_parallel_rank as _ранг
            if _ранг() != 0:
                return
        except Exception:
            return          # не смогли опознать ранг -- лучше НЕ писать, чем писать вдвое
        try:
            import os as _os
            _os.makedirs(_ДАМП_MTP, exist_ok=True)
            _n = _ДАМП_MTP_N[0]
            _ДАМП_MTP_N[0] = _n + 1
            torch.save({"h": признаки.detach().to(torch.float16).cpu(),
                        "t": токены.detach().to(torch.int32).cpu()},
                       _os.path.join(_ДАМП_MTP, f"kusok_{_os.getpid()}_{_n:06d}.pt"))
            if _n in (0, 9, 99, 999):
                print(f"[fa2_sm70 СЪЁМ MTP] кусков {_n + 1}, позиций в этом {признаки.shape[0]}",
                      file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[fa2_sm70 СЪЁМ MTP] отказ записи: {e}", file=sys.stderr, flush=True)

    def _фаза_карты(self, имя: str):
        """ФАЗА НА КАРТЕ, а не на хозяине. Возвращает менеджер контекста.

        ЗАЧЕМ. Прежний `_фаза_шага` меряет `perf_counter` вокруг вызова, то есть время
        ВЫДАЧИ ядер: проход асинхронный, хозяин возвращается сразу. Положительный контроль
        27.08 это доказал -- вставка заведомых 5 мс внутрь прохода (`torch.cuda._sleep`)
        подняла ПЕРИОД 44.28 -> 47.01, а фазу «прямой проход цели» не сдвинула вовсе
        (31.01 -> 31.02). Значит все раскладки по фазам были про ХОЗЯИНА, а не про карту.
        Здесь время берётся СОБЫТИЯМИ CUDA -- они лежат в потоке и меряют именно карту.
        Включение: FA2SM70_STEP_PHASE_DEV=1 (вместе с FA2SM70_STEP_PHASE=N для печати).
        """
        import contextlib

        if not _ФАЗА_ДЕВ:
            return contextlib.nullcontext()

        runner = self

        @contextlib.contextmanager
        def _окно():
            пара = getattr(runner, "_соб_пул", None)
            if пара is None:
                пара = runner._соб_пул = {}
            если = пара.get(имя)
            if если is None:
                если = пара[имя] = (torch.cuda.Event(enable_timing=True),
                                    torch.cuda.Event(enable_timing=True))
            если[0].record()
            try:
                yield
            finally:
                если[1].record()
                # Читаем ПРОШЛУЮ пару: синхронизация здесь убила бы то, что меряем.
                прошл = getattr(runner, "_соб_готово", None)
                if прошл is None:
                    прошл = runner._соб_готово = {}
                гот = прошл.get(имя)
                if гот is not None and гот[1].query():
                    runner._фаза_шага("[карта] " + имя, гот[0].elapsed_time(гот[1]) / 1e3)
                прошл[имя] = если
                пара[имя] = (torch.cuda.Event(enable_timing=True),
                             torch.cuda.Event(enable_timing=True))

        return _окно()

    def _фаза_шага(self, имя: str, дт: float) -> None:
        н = _ФАЗА_ШАГА_N
        if not н:
            return
        д = getattr(self, "_фазы_шага", None)
        if д is None:
            д = self._фазы_шага = {}
            self._фазы_шага_счёт = 0
        д[имя] = д.get(имя, 0.0) + дт * 1e3
        if имя == "выборка+отбраковка":
            self._фазы_шага_счёт += 1
            if self._фазы_шага_счёт % н == 0:
                c = н  # ОКНО, а не вся история: иначе средние загрязнены префилльными шагами,
                # где проход в разы дороже декодного (проверено -- расхождение вчетверо).
                стр = "  ".join(f"{и}={в / c:.2f}" for и, в in sorted(д.items(), key=lambda x: -x[1]))
                logger.info(
                    "[fa2_sm70 ФАЗЫ ШАГА, мс/шаг, окно %d] %s | ПЕРИОД макс=%.1f сумма=%.2f с за %d заходов",
                    c, стр, getattr(self, "_пер_макс", 0.0) * 1e3,
                    getattr(self, "_пер_сумма", 0.0), getattr(self, "_пер_счёт", 0),
                )
                self._пер_макс = 0.0
                д.clear()

    def _sample(
        self,
        logits: torch.Tensor | None,
        spec_decode_metadata: SpecDecodeMetadata | None,
    ) -> SamplerOutput:
        # Sample the next token and get logprobs if needed.
        sampling_metadata = self.input_batch.sampling_metadata
        # Update output token ids with tokens sampled in last step
        # if async scheduling and required by current sampling params.
        self.input_batch.update_async_output_token_ids()
        if spec_decode_metadata is None:
            return self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )

        # Update spec_token_ids with real draft tokens from pre step only when
        # output_token_ids is needed (penalties or bad_words are in use).
        if self.use_async_scheduling and self._draft_token_req_ids is not None:
            draft_token_ids_cpu, _ = self._get_draft_token_ids_cpu()
            self.input_batch.update_async_spec_token_ids(draft_token_ids_cpu)

        # [СБОР ЛОГИТОВ ЦЕЛИ ДЛЯ ДИСТИЛЛЯЦИИ ЧЕРНОВИКА -- 01.09]
        # Обучение по ПРИНЯТЫМ ТОКЕНАМ вышло на плато (записка 25 §94d): один правильный ответ
        # на позицию несёт слишком мало сигнала. Черновик обязан воспроизводить РАСПРЕДЕЛЕНИЕ
        # цели, поэтому собираем top-k её логитов -- 64 значения с индексами это 512 байт на
        # позицию, то есть корпус ДЕШЕВЛЕ нынешнего. Ветка мертва без переменной.
        if _ЛОГИТЫ_ТОП:
            try:
                _зн, _ид = logits.topk(_ЛОГИТЫ_ТОП, dim=-1)
                self._fa2_логиты = (_зн.detach().to("cpu", torch.float16),
                                    _ид.detach().to("cpu", torch.int32))
            except Exception:
                self._fa2_логиты = None
        _смд = spec_decode_metadata
        if _ДЕРЕВО_W > 0:
            _смд = self._дерево_ветвь(spec_decode_metadata, logits) or spec_decode_metadata
        sampler_output = self.rejection_sampler(
            _смд,
            None,  # draft_probs
            logits,
            sampling_metadata,
        )
        return sampler_output

    def _дерево_ветвь(self, smd, logits):
        """[ДЕРЕВО, ШАГ 3] ВЫБОР ВЕТВИ ДО ПРИЁМКИ.

        Приёмка vLLM линейна по построению: ядро идёт по цепи и встаёт на первом несовпадении.
        Ветвление в неё не вписать -- но и не нужно. Черновик отдаёт 2W токенов как ДВЕ ветви
        по W (ветвление на позиции 0), а здесь выбирается та ветвь, чей первый токен совпал с
        выбором цели, и приёмке подаётся уже ЦЕПЬ. Собственное ядро приёмки не требуется.

        Приёмка сама отбирает строки по `target_logits_indices`, поэтому переставляются ИНДЕКСЫ,
        а тензор логитов не трогается вовсе.
        """
        W = _ДЕРЕВО_W
        nd = smd.num_draft_tokens
        if not nd or any(n != 2 * W for n in nd):
            return None  # форма не наша (префилл, разнобой) -- идём цепью
        B = len(nd)
        dev = logits.device
        tli = smd.target_logits_indices.view(B, 2 * W)
        dti = smd.draft_token_ids.view(B, 2 * W)
        # Что цель хочет на позиции 0. Строка tli[:,0] -- та самая позиция; у ветви B строка
        # tli[:,W] стоит на ТОЙ ЖЕ позиции и с тем же контекстом (b0 видит лишь якорь).
        t0 = logits[tli[:, 0]].argmax(dim=-1).to(dti.dtype)
        берём_B = (dti[:, W] == t0) & (dti[:, 0] != t0)
        if _ДЕРЕВО_СЧЁТ:
            # ФАЛЬСИФИКАТОР ПРИЗА: считаем, ЧТО дало бы дерево, ничего не меняя в выдаче.
            # Три числа: шагов всего; шагов, где ветвь A угадала позицию 0; шагов, где её
            # угадала ТОЛЬКО ветвь B (это и есть чистая прибавка дерева).
            self._дер_шагов = getattr(self, "_дер_шагов", 0) + B
            self._дер_A = getattr(self, "_дер_A", 0) + int((dti[:, 0] == t0).sum())
            self._дер_B = getattr(self, "_дер_B", 0) + int(берём_B.sum())
            if self._дер_шагов % _ДЕРЕВО_СЧЁТ < B:
                n = self._дер_шагов
                import sys as _s
                print(f"[ДЕРЕВО-ПРИЗ {n} shagov] A_ugadala={self._дер_A / n * 100:.1f}%  "
                      f"tolko_B={self._дер_B / n * 100:.1f}%  "
                      f"vmeste={(self._дер_A + self._дер_B) / n * 100:.1f}%",
                      file=_s.stderr, flush=True)
        if _ДЕРЕВО_ФАЙЛ:
            # Переключатель без перезапуска: 0 = только ветвь A, 1 = дерево с выбором.
            self._дер_опрос = getattr(self, "_дер_опрос", 0) + 1
            if self._дер_опрос % 24 == 1:
                # РЕЖИМ -- БИТОВАЯ МАСКА, ЧТОБЫ БИСЕКЦИЯ ШЛА В ОДНОМ ПОДЪЁМЕ.
                # Качество сравнимо только внутри одного подъёма (тюнер GEMM
                # свой у каждого), а рычагов у дерева три. Поэтому файл несёт
                # число: бит0 -- ветвь B вообще; бит1 -- ЗАПРЕТ подмены conv-окна;
                # бит2 -- ЗАПРЕТ копии KV; бит3 -- ЗАПРЕТ off-сдвига чтения SSM.
                try:
                    with open(_ДЕРЕВО_ФАЙЛ) as _f:
                        self._дер_реж = int(_f.read(8).strip() or "0")
                except Exception:
                    self._дер_реж = 0
            if not (getattr(self, "_дер_реж", 0) & 1):
                берём_B = torch.zeros_like(берём_B)
        elif _ДЕРЕВО_ТОЛЬКО_A:
            берём_B = torch.zeros_like(берём_B)
        # БИТ 7 -- ПРИНУДИТЕЛЬНОЕ ВЕТВЛЕНИЕ ДЛЯ ГЕЙТА ТОЖДЕСТВА. При FA2SM70_TREE_GATE=1
        # ветвь B несёт ТОТ ЖЕ токен, что и A, и строки совпадают побайтово (гейт
        # прошёл: max|a-b|=0). Тогда подмена окна пишет ЗАВЕДОМО ТО ЖЕ содержимое, и
        # порча означала бы, что вредит САМО действие (адрес, поток, время), а не
        # содержимое. Без принуждения условие (a0 != t0) на тождестве не выполнимо.
        _рж = getattr(self, "_дер_реж", 0) if _ДЕРЕВО_ФАЙЛ else 0
        if (_рж & 1) and (_рж & 128):
            берём_B = (dti[:, W] == t0)
        if _ДЕРЕВО_КВ:
            self._дерево_принять_B(берём_B, W, B, smd)
        # РАСКЛАДКА СТРОК ШАГА: [якорь, a0..a(W-1), b0..b(W-1)], и строка i даёт предсказание
        # ПОСЛЕ токена этой строки. Отсюда:
        #   ветвь A: позиции берут строки [якорь, a0, a1] = tli[0], tli[1], tli[2]; бонус tli[W]
        #   ветвь B: позиции берут строки [якорь, b0, b1] = tli[0], tli[W+1], tli[W+2];
        #            бонус -- исходный bonus_logits_indices (строка после b(W-1))
        # ПЕРВЫЙ элемент у обеих ветвей ОДИН И ТОТ ЖЕ -- строка якоря: обе ветви стоят на
        # позиции 0 и делят её контекст. Прежняя редакция брала для B [W, W+1, W+2], то есть
        # сверяла b0 с предсказанием ПОСЛЕ a(W-1); гейт поймал это как расхождение 2/5.
        # Бонус тоже был исходным при обеих ветвях -- при ветви A он предсказан в контексте
        # всех 2W строк, а обязан быть предсказанием после a(W-1). Это и портило «всегда A».
        ряд = torch.arange(W, device=dev, dtype=torch.long).view(1, W)
        сдв = torch.where(ряд == 0, 0, ряд + torch.where(берём_B, W, 0).view(B, 1))
        бонус = torch.where(берём_B, smd.bonus_logits_indices, tli[:, W])
        return SpecDecodeMetadata(
            draft_token_ids=dti.gather(1, torch.where(
                ряд == 0, torch.where(берём_B, W, 0).view(B, 1), ряд + torch.where(
                    берём_B, W, 0).view(B, 1))).reshape(-1).contiguous(),
            num_draft_tokens=[W] * B,
            cu_num_draft_tokens=torch.arange(
                1, B + 1, device=dev, dtype=smd.cu_num_draft_tokens.dtype
            ) * W,
            cu_num_sampled_tokens=smd.cu_num_sampled_tokens,
            target_logits_indices=tli.gather(1, сдв).reshape(-1).contiguous(),
            bonus_logits_indices=бонус.contiguous(),
            logits_indices=smd.logits_indices,
        )

    def _дерево_принять_B(self, берём_B, W, B, smd) -> None:
        """[ДЕРЕВО, 05.09] Делает принятие ветви B корректным: off-буфер + слот-копия KV.

        Состояния GDN НЕ копируются вовсе: ядро рекуррента уже ветвится (branch_at), а
        чтение следующего шага направляется сдвигом num_accepted на W (off-буфер, который
        читает gdn_attn.build). Копируется только K/V полного внимания: строки ветви B
        писали в слоты физических позиций ctx+W+1+j, а контекст следующего шага собирается
        из слотов ctx+1+j.
        """
        import sys as _s8
        try:
            буф = getattr(self, "_дерево_off", None)
            if буф is None:
                from vllm.model_executor.models import qwen3_next as _qn
                буф = _qn.ДЕРЕВО_OFF.get("буф")      # создан билдером ДО захвата графов
                if буф is None:
                    буф = torch.zeros(self.max_num_reqs, dtype=torch.int32,
                                      device=берём_B.device)
                    _qn.ДЕРЕВО_OFF["буф"] = буф
                self._дерево_off = буф
            _реж = getattr(self, "_дер_реж", 1) if _ДЕРЕВО_ФАЙЛ else 1
            буф.zero_()
            if _ДЕРЕВО_OFF_ВКЛ and not (_реж & 8):
                буф[:B] = берём_B.to(torch.int32) * W
            self._дерево_зов = getattr(self, "_дерево_зов", 0) + 1
            _дн = self._дерево_зов <= 8
            if not _ДЕРЕВО_КВ_КОПИЯ:
                return
            _без_кв = bool(_реж & 4)      # запрет копии KV -- рычаг бисекции
            _без_окна = bool(_реж & 2)    # запрет подмены conv-окна
            слотыг = getattr(self, "_дерево_слотыг", None)
            if not слотыг:
                if _дн: print("[ДЕРЕВО ?] слотыг пуст", file=_s8.stderr, flush=True)
                return
            # ГРУППЫ -- ПО ТИПУ СПЕКА, НЕ ПО НОМЕРУ. Прибор показал: группа 0 здесь --
            # linear_attn (GDN, 48 слоёв), а полное внимание -- другая группа. Номер один
            # раз находится по имени класса спека и запоминается.
            _гид_attn = getattr(self, "_дерево_гид_attn", None)
            if _гид_attn is None:
                _гид_attn = -1
                for _g, _гр in enumerate(self.kv_cache_config.kv_cache_groups):
                    if "FullAttention" in type(_гр.kv_cache_spec).__name__:
                        _гид_attn = _g
                        break
                self._дерево_гид_attn = _гид_attn
                self._дерево_гиды_гдн = [
                    _g for _g, _гр in enumerate(self.kv_cache_config.kv_cache_groups)
                    if "Mamba" in type(_гр.kv_cache_spec).__name__]
                print(f"[ДЕРЕВО KV] группы: полное внимание={_гид_attn} "
                      f"GDN={self._дерево_гиды_гдн}", file=_s8.stderr, flush=True)
            if _гид_attn < 0:
                return
            sm = слотыг.get(_гид_attn)
            if sm is None:
                if _дн: print(f"[ДЕРЕВО ?] нет группы {_гид_attn}, ключи={list(слотыг)[:5]}",
                              file=_s8.stderr, flush=True)
                return
            q = 2 * W + 1
            if sm.shape[0] < B * q:
                if _дн: print(f"[ДЕРЕВО ?] sm короток: {tuple(sm.shape)} < {B}x{q}",
                              file=_s8.stderr, flush=True)
                return
            if _дн:
                try:
                    from vllm.model_executor.models.qwen3_next import ДЕРЕВО_OFF as _ДО9
                    _дг9 = _ДО9.get("дозор") or {}
                    _нп = (f" nacc(об/ssm)={int(_дг9['nacc'][0])}/{int(_дг9['nacc'][1])}"
                           if "nacc" in _дг9 else "")
                except Exception:
                    _нп = ""
                print(f"[ДЕРЕВО ?] зов#{self._дерево_зов}: B={B} W={W} "
                      f"беру={int(берём_B.sum())}{_нп}", file=_s8.stderr, flush=True)
            # ДОШЁЛ ЛИ OFF-СДВИГ ДО ЯДРА: печатаем пару (nacc, nacc_ssm) на ШАГЕ
            # ПОСЛЕ принятия ветви B -- именно там сдвиг обязан дать m-1+W. Дозор
            # заполняется ВНУТРИ графа, поэтому это показание самого ядра, а не
            # намерения строителя.
            смв = sm[: B * q].view(B, q)
            src = смв[:, W + 1: 2 * W + 1][берём_B]      # слоты строк ветви B
            dst = смв[:, 1: W + 1][берём_B]              # слоты строк ветви A
            if src.numel() == 0:
                return
            src = src.reshape(-1); dst = dst.reshape(-1)
            кэши = getattr(self, "_дерево_кэши", None)
            if кэши is None:
                кэши = []
                try:
                    гр0 = self.kv_cache_config.kv_cache_groups[_гид_attn]
                    ктх = self.compilation_config.static_forward_context
                    for имя in гр0.layer_names:
                        сл = ктх.get(имя)
                        кв = getattr(сл, "kv_cache", None)
                        if кв is not None and len(кв) > 0 and torch.is_tensor(кв[0]):
                            кэши.append(кв[0])
                    print(f"[ДЕРЕВО KV] сборка: слоёв гр0={len(гр0.layer_names)} "
                          f"собрано={len(кэши)} пример={гр0.layer_names[:1]} "
                          f"тип_кв={type(getattr(ктх.get(гр0.layer_names[0]), 'kv_cache', None))}",
                          file=_s8.stderr, flush=True)
                except Exception as _e:
                    print(f"[ДЕРЕВО KV] кэши не собраны: {_e}", file=_s8.stderr, flush=True)
                self._дерево_кэши = кэши
            if not кэши:
                return
            if not _без_кв:
                for кв in кэши:
                    bs = кв.shape[2]
                    кв[dst // bs, :, dst % bs] = кв[src // bs, :, src % bs]
            self._дерево_событий = getattr(self, "_дерево_событий", 0) + 1
            if self._дерево_событий <= 5:
                print(f"[ДЕРЕВО ПРИНЯТИЕ] #{self._дерево_событий}: строкB={int(берём_B.sum())} "
                      f"kv_слоёв={len(кэши)} src={src[:4].tolist()} dst={dst[:4].tolist()}",
                      file=_s8.stderr, flush=True)
            # --- СВЁРТОЧНОЕ ОКНО GDN: подмена из ЧЕРНОВОГО слота -----------------------
            # Свёртка ветви B посчитана ВТОРЫМ вызовом в черновой слот (qwen3_next,
            # ДЕРЕВО_OFF["черн"]): его окно -- ровно окно принятой цепи [.., якорь, b0, b1].
            # При берём_B оно копируется в основной слот, и следующий шаг живёт обычной
            # цепной арифметикой nacc=m (интеграционный тест test_derevo_conv.py:
            # relL2 ~6e-08 на ветвях A/B, m=1..3). Прежняя копия якоря в столбец +W --
            # ОТМЕНЕНА тем же тестом: ряд не выражается точечной правкой.
            try:
                from vllm.model_executor.models.qwen3_next import ДЕРЕВО_OFF as _ДО2
                _реестр = _ДО2.get("черн") or {}
                # СЛОТЫ -- ПО ГРУППАМ: у каждой KV-группы GDN своя таблица, и слот того же
                # запроса в другой группе другой. Слои сопоставляются группе по имени.
                _ктоB = берём_B.nonzero(as_tuple=True)[0]
                if (_ДЕРЕВО_CONV_SWAP and not _без_окна
                        and _ктоB.numel() and _реестр):
                    for гид in getattr(self, "_дерево_гиды_гдн", []):
                        гр = self.attn_groups[гид][0]
                        стр = гр.metadata_builders[0] if гр.metadata_builders else None
                        таб = getattr(стр, "spec_state_indices_tensor", None)
                        if таб is None or таб.shape[0] < B:
                            continue
                        _слоты0 = таб[:B, 0].to(torch.long)[берём_B]
                        for имя in гр.layer_names:
                            з2 = _реестр.get(имя)
                            if з2 is None:
                                continue
                            _черн, _осн = з2
                            # СВОП ОКНА ЗАМЕНЁН ТОЧЕЧНОЙ КОПИЕЙ. Полная подмена окна из
                            # чернового слота, будучи верной по содержимому (дозор это
                            # показал), в бою ДОБАВЛЯЛА порчу (136 «!» против 8 без неё) --
                            # взаимодействие 5- и 7-элементной семантик окна между шагами.
                            # Здесь ряд чинится в САМОМ основном окне: столбцы строк ветви B
                            # (sl-T+W+1+j) копируются на места строк ветви A (sl-T+1+j) --
                            # тогда ряды свёртки следующего шага при обычном nacc=m верны
                            # для всех m (арифметика та же, что в test_derevo_conv.py).
                            # ЗОНДЫ ПРИЧИНЫ (бит4 -- запись САМОГО СЕБЯ, бит5 --
                            # только читаемые позиции 2..W+2). Ядро читает историю
                            # окно[m-1:m+2] при m<=W+1, то есть дальше индекса W+2
                            # не заглядывает НИКОГДА (замер sem_read.py). Если
                            # тождественная запись тоже портит -- виновата САМА
                            # запись, а не содержимое.
                            _осн[_слоты0] = _черн[_ктоB]
                            self._дерево_конв_ок = getattr(self, "_дерево_конв_ок", 0) + 1
                if getattr(self, "_дерево_событий", 0) <= 5 and _ктоB.numel():
                    print(f"[ДЕРЕВО conv] окон переложено (слоёв всего): "
                          f"{getattr(self, '_дерево_конв_ок', 0)}",
                          file=_s8.stderr, flush=True)
            except Exception as _e:
                self._дерево_ошибка_конв = getattr(self, "_дерево_ошибка_конв", 0) + 1
                if self._дерево_ошибка_конв <= 3:
                    print(f"[ДЕРЕВО conv] отказ: {type(_e).__name__}: {_e}",
                          file=_s8.stderr, flush=True)
        except Exception as _e:
            self._дерево_ошибка = getattr(self, "_дерево_ошибка", 0) + 1
            if self._дерево_ошибка <= 3:
                print(f"[ДЕРЕВО KV] отказ: {type(_e).__name__}: {_e}",
                      file=_s8.stderr, flush=True)

    def _bookkeeping_sync(
        self,
        scheduler_output: "SchedulerOutput",
        sampler_output: SamplerOutput,
        logits: torch.Tensor | None,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: int,
        spec_decode_metadata: SpecDecodeMetadata | None,
    ) -> tuple[
        dict[str, int],
        LogprobsLists | None,
        list[list[int]],
        dict[str, LogprobsTensors | None],
        list[str],
        dict[str, int],
        list[int],
    ]:
        num_nans_in_logits = {}
        if envs.VLLM_COMPUTE_NANS_IN_LOGITS:
            num_nans_in_logits = self._get_nans_in_logits(logits)

        num_reqs = self.input_batch.num_reqs
        discard_sampled_tokens_req_indices = np.nonzero(
            self.discard_request_mask.np[:num_reqs]
        )[0]
        for i in discard_sampled_tokens_req_indices:
            gen = self.input_batch.generators.get(int(i))
            if gen is not None:
                gen.set_offset(gen.get_offset() - 4)

        # Copy some objects so they don't get modified after returning.
        # This is important when using async scheduling.
        req_ids_output_copy = self.input_batch.req_ids.copy()
        req_id_to_index_output_copy = self.input_batch.req_id_to_index.copy()

        num_sampled_tokens = sampler_output.sampled_token_ids.shape[0]
        sampled_token_ids = sampler_output.sampled_token_ids
        logprobs_tensors = sampler_output.logprobs_tensors
        invalid_req_indices = []
        logprobs_lists = None
        if not self.use_async_scheduling:
            # Get the valid generated tokens.
            max_gen_len = sampled_token_ids.shape[-1]
            if max_gen_len == 1:
                # No spec decode tokens.
                valid_sampled_token_ids = self._to_list(sampled_token_ids)
                # Mask out the sampled tokens that should not be sampled.
                for i in discard_sampled_tokens_req_indices:
                    valid_sampled_token_ids[int(i)].clear()

                if logprobs_tensors is not None:
                    logprobs_lists = logprobs_tensors.tolists()
            else:
                # Includes spec decode tokens.
                valid_sampled_token_ids, logprobs_lists = RejectionSampler.parse_output(
                    sampled_token_ids,
                    self.input_batch.vocab_size,
                    discard_sampled_tokens_req_indices,
                    logprobs_tensors=logprobs_tensors,
                )
        else:
            valid_sampled_token_ids = []
            invalid_req_indices = discard_sampled_tokens_req_indices.tolist()
            invalid_req_indices_set = set(invalid_req_indices)

            # Cache the sampled tokens on the GPU and avoid CPU sync.
            # These will be copied into input_ids in the next step
            # when preparing inputs.
            # With spec decoding, this is done in propose_draft_token_ids().
            if self.input_batch.prev_sampled_token_ids is None:
                assert sampled_token_ids.shape[-1] == 1
                self.input_batch.prev_sampled_token_ids = sampled_token_ids
            self.input_batch.prev_req_id_to_index = {
                req_id: i
                for i, req_id in enumerate(self.input_batch.req_ids)
                if i not in invalid_req_indices_set
            }

        # Cache the sampled tokens in the model runner, so that the scheduler
        # doesn't need to send them back.
        # NOTE(woosuk): As an exception, when using PP, the scheduler sends
        # the sampled tokens back, because there's no direct communication
        # between the first-stage worker and the last-stage worker.
        req_ids = self.input_batch.req_ids
        for req_idx in range(num_sampled_tokens):
            if self.use_async_scheduling:
                sampled_ids = [-1] if req_idx not in invalid_req_indices_set else None
            else:
                sampled_ids = valid_sampled_token_ids[req_idx]

            num_sampled_ids: int = len(sampled_ids) if sampled_ids else 0

            if not sampled_ids:
                continue

            start_idx = self.input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + num_sampled_ids
            assert end_idx <= self.max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self.max_model_len}"
            )

            self.input_batch.token_ids_cpu[req_idx, start_idx:end_idx] = sampled_ids
            self.input_batch.is_token_ids[req_idx, start_idx:end_idx] = True
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx

            req_id = req_ids[req_idx]
            req_state = self.requests[req_id]
            req_state.output_token_ids.extend(sampled_ids)

        # Compute prompt logprobs if needed.
        prompt_logprobs_dict = self._get_prompt_logprobs_dict(
            hidden_states[:num_scheduled_tokens],
            scheduler_output.num_scheduled_tokens,
        )

        return (
            num_nans_in_logits,
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        )

    @contextmanager
    def synchronize_input_prep(self):
        if self.prepare_inputs_event is None:
            yield
            return

        # Ensure prior step has finished with reused CPU tensors.
        # This is required in the async scheduling case because
        # the CPU->GPU transfer happens async.
        self.prepare_inputs_event.synchronize()
        try:
            yield
        finally:
            self.prepare_inputs_event.record()

    def _model_forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs: dict[str, Any],
    ) -> Any:
        """Helper method to call the model forward pass.

        This method can be overridden by subclasses for model execution.
        Motivation: We can inspect only this method versus
        the whole execute_model, which has additional logic.

        Args:
            input_ids: Input token IDs
            positions: Token positions
            intermediate_tensors: Tensors from previous pipeline stages
            inputs_embeds: Input embeddings (alternative to input_ids)
            **model_kwargs: Additional model arguments

        Returns:
            Model output tensor
        """
        # [fa2_sm70] МЕГАЯДРО ВЫШЕ КОМПИЛЯЦИИ. Внутри скомпилированного тела не проходит ничего,
        # кроме чистых вычислений (проверено шестью отказами: мутации, атрибут функции, логгер).
        # Здесь -- обычный питон, поэтому и ветвление, и наблюдение разрешены. Метод для этого и
        # заведён авторами ("we can inspect only this method versus the whole execute_model").
        import os as _os
        if _os.environ.get("FA2SM70_MEGA") == "1":
            try:
                import fa2_sm70.megastep as _ms
                # РАННЕР ОТДАЁТСЯ ЯВНО: ярус Я5 (весь шаг одним пуском, включая сэмплер) не может
                # быть построен, пока видна только модель -- сэмплер и голова живут на раннере.
                _ms._РАННЕР = self
                _r = _ms.попробовать_проход(self.model, input_ids, positions,
                                            intermediate_tensors, inputs_embeds)
                if _r is not None:
                    return _r
            except Exception:
                raise                    # НЕ глотать: тихий откат скрыл бы неработающую цепь
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )

    @staticmethod
    def _is_uniform_decode(
        max_num_scheduled_tokens: int,
        uniform_decode_query_len: int,
        num_tokens: int,
        num_reqs: int,
        force_uniform_decode: bool | None = None,
    ) -> bool:
        """
        Checks if it's a decode batch with same amount scheduled tokens
        across all requests.
        """
        return (
            (
                (max_num_scheduled_tokens == uniform_decode_query_len)
                and (num_tokens == max_num_scheduled_tokens * num_reqs)
            )
            if force_uniform_decode is None
            else force_uniform_decode
        )

    def _determine_batch_execution_and_padding(
        self,
        num_tokens: int,
        num_reqs: int,
        num_scheduled_tokens_np: np.ndarray,
        max_num_scheduled_tokens: int,
        use_cascade_attn: bool,
        allow_microbatching: bool = True,
        force_eager: bool = False,
        # For cudagraph capture TODO(lucas): Refactor how we capture cudagraphs (will
        # be improved in model runner v2)
        force_uniform_decode: bool | None = None,
        force_has_lora: bool | None = None,
        force_num_active_loras: int | None = None,
        num_encoder_reqs: int = 0,
    ) -> tuple[
        CUDAGraphMode,
        BatchDescriptor,
        bool,
        torch.Tensor | None,
        CUDAGraphStat | None,
    ]:
        uniform_decode = self._is_uniform_decode(
            max_num_scheduled_tokens=max_num_scheduled_tokens,
            uniform_decode_query_len=self.uniform_decode_query_len,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            force_uniform_decode=force_uniform_decode,
        )
        # Encoder-decoder models only support CG for decoder_step > 0 (no enc_output
        # is present). Also, chunked-prefill is disabled, so batch are uniform.
        has_encoder_output = (
            self.model_config.is_encoder_decoder and num_encoder_reqs > 0
        )

        # Compute LoRA state for cudagraph dispatch
        num_active_loras = (
            force_num_active_loras
            if force_num_active_loras is not None
            else len(self.input_batch.lora_id_to_lora_request)
        )
        has_lora = num_active_loras > 0 if force_has_lora is None else force_has_lora

        num_tokens_padded = self._pad_for_sequence_parallelism(num_tokens)
        dispatch_cudagraph = (
            lambda num_tokens, disable_full: self.cudagraph_dispatcher.dispatch(
                num_tokens=num_tokens,
                has_lora=has_lora,
                uniform_decode=uniform_decode,
                disable_full=disable_full,
                num_active_loras=num_active_loras,
            )
            if not force_eager
            else (CUDAGraphMode.NONE, BatchDescriptor(num_tokens_padded))
        )

        cudagraph_mode, batch_descriptor = dispatch_cudagraph(
            num_tokens_padded, use_cascade_attn or has_encoder_output
        )
        # [ПРОБА РЕЖИМА ГРАФА, 07.09] Пофазный секундомер показал: прямой проход цели
        # со спекуляцией 36.3 мс при работе GPU ~19 -- подпись пути БЕЗ полного графа.
        # Здесь печатается решение диспетчера: какой режим и на сколько токенов.
        if os.environ.get("FA2SM70_CG_PROBE", "0") == "1":
            _к = (str(cudagraph_mode), int(batch_descriptor.num_tokens), bool(uniform_decode))
            _п = getattr(self, "_реж_графа", None)
            if _п is None:
                _п = self._реж_графа = {}
            _п[_к] = _п.get(_к, 0) + 1
            if sum(_п.values()) in (5, 50) or sum(_п.values()) % 200 == 0:
                print("[РЕЖИМ ГРАФА] " + "  ".join(
                    f"{м}/ток={т}/uniform={u}: {n}" for (м, т, u), n in
                    sorted(_п.items(), key=lambda x: -x[1])[:5]),
                    file=__import__("sys").stderr, flush=True)
        num_tokens_padded = batch_descriptor.num_tokens
        if self.compilation_config.pass_config.enable_sp:
            assert (
                batch_descriptor.num_tokens
                % self.vllm_config.parallel_config.tensor_parallel_size
                == 0
            ), (
                "Sequence parallelism requires num_tokens to be "
                "a multiple of tensor parallel size"
            )

        # Extra coordination when running data-parallel since we need to coordinate
        # across ranks
        should_ubatch, num_tokens_across_dp = False, None
        if self.vllm_config.parallel_config.data_parallel_size > 1:
            # Disable DP padding when running eager to avoid excessive padding when
            # running prefills. This lets us set cudagraph_mode="NONE" on the prefiller
            # in a P/D setup and still use CUDA graphs (enabled by this padding) on the
            # decoder.
            allow_dp_padding = (
                self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
            )

            should_ubatch, num_tokens_across_dp, synced_cudagraph_mode = (
                coordinate_batch_across_dp(
                    num_tokens_unpadded=num_tokens,
                    parallel_config=self.parallel_config,
                    allow_microbatching=allow_microbatching,
                    allow_dp_padding=allow_dp_padding,
                    num_tokens_padded=num_tokens_padded,
                    uniform_decode=uniform_decode,
                    num_scheduled_tokens_per_request=num_scheduled_tokens_np,
                    cudagraph_mode=cudagraph_mode.value,
                )
            )

            # Extract DP-synced values
            if num_tokens_across_dp is not None:
                dp_rank = self.parallel_config.data_parallel_rank
                num_tokens_padded = int(num_tokens_across_dp[dp_rank].item())
                # Re-dispatch with DP padding so we have the correct batch_descriptor
                cudagraph_mode, batch_descriptor = dispatch_cudagraph(
                    num_tokens_padded,
                    disable_full=synced_cudagraph_mode <= CUDAGraphMode.PIECEWISE.value,
                )
                # Assert to make sure the agreed upon token count is correct otherwise
                # num_tokens_across_dp will no-longer be valid
                assert batch_descriptor.num_tokens == num_tokens_padded

        cudagraph_stats = None
        if self.vllm_config.observability_config.cudagraph_metrics:
            cudagraph_stats = CUDAGraphStat(
                num_unpadded_tokens=num_tokens,
                num_padded_tokens=batch_descriptor.num_tokens,
                num_paddings=batch_descriptor.num_tokens - num_tokens,
                runtime_mode=str(cudagraph_mode),
            )

        return (
            cudagraph_mode,
            batch_descriptor,
            should_ubatch,
            num_tokens_across_dp,
            cudagraph_stats,
        )

    def _register_layerwise_nvtx_hooks(self) -> None:
        """
        Register layerwise NVTX hooks if --enable-layerwise-nvtx-tracing is enabled
        to trace detailed information of each layer or module in the model.
        """

        if (
            self.vllm_config.observability_config.enable_layerwise_nvtx_tracing
            and not self.layerwise_nvtx_hooks_registered
        ):
            if self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
                logger.debug_once(
                    "layerwise NVTX tracing is not supported when CUDA graph is "
                    "turned off; you may observe part or all of the model "
                    "missing NVTX markers"
                )

            # In STOCK_TORCH_COMPILE mode, after registering hooks here,
            # the __call__ function of nn.module will be recompiled with
            # fullgraph=True. Since nvtx.range_push/pop are not traceable
            # by torch dynamo, we can't register hook functions here
            # because hook functions will also be traced by torch dynamo.
            if (
                self.vllm_config.compilation_config.mode
                == CompilationMode.STOCK_TORCH_COMPILE
            ):
                logger.debug_once(
                    "layerwise NVTX tracing is not supported when "
                    "CompilationMode is STOCK_TORCH_COMPILE, skipping "
                    "function hooks registration"
                )
            else:
                pyt_hooks = PytHooks()
                pyt_hooks.register_hooks(self.model, self.model.__class__.__name__)
                self.layerwise_nvtx_hooks_registered = True

    def _get_slot_mappings(
        self,
        num_tokens_padded: int,
        num_reqs_padded: int,
        num_tokens_unpadded: int,
        ubatch_slices: "UBatchSlices | None" = None,
    ) -> tuple[
        dict[int, torch.Tensor] | None,
        dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None,
    ]:
        """
        Build slot mappings in both formats needed by the system.

        Args:
            num_tokens_padded: Total number of tokens (padded)
            num_reqs_padded: Total number of requests (padded)
            num_tokens_unpadded: Actual number of tokens (unpadded)
            ubatch_slices: Optional ubatch slicing info for DBO

        Returns:
            A tuple of:
            - slot_mappings_by_gid: dict[int, torch.Tensor] for attention metadata
            - slot_mappings_by_layer: dict[str, torch.Tensor] or list for ForwardContext
        """
        if not (
            hasattr(self, "kv_cache_config")
            and self.kv_cache_config is not None
            and len(self.kv_cache_config.kv_cache_groups) > 0
        ):
            return None, None

        def _get_slot_mapping(kv_cache_gid: int):
            assert num_reqs_padded is not None and num_tokens_padded is not None
            kv_cache_spec = self.kv_cache_config.kv_cache_groups[
                kv_cache_gid
            ].kv_cache_spec
            if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
                slot_mapping = torch.zeros(
                    (num_tokens_padded,),
                    dtype=torch.int64,
                    device=self.device,
                )
            else:
                blk_table = self.input_batch.block_table[kv_cache_gid]
                slot_mapping = blk_table.slot_mapping.gpu[:num_tokens_padded]

            # Fill unused with -1. Needed for reshape_and_cache in full cuda
            # graph mode. `blk_table_tensor` -1 to match mamba PAD_SLOT_ID
            slot_mapping[num_tokens_unpadded:num_tokens_padded].fill_(-1)

            return slot_mapping

        slot_mappings_by_gid = {
            gid: _get_slot_mapping(gid)
            for gid, _ in enumerate(self.kv_cache_config.kv_cache_groups)
        }

        slot_mappings_by_layer: dict[str, torch.Tensor] = {}
        for gid, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups):
            slot_mapping = slot_mappings_by_gid[gid]
            for layer_name in kv_cache_group.layer_names:
                slot_mappings_by_layer[layer_name] = slot_mapping

        if ubatch_slices is not None:
            result: list[dict[str, torch.Tensor]] = []
            for ubatch in ubatch_slices:
                sliced_mappings: dict[str, torch.Tensor] = {}
                for layer_name, slot_mapping in slot_mappings_by_layer.items():
                    sliced_mappings[layer_name] = slot_mapping[ubatch.token_slice]
                result.append(sliced_mappings)
            return slot_mappings_by_gid, result

        return slot_mappings_by_gid, slot_mappings_by_layer

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors | None:
        if _ФАЗА_ШАГА_N:
            # ПРОМЕЖУТОК МЕЖДУ ШАГАМИ: воркер простаивает, работает движок/планировщик.
            _т = time.perf_counter()
            if getattr(self, "_t_конец_шага", None) is not None:
                self._фаза_шага("МЕЖДУ шагами (движок)", _т - self._t_конец_шага)
            # ПЕРИОД МЕЖДУ ВХОДАМИ -- однозначная мера: сколько идёт ПОЛНЫЙ цикл шага, включая всё,
            # что делается вне воркера. Разность «период минус воркер» и есть искомое.
            if getattr(self, "_t_вход_пред", None) is not None:
                _пер = _т - self._t_вход_пред
                self._фаза_шага("ПЕРИОД (вход->вход)", _пер)
                # МАКСИМУМ И СУММА: среднее занижается, если между настоящими шагами движок делает
                # короткие холостые заходы. Сумма периодов -- это СТЕНОЧНОЕ время окна, её и надо
                # сверять с секундомером снаружи.
                self._пер_макс = max(getattr(self, "_пер_макс", 0.0), _пер)
                self._пер_сумма = getattr(self, "_пер_сумма", 0.0) + _пер
                self._пер_счёт = getattr(self, "_пер_счёт", 0) + 1
            self._t_вход_пред = _т
            self._t_шаг = _т
        if _ПОДРЕЗ_МиБ:
            # [ИСПРАВЛЕНО 10.09 -- МОЯ ЖЕ РЕГРЕССИЯ, НАЙДЕНА ТРАССОЙ СО СТЕКАМИ]
            # Стояло `max(num_scheduled_tokens.values()) > 1` с пометкой «только префилльный
            # шаг». При СПЕКУЛЯЦИИ декодный шаг планирует k+1=4 токена на запрос, поэтому
            # условие было истинно НА КАЖДОМ ШАГЕ. А в комментарии я написал, что проверка
            # «стоит наносекунды»: тоже неверно -- `torch.cuda.memory_reserved()` идёт через
            # `memory_stats()`, и трасса показала 322 вызова `_recurse_add_to_result` НА ШАГ.
            # То есть я сам внёс питон в горячий путь -- ровно то, что запрещает пункт 5 цели.
            # Лечение: отсекать по ОБЩЕМУ числу токенов шага (целое из планировщика, дешевле
            # некуда), и только потом трогать распределитель.
            _всего = getattr(scheduler_output, "total_num_scheduled_tokens", 0)
            if _всего > _ПОДРЕЗ_ТОКЕНОВ:
                _дер = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
                if _дер > _ПОДРЕЗ_МиБ * 2 ** 20:
                    torch.cuda.empty_cache()
                    self._подрез_счёт = getattr(self, "_подрез_счёт", 0) + 1
                    if self._подрез_счёт <= 3 or self._подрез_счёт % 64 == 0:
                        logger.info("[fa2_sm70 ПОДРЕЗКА] вернул %.0f МиБ держанных "
                                    "(всего подрезок %d)", _дер / 2 ** 20,
                                    self._подрез_счёт)
        if self.execute_model_state is not None:
            raise RuntimeError(
                "State error: sample_tokens() must be called "
                "after execute_model() returns None."
            )

        if self.vllm_config.model_config.enable_return_routed_experts:
            capturer = RoutedExpertsCapturer.get_instance()
            if capturer is not None:
                capturer.clear_buffer()  # noqa
            else:
                logger.error("RoutedExpertsCapturer not initialized.")

        if scheduler_output.preempted_req_ids and has_kv_transfer_group():
            get_kv_transfer_group().handle_preemptions(
                scheduler_output.preempted_req_ids
            )

        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        with (
            record_function_or_nullcontext("gpu_model_runner: preprocess"),
            self.synchronize_input_prep(),
        ):
            # Update persistent batch states.
            self._update_states(scheduler_output)

            if has_ec_transfer() and get_ec_transfer().is_producer:
                with self.maybe_get_ec_connector_output(
                    scheduler_output,
                    encoder_cache=self.encoder_cache,
                ) as ec_connector_output:
                    self._execute_mm_encoder(scheduler_output)
                    return make_empty_encoder_model_runner_output(scheduler_output)

            if not num_scheduled_tokens:
                if (
                    self.parallel_config.distributed_executor_backend
                    == "external_launcher"
                    and self.parallel_config.data_parallel_size > 1
                ):
                    # this is a corner case when both external launcher
                    # and DP are enabled, num_scheduled_tokens could be
                    # 0, and has_unfinished_requests in the outer loop
                    # returns True. before returning early here we call
                    # dummy run to ensure coordinate_batch_across_dp
                    # is called into to avoid out of sync issues.
                    self._dummy_run(1)
                if not has_kv_transfer_group():
                    # Return empty ModelRunnerOutput if no work to do.
                    return EMPTY_MODEL_RUNNER_OUTPUT
                return self.kv_connector_no_forward(scheduler_output, self.vllm_config)

            if self.cache_config.kv_sharing_fast_prefill:
                assert not self.num_prompt_logprobs, (
                    "--kv-sharing-fast-prefill produces incorrect "
                    "logprobs for prompt tokens, tokens, please disable "
                    "it when the requests need prompt logprobs"
                )

            num_reqs = self.input_batch.num_reqs
            req_ids = self.input_batch.req_ids
            tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
            num_scheduled_tokens_np = np.array(tokens, dtype=np.int32)
            max_num_scheduled_tokens = int(num_scheduled_tokens_np.max())
            num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens

            _тп = time.perf_counter()
            logits_indices, spec_decode_metadata = self._prepare_inputs(
                scheduler_output,
                num_scheduled_tokens_np,
            )
            if _ФАЗА_ШАГА_N:
                self._фаза_шага("подготовка входа", time.perf_counter() - _тп)
                self._t_после_подготовки = time.perf_counter()

            cascade_attn_prefix_lens = None
            # Disable cascade attention when using microbatching (DBO)
            if self.cascade_attn_enabled and not self.parallel_config.use_ubatching:
                # Pre-compute cascade attention prefix lengths
                cascade_attn_prefix_lens = self._compute_cascade_attn_prefix_lens(
                    num_scheduled_tokens_np,
                    self.input_batch.num_computed_tokens_cpu[:num_reqs],
                    scheduler_output.num_common_prefix_blocks,
                )

            (
                cudagraph_mode,
                batch_desc,
                should_ubatch,
                num_tokens_across_dp,
                cudagraph_stats,
            ) = self._determine_batch_execution_and_padding(
                num_tokens=num_tokens_unpadded,
                num_reqs=num_reqs,
                num_scheduled_tokens_np=num_scheduled_tokens_np,
                max_num_scheduled_tokens=max_num_scheduled_tokens,
                use_cascade_attn=cascade_attn_prefix_lens is not None,
                num_encoder_reqs=len(scheduler_output.scheduled_encoder_inputs),
            )

            logger.debug(
                "Running batch with cudagraph_mode: %s, batch_descriptor: %s, "
                "should_ubatch: %s, num_tokens_across_dp: %s",
                cudagraph_mode,
                batch_desc,
                should_ubatch,
                num_tokens_across_dp,
            )

            num_tokens_padded = batch_desc.num_tokens
            num_reqs_padded = (
                batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
            )
            ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
                should_ubatch,
                num_scheduled_tokens_np,
                num_tokens_padded,
                num_reqs_padded,
                self.parallel_config.num_ubatches,
            )

            logger.debug(
                "ubatch_slices: %s, ubatch_slices_padded: %s",
                ubatch_slices,
                ubatch_slices_padded,
            )

            # True if any attention backend handles KV cache update separately
            # from forward() (i.e., forward_includes_kv_cache_update=False). When true,
            # slot_mappings must use padded dimensions to match the key/value tensors.
            has_separate_kv_update = not all(
                all(
                    g.backend.forward_includes_kv_cache_update
                    for g in self.attn_groups[id]
                )
                for id, spec in enumerate(self.kv_cache_config.kv_cache_groups)
                if not isinstance(spec.kv_cache_spec, EncoderOnlyAttentionSpec)
            )
            pad_attn = cudagraph_mode == CUDAGraphMode.FULL

            if self.cache_config.mamba_cache_mode == "align":
                mamba_utils.preprocess_mamba(
                    scheduler_output,
                    self.kv_cache_config,
                    self.cache_config,
                    self.mamba_state_idx,
                    self.input_batch,
                    self.requests,
                    self.compilation_config.static_forward_context,
                    self.model.get_mamba_state_copy_func(),
                )

            use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
            ubatch_slices_attn = ubatch_slices_padded if pad_attn else ubatch_slices

            _тс = time.perf_counter()
            slot_mappings_by_group, slot_mappings = self._get_slot_mappings(
                num_tokens_padded=num_tokens_padded
                if pad_attn or has_separate_kv_update
                else num_tokens_unpadded,
                num_reqs_padded=(
                    num_reqs_padded if pad_attn or has_separate_kv_update else num_reqs
                ),
                num_tokens_unpadded=num_tokens_unpadded,
                ubatch_slices=ubatch_slices_padded,
            )
            self._дерево_слотыг = slot_mappings_by_group   # [ДЕРЕВО] для слот-копии KV

            if _ФАЗА_ШАГА_N:
                self._фаза_шага("  слоты по слоям", time.perf_counter() - _тс)
            _тм = time.perf_counter()
            attn_metadata, spec_decode_common_attn_metadata = (
                self._build_attention_metadata(
                    num_tokens=num_tokens_unpadded,
                    num_tokens_padded=num_tokens_padded if pad_attn else None,
                    num_reqs=num_reqs,
                    num_reqs_padded=num_reqs_padded if pad_attn else None,
                    max_query_len=max_num_scheduled_tokens,
                    ubatch_slices=ubatch_slices_attn,
                    logits_indices=logits_indices,
                    use_spec_decode=use_spec_decode,
                    num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
                    cascade_attn_prefix_lens=cascade_attn_prefix_lens,
                    slot_mappings=slot_mappings_by_group,
                )
            )
            if _ФАЗА_ШАГА_N:
                self._фаза_шага("  метаданные внимания", time.perf_counter() - _тм)

            (
                input_ids,
                inputs_embeds,
                positions,
                intermediate_tensors,
                model_kwargs,
                ec_connector_output,
            ) = self._preprocess(
                scheduler_output, num_tokens_padded, intermediate_tensors
            )

        # Set cudagraph mode to none if calc_kv_scales is true.
        # KV scales calculation involves dynamic operations that are incompatible
        # with CUDA graph capture.
        if self.calculate_kv_scales:
            cudagraph_mode = CUDAGraphMode.NONE
            # Mark KV scales as calculated after the first forward pass
            self.calculate_kv_scales = False

        # Encoder-decoder models can only compile the pure decode steps where no
        # encoder inputs are present. Use eager for the first pass.
        num_encoder_reqs = len(scheduler_output.scheduled_encoder_inputs)
        has_encoder_input = (
            self.model_config.is_encoder_decoder and num_encoder_reqs > 0
        )

        # Run the model.
        # Use persistent buffers for CUDA graphs.
        with (
            set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens_padded,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_mode,
                batch_descriptor=batch_desc,
                ubatch_slices=ubatch_slices_padded,
                slot_mapping=slot_mappings,
                skip_compiled=has_encoder_input,
            ),
            record_function_or_nullcontext("gpu_model_runner: forward"),
            self.maybe_get_kv_connector_output(scheduler_output) as kv_connector_output,
        ):
            _тф = time.perf_counter()
            if _ФАЗА_ШАГА_N and getattr(self, "_t_после_подготовки", None):
                # УЧАСТОК МЕЖДУ подготовкой входа и проходом: метаданные внимания, слоты по слоям,
                # контекст прохода. Именно сюда указала разность зондов (12 мс со спекуляцией).
                self._фаза_шага("метаданные+слоты до прохода",
                                _тф - self._t_после_подготовки)
            with self._фаза_карты("проход цели"):
                model_output = self._model_forward(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    **model_kwargs,
                )
            if _ФАЗА_ШАГА_N:
                if not _ФАЗА_БЕЗ_СИНХР:
                    torch.cuda.synchronize()
                self._фаза_шага("прямой проход цели", time.perf_counter() - _тф)

        with record_function_or_nullcontext("gpu_model_runner: postprocess"):
            if self.use_aux_hidden_state_outputs:
                # True when EAGLE 3 is used.
                hidden_states, aux_hidden_states = model_output
            else:
                # Common case.
                hidden_states = model_output
                aux_hidden_states = None

            if not self.broadcast_pp_output:
                # Common case.
                if not get_pp_group().is_last_rank:
                    # Return the intermediate tensors.
                    assert isinstance(hidden_states, IntermediateTensors)
                    hidden_states.kv_connector_output = kv_connector_output
                    self.kv_connector_output = kv_connector_output
                    return hidden_states

                if self.is_pooling_model:
                    # Return the pooling output.
                    return self._pool(
                        hidden_states,
                        num_scheduled_tokens,
                        num_scheduled_tokens_np,
                        kv_connector_output,
                    )

                _тл = time.perf_counter()
                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states)
                if _ФАЗА_ШАГА_N:
                    torch.cuda.synchronize()
                    self._фаза_шага("словарь (logits)", time.perf_counter() - _тл)
            else:
                # Rare case.
                assert not self.is_pooling_model

                sample_hidden_states = hidden_states[logits_indices]
                if not get_pp_group().is_last_rank:
                    all_gather_tensors = {
                        "residual": not is_residual_scattered_for_sp(
                            self.vllm_config, num_tokens_padded
                        )
                    }
                    get_pp_group().send_tensor_dict(
                        hidden_states.tensors,
                        all_gather_group=get_tp_group(),
                        all_gather_tensors=all_gather_tensors,
                    )
                    logits = None
                else:
                    logits = self.model.compute_logits(sample_hidden_states)

                model_output_broadcast_data: dict[str, Any] = {}
                if logits is not None:
                    model_output_broadcast_data["logits"] = logits.contiguous()

                broadcasted = get_pp_group().broadcast_tensor_dict(
                    model_output_broadcast_data, src=len(get_pp_group().ranks) - 1
                )
                assert broadcasted is not None
                logits = broadcasted["logits"]

        self.execute_model_state = ExecuteModelState(
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            ec_connector_output,
            cudagraph_stats,
            slot_mappings,
        )
        self.kv_connector_output = kv_connector_output
        if _ФАЗА_ШАГА_N and getattr(self, "_t_шаг", None):
            self._фаза_шага("execute_model целиком", time.perf_counter() - self._t_шаг)
        return None

    @torch.inference_mode
    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
        kv_connector_output = self.kv_connector_output
        self.kv_connector_output = None

        if self.execute_model_state is None:
            # receive sampled token ids from the last PP rank.
            if self.use_async_scheduling and get_pp_group().world_size > 1:
                self._pp_receive_prev_sampled_token_ids_to_input_batch()
            if not kv_connector_output:
                return None  # type: ignore[return-value]

            # In case of PP with kv transfer, we need to pass through the
            # kv_connector_output
            if kv_connector_output.is_empty():
                return EMPTY_MODEL_RUNNER_OUTPUT

            output = copy(EMPTY_MODEL_RUNNER_OUTPUT)
            output.kv_connector_output = kv_connector_output
            return output

        # Unpack ephemeral state.
        (
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            ec_connector_output,
            cudagraph_stats,
            slot_mappings,
        ) = self.execute_model_state
        # Clear ephemeral state.
        self.execute_model_state = None

        # Apply structured output bitmasks if present.
        if grammar_output is not None:
            apply_grammar_bitmask(
                scheduler_output, grammar_output, self.input_batch, logits
            )

        with record_function_or_nullcontext("gpu_model_runner: sample"):
            _тв = time.perf_counter()
            # [ОКНО НА КАРТУ ДЛЯ ВЫБОРКИ -- 10.09, записка 25 §279]
            # Фаза «выборка+отбраковка» (2.14 мс) считалась ПРОСТОЕМ карты, но внутри отбраковки
            # стоит `output_token_ids.cpu()` -- значит часть этого времени это ОЖИДАНИЕ ядер
            # выборки, то есть работа карты. Без этого окна работа по порядку шага делалась бы
            # по завышенной щели.
            with self._фаза_карты("выборка на карте"):
                sampler_output = self._sample(logits, spec_decode_metadata)
            if _ФАЗА_ШАГА_N:
                torch.cuda.synchronize()
                self._фаза_шага("выборка+отбраковка", time.perf_counter() - _тв)

        # [ПРОБА ПОКРЫТИЯ ЧЕРНОВИКА, ЕДИНИЦА СРАВНЕНИЯ СОВПАДАЕТ] Кандидаты позиции 0, которые
        # черновик предложил ДЛЯ ЭТОГО ЖЕ шага, против токена, который цель на этой позиции реально
        # выдала (`sampled_token_ids[:, 0]` -- либо принятый черновиковый, либо замена). Отвечает на
        # вопрос, от которого зависит, окупится ли ДЕРЕВО: черновик промахивается мимо цели или
        # только плохо её ранжирует.
        _пк = int(os.environ.get("FA2SM70_DFLASH2_COVER", "0"))
        if _пк and getattr(self, "drafter", None) is not None:
            _к0 = getattr(self.drafter, "канд0", None)
            if _к0 is not None and sampler_output.sampled_token_ids is not None:
                _ф = sampler_output.sampled_token_ids[: _к0.shape[0], 0:1]
                _сум = getattr(self, "_покр_сум", None)
                if _сум is None:
                    _сум = self._покр_сум = {}
                    self._покр_н = 0
                for _n in (1, 2, 4, 8, 16):
                    _сум[_n] = _сум.get(_n, 0) + int((_к0[:, :_n] == _ф).any(dim=1).sum())
                self._покр_н += int(_к0.shape[0])
                # УСЛОВИЕ ПЕЧАТИ БЫЛО ТОЧНЫМ КРАТНЫМ, а счётчик растёт на РАЗМЕР
                # БАТЧА -- кратное почти никогда не совпадало, и годная проба молчала,
                # пока в журнале печаталась СОСЕДНЯЯ, испорченная (та давала top-1=7 %
                # при истинной приёмке позиции 0 равной 56 %). Печатаем по порогу.
                if self._покр_н - getattr(self, "_покр_печ", 0) >= _пк:
                    self._покр_печ = self._покр_н
                    logger.info("[fa2_sm70 ПОКРЫТИЕ0, %d шагов] %s", self._покр_н,
                                "  ".join(f"top-{_n}={_сум.get(_n,0)/self._покр_н*100:.1f}%"
                                          for _n in (1, 2, 4, 8, 16)))
                self.drafter.канд0 = None
        if _ВЫБОРКА_ФАЙЛ and getattr(self, "drafter", None) is not None:
            _с0 = getattr(self.drafter, "скрыт0", None)
            _у0 = getattr(self.drafter, "унар0", None)
            _кд = getattr(self.drafter, "_канд_для_выборки", None)
            if (_с0 is not None and _у0 is not None and _кд is not None
                    and sampler_output.sampled_token_ids is not None):
                _б = getattr(self, "_выб_буфер", None)
                if _б is None:
                    _б = self._выб_буфер = []
                    self._выб_всего = 0
                _n = min(_с0.shape[0], sampler_output.sampled_token_ids.shape[0])
                if _n > 0 and self._выб_всего < _ВЫБОРКА_ПРЕДЕЛ:
                    _ц0 = getattr(self.drafter, "цель0", None)
                    _б.append((
                        _с0[:_n].to(torch.float16).cpu(),
                        _кд[:_n].to(torch.int32).cpu(),
                        _у0[:_n].to(torch.float16).cpu(),
                        sampler_output.sampled_token_ids[:_n, 0].to(torch.int32).cpu(),
                        None if _ц0 is None else _ц0[:_n].cpu(),
                    ))
                    self._выб_всего += _n
                    if len(_б) >= 200 or self._выб_всего >= _ВЫБОРКА_ПРЕДЕЛ:
                        import os as _o
                        # КУСКАМИ, А НЕ ПЕРЕЗАПИСЬЮ ЦЕЛИКОМ: перезапись растёт
                        # квадратично и на 40 тыс. строк съела бы сбор целиком.
                        _к = getattr(self, "_выб_кусок", 0)
                        self._выб_кусок = _к + 1
                        _путь = f"{_ВЫБОРКА_ФАЙЛ}.{_o.getpid()}.{_к:04d}.pt"
                        torch.save(_б, _путь)
                        _б.clear()
                        print(f"[ВЫБОРКА] сохранено {self._выб_всего} строк -> {_путь}",
                              file=__import__("sys").stderr, flush=True)
            self.drafter.скрыт0 = None
            self.drafter.унар0 = None
            self.drafter.цель0 = None

        if _ВОЗВРАТ_ПАМЯТИ and not getattr(self, "_мем_возвращено", False):
            self._мем_возвращено = True
            _до_св, _вс = torch.cuda.mem_get_info()
            torch.cuda.empty_cache()
            _по_св, _ = torch.cuda.mem_get_info()
            logger.info("[fa2_sm70 ПАМЯТЬ] возвращено драйверу %.2f ГиБ "
                        "(свободно %.2f -> %.2f из %.2f)",
                        (_по_св - _до_св) / 2**30, _до_св / 2**30, _по_св / 2**30, _вс / 2**30)
        if _ОПИСЬ_ПАМЯТИ and getattr(self, "_мем_печ", 0) < _ОПИСЬ_ПАМЯТИ:
            self._мем_печ = getattr(self, "_мем_печ", 0) + 1
            _св, _всего = torch.cuda.mem_get_info()
            logger.info(
                "[fa2_sm70 ПАМЯТЬ] занято торчем %.2f ГиБ, ДЕРЖИТ (reserved) %.2f, "
                "не отдано драйверу %.2f | свободно у драйвера %.2f из %.2f",
                torch.cuda.memory_allocated() / 2**30,
                torch.cuda.memory_reserved() / 2**30,
                (torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) / 2**30,
                _св / 2**30, _всего / 2**30)
        self._update_states_after_model_execute(
            sampler_output.sampled_token_ids, scheduler_output
        )
        if self.use_async_scheduling:
            pp = get_pp_group()
            # For torchrun external_launcher PP mode with broadcast_pp_output=True,
            # PP outputs have been broadcasted to all ranks at logits computation.
            # Therefore, here is no need to send sampled token ids again in this case.
            if not self.broadcast_pp_output and pp.world_size > 1 and pp.is_last_rank:
                self._pp_broadcast_prev_sampled_token_ids(
                    sampler_output.sampled_token_ids
                )

        self._draft_token_ids = None
        self._draft_token_req_ids = None
        self.input_batch.prev_sampled_token_ids = None

        # [ПРИБОР СЛОЁВ, 29.08] Снимок буферов, заполненных ВНУТРИ графа, читается ЗДЕСЬ --
        # после шага и вне графа. Иначе локализовать расхождение нечем: питон при повторе
        # графа не исполняется, хуки и печать из слоя молчат.
        if _ДБГ_СЛОИ_ВКЛ:
            try:
                from vllm.model_executor.models.qwen3_next import дбг_снимок
                _ш = getattr(self, "_дбг_шаг", 0) + 1
                self._дбг_шаг = _ш
                if _ш <= int(_os.environ.get("FA2SM70_DBG_STEPS", "3")):
                    import sys as _sdbg
                    for _i, _в in дбг_снимок().items():
                        print(f"[СЛОЙ шаг={_ш} слой={_i:02d}] "
                              + " ".join(f"{x:+.6f}" for x in _в[:4]),
                              file=_sdbg, flush=True)
            except Exception:
                pass

        def propose_draft_token_ids(sampled_token_ids):
            assert spec_decode_common_attn_metadata is not None
            with record_function_or_nullcontext("gpu_model_runner: draft"):
                _тч = time.perf_counter()
                self._draft_token_ids = self.propose_draft_token_ids(
                    scheduler_output,
                    sampled_token_ids,
                    self.input_batch.sampling_metadata,
                    hidden_states,
                    sample_hidden_states,
                    aux_hidden_states,
                    spec_decode_metadata,
                    spec_decode_common_attn_metadata,
                    slot_mappings,
                )
                if _ФАЗА_ШАГА_N:
                    torch.cuda.synchronize()
                    self._фаза_шага("черновик", time.perf_counter() - _тч)
                self._copy_draft_token_ids_to_cpu(scheduler_output)

        spec_config = self.speculative_config
        propose_drafts_after_bookkeeping = False
        if spec_config is not None:
            input_fits_in_drafter = spec_decode_common_attn_metadata is not None and (
                spec_decode_common_attn_metadata.max_seq_len + self.num_spec_tokens
                <= self.effective_drafter_max_model_len
            )
            use_gpu_toks = (
                spec_config.use_eagle() or spec_config.uses_draft_model()
            ) and not spec_config.disable_padded_drafter_batch
            if use_gpu_toks:
                # EAGLE/DraftModel speculative decoding can use the GPU sampled tokens
                # as inputs, and does not need to wait for bookkeeping to finish.
                assert isinstance(self.drafter, EagleProposer | DraftModelProposer | DFlash2Proposer)
                sampled_token_ids = sampler_output.sampled_token_ids
                if input_fits_in_drafter:
                    propose_draft_token_ids(sampled_token_ids)
                elif self.valid_sampled_token_count_event is not None:
                    assert spec_decode_common_attn_metadata is not None
                    next_token_ids, valid_sampled_tokens_count = (
                        self.drafter.prepare_next_token_ids_padded(
                            spec_decode_common_attn_metadata,
                            sampled_token_ids,
                            self.requests,
                            self.input_batch,
                            self.discard_request_mask.gpu,
                        )
                    )
                    self._copy_valid_sampled_token_count(
                        next_token_ids, valid_sampled_tokens_count
                    )
                    # Since we couldn't run the drafter,
                    # just use zeros for the draft tokens.
                    self._draft_token_ids = torch.zeros(
                        1, device=self.device, dtype=torch.int32
                    ).expand(len(self.input_batch.req_ids), self.num_spec_tokens)
                    self._copy_draft_token_ids_to_cpu(scheduler_output, zeros_only=True)
            else:
                propose_drafts_after_bookkeeping = input_fits_in_drafter

        with record_function_or_nullcontext("gpu_model_runner: bookkeep"):
            _тб = time.perf_counter()
            (
                num_nans_in_logits,
                logprobs_lists,
                valid_sampled_token_ids,
                prompt_logprobs_dict,
                req_ids_output_copy,
                req_id_to_index_output_copy,
                invalid_req_indices,
            ) = self._bookkeeping_sync(
                scheduler_output,
                sampler_output,
                logits,
                hidden_states,
                scheduler_output.total_num_scheduled_tokens,
                spec_decode_metadata,
            )
            if _ФАЗА_ШАГА_N:
                self._фаза_шага("учёт (bookkeeping)", time.perf_counter() - _тб)

        if propose_drafts_after_bookkeeping:
            # ngram and other speculative decoding methods use the sampled
            # tokens on the CPU, so they are run after bookkeeping.
            propose_draft_token_ids(valid_sampled_token_ids)

        with record_function_or_nullcontext("gpu_model_runner: eplb"):
            self.eplb_step()

        with record_function_or_nullcontext("gpu_model_runner: ModelRunnerOutput"):
            if self.model_config.enable_return_routed_experts:
                capturer = RoutedExpertsCapturer.get_instance()
                if capturer is not None:
                    capturer.save_captured_experts(indices=self.slot_mapping)  # noqa
                else:
                    logger.error("RoutedExpertsCapturer not initialized.")

            output = ModelRunnerOutput(
                req_ids=req_ids_output_copy,
                req_id_to_index=req_id_to_index_output_copy,
                sampled_token_ids=valid_sampled_token_ids,
                logprobs=logprobs_lists,
                prompt_logprobs_dict=prompt_logprobs_dict,
                kv_connector_output=kv_connector_output,
                ec_connector_output=ec_connector_output
                if self.supports_mm_inputs
                else None,
                num_nans_in_logits=num_nans_in_logits,
                cudagraph_stats=cudagraph_stats,
            )

        if not self.use_async_scheduling:
            if _ФАЗА_ШАГА_N and getattr(self, "_t_шаг", None):
                self._t_конец_шага = time.perf_counter()
                self._фаза_шага("воркер целиком", self._t_конец_шага - self._t_шаг)
            return output

        with record_function_or_nullcontext(
            "gpu_model_runner: AsyncGPUModelRunnerOutput"
        ):
            async_output = AsyncGPUModelRunnerOutput(
                model_runner_output=output,
                sampled_token_ids=sampler_output.sampled_token_ids,
                logprobs_tensors=sampler_output.logprobs_tensors,
                invalid_req_indices=invalid_req_indices,
                async_output_copy_stream=self.async_output_copy_stream,
                vocab_size=self.input_batch.vocab_size,
                гран_ctx=(
                    self.input_batch.num_computed_tokens_cpu[
                        : self.input_batch.num_reqs
                    ].tolist()
                    if (_ГРАН_БОНУС and self.num_spec_tokens)
                    else None
                ),
                гран_B=int(getattr(self.cache_config, "block_size", 0) or 0),
            )
        with record_function_or_nullcontext(
            "gpu_model_runner: set_async_sampled_token_ids"
        ):
            # Save ref of sampled_token_ids CPU tensor if the batch contains
            # any requests with sampling params that require output ids.
            self.input_batch.set_async_sampled_token_ids(
                async_output.sampled_token_ids_cpu,
                async_output.async_copy_ready_event,
            )

        if _ФАЗА_ШАГА_N and getattr(self, "_t_шаг", None):
            self._t_конец_шага = time.perf_counter()
            self._фаза_шага("воркер целиком", self._t_конец_шага - self._t_шаг)
        return async_output

    def _pp_broadcast_prev_sampled_token_ids(
        self, sampled_token_ids: torch.Tensor
    ) -> None:
        """Broadcast sampled token ids (GPU) from last PP stage"""
        pp = get_pp_group()
        assert pp.is_last_rank
        # `prev_sampled_token_ids` is expected to have shape [num_reqs, 1].
        assert sampled_token_ids.dim() == 2 and sampled_token_ids.shape[-1] == 1, (
            "PP+async expects sampled_token_ids to have shape [num_reqs, 1]"
        )
        torch.distributed.broadcast(
            sampled_token_ids, src=pp.rank, group=pp.device_group
        )

    def _pp_receive_prev_sampled_token_ids_to_input_batch(self) -> None:
        """Receive sampled token ids broadcast from last PP stage"""
        pp = get_pp_group()
        assert not pp.is_last_rank
        num_reqs = self.input_batch.num_reqs
        # `prev_sampled_token_ids` is expected to have shape [num_reqs, 1].
        recv = torch.empty((num_reqs, 1), dtype=torch.int32, device=self.device)
        torch.distributed.broadcast(recv, src=pp.last_rank, group=pp.device_group)
        self.input_batch.prev_sampled_token_ids = recv

        # construct `prev_req_id_to_index` here so `_prepare_input_ids`
        # can map req_id -> previous batch row
        discard_req_indices = np.nonzero(self.discard_request_mask.np[:num_reqs])[0]
        discard_req_indices_set = set(discard_req_indices)
        prev_req_id_to_index: dict[str, int] = {}
        for i, req_id in enumerate(self.input_batch.req_ids):
            if i in discard_req_indices_set:
                continue
            prev_req_id_to_index[req_id] = i
            # PP+async scheduling: advance per-request local cached output length by
            # appending a placeholder (-1) token id.
            if (req_state := self.requests.get(req_id)) is not None:
                req_state.output_token_ids.append(-1)
        self.input_batch.prev_req_id_to_index = prev_req_id_to_index

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        if not self.num_spec_tokens or not self._draft_token_req_ids:
            return None
        draft_token_ids, req_ids = self._get_draft_token_ids_cpu()
        return DraftTokenIds(req_ids, draft_token_ids)

    def _copy_draft_token_ids_to_cpu(
        self, scheduler_output: "SchedulerOutput", zeros_only: bool = False
    ) -> None:
        # Check if we need to copy draft tokens to CPU. In async scheduling,
        # we only copy when needed for structured output, penalties or bad_words.
        # [ГРАНИЦА БЛОКА СОСТОЯНИЯ -- ИСКЛЮЧЕНИЕ ИЗ РАННЕГО ВЫХОДА, 01.09]
        # При async черновик НЕ сходит на процессор, и потому пропуск спекуляции у границы
        # (`_гран_пропуск`) был мёртв: функция ниже не звалась НИ РАЗУ (поймано счётчиком).
        # Это и есть причина, по которой обрыв на границе не ушёл на сборке 31.08
        # (reports/vllm-granica-ne-ushla-2026-08-31.md: 5 событий на 23 генерациях, две точные
        # подписи 4095 и 4097). Копируем ВОПРЕКИ раннему выходу, но только когда хотя бы один
        # запрос стоит вплотную к границе -- это 5 шагов из 4096, то есть 0.12 % шагов.
        if self.use_async_scheduling and not (
            scheduler_output.has_structured_output_requests
            or self.input_batch.sampling_metadata.output_token_ids
        ):
            if not (_ГРАН_ПРОПУСК and self._гран_рядом()):
                return
        # We must also set the corresponding request ids.
        self._draft_token_req_ids = self.input_batch.req_ids.copy()

        draft_token_ids: torch.Tensor = self._draft_token_ids
        if not torch.is_tensor(draft_token_ids):
            return
        assert self.draft_token_ids_event is not None
        assert self.draft_token_ids_copy_stream is not None
        assert self.draft_token_ids_cpu is not None
        default_stream = torch.cuda.current_stream()
        num_reqs = draft_token_ids.shape[0]
        with torch.cuda.stream(self.draft_token_ids_copy_stream):
            if not zeros_only:
                # Trigger async copy of draft token ids to cpu.
                self.draft_token_ids_copy_stream.wait_stream(default_stream)
                self.draft_token_ids_cpu[:num_reqs].copy_(
                    draft_token_ids, non_blocking=True
                )
            else:
                # No copy needed, just zero-out cpu tensor.
                self.draft_token_ids_cpu[:num_reqs] = 0
            self.draft_token_ids_event.record()

    def _гран_слот_проверить(self, req_indices, positions_np, num_reqs) -> None:
        """Инвариант границы: блок под каждую позицию шага обязан быть уже в таблице.

        `compute_slot_mapping` адресует block_table[req, pos // block_size] БЕЗ проверки
        ширины: если планировщик не выделил блок под последнюю позицию шага, читается
        нулевая ячейка, и KV этой позиции уезжает в блок 0 -- чужой контекст. Прибор ищет
        ровно это, по всем KV-группам (у гибрида их десять, размеры блока разные).
        """
        import sys as _s4
        for гр, таб in enumerate(self.input_batch.block_table.block_tables):
            B = int(таб.block_size)
            шир = таб.num_blocks_per_row
            for i in range(num_reqs):
                маска = req_indices == i
                if not маска.any():
                    continue
                макс = int(positions_np[маска].max())
                надо = макс // B + 1
                есть = int(шир[i])
                if надо > есть:
                    print(f"[ГРАН-СЛОТ] группа={гр} строка={i} макс_поз={макс} B={B} "
                          f"надо блоков={надо} есть={есть} -> KV уедет в блок 0",
                          file=_s4.stderr, flush=True)
                elif макс % B < 4 or (B - 1 - макс % B) < 4:
                    print(f"[ГРАН-СЛОТ] у границы: группа={гр} строка={i} макс_поз={макс} "
                          f"B={B} блоков={есть} q={int(маска.sum())}",
                          file=_s4.stderr, flush=True)

    def _get_draft_token_ids_cpu(self) -> tuple[list[list[int]], list[str]]:
        if isinstance(self._draft_token_ids, list):
            return self._гран_пропуск(self._draft_token_ids, self.input_batch.req_ids), self.input_batch.req_ids
        req_ids = self._draft_token_req_ids
        if req_ids is None:
            return [], []
        assert self.draft_token_ids_event is not None
        assert self.draft_token_ids_cpu is not None
        self.draft_token_ids_event.synchronize()
        _спис = self.draft_token_ids_cpu[: len(req_ids)].tolist()
        return self._гран_пропуск(_спис, req_ids), req_ids

    def _гран_рядом(self) -> bool:
        """Стоит ли хоть один запрос вплотную к границе блока состояния.

        Считается ТОЛЬКО по процессорным копиям (`num_computed_tokens_cpu`) -- ни одного
        обращения к карте, поэтому проверка не рвёт асинхронность: она стоит несколько
        десятков наносекунд и истинна на 5 шагах из 4096.
        """
        if not self.num_spec_tokens:
            return False
        try:
            B = self._гран_блок
        except AttributeError:
            B = getattr(self.cache_config, "mamba_block_size", None) or 0
            if not B:
                try:
                    for _g in (self.kv_cache_config.kv_cache_groups or ()):
                        if type(_g.kv_cache_spec).__name__ == "MambaSpec":
                            B = int(_g.kv_cache_spec.block_size); break
                except Exception:
                    B = 0
                if not B:
                    B = int(getattr(self.cache_config, "block_size", 0) or 0)
            self._гран_блок = B
        if not B:
            return False
        n = self.input_batch.num_reqs
        нкт = self.input_batch.num_computed_tokens_cpu
        k = self.num_spec_tokens
        for i in range(n):
            ctx = int(нкт[i])
            # ОКНО ОСТАВЛЕНО УЗКИМ. Расширение (+-k) и второй источник длины проверены
            # статистически 03.09: обрывов 9 из 20 и с ними, и БЕЗ механизма вовсе -- то есть
            # снятие спекуляции у границы этот дефект не лечит, а лишняя ширина только чаще
            # выключает спекуляцию. Гипотеза §58-59 для обрыва 8192 ОПРОВЕРГНУТА замером.
            if ctx > 0 and (ctx - 1) // B != (ctx + k + 1) // B:
                return True
        return False



    def _гран_пропуск(self, спис, req_ids):
        """ПУТЬ 3 ЛЕЧЕНИЯ ГРАНИЦЫ БЛОКА СОСТОЯНИЯ (записка 25 §58-59).

        На шаге, где k+1 черновых токенов переходят границу блока состояния GDN, align-таблица
        спекуляции начинается уже с НОВОГО блока, а состояние лежит в старом -- блока со
        состоянием в её колонках нет вовсе. Замерено прибором:
            seq_len=4097..4100 ctx=4093..4096 start_po_seq=1 start_po_ctx=0
        Прямое лечение (дать спекуляции опору по посчитанным) ОПРОВЕРГНУТО: генерация
        вырождается в нули. Поэтому такой шаг просто идёт БЕЗ спекуляции: пустой черновик
        делает строку не-спекулятивной, а не-спек путь границу проходит верно (read по
        посчитанным, write по запланированным).

        Цена -- один шаг из block_size, то есть 0.02 %.
        """
        if _ГРАН_ЛОГ:
            self._гран_зовов = getattr(self, "_гран_зовов", 0) + 1
            if self._гран_зовов % 500 == 1:
                import sys as _s6
                print(f"[УРЕЗАНИЕ] вызовов пропуска: {self._гран_зовов} "
                      f"(строк {len(спис) if спис else 0})", file=_s6.stderr, flush=True)
        if not _ГРАН_ПРОПУСК or not спис:
            return спис
        try:
            B = self._гран_блок
        except AttributeError:
            B = getattr(self.cache_config, "mamba_block_size", None) or 0
            if not B:
                # mamba_block_size может быть не заполнен; берём размер блока у СПЕЦИФИКАЦИИ
                # групп состояния, а последним доводом -- блок внимания (движок подгоняет его
                # под страницу mamba: см. «Setting attention block size to 4096 tokens»).
                try:
                    for _g in (self.kv_cache_config.kv_cache_groups or ()):
                        _sp = _g.kv_cache_spec
                        if type(_sp).__name__ == "MambaSpec":
                            B = int(_sp.block_size); break
                except Exception:
                    B = 0
                if not B:
                    B = int(getattr(self.cache_config, "block_size", 0) or 0)
            self._гран_блок = B
        if not B:
            return спис
        нкт = self.input_batch.num_computed_tokens_cpu
        for i, стр in enumerate(спис):
            if not стр:
                continue
            ctx = int(нкт[i])
            # [УЛЬТРАКОРОТКИЙ КОНТЕКСТ -- ВТОРАЯ НАХОДКА ОТЧЁТА 31.08]
            # Отчёт (раздел 5) фиксирует РОСТ невоспроизводимости однотокенных запросов:
            # 'раз' давал 3 из 40 различных ответов 29.08 и 14 из 40 31.08. Замер 01.09 назвал
            # виновника снятием фазы: БЕЗ спекуляции 'раз' даёт 1 вариант из 40 и НОЛЬ пустых,
            # с ней -- 3 варианта и 8 ПУСТЫХ ответов. То есть на вырожденном контексте черновик
            # предлагает конец последовательности, и приёмка его берёт.
            # Лечение то же, что у границы: на таком шаге спекуляции просто нет.
            if 0 < ctx < _МИН_КОНТЕКСТ:
                спис[i] = []
                self._гран_коротко = getattr(self, "_гран_коротко", 0) + 1
                continue
            if ctx <= 0:
                # Начало последовательности: (0-1)//B = -1, и наивное сравнение давало ложное
                # срабатывание на первом же шаге. Поймано печатью позиций: ctx=0 в списке снятий.
                continue
            # Окно с запасом: на момент вызова число посчитанных ещё не учло принятые этого
            # шага, поэтому граница проверяется не точкой, а отрезком ctx-1 .. ctx+k+1.
            # Замер (промпт 3889, k=3, B=4096) дал снятия ровно на ctx=4092..4096 -- пять шагов
            # у границы и ни одного лишнего.
            # [04.09 ЗАПАС НА АСИНХРОННОСТЬ] Черновик готовится для СЛЕДУЮЩЕГО шага, а `ctx`
            # здесь -- от текущего: при асинхронном планировании они расходятся ровно на число
            # принятых (до k+1). Прибор [СДВИГ ОПОРЫ] ловил шаги ctx=8190 q=4, то есть урезание
            # считало место по контексту, который к исполнению уже ушёл за границу. Поэтому
            # зона снятия берётся с запасом на этот разбег: 2(k+1) позиций, то есть ~8 шагов
            # из 4096 (0.2 %), и внутри неё черновик снимается ЦЕЛИКОМ, а не урезается.
            _запас = 2 * (len(стр) + 1)
            if _ГРАН_ВСЕГДА or (ctx - 1) // B != (ctx + _запас) // B:
                спис[i] = []
                self._гран_снято = getattr(self, "_гран_снято", 0) + 1
                if _ГРАН_ЛОГ:
                    import sys as _s7
                    print(f"[СНЯТИЕ] ctx={ctx} было={len(стр)} запас={_запас} B={B} "
                          f"всего={self._гран_снято}", file=_s7.stderr, flush=True)
                continue
            if (ctx - 1) // B != (ctx + len(стр) + 1) // B:
                # [УРЕЗАНИЕ ВМЕСТО СНЯТИЯ -- 03.09]
                # Полное снятие черновика у границы дефект НЕ лечит: 9 обрывов из 20 и с ним,
                # и без него (записка 25 §121). Зато замер по глубине спекуляции показал прямую
                # зависимость: k=1 -> 5 % обрывов, k=3 -> 45 %. Значит вредит не спекуляция как
                # таковая, а ЧИСЛО позиций, пересекающих границу. Оставляем ровно столько
                # черновых токенов, сколько влезает ДО границы: строка остаётся спекулятивной
                # (однородность батча не рвётся), но границу за шаг не переходит.
                # [04.09 ФОРМУЛА СЧИТАЛА ОТ ctx, А ОПОРА СТОИТ НА ctx-1]
                # Опора spec-таблицы -- блок ПОСЛЕДНЕЙ позиции шага, а состояние прошлого
                # шага лежит в блоке позиции ctx-1. Значит все позиции шага обязаны лежать в
                # блоке (ctx-1)//B. Прежняя формула считала место до конца блока, где лежит
                # САМ ctx, и при ctx, кратном блоку, давала 4095 -- то есть не урезала ничего
                # ровно на том шаге, где опора и уезжает (прибор: ctx=8192 было=3 стало=4095,
                # следом [СДВИГ ОПОРЫ] start_тек=2 start_зап=1).
                _влезает = max(0, ((ctx - 1) // B + 1) * B - ctx - 1)
                спис[i] = стр[:_влезает] if _влезает else []
                self._гран_снято = getattr(self, "_гран_снято", 0) + 1
                if _ГРАН_ЛОГ:
                    import sys as _s5
                    print(f"[УРЕЗАНИЕ] ctx={ctx} было={len(стр)} стало={_влезает} "
                          f"B={B} всего={self._гран_снято}", file=_s5.stderr, flush=True)
                if _ГРАН_ЛОГ:
                    # Печать РЕДКАЯ по построению: событие бывает 5 раз на block_size шагов
                    # (0.12 %), поэтому журнал она не засоряет, а доказательство даёт.
                    import sys as _s
                    print(f"[ГРАНИЦА] спекуляция снята: ctx={ctx} k={len(стр)} B={B} "
                          f"всего={self._гран_снято}", file=_s.stderr, flush=True)
        return спис

    def _copy_valid_sampled_token_count(
        self, next_token_ids: torch.Tensor, valid_sampled_tokens_count: torch.Tensor
    ) -> None:
        if self.valid_sampled_token_count_event is None:
            return

        default_stream = torch.cuda.current_stream()
        # Initialize a new stream to overlap the copy operation with
        # prepare_input of draft model.
        with torch.cuda.stream(self.valid_sampled_token_count_copy_stream):
            self.valid_sampled_token_count_copy_stream.wait_stream(default_stream)  # type: ignore
            counts = valid_sampled_tokens_count
            counts_cpu = self.valid_sampled_token_count_cpu
            assert counts_cpu is not None
            counts_cpu[: counts.shape[0]].copy_(counts, non_blocking=True)
            self.valid_sampled_token_count_event.record()

        self.input_batch.prev_sampled_token_ids = next_token_ids.unsqueeze(1)

    def _get_valid_sampled_token_count(self) -> list[int]:
        # Wait until valid_sampled_tokens_count is copied to cpu,
        prev_sampled_token_ids = self.input_batch.prev_sampled_token_ids
        sampled_count_event = self.valid_sampled_token_count_event
        if sampled_count_event is None or prev_sampled_token_ids is None:
            return []

        counts_cpu = self.valid_sampled_token_count_cpu
        assert counts_cpu is not None
        sampled_count_event.synchronize()
        return counts_cpu[: prev_sampled_token_ids.shape[0]].tolist()

    def propose_draft_token_ids(
        self,
        scheduler_output: "SchedulerOutput",
        sampled_token_ids: torch.Tensor | list[list[int]],
        sampling_metadata: SamplingMetadata,
        hidden_states: torch.Tensor,
        sample_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        spec_decode_metadata: SpecDecodeMetadata | None,
        common_attn_metadata: CommonAttentionMetadata,
        slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None,
    ) -> list[list[int]] | torch.Tensor:
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        spec_config = self.speculative_config
        assert spec_config is not None
        # [ЖИВОЙ РЯД ДЛЯ n-ГРАММ 31.08] Отдаём предлагателю ПРИНЯТЫЕ токены шага. Это
        # единственный источник, который их содержит: `token_ids_cpu` движок пишет только при
        # добавлении запроса, а `req_output_token_ids` держит заполнители -1, пока реальные
        # значения не потребуются штрафам. Без этого поиск n-грамм шёл по ОДНОМУ ПРОМПТУ и
        # не срабатывал ни разу за прогон (замер: `нет совпадения=149 из 149`, длина
        # совпадения ровно 1 на каждом шаге).
        self._fa2_принятые = sampled_token_ids
        if _ДЕРЕВО_ТАУ and sampled_token_ids is not None:
            # [ЗАМЕР TAU ЧЕРЕДОВАНИЕМ ВНУТРИ ОДНОГО ПРОЦЕССА]
            # Скорость на этой машине мерить нельзя: соседний боевой даёт монотонный дрейф
            # (одна и та же база дала 26.53 -> 38.04 мс/ток подряд, записка 25 §65). Но tau --
            # это КАЧЕСТВО черновика, и от фона она не зависит. Режим переключается ФАЙЛОМ,
            # без перезапуска, поэтому оба плеча меряются в одном временном окне.
            # Принятые приходят СПИСКОМ строк (n-граммы) либо ТЕНЗОРОМ [B, k+1] с
            # заполнителем -1 у отвергнутых. Первая редакция знала только про список, и
            # замер молчал весь прогон -- признак тот же, что и всегда: НИ ОДНОЙ строки.
            if isinstance(sampled_token_ids, list):
                if not sampled_token_ids:
                    return
                _пр = sum(len(r) for r in sampled_token_ids) / len(sampled_token_ids)
            else:
                _т = sampled_token_ids
                _пр = float((_т >= 0).sum().item()) / max(_т.shape[0], 1)
            _реж = getattr(self, "_дер_реж", 0)
            self._тау_сум = getattr(self, "_тау_сум", {0: 0.0, 1: 0.0})
            self._тау_н = getattr(self, "_тау_н", {0: 0, 1: 0})
            self._тау_сум[_реж] += _пр
            self._тау_н[_реж] += 1
            if (self._тау_н[0] + self._тау_н[1]) % _ДЕРЕВО_ТАУ == 0:
                import sys as _s
                _a = self._тау_сум[0] / max(self._тау_н[0], 1)
                _b = self._тау_сум[1] / max(self._тау_н[1], 1)
                print(f"[TAU] tolko_A={_a:.3f} ({self._тау_н[0]} shagov)  "
                      f"derevo={_b:.3f} ({self._тау_н[1]} shagov)  "
                      f"priz={(_b / _a - 1) * 100 if _a > 0 else 0:+.1f}%",
                      file=_s.stderr, flush=True)
        if spec_config.method == "ngram":
            from vllm.v1.spec_decode.ngram_proposer import NgramProposer

            assert isinstance(sampled_token_ids, list)
            assert isinstance(self.drafter, NgramProposer)
            draft_token_ids = self.drafter.propose(
                sampled_token_ids,
                self.input_batch.num_tokens_no_spec,
                self.input_batch.token_ids_cpu,
                slot_mappings=slot_mappings,
            )
        elif spec_config.method == "suffix":
            assert isinstance(sampled_token_ids, list)
            assert isinstance(self.drafter, SuffixDecodingProposer)
            draft_token_ids = self.drafter.propose(
                self.input_batch, sampled_token_ids, slot_mappings=slot_mappings
            )
        elif spec_config.method == "medusa":
            assert isinstance(sampled_token_ids, list)
            assert isinstance(self.drafter, MedusaProposer)

            if sample_hidden_states.shape[0] == len(sampled_token_ids):
                # The input to the target model does not include draft tokens.
                hidden_states = sample_hidden_states
            else:
                indices = []
                offset = 0
                assert spec_decode_metadata is not None, (
                    "No spec decode metadata for medusa"
                )
                for num_draft, tokens in zip(
                    spec_decode_metadata.num_draft_tokens, sampled_token_ids
                ):
                    indices.append(offset + len(tokens) - 1)
                    offset += num_draft + 1
                indices = torch.tensor(indices, device=self.device)
                hidden_states = sample_hidden_states[indices]

            draft_token_ids = self.drafter.propose(
                target_hidden_states=hidden_states,
                sampling_metadata=sampling_metadata,
                slot_mappings=slot_mappings,
            )
        elif spec_config.use_eagle() or spec_config.uses_draft_model():
            assert isinstance(self.drafter, EagleProposer | DraftModelProposer | DFlash2Proposer)

            if spec_config.disable_padded_drafter_batch:
                # When padded-batch is disabled, the sampled_token_ids should be
                # the cpu-side list[list[int]] of valid sampled tokens for each
                # request, with invalid requests having empty lists.
                assert isinstance(sampled_token_ids, list), (
                    "sampled_token_ids should be a python list when"
                    "padded-batch is disabled."
                )
                next_token_ids = self.drafter.prepare_next_token_ids_cpu(
                    sampled_token_ids,
                    self.requests,
                    self.input_batch,
                    scheduler_output.num_scheduled_tokens,
                )
            else:
                # When using padded-batch, the sampled_token_ids should be
                # the gpu tensor of sampled tokens for each request, of shape
                # (num_reqs, num_spec_tokens + 1) with rejected tokens having
                # value -1.
                assert isinstance(sampled_token_ids, torch.Tensor), (
                    "sampled_token_ids should be a torch.Tensor when"
                    "padded-batch is enabled."
                )
                next_token_ids, valid_sampled_tokens_count = (
                    self.drafter.prepare_next_token_ids_padded(
                        common_attn_metadata,
                        sampled_token_ids,
                        self.requests,
                        self.input_batch,
                        self.discard_request_mask.gpu,
                    )
                )
                self._copy_valid_sampled_token_count(
                    next_token_ids, valid_sampled_tokens_count
                )

            num_rejected_tokens_gpu = None
            if spec_decode_metadata is None:
                token_indices_to_sample = None
                # input_ids can be None for multimodal models.
                target_token_ids = self.input_ids.gpu[:num_scheduled_tokens]
                target_positions = self._get_positions(num_scheduled_tokens)
                if self.use_aux_hidden_state_outputs:
                    assert aux_hidden_states is not None
                    target_hidden_states = torch.cat(
                        [h[:num_scheduled_tokens] for h in aux_hidden_states], dim=-1
                    )
                else:
                    target_hidden_states = hidden_states[:num_scheduled_tokens]
                if _ДАМП_MTP and num_scheduled_tokens >= _ДАМП_MTP_МИН:
                    self._снять_для_головы(target_hidden_states, target_token_ids)
            else:
                if spec_config.disable_padded_drafter_batch:
                    token_indices_to_sample = None
                    common_attn_metadata, token_indices = self.drafter.prepare_inputs(
                        common_attn_metadata,
                        sampled_token_ids,
                        spec_decode_metadata.num_draft_tokens,
                    )
                    target_token_ids = self.input_ids.gpu[token_indices]
                    target_positions = self._get_positions(token_indices)
                    if self.use_aux_hidden_state_outputs:
                        assert aux_hidden_states is not None
                        target_hidden_states = torch.cat(
                            [h[token_indices] for h in aux_hidden_states], dim=-1
                        )
                    else:
                        target_hidden_states = hidden_states[token_indices]
                else:
                    (
                        common_attn_metadata,
                        token_indices_to_sample,
                        num_rejected_tokens_gpu,
                    ) = self.drafter.prepare_inputs_padded(
                        common_attn_metadata,
                        spec_decode_metadata,
                        valid_sampled_tokens_count,
                    )
                    total_num_tokens = common_attn_metadata.num_actual_tokens
                    # When padding the batch, token_indices is just a range
                    target_token_ids = self.input_ids.gpu[:total_num_tokens]
                    target_positions = self._get_positions(total_num_tokens)
                    if self.use_aux_hidden_state_outputs:
                        assert aux_hidden_states is not None
                        target_hidden_states = torch.cat(
                            [h[:total_num_tokens] for h in aux_hidden_states], dim=-1
                        )
                    else:
                        target_hidden_states = hidden_states[:total_num_tokens]

            if self.supports_mm_inputs and self.drafter.supports_mm_inputs:
                mm_embed_inputs = self._gather_mm_embeddings(
                    scheduler_output,
                    shift_computed_tokens=1,
                )
            else:
                mm_embed_inputs = None

            # [ПРОХОД ЧЕРНОВИКА: КАРТА ОТДЕЛЬНО ОТ ХОЗЯИНА -- 09.09, записка 25 §249]
            # §236 намерил «проход черновика 10.52 мс» ХОЗЯЙСКИМ таймером, то есть время
            # ВЫДАЧИ ядер. Пол чтения весов у трёх проходов головы ~5.4 мс, и вопрос, на
            # который нельзя ответить хозяйским таймером: остальные ~5 мс -- это работа
            # карты или ЩЕЛИ между запусками (черновик у vLLM принципиально PIECEWISE,
            # eagle.py:289). Ответ решает, стоит ли полный граф черновика.
            _тчер = time.perf_counter() if _ФАЗА_ШАГА_N else 0.0
            with self._фаза_карты("проход черновика"):
                draft_token_ids = self.drafter.propose(
                    target_token_ids=target_token_ids,
                    target_positions=target_positions,
                    target_hidden_states=target_hidden_states,
                    next_token_ids=next_token_ids,
                    last_token_indices=token_indices_to_sample,
                    sampling_metadata=sampling_metadata,
                    common_attn_metadata=common_attn_metadata,
                    mm_embed_inputs=mm_embed_inputs,
                    num_rejected_tokens_gpu=num_rejected_tokens_gpu,
                    slot_mappings=slot_mappings,
                )
            if _ФАЗА_ШАГА_N:
                self._фаза_шага("проход черновика (хозяин)",
                                time.perf_counter() - _тчер)

        return draft_token_ids

    def update_config(self, overrides: dict[str, Any]) -> None:
        allowed_config_names = {"load_config", "model_config"}
        for config_name, config_overrides in overrides.items():
            assert config_name in allowed_config_names, (
                f"Config `{config_name}` not supported. "
                f"Allowed configs: {allowed_config_names}"
            )
            config = getattr(self, config_name)
            new_config = update_config(config, config_overrides)
            setattr(self, config_name, new_config)

    def load_model(self, eep_scale_up: bool = False) -> None:
        """
        Args:
            eep_scale_up: the model loading is for elastic EP scale up.
        """
        logger.info_once(
            "Starting to load model %s...",
            self.model_config.model,
            scope="global",
        )
        global_expert_loads, old_global_expert_indices_per_model, rank_mapping = (
            EplbState.get_eep_state(self.parallel_config)
            if eep_scale_up
            else (None, None, None)
        )

        if self.parallel_config.enable_eplb:
            self.eplb_state = EplbState(self.parallel_config, self.device)
            eplb_models = 0

        try:
            with DeviceMemoryProfiler() as m:
                time_before_load = time.perf_counter()
                model_loader = get_model_loader(self.load_config)
                self.model = model_loader.load_model(
                    vllm_config=self.vllm_config, model_config=self.model_config
                )
                if self.lora_config:
                    self.model = self.load_lora_model(
                        self.model, self.vllm_config, self.device
                    )
                if hasattr(self, "drafter"):
                    logger.info_once("Loading drafter model...")
                    self.drafter.load_model(self.model)
                    if (
                        hasattr(self.drafter, "model")
                        and is_mixture_of_experts(self.drafter.model)
                        and self.parallel_config.enable_eplb
                    ):
                        spec_config = self.vllm_config.speculative_config
                        assert spec_config is not None
                        assert spec_config.draft_model_config is not None
                        logger.info_once(
                            "EPLB is enabled for drafter model %s.",
                            spec_config.draft_model_config.model,
                        )

                        global_expert_load = (
                            global_expert_loads[eplb_models]
                            if global_expert_loads
                            else None
                        )
                        old_global_expert_indices = (
                            old_global_expert_indices_per_model[eplb_models]
                            if old_global_expert_indices_per_model
                            else None
                        )
                        if self.eplb_state is None:
                            self.eplb_state = EplbState(
                                self.parallel_config, self.device
                            )
                        self.eplb_state.add_model(
                            self.drafter.model,
                            spec_config.draft_model_config,
                            global_expert_load,
                            old_global_expert_indices,
                            rank_mapping,
                        )
                        eplb_models += 1

                if self.use_aux_hidden_state_outputs:
                    if not supports_eagle3(self.get_model()):
                        raise RuntimeError(
                            "Model does not support EAGLE3 interface but "
                            "aux_hidden_state_outputs was requested"
                        )

                    # Try to get auxiliary layers from speculative config,
                    # otherwise use model's default layers
                    aux_layers = self._get_eagle3_aux_layers_from_config()
                    if aux_layers:
                        logger.info(
                            "Using auxiliary layers from speculative config: %s",
                            aux_layers,
                        )
                    else:
                        aux_layers = self.model.get_eagle3_aux_hidden_state_layers()

                    self.model.set_aux_hidden_state_layers(aux_layers)
                time_after_load = time.perf_counter()
            self.model_memory_usage = m.consumed_memory
        except torch.cuda.OutOfMemoryError as e:
            msg = (
                "Failed to load model - not enough GPU memory. "
                "Try lowering --gpu-memory-utilization to free memory for weights, "
                "increasing --tensor-parallel-size, or using --quantization. "
                "See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ "
                "for more tips."
            )
            combined_msg = f"{msg} (original error: {e})"
            logger.error(combined_msg)
            raise e
        logger.info_once(
            "Model loading took %s GiB memory and %.6f seconds",
            format_gib(self.model_memory_usage),
            time_after_load - time_before_load,
            scope="local",
        )
        prepare_communication_buffer_for_model(self.model)
        if (drafter := getattr(self, "drafter", None)) and (
            drafter_model := getattr(drafter, "model", None)
        ):
            prepare_communication_buffer_for_model(drafter_model)
        mm_config = self.model_config.multimodal_config
        self.is_multimodal_pruning_enabled = (
            supports_multimodal_pruning(self.get_model())
            and mm_config is not None
            and mm_config.is_multimodal_pruning_enabled()
        )

        if is_mixture_of_experts(self.model) and self.parallel_config.enable_eplb:
            logger.info_once("EPLB is enabled for model %s.", self.model_config.model)
            global_expert_load = (
                global_expert_loads[eplb_models] if global_expert_loads else None
            )
            old_global_expert_indices = (
                old_global_expert_indices_per_model[eplb_models]
                if old_global_expert_indices_per_model
                else None
            )
            assert self.eplb_state is not None
            self.eplb_state.add_model(
                self.model,
                self.model_config,
                global_expert_load,
                old_global_expert_indices,
                rank_mapping,
            )
            if self.eplb_state.is_async:
                self.eplb_state.start_async_loop(rank_mapping=rank_mapping)

        if (
            self.vllm_config.compilation_config.mode
            == CompilationMode.STOCK_TORCH_COMPILE
        ):
            backend = self.vllm_config.compilation_config.init_backend(self.vllm_config)
            compilation_counter.stock_torch_compile_count += 1
            self.model.compile(fullgraph=True, backend=backend)
            return
        # for other compilation modes, cudagraph behavior is controlled by
        # CudagraphWraper and CudagraphDispatcher of vllm.

        # wrap the model with full cudagraph wrapper if needed.
        cudagraph_mode = self.compilation_config.cudagraph_mode
        assert cudagraph_mode is not None
        if (
            cudagraph_mode.has_full_cudagraphs()
            and not self.parallel_config.use_ubatching
        ):
            self.model = CUDAGraphWrapper(
                self.model, self.vllm_config, runtime_mode=CUDAGraphMode.FULL
            )
        elif self.parallel_config.use_ubatching:
            if cudagraph_mode.has_full_cudagraphs():
                self.model = UBatchWrapper(
                    self.model, self.vllm_config, CUDAGraphMode.FULL, self.device
                )
            else:
                self.model = UBatchWrapper(
                    self.model, self.vllm_config, CUDAGraphMode.NONE, self.device
                )

    def _get_eagle3_aux_layers_from_config(self) -> tuple[int, ...] | None:
        """Extract Eagle3 auxiliary layer indices from speculative config.

        These indices specify which hidden states from the base model should
        be used as auxiliary inputs for the Eagle3 drafter model during
        speculative decoding.

        Returns:
            Tuple of layer indices if found in draft model config,
            None otherwise.
        """
        if not (self.speculative_config and self.speculative_config.draft_model_config):
            return None

        hf_config = self.speculative_config.draft_model_config.hf_config
        if not hasattr(hf_config, "eagle_aux_hidden_state_layer_ids"):
            # [FA2/SM70 25.08] РАНЬШЕ ЗДЕСЬ БЫЛ ГЛУХОЙ ВЫХОД, и он молча уводил на умолчание
            # eagle3 (ТРИ слоя). У чекпойнта DFlash2 этого поля нет вовсе -- слои названы
            # `dflash_config.target_layer_ids`. Тихий откат на умолчание -- худший вид отказа:
            # сеть поднимается, а черновик кормится не тем; ловится только формой (15360 != 25600).
            return self._dflash_aux_layers()

        layer_ids = hf_config.eagle_aux_hidden_state_layer_ids
        if layer_ids and isinstance(layer_ids, (list, tuple)):
            return tuple(layer_ids)

        return self._dflash_aux_layers()

    def _dflash_aux_layers(self) -> tuple[int, ...] | None:
        """[FA2/SM70 25.08] Слои-доноры для DFlash/DFlash2 -- ДОПИСАНО, прежнее не тронуто.

        Метод выше знает лишь про `eagle_aux_hidden_state_layer_ids`; у чекпойнта DFlash2 слои
        названы иначе -- `dflash_config.target_layer_ids` = [5,19,33,47,61], и у них СВОЯ
        нумерация: движку нужен индекс СЛЕДУЮЩЕГО слоя (+1), потому что признак снимается на
        входе слоя, а не на выходе. Без этого черновик получал ТРИ признака вместо пяти
        (умолчание eagle3), и `fc` ждал 25600 против приехавших 15360 -- ровно то, на чём
        споткнулся первый пуск. Полный разбор нумерации живёт в gpu/spec_decode/eagle3_utils.py,
        мы его здесь и переиспользуем, а не копируем.
        """
        try:
            from vllm.v1.worker.gpu.spec_decode.eagle3_utils import (
                get_eagle3_aux_layers_from_config,
            )
        except ImportError:
            return None
        return get_eagle3_aux_layers_from_config(self.speculative_config)

    def reload_weights(
        self,
        weights_iterator: Iterable[tuple[str, torch.Tensor]] | None = None,
        weights_path: str | None = None,
        is_checkpoint_format: bool = True,
    ) -> None:
        """
        Reload weights from a weights iterator or from disk

        :param weights_iterator: weights to load into model
        :param weights_path: path to load weights from if weights_iterator is not
            provided. Use path of original model if neither is provided.
        :param is_checkpoint_format: set to False if weights have already been processed
            into kernel format (repacking, renaming, ect.)
        """
        # TODO(@kylesayrs): generalize to all runners and loaders
        # argument validation
        if weights_iterator is None and not is_checkpoint_format:
            logger.warning(
                "Reloading from disk means that weights will be in checkpoint format. "
                "Please use `is_checkpoint_format=True` "
                "to avoid weight reloading errors"
            )

        model = self.get_model()
        weights_to_load = {name for name, _ in model.named_parameters()}
        counter_before_reloading = time.perf_counter()

        # load weights from disk if none are provided
        if weights_iterator is None:
            model_loader = get_model_loader(self.load_config)
            if not hasattr(model_loader, "get_all_weights"):
                raise NotImplementedError(
                    f"Model reloading with `{self.load_config.load_format}` format"
                )

            if weights_path is not None:
                self.model_config.model = weights_path
            weights_iterator = model_loader.get_all_weights(self.model_config, model)
            weights_iterator = cast(
                Iterable[tuple[str, torch.Tensor]], weights_iterator
            )

        # begin loading weights
        logger.info_once("Reloading weights inplace...", scope="local")
        load_device = (
            self.vllm_config.load_config.device or self.vllm_config.device_config.device
        )
        with torch.device(load_device):
            if is_checkpoint_format:
                # load weights from checkpoint/ original model format
                initialize_layerwise_reload(model)
                loaded_weights = model.load_weights(weights_iterator)
                finalize_layerwise_reload(model, self.model_config)

            else:
                # load weights from kernel format
                logger.warning_once(
                    "Reloading with `is_checkpoint_format=True` requires that "
                    "weights be in kernel format and already sharded",
                    scope="local",
                )
                loaded_weights = set()
                for name, loaded_weight in weights_iterator:
                    param = model.get_parameter(name)  # TODO: buffers?
                    param.copy_(loaded_weight)
                    loaded_weights.add(name)

        # logging and validation
        counter_after_reloading = time.perf_counter()
        diff_seconds = counter_after_reloading - counter_before_reloading
        logger.info_once(
            "Reloading and processing weights took %.2f seconds",
            diff_seconds,
            scope="local",
        )
        if self.model_config.quantization is None and loaded_weights is not None:
            weights_not_loaded = weights_to_load - loaded_weights
            if weights_not_loaded:
                logger.warning(
                    "Following weights were not loaded from checkpoint: %s",
                    weights_not_loaded,
                )

    def save_tensorized_model(
        self,
        tensorizer_config: "TensorizerConfig",
    ) -> None:
        TensorizerLoader.save_model(
            self.get_model(),
            tensorizer_config=tensorizer_config,
            model_config=self.model_config,
        )

    def _get_prompt_logprobs_dict(
        self,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: dict[str, int],
    ) -> dict[str, LogprobsTensors | None]:
        num_prompt_logprobs_dict = self.num_prompt_logprobs
        if not num_prompt_logprobs_dict:
            return {}

        in_progress_dict = self.input_batch.in_progress_prompt_logprobs_cpu
        prompt_logprobs_dict: dict[str, LogprobsTensors | None] = {}

        # Since prompt logprobs are a rare feature, prioritize simple,
        # maintainable loop over optimal performance.
        completed_prefill_reqs = []
        for req_id, num_prompt_logprobs in num_prompt_logprobs_dict.items():
            num_tokens = num_scheduled_tokens.get(req_id)
            if num_tokens is None:
                # This can happen if the request was preempted in prefill stage.
                continue

            # Get metadata for this request.
            request = self.requests[req_id]
            if request.prompt_token_ids is None:
                # Prompt logprobs is incompatible with prompt embeddings
                continue

            num_prompt_tokens = len(request.prompt_token_ids)
            prompt_token_ids = torch.tensor(request.prompt_token_ids).to(
                self.device, non_blocking=True
            )

            # Set up target LogprobsTensors object.
            logprobs_tensors = in_progress_dict.get(req_id)
            if not logprobs_tensors:
                # Create empty logprobs CPU tensors for the entire prompt.
                # If chunked, we'll copy in slice by slice.
                logprobs_tensors = LogprobsTensors.empty_cpu(
                    num_prompt_tokens - 1, num_prompt_logprobs + 1
                )
                in_progress_dict[req_id] = logprobs_tensors

            # Determine number of logits to retrieve.
            start_idx = request.num_computed_tokens
            start_tok = start_idx + 1
            num_remaining_tokens = num_prompt_tokens - start_tok
            if num_tokens <= num_remaining_tokens:
                # This is a chunk, more tokens remain.
                # In the == case, there are no more prompt logprobs to produce
                # but we want to defer returning them to the next step where we
                # have new generated tokens to return.
                num_logits = num_tokens
            else:
                # This is the last chunk of prompt tokens to return.
                num_logits = num_remaining_tokens
                completed_prefill_reqs.append(req_id)
                prompt_logprobs_dict[req_id] = logprobs_tensors

            if num_logits <= 0:
                # This can happen for the final chunk if we prefilled exactly
                # (num_prompt_tokens - 1) tokens for this request in the prior
                # step. There are no more prompt logprobs to produce.
                continue

            # Get the logits corresponding to this req's prompt tokens.
            # If this is a partial request (i.e. chunked prefill),
            # then there is prompt logprob generated for each index.
            req_idx = self.input_batch.req_id_to_index[req_id]
            offset = self.query_start_loc.np[req_idx].item()
            prompt_hidden_states = hidden_states[offset : offset + num_logits]
            logits = self.model.compute_logits(prompt_hidden_states)

            # Get the "target" tokens for each index. For prompt at index i,
            # the token at prompt index i+1 is the "sampled" token we want
            # to gather the logprob for.
            tgt_token_ids = prompt_token_ids[start_tok : start_tok + num_logits]

            # Compute prompt logprobs.
            logprobs = self.sampler.compute_logprobs(logits)
            token_ids, logprobs, ranks, _ = self.sampler.gather_logprobs(
                logprobs, num_prompt_logprobs, tgt_token_ids
            )

            # Transfer GPU->CPU async.
            chunk_slice = slice(start_idx, start_idx + num_logits)
            logprobs_tensors.logprob_token_ids[chunk_slice].copy_(
                token_ids, non_blocking=True
            )
            logprobs_tensors.logprobs[chunk_slice].copy_(logprobs, non_blocking=True)
            logprobs_tensors.selected_token_ranks[chunk_slice].copy_(
                ranks, non_blocking=True
            )

        # Remove requests that have completed prefill from the batch
        # num_prompt_logprobs_dict.
        for req_id in completed_prefill_reqs:
            del num_prompt_logprobs_dict[req_id]
            del in_progress_dict[req_id]

        # Must synchronize the non-blocking GPU->CPU transfers.
        if prompt_logprobs_dict:
            self._sync_device()

        return prompt_logprobs_dict

    def _get_nans_in_logits(
        self,
        logits: torch.Tensor | None,
    ) -> dict[str, int]:
        try:
            if logits is None:
                return {req_id: 0 for req_id in self.input_batch.req_ids}

            num_nans_in_logits = {}
            num_nans_for_index = logits.isnan().sum(dim=-1).cpu().numpy()
            for req_id in self.input_batch.req_ids:
                req_index = self.input_batch.req_id_to_index[req_id]
                num_nans_in_logits[req_id] = (
                    int(num_nans_for_index[req_index])
                    if num_nans_for_index is not None and req_index < logits.shape[0]
                    else 0
                )
            return num_nans_in_logits
        except IndexError:
            return {}

    @contextmanager
    def maybe_randomize_inputs(
        self, input_ids: torch.Tensor | None, inputs_embeds: torch.Tensor | None
    ):
        """
        Randomize input_ids if VLLM_RANDOMIZE_DP_DUMMY_INPUTS is set.
        This is to help balance expert-selection
         - during profile_run
         - during DP rank dummy run
        """

        dp_size = self.vllm_config.parallel_config.data_parallel_size
        randomize_inputs = envs.VLLM_RANDOMIZE_DP_DUMMY_INPUTS and dp_size > 1
        if not randomize_inputs:
            yield
        elif input_ids is not None:

            @functools.cache
            def rand_input_ids() -> torch.Tensor:
                return torch.randint_like(
                    self.input_ids.gpu,
                    low=0,
                    high=self.model_config.get_vocab_size(),
                )

            logger.debug_once("Randomizing dummy input_ids for DP Rank")
            input_ids.copy_(rand_input_ids()[: input_ids.size(0)], non_blocking=True)
            yield
            input_ids.fill_(0)
        else:

            @functools.cache
            def rand_inputs_embeds() -> torch.Tensor:
                return torch.randn_like(
                    self.inputs_embeds.gpu,
                )

            assert inputs_embeds is not None
            logger.debug_once("Randomizing dummy inputs_embeds for DP Rank")
            inputs_embeds.copy_(
                rand_inputs_embeds()[: inputs_embeds.size(0)], non_blocking=True
            )
            yield
            inputs_embeds.fill_(0)

    def _get_mm_dummy_batch(
        self,
        modality: str,
        max_items_per_batch: int,
    ) -> BatchedTensorInputs:
        """Dummy data for profiling and precompiling multimodal models."""
        assert self.mm_budget is not None

        # Don't use `max_items_per_batch` here to avoid redundant computation
        dummy_mm_inputs = self.mm_registry.get_dummy_mm_inputs(
            self.model_config,
            mm_counts={modality: 1},
            cache=self.mm_budget.cache,
        )
        dummy_mm_item = dummy_mm_inputs["mm_kwargs"][modality][0]

        # We use the cache so that the item is saved to the cache,
        # but not read from the cache
        assert dummy_mm_item is not None, "Item should not already be cached"

        return next(
            mm_kwargs_group
            for _, _, mm_kwargs_group in group_mm_kwargs_by_modality(
                [(modality, dummy_mm_item)] * max_items_per_batch,
                device=self.device,
                pin_memory=self.pin_memory,
            )
        )

    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        cudagraph_runtime_mode: CUDAGraphMode | None = None,
        force_attention: bool = False,
        uniform_decode: bool = False,
        allow_microbatching: bool = True,
        skip_eplb: bool = False,
        is_profile: bool = False,
        create_mixed_batch: bool = False,
        remove_lora: bool = True,
        is_graph_capturing: bool = False,
        num_active_loras: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run a dummy forward pass to warm up/profile run or capture the
        CUDA graph for the model.

        Args:
            num_tokens: Number of tokens to run the dummy forward pass.
            cudagraph_runtime_mode: used to control the behavior.
                - if not set will determine the cudagraph mode based on using
                    the self.cudagraph_dispatcher.
                - CUDAGraphMode.NONE: No cudagraph, for warm up and profile run
                - CUDAGraphMode.PIECEWISE: Piecewise cudagraph.
                - CUDAGraphMode.FULL: Full cudagraph, attention metadata is
                    needed.
            force_attention: If True, always create attention metadata. Used to
                warm up attention backend when mode is NONE.
            uniform_decode: If True, the batch is a uniform decode batch.
            skip_eplb: If True, skip EPLB state update.
            is_profile: If True, this is a profile run.
            create_mixed_batch: If True, create a mixed batch with both decode
                (1 token) and prefill (multiple tokens) requests.
            remove_lora: If False, dummy LoRAs are not destroyed after the run
            num_active_loras: Number of distinct active LoRAs to capture for.
                LoRA is activated when num_active_loras > 0.
        """
        mm_config = self.vllm_config.model_config.multimodal_config
        if mm_config and mm_config.mm_encoder_only:
            # The current dummy run only covers LM execution, so we can skip it.
            # mm encoder dummy run may need to add in the future.
            return torch.tensor([]), torch.tensor([])

        assert (
            cudagraph_runtime_mode is None
            or cudagraph_runtime_mode.valid_runtime_modes()
        )

        # If cudagraph_mode.decode_mode() == FULL and
        # cudagraph_mode.separate_routine(). This means that we are using
        # different graphs and/or modes for mixed prefill-decode batches vs.
        # uniform decode batches. A uniform decode batch means that all
        # requests have identical query length, except a potential virtual
        # request (shorter) in the batch account for padding.
        # Uniform decode batch could either be common pure decode, where
        # max_query_len == 1, or speculative decode, where
        # max_query_len == 1 + num_spec_decode_tokens.

        # When setting max_query_len = 1, we switch to and capture the optimized
        # routine of FA2 for pure decode, i.e., Flashdecode + an optimization
        # for GQA/MQA.
        max_query_len = self.uniform_decode_query_len if uniform_decode else num_tokens

        # Set num_scheduled_tokens based on num_tokens and max_num_seqs
        # for dummy run with LoRA so that the num_reqs collectively
        # has num_tokens in total.
        assert num_tokens <= self.scheduler_config.max_num_batched_tokens
        max_num_reqs = self.scheduler_config.max_num_seqs
        if create_mixed_batch:
            assert not uniform_decode
            # Create mixed batch:
            # first half decode tokens, second half one prefill
            num_decode_tokens = min(max_num_reqs - 1, num_tokens // 2)
            num_prefill_tokens = num_tokens - num_decode_tokens
            num_reqs = num_decode_tokens + 1

            # Create decode requests (1 token each) followed by prefill request
            num_scheduled_tokens_list = [1] * num_decode_tokens + [num_prefill_tokens]
            # Note: Overriding max_query_len to be the prefill tokens
            max_query_len = num_prefill_tokens
        elif uniform_decode:
            assert not create_mixed_batch
            num_reqs = min(max_num_reqs, cdiv(num_tokens, max_query_len))
            num_scheduled_tokens_list = [max_query_len] * num_reqs
            if num_tokens % max_query_len != 0:
                num_scheduled_tokens_list[-1] = num_tokens % max_query_len
        else:
            num_reqs = min(num_tokens, max_num_reqs)
            min_tokens_per_req = num_tokens // num_reqs
            num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
            num_scheduled_tokens_list[-1] += num_tokens % num_reqs

        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs
        num_scheduled_tokens = np.array(num_scheduled_tokens_list, dtype=np.int32)
        num_tokens_unpadded = int(num_scheduled_tokens.sum())

        num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)

        _cudagraph_mode, batch_desc, should_ubatch, num_tokens_across_dp, _ = (
            self._determine_batch_execution_and_padding(
                num_tokens=num_tokens_unpadded,
                num_reqs=num_reqs,
                num_scheduled_tokens_np=num_scheduled_tokens,
                max_num_scheduled_tokens=max_query_len,
                use_cascade_attn=False,
                allow_microbatching=allow_microbatching,
                force_eager=is_profile
                or (cudagraph_runtime_mode == CUDAGraphMode.NONE),
                # `force_uniform_decode` is used for cudagraph capture; because for
                # capturing mixed prefill-decode batches, we sometimes use
                # num_tokens == num_reqs which looks like a uniform decode batch to the
                # dispatcher; but we actually want to capture a piecewise cudagraph
                force_uniform_decode=uniform_decode,
                # `force_has_lora` is used for cudagraph capture; because LoRA is
                # activated later in the context manager, but we need to know the
                # LoRA state when determining the batch descriptor for capture
                force_has_lora=num_active_loras > 0,
                # `force_num_active_loras` is used for cudagraph capture; because we
                # need to capture graphs for specific num_active_loras counts
                force_num_active_loras=num_active_loras,
            )
        )

        if cudagraph_runtime_mode is None:
            cudagraph_runtime_mode = _cudagraph_mode
        else:
            assert cudagraph_runtime_mode == _cudagraph_mode, (
                f"Cudagraph runtime mode mismatch in dummy_run. "
                f"Expected {_cudagraph_mode}, but got {cudagraph_runtime_mode}."
            )

        num_tokens_padded = batch_desc.num_tokens
        num_reqs_padded = (
            batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
        )
        ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
            should_ubatch,
            num_scheduled_tokens,
            num_tokens_padded,
            num_reqs_padded,
            self.vllm_config.parallel_config.num_ubatches,
        )
        logger.debug(
            "ubatch_slices: %s, ubatch_slices_padded: %s",
            ubatch_slices,
            ubatch_slices_padded,
        )

        attn_metadata: PerLayerAttnMetadata | None = None

        slot_mappings_by_group, slot_mappings = self._get_slot_mappings(
            num_tokens_padded=num_tokens,
            num_reqs_padded=num_reqs_padded,
            num_tokens_unpadded=num_tokens_unpadded,
            ubatch_slices=ubatch_slices_padded,
        )

        # If force_attention is True, we always capture attention. Otherwise,
        # it only happens for cudagraph_runtime_mode=FULL.
        if force_attention or cudagraph_runtime_mode == CUDAGraphMode.FULL:
            if create_mixed_batch:
                # In the mixed batch mode (used for FI warmup), we use
                # shorter sequence lengths to run faster.
                # TODO(luka) better system for describing dummy batches
                seq_lens = [1] * num_decode_tokens + [num_prefill_tokens + 1]
            else:
                seq_lens = max_query_len  # type: ignore[assignment]
            self.seq_lens.np[:num_reqs] = seq_lens
            self.seq_lens.np[num_reqs:] = 0
            self.seq_lens.copy_to_gpu()

            cum_num_tokens, _ = self._get_cumsum_and_arange(num_scheduled_tokens)
            self.query_start_loc.np[1 : num_reqs + 1] = cum_num_tokens
            self.query_start_loc.copy_to_gpu()

            pad_attn = cudagraph_runtime_mode == CUDAGraphMode.FULL
            attn_metadata, _ = self._build_attention_metadata(
                num_tokens=num_tokens_unpadded,
                num_reqs=num_reqs_padded,
                max_query_len=max_query_len,
                ubatch_slices=ubatch_slices_padded if pad_attn else ubatch_slices,
                for_cudagraph_capture=is_graph_capturing,
                slot_mappings=slot_mappings_by_group,
            )

        with self.maybe_dummy_run_with_lora(
            self.lora_config,
            num_scheduled_tokens,
            num_sampled_tokens,
            remove_lora,
            num_active_loras,
        ):
            # Make sure padding doesn't exceed max_num_tokens
            assert num_tokens_padded <= self.max_num_tokens
            model_kwargs = self._init_model_kwargs()
            if self.supports_mm_inputs and not self.model_config.is_encoder_decoder:
                input_ids, inputs_embeds = self._prepare_mm_inputs(num_tokens_padded)

                model_kwargs = {
                    **model_kwargs,
                    **self._dummy_mm_kwargs(num_reqs),
                }
            elif self.enable_prompt_embeds:
                input_ids = None
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded]
                model_kwargs = self._init_model_kwargs()
            else:
                input_ids = self.input_ids.gpu[:num_tokens_padded]
                inputs_embeds = None

            if self.uses_mrope:
                positions = self.mrope_positions.gpu[:, :num_tokens_padded]
            elif self.uses_xdrope_dim > 0:
                positions = self.xdrope_positions.gpu[:, :num_tokens_padded]
            else:
                positions = self.positions.gpu[:num_tokens_padded]

            if get_pp_group().is_first_rank:
                intermediate_tensors = None
            else:
                if self.intermediate_tensors is None:
                    self.intermediate_tensors = (
                        self.model.make_empty_intermediate_tensors(
                            batch_size=self.max_num_tokens,
                            dtype=self.model_config.dtype,
                            device=self.device,
                        )
                    )

                intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                    num_tokens_padded, None, False
                )

            if ubatch_slices_padded is not None:
                # Adjust values to reflect a single ubatch.
                # TODO(sage,lucas): this is cruft that should be addressed in
                #  the padding refactor.
                num_tokens_padded = ubatch_slices_padded[0].num_tokens
                if num_tokens_across_dp is not None:
                    num_tokens_across_dp[:] = num_tokens_padded

            with (
                self.maybe_randomize_inputs(input_ids, inputs_embeds),
                set_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens_padded,
                    num_tokens_across_dp=num_tokens_across_dp,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    batch_descriptor=batch_desc,
                    ubatch_slices=ubatch_slices_padded,
                    slot_mapping=slot_mappings,
                ),
            ):
                outputs = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    **model_kwargs,
                )

            if self.use_aux_hidden_state_outputs:
                hidden_states, _ = outputs
            else:
                hidden_states = outputs

            if self.speculative_config and (
                self.speculative_config.use_eagle()
                or self.speculative_config.uses_draft_model()
            ):
                assert isinstance(self.drafter, EagleProposer | DraftModelProposer | DFlash2Proposer)
                assert self.speculative_config is not None
                # Eagle currently only supports PIECEWISE cudagraphs.
                # Therefore only use cudagraphs if the main model uses PIECEWISE
                # NOTE(lucas): this is a hack, need to clean up.
                use_cudagraphs = (
                    (
                        is_graph_capturing
                        and cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
                    )
                    or (
                        not is_graph_capturing
                        and cudagraph_runtime_mode != CUDAGraphMode.NONE
                    )
                ) and not self.speculative_config.enforce_eager

                # Note(gnovack) - We need to disable cudagraphs for one of the two
                # lora cases when cudagraph_specialize_lora is enabled. This is a
                # short term mitigation for issue mentioned in
                # https://github.com/vllm-project/vllm/issues/28334
                if (
                    self.compilation_config.cudagraph_specialize_lora
                    and num_active_loras > 0
                ):
                    use_cudagraphs = False

                self.drafter.dummy_run(
                    num_tokens,
                    use_cudagraphs=use_cudagraphs,
                    is_graph_capturing=is_graph_capturing,
                    slot_mappings=slot_mappings,
                )

        # We register layerwise NVTX hooks here after the first dynamo tracing is
        # done to avoid nvtx operations in hook functions being traced by
        # torch dynamo and causing graph breaks.
        # Note that for DYNAMO_ONCE and VLLM_COMPILE mode,
        # compiled model's dynamo tracing is only done once and the compiled model's
        # __call__ function is replaced by calling the compiled function.
        # So it's safe to register hooks here. Hooks will be registered to
        # both compiled and uncompiled models but they will never
        # be called on the compiled model execution path.
        self._register_layerwise_nvtx_hooks()

        # This is necessary to avoid blocking DP.
        # For dummy runs, we typically skip EPLB since we don't have any real
        # requests to process.
        # However, in DP settings, there may be cases when some DP ranks do
        # not have any requests to process, so they're executing dummy batches.
        # In such cases, we still have to trigger EPLB to make sure
        # ranks execute the rearrangement in synchronization.
        if not skip_eplb:
            self.eplb_step(is_dummy=True, is_profile=is_profile)

        logit_indices = np.cumsum(num_scheduled_tokens) - 1
        logit_indices_device = torch.from_numpy(logit_indices).to(
            self.device, non_blocking=True
        )
        return hidden_states, hidden_states[logit_indices_device]

    @torch.inference_mode()
    def _dummy_sampler_run(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # The dummy hidden states may contain special values,
        # like `inf` or `nan`.
        # To avoid breaking the sampler, we use a random tensor here instead.

        mm_config = self.vllm_config.model_config.multimodal_config
        if mm_config and mm_config.mm_encoder_only:
            # MM Encoder only model no need to run sampler.
            return torch.tensor([])

        hidden_states = torch.rand_like(hidden_states)

        logits = self.model.compute_logits(hidden_states)
        num_reqs = logits.size(0)

        dummy_tensors = lambda v: torch.full((num_reqs,), v, device=self.device)

        dummy_metadata = SamplingMetadata(
            temperature=dummy_tensors(0.5),
            all_greedy=False,
            all_random=False,
            top_p=dummy_tensors(0.9),
            top_k=dummy_tensors(logits.size(1) - 1),
            generators={},
            max_num_logprobs=None,
            no_penalties=True,
            prompt_token_ids=None,
            frequency_penalties=dummy_tensors(0.1),
            presence_penalties=dummy_tensors(0.1),
            repetition_penalties=dummy_tensors(0.1),
            output_token_ids=[[] for _ in range(num_reqs)],
            spec_token_ids=[[] for _ in range(num_reqs)],
            allowed_token_ids_mask=None,
            bad_words_token_ids={},
            logitsprocs=LogitsProcessors(),
        )
        try:
            sampler_output = self.sampler(
                logits=logits, sampling_metadata=dummy_metadata
            )
        except RuntimeError as e:
            if "out of memory" in str(e):
                raise RuntimeError(
                    "CUDA out of memory occurred when warming up sampler with "
                    f"{num_reqs} dummy requests. Please try lowering "
                    "`max_num_seqs` or `gpu_memory_utilization` when "
                    "initializing the engine."
                ) from e
            else:
                raise e
        if self.speculative_config:
            draft_token_ids = [[0] for _ in range(num_reqs)]
            dummy_spec_decode_metadata = SpecDecodeMetadata.make_dummy(
                draft_token_ids, self.device
            )

            num_tokens = sum(len(ids) for ids in draft_token_ids)
            # draft_probs = torch.randn(
            #     num_tokens, logits.shape[-1], device=self.device,
            #     dtype=logits.dtype)
            draft_probs = None
            logits = torch.randn(
                num_tokens + num_reqs,
                logits.shape[-1],
                device=self.device,
                dtype=logits.dtype,
            )
            self.rejection_sampler(
                dummy_spec_decode_metadata,
                draft_probs,
                logits,
                dummy_metadata,
            )
        return sampler_output

    def _dummy_pooler_run_task(
        self,
        hidden_states: torch.Tensor,
        task: PoolingTask,
    ) -> PoolerOutput:
        num_tokens = hidden_states.shape[0]
        max_num_reqs = self.scheduler_config.max_num_seqs
        num_reqs = min(num_tokens, max_num_reqs)
        min_tokens_per_req = num_tokens // num_reqs
        num_scheduled_tokens_np = np.full(num_reqs, min_tokens_per_req)
        num_scheduled_tokens_np[-1] += num_tokens % num_reqs
        assert np.sum(num_scheduled_tokens_np) == num_tokens
        assert len(num_scheduled_tokens_np) == num_reqs

        req_num_tokens = num_tokens // num_reqs

        dummy_prompt_lens = torch.from_numpy(num_scheduled_tokens_np)
        dummy_token_ids = torch.zeros(
            (num_reqs, req_num_tokens), dtype=torch.int32, device=self.device
        )

        model = cast(VllmModelForPooling, self.get_model())
        dummy_pooling_params = PoolingParams(task=task)
        dummy_pooling_params.verify(task=task, model_config=self.model_config)
        to_update = model.pooler.get_pooling_updates(task)
        to_update.apply(dummy_pooling_params)

        dummy_metadata = PoolingMetadata(
            prompt_lens=dummy_prompt_lens,
            prompt_token_ids=dummy_token_ids,
            pooling_params=[dummy_pooling_params] * num_reqs,
            pooling_states=[PoolingStates() for i in range(num_reqs)],
        )

        dummy_metadata.build_pooling_cursor(
            num_scheduled_tokens_np,
            seq_lens_cpu=dummy_prompt_lens,
            device=hidden_states.device,
        )

        try:
            return model.pooler(
                hidden_states=hidden_states, pooling_metadata=dummy_metadata
            )
        except RuntimeError as e:
            if "out of memory" in str(e):
                raise RuntimeError(
                    "CUDA out of memory occurred when warming up pooler "
                    f"({task=}) with {num_reqs} dummy requests. Please try "
                    "lowering `max_num_seqs` or `gpu_memory_utilization` when "
                    "initializing the engine."
                ) from e
            else:
                raise e

    @torch.inference_mode()
    def _dummy_pooler_run(
        self,
        hidden_states: torch.Tensor,
    ) -> PoolerOutput:
        mm_config = self.vllm_config.model_config.multimodal_config
        if mm_config and mm_config.mm_encoder_only:
            # MM Encoder only model not need to run pooler.
            return torch.tensor([])

        # Find the task that has the largest output for subsequent steps
        supported_pooling_tasks = self.get_supported_pooling_tasks()

        if not supported_pooling_tasks:
            raise RuntimeError(
                f"Model {self.model_config.model} does not support "
                "any pooling tasks. See "
                "https://docs.vllm.ai/en/latest/models/pooling_models.html "
                "to learn more."
            )

        output_size = dict[PoolingTask, float]()
        for task in supported_pooling_tasks:
            # Run a full batch with each task to ensure none of them OOMs
            output = self._dummy_pooler_run_task(hidden_states, task)
            output_size[task] = sum(o.nbytes for o in output if o is not None)
            del output  # Allow GC

        max_task = max(output_size.items(), key=lambda x: x[1])[0]
        return self._dummy_pooler_run_task(hidden_states, max_task)

    def profile_run(self) -> None:
        # Profile with multimodal encoder & encoder cache.
        if self.supports_mm_inputs:
            mm_config = self.model_config.multimodal_config
            if mm_config is not None and mm_config.skip_mm_profiling:
                logger.info(
                    "Skipping memory profiling for multimodal encoder and "
                    "encoder cache."
                )
            else:
                mm_budget = self.mm_budget
                assert mm_budget is not None

                if (encoder_budget := mm_budget.get_encoder_budget()) > 0:
                    # NOTE: Currently model is profiled with a single non-text
                    # modality with the max possible input tokens even when
                    # it supports multiple.
                    dummy_modality = mm_budget.get_modality_with_max_tokens()
                    max_mm_items_per_batch = mm_budget.mm_max_items_per_batch[
                        dummy_modality
                    ]

                    logger.info(
                        "Encoder cache will be initialized with a budget of "
                        "%s tokens, and profiled with %s %s items of the "
                        "maximum feature size.",
                        encoder_budget,
                        max_mm_items_per_batch,
                        dummy_modality,
                    )

                    # Create dummy batch of multimodal inputs.
                    batched_dummy_mm_inputs = self._get_mm_dummy_batch(
                        dummy_modality,
                        max_mm_items_per_batch,
                    )

                    # Run multimodal encoder.
                    dummy_encoder_outputs = self.model.embed_multimodal(
                        **batched_dummy_mm_inputs
                    )

                    sanity_check_mm_encoder_outputs(
                        dummy_encoder_outputs,
                        expected_num_items=max_mm_items_per_batch,
                    )
                    for i, output in enumerate(dummy_encoder_outputs):
                        self.encoder_cache[f"tmp_{i}"] = output

        # Add `is_profile` here to pre-allocate communication buffers
        hidden_states, last_hidden_states = self._dummy_run(
            self.max_num_tokens, is_profile=True
        )
        if get_pp_group().is_last_rank:
            if self.is_pooling_model:
                output = self._dummy_pooler_run(hidden_states)
            else:
                output = self._dummy_sampler_run(last_hidden_states)
        else:
            output = None
        self._sync_device()
        del hidden_states, output
        self.encoder_cache.clear()
        gc.collect()

    def capture_model(self) -> int:
        if self.compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
            logger.warning(
                "Skipping CUDA graph capture. To turn on CUDA graph capture, "
                "ensure `cudagraph_mode` was not manually set to `NONE`"
            )
            return 0

        compilation_counter.num_gpu_runner_capture_triggers += 1

        # [МИНА ГРАФА, 06.09] ТАБЛИЦЫ МАСШТАБОВ int16 СОЗДАЮТСЯ ЗДЕСЬ, ДО ЗАХВАТА.
        # Они привязываются лениво к тензору состояния (`fa2sm70_gdn_i16.таблица`), а
        # первый вызов на НАСТОЯЩЕМ пуле -- это и есть захват графа: тензор, выделенный
        # во время захвата, живёт в пуле графа, после захвата эта память переиспользуется,
        # а запечённый указатель продолжает по ней писать. Снаружи -- Xid 13 и смерть
        # воркера. Бисекция 06.09 (§153) показала: дефекту нужны РОВНО три условия --
        # спекуляция, полный граф и int16-пул; снятие любого лечит.
        try:
            import fa2sm70_gdn_i16 as _i16м
            if _i16м.включён():
                _сд = 0
                for _имя, _сл in self.compilation_config.static_forward_context.items():
                    _кв = getattr(_сл, "kv_cache", None)
                    if not _кв:
                        continue
                    for _вэ in _кв:
                        _сост = _вэ[1] if isinstance(_вэ, (list, tuple)) and len(_вэ) > 1 else None
                        if _сост is not None and getattr(_сост, "dtype", None) == torch.int16:
                            _i16м.таблица(_сост)
                            _сд += 1
                logger.info("[fa2_sm70] таблиц масштабов int16 заведено ДО захвата: %d", _сд)
        except Exception as _e:
            logger.warning("[fa2_sm70] масштабы int16 до захвата не заведены: %s", _e)

        start_time = time.perf_counter()

        @contextmanager
        def freeze_gc():
            # Optimize garbage collection during CUDA graph capture.
            # Clean up, then freeze all remaining objects from being included
            # in future collections.
            gc.collect()
            should_freeze = not envs.VLLM_ENABLE_CUDAGRAPH_GC
            if should_freeze:
                gc.freeze()
            try:
                yield
            finally:
                if should_freeze:
                    gc.unfreeze()
                    gc.collect()

        # Trigger CUDA graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        set_cudagraph_capturing_enabled(True)
        with freeze_gc(), graph_capture(device=self.device):
            start_free_gpu_memory = torch.cuda.mem_get_info()[0]

            for (
                runtime_mode,
                batch_descs,
            ) in self.cudagraph_dispatcher.get_capture_descs():
                self._capture_cudagraphs(
                    batch_descriptors=batch_descs,
                    cudagraph_runtime_mode=runtime_mode,
                )

            torch.cuda.synchronize()
            end_free_gpu_memory = torch.cuda.mem_get_info()[0]

        # Disable cudagraph capturing globally, so any unexpected cudagraph
        # capturing will be detected and raise an error after here.
        # Note: We don't put it into graph_capture context manager because
        # we may do lazy capturing in future that still allows capturing
        # after here.
        set_cudagraph_capturing_enabled(False)

        # Lock workspace to prevent resizing during execution.
        # Max workspace sizes should have been captured during warmup/profiling.
        lock_workspace()

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
        # This usually takes 5~20 seconds.
        logger.info_once(
            "Graph capturing finished in %.0f secs, took %.2f GiB",
            elapsed_time,
            cuda_graph_size / (1 << 30),
            scope="local",
        )
        return cuda_graph_size

    def _capture_cudagraphs(
        self,
        batch_descriptors: list[BatchDescriptor],
        cudagraph_runtime_mode: CUDAGraphMode,
    ):
        assert (
            cudagraph_runtime_mode != CUDAGraphMode.NONE
            and cudagraph_runtime_mode.valid_runtime_modes()
        ), f"Invalid cudagraph runtime mode: {cudagraph_runtime_mode}"

        if not batch_descriptors:
            return

        uniform_decode = batch_descriptors[0].uniform
        force_attention = cudagraph_runtime_mode == CUDAGraphMode.FULL

        dummy_run = functools.partial(
            self._dummy_run,
            uniform_decode=uniform_decode,
            skip_eplb=True,
            remove_lora=False,
            force_attention=force_attention,
        )

        # Only rank 0 should print progress bar during capture
        if is_global_first_rank():
            batch_descriptors = tqdm(
                batch_descriptors,
                disable=not self.load_config.use_tqdm_on_load,
                desc="Capturing CUDA graphs ({}, {})".format(
                    "decode" if uniform_decode else "mixed prefill-decode",
                    cudagraph_runtime_mode.name,
                ),
            )

        # We skip EPLB here since we don't want to record dummy metrics
        for batch_desc in batch_descriptors:
            num_tokens = batch_desc.num_tokens
            num_active_loras = batch_desc.num_active_loras

            # We currently only capture ubatched graphs when its a FULL
            # cudagraph, a uniform decode batch, and the number of tokens
            # is above the threshold. Otherwise we just capture a non-ubatched
            # version of the graph
            allow_microbatching = (
                self.parallel_config.use_ubatching
                and cudagraph_runtime_mode == CUDAGraphMode.FULL
                and uniform_decode
                and check_ubatch_thresholds(
                    config=self.vllm_config.parallel_config,
                    num_tokens=num_tokens,
                    uniform_decode=uniform_decode,
                )
            )

            for _ in range(self.compilation_config.cudagraph_num_of_warmups):
                # Use CUDAGraphRuntimeStyle.NONE (default) for warmup.
                # But be careful, warm up with `NONE` is orthogonal to
                # if we want to warm up attention or not. This is
                # different from the case where `FULL` implies capture
                # attention while `PIECEWISE` implies no attention.
                dummy_run(
                    num_tokens,
                    cudagraph_runtime_mode=CUDAGraphMode.NONE,
                    allow_microbatching=allow_microbatching,
                    num_active_loras=num_active_loras,
                )

            # Capture run
            dummy_run(
                num_tokens,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                allow_microbatching=allow_microbatching,
                num_active_loras=num_active_loras,
                is_graph_capturing=True,
            )
        self.maybe_remove_all_loras(self.lora_config)

    def initialize_attn_backend(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize the attention backends and attention metadata builders.
        """
        assert len(self.attn_groups) == 0, "Attention backends are already initialized"

        class AttentionGroupKey(NamedTuple):
            attn_backend: type[AttentionBackend]
            kv_cache_spec: KVCacheSpec

        def get_attn_backends_for_group(
            kv_cache_group_spec: KVCacheGroupSpec,
        ) -> tuple[dict[AttentionGroupKey, list[str]], set[type[AttentionBackend]]]:
            layer_type = cast(type[Any], AttentionLayerBase)
            layers = get_layers_from_vllm_config(
                self.vllm_config, layer_type, kv_cache_group_spec.layer_names
            )
            attn_backends = {}
            attn_backend_layers = defaultdict(list)
            # Dedupe based on full class name; this is a bit safer than
            # using the class itself as the key because when we create dynamic
            # attention backend subclasses (e.g. ChunkedLocalAttention) unless
            # they are cached correctly, there will be different objects per
            # layer.
            for layer_name in kv_cache_group_spec.layer_names:
                attn_backend = layers[layer_name].get_attn_backend()

                if layer_name in self.kv_sharing_fast_prefill_eligible_layers:
                    attn_backend = create_fast_prefill_custom_backend(
                        "FastPrefill",
                        attn_backend,  # type: ignore[arg-type]
                    )

                full_cls_name = attn_backend.full_cls_name()
                layer_kv_cache_spec = kv_cache_group_spec.kv_cache_spec
                if isinstance(layer_kv_cache_spec, UniformTypeKVCacheSpecs):
                    layer_kv_cache_spec = layer_kv_cache_spec.kv_cache_specs[layer_name]
                key = (full_cls_name, layer_kv_cache_spec)
                attn_backends[key] = AttentionGroupKey(
                    attn_backend, layer_kv_cache_spec
                )
                attn_backend_layers[key].append(layer_name)
            return (
                {attn_backends[k]: v for k, v in attn_backend_layers.items()},
                set(group_key.attn_backend for group_key in attn_backends.values()),
            )

        def create_attn_groups(
            attn_backends_map: dict[AttentionGroupKey, list[str]],
            kv_cache_group_id: int,
        ) -> list[AttentionGroup]:
            attn_groups: list[AttentionGroup] = []
            for (attn_backend, kv_cache_spec), layer_names in attn_backends_map.items():
                attn_group = AttentionGroup(
                    attn_backend,
                    layer_names,
                    kv_cache_spec,
                    kv_cache_group_id,
                )

                attn_groups.append(attn_group)
            return attn_groups

        attention_backend_maps = []
        attention_backend_list = []
        for kv_cache_group_spec in kv_cache_config.kv_cache_groups:
            attn_backends = get_attn_backends_for_group(kv_cache_group_spec)
            attention_backend_maps.append(attn_backends[0])
            attention_backend_list.append(attn_backends[1])

        # Resolve cudagraph_mode before actually initialize metadata_builders
        self._check_and_update_cudagraph_mode(
            attention_backend_list, kv_cache_config.kv_cache_groups
        )

        # Check if attention backend supports PCP&DCP and related features.
        check_attention_cp_compatibility(self.vllm_config)

        for i, attn_backend_map in enumerate(attention_backend_maps):
            self.attn_groups.append(create_attn_groups(attn_backend_map, i))

    def initialize_metadata_builders(
        self, kv_cache_config: KVCacheConfig, kernel_block_sizes: list[int]
    ) -> None:
        """
        Create the metadata builders for all KV cache groups and attn groups.
        """
        for kv_cache_group_id in range(len(kv_cache_config.kv_cache_groups)):
            for attn_group in self.attn_groups[kv_cache_group_id]:
                attn_group.create_metadata_builders(
                    self.vllm_config,
                    self.device,
                    kernel_block_sizes[kv_cache_group_id]
                    if kv_cache_group_id < len(kernel_block_sizes)
                    else None,
                    num_metadata_builders=1
                    if not self.parallel_config.use_ubatching
                    else self.parallel_config.num_ubatches,
                )
        # Calculate reorder batch threshold (if needed)
        # Note (tdoublep): do this *after* constructing builders,
        # because some of them change the threshold at init time.
        self.calculate_reorder_batch_threshold()

    def _check_and_update_cudagraph_mode(
        self,
        attention_backends: list[set[type[AttentionBackend]]],
        kv_cache_groups: list[KVCacheGroupSpec],
    ) -> None:
        """
        Resolve the cudagraph_mode when there are multiple attention
        groups with potential conflicting CUDA graph support.
        Then initialize the cudagraph_dispatcher based on the resolved
        cudagraph_mode.
        """
        min_cg_support = AttentionCGSupport.ALWAYS
        min_cg_backend_name = None

        for attn_backend_set, kv_cache_group in zip(
            attention_backends, kv_cache_groups
        ):
            for attn_backend in attn_backend_set:
                builder_cls = attn_backend.get_builder_cls()

                cg_support = builder_cls.get_cudagraph_support(
                    self.vllm_config, kv_cache_group.kv_cache_spec
                )
                if cg_support.value < min_cg_support.value:
                    min_cg_support = cg_support
                    min_cg_backend_name = attn_backend.__name__
        # Flexible resolve the cudagraph mode
        cudagraph_mode = self.compilation_config.cudagraph_mode
        assert cudagraph_mode is not None
        # check cudagraph for mixed batch is supported
        if (
            cudagraph_mode.mixed_mode() == CUDAGraphMode.FULL
            and min_cg_support != AttentionCGSupport.ALWAYS
        ):
            msg = (
                f"CUDAGraphMode.{cudagraph_mode.name} is not supported "
                f"with {min_cg_backend_name} backend (support: "
                f"{min_cg_support})"
            )
            if min_cg_support == AttentionCGSupport.NEVER:
                # if not supported any full cudagraphs, just raise it.
                msg += (
                    "; please try cudagraph_mode=PIECEWISE, and "
                    "make sure compilation mode is VLLM_COMPILE"
                )
                raise ValueError(msg)

            # attempt to resolve the full cudagraph related mode
            if self.compilation_config.splitting_ops_contain_attention():
                msg += "; setting cudagraph_mode=FULL_AND_PIECEWISE"
                cudagraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.FULL_AND_PIECEWISE
                )
            else:
                msg += "; setting cudagraph_mode=FULL_DECODE_ONLY"
                cudagraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.FULL_DECODE_ONLY
                )
            logger.warning(msg)

        # check that if we are doing decode full-cudagraphs it is supported
        if (
            cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            and min_cg_support == AttentionCGSupport.NEVER
        ):
            msg = (
                f"CUDAGraphMode.{cudagraph_mode.name} is not supported "
                f"with {min_cg_backend_name} backend (support: "
                f"{min_cg_support})"
            )
            if self.compilation_config.mode == CompilationMode.VLLM_COMPILE and (
                self.compilation_config.splitting_ops_contain_attention()
                or self.compilation_config.use_inductor_graph_partition
            ):
                msg += (
                    "; setting cudagraph_mode=PIECEWISE because "
                    "attention is compiled piecewise"
                )
                cudagraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.PIECEWISE
                )
            else:
                msg += (
                    "; setting cudagraph_mode=NONE because "
                    "attention is not compiled piecewise"
                )
                cudagraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.NONE
                )
            logger.warning(msg)

        # check that if we are doing spec-decode + decode full-cudagraphs it is
        # supported
        if (
            cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            and self.uniform_decode_query_len > 1
            and min_cg_support.value < AttentionCGSupport.UNIFORM_BATCH.value
        ):
            msg = (
                f"CUDAGraphMode.{cudagraph_mode.name} is not supported"
                f" with spec-decode for attention backend "
                f"{min_cg_backend_name} (support: {min_cg_support})"
            )
            if self.compilation_config.splitting_ops_contain_attention():
                msg += "; setting cudagraph_mode=PIECEWISE"
                cudagraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.PIECEWISE
                )
            else:
                msg += "; setting cudagraph_mode=NONE"
                cudagraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.NONE
                )
            logger.warning(msg)

        # double check that we can support full cudagraph if they are requested
        # even after automatic downgrades
        if (
            cudagraph_mode.has_full_cudagraphs()
            and min_cg_support == AttentionCGSupport.NEVER
        ):
            raise ValueError(
                f"CUDAGraphMode.{cudagraph_mode.name} is not "
                f"supported with {min_cg_backend_name} backend ("
                f"support:{min_cg_support}) "
                "; please try cudagraph_mode=PIECEWISE, "
                "and make sure compilation mode is VLLM_COMPILE"
            )

        # if we have dedicated decode cudagraphs, and spec-decode is enabled,
        # we need to adjust the cudagraph sizes to be a multiple of the uniform
        # decode query length to avoid: https://github.com/vllm-project/vllm/issues/28207
        # temp-fix: https://github.com/vllm-project/vllm/issues/28207#issuecomment-3504004536
        # Will be removed in the near future when we have separate cudagraph capture
        # sizes for decode and mixed prefill-decode.
        if (
            cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            and cudagraph_mode.separate_routine()
            and self.uniform_decode_query_len > 1
        ):
            self.compilation_config.adjust_cudagraph_sizes_for_spec_decode(
                self.uniform_decode_query_len, self.parallel_config.tensor_parallel_size
            )
            capture_sizes = self.compilation_config.cudagraph_capture_sizes
            self.cudagraph_batch_sizes = (
                capture_sizes if capture_sizes is not None else []
            )

        # Trigger cudagraph dispatching keys initialization after
        # resolved cudagraph mode.
        self.compilation_config.cudagraph_mode = cudagraph_mode
        self.cudagraph_dispatcher.initialize_cudagraph_keys(
            cudagraph_mode, self.uniform_decode_query_len
        )

        # Initialize eagle's cudagraph dispatcher if using eagle spec decode.
        if self.speculative_config and self.speculative_config.use_eagle():
            assert isinstance(self.drafter, EagleProposer | DFlash2Proposer)
            self.drafter.initialize_cudagraph_keys(cudagraph_mode)

    def calculate_reorder_batch_threshold(self) -> None:
        """
        Choose the minimum reorder batch threshold from all attention groups.
        Backends should be able to support lower threshold then what they request
        just may have a performance penalty due to that backend treating decodes
        as prefills.
        """
        min_none_high = lambda a, b: a if b is None else b if a is None else min(a, b)

        reorder_batch_thresholds: list[int | None] = [
            group.get_metadata_builder().reorder_batch_threshold
            for group in self._attn_group_iterator()
        ]
        # If there are no attention groups (attention-free model) or no backend
        # reports a threshold, leave reordering disabled.
        if len(reorder_batch_thresholds) == 0:
            self.reorder_batch_threshold = None
            return
        self.reorder_batch_threshold = reduce(min_none_high, reorder_batch_thresholds)  # type: ignore[assignment]

    @staticmethod
    def select_common_block_size(
        kv_manager_block_size: int, attn_groups: list[AttentionGroup]
    ) -> int:
        """
        Select a block size that is supported by all backends and is a factor of
        kv_manager_block_size.

        If kv_manager_block_size is supported by all backends, return it directly.
        Otherwise, return the max supported size.

        Args:
            kv_manager_block_size: Block size of KV cache
            attn_groups: List of attention groups

        Returns:
            The selected block size

        Raises:
            ValueError: If no valid block size found
        """

        def block_size_is_supported(
            backends: list[type[AttentionBackend]], block_size: int
        ) -> bool:
            """
            Check if the block size is supported by all backends.
            """
            for backend in backends:
                is_supported = False
                for supported_size in backend.get_supported_kernel_block_sizes():
                    if isinstance(supported_size, int):
                        if block_size == supported_size:
                            is_supported = True
                    elif isinstance(supported_size, MultipleOf):
                        if block_size % supported_size.base == 0:
                            is_supported = True
                    else:
                        raise ValueError(f"Unknown supported size: {supported_size}")
                if not is_supported:
                    return False
            return True

        backends = [group.backend for group in attn_groups]

        # Case 1: if the block_size of kv cache manager is supported by all backends,
        # return it directly
        if block_size_is_supported(backends, kv_manager_block_size):
            return kv_manager_block_size

        # Case 2: otherwise, the block_size must be an `int`-format supported size of
        # at least one backend. Iterate over all `int`-format supported sizes in
        # descending order and return the first one that is supported by all backends.
        # Simple proof:
        # If the supported size b is in MultipleOf(x_i) format for all attention
        # backends i, and b a factor of kv_manager_block_size, then
        # kv_manager_block_size also satisfies MultipleOf(x_i) for all i. We will
        # return kv_manager_block_size in case 1.
        all_int_supported_sizes = set(
            supported_size
            for backend in backends
            for supported_size in backend.get_supported_kernel_block_sizes()
            if isinstance(supported_size, int)
        )

        for supported_size in sorted(all_int_supported_sizes, reverse=True):
            if kv_manager_block_size % supported_size != 0:
                continue
            if block_size_is_supported(backends, supported_size):
                return supported_size
        raise ValueError(f"No common block size for {kv_manager_block_size}. ")

    def may_reinitialize_input_batch(
        self, kv_cache_config: KVCacheConfig, kernel_block_sizes: list[int]
    ) -> None:
        """
        Re-initialize the input batch if the block sizes are different from
        `[self.cache_config.block_size]`. This usually happens when there
        are multiple KV cache groups.

        Args:
            kv_cache_config: The KV cache configuration.
            kernel_block_sizes: The kernel block sizes for each KV cache group.
        """
        block_sizes = [
            kv_cache_group.kv_cache_spec.block_size
            for kv_cache_group in kv_cache_config.kv_cache_groups
            if not isinstance(kv_cache_group.kv_cache_spec, EncoderOnlyAttentionSpec)
        ]
        max_num_blocks = []
        max_model_len = max(self.max_model_len, self.max_encoder_len)
        for i, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            if isinstance(kv_cache_group.kv_cache_spec, EncoderOnlyAttentionSpec):
                continue
            max_num_blocks_per_req = cdiv(
                max_model_len, block_sizes[i] * get_total_cp_world_size()
            )
            if isinstance(kv_cache_group.kv_cache_spec, MambaSpec):
                mamba_blocks_per_req = (
                    max_num_blocks_per_req
                    if self.cache_config.enable_prefix_caching
                    else 1
                ) + kv_cache_group.kv_cache_spec.num_speculative_blocks
                max_num_blocks_per_req = max(
                    max_num_blocks_per_req, mamba_blocks_per_req
                )
            max_num_blocks.append(max_num_blocks_per_req)

        if block_sizes != [self.cache_config.block_size] or kernel_block_sizes != [
            self.cache_config.block_size
        ]:
            assert self.cache_config.cpu_offload_gb == 0, (
                "Cannot re-initialize the input batch when CPU weight "
                "offloading is enabled. See https://github.com/vllm-project/vllm/pull/18298 "  # noqa: E501
                "for more details."
            )
            self.input_batch = InputBatch(
                max_num_reqs=self.max_num_reqs,
                max_model_len=max_model_len,
                max_num_batched_tokens=self.max_num_tokens,
                device=self.device,
                pin_memory=self.pin_memory,
                vocab_size=self.model_config.get_vocab_size(),
                block_sizes=block_sizes,
                kernel_block_sizes=kernel_block_sizes,
                max_num_blocks_per_req=max_num_blocks,
                is_spec_decode=bool(self.vllm_config.speculative_config),
                logitsprocs=self.input_batch.logitsprocs,
                logitsprocs_need_output_token_ids=self.input_batch.logitsprocs_need_output_token_ids,
                is_pooling_model=self.is_pooling_model,
            )

    def _allocate_kv_cache_tensors(
        self, kv_cache_config: KVCacheConfig
    ) -> dict[str, torch.Tensor]:
        """
        Initializes the KV cache buffer with the correct size. The buffer needs
        to be reshaped to the desired shape before being used by the models.

        Args:
            kv_cache_config: The KV cache config
        Returns:
            dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        kv_cache_raw_tensors: dict[str, torch.Tensor] = {}
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            tensor = torch.zeros(
                kv_cache_tensor.size, dtype=torch.int8, device=self.device
            )
            for layer_name in kv_cache_tensor.shared_by:
                kv_cache_raw_tensors[layer_name] = tensor

        layer_names = set()
        for group in kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                layer_names.add(layer_name)
        assert layer_names == set(kv_cache_raw_tensors.keys()), (
            "Some layers are not correctly initialized"
        )
        return kv_cache_raw_tensors

    def _attn_group_iterator(self) -> Iterator[AttentionGroup]:
        return itertools.chain.from_iterable(self.attn_groups)

    def _kv_cache_spec_attn_group_iterator(self) -> Iterator[AttentionGroup]:
        if not self.kv_cache_config.kv_cache_groups:
            return
        for attn_groups in self.attn_groups:
            yield from attn_groups

    def _prepare_kernel_block_sizes(self, kv_cache_config: KVCacheConfig) -> list[int]:
        """
        Generate kernel_block_sizes that matches each block_size.

        For attention backends that support virtual block splitting,
        use the supported block sizes from the backend.
        For other backends (like Mamba), use the same block size (no splitting).

        Args:
            kv_cache_config: The KV cache configuration.

        Returns:
            list[int]: List of kernel block sizes for each cache group.
        """
        kernel_block_sizes = []
        for kv_cache_gid, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            kv_cache_spec = kv_cache_group.kv_cache_spec
            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
                # All layers in the UniformTypeKVCacheSpecs have the same type,
                # Pick an arbitrary one to dispatch.
                kv_cache_spec = next(iter(kv_cache_spec.kv_cache_specs.values()))
            if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
                continue
            elif isinstance(kv_cache_spec, AttentionSpec):
                # This is an attention backend that supports virtual
                # block splitting. Get the supported block sizes from
                # all backends in the group.
                attn_groups = self.attn_groups[kv_cache_gid]
                kv_manager_block_size = kv_cache_group.kv_cache_spec.block_size
                selected_kernel_size = self.select_common_block_size(
                    kv_manager_block_size, attn_groups
                )
                kernel_block_sizes.append(selected_kernel_size)
            elif isinstance(kv_cache_spec, MambaSpec):
                # This is likely Mamba or other non-attention cache,
                # no splitting.
                kernel_block_sizes.append(kv_cache_spec.block_size)
            else:
                raise NotImplementedError(
                    f"unknown kv cache spec {kv_cache_group.kv_cache_spec}"
                )
        return kernel_block_sizes

    def _reshape_kv_cache_tensors(
        self,
        kv_cache_config: KVCacheConfig,
        kv_cache_raw_tensors: dict[str, torch.Tensor],
        kernel_block_sizes: list[int],
    ) -> dict[str, torch.Tensor]:
        """
        Reshape the KV cache tensors to the desired shape and dtype.

        Args:
            kv_cache_config: The KV cache config
            kv_cache_raw_tensors: The KV cache buffer of each layer, with
                correct size but uninitialized shape.
            kernel_block_sizes: The kernel block sizes for each KV cache group.
        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        kv_caches: dict[str, torch.Tensor] = {}
        has_attn, has_mamba = False, False
        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            attn_backend = group.backend
            if group.kv_cache_group_id == len(kernel_block_sizes):
                # There may be a last group for layers without kv cache.
                continue
            kernel_block_size = kernel_block_sizes[group.kv_cache_group_id]
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                raw_tensor = kv_cache_raw_tensors[layer_name]
                assert raw_tensor.numel() % kv_cache_spec.page_size_bytes == 0
                num_blocks = raw_tensor.numel() // kv_cache_spec.page_size_bytes
                if isinstance(kv_cache_spec, AttentionSpec):
                    has_attn = True
                    num_blocks_per_kv_block = (
                        kv_cache_spec.block_size // kernel_block_size
                    )
                    kernel_num_blocks = num_blocks * num_blocks_per_kv_block

                    kv_cache_shape = attn_backend.get_kv_cache_shape(
                        kernel_num_blocks,
                        kernel_block_size,
                        kv_cache_spec.num_kv_heads,
                        kv_cache_spec.head_size,
                        cache_dtype_str=self.cache_config.cache_dtype,
                    )
                    dtype = kv_cache_spec.dtype
                    try:
                        kv_cache_stride_order = attn_backend.get_kv_cache_stride_order()
                        assert len(kv_cache_stride_order) == len(kv_cache_shape)
                    except (AttributeError, NotImplementedError):
                        kv_cache_stride_order = tuple(range(len(kv_cache_shape)))
                    # The allocation respects the backend-defined stride order
                    # to ensure the semantic remains consistent for each
                    # backend. We first obtain the generic kv cache shape and
                    # then permute it according to the stride order which could
                    # result in a non-contiguous tensor.
                    kv_cache_shape = tuple(
                        kv_cache_shape[i] for i in kv_cache_stride_order
                    )
                    # Maintain original KV shape view.
                    inv_order = [
                        kv_cache_stride_order.index(i)
                        for i in range(len(kv_cache_stride_order))
                    ]
                    kv_caches[layer_name] = (
                        kv_cache_raw_tensors[layer_name]
                        .view(dtype)
                        .view(kv_cache_shape)
                        .permute(*inv_order)
                    )
                elif isinstance(kv_cache_spec, MambaSpec):
                    has_mamba = True
                    raw_tensor = kv_cache_raw_tensors[layer_name]
                    state_tensors = []
                    storage_offset_bytes = 0
                    for shape, dtype in zip(kv_cache_spec.shapes, kv_cache_spec.dtypes):
                        dtype_size = get_dtype_size(dtype)
                        num_element_per_page = (
                            kv_cache_spec.page_size_bytes // dtype_size
                        )
                        target_shape = (num_blocks, *shape)
                        stride = torch.empty(target_shape).stride()
                        target_stride = (num_element_per_page, *stride[1:])
                        assert storage_offset_bytes % dtype_size == 0
                        tensor = torch.as_strided(
                            raw_tensor.view(dtype),
                            size=target_shape,
                            stride=target_stride,
                            storage_offset=storage_offset_bytes // dtype_size,
                        )
                        state_tensors.append(tensor)
                        storage_offset_bytes += stride[0] * dtype_size

                    kv_caches[layer_name] = state_tensors
                else:
                    raise NotImplementedError

        if has_attn and has_mamba:
            self._update_hybrid_attention_mamba_layout(kv_caches)

        return kv_caches

    def _update_hybrid_attention_mamba_layout(
        self, kv_caches: dict[str, torch.Tensor]
    ) -> None:
        """
        Update the layout of attention layers from (2, num_blocks, ...) to
        (num_blocks, 2, ...).

        Args:
            kv_caches: The KV cache buffer of each layer.
        """

        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            for layer_name in group.layer_names:
                kv_cache = kv_caches[layer_name]
                if isinstance(kv_cache_spec, AttentionSpec) and kv_cache.shape[0] == 2:
                    assert kv_cache.shape[1] != 2, (
                        "Fail to determine whether the layout is "
                        "(2, num_blocks, ...) or (num_blocks, 2, ...) for "
                        f"a tensor of shape {kv_cache.shape}"
                    )
                    hidden_size = kv_cache.shape[2:].numel()
                    kv_cache.as_strided_(
                        size=kv_cache.shape,
                        stride=(hidden_size, 2 * hidden_size, *kv_cache.stride()[2:]),
                    )

    def initialize_kv_cache_tensors(
        self, kv_cache_config: KVCacheConfig, kernel_block_sizes: list[int]
    ) -> dict[str, torch.Tensor]:
        """
        Initialize the memory buffer for KV cache.

        Args:
            kv_cache_config: The KV cache config
            kernel_block_sizes: The kernel block sizes for each KV cache group.

        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """

        # Try creating KV caches optimized for kv-connector transfers
        cache_dtype = self.cache_config.cache_dtype
        if self.use_uniform_kv_cache(self.attn_groups, cache_dtype):
            kv_caches, cross_layers_kv_cache, attn_backend = (
                self.allocate_uniform_kv_caches(
                    kv_cache_config,
                    self.attn_groups,
                    cache_dtype,
                    self.device,
                    kernel_block_sizes,
                )
            )
            self.cross_layers_kv_cache = cross_layers_kv_cache
            self.cross_layers_attn_backend = attn_backend
        else:
            # Fallback to the general case
            # Initialize the memory buffer for KV cache
            kv_cache_raw_tensors = self._allocate_kv_cache_tensors(kv_cache_config)

            # Change the memory buffer to the desired shape
            kv_caches = self._reshape_kv_cache_tensors(
                kv_cache_config, kv_cache_raw_tensors, kernel_block_sizes
            )

        # Set up cross-layer KV cache sharing
        for layer_name, target_layer_name in self.shared_kv_cache_layers.items():
            logger.debug("%s reuses KV cache of %s", layer_name, target_layer_name)
            kv_caches[layer_name] = kv_caches[target_layer_name]

        num_attn_module = (
            2 if self.model_config.hf_config.model_type == "longcat_flash" else 1
        )
        bind_kv_cache(
            kv_caches,
            self.compilation_config.static_forward_context,
            self.kv_caches,
            num_attn_module,
        )
        return kv_caches

    def maybe_add_kv_sharing_layers_to_kv_cache_groups(
        self, kv_cache_config: KVCacheConfig
    ) -> None:
        """
        Add layers that re-use KV cache to KV cache group of its target layer.
        Mapping of KV cache tensors happens in `initialize_kv_cache_tensors()`
        """
        if not self.shared_kv_cache_layers:
            # No cross-layer KV sharing, return
            return

        add_kv_sharing_layers_to_kv_cache_groups(
            self.shared_kv_cache_layers,
            kv_cache_config.kv_cache_groups,
            self.runner_only_attn_layers,
        )

        if self.cache_config.kv_sharing_fast_prefill:
            # In You Only Cache Once (https://arxiv.org/abs/2405.05254) or other
            # similar KV sharing setups, only the layers that generate KV caches
            # are involved in the prefill phase, enabling prefill to early exit.
            attn_layers = get_layers_from_vllm_config(self.vllm_config, Attention)
            for layer_name in reversed(attn_layers):
                if layer_name in self.shared_kv_cache_layers:
                    self.kv_sharing_fast_prefill_eligible_layers.add(layer_name)
                else:
                    break

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        kv_cache_config = deepcopy(kv_cache_config)
        self.kv_cache_config = kv_cache_config
        self.may_add_encoder_only_layers_to_kv_cache_config()
        self.maybe_add_kv_sharing_layers_to_kv_cache_groups(kv_cache_config)
        self.initialize_attn_backend(kv_cache_config)
        # The kernel block size for all KV cache groups. For example, if
        # kv_cache_manager uses block_size 256 for a given group, but the attention
        # backends for that group only supports block_size 64, we will return
        # kernel_block_size 64 and split the 256-token-block to 4 blocks with 64
        # tokens each.
        kernel_block_sizes = self._prepare_kernel_block_sizes(kv_cache_config)

        # create metadata builders
        self.initialize_metadata_builders(kv_cache_config, kernel_block_sizes)

        # Reinitialize need to after initialize_attn_backend
        self.may_reinitialize_input_batch(kv_cache_config, kernel_block_sizes)
        kv_caches = self.initialize_kv_cache_tensors(
            kv_cache_config, kernel_block_sizes
        )

        if self.speculative_config and (
            self.speculative_config.use_eagle()
            or self.speculative_config.uses_draft_model()
        ):
            assert isinstance(self.drafter, EagleProposer | DraftModelProposer | DFlash2Proposer)
            # validate all draft model layers belong to the same kv cache
            # group
            self.drafter.validate_same_kv_cache_group(kv_cache_config)

        if has_kv_transfer_group():
            kv_transfer_group = get_kv_transfer_group()
            if self.cross_layers_kv_cache is not None:
                assert self.cross_layers_attn_backend is not None
                kv_transfer_group.register_cross_layers_kv_cache(
                    self.cross_layers_kv_cache, self.cross_layers_attn_backend
                )
            else:
                kv_transfer_group.register_kv_caches(kv_caches)
            kv_transfer_group.set_host_xfer_buffer_ops(copy_kv_blocks)

        if self.model_config.enable_return_routed_experts:
            self.init_routed_experts_capturer()

    def init_routed_experts_capturer(self):
        logger.info(
            "Initializing routed experts capturer, enable_return_routed_experts: %s",
            self.model_config.enable_return_routed_experts,
        )
        routed_experts_capturer = RoutedExpertsCapturer.create()
        block_size = self.cache_config.block_size
        self.max_num_kv_tokens = (
            self.kv_cache_config.num_blocks // len(self.kv_cache_config.kv_cache_groups)
            + 1
        ) * block_size
        routed_experts_capturer.init_buffer(
            max_num_batched_tokens=self.scheduler_config.max_num_batched_tokens,
            max_num_kv_tokens=self.max_num_kv_tokens,
            vllm_config=self.vllm_config,
        )
        self._bind_routed_experts_capturer(routed_experts_capturer)

    def _bind_routed_experts_capturer(self, capturer: RoutedExpertsCapturer) -> None:
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE
        from vllm.model_executor.layers.fused_moe.router.base_router import (
            BaseRouter,
        )

        for module in self.compilation_config.static_forward_context.values():
            if isinstance(module, FusedMoE) and isinstance(module.router, BaseRouter):
                layer_id = module.layer_id

                def _capture_fn(topk_ids, _layer_id=layer_id, _capturer=capturer):
                    _capturer.capture(_layer_id, topk_ids)

                module.router.set_capture_fn(_capture_fn)

    def may_add_encoder_only_layers_to_kv_cache_config(self) -> None:
        """
        Add encoder-only layers to the KV cache config.
        """
        block_size = self.vllm_config.cache_config.block_size
        encoder_only_attn_specs: dict[AttentionSpec, list[str]] = defaultdict(list)
        attn_layers = get_layers_from_vllm_config(self.vllm_config, Attention)
        for layer_name, attn_module in attn_layers.items():
            if attn_module.attn_type == AttentionType.ENCODER_ONLY:
                attn_spec: AttentionSpec = EncoderOnlyAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=attn_module.num_kv_heads,
                    head_size=attn_module.head_size,
                    dtype=self.kv_cache_dtype,
                )
                encoder_only_attn_specs[attn_spec].append(layer_name)
                self.runner_only_attn_layers.add(layer_name)
        if len(encoder_only_attn_specs) > 0:
            assert len(encoder_only_attn_specs) == 1, (
                "Only support one encoder-only attention spec now"
            )
            spec, layer_names = encoder_only_attn_specs.popitem()
            self.kv_cache_config.kv_cache_groups.append(
                KVCacheGroupSpec(layer_names=layer_names, kv_cache_spec=spec)
            )

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """
        if has_ec_transfer() and get_ec_transfer().is_producer:
            return {}
        kv_cache_spec: dict[str, KVCacheSpec] = {}
        layer_type = cast(type[Any], AttentionLayerBase)
        attn_layers = get_layers_from_vllm_config(self.vllm_config, layer_type)
        for layer_name, attn_module in attn_layers.items():
            if isinstance(attn_module, Attention) and (
                kv_tgt_layer := attn_module.kv_sharing_target_layer_name
            ):
                # The layer doesn't need its own KV cache and will use that of
                # the target layer. We skip creating a KVCacheSpec for it, so
                # that KV cache management logic will act as this layer does
                # not exist, and doesn't allocate KV cache for the layer. This
                # enables the memory saving of cross-layer kv sharing, allowing
                # a given amount of memory to accommodate longer context lengths
                # or enable more requests to be processed simultaneously.
                self.shared_kv_cache_layers[layer_name] = kv_tgt_layer
                continue
            # Skip modules that don't need KV cache (eg encoder-only attention)
            if spec := attn_module.get_kv_cache_spec(self.vllm_config):
                kv_cache_spec[layer_name] = spec

        return kv_cache_spec

    def _to_list(self, sampled_token_ids: torch.Tensor) -> list[list[int]]:
        # This is a short term mitigation for issue mentioned in
        # https://github.com/vllm-project/vllm/issues/22754.
        # `tolist` would trigger a cuda wise stream sync, which
        # would block other copy ops from other cuda streams.
        # A cuda event sync would avoid such a situation. Since
        # this is in the critical path of every single model
        # forward loop, this has caused perf issue for a disagg
        # setup.
        pinned = self.sampled_token_ids_pinned_cpu[: sampled_token_ids.shape[0]]
        pinned.copy_(sampled_token_ids, non_blocking=True)
        self.transfer_event.record()
        self.transfer_event.synchronize()
        return pinned.tolist()

    def get_encoder_timing_stats(self) -> dict[str, dict[str, float | int]]:
        """
        Get encoder timing stats for all requests and clear the registry.

        Returns:
            Dictionary mapping request_id to stats dict.
        """
        with self._encoder_timing_lock:
            stats = {
                req_id: stats_obj.to_dict()
                for req_id, stats_obj in self.encoder_timing_registry.items()
            }
            self.encoder_timing_registry.clear()
            return stats

    @contextmanager
    def timed_encoder_operation(
        self,
        should_time: bool,
        group_lora_refs: list[tuple[str, Any]],
        current_item_idx: int,
        num_items: int,
    ):
        """
        Context manager to time encoder forward operations.

        Args:
            should_time: Whether timing is enabled
            group_lora_refs: Full list of (request_id, pos_info) tuples
            current_item_idx: Starting index for this group
            num_items: Number of items in this group
        """
        if not should_time:
            yield
            return

        group_refs = group_lora_refs[current_item_idx : current_item_idx + num_items]
        group_request_ids = {req_id for req_id, _ in group_refs}

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        try:
            yield
        finally:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time

            per_request_time = elapsed / max(len(group_request_ids), 1)

            with self._encoder_timing_lock:
                for req_id in group_request_ids:
                    if req_id not in self.encoder_timing_registry:
                        self.encoder_timing_registry[req_id] = EncoderTimingStats()

                    stats = self.encoder_timing_registry[req_id]
                    stats.encoder_forward_time += per_request_time
                    stats.num_encoder_calls += 1


@dataclass
class EncoderTimingStats:
    """Per-request timing statistics for encoder forward pass."""

    encoder_forward_time: float = 0.0
    """Time spent in vision encoder forward pass (seconds)."""

    num_encoder_calls: int = 0
    """Number of times encoder was called for this request."""

    def to_dict(self) -> dict[str, float | int]:
        return {
            "encoder_forward_time": self.encoder_forward_time,
            "num_encoder_calls": self.num_encoder_calls,
        }
