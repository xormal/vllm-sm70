# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ast
from dataclasses import replace
from importlib.util import find_spec
from typing import cast

import numpy as np
import torch
import os as _os
import torch as _torch
import torch.nn as nn

from vllm.config import (
    CUDAGraphMode,
    VllmConfig,
    get_layers_from_vllm_config,
)
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models import supports_multimodal
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.attention.backend import (
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.backends.tree_attn import (
    TreeAttentionMetadata,
    TreeAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import _SAMPLING_EPS
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.utils import (
    eagle_prepare_inputs_padded_kernel,
    eagle_prepare_next_token_padded_kernel,
)
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.dp_utils import coordinate_batch_across_dp
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

logger = init_logger(__name__)

# Проба потолка дерева: печатать раз в N шагов; 0 = ветка мертва (см. §234).
_ПРОБА_ТОП2 = int(__import__("os").environ.get("FA2SM70_MTP_TOP2", "0"))
# Сбор скрытых состояний черновика для проверки двухступенчатого аргмакса (0 = ветка мертва).
_СБОР_СКРЫТЫХ = int(__import__("os").environ.get("FA2SM70_MTP_DUMP_H", "0"))
# Двухступенчатый аргмакс словаря черновика: 0 = ветка мертва, боевой путь не меняется.
_И4 = int(__import__("os").environ.get("FA2SM70_MTP_I4", "0"))
_И4_СПИСОК = int(__import__("os").environ.get("FA2SM70_MTP_I4_N", "16"))
_И4_MMAX = int(__import__("os").environ.get("FA2SM70_MTP_I4_MMAX", "2"))
# [СРЕЗ ИЗМЕРЕНИЙ У ПЕРВОЙ СТУПЕНИ -- 09.09, записка 25 §253]
# Полный int4-словарь (303 МиБ) не уживается с 262 144 токенами -- проверено семью путями.
# Но первой ступени не нужен точный счёт: ей нужно, чтобы верная строка ПОПАЛА В СПИСОК, а
# вторая ступень пересчитает список точно по полному весу. Значит можно упаковать СРЕЗ первых
# r измерений. Ядро i4 берёт K во время исполнения -- новой сборки не нужно.
# Полнота на 700 НАСТОЯЩИХ скрытых состояниях черновика (доля ранга, 124160 строк):
#   r=2048 (121 МиБ) top64= 97.6 %      r=3072 (182 МиБ) top64=100.0 %
#   r=2560 (152 МиБ) top64= 99.9 %      r=4096 (243 МиБ) top16=100.0 %
# Случайная проекция той же памяти ХУЖЕ среза (r=2048: 82.9 против 97.6) -- энергия скрытого
# состояния лежит в исходных координатах, а не размазана.
# 0 = без среза (прежнее поведение, весь K).
_И4_СРЕЗ = int(__import__("os").environ.get("FA2SM70_MTP_I4_R", "0"))
# [СВОЙ ГРАФ ТЕЛА ЧЕРНОВИКА -- 09.09, записка 25 §257]
# Проход черновика идёт PIECEWISE (vLLM даёт ему только его, eagle.py:289). Замер: карта
# 7.06 мс при ядрах ~4.3, выдача хозяином 6.85 -- то есть щель ~2.2 мс и она вся в ленивой
# выдаче. Здесь тело головы захватывается в СВОЙ граф по образцу dflash2 (§212): входы уже
# живут в постоянных буферах, у меты держатся четыре тензора и копируются свежие значения.
# БЕЗОПАСНОСТЬ: любой отказ хоронит ключ навсегда и путь возвращается к прежнему; а ошибка
# черновика в принципе не может испортить выход -- цель проверяет каждый токен.
_ГР_ЧЕРН = __import__("os").environ.get("FA2SM70_MTP_GRAPH", "0") == "1"
_ГР_ЧЕРН_ФАЙЛ = __import__("os").environ.get("FA2SM70_MTP_GRAPH_FILE", "")
# Длина контекста входит в ключ ВЕДРОМ: выбор пути внутри бэкенда зависит от max_seq_len,
# и запекать его без ведра значило бы переиграть чужой путь на выросшем контексте.
_ГР_ЧЕРН_ВЕДРО = int(__import__("os").environ.get("FA2SM70_MTP_GRAPH_BUCKET", "16384"))
# Подняты из тела `_i4_argmax`: там они выполнялись на КАЖДОМ вызове (три за шаг).
from vllm.distributed import (
    get_tensor_model_parallel_world_size as _tp_size,
)
from vllm.distributed.communication_op import (
    tensor_model_parallel_all_gather,
)
# Тумблер местного аргмакса черновика (§250). Пустой путь = ветка мертва.
_МЕСТ_ФАЙЛ = __import__("os").environ.get("FA2SM70_MTP_LOCALARGMAX_FILE", "")
# Окна на событиях зовутся ШЕСТЬ раз за шаг -- признак читается один раз, при загрузке.
_ФАЗА_ДЕВ = int(__import__("os").environ.get("FA2SM70_STEP_PHASE_DEV", "0"))
_ФАЗА_N = int(__import__("os").environ.get("FA2SM70_STEP_PHASE", "0"))
import time as _t_eagle


# [ЗАКРЕПЛЁННАЯ ПЛОЩАДКА ДЛЯ H2D -- 11.09, записка 25 §291]
# Трасса 11.09 (стенд, декод): 68 синхронных копий хозяин->карта на ~17 шагов, то есть 4 на шаг.
# Причина: `torch.tensor(список, device=cuda)` и `.to(device, non_blocking=True)` по ВЫЧИСЛЕННОМУ
# (а значит НЕ закреплённому) CPU-тензору. На непиннованной памяти `non_blocking` молча
# игнорируется, копия идёт синхронно и РВЁТ КОНВЕЙЕР: карта доезжает до конца очереди, хозяин
# продолжает только после. Собственно копия стоит микросекунды, платит простой.
# Лечение: постоянные пары (закреплённый CPU-буфер, буфер карты) на каждое поле, копия
# non_blocking в тот же буфер. Адреса стабильны -- значит законно и внутри графа.
# Умолчание 0: venv общий с боевым.
_ЗАКРЕП = _os.environ.get("FA2SM70_PIN_HOST", "0") == "1"
_ЗАКРЕП_БУФ: dict = {}


def _закреп(имя, знач, устр):
    """CPU-тензор -> карта через постоянную закреплённую площадку. Возвращает вид буфера карты.

    `знач` -- одномерный CPU-тензор или numpy-массив. При отказе (нет pin_memory) возвращает
    обычный перенос: ускорение не обязано быть условием работы.
    """
    try:
        import numpy as _np
        if isinstance(знач, _np.ndarray):
            знач = _torch.from_numpy(знач)
        n = знач.numel()
        ключ = (имя, знач.dtype, str(устр))
        пара = _ЗАКРЕП_БУФ.get(ключ)
        if пара is None or пара[0].numel() < n:
            м = max(n, 128)
            cpu = _torch.empty(м, dtype=знач.dtype, pin_memory=True)
            gpu = _torch.empty(м, dtype=знач.dtype, device=устр)
            пара = (cpu, gpu)
            _ЗАКРЕП_БУФ[ключ] = пара
        cpu, gpu = пара
        cpu[:n].copy_(знач)                      # хозяин -> хозяин, без синхронизации
        gpu[:n].copy_(cpu[:n], non_blocking=True)  # закреплённая -> карта, асинхронно
        return gpu[:n]
    except Exception:
        return знач.to(устр)


# [ГЕЙТ ДЛЯ БУДУЩЕГО ОБУЧЕНИЯ ГОЛОВЫ MTP -- 11.09]
# Прошлые четыре захода обучения черновика сгорели на ПРИБОРЕ: он не воспроизводил боевое число,
# и дни уходили на оптимизацию не той функции (записка 25, урок «прибор врёт про разность»).
# Поэтому порядок здесь обратный: сперва гейт, потом обучение. Гейт возможен только если снять
# ТРОЙКУ за один шаг: (признаки цели, токены входа, токен, который движок выдал САМ). Тогда
# офлайновый проход головы обязан совпасть с движком ПОБАЙТОВО на тех же буферах -- и лишь после
# этого его разность между моделями можно считать осмысленной.
# Умолчание пусто: venv общий с боевым, и в выключенном виде это одна проверка строки на вызов.
_ГЕЙТ_ДАМП = _os.environ.get("FA2SM70_MTP_GATE_DUMP", "")
_ГЕЙТ_N = [0]
_ГЕЙТ_МАКС = int(_os.environ.get("FA2SM70_MTP_GATE_MAX", "200"))


def _снять_gejt(признаки, входные, токены):
    """Кладёт (признаки, входные токены, выданные черновиком токены) одного шага на диск."""
    if _ГЕЙТ_N[0] >= _ГЕЙТ_МАКС:
        return
    try:
        _os.makedirs(_ГЕЙТ_ДАМП, exist_ok=True)
        n = _ГЕЙТ_N[0]; _ГЕЙТ_N[0] = n + 1
        _torch.save({"h": признаки.detach().to(_torch.float16).cpu(),
                     "vh": входные.detach().to(_torch.int32).cpu() if входные is not None else None,
                     "dr": токены.detach().to(_torch.int32).cpu()},
                    _os.path.join(_ГЕЙТ_ДАМП, f"shag_{_os.getpid()}_{n:05d}.pt"))
    except Exception:
        pass          # гейт -- прибор, а не условие работы


def _хф(runner, имя, т0):
    """ХОЗЯЙСКАЯ фаза внутри прохода черновика (записка 25 §256). Карта в проходе черновика
    занята 7.35 мс, а хозяин выдаёт 7.12 -- то есть выдача и есть узкое место, и надо знать,
    что именно в ней питон, а что запуски ядер."""
    if _ФАЗА_N and runner is not None:
        ф = getattr(runner, "_фаза_шага", None)
        if ф is not None:
            ф("  чер/х: " + имя, _t_eagle.perf_counter() - т0)

PADDING_SLOT_ID = -1
# [18.10] Возврат заглушек в хвост дополненной меты черновика (см. _дополнить_мету). Умолчание 0.
_PAD_META_FIX = __import__("os").environ.get("FA2SM70_PAD_META_FIX", "0") == "1"
_PAD_META_PROBE = __import__("os").environ.get("FA2SM70_PAD_META_PROBE", "0") == "1"
if _PAD_META_FIX:
    print(f"[fa2_sm70 18.10] PAD_META_FIX=1 ({__file__})", flush=True)


class SpecDecodeBaseProposer:
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        pass_hidden_states_to_model: bool,
        runner=None,
    ):
        self.vllm_config = vllm_config
        assert vllm_config.speculative_config is not None
        self.speculative_config = vllm_config.speculative_config
        self.draft_model_config = self.speculative_config.draft_model_config
        self.method = self.speculative_config.method
        self.pass_hidden_states_to_model = pass_hidden_states_to_model

        self.runner = runner
        self.device = device
        self.dtype = vllm_config.model_config.dtype
        self.max_model_len = vllm_config.model_config.max_model_len
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank
        self.num_speculative_tokens = self.speculative_config.num_speculative_tokens
        self.use_local_argmax_reduction = (
            self.speculative_config.use_local_argmax_reduction
        )
        # The drafter can get longer sequences than the target model.
        max_batch_size = vllm_config.scheduler_config.max_num_seqs
        self.max_num_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens + max_batch_size
        )
        self.token_arange_np = np.arange(self.max_num_tokens)
        # We need to get the hidden size from the draft model config because
        # the draft model's hidden size can be different from the target model's
        # hidden size (e.g., Llama 3.3 70B).
        self.hidden_size = self.draft_model_config.get_hidden_size()
        self.inputs_embeds_size = self.draft_model_config.get_inputs_embeds_size()

        # Multi-modal data support
        self.mm_registry = MULTIMODAL_REGISTRY
        self.supports_mm_inputs = self.mm_registry.supports_multimodal_inputs(
            vllm_config.model_config
        )

        self.attn_metadata_builder: AttentionMetadataBuilder | None = None
        self.draft_indexer_metadata_builder: AttentionMetadataBuilder | None = None
        self.spec_decode_moe_layers: list[str] | None = None
        self.attn_layer_names: list[str] = []
        self.indexer_layer_names: list[str] = []
        self.eagle3_use_aux_hidden_state: bool = (
            self._get_eagle3_use_aux_hidden_state_from_config()
        )

        self.compilation_config = self.vllm_config.compilation_config

        # Cudagraph dispatcher for PIECEWISE-only dispatching in eagle.
        # Keys are initialized later via initialize_cudagraph_keys() called from
        # gpu_model_runner._check_and_update_cudagraph_mode after
        # adjust_cudagraph_sizes_for_spec_decode is called.
        self.cudagraph_dispatcher = CudagraphDispatcher(self.vllm_config)

        # persistent buffers for cuda graph
        self.input_ids = torch.zeros(
            self.max_num_tokens, dtype=torch.int32, device=device
        )
        # Use draft model's M-RoPE setting, not target model's
        # Draft models may be text-only even if target is multimodal
        self.uses_mrope = self.draft_model_config.uses_mrope
        self.uses_xdrope_dim = self.vllm_config.model_config.uses_xdrope_dim
        self.draft_uses_xdrope_dim = self.draft_model_config.uses_xdrope_dim
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
            self.mrope_positions = torch.zeros(
                (3, self.max_num_tokens + 1), dtype=torch.int64, device=device
            )
        elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
            self.xdrope_positions = torch.zeros(
                (self.uses_xdrope_dim, self.max_num_tokens + 1),
                dtype=torch.int64,
                device=device,
            )
        else:
            # RoPE need (max_num_tokens,)
            self.positions = torch.zeros(
                self.max_num_tokens, dtype=torch.int64, device=device
            )
        self.hidden_states = torch.zeros(
            (self.max_num_tokens, self.hidden_size), dtype=self.dtype, device=device
        )

        # We need +1 here because the arange is used to set query_start_loc,
        # which has one more element than batch_size.
        max_num_slots_for_arange = max(max_batch_size + 1, self.max_num_tokens)
        self.arange = torch.arange(
            max_num_slots_for_arange, device=device, dtype=torch.int32
        )

        self.inputs_embeds = torch.zeros(
            (self.max_num_tokens, self.inputs_embeds_size),
            dtype=self.dtype,
            device=device,
        )

        self.backup_next_token_ids = CpuGpuBuffer(
            max_batch_size,
            dtype=torch.int32,
            pin_memory=is_pin_memory_available(),
            device=device,
            with_numpy=True,
        )

        self._slot_mapping_buffer = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=device
        )

        # Determine allowed attention backends once during initialization.
        self.allowed_attn_types: tuple | None = None
        if current_platform.is_rocm():
            from vllm.v1.attention.backends.rocm_attn import RocmAttentionMetadata

            rocm_types = [
                TritonAttentionMetadata,
                RocmAttentionMetadata,
            ]
            # ROCM_AITER_FA is an optional backend
            from vllm._aiter_ops import rocm_aiter_ops

            if rocm_aiter_ops.is_enabled() and find_spec(
                AttentionBackendEnum.ROCM_AITER_FA.get_path(include_classname=False)
            ):
                from vllm.v1.attention.backends.rocm_aiter_fa import (
                    AiterFlashAttentionMetadata,
                )

                rocm_types.append(AiterFlashAttentionMetadata)

            # TRITON_MLA backend support for MLA models (e.g., DeepSeek)
            from vllm.model_executor.layers.attention.mla_attention import (
                MLACommonMetadata,
            )

            rocm_types.append(MLACommonMetadata)

            # FlexAttention backend support
            from vllm.v1.attention.backends.flex_attention import FlexAttentionMetadata

            rocm_types.append(FlexAttentionMetadata)

            self.allowed_attn_types = tuple(rocm_types)

        # Parse the speculative token tree.
        spec_token_tree = self.speculative_config.speculative_token_tree
        assert spec_token_tree is not None
        self.tree_choices: list[tuple[int, ...]] = ast.literal_eval(spec_token_tree)
        tree_depth = len(self.tree_choices[-1])
        # Precompute per-level properties of the tree.
        num_drafts_per_level = [0] * tree_depth
        for node in self.tree_choices:
            num_drafts_per_level[len(node) - 1] += 1
        self.cu_drafts_per_level = [num_drafts_per_level[0]]
        self.child_drafts_per_level = [num_drafts_per_level[0]]
        for level in range(1, tree_depth):
            self.cu_drafts_per_level.append(
                self.cu_drafts_per_level[-1] + num_drafts_per_level[level]
            )
            self.child_drafts_per_level.append(
                num_drafts_per_level[level] // num_drafts_per_level[level - 1]
            )
        # Precompute draft position offsets in flattened tree.
        self.tree_draft_pos_offsets = torch.arange(
            1, len(self.tree_choices) + 1, device=device, dtype=torch.int32
        ).repeat(max_batch_size, 1)

    def _get_positions(self, num_tokens: int):
        if self.uses_mrope:
            return self.mrope_positions[:, :num_tokens]
        if self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
            return self.xdrope_positions[:, :num_tokens]
        return self.positions[:num_tokens]

    def _set_positions(self, num_tokens: int, positions: torch.Tensor):
        if self.uses_mrope:
            self.mrope_positions[:, :num_tokens] = positions
        elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
            self.xdrope_positions[:, :num_tokens] = positions
        else:
            # Convert M-RoPE positions if target model uses M-RoPE
            # but draft doesn't, For text inputs, all M-RoPE
            # dimensions are identical
            if self.vllm_config.model_config.uses_mrope:
                positions = positions[0]
            self.positions[:num_tokens] = positions

    def _get_slot_mapping(
        self,
        num_tokens: int,
        slot_mapping: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return slot_mapping dict for EAGLE layers.

        If slot_mapping is provided, copies it into the buffer first.
        """
        if slot_mapping is not None:
            num_actual = slot_mapping.shape[0]
            self._slot_mapping_buffer[:num_actual].copy_(slot_mapping)
            if num_tokens > num_actual:
                self._slot_mapping_buffer[num_actual:num_tokens].fill_(PADDING_SLOT_ID)

        view = self._slot_mapping_buffer[:num_tokens]
        return {name: view for name in self.attn_layer_names + self.indexer_layer_names}

    def initialize_cudagraph_keys(self, cudagraph_mode: CUDAGraphMode) -> None:
        """Initialize cudagraph dispatcher keys for eagle.

        Eagle only supports PIECEWISE cudagraphs (via mixed_mode).
        This should be called after adjust_cudagraph_sizes_for_spec_decode.
        """
        if (
            not self.speculative_config.enforce_eager
            and cudagraph_mode.mixed_mode()
            in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL]
        ):
            eagle_cudagraph_mode = CUDAGraphMode.PIECEWISE
        else:
            eagle_cudagraph_mode = CUDAGraphMode.NONE

        self.cudagraph_dispatcher.initialize_cudagraph_keys(eagle_cudagraph_mode)

    # [ДВУХСТУПЕНЧАТЫЙ АРГМАКС СЛОВАРЯ ЧЕРНОВИКА -- 09.09, записка 25 §238-§240]
    # Черновик MTP идёт цепью и на КАЖДОМ шаге проецирует скрытое в весь словарь: 1.27 ГБ на
    # ранг, 1.93 мс. Но ему нужен не логит, а АРГМАКС. Здесь первый этап считается по int4
    # (303 МиБ), берётся top-16, и он досчитывается ТОЧНО по fp16-строкам. Совпадение с точным
    # аргмаксом 99.86 % на 700 настоящих состояниях; расхождение безвредно -- цель проверяет
    # каждый черновой токен, цена в приёмке доли процента.
    # ПУТЬ ВЫБИРАЕТСЯ ПО M: наше ядро скалярное и выигрывает при M<=2 (x3.00 при M=1),
    # а при M=4 чужой torch идёт тензорным и быстрее -- отдаём ему.
    def _i4_подготовить(self):
        if getattr(self, "_i4_пак", None) is not None:
            return self._i4_пак
        if torch.cuda.is_current_stream_capturing():
            return None            # ленивое выделение внутри ЗАХВАТА -- закон, ловилось трижды
        try:
            import fa2_sm70._ext as _E
            W = self.model.lm_head.weight
            if W.dtype != torch.float16 or W.dim() != 2 or W.shape[1] % 8:
                raise ValueError(f"голова не годится: {tuple(W.shape)} {W.dtype}")
            _r = W.shape[1] if not _И4_СРЕЗ else min(_И4_СРЕЗ, W.shape[1])
            if _r % 8:
                raise ValueError(f"срез r={_r} не кратен 8")
            _ext = _E.i4_ext()
            # УПАКОВКА КУСКАМИ СТРОК. Сама таблица мала (182 МиБ при r=3072), но `W[:, :r]`
            # одним куском требует 728 МиБ ВРЕМЕННО -- и именно на этом подготовка отказала.
            # Кусок в 8192 строки держит пик на ~48 МиБ. Раскладка построчная, поэтому куски
            # складываются в готовый буфер без склейки.
            _куc = int(__import__("os").environ.get("FA2SM70_MTP_I4_CHUNK", "8192"))
            if _r == W.shape[1] and _куc >= W.shape[0]:
                Q, S = _ext.pack(W.contiguous())
            else:
                Q = S = None
                for _i in range(0, W.shape[0], _куc):
                    _q, _s = _ext.pack(W[_i:_i + _куc, :_r].contiguous())
                    if Q is None:
                        Q = torch.empty((W.shape[0],) + tuple(_q.shape[1:]),
                                        dtype=_q.dtype, device=_q.device)
                        S = torch.empty((W.shape[0],) + tuple(_s.shape[1:]),
                                        dtype=_s.dtype, device=_s.device)
                    Q[_i:_i + _q.shape[0]] = _q
                    S[_i:_i + _s.shape[0]] = _s
                    del _q, _s
                torch.cuda.empty_cache()
            self._i4_пак = (Q, S, _ext, _r)
            print(f"[MTP i4] словарь черновика упакован: {tuple(Q.shape)} "
                  f"({Q.numel() / 2**20:.0f} МиБ против {W.numel()*2/2**20:.0f}, "
                  f"срез r={_r} из {W.shape[1]})",
                  file=__import__("sys").stderr, flush=True)
        except Exception as _е:
            self._i4_пак = None
            globals()["_И4"] = 0
            print(f"[MTP i4] отказ подготовки: {_е} -- иду прежним путём",
                  file=__import__("sys").stderr, flush=True)
        return self._i4_пак

    def _i4_argmax(self, hidden_states: torch.Tensor):
        пак = self._i4_подготовить()
        if пак is None:
            return None
        Q, S, ext, _r = пак
        h = hidden_states.to(torch.float16)
        W = self.model.lm_head.weight
        # ПЕРВАЯ ступень -- по срезу (дешёвое чтение), ВТОРАЯ -- по полному весу.
        _hs = h[:, :_r].contiguous() if _r != h.shape[1] else h.contiguous()
        топ = ext.scores(_hs, Q, S).topk(_И4_СПИСОК, dim=-1).indices        # [M, N]
        h = h.contiguous()
        # [ПИТОН В ПУТИ -- 09.09, записка 25 §256] Здесь стоял `torch.einsum("mk,mnk->mn")`.
        # einsum РАЗБИРАЕТ формулу строкой при каждом вызове, а вызовов три за шаг; замер
        # хозяйской фазы дал 1.35 мс выдачи против 1.61 мс работы карты на те же три вызова.
        # `bmm` даёт ровно то же произведение без разбора формулы.
        оц = torch.bmm(W[топ].float(), h.float().unsqueeze(-1)).squeeze(-1)  # точная досчитка
        лок_зн, лок_поз = оц.max(dim=-1)
        лок_идx = топ.gather(1, лок_поз.unsqueeze(1)).squeeze(1)
        нач = self.model.lm_head.shard_indices.org_vocab_start_index
        глоб = лок_идx + нач
        tp = _tp_size()
        if tp == 1:
            return глоб.to(torch.int64)
        пара = torch.stack([лок_зн.float(), глоб.float()], dim=-1)
        соб = tensor_model_parallel_all_gather(пара, dim=-1).view(h.shape[0], tp, 2)
        лучш = соб[:, :, 0].argmax(dim=-1, keepdim=True)
        return соб[:, :, 1].gather(dim=-1, index=лучш).squeeze(-1).to(torch.int64)

    # [РАЗБОР ПРОХОДА ЧЕРНОВИКА ПО КАРТЕ -- 09.09, записка 25 §249]
    # Карта в проходе черновика занята 10.3 мс, хозяин выдаёт за 8.4 -- значит проход
    # ограничен РАБОТОЙ, а не запусками, и полный граф черновика приза не имеет. Остаётся
    # назвать, чем карта занята: тело головы (fc + один слой) или голова словаря (1.27 ГБ
    # на ранг на КАЖДЫЙ из трёх проходов). Окна берут время СОБЫТИЯМИ, включаются тем же
    # FA2SM70_STEP_PHASE_DEV=1 и при выключенном рычаге стоят ноль.
    def _дополнить_мету(self, cad, реальных: int, всего: int):
        """Метаданные на ДОПОЛНЕННУЮ форму: `всего` запросов по одной строке.

        ЗАЧЕМ. Движок дополняет размер прохода черновика с `реальных` до `всего` (округление
        размеров захвата под q=k+1 цели, §256), и тогда метаданные описывают одну строку, а
        вход -- четыре. Однородный путь внимания такое рассогласование законно отвергает, и
        свой граф захватить нельзя (§257). Тело при этом трогать НЕЛЬЗЯ: оно скомпилировано
        под форму 4, а движок стоит с `evaluate_guards: False` -- вызов с другой формой падает
        на `assert_size_stride` даже вне захвата (проверено).
        Выход -- согласовать вторую сторону: объявить `всего` запросов по одной строке.
        Настоящие идут первыми и не меняются НИ НА БИТ; добавочные получают длину 1 и слот
        PADDING_SLOT_ID -- они ничего не пишут в кэш, а их выход отбрасывается вызывающим.
        Буферы постоянные: иначе граф запёк бы адреса разовых тензоров.
        """
        import dataclasses as _dc
        б = getattr(self, "_доп_буф", None)
        if б is None or б["всего"] != всего or б["блоков"] != cad.block_table_tensor.shape[1]:
            _у = cad.seq_lens.device
            б = self._доп_буф = {
                "всего": всего,
                "блоков": cad.block_table_tensor.shape[1],
                "qsl": torch.arange(всего + 1, device=_у, dtype=torch.int32),
                "qsl_cpu": torch.arange(всего + 1, dtype=torch.int32),
                "seq": torch.ones(всего, device=_у, dtype=cad.seq_lens.dtype),
                "seq_cpu": torch.ones(всего, dtype=torch.int32),
                "bt": torch.zeros((всего, cad.block_table_tensor.shape[1]),
                                  device=_у, dtype=cad.block_table_tensor.dtype),
                "sl": torch.full((всего,), PADDING_SLOT_ID, device=_у,
                                 dtype=cad.slot_mapping.dtype),
            }
        if _PAD_META_PROBE:
            # прибор стенда 18.10: сколько строк хвоста уйдёт черновику с НАСТОЯЩИМ (чужим) слотом
            _пр = self.__dict__.setdefault("_pad_probe", {"вызовов": 0, "сжатий": 0, "строк_чужих": 0, "вм": 0})
            _пр["вызовов"] += 1
            _вм = max(_пр["вм"], реальных)
            if _вм > реальных:
                _пр["сжатий"] += 1
                if not _PAD_META_FIX:
                    _пр["строк_чужих"] += _вм - реальных
            _пр["вм"] = реальных if _PAD_META_FIX else _вм
            if _пр["вызовов"] % 2000 == 0:
                logger.info("[fa2_sm70 PAD_META_PROBE] fix=%d %s", int(_PAD_META_FIX), dict(_пр))
        if _PAD_META_FIX:
            # [ФАЗЗИНГ 14.10, encsched] Хвост [реальных, грязь) хранит НАСТОЯЩИЕ seq_len / строку
            # таблицы / СЛОТ прошлого вызова с большим `реальных` (2 из 4 -> 1 из 4): слой черновика
            # писал K/V в слот чужого или завершённого запроса. Возвращаем заглушки только там,
            # где `реальных` уменьшилось -- в обычном шаге лишних пусков нет.
            _гр = б.get("грязь", 0)
            if _гр > реальных:
                б["seq"][реальных:_гр].fill_(1)
                б["bt"][реальных:_гр].zero_()
                б["sl"][реальных:_гр].fill_(PADDING_SLOT_ID)
            б["грязь"] = реальных
        б["seq"][:реальных].copy_(cad.seq_lens[:реальных], non_blocking=True)
        б["bt"][:реальных].copy_(cad.block_table_tensor[:реальных], non_blocking=True)
        б["sl"][:реальных].copy_(cad.slot_mapping[:реальных], non_blocking=True)
        # Хвост (длина 1, слот-заглушка, нулевая строка таблицы) заведён при создании буферов
        # и неизменен -- переписывать его каждый проход значило бы два лишних пуска ядер.
        # [КЕШ ОБЪЕКТА ОТМЕНЁН ЗАМЕРОМ -- 09.09] Кешировать результат `replace` и править у него
        # только `max_seq_len` выглядело даровой экономией 0.41 мс. Приёмка сказала иначе:
        # с кешем tau дважды просела 2.45 -> 2.19 (-10 %), без кеша держится 2.43-2.47.
        # Значит объект несёт поле, меняющееся ПОМИМО тех, что мы обновляем, и заморозка его
        # портит черновик. Пересобираем каждый вызов -- цена честнее ошибки.
        return _dc.replace(
            cad,
            query_start_loc=б["qsl"], query_start_loc_cpu=б["qsl_cpu"],
            seq_lens=б["seq"], num_reqs=всего, num_actual_tokens=всего,
            max_query_len=1, max_seq_len=int(cad.max_seq_len),
            block_table_tensor=б["bt"], slot_mapping=б["sl"],
            _seq_lens_cpu=None, _num_computed_tokens_cpu=None,
        )

    _ГР_ПОЛЯ = ("query_start_loc", "seq_lens", "block_table", "slot_mapping")

    def _тело_графом(self, мета, входных, cad, model_kwargs, шаг):
        """Проход тела головы одним CUDA-графом; None означает «иди обычным путём»."""
        if not _ГР_ЧЕРН:
            return None
        if _ГР_ЧЕРН_ФАЙЛ:
            self._гр_черед = getattr(self, "_гр_черед", 0) + 1
            if self._гр_черед % 64 == 1:
                try:
                    with open(_ГР_ЧЕРН_ФАЙЛ) as _ф:
                        self._гр_вкл = _ф.read(1) == "1"
                except OSError:
                    self._гр_вкл = True
            if not getattr(self, "_гр_вкл", True):
                return None
        _мсл = int(getattr(мета, "max_seq_len", 0) or 0)
        ключ = (int(входных), int(getattr(мета, "num_reqs", 0)),
                int(getattr(мета, "max_query_len", 0)), bool(getattr(мета, "causal", True)),
                int(шаг), _мсл // max(_ГР_ЧЕРН_ВЕДРО, 1))
        гр = getattr(self, "_гр_тела", None)
        if гр is None:
            гр = self._гр_тела = {}
        з = гр.setdefault(ключ, {"счёт": 0, "мёртв": False})
        if з["мёртв"]:
            return None
        if "граф" not in з:
            з["счёт"] += 1
            if з["счёт"] < 4:
                # ПРОГРЕВ ИДЁТ НАСТОЯЩЕЙ ФОРМОЙ, а не дополненной. dynamo специализируется на
                # ПЕРВОЙ увиденной форме и объявляет размерность динамической только увидев
                # вторую; тело черновика видело лишь дополненную 4, и захват падал на
                # `assert_size_stride: expected size 4==1`. Здесь оно видит и 1 -- если это и
                # была причина, размерность станет динамической сама. Отказ ловится ниже и
                # хоронит ключ, как всякий другой.
                try:
                    _по_слоям = {имя: мета for имя in self.attn_layer_names}
                    _слоты = self._get_slot_mapping(входных, cad.slot_mapping)
                    with set_forward_context(
                        _по_слоям, self.vllm_config, num_tokens=входных,
                        num_tokens_across_dp=None,
                        cudagraph_runtime_mode=CUDAGraphMode.NONE,
                        slot_mapping=_слоты,
                        additional_kwargs=self._get_forward_context_additional_kwargs(),
                    ):
                        return self.model(**model_kwargs)
                except Exception as _e:           # noqa: BLE001
                    з["мёртв"] = True
                    print(f"[MTP ГРАФ ТЕЛА] прогрев настоящей формой не прошёл "
                          f"({type(_e).__name__}: {str(_e)[:120]}) -- прежним путём",
                          file=__import__("sys").stderr, flush=True)
                    return None
            try:
                if getattr(self, "_гр_пул", None) is None:
                    self._гр_пул = torch.cuda.graph_pool_handle()
                _по_слоям = {имя: мета for имя in self.attn_layer_names}
                _слоты = self._get_slot_mapping(входных, cad.slot_mapping)
                г = torch.cuda.CUDAGraph()
                with set_forward_context(
                    _по_слоям, self.vllm_config, num_tokens=входных,
                    num_tokens_across_dp=None, cudagraph_runtime_mode=CUDAGraphMode.NONE,
                    slot_mapping=_слоты,
                    additional_kwargs=self._get_forward_context_additional_kwargs(),
                ):
                    with torch.cuda.graph(г, pool=self._гр_пул):
                        вых = self.model(**model_kwargs)
                з.update(граф=г, выход=вых,
                         буферы=[(и, getattr(мета, и, None)) for и in self._ГР_ПОЛЯ])
                г.replay()                        # захват ядра не исполнял -- шаг берёт своё
                print(f"[MTP ГРАФ ТЕЛА] захвачен: ключ={ключ}",
                      file=__import__("sys").stderr, flush=True)
                return вых
            except Exception as _e:               # noqa: BLE001
                з["мёртв"] = True
                print(f"[MTP ГРАФ ТЕЛА] отказ захвата ({type(_e).__name__}: {_e}) -- "
                      f"навсегда прежним путём для ключа {ключ}",
                      file=__import__("sys").stderr, flush=True)
                return None
        for и, буф in з["буферы"]:
            нов = getattr(мета, и, None)
            if буф is None or нов is None or нов.shape != буф.shape:
                return None
            if нов.data_ptr() != буф.data_ptr():
                буф.copy_(нов, non_blocking=True)
        self._get_slot_mapping(входных, cad.slot_mapping)
        з["граф"].replay()
        return з["выход"]

    def _карта(self, имя: str):
        if not _ФАЗА_ДЕВ:
            import contextlib
            return contextlib.nullcontext()
        р = getattr(self, "runner", None)
        ф = getattr(р, "_фаза_карты", None)
        if ф is None:
            import contextlib
            return contextlib.nullcontext()
        return ф(имя)

    def _greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
        with self._карта("чер: словарь+argmax"):
            return self._greedy_sample_ядро(hidden_states)

    def _greedy_sample_ядро(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Greedy-sample draft tokens from hidden states."""
        if _И4 and hidden_states.shape[0] <= _И4_MMAX:
            _т = self._i4_argmax(hidden_states)
            if _т is not None:
                return _т
        # [МЕСТНЫЙ АРГМАКС ВМЕСТО СБОРА ЛОГИТОВ -- 09.09, записка 25 §250]
        # Черновик трижды за шаг берёт аргмакс по словарю. При TP=2 обычный путь СОБИРАЕТ
        # все логиты (248320 float на ранг = 1 МиБ x3 за шаг), хотя нужен один номер.
        # `get_top_tokens` считает максимум по СВОЕЙ доле словаря и сводит пары
        # (значение, номер): обмен падает с O(словарь) до O(2*ранга). У нашей модели метод
        # есть (qwen3_5_mtp.py:509), а рычаг движка `use_local_argmax_reduction` выключен по
        # умолчанию. Здесь он ещё и чередуемый файлом -- парно внутри ОДНОГО экземпляра,
        # иначе разброс перезапуска (+-1.2 мс) съедает такой приз.
        if self._местный_аргмакс() and hasattr(self.model, "get_top_tokens"):
            return self.model.get_top_tokens(hidden_states)
        return self.model.compute_logits(hidden_states).argmax(dim=-1)

    def _местный_аргмакс(self) -> bool:
        if not _МЕСТ_ФАЙЛ:
            return self.use_local_argmax_reduction
        с = getattr(self, "_мест_сост", None)
        if с is None:
            с = self._мест_сост = {"вкл": False, "счёт": 0}
        с["счёт"] += 1
        if с["счёт"] % 64 == 1:
            try:
                with open(_МЕСТ_ФАЙЛ) as ф:
                    с["вкл"] = ф.read(1) == "1"
            except OSError:
                с["вкл"] = False
        return с["вкл"]

    def _collect_spec_decode_moe_layers(self) -> list[str] | None:
        if not self.compilation_config.fast_moe_cold_start:
            return None
        moe_layers = [
            module.layer_name
            for module in self.model.modules()
            if isinstance(module, FusedMoE)
        ]
        return moe_layers or None

    def _get_forward_context_additional_kwargs(self) -> dict[str, object] | None:
        if self.spec_decode_moe_layers:
            return {"spec_decode_moe_layers": self.spec_decode_moe_layers}
        return None

    # [n-ГРАММЫ В РОДНОМ MTP -- 15.09] Машинка n-грамм жила только в dflash2 (приёмка 3.86 против 2.56,
    # x3.9 на повторяемом тексте), а dflash2 на длинном контексте вредит и обнуляет префикс-кэш -- в бою стоит
    # родной MTP, и n-граммы туда не доставали. Здесь тот же объект (spec_decode/ngram_iz_konteksta.py)
    # спрашивается ПЕРЕД проходом черновика: если контекст сам себя продолжает, черновик берётся оттуда РОВНО
    # длины k (однородный декод и метаданные GDN не трогаются), а проход головы пропускается целиком. Выход не
    # меняется ПО ПОСТРОЕНИЮ: цель сверяет каждый черновой токен точным совпадением. Рычаг FA2SM70_MTP_NGRAM=1.
    def _нгр(self):
        о = getattr(self, "_нгр_объект", None)
        if о is None:
            import os as _o
            if _o.environ.get("FA2SM70_MTP_NGRAM", "0") != "1":
                self._нгр_объект = False
                return None
            from vllm.v1.spec_decode.ngram_iz_konteksta import НГраммыИзКонтекста
            о = self._нгр_объект = НГраммыИзКонтекста(
                getattr(self, "runner", None), int(self.num_speculative_tokens))
            print(f"[fa2_sm70] черновик MTP: n-ГРАММЫ ИЗ КОНТЕКСТА включены (k={self.num_speculative_tokens})",
                  flush=True)
        return о or None

    def propose(
        self,
        # [num_tokens]
        target_token_ids: torch.Tensor,
        # [num_tokens] or [3, num_tokens] when M-RoPE is enabled
        target_positions: torch.Tensor,
        # [num_tokens, hidden_size]
        target_hidden_states: torch.Tensor,
        # [batch_size]
        next_token_ids: torch.Tensor,
        last_token_indices: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,
    ) -> torch.Tensor:
        batch_size = common_attn_metadata.batch_size()

        _нг_об = self._нгр() if self.method == "mtp" else None
        if _нг_об is not None:
            _нг = _нг_об.черновик(batch_size, int(self.num_speculative_tokens), next_token_ids)
            if _нг is not None:
                return _нг

        if self.method == "eagle3":
            assert isinstance(self.model, Eagle3LlamaForCausalLM)
            target_hidden_states = self.model.combine_hidden_states(
                target_hidden_states
            )
            assert target_hidden_states.shape[-1] == self.hidden_size

        num_tokens, last_token_indices, common_attn_metadata = (
            self.set_inputs_first_pass(
                target_token_ids=target_token_ids,
                next_token_ids=next_token_ids,
                target_positions=target_positions,
                last_token_indices=last_token_indices,
                cad=common_attn_metadata,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            )
        )

        assert self.runner is not None

        if self.attn_metadata_builder is None:
            attn_metadata_builder = self._get_attention_metadata_builder()
        else:
            attn_metadata_builder = self.attn_metadata_builder

        attn_metadata = attn_metadata_builder.build_for_drafting(
            common_attn_metadata=common_attn_metadata, draft_index=0
        )
        # FIXME: support hybrid kv for draft model (remove separate indexer)
        if self.draft_indexer_metadata_builder:
            draft_indexer_metadata = (
                self.draft_indexer_metadata_builder.build_for_drafting(
                    common_attn_metadata=common_attn_metadata,
                    draft_index=0,
                )
            )
        else:
            draft_indexer_metadata = None
        # At this moment, we assume all eagle layers belong to the same KV
        # cache group, thus using the same attention metadata.
        per_layer_attn_metadata = {}
        for layer_name in self.attn_layer_names:
            per_layer_attn_metadata[layer_name] = attn_metadata

        for layer_name in self.indexer_layer_names:
            assert draft_indexer_metadata is not None
            per_layer_attn_metadata[layer_name] = draft_indexer_metadata

        num_tokens_dp_padded, num_tokens_across_dp = self._pad_batch_across_dp(
            num_tokens_unpadded=num_tokens, num_tokens_padded=num_tokens
        )

        cudagraph_runtime_mode, batch_desc = self.cudagraph_dispatcher.dispatch(
            num_tokens_dp_padded
        )
        num_input_tokens = batch_desc.num_tokens
        if num_tokens_across_dp is not None:
            num_tokens_across_dp[self.dp_rank] = num_input_tokens

        if self.pass_hidden_states_to_model:
            # target_hidden_states and self.hidden_states can have different
            # hidden dims. E.g. large target model and small draft model.
            self.hidden_states[:num_tokens] = target_hidden_states

        if self.supports_mm_inputs:
            mm_embeds, is_mm_embed = mm_embed_inputs or (None, None)

            self.inputs_embeds[:num_tokens] = self.model.embed_input_ids(
                self.input_ids[:num_tokens],
                multimodal_embeddings=mm_embeds,
                is_multimodal=is_mm_embed,
            )

            input_ids = None
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
        else:
            input_ids = self.input_ids[:num_input_tokens]
            inputs_embeds = None

        model_kwargs = {
            "input_ids": input_ids,
            "positions": self._get_positions(num_input_tokens),
            "inputs_embeds": inputs_embeds,
        }
        if self.pass_hidden_states_to_model:
            model_kwargs["hidden_states"] = self.hidden_states[:num_input_tokens]
        if self.method == "mtp":
            model_kwargs["spec_step_idx"] = 0

        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=self._get_slot_mapping(
                num_input_tokens, common_attn_metadata.slot_mapping
            ),
            additional_kwargs=self._get_forward_context_additional_kwargs(),
        ):
            with self._карта("чер: тело головы"):
                ret_hidden_states = self.model(**model_kwargs)
            if not self.model_returns_tuple():
                last_hidden_states = ret_hidden_states
                hidden_states = last_hidden_states
            else:
                last_hidden_states, hidden_states = ret_hidden_states

        sample_hidden_states = last_hidden_states[last_token_indices]

        # Early exit if there is only one draft token to be generated.
        if self.num_speculative_tokens == 1:
            draft_token_ids = self._greedy_sample(sample_hidden_states)
            return draft_token_ids.view(-1, 1)

        if self.uses_mrope:
            positions = self.mrope_positions[:, last_token_indices]
        else:
            positions = self.positions[last_token_indices]
        if self.method in (
            "deepseek_mtp",
            "ernie_mtp",
            "longcat_flash_mtp",
            "pangu_ultra_moe_mtp",
        ):
            hidden_states = self.hidden_states[last_token_indices]
        else:
            hidden_states = hidden_states[last_token_indices]

        if isinstance(attn_metadata, TreeAttentionMetadata):
            # Draft using tree attention.
            logits = self.model.compute_logits(sample_hidden_states)
            draft_token_ids_list = self.propose_tree(
                batch_size=batch_size,
                logits=logits,
                positions=positions,
                hidden_states=hidden_states,
                common_attn_metadata=common_attn_metadata,
                slot_mappings=slot_mappings,
            )
            # [batch_size, num_tree_tokens]
            return torch.cat(draft_token_ids_list, dim=1)

        draft_token_ids = self._greedy_sample(sample_hidden_states)

        # [ПРОБА ПРИЗА ДЕРЕВА -- 09.09, записка 25 §234]
        # Уравнение пункта 4 требует приёмки ~3.4 против нынешних 2.4; дерево кандидатов --
        # единственный измеренный путь, и его потолок -- доля случаев, когда верный токен
        # позиции 0 стоит у черновика ВТОРЫМ. Логиты уже посчитаны, проба даровая.
        # СВЕРКУ ДЕЛАЕТ РАННЕР, А НЕ МЫ. Первая редакция сверяла здесь, с `next_token_ids`, --
        # и врала: бонусный токен сдвинут на число ПРИНЯТЫХ, выравнивание не то (та же ошибка
        # уже ловилась у пробы покрытия DFlash2). Кладём кандидатов в `канд0`, а сверяет их
        # готовая проба раннера (FA2SM70_DFLASH2_COVER) -- против `sampled_token_ids[:, 0]`,
        # то есть против токена, который цель выдала НА ЭТОЙ ЖЕ позиции.
        if _СБОР_СКРЫТЫХ:
            # [СБОР ДЛЯ ДВУХСТУПЕНЧАТОГО АРГМАКСА -- 09.09, записка 25 §238-§239]
            # Черновик читает голову словаря (1.27 ГБ на ранг) ТРИЖДЫ за шаг ради трёх
            # аргмаксов -- 14 % шага на четыре чтения одной матрицы. Чтобы проверить замену
            # (короткий список по сжатому представлению + точная досчитка), нужны НАСТОЯЩИЕ
            # скрытые состояния черновика: на синтетике распределение другое, и оценка
            # полноты списка была бы недействительной.
            try:
                _б = getattr(self, "_скр_буф", None)
                if _б is None:
                    _б = self._скр_буф = []
                if len(_б) * sample_hidden_states.shape[0] < _СБОР_СКРЫТЫХ:
                    _б.append(sample_hidden_states.detach().to("cpu", torch.float16))
                elif _б:
                    import os as _o
                    torch.save(torch.cat(_б), _o.path.join(_o.environ.get("FA2SM70_DUMP_DIR", "/tmp"), f"mtp_skrytye_{_o.getpid()}.pt"))
                    print(f"[MTP СБОР] сохранено {sum(x.shape[0] for x in _б)} строк",
                          file=__import__("sys").stderr, flush=True)
                    self._скр_буф = []
                    globals()["_СБОР_СКРЫТЫХ"] = 0
            except Exception as _е:
                globals()["_СБОР_СКРЫТЫХ"] = 0
                print(f"[MTP СБОР] отказ: {_е}", file=__import__("sys").stderr, flush=True)
        if _ПРОБА_ТОП2:
            try:
                self.канд0 = self.model.compute_logits(
                    sample_hidden_states).topk(16, dim=-1).indices
            except Exception as _е:
                globals()["_ПРОБА_ТОП2"] = 0
                print(f"[MTP ТОП-2] отказ: {_е}", file=__import__("sys").stderr, flush=True)

        if self.allowed_attn_types is not None and not isinstance(
            attn_metadata, self.allowed_attn_types
        ):
            raise ValueError(
                f"Unsupported attention metadata type for speculative "
                "decoding with num_speculative_tokens > 1: "
                f"{type(attn_metadata)}. Supported types are: "
                f"{self.allowed_attn_types}"
            )

        # Generate the remaining draft tokens.
        draft_token_ids_list = [draft_token_ids]

        batch_size_dp_padded, batch_size_across_dp = self._pad_batch_across_dp(
            num_tokens_unpadded=batch_size, num_tokens_padded=batch_size
        )

        cudagraph_runtime_mode, batch_desc = self.cudagraph_dispatcher.dispatch(
            batch_size_dp_padded
        )
        input_batch_size = batch_desc.num_tokens
        # [ОДНОКРАТНАЯ УЛИКА -- 09.09, записка 25 §256] Выдача прохода черновика стоит 6.9 мс
        # при карточной работе 4.3: подозрение, что тело головы идёт БЕЗ графа. Режим печатаем
        # ОДИН раз -- гадать про это по журналу захвата нельзя, там своя разбивка.
        if not getattr(self, "_реж_напечатан", False):
            self._реж_напечатан = True
            print(f"[MTP граф] режим прохода черновика: {cudagraph_runtime_mode} "
                  f"(размер {input_batch_size}, просили {batch_size_dp_padded})",
                  file=__import__("sys").stderr, flush=True)
        if batch_size_across_dp is not None:
            batch_size_across_dp[self.dp_rank] = input_batch_size

        common_attn_metadata.num_actual_tokens = batch_size
        common_attn_metadata.max_query_len = 1
        common_attn_metadata.query_start_loc = self.arange[: batch_size + 1]
        common_attn_metadata.query_start_loc_cpu = torch.from_numpy(
            self.token_arange_np[: batch_size + 1]
        ).clone()

        # In padded drafter batch, we need to adjust the sequence lengths
        # to remove the "padding" (i.e. rejected tokens).
        # Only apply this adjustment when we have rejected tokens
        # (i.e., not the first proposal).
        if self.num_speculative_tokens > 1 and num_rejected_tokens_gpu is not None:
            common_attn_metadata.seq_lens -= num_rejected_tokens_gpu
            # Invalidate the CPU-side shadows to avoid H<>D sync.
            common_attn_metadata._seq_lens_cpu = None
            common_attn_metadata._num_computed_tokens_cpu = None

        for token_index in range(self.num_speculative_tokens - 1):
            _т_вход = _t_eagle.perf_counter()
            # Update the inputs.
            # cast to int32 is crucial when eagle model is compiled.
            # tensor.argmax() returns int64 by default.
            input_ids = draft_token_ids_list[-1].int()
            if self.uses_mrope:
                positions += 1
                # NOTE(woosuk): We should handle the case where the draft model
                # generates tokens beyond the max model length.
                # Since it is complex to remove such requests from the batch,
                # we keep them in the batch but adjust the position ids
                # and slot mappings to avoid the
                # out-of-range access during the model execution.
                # The draft tokens generated with this adjustment
                # should be ignored.
                exceeds_max_model_len = positions[0] >= self.max_model_len
                # Mask out the position ids that exceed the max model length.
                # Otherwise, we may get out-of-range error in RoPE.
                clamped_positions = torch.where(
                    exceeds_max_model_len.unsqueeze(0),
                    torch.zeros_like(positions),
                    positions,
                )
            else:
                positions += 1
                exceeds_max_model_len = positions >= self.max_model_len
                clamped_positions = torch.where(exceeds_max_model_len, 0, positions)
            # For data integrity when async scheduling, we shouldn't use in place
            # operations in case they are modified in next step's `prepare_input`
            # of main model.
            # Increment the sequence lengths.
            common_attn_metadata.seq_lens += 1
            # For the requests that exceed the max model length, we set the
            # sequence length to 1 to minimize their overheads in attention.
            common_attn_metadata.seq_lens.masked_fill_(exceeds_max_model_len, 1)
            # Increment the maximum sequence length. We increment max_seq_len
            # unconditionally even though some seq_lens may have been capped above,
            # as max_seq_len serves as an upper bound for sequence lengths.
            common_attn_metadata.max_seq_len = min(
                common_attn_metadata.max_seq_len + 1, self.max_model_len
            )

            # Also update the CPU-side shadow; NOTE: this is hacky and should be
            # removed in when common_attn_metadata.seq_lens_cpu is deprecated.
            if common_attn_metadata._seq_lens_cpu is not None:
                common_attn_metadata._seq_lens_cpu += 1
            if common_attn_metadata._num_computed_tokens_cpu is not None:
                common_attn_metadata._num_computed_tokens_cpu += 1

            # Compute the slot mapping.
            block_size = attn_metadata_builder.kv_cache_spec.block_size
            if self.uses_mrope:
                # all dimensions of positions are the same
                block_numbers = clamped_positions[0] // block_size
            else:
                block_numbers = clamped_positions // block_size
            block_ids = common_attn_metadata.block_table_tensor.gather(
                dim=1, index=block_numbers.view(-1, 1)
            )
            block_ids = block_ids.view(-1)
            if self.uses_mrope:
                common_attn_metadata.slot_mapping = (
                    block_ids * block_size + clamped_positions[0] % block_size
                )
            else:
                common_attn_metadata.slot_mapping = (
                    block_ids * block_size + clamped_positions % block_size
                )
            # Mask out the slot mappings that exceed the max model length.
            # Otherwise, the KV cache will be inadvertently updated with the
            # padding tokens.
            common_attn_metadata.slot_mapping.masked_fill_(
                exceeds_max_model_len, PADDING_SLOT_ID
            )
            _хф(getattr(self, "runner", None), "позиции+слоты", _т_вход)

            # Rebuild attention metadata
            _т_мета = _t_eagle.perf_counter()
            _cad = common_attn_metadata
            if _ГР_ЧЕРН and input_batch_size != batch_size:
                _cad = self._дополнить_мету(_cad, batch_size, input_batch_size)
            attn_metadata = attn_metadata_builder.build_for_drafting(  # type: ignore
                common_attn_metadata=_cad, draft_index=token_index + 1
            )
            for layer_name in self.attn_layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata
            _хф(getattr(self, "runner", None), "метаданные", _т_мета)

            # copy inputs to buffer for cudagraph
            self.input_ids[:batch_size] = input_ids
            self._set_positions(batch_size, clamped_positions)
            self.hidden_states[:batch_size] = hidden_states
            if self.supports_mm_inputs:
                self.inputs_embeds[:batch_size] = self.model.embed_input_ids(input_ids)

                input_ids = None
                inputs_embeds = self.inputs_embeds[:input_batch_size]
            else:
                input_ids = self.input_ids[:input_batch_size]
                inputs_embeds = None

            # Run the model.
            model_kwargs = {
                "input_ids": input_ids,
                "positions": self._get_positions(input_batch_size),
                "inputs_embeds": inputs_embeds,
            }
            if self.pass_hidden_states_to_model:
                model_kwargs["hidden_states"] = self.hidden_states[:input_batch_size]
            if self.method == "mtp":
                model_kwargs["spec_step_idx"] = token_index + 1

            _т_выд = _t_eagle.perf_counter()
            with set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=input_batch_size,
                num_tokens_across_dp=batch_size_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                slot_mapping=self._get_slot_mapping(
                    input_batch_size, common_attn_metadata.slot_mapping
                ),
                additional_kwargs=self._get_forward_context_additional_kwargs(),
            ):
                _хф(getattr(self, "runner", None), "контекст прохода", _т_выд)
                _т_тело = _t_eagle.perf_counter()
                with self._карта("чер: тело головы"):
                    # [БЕЗ ДОПОЛНЕНИЯ -- 09.09, записка 25 §257] Движок дополняет размер с 1
                    # до 4: `adjust_cudagraph_sizes_for_spec_decode` округляет ВСЕ размеры
                    # захвата вверх до кратного k+1 (цели нужен q=4), а у черновика размер
                    # считается в ЗАПРОСАХ. Из-за этого метаданные описывают 1 строку, а вход
                    # 4 -- и замок бэкенда справедливо отказал в захвате («однородный путь НЕ
                    # взят»). СВОЕМУ графу чужие размеры захвата не нужны: берём настоящие.
                    ret_hidden_states = None
                    if _ГР_ЧЕРН:            # ветвь мертва по умолчанию -- ни одной операции
                        # Форма НЕ меняется (тело запечено под неё) -- меняется только мета.
                        ret_hidden_states = self._тело_графом(
                            attn_metadata, input_batch_size, _cad,
                            model_kwargs, token_index + 1)
                    if ret_hidden_states is None:
                        ret_hidden_states = self.model(**model_kwargs)
                _хф(getattr(self, "runner", None), "ВЫДАЧА тела", _т_тело)
                if not self.model_returns_tuple():
                    last_hidden_states = ret_hidden_states
                    hidden_states = ret_hidden_states
                else:
                    last_hidden_states, hidden_states = ret_hidden_states

            hidden_states = hidden_states[:batch_size]
            _т_выб = _t_eagle.perf_counter()
            draft_token_ids = self._greedy_sample(last_hidden_states[:batch_size])
            draft_token_ids_list.append(draft_token_ids)
            _хф(getattr(self, "runner", None), "словарь+argmax", _т_выб)

        # [batch_size, num_speculative_tokens]
        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        if _ГЕЙТ_ДАМП:
            _снять_gejt(target_hidden_states, input_ids if "input_ids" in dir() else None,
                        draft_token_ids)
        return draft_token_ids

    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        last_token_indices: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata]:
        if last_token_indices is None:
            last_token_indices = cad.query_start_loc[1:] - 1

        num_tokens = target_token_ids.shape[0]
        # Shift the input ids by one token.
        # E.g., [a1, b1, b2, c1, c2, c3] -> [b1, b2, c1, c2, c3, c3]
        self.input_ids[: num_tokens - 1] = target_token_ids[1:]
        # Replace the last token with the next token.
        # E.g., [b1, b2, c1, c2, c3, c3] -> [a2, b2, b3, c2, c3, c4]
        self.input_ids[last_token_indices] = next_token_ids

        # copy inputs to buffer for cudagraph
        if self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim == 0:
            target_positions = target_positions[0]
        self._set_positions(num_tokens, target_positions)

        return num_tokens, last_token_indices, cad

    def model_returns_tuple(self) -> bool:
        return self.method not in ("mtp", "draft_model")

    def prepare_next_token_ids_cpu(
        self,
        sampled_token_ids: list[list[int]],
        requests: dict[str, CachedRequestState],
        gpu_input_batch: InputBatch,
        num_scheduled_tokens: dict[str, int],
    ) -> torch.Tensor:
        """
        This function is used to prepare the inputs for speculative decoding.
        It calculates the next token ids for each request based on the sampled
        token ids from the CPU. If a request has no sampled token ids (e.g.,
        during the initial decoding steps), it falls back to using the request
        state to get the next token id.
        """
        req_ids = gpu_input_batch.req_ids
        next_token_ids: list[int] = []
        for i, token_ids in enumerate(sampled_token_ids):
            if token_ids:
                # Common case.
                next_token_id = token_ids[-1]
            else:
                # Partial prefill (rare case).
                # Get the next token id from the request state.
                req_id = req_ids[i]
                req_state = requests[req_id]
                seq_len = req_state.num_computed_tokens + num_scheduled_tokens[req_id]
                next_token_id = req_state.get_token_id(seq_len)
            next_token_ids.append(next_token_id)
        if _ЗАКРЕП:
            # было: torch.tensor(список, device=cuda) -- синхронная копия на каждом шаге
            next_token_ids = _закреп("next_tok",
                                     torch.tensor(next_token_ids, dtype=torch.int32),
                                     self.input_ids.device)
        else:
            next_token_ids = torch.tensor(
                next_token_ids, dtype=torch.int32, device=self.input_ids.device
            )
        return next_token_ids

    def prepare_next_token_ids_padded(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        sampled_token_ids: torch.Tensor,
        requests: dict[str, CachedRequestState],
        gpu_input_batch: InputBatch,
        discard_request_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding.
        It calculates the next token ids and the number of valid sampled tokens
        for each request, considering the "discarded" requests whose next token
        is not sampled and comes from `request.get_token_id()` instead. This is denoted
        the "backup" token id. It also counts rejected tokens via `sampled_token_ids`.
        """
        # Precompute get_token_id for when there is no valid next token
        num_reqs = gpu_input_batch.num_reqs
        self.backup_next_token_ids.np[:num_reqs] = np.array(
            [
                requests[gpu_input_batch.req_ids[i]].get_token_id(
                    common_attn_metadata.seq_lens_cpu[i].item()
                )
                for i in range(num_reqs)
            ],
            dtype=np.int32,
        )
        self.backup_next_token_ids.copy_to_gpu(num_reqs)
        backup_tokens_gpu = self.backup_next_token_ids.gpu

        batch_size, num_tokens = sampled_token_ids.shape
        device = sampled_token_ids.device

        assert discard_request_mask.dtype == torch.bool
        assert backup_tokens_gpu.dtype == torch.int32

        next_token_ids = torch.empty(batch_size, dtype=torch.int32, device=device)
        valid_sampled_tokens_count = next_token_ids.new_empty(batch_size)

        # Kernel grid: one program per request (row)
        grid = (batch_size,)

        # Find the next power of 2 for block sizes
        BLOCK_SIZE_TOKENS = triton.next_power_of_2(num_tokens)
        eagle_prepare_next_token_padded_kernel[grid](
            sampled_token_ids,
            discard_request_mask,
            backup_tokens_gpu,
            next_token_ids,
            valid_sampled_tokens_count,
            gpu_input_batch.vocab_size,
            num_tokens,
            batch_size,
            sampled_token_ids.stride(0),
            BLOCK_SIZE_TOKENS=BLOCK_SIZE_TOKENS,
        )

        return next_token_ids, valid_sampled_tokens_count

    def prepare_inputs_padded(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        spec_decode_metadata: SpecDecodeMetadata,
        valid_sampled_tokens_count: torch.Tensor,
    ) -> tuple[CommonAttentionMetadata, torch.Tensor, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding
        It updates the common_attn_metadata for speculative decoding,
        but does not consider the rejected tokens. Instead, all tokens
        are included as inputs to the speculator, with the rejected tokens
        used as padding and filtered out later by `token_indices_to_sample`.
        No blocking CPU operations should be introduced in this function.
        """
        num_reqs = common_attn_metadata.num_reqs
        device = valid_sampled_tokens_count.device

        token_indices_to_sample = torch.empty(
            (num_reqs,), dtype=torch.int32, device=device
        )
        num_rejected_tokens_gpu = torch.empty(
            (num_reqs,), dtype=torch.int32, device=device
        )

        grid = (num_reqs,)
        eagle_prepare_inputs_padded_kernel[grid](
            spec_decode_metadata.cu_num_draft_tokens,
            valid_sampled_tokens_count,
            common_attn_metadata.query_start_loc,
            token_indices_to_sample,
            num_rejected_tokens_gpu,
            num_reqs,
        )

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        new_query_len_per_req = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

        total_num_tokens = query_start_loc_cpu[-1].item()

        spec_common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=common_attn_metadata.query_start_loc,
            seq_lens=common_attn_metadata.seq_lens,
            query_start_loc_cpu=query_start_loc_cpu,
            _seq_lens_cpu=common_attn_metadata._seq_lens_cpu,
            _num_computed_tokens_cpu=common_attn_metadata._num_computed_tokens_cpu,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=total_num_tokens,
            max_query_len=new_query_len_per_req.max().item(),
            max_seq_len=common_attn_metadata.seq_lens_cpu.max().item(),
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping[:total_num_tokens],
            causal=True,
            dcp_local_seq_lens=common_attn_metadata.dcp_local_seq_lens,
        )

        return (
            spec_common_attn_metadata,
            token_indices_to_sample,
            num_rejected_tokens_gpu,
        )

    def propose_tree(
        self,
        batch_size: int,
        # [num_tokens, vocab_size]
        logits: torch.Tensor,
        # [num_tokens]
        positions: torch.Tensor,
        # [num_tokens, hidden_size]
        hidden_states: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,
    ) -> list[torch.Tensor]:
        tree_attn_metadata_builder = self.runner.attn_groups[0][
            0
        ].get_metadata_builder()
        assert isinstance(tree_attn_metadata_builder, TreeAttentionMetadataBuilder)

        total_num_drafts = self.cu_drafts_per_level[0]
        level_num_drafts = total_num_drafts
        # Sample a draft token for each child at the tree root level.
        num_children = self.child_drafts_per_level[0]
        if num_children == 1:
            draft_token_ids = logits.argmax(dim=-1).view(batch_size, -1)
        else:
            draft_token_ids = torch.topk(logits, num_children, dim=-1).indices.view(
                batch_size, -1
            )
        draft_token_ids_list = [draft_token_ids]
        draft_hidden_states = hidden_states.view(batch_size, 1, -1)

        # Initialize empty tensors for concatenation with the level outputs.
        tree_input_ids = torch.empty(
            0, device=self.input_ids.device, dtype=self.input_ids.dtype
        )
        tree_positions = torch.empty(
            0, device=self.positions.device, dtype=self.positions.dtype
        )
        tree_hidden_states = torch.empty(
            0, device=self.hidden_states.device, dtype=self.hidden_states.dtype
        )
        # Precompute the draft token positions.
        flattened_draft_positions = (
            positions.view(batch_size, -1) + self.tree_draft_pos_offsets[:batch_size, :]
        )
        tree_depth = len(self.cu_drafts_per_level)
        for level in range(tree_depth - 1):
            # Get draft positions for RoPE.
            draft_positions = positions + (level + 1)
            exceeds_max_model_len = (positions + total_num_drafts) >= self.max_model_len
            # Mask out the position ids that exceed the max model length.
            # Otherwise, we may get out-of-range error in RoPE.
            draft_positions = torch.where(
                exceeds_max_model_len,
                0,
                draft_positions,
            ).view(batch_size, -1)

            if level_num_drafts > 1:
                # Repeat the positions for each draft at this level.
                draft_positions = draft_positions.repeat_interleave(
                    level_num_drafts, dim=1
                )

            if num_children > 1:
                # Repeat draft hidden states for each child.
                draft_hidden_states = draft_hidden_states.repeat_interleave(
                    num_children, dim=1
                )

            # Concatenate the draft tokens, positions, and hidden states.
            tree_input_ids = torch.cat([tree_input_ids, draft_token_ids], dim=1)
            tree_positions = torch.cat([tree_positions, draft_positions], dim=1)
            tree_hidden_states = torch.cat(
                [tree_hidden_states, draft_hidden_states], dim=1
            )

            # Build new attention metadata for the next level of drafts.
            # This is necessary to support tree attention.
            query_len = total_num_drafts
            common_attn_metadata = replace(
                common_attn_metadata,
                query_start_loc=query_len * self.arange[: batch_size + 1],
                seq_lens=common_attn_metadata.seq_lens + level_num_drafts,
                num_actual_tokens=batch_size * query_len,
                max_query_len=query_len,
            )
            attn_metadata = tree_attn_metadata_builder.build_for_drafting(
                common_attn_metadata=common_attn_metadata, draft_index=level + 1
            )

            # Apply new attention metadata to all layers.
            per_layer_attn_metadata = {}
            for layer_name in self.attn_layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata

            # Consider max model length.
            attn_metadata.max_seq_len = min(
                attn_metadata.max_seq_len, self.max_model_len
            )
            # For the requests that exceed the max model length, we set the
            # sequence length to 1 to minimize their overheads in attention.
            attn_metadata.seq_lens.masked_fill_(exceeds_max_model_len, 1)

            # Compute the slot mapping.
            block_size = tree_attn_metadata_builder.kv_cache_spec.block_size
            query_positions = flattened_draft_positions[:, level : level + query_len]
            block_numbers = query_positions // block_size
            block_ids = attn_metadata.block_table.gather(dim=1, index=block_numbers)
            slot_mapping = block_ids * block_size + query_positions % block_size
            # Mask out the slot mappings that exceed the max model length.
            # Otherwise, the KV cache will be inadvertently updated with the
            # padding tokens.
            slot_mapping[exceeds_max_model_len] = PADDING_SLOT_ID
            attn_metadata.slot_mapping = slot_mapping.view(-1)

            # Copy inputs to buffer for cudagraph.
            num_tokens = attn_metadata.num_actual_tokens
            input_ids = tree_input_ids.view(-1)
            self.input_ids[:num_tokens] = input_ids
            self.positions[:num_tokens] = tree_positions.view(-1)
            self.hidden_states[:num_tokens] = tree_hidden_states.view(num_tokens, -1)

            cudagraph_runtime_mode, batch_desc = self.cudagraph_dispatcher.dispatch(
                num_tokens
            )
            num_input_tokens = batch_desc.num_tokens
            # Run the model.
            with set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                slot_mapping=self._get_slot_mapping(
                    num_input_tokens, attn_metadata.slot_mapping
                ),
                additional_kwargs=self._get_forward_context_additional_kwargs(),
            ):
                last_hidden_states, hidden_states = self.model(
                    input_ids=self.input_ids[:num_input_tokens],
                    positions=self.positions[:num_input_tokens],
                    hidden_states=self.hidden_states[:num_input_tokens],
                    inputs_embeds=None,
                    spec_step_idx=level + 1,
                )

            # Get the output hidden states for the draft tokens.
            draft_hidden_states = hidden_states[:num_tokens].view(
                batch_size, query_len, -1
            )[:, -level_num_drafts:]
            draft_last_hidden_states = last_hidden_states[:num_tokens].view(
                batch_size, query_len, -1
            )[:, -level_num_drafts:]

            # Get the output logits for the draft tokens.
            logits = self.model.compute_logits(
                draft_last_hidden_states.reshape(batch_size * level_num_drafts, -1)
            )

            # Sample a draft token for each child at the next tree level.
            num_children = self.child_drafts_per_level[level + 1]
            if num_children == 1:
                draft_token_ids = logits.argmax(dim=-1).view(batch_size, -1)
            else:
                draft_token_ids = torch.topk(logits, num_children, dim=-1).indices.view(
                    batch_size, -1
                )
            draft_token_ids_list.append(draft_token_ids)

            # Update the # drafts counters for the next tree level.
            level_num_drafts = self.cu_drafts_per_level[level + 1] - total_num_drafts
            total_num_drafts = self.cu_drafts_per_level[level + 1]
        return draft_token_ids_list

    def prepare_inputs(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        sampled_token_ids: list[list[int]],
        num_draft_tokens: list[int],
    ) -> tuple[CommonAttentionMetadata, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding.
        It updates to the common_attn_metadata to account for the rejected
        tokens (and newly sampled tokens). It also returns the token indices
        of the tokens that should be fed to the speculator.
        """
        # E.g.
        #  common_attn_metadata.query_start_loc{_cpu}:
        #       [0, q1, q1 + q2, q1 + q2 + q3]
        #  common_attn_metadata.seq_lens{_cpu}: [s1, s2, s3]
        #  num_rejected_tokens: [n1, n2, n3]
        # This function computes the intermediate values:
        #  num_tokens_per_req: [q1 - n1, q2 - n2, q3 - n3]
        # And returns:
        #  common_attn_metadata.query_start_loc{_cpu}:
        #       [0, q1 - n1, q1 + q2 - n1 - n2, q1 + q2 + q3 - n1 - n2 - n3]
        #  common_attn_metadata.seq_lens{_cpu}:
        #       [s1 - n1 + 1, s2 - n2 + 1, s3 - n3 + 1]
        #  token_indices: [0, 1, ..., q1 - n1 - 1,
        #                 q1, q1 + 1, ..., q1 + q2 - n2 - 1,
        #                 q1 + q2, q1 + q2 + 1, ..., q1 + q2 + q3 - n3 - 1]

        num_rejected_tokens = [
            n + 1 - len(sampled_token_ids[i]) if n > 0 else 0
            for i, n in enumerate(num_draft_tokens)
        ]
        num_rejected_tokens = torch.tensor(num_rejected_tokens, dtype=torch.int32)

        device = common_attn_metadata.query_start_loc.device
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        new_seq_lens_cpu = common_attn_metadata.seq_lens_cpu - num_rejected_tokens

        # [0, q1, q1 + q2, q1 + q2 + q3] -> [q1, q2, q3]
        new_query_len_per_req = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        # [q1, q2, q3] -> [q1 - n1, q2 - n2, q3 - n3]
        new_num_tokens_per_req = new_query_len_per_req - num_rejected_tokens
        new_num_tokens_per_req_np = new_num_tokens_per_req.numpy()

        # [q1 - n1, q2 - n2, q3 - n3] ->
        # [0, q1 - n1, q1 + q2 - n1 - n2, q1 + q2 + q3 - n1 - n2 - n3]
        new_query_start_loc_cpu = torch.zeros(
            query_start_loc_cpu.shape,
            dtype=torch.int32,
            pin_memory=is_pin_memory_available(),
        )
        new_query_start_loc_np = new_query_start_loc_cpu.numpy()
        np.cumsum(new_num_tokens_per_req_np, out=new_query_start_loc_np[1:])

        total_num_tokens = new_query_start_loc_np[-1]
        # Example assuming num_tokens_per_req_np = [2, 4, 3]
        # this implies that `new_query_start_locs` is:
        # [0, 2, 6, 9] ->
        # [0, 0, 2, 2, 2, 2, 6, 6, 6]
        #  _r1_  ____r2____  ___r3__
        new_query_start_locs_expanded = np.repeat(
            new_query_start_loc_np[:-1], new_num_tokens_per_req_np
        )
        # [0, 1, 2, 3, 4, 5, 6, 7, 8] ->
        # [0, 1, 0, 1, 2, 3, 0, 1, 2]
        #  _r1_  ____r2____  ___r3__
        token_offsets = (
            self.token_arange_np[:total_num_tokens] - new_query_start_locs_expanded
        )

        # Expand starting positions to match token pattern
        # [0, q1, q1 + q2] ->
        # [0, 0, q1, q1, q1, q1, q1 + q2, q1 + q2, q1 + q2]
        #  _r1_  _____r2_______  ___________r3____________
        old_query_start_locs_expanded = np.repeat(
            query_start_loc_cpu[:-1].numpy(), new_num_tokens_per_req_np
        )
        # Final token indices are:
        # [0, 1,                                // req 1
        #  q1 + 0, q1 + 1, q1 + 2, q1 + 3,       // req 2
        #  q1 + q2 + 0, q1 + q2 + 1, q1 + q2 + 2] // req 3
        token_indices_np = token_offsets + old_query_start_locs_expanded
        if _ЗАКРЕП:
            token_indices = _закреп("tok_idx", token_indices_np, device)
            _qsl = _закреп("qsl", new_query_start_loc_cpu, device)
            _sl = _закреп("slens", new_seq_lens_cpu, device)
        else:
            token_indices = torch.from_numpy(token_indices_np).to(device, non_blocking=True)
            _qsl = new_query_start_loc_cpu.to(device, non_blocking=True)
            _sl = new_seq_lens_cpu.to(device, non_blocking=True)

        spec_common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=_qsl,
            seq_lens=_sl,
            query_start_loc_cpu=new_query_start_loc_cpu,
            _seq_lens_cpu=new_seq_lens_cpu,
            _num_computed_tokens_cpu=common_attn_metadata._num_computed_tokens_cpu,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=total_num_tokens,
            max_query_len=new_query_len_per_req.max().item(),
            max_seq_len=new_seq_lens_cpu.max().item(),
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping[token_indices],
            causal=True,
            dcp_local_seq_lens=common_attn_metadata.dcp_local_seq_lens,
        )

        return spec_common_attn_metadata, token_indices

    def get_model_name(self, model: nn.Module) -> str:
        if hasattr(model, "module"):  # multi-GPU
            model = model.module
        return model.__class__.__name__

    def load_model(self, target_model: nn.Module) -> None:
        draft_model_config = self.speculative_config.draft_model_config
        target_attn_layer_names = set(
            get_layers_from_vllm_config(
                self.vllm_config,
                AttentionLayerBase,  # type: ignore[type-abstract]
            ).keys()
        )
        # FIXME: support hybrid kv for draft model
        target_indexer_layer_names = set(
            get_layers_from_vllm_config(
                self.vllm_config, DeepseekV32IndexerCache
            ).keys()
        )

        from vllm.compilation.backends import set_model_tag

        with set_model_tag("eagle_head"):
            self.model = get_model(
                vllm_config=self.vllm_config, model_config=draft_model_config
            )

        draft_attn_layer_names = (
            get_layers_from_vllm_config(
                self.vllm_config,
                AttentionLayerBase,  # type: ignore[type-abstract]
            ).keys()
            - target_attn_layer_names
        )
        indexer_layers = get_layers_from_vllm_config(
            self.vllm_config, DeepseekV32IndexerCache
        )
        draft_indexer_layer_names = indexer_layers.keys() - target_indexer_layer_names
        self.attn_layer_names = list(draft_attn_layer_names - draft_indexer_layer_names)
        self.indexer_layer_names = list(draft_indexer_layer_names)

        if self.indexer_layer_names:
            first_layer = self.indexer_layer_names[0]
            self.draft_indexer_metadata_builder = (
                indexer_layers[first_layer]
                .get_attn_backend()
                .get_builder_cls()(
                    indexer_layers[first_layer].get_kv_cache_spec(self.vllm_config),
                    self.indexer_layer_names,
                    self.vllm_config,
                    self.device,
                )
            )
        else:
            self.draft_indexer_metadata_builder = None

        if self.supports_mm_inputs:
            # Even if the target model is multimodal, we can also use
            # text-only draft models
            try:
                dummy_input_ids = torch.tensor([[1]], device=self.input_ids.device)
                self.model.embed_input_ids(dummy_input_ids, multimodal_embeddings=None)
            except (NotImplementedError, AttributeError, TypeError):
                logger.warning(
                    "Draft model does not support multimodal inputs, "
                    "falling back to text-only mode"
                )
                self.supports_mm_inputs = False

        if supports_multimodal(target_model):
            # handle multimodality
            assert hasattr(target_model, "config")
            if self.get_model_name(target_model) in [
                "Qwen2_5_VLForConditionalGeneration",
                "Qwen3VLForConditionalGeneration",
                "Qwen3VLMoeForConditionalGeneration",
                "Qwen3_5ForConditionalGeneration",
                "Qwen3_5MoeForConditionalGeneration",
                "HunYuanVLForConditionalGeneration",
                "GlmOcrForConditionalGeneration",
            ]:
                self.model.config.image_token_index = target_model.config.image_token_id
            elif self.get_model_name(target_model) == "PixtralForConditionalGeneration":
                self.model.config.image_token_index = (
                    target_model.config.vision_config.image_token_id
                )
            else:
                self.model.config.image_token_index = (
                    target_model.config.image_token_index
                )
            target_language_model = cast(
                SupportsMultiModal, target_model
            ).get_language_model()
        else:
            target_language_model = target_model

        # share embed_tokens with the target model if needed
        if get_pp_group().world_size == 1:
            inner_model = getattr(target_language_model, "model", None)
            if inner_model is None:
                raise AttributeError("Target model does not have 'model' attribute")
            if hasattr(inner_model, "embed_tokens"):
                target_embed_tokens = inner_model.embed_tokens
            elif hasattr(inner_model, "embedding"):
                target_embed_tokens = inner_model.embedding
            else:
                raise AttributeError(
                    "Target model does not have 'embed_tokens' or 'embedding' attribute"
                )

            share_embeddings = False
            if hasattr(self.model, "has_own_embed_tokens"):
                # EAGLE model
                if not self.model.has_own_embed_tokens:
                    share_embeddings = True
                    logger.info(
                        "Detected EAGLE model without its own embed_tokens in the"
                        " checkpoint. Sharing target model embedding weights with the"
                        " draft model."
                    )
                elif (
                    isinstance(target_embed_tokens.weight, torch.Tensor)
                    and isinstance(self.model.model.embed_tokens.weight, torch.Tensor)
                    # TODO: Offload to CPU for comparison to avoid extra GPU memory
                    # usage in CI testing environments with limited GPU memory
                    and torch.equal(
                        target_embed_tokens.weight.cpu(),
                        self.model.model.embed_tokens.weight.cpu(),
                    )
                ):
                    share_embeddings = True
                    logger.info(
                        "Detected EAGLE model with embed_tokens identical to the target"
                        " model. Sharing target model embedding weights with the draft"
                        " model."
                    )
                else:
                    logger.info(
                        "Detected EAGLE model with distinct embed_tokens weights. "
                        "Keeping separate embedding weights from the target model."
                    )
            else:
                # MTP model
                share_embeddings = True
                logger.info(
                    "Detected MTP model. "
                    "Sharing target model embedding weights with the draft model."
                )

            if share_embeddings:
                if hasattr(self.model.model, "embed_tokens"):
                    del self.model.model.embed_tokens
                self.model.model.embed_tokens = target_embed_tokens
        else:
            logger.info(
                "The draft model's vocab embedding will be loaded separately"
                " from the target model."
            )

        # share lm_head with the target model if needed
        share_lm_head = False
        if hasattr(self.model, "has_own_lm_head"):
            # EAGLE model
            if not self.model.has_own_lm_head:
                share_lm_head = True
                logger.info(
                    "Detected EAGLE model without its own lm_head in the checkpoint. "
                    "Sharing target model lm_head weights with the draft model."
                )
            elif (
                hasattr(target_language_model, "lm_head")
                and isinstance(target_language_model.lm_head.weight, torch.Tensor)
                and isinstance(self.model.lm_head.weight, torch.Tensor)
                # TODO: Offload to CPU for comparison to avoid extra GPU memory
                # usage in CI testing environments with limited GPU memory
                and torch.equal(
                    target_language_model.lm_head.weight.cpu(),
                    self.model.lm_head.weight.cpu(),
                )
            ):
                share_lm_head = True
                logger.info(
                    "Detected EAGLE model with lm_head identical to the target model. "
                    "Sharing target model lm_head weights with the draft model."
                )
            else:
                logger.info(
                    "Detected EAGLE model with distinct lm_head weights. "
                    "Keeping separate lm_head weights from the target model."
                )
        else:
            # MTP model
            share_lm_head = True
            logger.info(
                "Detected MTP model. "
                "Sharing target model lm_head weights with the draft model."
            )

        if share_lm_head and hasattr(target_language_model, "lm_head"):
            if hasattr(self.model, "lm_head"):
                del self.model.lm_head
            self.model.lm_head = target_language_model.lm_head

        self.spec_decode_moe_layers = self._collect_spec_decode_moe_layers()

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
    ) -> None:
        # FIXME: when using tree-based specdec, adjust number of forward-passes
        # according to the depth of the tree.
        for fwd_idx in range(
            self.num_speculative_tokens if not is_graph_capturing else 1
        ):
            if fwd_idx <= 1:
                num_tokens_dp_padded, num_tokens_across_dp = self._pad_batch_across_dp(
                    num_tokens_unpadded=num_tokens, num_tokens_padded=num_tokens
                )
                if use_cudagraphs:
                    cudagraph_runtime_mode, batch_desc = (
                        self.cudagraph_dispatcher.dispatch(num_tokens_dp_padded)
                    )
                    num_input_tokens = batch_desc.num_tokens
                else:
                    cudagraph_runtime_mode = CUDAGraphMode.NONE
                    num_input_tokens = num_tokens_dp_padded
                if num_tokens_across_dp is not None:
                    num_tokens_across_dp[self.dp_rank] = num_input_tokens

            # Make sure to use EAGLE's own buffer during cudagraph capture.
            if (
                self.attn_layer_names
                and slot_mappings is not None
                and self.attn_layer_names[0] in slot_mappings
            ):
                slot_mapping_dict = self._get_slot_mapping(num_input_tokens)
            else:
                slot_mapping_dict = slot_mappings or {}

            with set_forward_context(
                None,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                slot_mapping=slot_mapping_dict,
                additional_kwargs=self._get_forward_context_additional_kwargs(),
            ):
                if self.supports_mm_inputs:
                    input_ids = None
                    inputs_embeds = self.inputs_embeds[:num_input_tokens]
                else:
                    input_ids = self.input_ids[:num_input_tokens]
                    inputs_embeds = None

                kwargs = dict(
                    input_ids=input_ids,
                    positions=self._get_positions(num_input_tokens),
                    inputs_embeds=inputs_embeds,
                )
                if self.pass_hidden_states_to_model:
                    kwargs["hidden_states"] = self.hidden_states[:num_input_tokens]
                if self.method == "mtp":
                    kwargs["spec_step_idx"] = fwd_idx
                self.model(**kwargs)

    def _get_attention_metadata_builder(self) -> AttentionMetadataBuilder:
        """Find and return the attention metadata builders for EAGLE layers.

        Returns:
            The metadata builders for EAGLE layers.

        Raises:
            AssertionError: If no metadata builders are found for EAGLE layers.
        """
        builder = None
        chosen_layer = self.attn_layer_names[0]

        for kv_cache_group in self.runner.attn_groups:
            for attn_group in kv_cache_group:
                if chosen_layer in attn_group.layer_names:
                    builder = attn_group.get_metadata_builder()
                    break
            if builder is not None:
                break

        assert builder is not None, (
            "Failed to find attention metadata builder for EAGLE layers."
        )
        return builder

    def _get_eagle3_use_aux_hidden_state_from_config(self) -> bool:
        """
        Some eagle3 heads (e.g., nvidia/gpt-oss-120b-Eagle3-v2) do not use auxiliary
        hidden states and directly uses the last layer output just like eagle1.
        They might indicate this by setting "use_aux_hidden_state" to False
        inside the "eagle_config" dict of their hf_config.
        """
        if self.method != "eagle3":
            return False
        # Assume that eagle3 heads use aux hidden states by default
        use_aux_hidden_state = True
        eagle_config = getattr(self.draft_model_config.hf_config, "eagle_config", None)
        if eagle_config is not None:
            use_aux_hidden_state = eagle_config.get("use_aux_hidden_state", True)
        return use_aux_hidden_state

    def validate_same_kv_cache_group(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Validate that all drafting layers belong to the same KVCacheGroup.
        Need this assumption to ensure all drafting layers can use the
        same AttentionMetadata.
        May extend to multiple AttentionMetadata in the future.
        """
        kv_cache_groups: dict[str, int] = {}
        for id, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            for layer_name in kv_cache_group.layer_names:
                kv_cache_groups[layer_name] = id
        assert (
            len(
                set(
                    [
                        kv_cache_groups[layer_name]
                        for layer_name in self.attn_layer_names
                    ]
                )
            )
            == 1
        ), "All drafting layers should belong to the same kv cache group"

    def _pad_batch_across_dp(
        self,
        num_tokens_unpadded: int,
        num_tokens_padded: int,
    ) -> tuple[int, torch.Tensor]:
        # TODO(Flechman): support DBO ubatching
        should_ubatch, num_toks_across_dp, _ = coordinate_batch_across_dp(
            num_tokens_unpadded=num_tokens_unpadded,
            parallel_config=self.vllm_config.parallel_config,
            allow_microbatching=False,
            allow_dp_padding=self.cudagraph_dispatcher.cudagraph_mode
            != CUDAGraphMode.NONE,
            num_tokens_padded=num_tokens_padded,
            uniform_decode=None,
            num_scheduled_tokens_per_request=None,
        )
        assert not should_ubatch, "DBO ubatching not implemented for EAGLE"

        num_tokens_dp_padded = num_tokens_padded
        if num_toks_across_dp is not None:
            num_tokens_dp_padded = int(num_toks_across_dp[self.dp_rank].item())
        return num_tokens_dp_padded, num_toks_across_dp


class EagleProposer(SpecDecodeBaseProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(
            vllm_config,
            device,
            pass_hidden_states_to_model=True,
            runner=runner,
        )


# NOTE(woosuk): Currently, the below code is not used and we always use argmax
# to sample the draft tokens. We will use this after we find a way to manage
# the draft prob tensor.
# Refer to https://github.com/vllm-project/vllm/pull/16899 for the details.
# FIXME(woosuk): The logic here is duplicated with the main sampling code.
# We should refactor this to reuse the same sampling implementation.
def compute_probs_and_sample_next_token(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sampling_metadata.all_greedy:
        # For greedy requests, draft_probs is not used in rejection sampling.
        # Therefore, we can just return the logits.
        probs = logits
        next_token_ids = logits.argmax(dim=-1)
        return next_token_ids, probs

    assert sampling_metadata.temperature is not None

    # Use epsilon comparison to detect greedy sampling (temperature ~ 0.0)
    # consistent with sampler.py's _SAMPLING_EPS threshold
    temperature = sampling_metadata.temperature
    # Avoid division by zero if there are greedy requests.
    if not sampling_metadata.all_random:
        is_greedy = temperature < _SAMPLING_EPS
        temperature = torch.where(is_greedy, 1.0, temperature)
    logits.div_(temperature.view(-1, 1))
    probs = logits.softmax(dim=-1, dtype=torch.float32)

    # NOTE(woosuk): Currently, we ignore most of the sampling parameters in
    # generating the draft tokens. We only use the temperature. While this
    # could degrade the acceptance rate, it does not affect the distribution
    # of the generated tokens after rejection sampling.

    # TODO(woosuk): Consider seeds.
    q = torch.empty_like(probs)
    q.exponential_()
    # NOTE(woosuk): We shouldn't use `probs.div_(q)` because the draft_probs
    # will be used later for rejection sampling.
    next_token_ids = probs.div(q).argmax(dim=-1).view(-1)
    if not sampling_metadata.all_random:
        greedy_token_ids = probs.argmax(dim=-1)
        next_token_ids = torch.where(is_greedy, greedy_token_ids, next_token_ids)
    return next_token_ids, probs
