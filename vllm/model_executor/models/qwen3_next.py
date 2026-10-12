# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen3Next model."""

import os
from collections.abc import Iterable
from itertools import islice

import torch
import os as _os_kva
_KVA_DIR = _os_kva.environ.get("FA2SM70_KVA_DUMP", "")       # прибор KVA (записка 26 §6), умолчание выкл.
_KVA_SPLIT = int(_os_kva.environ.get("FA2SM70_KVA_SPLIT", "32"))
_KVA_N = [0]
_KVA_SLED = [False]
# [ПРИБОР KVA] Дамп потока через CUSTOM OP: под torch.compile Python-проверка файла замораживалась при трассировке
# (ветка становилась мёртвой), а custom op -- непрозрачен для Dynamo и исполняется на каждом вызове.
# mutates_args=("h",) -- чтобы компилятор не выбросил op без выходов как мёртвый код.
@torch.library.custom_op("fa2sm70::kva_dump", mutates_args=())
def _kva_dump_op(h: torch.Tensor, r: torch.Tensor, pos: torch.Tensor, layer: int) -> torch.Tensor:
    # возвращает копию h, которую цикл слоёв подставляет дальше: иначе op без потребителя выбрасывается как мёртвый код
    if _KVA_DIR and h.shape[0] >= 1024 and int(torch.cuda.current_device()) == 0 and _os_kva.path.exists(_KVA_DIR + "/ON"):
        if not _KVA_SLED[0]:
            _KVA_SLED[0] = True; print(f"[fa2_sm70 KVA] op дампа слоя {layer} исполняется: строк {h.shape[0]}", flush=True)
        _KVA_N[0] += 1
        torch.save({"h": (h + r).detach().to(torch.float16).cpu(), "pos": pos.detach().cpu()}, f"{_KVA_DIR}/h{layer}_{_KVA_N[0]:06d}.pt")
    return h.clone()
@_kva_dump_op.register_fake
def _kva_dump_fake(h: torch.Tensor, r: torch.Tensor, pos: torch.Tensor, layer: int) -> torch.Tensor:
    return torch.empty_like(h)
_KVA_GN: dict = {}

def _kva_gdn_dump(self, key_ns, value_ns, g_ns, beta_ns, extra=None):
    """[ПРИБОР KVA] цели projector'а поздних GDN-слоёв: пост-conv k, v, g, beta по позициям (см. крюк в цикле слоёв)."""
    if not _KVA_DIR or getattr(self, "layer_idx", -1) < _KVA_SPLIT or key_ns is None or key_ns.shape[1] < 1024 \
            or g_ns is None or beta_ns is None or not _os_kva.path.exists(_KVA_DIR + "/ON") or not _os_kva.path.exists(_KVA_DIR + "/G"):
        return
    _rk = int(torch.cuda.current_device()); _key = (self.layer_idx, _rk)
    _KVA_GN[_key] = _KVA_GN.get(_key, 0) + 1
    d = {"k": key_ns[0].detach().to(torch.float16).cpu(), "v": value_ns[0].detach().to(torch.float16).cpu(),
         "g": g_ns[0].detach().float().cpu(), "beta": beta_ns[0].detach().float().cpu()}
    if extra: d.update(extra)
    torch.save(d, f"{_KVA_DIR}/gdn_L{self.layer_idx}_r{_rk}_{_KVA_GN[_key]:06d}.pt")

from torch import nn
from transformers.activations import ACT2FN

from vllm.compilation.decorators import support_torch_compile
from vllm.config import (
    CacheConfig,
    ModelConfig,
    SpeculativeConfig,
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed import (
    divide,
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger

try:  # приватная обвязка int16-состояния GDN (задача 195); её может не быть в чужом дереве
    import fa2sm70_gdn_i16 as _i16
except Exception:  # noqa: BLE001
    _i16 = None

# [ФАЛЬСИФИКАТОРЫ -- КОНСТАНТЫ, А НЕ ЧТЕНИЕ НА КАЖДЫЙ СЛОЙ, 31.08]
# Три рычага снятия фаз читались ВНУТРИ горячих функций: `SKIP_GDN` -- на каждый из 48 слоёв
# GDN, `SKIP_FULLATTN` -- на каждый из 16 слоёв внимания, `SKIP_CONV` -- в префилльной ветке,
# и рядом с каждым стоял ещё `import os` внутри функции. Это шестьдесят с лишним обращений к
# словарю окружения за шаг при ВЫКЛЮЧЕННЫХ фальсификаторах. Окружение в рантайме не меняется.
import os as _os_фальс
_ПРОПУСК_GDN = _os_фальс.environ.get("FA2SM70_SKIP_GDN", "0") == "1"
_ПРОПУСК_СВЁРТКИ = _os_фальс.environ.get("FA2SM70_SKIP_CONV", "0") == "1"
_ПРОПУСК_ВНИМАНИЯ = _os_фальс.environ.get("FA2SM70_SKIP_FULLATTN", "0") == "1"


def _вещ(пул):
    """Вещественный тип, в котором с состоянием работает питон-путь префилла."""
    return torch.float32 if пул.dtype == torch.int16 else пул.dtype


def _чит(пул, индекс):
    """Состояние по индексу слота -- всегда в вещественном виде."""
    if пул.dtype != torch.int16:
        return пул[индекс]
    return _i16.читать(пул, индекс, torch.float32)


def _пиш(пул, индекс, значение):
    if пул.dtype != torch.int16:
        пул[индекс] = значение.to(пул.dtype)
        return
    _i16.писать(пул, индекс, значение)

from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fla.ops import (
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.model_executor.layers.fused_moe import SharedFusedMoE
from vllm.model_executor.layers.layernorm import (
    GemmaRMSNorm as Qwen3NextRMSNorm,
)
from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_mixer2 import mamba_v2_sharded_weight_loader
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
    sharded_weight_loader,
)
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs import Qwen3NextConfig
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from .interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsLoRA,
    SupportsPP,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)

# [ОБЩИЕ ИНДЕКСЫ СПЕКУЛЯТИВНОЙ ВЕТВИ GDN НА ШАГ -- 11.09] См. разбор в `_forward_core`.
# Умолчание 0: venv общий с боевым, включение -- FA2SM70_GDN_SHARE_IDX=1.
_ОБЩ_ИДX = __import__("os").environ.get("FA2SM70_GDN_SHARE_IDX", "0") == "1"
QWEN3_NEXT_SM70_TRACE = os.getenv("VLLM_QWEN3_NEXT_SM70_TRACE", "0") == "1"
# Рычаг «без трёх копий q/k/v в GDN» (пункт цели: убрать параллельную работу). Умолчание 0.
_БЕЗ_КОПИЙ_QKV = os.environ.get("FA2SM70_GDN_QKV_NOCOPY", "0") == "1"
QWEN3_NEXT_FUSED_SIGMOID_GATING_ENABLED = (
    os.getenv("VLLM_QWEN3_NEXT_FUSED_SIGMOID_GATING", "1") == "1"
)
_QWEN3_NEXT_SM70_SEEN: set[tuple[str, int, int, int]] = set()

# ФАЛЬСИФИКАТОР СНЯТИЕМ ФАЗЫ (ответ заведомо неверен, читается ТОЛЬКО время): убирает
# рекуррент GDN на спекулятивной ветке. Флаг читается ОДИН РАЗ при загрузке -- чтение
# окружения внутри forward рвёт torch.compile.
FA2SM70_NOGDN_SPEC = os.getenv("FA2SM70_NOGDN_SPEC", "0") == "1"
# Снятие ВСЕЙ фазы GDN (не только рекуррента) -- см. пояснение в forward ниже.
FA2SM70_FALSE_GDN = os.getenv("FA2SM70_FALSE_GDN", "0") == "1"
FA2SM70_FALSE_MLP = os.getenv("FA2SM70_FALSE_MLP", "0") == "1"
_FALSE_GDN_СЧЁТ = 0
_FALSE_MLP_СЧЁТ = 0
FA2SM70_PHASE_PROBE_MS = float(os.getenv("FA2SM70_PHASE_PROBE_MS", "0"))

KVCache = tuple[torch.Tensor, torch.Tensor]


def _maybe_sm70_projection(
    layer: nn.Module,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None] | None:
    # Let the generic LinearBase path handle compiled regions. Its SM70 dense
    # fast path already routes through the graph-safe raw op and avoids feeding
    # Python ints through the piecewise compiled graph.
    if torch.compiler.is_compiling():
        return None

    if not getattr(layer, "_sm70_f16_prepared", False):
        return None

    if not hasattr(torch.ops._C, "sm70_f16_gemm"):
        return None

    # Row-parallel layers that still expect an implicit TP input split cannot
    # use the raw SM70 projection shortcut. Bypassing that split breaks the
    # layer semantics and can also confuse torch.compile's mutation tracking.
    if hasattr(layer, "input_is_parallel") and not layer.input_is_parallel:
        return None

    # СЛОЙ, ОБЯЗАННЫЙ СВОДИТЬ РЕЗУЛЬТАТ, ЯРЛЫКОМ НЕ ОБСЛУЖИВАЕТСЯ. Этот путь возвращает
    # произведение НАПРЯМУЮ, минуя RowParallelLinear.forward, а значит и минуя
    # tensor_model_parallel_all_reduce -- наружу уходит ЧАСТИЧНАЯ СУММА.
    # До сих пор это было прикрыто проверкой is_compiling(): внутри компилируемой области
    # ярлык не берётся. Любой вызов слоя ВНЕ этой области (например, разбор по слоям или
    # мегашаг) попадал на ветвь, верную только внутри неё. ЗАМЕРЕНО: вклад слоя 0 различался
    # по рангам 4.8671 против 10.8121 при совпадающих входе и выходе MLP.
    if getattr(layer, "reduce_results", False) and get_tensor_model_parallel_world_size() > 1:
        return None

    x_2d = x.reshape(-1, x.shape[-1])
    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()
    if QWEN3_NEXT_SM70_TRACE:
        key = (
            getattr(layer, "prefix", "<unknown>"),
            x_2d.size(0),
            layer.weight.shape[0],
            x_2d.size(1),
        )
        if key not in _QWEN3_NEXT_SM70_SEEN:
            _QWEN3_NEXT_SM70_SEEN.add(key)
            logger.info(
                "QWEN3_NEXT SM70 projection prefix=%s m=%d n=%d k=%d",
                key[0],
                key[1],
                key[2],
                key[3],
            )
    output = torch.ops._C.sm70_f16_gemm(x_2d, layer.weight)
    output = output.reshape(*x.shape[:-1], output.shape[-1])
    if getattr(layer, "bias", None) is not None and not getattr(
        layer, "skip_bias_add", False
    ):
        output = output + layer.bias
    output_bias = layer.bias if getattr(layer, "skip_bias_add", False) else None
    return output, output_bias


# [ДЕРЕВО, 05.09] Ветвление рекуррента и сдвиг чтения при принятой ветви B.
# Реестр заполняет раннер (буфер постоянный -- живёт с графами); ядро GDN получает
# branch_at=W+1 (строка b0 продолжает якорь, а не хвост ветви A -- сверено бит в бит
# офлайн-тестом test_branch.py). Сдвиг nacc на W ложится в gdn_attn.build.
_ДЕРЕВО_W_ENV = int(__import__("os").environ.get("FA2SM70_TREE_W", "0"))
_ДЕРЕВО_БЕЗ_BRANCH = __import__("os").environ.get("FA2SM70_TREE_NOBRANCH", "0") == "1"
# [СТРАЖ ИНДЕКСОВ, 08.09] Тот же приём, что дал первое исключение по состоянию GDN:
# ядро-проверка запускается в потоке и потому попадает в ЗАХВАТ ГРАФА -- значит видит
# то, что исполняется на самом деле, а не то, что показывает трасса. Здесь накрываются
# таблицы блоков свёртки: и не-спекулятивная, и спекулятивная.
_СТРАЖ = __import__("os").environ.get("FA2SM70_GDN_GUARD", "0") == "1"
_страж_модуль = None


def _страж_идx(табл, предел, вид):
    global _страж_модуль
    if not _СТРАЖ or табл is None:
        return
    try:
        if _страж_модуль is None:
            import fa2_sm70._ext as _e
            _страж_модуль = _e.gdn_ext()
        _страж_модуль.gdn_check_idx(табл, int(предел), int(вид))
    except Exception:
        pass
ДЕРЕВО_OFF = {"буф": None}


class Qwen3NextSparseMoeBlock(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config

        self.tp_size = get_tensor_model_parallel_world_size()

        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts = config.num_experts

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        # Load balancing settings.
        vllm_config = get_current_vllm_config()
        eplb_config = vllm_config.parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb

        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate",
        )

        self.shared_expert_gate = ReplicatedLinear(
            config.hidden_size,
            1,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.shared_expert_gate",
        )

        if config.shared_expert_intermediate_size > 0:
            self.shared_expert = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                expert_gate=self.shared_expert_gate,
                prefix=f"{prefix}.shared_expert",
            )
        else:
            self.shared_expert = None

        self.experts = SharedFusedMoE(
            shared_experts=self.shared_expert,
            gate=self.gate,
            num_experts=self.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=getattr(config, "norm_topk_prob", True),
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        if self.experts.is_internal_router:
            # In this case, the gate/router runs inside the FusedMoE class
            final_hidden_states = self.experts(
                hidden_states=hidden_states, router_logits=hidden_states
            )
        else:
            # router_logits: (num_tokens, n_experts)
            router_logits, _ = self.gate(hidden_states)
            final_hidden_states = self.experts(
                hidden_states=hidden_states, router_logits=router_logits
            )

        if self.shared_expert is not None:
            final_hidden_states = final_hidden_states[0] + final_hidden_states[1]

        if self.is_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )
            final_hidden_states = final_hidden_states[:num_tokens]
        elif self.tp_size > 1:
            final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(  # noqa E501
                final_hidden_states
            )

        return final_hidden_states.view(orig_shape)



# [ПРОБА КОЛОНКИ ЗАПИСИ, 07.09] Ядро свёртки проверяет координату ЧТЕНИЯ на
# pad_slot_id и делает ранний возврат, а координату ЗАПИСИ не проверяет ничем.
# Прибор считает, сколько строк за пуск имеют в колонке записи паддинг (запись
# ушла бы ДО начала тензора) или саму колонку за краем таблицы (чтение за буфер).
# Под графом не срабатывает (питон в повторе не исполняется) -- мерить с CG=0.
from vllm.v1.attention.backends.utils import PAD_SLOT_ID  # прибор ниже
_ПРОБА_ЗАП = os.environ.get("FA2SM70_CONV_PROBE", "0") == "1"
_ПРОБА_ИТОГ: dict = {}


def _проба_записи(таблица, кол_чт, кол_зап, метка):
    if not _ПРОБА_ЗАП or таблица is None or кол_зап is None:
        return
    try:
        n = min(таблица.shape[0], кол_зап.shape[0])
        if n <= 0:
            return
        т = таблица[:n]
        зап = кол_зап[:n].reshape(-1).long()
        за_краем = int((зап >= т.shape[1]).sum()) + int((зап < 0).sum())
        безопасн = зап.clamp(0, т.shape[1] - 1).unsqueeze(1)
        w = т.gather(1, безопасн).reshape(-1)
        в_паддинге = int((w == PAD_SLOT_ID).sum())
        расх = 0
        if кол_чт is not None:
            чт = кол_чт[:n].reshape(-1).long().clamp(0, т.shape[1] - 1).unsqueeze(1)
            расх = int((т.gather(1, чт).reshape(-1) != w).sum())
        а = _ПРОБА_ИТОГ.setdefault(метка, [0, 0, 0, 0])
        а[0] += 1; а[1] += в_паддинге; а[2] += за_краем; а[3] += расх
        if в_паддинге or за_краем or а[0] == 1 or а[0] % 4000 == 0:
            print(f"[ПРОБА-ЗАП {метка}] пусков={а[0]} строк_в_паддинге={а[1]} "
                  f"колонка_за_краем={а[2]} чт!=зап={а[3]}", flush=True)
    except Exception as e:
        print(f"[ПРОБА-ЗАП {метка}] сбой прибора: {e}", flush=True)



class Qwen3NextGatedDeltaNet(nn.Module, MambaBase):
    @property
    def mamba_type(self) -> str:
        return "gdn_attention"

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        типы = MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
            self.cache_config.mamba_ssm_cache_dtype,
        )
        # СОСТОЯНИЕ В int16 (задача 195). Два байта, как у fp16, но сетка РАВНОМЕРНАЯ:
        # замерено на 2000 шагов рекуррента -- выход точнее в 3.0 раза, само состояние в
        # 5.7 раза. Это лечит вырождение длинной генерации, которым fp16 оплачивал ёмкость.
        # Масштабы живут в приватной таблице рядом с пулом (fa2sm70_gdn_i16).
        if _i16 is not None and _i16.включён():
            # МИНА, КОТОРУЮ ЗАКРЫВАЕМ ЗАРАНЕЕ: в режиме 'align' движок КОПИРУЕТ состояние
            # между слотами своей функцией (get_mamba_state_copy_func). Она сдвинет КОДЫ и
            # не тронет приватную таблицу масштабов -- состояние разъедется тихо. В режиме
            # 'all' (наш боевой) копирования нет: питон и ядро пишут по индексу, масштаб
            # едет вместе со значением.
            if getattr(self.cache_config, "mamba_cache_mode", "none") == "align":
                raise ValueError(
                    "[fa2_sm70] FA2SM70_GDN_I16 несовместим с mamba_cache_mode='align': "
                    "движок копирует состояние между слотами мимо таблицы масштабов. "
                    "Используйте режим 'all' (FA2SM70_MAMBA_ALL=1) или выключите int16.")
            типы = (типы[0], torch.int16)
        return типы

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            self.tp_size,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            self.conv_kernel_size,
            self.num_spec,
        )

    def __init__(
        self,
        config: Qwen3NextConfig,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        speculative_config: SpeculativeConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = extract_layer_index(prefix)
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.layer_norm_epsilon = config.rms_norm_eps
        self.prefix = prefix

        self.config = config
        self.model_config = model_config
        self.cache_config = cache_config
        self.quant_config = quant_config
        self.speculative_config = speculative_config
        self.num_spec = (
            self.speculative_config.num_speculative_tokens
            if self.speculative_config
            else 0
        )

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_kernel_size,
            output_size=self.conv_dim,
            bias=False,
            prefix=f"{prefix}.conv1d",
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        # projection of the input hidden states
        self.projection_size_qkvz = self.key_dim * 2 + self.value_dim * 2
        self.projection_size_ba = self.num_v_heads * 2
        self.in_proj_qkvz = self.create_qkvz_proj(
            hidden_size=self.hidden_size,
            key_dim=self.key_dim,
            value_dim=self.value_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj_qkvz",
        )
        # ba_proj doesn't support blockwise fp8 quantization.
        self.in_proj_ba = self.create_ba_proj(
            hidden_size=self.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj_ba",
        )

        query_key_settings = (self.key_dim, 0, False)
        value_settings = (self.value_dim, 0, False)

        delattr(self.conv1d.weight, "weight_loader")
        set_weight_attrs(
            self.conv1d.weight,
            {
                "weight_loader": mamba_v2_sharded_weight_loader(
                    [
                        query_key_settings,
                        query_key_settings,
                        value_settings,
                    ],
                    self.tp_size,
                    self.tp_rank,
                )
            },
        )

        # selective projection used to make dt, B and C input dependant

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(
            torch.ones(self.num_v_heads // self.tp_size),
        )
        self.A_log = nn.Parameter(
            torch.empty(
                divide(self.num_v_heads, self.tp_size),
            )
        )

        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(0)})
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        # Use model_config.dtype (runtime dtype, e.g. fp16 on SM70)
        # instead of config.dtype (HF config dtype, e.g. bf16)
        _norm_dtype = model_config.dtype if model_config is not None else config.dtype
        self.norm = RMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            group_size=None,
            norm_before_gate=True,
            device=current_platform.current_device(),
            dtype=_norm_dtype,
        )

        self.out_proj = RowParallelLinear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def create_qkvz_proj(
        self,
        hidden_size: int,
        key_dim: int,
        value_dim: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> ColumnParallelLinear:
        """Create the QKVZ projection layer. Subclasses can override."""
        return ColumnParallelLinear(
            input_size=hidden_size,
            output_size=key_dim * 2 + value_dim * 2,
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
        )

    def create_ba_proj(
        self,
        hidden_size: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> MergedColumnParallelLinear | None:
        return MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[self.num_v_heads] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
        )

    def fix_query_key_value_ordering(
        self,
        mixed_qkvz,
        mixed_ba,
    ):
        """
        Derives `query`, `key` and `value` tensors from `mixed_qkvzba`.
        """
        new_tensor_shape_qkvz = mixed_qkvz.size()[:-1] + (
            self.num_k_heads // self.tp_size,
            (
                self.head_k_dim
                + self.head_k_dim
                + (self.head_v_dim + self.head_v_dim)
                * self.num_v_heads
                // self.num_k_heads
            ),
        )
        new_tensor_shape_ba = mixed_qkvz.size()[:-1] + (
            self.num_k_heads // self.tp_size,
            2 * self.num_v_heads // self.num_k_heads,
        )

        mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)

        split_arg_list_qkvz = [
            self.head_k_dim,
            self.head_k_dim,
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
        ]
        split_arg_list_ba = [
            self.num_v_heads // self.num_k_heads,
            self.num_v_heads // self.num_k_heads,
        ]

        # [b, sq, ng, (hn + hn + np/ng * hn + np/ng + np/ng)]
        # --> [b, sq, ng, hn], [b, sq, ng, hn], [b, sq, ng, np/ng * hn],
        #  [b, sq, ng, np/ng * hn], [b, sq, ng, np/ng], [b, sq, ng, np/ng]
        (query, key, value, z) = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=2)
        (b, a) = torch.split(mixed_ba, split_arg_list_ba, dim=2)

        # [b, sq, ng, np/ng * hn] -> [b, sq, np, hn]
        value = value.reshape(value.size(0), -1, self.head_v_dim)
        z = z.reshape(z.size(0), -1, self.head_v_dim)
        b = b.reshape(b.size(0), self.num_v_heads // self.tp_size)
        a = a.reshape(a.size(0), self.num_v_heads // self.tp_size)

        return query, key, value, z, b, a

    def _дерево_ssm_nacc(self, attn_metadata, nacc):
        """nacc для SSM-чтения: сдвинутый буфер, если билдер его дал; плюс дозор."""
        ssm = getattr(attn_metadata, "num_accepted_ssm", None)
        итог = ssm if ssm is not None else nacc
        if getattr(self, "prefix", "").endswith("layers.0.linear_attn"):
            _дг = ДЕРЕВО_OFF.get("дозор")
            if _дг is not None and итог is not None and nacc is not None:
                if "nacc" not in _дг:
                    _дг["nacc"] = torch.zeros(2, dtype=torch.int32, device=nacc.device)
                _дг["nacc"][0].copy_(nacc[0])
                _дг["nacc"][1].copy_(итог[0])
        return итог

    def rearrange_mixed_qkv(self, mixed_qkv):
        if mixed_qkv is None:
            return None, None, None
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim // self.tp_size,
                self.key_dim // self.tp_size,
                self.value_dim // self.tp_size,
            ],
            dim=-1,
        )
        query = query.view(1, query.size(0), -1, self.head_k_dim)
        key = key.view(1, key.size(0), -1, self.head_k_dim)
        value = value.view(1, value.size(0), -1, self.head_v_dim)
        # [ТРИ КОПИИ НА КАЖДЫЙ СЛОЙ GDN -- задача «убрать параллельную работу»]
        # `split` по последней оси даёт НЕсмежные виды, и три `.contiguous()` копируют их
        # обратно. Трасса: 768 clone + 768 contiguous + 768 copy_ + 768 empty_like на форме
        # [1,4,8,128] за 12 шагов -- 256 операций на шаг. Байт там мало (8 КиБ на копию), но
        # каждая операция стоит диспетчеризации торча, а это единицы микросекунд НА КАЖДУЮ.
        # ОТВЕТ ЗАМЕРОМ (31.08): копии НУЖНЫ. Без них ядро GDN (Triton) падает на подъёме --
        # `Triton Error [CUDA]: out of memory`: на нес смежных входах оно строит другой план
        # и не укладывается в память. Рычаг оставлен выключенным как документированный отказ.
        # Цена самих копий при этом мала: 64 вызова `.contiguous()` на шаг, около 0.6 мс из 85,
        # то есть меньше процента -- убирать их не стоило бы и в случае успеха.
        if _БЕЗ_КОПИЙ_QKV:
            return query, key, value
        return query.contiguous(), key.contiguous(), value.contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        """
        Forward pass with three parts:
        1. Input projection
        2. Core attention (custom op)
        3. Output projection
        """
        # [FA2/SM70 27.08] ФАЛЬСИФИКАТОР СНЯТИЕМ ВСЕЙ ФАЗЫ GDN (FA2SM70_FALSE_GDN=1).
        # Ответ заведомо неверен, читается ТОЛЬКО время. Нужен потому, что прежний
        # FA2SM70_NOGDN_SPEC снимал лишь РЕКУРРЕНТ, а потокенная работа слоя (свёртка,
        # гейтование, правило дельты) оставалась -- и наклон по позициям приписывался телу,
        # хотя микрозамер tm8 по всем боевым формам показал, что тело ПЛОСКОЕ по M.
        if FA2SM70_FALSE_GDN:
            # ДОКАЗАТЕЛЬСТВО ИСПОЛНЕНИЯ: без него «фальсификатор выставлен» не значит
            # «фальсификатор работает» -- ровно на этом я сегодня уже обжёгся.
            global _FALSE_GDN_СЧЁТ
            _FALSE_GDN_СЧЁТ += 1
            if _FALSE_GDN_СЧЁТ in (1, 1000, 100000):
                print(f"[fa2_sm70] ФАЛЬСИФИКАТОР GDN сработал {_FALSE_GDN_СЧЁТ} раз",
                      file=__import__("sys").stderr, flush=True)
            output.zero_()
            return
        num_tokens = hidden_states.size(0)

        # ============================================================
        # Part 1: Input Projection
        # ============================================================
        projected_states_qkvz = _maybe_sm70_projection(
            self.in_proj_qkvz, hidden_states
        )
        if projected_states_qkvz is None:
            projected_states_qkvz = self.in_proj_qkvz(hidden_states)
        projected_states_qkvz, _ = projected_states_qkvz

        projected_states_ba = _maybe_sm70_projection(
            self.in_proj_ba, hidden_states
        )
        if projected_states_ba is None:
            projected_states_ba = self.in_proj_ba(hidden_states)
        projected_states_ba, _ = projected_states_ba
        query, key, value, z, b, a = self.fix_query_key_value_ordering(
            projected_states_qkvz, projected_states_ba
        )
        query = query.reshape(query.size(0), -1)
        key = key.reshape(key.size(0), -1)
        value = value.reshape(value.size(0), -1)
        mixed_qkv = torch.cat((query, key, value), dim=-1)

        # ============================================================
        # Part 2: Core Attention (Custom Op)
        # ============================================================
        # Note: we should not use torch.empty here like other attention backends,
        # see discussions in https://github.com/vllm-project/vllm/pull/28182
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # [FA2/SM70] ФАЛЬСИФИКАТОР СНЯТИЕМ ФАЗЫ (FA2SM70_SKIP_GDN=1): ответ заведомо неверен,
        # читается ТОЛЬКО время -- доля всей линейной мешалки (свёртка + скан) в стене.
        if _ПРОПУСК_GDN:
            core_attn_out.zero_()
        else:
            torch.ops.vllm.gdn_attention_core(
                mixed_qkv,
                b,
                a,
                core_attn_out,
                self.prefix,
            )

        # ============================================================
        # Part 3: Output Projection
        # ============================================================
        z_shape_og = z.shape
        # Reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(
            *core_attn_out.shape[:-2], -1)
        out_proj = _maybe_sm70_projection(self.out_proj, core_attn_out)
        if out_proj is None:
            out_proj = self.out_proj(core_attn_out)
        output[:num_tokens], _ = out_proj

    def _forward_core(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        """
        Core attention computation (called by custom op).
        """
        forward_context = get_forward_context()
        attn_metadata: AttentionMetadata = forward_context.attn_metadata

        if attn_metadata is None:
            # V1 profile run
            return

        assert isinstance(attn_metadata, dict)
        attn_metadata = attn_metadata[self.prefix]
        assert isinstance(attn_metadata, GDNAttentionMetadata)
        has_initial_state = attn_metadata.has_initial_state
        spec_query_start_loc = attn_metadata.spec_query_start_loc
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        spec_token_indx = attn_metadata.spec_token_indx
        non_spec_token_indx = attn_metadata.non_spec_token_indx
        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor  # noqa: E501
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        self_kv_cache = self.kv_cache[forward_context.virtual_engine]
        conv_state = self_kv_cache[0].transpose(-1, -2)
        ssm_state = self_kv_cache[1]
        if ssm_state.dtype == torch.int16 and _i16 is not None:
            _i16.таблица(ssm_state)   # ленивая привязка масштабов К ЭТОМУ пулу
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_accepted_tokens = attn_metadata.num_accepted_tokens

        # --- РЕЖИМ 'all': ПРЕФИКС-КЭШ ДЛЯ РЕКУРРЕНТНОГО СОСТОЯНИЯ (задача 194) -------------
        # Признак режима -- ненулевой mamba_block_size в метаданных. В нём индексы состояний
        # ДВУМЕРНЫЕ (запрос x блоки): состояние берётся из блока последнего ПОСЧИТАННОГО
        # токена, а не из единственного слота запроса, и сохраняется на КАЖДОЙ границе блока.
        # Именно этого не хватало, чтобы префикс-кэш и спекуляция работали вместе.
        mamba_block_size = attn_metadata.mamba_block_size
        mamba_all = mamba_block_size > 0
        if mamba_all and non_spec_state_indices_tensor is not None:
            _blk_computed = attn_metadata.block_idx_last_computed_token
            _blk_last = attn_metadata.block_idx_last_scheduled_token
            # Одномерные индексы для потребителей, которым нужен ОДИН блок:
            #   ..._read  -- откуда взять состояние (последний посчитанный токен),
            #   ..._write -- куда его положить после шага (последний запланированный).
            # Они различаются ровно в тот шаг, когда токен переходит границу блока.
            # ЭТИ ДВА СБОРА ЗАВИСЯТ ТОЛЬКО ОТ МЕТАДАННЫХ, А НЕ ОТ СЛОЯ. Метаданные строятся
            # ОДИН раз на KV-группу за шаг, а слоёв в группе 4-5, и каждый пересчитывал их
            # заново (97 сборов за шаг по профилю). Память держим НА САМОМ ОБЪЕКТЕ метаданных:
            # он создаётся заново каждый шаг, значит устареть не может -- в отличие от кэша по
            # id(), который у нас уже давал тихую порчу (см. `_общий_ключ` в gdn_attn.py).
            _пам = getattr(attn_metadata, "_fa2_idx_pam", None)
            if _пам is None:
                _idx_read = non_spec_state_indices_tensor.gather(
                    1, _blk_computed.unsqueeze(1).long()
                ).squeeze(1)
                _idx_write = non_spec_state_indices_tensor.gather(
                    1, _blk_last.unsqueeze(1).long()
                ).squeeze(1)
                try:
                    attn_metadata._fa2_idx_pam = (_idx_read, _idx_write)
                except Exception:
                    pass
            else:
                _idx_read, _idx_write = _пам
        else:
            _blk_computed = _blk_last = None
            _idx_read = _idx_write = non_spec_state_indices_tensor

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        # 1. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        # Fast path for normal decode-only (no speculative tokens):
        # avoid extra branch/index plumbing in the generic mixed path.
        if (
            spec_sequence_masks is None
            and attn_metadata.num_prefills == 0
            and attn_metadata.num_decodes > 0
        ):
            mixed_qkv_decode = causal_conv1d_update(
                mixed_qkv,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=non_spec_state_indices_tensor[
                    : attn_metadata.num_actual_tokens
                ],
                # В 'all' свёрточное состояние ЧИТАЕТСЯ из блока последнего посчитанного
                # токена и ПИШЕТСЯ в блок последнего запланированного: на шаге, где токен
                # переходит границу блока, это разные блоки. Ядро это умеет само.
                block_idx_last_scheduled_token=(
                    _blk_last[: attn_metadata.num_actual_tokens] if mamba_all else None
                ),
                initial_state_idx=(
                    _blk_computed[: attn_metadata.num_actual_tokens]
                    if mamba_all
                    else None
                ),
                validate_data=False,
            )
            query_decode, key_decode, value_decode = self.rearrange_mixed_qkv(
                mixed_qkv_decode
            )
            if QWEN3_NEXT_FUSED_SIGMOID_GATING_ENABLED:
                core_attn_out_decode, _ = fused_sigmoid_gating_delta_rule_update(
                    A_log=self.A_log,
                    a=a,
                    b=b,
                    dt_bias=self.dt_bias,
                    q=query_decode,
                    k=key_decode,
                    v=value_decode,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=non_spec_query_start_loc[
                        : attn_metadata.num_decodes + 1
                    ],
                    # В 'all' индексы РАЗНЫЕ на чтение и на запись (переход границы блока),
                    # в остальных режимах это один и тот же тензор -- ветки в ядре нет.
                    ssm_state_indices=_idx_read,
                    ssm_state_indices_out=_idx_write if mamba_all else None,
                    use_qk_l2norm_in_kernel=True,
                )
            else:
                g_decode, beta_decode = fused_gdn_gating(
                    self.A_log, a, b, self.dt_bias
                )
                core_attn_out_decode, _ = fused_recurrent_gated_delta_rule(
                    q=query_decode,
                    k=key_decode,
                    v=value_decode,
                    g=g_decode,
                    beta=beta_decode,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=non_spec_query_start_loc[
                        : attn_metadata.num_decodes + 1
                    ],
                    ssm_state_indices=_idx_read,
                    ssm_state_indices_out=_idx_write if mamba_all else None,
                    use_qk_l2norm_in_kernel=True,
                )
            core_attn_out[:num_actual_tokens] = core_attn_out_decode.squeeze(0)
            return

        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                mixed_qkv_spec = mixed_qkv
                mixed_qkv_non_spec = None
            else:
                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)
        else:
            mixed_qkv_spec = None
            mixed_qkv_non_spec = mixed_qkv

        # 1.1: Process the multi-query part
        if spec_sequence_masks is not None and _ДЕРЕВО_W_ENV > 0:
            # [ДЕРЕВО, 05.09-2] Свёртка ветви B считается ВТОРЫМ вызовом с тем же начальным
            # окном в ЧЕРНОВОЙ слот: в основном окне между якорем и b0 лежат строки ветви A,
            # и никакой скалярный сдвиг nacc не даёт ряда [якорь, b0, b1] (§128c). Выходы
            # строк 1..W подменяют conv-выходы b-строк; при принятии B раннер копирует
            # ЧЕРНОВОЕ окно в основное, и следующий шаг идёт обычным nacc=m.
            # Интеграционный тест: tools/test_derevo_conv.py -- relL2 ~6e-08, ветви A/B, m=1..3.
            _Wд = _ДЕРЕВО_W_ENV
            _нсд2 = attn_metadata.num_spec_decodes
            _Т2 = 2 * _Wд + 1
            _окно = conv_state.shape[-1]
            _черн = getattr(self, "_дерево_черн", None)
            if (_черн is None or _черн.shape[0] < _нсд2
                    or _черн.shape[1] != conv_state.shape[1]
                    or _черн.shape[2] != _окно):
                # РАСКЛАДКА ЧЕРНОВОГО ОКНА ОБЯЗАНА ПОВТОРЯТЬ БОЕВУЮ, А НЕ ФОРМУ.
                # Боевое conv_state -- это `self_kv_cache[0].transpose(-1, -2)`, то есть
                # шаги (dim*SL, 1, dim); плотный torch.zeros даёт (dim*SL, SL, 1). Форма
                # у них одна, а ядро свёртки читает ШАГИ, и на чужой раскладке оно
                # писало верно лишь НУЛЕВОЙ канал -- по нему сверка (протокол,
                # _черн[0,0,:]) совпадала с основным окном побайтово, а max|d| по ВСЕМ
                # каналам был 27.7 при норме 67.8. Отсюда и вся порча ветви B: смысл
                # подмены верен (различитель реж=65 -- 36 «!» против 196), негодны были
                # ДАННЫЕ. Офлайн-тесты этого не ловили: там пул выделялся плотным.
                _черн = self._дерево_черн = torch.zeros(
                    max(_нсд2, 16), _окно, conv_state.shape[1],
                    dtype=conv_state.dtype, device=conv_state.device
                ).transpose(-1, -2)
                try:
                    ДЕРЕВО_OFF.setdefault("черн", {})[str(getattr(self, "prefix", id(self)))] = (_черн, conv_state)
                except Exception:
                    pass
            # ЗАЩИТА РАЗМЕРОВ: на прогреве и захвате нсд2 приходит dummy-значением и может
            # превышать число строк mixed -- index_copy тогда бьёт за пределы и оставляет в
            # потоке device-side assert, который всплывает много позже чужой синхронизацией.
            if _нсд2 * _Т2 > int(mixed_qkv_spec.shape[0]):
                _нсд2 = int(mixed_qkv_spec.shape[0]) // _Т2
            if _нсд2 <= 0:
                _нсд2 = 0
            # [ОБЩИЕ ИНДЕКСЫ НА ШАГ -- 11.09, записка 25 §298]
            # Эти тензоры зависят ТОЛЬКО от метаданных шага (число спекулятивных строк и окно
            # свёртки), но считались в КАЖДОМ из 48 слоёв GDN: ~10 мелких ядер на слой, то есть
            # ~480 лишних запусков на шаг. При средней цене мелкого ядра 4.2 мкс (опись §298)
            # это около 2.0 мс шага. Тот же класс дефекта, что `fused_gdn_gating`, который
            # считался 48 раз впустую.
            # Кэш лежит НА ОБЪЕКТЕ МЕТАДАННЫХ: строитель создаёт его заново каждый шаг
            # (`attn_metadata = GDNAttentionMetadata(...)`), поэтому устаревших значений быть не
            # может. Под полным CUDA-графом выигрыш запекается при ЗАХВАТЕ: слой 1 считает, слои
            # 2..48 читают готовое, и в графе остаётся одна копия вычисления вместо 48.
            # Умолчание 0: venv общий с боевым.
            _общ_ключ = (int(_нсд2), int(_Wд), int(_Т2))
            _общ = getattr(attn_metadata, "_fa2_общ_идx", None) if _ОБЩ_ИДX else None
            if _общ is not None and _общ[0] == _общ_ключ:
                _слот0, _ряд, _инд_B, _ряд32, _qsl_conv = _общ[1]
            else:
                _слот0 = spec_state_indices_tensor[:_нсд2, 0].to(torch.long).clamp(min=0)
                _ряд = torch.arange(max(_нсд2, 0), device=mixed_qkv_spec.device)
                _инд_B = torch.cat([
                    (_ряд * _Т2).unsqueeze(1),
                    (_ряд * _Т2).unsqueeze(1) + _Wд + 1
                    + torch.arange(_Wд, device=_ряд.device).unsqueeze(0),
                ], dim=1).reshape(-1)
                _ряд32 = _ряд.to(torch.int32)
                _qsl_conv = (torch.arange(_нсд2 + 1, device=_ряд.device,
                                          dtype=torch.int32) * (_Wд + 1))
                if _ОБЩ_ИДX:
                    try:
                        attn_metadata._fa2_общ_идx = (
                            _общ_ключ, (_слот0, _ряд, _инд_B, _ряд32, _qsl_conv))
                    except Exception:
                        pass            # кэш -- ускорение, а не условие работы
            _черн[:_нсд2].copy_(conv_state[_слот0])
            _xB = mixed_qkv_spec.index_select(0, _инд_B)
            _yB = causal_conv1d_update(
                _xB, _черн, conv_weights, self.conv1d.bias, self.activation,
                conv_state_indices=_ряд32,
                num_accepted_tokens=(getattr(attn_metadata, "num_accepted_conv", None)
                                     if getattr(attn_metadata, "num_accepted_conv", None)
                                     is not None else num_accepted_tokens),
                query_start_loc=_qsl_conv,
                max_query_len=_Wд + 1,
                validate_data=False)
            mixed_qkv_spec = causal_conv1d_update(
                mixed_qkv_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=spec_state_indices_tensor[:, 0][:_нсд2],
                num_accepted_tokens=(getattr(attn_metadata, "num_accepted_conv", None)
                                     if getattr(attn_metadata, "num_accepted_conv", None)
                                     is not None else num_accepted_tokens),
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_state_indices_tensor.size(-1),
                validate_data=False,
            )
            # ВЫРОЖДЕННЫЙ СЛУЧАЙ ДЕРЕВА: при нуле спекулятивных запросов _yB пуст,
            # и view на [0, W+1, -1] бросает ("cannot reshape tensor of 0 elements").
            # На W=1 такого шага не случалось, на W=2 он уронил подъём. Предусловие
            # ФОРМЫ обязано деградировать, а не бросать -- закон, оплаченный падением
            # боевого 16.08.
            if _нсд2 > 0 and _yB.numel() > 0:
                _инд_Bт = _инд_B.view(_нсд2, _Wд + 1)[:, 1:].reshape(-1)
                _yBт = _yB.view(_нсд2, _Wд + 1, -1)[:, 1:].reshape(_инд_Bт.shape[0], -1)
                mixed_qkv_spec = mixed_qkv_spec.index_copy(0, _инд_Bт, _yBт)
        elif spec_sequence_masks is not None:
            _страж_идx(spec_state_indices_tensor, conv_state.shape[0], 3)
            # [СВЁРТКА У ГРАНИЦЫ БЛОКА, 04.09] Спекулятивная ветка брала ОДИН слот --
            # `spec_state_indices_tensor[:, 0]`, то есть блок опоры. Опора же считается от
            # seq_len и на границе блока сдвигается, поэтому свёрточное окно на каждой границе
            # читалось из нового, ещё пустого блока. Обычный декод такого не делает: он даёт
            # ядру ПАРУ указателей (читать из блока последнего посчитанного, писать в блок
            # последнего запланированного), и ядро умеет это вместе со спекуляцией
            # (IS_APC_ENABLED и IS_SPEC_DECODING -- независимые константы шаблона).
            # Здесь та же пара и передаётся; при несдвинутой опоре обе колонки совпадают и
            # поведение в точности прежнее.
            _кб = getattr(attn_metadata, "spec_conv_блоки", None)
            _кч = getattr(attn_metadata, "spec_conv_чт", None)
            _кз = getattr(attn_metadata, "spec_conv_зап", None)
            _нсд = attn_metadata.num_spec_decodes
            # РАЗМЕР СВЕРЯЕТСЯ У ВСЕХ ТРЁХ, А НЕ ТОЛЬКО У ТАБЛИЦЫ. Указатели строятся по
            # числу строк, где метаданные сошлись, и оно бывает МЕНЬШЕ числа спекулятивных
            # запросов; ядро же берёт batch по conv_state_indices и читает указатели на всю
            # эту длину -- выход за буфер всплывал позже как illegal memory access в чужом
            # ядре (турбомайндовый тюнер GEMM), то есть за сотни строк от места причины.
            if (_кб is not None and _кч is not None and _кз is not None
                    and _кб.shape[0] >= _нсд and _кч.shape[0] >= _нсд
                    and _кз.shape[0] >= _нсд):
                mixed_qkv_spec = causal_conv1d_update(
                    mixed_qkv_spec,
                    conv_state,
                    conv_weights,
                    self.conv1d.bias,
                    self.activation,
                    conv_state_indices=_кб[:_нсд],
                    num_accepted_tokens=num_accepted_tokens,
                    query_start_loc=spec_query_start_loc,
                    max_query_len=spec_state_indices_tensor.size(-1),
                    block_idx_last_scheduled_token=_кз[:_нсд],
                    initial_state_idx=_кч[:_нсд],
                    validate_data=False,
                )
            else:
                mixed_qkv_spec = causal_conv1d_update(
                    mixed_qkv_spec,
                    conv_state,
                    conv_weights,
                    self.conv1d.bias,
                    self.activation,
                    conv_state_indices=spec_state_indices_tensor[:, 0][
                        : attn_metadata.num_spec_decodes
                    ],
                    num_accepted_tokens=num_accepted_tokens,
                    query_start_loc=spec_query_start_loc,
                    max_query_len=spec_state_indices_tensor.size(-1),
                    validate_data=False,
                )

        # 1.2: Process the remaining part
        if attn_metadata.num_prefills > 0:
            # [FA2/SM70] ФАЛЬСИФИКАТОР (FA2SM70_SKIP_CONV=1): снять причинную свёртку.
            # Ответ заведомо неверен, читается только время -- доля свёртки в стене.
            _пропуск_св = _ПРОПУСК_СВЁРТКИ
            mixed_qkv_non_spec_T = mixed_qkv_non_spec.transpose(0, 1)
            # - "cache_indices" updates the conv_state cache in positions
            #   pointed to by "state_indices_tensor"
            mixed_qkv_non_spec = mixed_qkv_non_spec if _пропуск_св else causal_conv1d_fn(
                mixed_qkv_non_spec_T,
                conv_weights,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=has_initial_state,
                # В 'all' ядру отдаётся ВСЯ таблица блоков (по всем не-спекулятивным
                # запросам -- этот вызов обрабатывает и декодные): оно само разложит
                # свёрточное состояние по границам (block_size_to_align).
                cache_indices=(_проба_записи(
                    non_spec_state_indices_tensor, _blk_computed, _blk_last, "префилл")
                    or non_spec_state_indices_tensor) if _ПРОБА_ЗАП and mamba_all
                    else non_spec_state_indices_tensor,
                block_idx_first_scheduled_token=(
                    attn_metadata.block_idx_first_scheduled_token if mamba_all else None
                ),
                block_idx_last_scheduled_token=(_blk_last if mamba_all else None),
                initial_state_idx=(_blk_computed if mamba_all else None),
                num_computed_tokens=(
                    attn_metadata.num_computed_tokens_ns if mamba_all else None
                ),
                block_size_to_align=(mamba_block_size if mamba_all else 0),
                query_start_loc=non_spec_query_start_loc,
                metadata=attn_metadata,
            ).transpose(0, 1)
        elif attn_metadata.num_decodes > 0:
            if _ПРОБА_ЗАП and mamba_all:
                _проба_записи(
                    non_spec_state_indices_tensor[: attn_metadata.num_decodes],
                    _blk_computed[: attn_metadata.num_decodes],
                    _blk_last[: attn_metadata.num_decodes], "декод")
            mixed_qkv_non_spec = causal_conv1d_update(
                mixed_qkv_non_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                # Срез оставлен ПРЕЖНИМ (по num_actual_tokens) вне режима 'all': менять его
                # заодно нельзя -- это чужая ветка, и запас индексов там намеренный.
                conv_state_indices=non_spec_state_indices_tensor[
                    : (
                        attn_metadata.num_decodes
                        if mamba_all
                        else attn_metadata.num_actual_tokens
                    )
                ],
                block_idx_last_scheduled_token=(
                    _blk_last[: attn_metadata.num_decodes] if mamba_all else None
                ),
                initial_state_idx=(
                    _blk_computed[: attn_metadata.num_decodes] if mamba_all else None
                ),
                validate_data=False,
            )
        else:
            mixed_qkv_non_spec = None

        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)
        query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(
            mixed_qkv_non_spec
        )

        # [29.08] ГЕЙТИНГ НЕ СЧИТАЕТСЯ ТАМ, ГДЕ ЕГО НИКТО НЕ ЧИТАЕТ.
        # В боевой ветке (слитый сигмоидный гейтинг + ЧИСТО спекулятивный шаг) `g`/`beta`
        # не нужны: гейтинг делается ВНУТРИ ядра рекуррента, которое берёт A_log/a/b/dt_bias
        # напрямую. При этом вызов стоял безусловно -- 48 слоёв за шаг считали и выбрасывали.
        # Условие выведено по ЧИТАТЕЛЯМ: единственные потребители -- строки со `g_spec`
        # (только при ВЫКЛЮЧЕННОМ слитом гейтинге) и `g_non_spec` (только при num_prefills>0
        # или num_decodes>0). Если ни одного читателя нет, вызов лишний по построению.
        # Гейт: побайтовое совпадение выдачи с прежним поведением (иначе читатель найдётся).
        _гейтинг_нужен = not (
            QWEN3_NEXT_FUSED_SIGMOID_GATING_ENABLED
            and spec_sequence_masks is not None
            and attn_metadata.num_prefills == 0
            and attn_metadata.num_decodes == 0
        )
        if _гейтинг_нужен:
            g, beta = fused_gdn_gating(self.A_log, a, b, self.dt_bias)
        else:
            g = beta = None

        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                g_spec = g
                beta_spec = beta
                g_non_spec = None
                beta_non_spec = None
            else:
                g_spec = g.index_select(1, spec_token_indx)
                beta_spec = beta.index_select(1, spec_token_indx)
                g_non_spec = g.index_select(1, non_spec_token_indx)
                beta_non_spec = beta.index_select(1, non_spec_token_indx)
        else:
            g_spec = None
            beta_spec = None
            g_non_spec = g
            beta_non_spec = beta

        # 2. Recurrent attention

        # 2.1: Process the multi-query part
        if spec_sequence_masks is not None and QWEN3_NEXT_FUSED_SIGMOID_GATING_ENABLED:
            # [FA2/SM70] СПЕКУЛЯТИВНЫЙ ДЕКОД -- ТОЖЕ НАШИМ ЯДРОМ.
            # Здесь стоял прямой вызов Triton, и из-за него весь декод со спекуляцией шёл
            # чужим телом: наша ветка слитого гейтинга подключена только к обычному декоду,
            # хотя ядро спекуляцию умеет (шаблон SPEC + num_accepted_tokens). Гейт и
            # рекуррент считаются одним пуском, лишний проход по состоянию не рождается.
            # [ТОЖДЕСТВЕННАЯ ВЫБОРКА -- ЭТО РАБОТА, КОТОРОЙ НЕТ. Опись пути, 25.08]
            # `indexSelectSmallIndex`: 96 пусков за шаг по 4.26 мкс = 0.41 мс. Ровно два на
            # каждый из 48 слоёв GDN -- вот эти. При ЧИСТО спекулятивном шаге (ни префилла, ни
            # обычного декода) `spec_token_indx` = 0..T-1, то есть выборка выбирает ВСЕ строки
            # подряд и является тождеством. Соседние две выборки (mixed_qkv, g/beta) этим гейтом
            # уже закрыты -- эта пара просто была пропущена.
            _всё_спек = (attn_metadata.num_prefills == 0
                         and attn_metadata.num_decodes == 0)
            if _всё_спек or spec_token_indx is None:
                _a_spec, _b_spec = a, b
            else:
                _a_spec = a.index_select(0, spec_token_indx)
                _b_spec = b.index_select(0, spec_token_indx)
            core_attn_out_spec, last_recurrent_state = (
                fused_sigmoid_gating_delta_rule_update(
                    A_log=self.A_log,
                    a=_a_spec,
                    b=_b_spec,
                    dt_bias=self.dt_bias,
                    q=query_spec,
                    k=key_spec,
                    v=value_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=spec_query_start_loc[
                        : attn_metadata.num_spec_decodes + 1
                    ],
                    ssm_state_indices=spec_state_indices_tensor,
                    # [ДЕРЕВО] чтение состояния -- по СДВИНУТОМУ nacc (+W при принятой
                    # ветви B, только при m>1); свёртка выше живёт обычным nacc.
                    num_accepted_tokens=self._дерево_ssm_nacc(
                        attn_metadata, num_accepted_tokens),
                    use_qk_l2norm_in_kernel=True,
                    branch_at=(0 if _ДЕРЕВО_БЕЗ_BRANCH else (_ДЕРЕВО_W_ENV + 1 if _ДЕРЕВО_W_ENV > 0 else 0)),
                )
            )
        elif spec_sequence_masks is not None:
            core_attn_out_spec, last_recurrent_state = fused_recurrent_gated_delta_rule(
                q=query_spec,
                k=key_spec,
                v=value_spec,
                g=g_spec,
                beta=beta_spec,
                initial_state=ssm_state,
                inplace_final_state=True,
                cu_seqlens=spec_query_start_loc[: attn_metadata.num_spec_decodes + 1],
                ssm_state_indices=spec_state_indices_tensor,
                num_accepted_tokens=num_accepted_tokens,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out_spec, last_recurrent_state = None, None

        # 2.2: Process the remaining part
        if attn_metadata.num_prefills > 0 and mamba_all:
            # РЕЖИМ 'all', ПРЕФИЛЛ. Состояние обязано быть сохранено НА КАЖДОЙ ГРАНИЦЕ
            # БЛОКА, иначе следующий запрос с тем же префиксом не сможет продолжить с
            # середины -- ради этого всё и делается. Чанковый скан отдаёт только КОНЕЧНОЕ
            # состояние, поэтому вызов режется границами блоков: конец каждого отрезка и
            # есть граница, а его конечное состояние -- то, что нужно положить в блок.
            #
            # Почему не через промежуточные состояния самого скана: у нас сшитое тело
            # (h+o+wu) НАМЕРЕННО не материализует h в общей памяти (задача 132), и вернуть
            # его -- значит отменить эту экономию на ВСЁМ префилле ради редкой границы.
            # Резка платит только лишними запусками (2-3 на чанк в 4096 токенов).
            qsl_cpu = attn_metadata.non_spec_query_start_loc_cpu
            ncomp_cpu = attn_metadata.num_computed_tokens_ns_cpu
            assert qsl_cpu is not None and ncomp_cpu is not None
            core_attn_out_non_spec = torch.empty(
                (1, query_non_spec.shape[1], value_non_spec.shape[2],
                 value_non_spec.shape[3]),
                dtype=value_non_spec.dtype,
                device=value_non_spec.device,
            )
            has_init_cpu = has_initial_state.to("cpu")
            _kva_gdn_dump(self, key_non_spec, value_non_spec, g_non_spec, beta_non_spec, {"qsl": qsl_cpu.clone(), "ncomp": ncomp_cpu.clone()})
            for i in range(qsl_cpu.numel() - 1):
                s, e = int(qsl_cpu[i]), int(qsl_cpu[i + 1])
                if e <= s:
                    continue
                C = int(ncomp_cpu[i])          # сколько токенов запроса уже посчитано
                # Начальное состояние -- из блока последнего ПОСЧИТАННОГО токена.
                if bool(has_init_cpu[i]):
                    st = _чит(ssm_state, _idx_read[i]).unsqueeze(0).contiguous()
                else:
                    st = torch.zeros(
                        (1, *ssm_state.shape[1:]),
                        dtype=_вещ(ssm_state), device=ssm_state.device)
                # Границы блоков внутри этого куска, в локальных координатах.
                резы = []
                гр = (C // mamba_block_size + 1) * mamba_block_size
                while гр < C + (e - s):
                    резы.append(гр - C)
                    гр += mamba_block_size
                участки = [0] + резы + [e - s]
                for j in range(len(участки) - 1):
                    a0, a1 = s + участки[j], s + участки[j + 1]
                    cu = torch.tensor(
                        [0, a1 - a0], dtype=torch.int32, device=query_non_spec.device
                    )
                    o_seg, st = chunk_gated_delta_rule(
                        q=query_non_spec[:, a0:a1],
                        k=key_non_spec[:, a0:a1],
                        v=value_non_spec[:, a0:a1],
                        g=g_non_spec[:, a0:a1],
                        beta=beta_non_spec[:, a0:a1],
                        initial_state=st,
                        output_final_state=True,
                        cu_seqlens=cu,
                        head_first=False,
                        use_qk_l2norm_in_kernel=True,
                    )
                    core_attn_out_non_spec[:, a0:a1] = o_seg
                    # Конец отрезка -- либо граница блока, либо конец куска. В обоих случаях
                    # состояние принадлежит блоку, в котором лежит ПОСЛЕДНИЙ его токен.
                    блок = (C + участки[j + 1] - 1) // mamba_block_size
                    _пиш(ssm_state, non_spec_state_indices_tensor[i, блок], st[0])
            last_recurrent_state = None
        elif attn_metadata.num_prefills > 0:
            _kva_gdn_dump(self, key_non_spec, value_non_spec, g_non_spec, beta_non_spec)
            initial_state = _чит(
                ssm_state, non_spec_state_indices_tensor).contiguous()
            initial_state[~has_initial_state, ...] = 0
            (
                core_attn_out_non_spec,
                last_recurrent_state,
            ) = chunk_gated_delta_rule(
                q=query_non_spec,
                k=key_non_spec,
                v=value_non_spec,
                g=g_non_spec,
                beta=beta_non_spec,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=non_spec_query_start_loc,
                head_first=False,
                use_qk_l2norm_in_kernel=True,
            )
            # Init cache
            _пиш(ssm_state, non_spec_state_indices_tensor, last_recurrent_state)
        elif attn_metadata.num_decodes > 0:
            core_attn_out_non_spec, last_recurrent_state = (
                fused_recurrent_gated_delta_rule(
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    g=g_non_spec,
                    beta=beta_non_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=non_spec_query_start_loc[
                        : attn_metadata.num_decodes + 1
                    ],
                    ssm_state_indices=_idx_read,
                    ssm_state_indices_out=_idx_write if mamba_all else None,
                    use_qk_l2norm_in_kernel=True,
                )
            )
        else:
            core_attn_out_non_spec, last_recurrent_state = None, None

        # 3. Merge core attention output
        if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
            merged_out = torch.empty(
                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                dtype=core_attn_out_non_spec.dtype,
                device=core_attn_out_non_spec.device,
            )
            merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
            merged_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
            core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
        elif spec_sequence_masks is not None:
            core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
        else:
            core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)


class Qwen3NextAttention(nn.Module):
    def __init__(
        self,
        config: Qwen3NextConfig,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim or (self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        self.attn_output_gate = getattr(config, "attn_output_gate", True)

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=getattr(config, "qkv_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
            dual_chunk_attention_config=self.dual_chunk_attention_config,
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            **{
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config": self.dual_chunk_attention_config,
            }
            if self.dual_chunk_attention_config
            else {},
        )

        self.q_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
    ):
        qkv_proj = _maybe_sm70_projection(self.qkv_proj, hidden_states)
        if qkv_proj is None:
            qkv_proj = self.qkv_proj(hidden_states)
        qkv, _ = qkv_proj

        if self.attn_output_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
            )
            orig_shape = q_gate.shape[:-1]
            q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
            q = q.reshape(*orig_shape, -1)
            gate = gate.reshape(*orig_shape, -1)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(
            -1, self.num_heads * self.head_dim
        )
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(
            -1, self.num_kv_heads * self.head_dim
        )

        q, k = self.rotary_emb(positions, q, k)

        # [FA2/SM70] ФАЛЬСИФИКАТОР СНЯТИЕМ ФАЗЫ (FA2SM70_SKIP_FULLATTN=1). Ответ при этом
        # ЗАВЕДОМО НЕВЕРЕН -- читается ТОЛЬКО время: разность со штатным прогоном и есть
        # доля полного внимания в стене префилла. Ставится здесь, а не в бэкенде, чтобы
        # снять ВСЮ фазу вместе с чтением страниц KV, а не одно умножение.
        if _ПРОПУСК_ВНИМАНИЯ:
            attn_output = torch.zeros_like(q)
        else:
            attn_output = self.attn(q, k, v)

        if self.attn_output_gate:
            gate = torch.sigmoid(gate)
            attn_output = attn_output * gate

        o_proj = _maybe_sm70_projection(self.o_proj, attn_output)
        if o_proj is None:
            o_proj = self.o_proj(attn_output)
        output[:], _ = o_proj


class Qwen3NextDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        speculative_config = vllm_config.speculative_config

        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)

        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3NextGatedDeltaNet(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                speculative_config=speculative_config,
                prefix=f"{prefix}.linear_attn",
            )
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")

        mlp_only_layers = (
            [] if not hasattr(config, "mlp_only_layers") else config.mlp_only_layers
        )
        if (self.layer_idx not in mlp_only_layers) and (
            config.num_experts > 0
            and (self.layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3NextSparseMoeBlock(
                vllm_config=vllm_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = Qwen3NextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3NextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_scale = getattr(config, "layer_scale", False)
        if self.layer_scale:
            self.attn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                    dtype=config.dtype,
                ),
            )
            self.ffn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                    dtype=config.dtype,
                ),
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor = None,
        **kwargs: object,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        self_attention_output = torch.empty_like(hidden_states)
        if self.layer_type == "linear_attention":
            self.linear_attn(
                hidden_states=hidden_states,
                output=self_attention_output,
            )
        elif self.layer_type == "full_attention":
            self.self_attn(
                hidden_states=hidden_states,
                output=self_attention_output,
                positions=positions,
            )
        else:
            raise ValueError("Invalid layer_type")
        hidden_states = self_attention_output

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * self._attn_scale_p1
            else:
                hidden_states = hidden_states * self._attn_scale_p1.unsqueeze(0)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # [FA2/SM70 27.08] ФАЛЬСИФИКАТОР СНЯТИЕМ ФАЗЫ MLP (FA2SM70_FALSE_MLP=1).
        # Ответ заведомо неверен, читается ТОЛЬКО время. Нужен, чтобы разложить остаток
        # наклона по позициям: обмен даёт 0.90 мс/позицию, GDN 0.53, а 0.90 ещё не отнесены
        # ни к чему, при том что внимание даёт лишь 0.07.
        if not FA2SM70_FALSE_MLP:
            hidden_states = self.mlp(hidden_states)
        else:
            global _FALSE_MLP_СЧЁТ
            _FALSE_MLP_СЧЁТ += 1
            if _FALSE_MLP_СЧЁТ in (1, 1000, 100000):
                print(f"[fa2_sm70] ФАЛЬСИФИКАТОР MLP сработал {_FALSE_MLP_СЧЁТ} раз",
                      file=__import__("sys").stderr, flush=True)

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * self._ffn_scale_p1
            else:
                hidden_states = hidden_states * self._ffn_scale_p1.unsqueeze(0)

        return hidden_states, residual


@support_torch_compile
class Qwen3NextModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config: Qwen3NextConfig = vllm_config.model_config.hf_config
        parallel_config = vllm_config.parallel_config

        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.config = config

        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        def get_layer(prefix: str):
            return Qwen3NextDecoderLayer(
                vllm_config,
                layer_type=config.layer_types[extract_layer_index(prefix)],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        if get_pp_group().is_last_rank:
            self.norm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        # [FA2/SM70 27.08] ПОЛОЖИТЕЛЬНЫЙ КОНТРОЛЬ ПРИБОРА ФАЗ (FA2SM70_PHASE_PROBE_MS=N).
        # Вставляет в проход ЗАВЕДОМО ИЗВЕСТНУЮ задержку. Прибор обязан показать РОВНО её
        # прибавку в фазе «прямой проход цели». Без такого контроля показания прибора не
        # доказывают ничего -- сегодня слепой гейт сведения стоил семи итераций отладки.
        if FA2SM70_PHASE_PROBE_MS:
            torch.cuda._sleep(int(FA2SM70_PHASE_PROBE_MS * 1.4e6))

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            # [FA2/SM70 13.09, ПРИБОР KVA] Дамп потока (hidden + residual) на входе слоя FA2SM70_KVA_SPLIT через custom op
            # (записка 26 §6); условия по константам -- при трассировке, проверка флага ON -- внутри op во время работы.
            if _KVA_DIR and getattr(layer, "layer_idx", -1) == _KVA_SPLIT and residual is not None:
                hidden_states = torch.ops.fa2sm70.kva_dump(hidden_states, residual, positions, _KVA_SPLIT)
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return SharedFusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            # Dense variants (e.g. Qwen3.5-27B) do not define num_experts.
            num_experts=getattr(self.config, "num_experts", 0),
            num_redundant_experts=self.num_redundant_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if name.startswith("mtp."):
                continue

            # Remapping the name of FP8 kv-scale.
            if name.endswith("scale"):
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                if "mlp.experts" in name:
                    continue

                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                # name = apply_attn_prefix(name, params_dict)
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias") or name.endswith("_bias")
                    ) and name not in params_dict:
                        continue
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        logger.warning_once(
                            f"Parameter {name} not found in params_dict, skip loading"
                        )
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)

        # Pre-compute layer_scale + 1 to reduce CUDA graph kernel count.
        # Eliminates per-forward dtype conversion, indexing, and scalar add.
        for layer in self.layers:
            if getattr(layer, 'layer_scale', False):
                with torch.no_grad():
                    # [1, 1, hidden] -> [1, hidden] for 2D decode path
                    layer._attn_scale_p1 = (
                        layer.attn_layer_scale + 1).squeeze(0).clone()
                    layer._ffn_scale_p1 = (
                        layer.ffn_layer_scale + 1).squeeze(0).clone()

        return loaded_params


class QwenNextMixtureOfExperts(MixtureOfExperts):
    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.model.layers:
            if isinstance(layer.mlp, Qwen3NextSparseMoeBlock):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def set_moe_parameters(self):
        self.expert_weights = []

        self.moe_layers = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, Qwen3NextDecoderLayer) and isinstance(
                layer.mlp, Qwen3NextSparseMoeBlock
            ):
                example_moe = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_moe is None:
            raise RuntimeError("No Qwen3Next layer found in the model.layers.")

        # Set MoE hyperparameters
        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_redundant_experts = example_moe.n_redundant_experts


class Qwen3NextForCausalLM(
    nn.Module,
    HasInnerState,
    SupportsLoRA,
    SupportsPP,
    QwenNextMixtureOfExperts,
    IsHybrid,
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config

        scheduler_config = vllm_config.scheduler_config
        # [FA2/SM70, задача 194] Запрет СНЯТ: режим 'all' на нашем пути реализован целиком --
        # метаданные (gdn_attn.py: mamba_block_size + три индекса блоков), декод (наш
        # volta_gdn_rec принимает ssm_state_indices_out -- запись в блок, отличный от блока
        # чтения) и префилл (сегментация куска по границам блоков с записью состояния в блок
        # последнего токена каждого отрезка). Это единственная настоящая починка пары
        # MTP + префикс-кэш: в 'align' движок копирует состояние между слотами мимо нашей
        # таблицы масштабов, из-за чего int16-состояние и спекуляция с кэшем не складывались.
        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = Qwen3NextModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        # Set MoE hyperparameters
        self.set_moe_parameters()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

        return hidden_states

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        config = vllm_config.model_config.hf_config
        parallel_config = vllm_config.parallel_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            config.linear_num_key_heads,
            config.linear_num_value_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim,
            config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor.get_top_tokens(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["mtp."],
        )
        return loader.load_weights(weights)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()


def gdn_attention_core(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
) -> None:
    """
    Custom op for the core attention computation.
    Only handles the convolution + recurrent attention part.
    Input/output projections are handled outside this op.
    """
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self._forward_core(
        mixed_qkv=mixed_qkv,
        b=b,
        a=a,
        core_attn_out=core_attn_out,
    )


def gdn_attention_core_fake(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
) -> None:
    """Fake implementation for torch.compile."""
    return


direct_register_custom_op(
    op_name="gdn_attention_core",
    op_func=gdn_attention_core,
    mutates_args=["core_attn_out"],
    fake_impl=gdn_attention_core_fake,
)


@triton.jit
def fused_gdn_gating_kernel(
    g,
    beta_output,
    A_log,
    a,
    b,
    dt_bias,
    seq_len,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
):
    i_b, i_s, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
    off = i_b * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off
    mask = head_off < NUM_HEADS
    blk_A_log = tl.load(A_log + head_off, mask=mask)
    blk_a = tl.load(a + off, mask=mask)
    blk_b = tl.load(b + off, mask=mask)
    blk_bias = tl.load(dt_bias + head_off, mask=mask)
    # If the model is loaded in fp16, without the .float() here, A might be -inf
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = tl.where(
        beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
    )
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)
    # compute beta_output = sigmoid(b)
    blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))
    tl.store(
        beta_output + off, blk_beta_output.to(beta_output.dtype.element_ty), mask=mask
    )


def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused computation of g and beta for Gated Delta Net.
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    beta_output = b.sigmoid()
    TODO maybe use torch.compile to replace this triton kernel
    """
    batch, num_heads = a.shape
    seq_len = 1
    grid = (batch, seq_len, triton.cdiv(num_heads, 8))
    g = torch.empty(1, batch, num_heads, dtype=torch.float32, device=a.device)
    beta_output = torch.empty(1, batch, num_heads, dtype=b.dtype, device=b.device)
    fused_gdn_gating_kernel[grid](
        g,
        beta_output,
        A_log,
        a,
        b,
        dt_bias,
        seq_len,
        num_heads,
        beta,
        threshold,
        8,
        num_warps=1,
    )
    return g, beta_output
