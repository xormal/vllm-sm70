# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash Attention V100 backend for SM70.

Prefill uses the dense Flash V100 kernel for strict no-prefix cases.
Decode uses a paged Flash V100 kernel that reads vLLM's KV cache directly.
"""

from __future__ import annotations

import os
import torch

from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionCGSupport, AttentionType
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadata,
    TritonAttentionMetadataBuilder,
)

logger = init_logger(__name__)

# Lazy imports: only resolve optional CUDA extensions when needed.
_flash_attn_func = None
_flash_attn_decode_paged = None
_flash_attn_prefill_paged = None
_paged_kv_utils = None
_warned_prefill_fallback = False
_warned_feature_fallback = False
_warned_decode_fallback = False
_logged_prefill_flash = False
_logged_prefill_prefix_flash = False
_logged_decode_flash = False
_logged_prefill_compare = False


def _get_flash_ops():
    """Lazy-load flash_attn_v100 ops if available."""
    global _flash_attn_func, _flash_attn_decode_paged, _flash_attn_prefill_paged
    if (_flash_attn_func is None or _flash_attn_decode_paged is None
            or _flash_attn_prefill_paged is None):
        try:
            from flash_attn_v100 import (flash_attn_decode_paged,
                                         flash_attn_func,
                                         flash_attn_prefill_paged)

            _flash_attn_func = flash_attn_func
            _flash_attn_decode_paged = flash_attn_decode_paged
            _flash_attn_prefill_paged = flash_attn_prefill_paged
        except ImportError:
            _flash_attn_func = None
            _flash_attn_decode_paged = None
            _flash_attn_prefill_paged = None
    return _flash_attn_func, _flash_attn_decode_paged, _flash_attn_prefill_paged


def _get_paged_kv_utils():
    """Lazy-load paged KV extraction CUDA extension."""
    global _paged_kv_utils
    if _paged_kv_utils is None:
        try:
            import paged_kv_utils

            _paged_kv_utils = paged_kv_utils
        except ImportError:
            _paged_kv_utils = None
    return _paged_kv_utils


_CALIB_SCALES = None          # {layer_name: (k_scale, v_scale)}
_CALIB_DEFAULT = (1.0, 1.0)   # conservative default for layers not in the file


def _load_calib_scales():
    """Load offline-calibrated static KV scales from KV_SCALES_FILE (once)."""
    global _CALIB_SCALES, _CALIB_DEFAULT
    if _CALIB_SCALES is not None:
        return
    _CALIB_SCALES = {}
    path = os.environ.get("KV_SCALES_FILE")
    if not path or not os.path.exists(path):
        return
    try:
        import json
        d = json.load(open(path))
        ks, vs = [], []
        for ln, e in d.items():
            k = float(e["k_scale"]); v = float(e["v_scale"])
            if k > 0 and v > 0:
                _CALIB_SCALES[ln] = (k, v); ks.append(k); vs.append(v)
        if ks:
            # Uncovered layers (e.g. layer 0) get the max scale -> no clipping.
            _CALIB_DEFAULT = (max(ks), max(vs))
        logger.info("FLASH_ATTN_V100: loaded %d calibrated KV scales from %s "
                    "(default for uncovered=%s)", len(_CALIB_SCALES), path, _CALIB_DEFAULT)
    except Exception as exc:
        logger.warning("FLASH_ATTN_V100: failed to load KV scales from %s: %s", path, exc)


def _kv_scales(layer):
    """Return (k_scale, v_scale): calibrated static if available, else the
    layer's own runtime scale."""
    _load_calib_scales()
    if _CALIB_SCALES:
        ln = getattr(layer, "layer_name", None)
        k, v = _CALIB_SCALES.get(ln, _CALIB_DEFAULT)
        return float(k), float(v)
    return float(layer._k_scale_float), float(layer._v_scale_float)


def _has_prefix_context(attn_metadata: TritonAttentionMetadata) -> bool:
    """Return True if any sequence has KV context before current query tokens."""
    query_lens = attn_metadata.query_start_loc[1:] - attn_metadata.query_start_loc[:-1]
    return not torch.equal(query_lens, attn_metadata.seq_lens)


def _extract_contiguous_kv_from_paged_cache(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    total_tokens: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract contiguous K/V from paged KV cache.

    Uses the CUDA extension when available and falls back to a Python path.
    """

    paged_kv_utils = _get_paged_kv_utils()

    if isinstance(kv_cache, (list, tuple)):
        key_cache, value_cache = kv_cache[0], kv_cache[1]
    else:
        if kv_cache.shape[0] == 2:
            key_cache, value_cache = kv_cache.unbind(0)
        elif kv_cache.shape[1] == 2:
            key_cache, value_cache = kv_cache.unbind(1)
        else:
            raise ValueError(
                f"Unexpected KV cache shape {tuple(kv_cache.shape)}; "
                "expected dimension 2 at axis 0 or 1"
            )

    if paged_kv_utils is not None and key_cache.dtype != torch.uint8:
        if hasattr(paged_kv_utils, "paged_kv_to_contiguous"):
            k_cont, v_cont = paged_kv_utils.paged_kv_to_contiguous(
                key_cache, value_cache, block_table, seq_lens)
        else:
            k_cont = paged_kv_utils.paged_to_contiguous(key_cache, block_table,
                                                        seq_lens)
            v_cont = paged_kv_utils.paged_to_contiguous(value_cache, block_table,
                                                        seq_lens)
        if total_tokens is None:
            total_tokens = int(seq_lens.sum().item())
        return k_cont[:total_tokens], v_cont[:total_tokens]

    # Slow Python fallback.
    batch_size = block_table.shape[0]
    if total_tokens is None:
        total_tokens = int(seq_lens.sum().item())

    k_cont = torch.empty(
        (total_tokens, num_kv_heads, head_dim),
        dtype=key_cache.dtype,
        device=key_cache.device,
    )
    v_cont = torch.empty(
        (total_tokens, num_kv_heads, head_dim),
        dtype=value_cache.dtype,
        device=value_cache.device,
    )

    token_offset = 0
    for batch_idx in range(batch_size):
        seq_len = int(seq_lens[batch_idx].item())
        if seq_len == 0:
            continue

        num_blocks = (seq_len + block_size - 1) // block_size
        for block_idx in range(num_blocks):
            physical_block_idx = int(block_table[batch_idx, block_idx].item())
            start_token = block_idx * block_size
            end_token = min(start_token + block_size, seq_len)
            n = end_token - start_token

            k_cont[token_offset:token_offset + n] = key_cache[physical_block_idx, :n]
            v_cont[token_offset:token_offset + n] = value_cache[physical_block_idx, :n]
            token_offset += n

    return k_cont, v_cont


def _fp8_dtype_from_cache_dtype(kv_cache_dtype: str) -> torch.dtype:
    if kv_cache_dtype in ("fp8", "fp8_e4m3"):
        return torch.float8_e4m3fn
    if kv_cache_dtype == "fp8_e5m2":
        return torch.float8_e5m2
    raise ValueError(f"Unsupported FLASH_ATTN_V100 fp8 dtype: {kv_cache_dtype}")


def _dequantize_fp8_contiguous_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: float,
    v_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not kv_cache_dtype.startswith("fp8"):
        return key, value
    fp8_dtype = _fp8_dtype_from_cache_dtype(kv_cache_dtype)
    key = key.view(fp8_dtype).to(torch.float16) * k_scale
    value = value.view(fp8_dtype).to(torch.float16) * v_scale
    return key, value


class FlashAttnV100MetadataBuilder(TritonAttentionMetadataBuilder):
    """Attach CPU metadata for the dense prefill path."""

    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def build(self, common_prefix_len, common_attn_metadata, fast_build: bool = False):
        attn_metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        attn_metadata.query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        attn_metadata.seq_lens_cpu = common_attn_metadata.seq_lens_cpu
        return attn_metadata


class FlashAttnV100Impl(TritonAttentionImpl):
    """Flash Attention V100 implementation with strict fallback policy."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        (self.flash_attn_func, self.flash_attn_decode_paged,
         self.flash_attn_prefill_paged) = _get_flash_ops()
        # V100 FA2 kernels consume fp16 Q. FP8 KV cache support is implemented
        # as storage compression only, with K/V dequantized inside FA2 kernels.
        self.supports_quant_query_input = False
        self.use_flash_v100 = self.flash_attn_func is not None
        self.use_flash_v100_decode = self.flash_attn_decode_paged is not None
        paged_prefill_enable = os.getenv("VLLM_FLASH_V100_ENABLE_PAGED_PREFILL")
        paged_prefill_disable = (
            os.getenv("VLLM_FLASH_V100_DISABLE_PAGED_PREFILL", "0") == "1")
        self.use_flash_v100_prefill_paged = (
            self.flash_attn_prefill_paged is not None
            and paged_prefill_enable == "1"
            and not paged_prefill_disable)
        self._decode_cache_k: torch.Tensor | None = None
        self._decode_cache_v: torch.Tensor | None = None
        self._decode_cache_len = 0
        self._decode_cache_capacity = 0

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        # SM70 fp8 KV write: Triton can't emit e4m3fn stores on Volta, so route
        # fp8 writes through the custom CUDA kernel (unblocks e4m3). Non-fp8
        # (auto/fp16) and encoder/kv-sharing cases defer to the parent.
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return
        if not self.kv_cache_dtype.startswith("fp8"):
            return super().do_kv_cache_update(layer, key, value, kv_cache,
                                              slot_mapping)
        if (self.kv_sharing_target_layer_name is not None
                or key is None or value is None):
            return
        try:
            from flash_attn_v100 import reshape_and_cache_fp8
        except ImportError:
            return super().do_kv_cache_update(layer, key, value, kv_cache,
                                              slot_mapping)
        key_cache, value_cache = kv_cache.unbind(1)  # uint8 paged caches
        # allow_i8: the private int8+per-position-scale layout is legal ONLY while every reader of
        # these bytes is ours. If this layer's attention does not run on our path (sliding window,
        # alibi, soft-cap ...), Triton reads the same bytes as e4m3 and the output is garbage. Gemma-4
        # has 40 such layers out of 48. The pool stays mixed on purpose -- the shim tags the format per
        # cache. e4m3 is still written by our kernel here, because Triton cannot emit e4m3 on Volta.
        reshape_and_cache_fp8(
            key, value, key_cache, value_cache, slot_mapping,
            self.kv_cache_dtype,
            _kv_scales(layer)[0], _kv_scales(layer)[1],
            allow_i8=self._supports_flash_v100_path(),
        )

    def _reset_decode_cache(self) -> None:
        self._decode_cache_k = None
        self._decode_cache_v = None
        self._decode_cache_len = 0
        self._decode_cache_capacity = 0

    def _ensure_decode_cache_capacity(
        self,
        required_len: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if (
            self._decode_cache_k is not None
            and self._decode_cache_v is not None
            and self._decode_cache_capacity >= required_len
            and self._decode_cache_k.shape[1] == num_kv_heads
            and self._decode_cache_k.shape[2] == head_dim
            and self._decode_cache_k.dtype == dtype
            and self._decode_cache_k.device == device
        ):
            return

        new_capacity = max(required_len, max(16, self._decode_cache_capacity * 2))
        new_k = torch.empty(
            (new_capacity, num_kv_heads, head_dim),
            dtype=dtype,
            device=device,
        )
        new_v = torch.empty(
            (new_capacity, num_kv_heads, head_dim),
            dtype=dtype,
            device=device,
        )

        if (
            self._decode_cache_k is not None
            and self._decode_cache_v is not None
            and self._decode_cache_len > 0
        ):
            new_k[:self._decode_cache_len].copy_(
                self._decode_cache_k[:self._decode_cache_len]
            )
            new_v[:self._decode_cache_len].copy_(
                self._decode_cache_v[:self._decode_cache_len]
            )

        self._decode_cache_k = new_k
        self._decode_cache_v = new_v
        self._decode_cache_capacity = new_capacity

    def _get_decode_kv_single_seq(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        seq_lens_cpu: torch.Tensor,
        block_size: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = int(seq_lens_cpu[0])
        q_len = int(attn_metadata.num_actual_tokens)
        num_kv_heads = key.shape[1]

        cache_hit = (
            self._decode_cache_k is not None
            and self._decode_cache_v is not None
            and seq_len > self._decode_cache_len
            and seq_len - q_len == self._decode_cache_len
        )

        if not cache_hit:
            k_cont, v_cont = _extract_contiguous_kv_from_paged_cache(
                kv_cache=kv_cache,
                block_table=attn_metadata.block_table,
                seq_lens=attn_metadata.seq_lens,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                block_size=block_size,
                total_tokens=seq_len,
            )
            self._ensure_decode_cache_capacity(
                seq_len,
                num_kv_heads,
                head_dim,
                k_cont.dtype,
                k_cont.device,
            )
            assert self._decode_cache_k is not None
            assert self._decode_cache_v is not None
            self._decode_cache_k[:seq_len].copy_(k_cont)
            self._decode_cache_v[:seq_len].copy_(v_cont)
            self._decode_cache_len = seq_len
            return (
                self._decode_cache_k[:seq_len],
                self._decode_cache_v[:seq_len],
            )

        self._ensure_decode_cache_capacity(
            seq_len,
            num_kv_heads,
            head_dim,
            key.dtype,
            key.device,
        )
        assert self._decode_cache_k is not None
        assert self._decode_cache_v is not None
        self._decode_cache_k[self._decode_cache_len:seq_len].copy_(key[:q_len])
        self._decode_cache_v[self._decode_cache_len:seq_len].copy_(value[:q_len])
        self._decode_cache_len = seq_len
        return (
            self._decode_cache_k[:seq_len],
            self._decode_cache_v[:seq_len],
        )

    def _supports_flash_v100_path(self) -> bool:
        """Check whether current layer/config can run Flash V100 safely."""
        supported_kv_dtype = (
            not self.kv_cache_dtype.startswith("fp8")
            or self.kv_cache_dtype in ("fp8", "fp8_e4m3", "fp8_e5m2")
        )
        return (
            self.use_flash_v100
            and self.attn_type == AttentionType.DECODER
            and self.alibi_slopes is None
            and self.logits_soft_cap == 0
            and self.sinks is None
            and self.sliding_window == (-1, -1)
            and supported_kv_dtype
        )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward path.

        - Prefill: use dense Flash V100 only when there is no prefix context.
        - Decode: use paged Flash V100 when available, otherwise fall back.
        """
        global _logged_decode_flash, _logged_prefill_flash
        global _logged_prefill_prefix_flash
        global _warned_decode_fallback, _warned_prefill_fallback
        global _warned_feature_fallback

        if attn_metadata is None:
            assert output is not None
            return output.fill_(0)

        if not self._supports_flash_v100_path():
            if self.use_flash_v100 and not _warned_feature_fallback:
                logger.warning(
                    "FLASH_ATTN_V100 fallback to Triton due to unsupported "
                    "attention features or KV cache dtype."
                )
                _warned_feature_fallback = True
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        is_prefill = attn_metadata.max_query_len > 1
        is_capturing = query.is_cuda and torch.cuda.is_current_stream_capturing()

        if is_prefill:
            if is_capturing:
                if not _warned_prefill_fallback:
                    logger.warning(
                        "FLASH_ATTN_V100 prefill fallback during CUDA graph "
                        "capture. Using Triton path for capture safety."
                    )
                    _warned_prefill_fallback = True
                return super().forward(
                    layer,
                    query,
                    key,
                    value,
                    kv_cache,
                    attn_metadata,
                    output,
                    output_scale,
                    output_block_scale,
                )
            if _has_prefix_context(attn_metadata):
                if not _logged_prefill_prefix_flash:
                    if self.use_flash_v100_prefill_paged:
                        logger.info(
                            "FLASH_ATTN_V100 prefill path active "
                            "(prefix/chunked via direct paged prefill kernel)."
                        )
                    else:
                        logger.info(
                            "FLASH_ATTN_V100 prefill path active "
                            "(prefix/chunked via paged-KV gather)."
                        )
                    _logged_prefill_prefix_flash = True
                self._reset_decode_cache()
                return self._flash_v100_prefill_with_prefix(
                    layer,
                    query,
                    kv_cache,
                    attn_metadata,
                    output,
                )
            if not _logged_prefill_flash:
                logger.info(
                    "FLASH_ATTN_V100 prefill path active (no prefix/chunked context)."
                )
                _logged_prefill_flash = True
            self._reset_decode_cache()
            return self._flash_v100_prefill(query, key, value, attn_metadata, output)

        if not self.use_flash_v100_decode:
            if self.use_flash_v100 and not _warned_decode_fallback:
                logger.warning(
                    "FLASH_ATTN_V100 decode fallback to Triton: paged decode op "
                    "is unavailable."
                )
                _warned_decode_fallback = True
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        if not _logged_decode_flash:
            logger.info(
                "FLASH_ATTN_V100 decode path active (paged KV kernel, CUDA-graph safe)."
            )
            _logged_decode_flash = True
        return self._flash_v100_decode(
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
        )

    def _flash_v100_prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Prefill path for no-prefix case (query_len == seq_len per sequence)."""
        num_actual_tokens = attn_metadata.num_actual_tokens
        query = query[:num_actual_tokens]
        key = key[:num_actual_tokens]
        value = value[:num_actual_tokens]
        out_view = output[:num_actual_tokens]

        query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
        query_start_loc = (
            query_start_loc_cpu if query_start_loc_cpu is not None else attn_metadata.query_start_loc
        )
        num_seqs = len(query_start_loc) - 1

        if num_seqs == 0:
            return output

        seq_lens = query_start_loc[1:] - query_start_loc[:-1]
        run_start = 0
        while run_start < num_seqs:
            run_seq_len = int(seq_lens[run_start].item())
            run_end = run_start + 1
            while (
                run_end < num_seqs
                and int(seq_lens[run_end].item()) == run_seq_len
            ):
                run_end += 1

            if run_seq_len > 0:
                tok_start = int(query_start_loc[run_start].item())
                tok_end = int(query_start_loc[run_end].item())
                batch_size = run_end - run_start

                q_batch = query[tok_start:tok_end].view(
                    batch_size, run_seq_len, query.shape[1], query.shape[2]
                )
                k_batch = key[tok_start:tok_end].view(
                    batch_size, run_seq_len, key.shape[1], key.shape[2]
                )
                v_batch = value[tok_start:tok_end].view(
                    batch_size, run_seq_len, value.shape[1], value.shape[2]
                )

                out_batch = self.flash_attn_func(
                    q_batch,
                    k_batch,
                    v_batch,
                    causal=True,
                    softmax_scale=self.scale,
                )
                out_view[tok_start:tok_end].copy_(
                    out_batch.view(tok_end - tok_start, out_batch.shape[2], out_batch.shape[3])
                )

            run_start = run_end

        return output

    def _flash_v100_decode(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Decode path using Flash V100 directly over paged KV cache."""
        num_actual_tokens = attn_metadata.num_actual_tokens
        query = query[:num_actual_tokens]
        out_view = output[:num_actual_tokens]

        if query.shape[0] == 0:
            return output

        if kv_cache.shape[0] == 2:
            key_cache, value_cache = kv_cache.unbind(0)
        else:
            key_cache, value_cache = kv_cache.unbind(1)

        self.flash_attn_decode_paged(
            query,
            key_cache,
            value_cache,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            softmax_scale=self.scale,
            out=out_view,
            kv_cache_dtype=self.kv_cache_dtype,
            k_scale=_kv_scales(layer)[0],
            v_scale=_kv_scales(layer)[1],
        )
        return output

    def _fa2sm70_prefill_kwargs(self) -> dict:
        """[fa2_sm70 patch] Optional kwargs of the loaded paged-prefill op, probed ONCE.

        `out=`         -- the op writes straight into the engine's output buffer (our kernels take
                          output strides), which removes a full copy of the attention output per
                          layer per chunk. Kill switch: VLLM_FLASH_V100_NO_DIRECT_OUT=1.
        `seq_lens_cpu=`-- the sequence length from the host, removing one device->host sync per
                          layer per chunk (the op otherwise reads it off a CUDA tensor).
        Both are OPTIONAL on purpose: an older op without them keeps the previous behaviour, and the
        pointer test at the bottom of the loop still copies when `out` was not honoured.
        """
        kw = getattr(self, "_fa2sm70_prefill_kw", None)
        if kw is None:
            import inspect
            try:
                params = inspect.signature(self.flash_attn_prefill_paged).parameters
            except (TypeError, ValueError):
                params = {}
            kw = {
                "out": ("out" in params)
                and os.getenv("VLLM_FLASH_V100_NO_DIRECT_OUT", "0") != "1",
                "seq_lens_cpu": ("seq_lens_cpu" in params)
                and os.getenv("VLLM_FLASH_V100_NO_CPU_SEQLENS", "0") != "1",
            }
            self._fa2sm70_prefill_kw = kw
            logger.info("FLASH_ATTN_V100 paged prefill: direct out=%s, host seq_lens=%s",
                        kw["out"], kw["seq_lens_cpu"])
        return kw

    def _flash_v100_prefill_with_prefix(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Prefill path for prefix/chunked context via gathered contiguous KV."""
        global _logged_prefill_compare
        num_actual_tokens = attn_metadata.num_actual_tokens
        query = query[:num_actual_tokens]
        out_view = output[:num_actual_tokens]

        query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
        query_start_loc = (
            query_start_loc_cpu
            if query_start_loc_cpu is not None
            else attn_metadata.query_start_loc
        )
        seq_lens_cpu = getattr(attn_metadata, "seq_lens_cpu", None)
        seq_lens = seq_lens_cpu if seq_lens_cpu is not None else attn_metadata.seq_lens
        num_seqs = len(query_start_loc) - 1

        if kv_cache.shape[0] == 2:
            key_cache, value_cache = kv_cache.unbind(0)
        else:
            key_cache, value_cache = kv_cache.unbind(1)
        block_size = key_cache.shape[1]
        num_kv_heads = key_cache.shape[2]
        head_dim = key_cache.shape[3]
        debug_compare = (os.getenv("VLLM_FLASH_V100_DEBUG_PREFILL_COMPARE", "0")
                         == "1")

        # [fa2_sm70 patch] Which optional kwargs the loaded prefill op understands (probed once).
        fa2_kw = self._fa2sm70_prefill_kwargs()

        for i in range(num_seqs):
            start = int(query_start_loc[i].item())
            end = int(query_start_loc[i + 1].item())
            if end <= start:
                continue

            out_slice = out_view[start:end]
            if self.use_flash_v100_prefill_paged:
                # [fa2_sm70 patch] Hand the kernel THIS buffer (it takes output strides) and the
                # host-side length. Without `out` the op writes its own buffer and we copy it below
                # -- 2 x q_len x H x d bytes per layer per chunk. Without `seq_lens_cpu` the op has
                # to read the length off the device: one full sync per layer per chunk.
                extra = {}
                if fa2_kw["out"]:
                    extra["out"] = out_slice.unsqueeze(0)
                if fa2_kw["seq_lens_cpu"] and seq_lens_cpu is not None:
                    extra["seq_lens_cpu"] = seq_lens_cpu[i:i + 1]
                out_seq = self.flash_attn_prefill_paged(
                    query[start:end].unsqueeze(0),
                    key_cache,
                    value_cache,
                    attn_metadata.block_table[i:i + 1],
                    attn_metadata.seq_lens[i:i + 1],
                    softmax_scale=self.scale,
                    kv_cache_dtype=self.kv_cache_dtype,
                    k_scale=_kv_scales(layer)[0],
                    v_scale=_kv_scales(layer)[1],
                    **extra,
                )
                if debug_compare and not _logged_prefill_compare:
                    seq_len = int(seq_lens[i].item())
                    k_cont, v_cont = _extract_contiguous_kv_from_paged_cache(
                        kv_cache=kv_cache,
                        block_table=attn_metadata.block_table[i:i + 1],
                        seq_lens=attn_metadata.seq_lens[i:i + 1],
                        num_kv_heads=num_kv_heads,
                        head_dim=head_dim,
                        block_size=block_size,
                        total_tokens=seq_len,
                    )
                    k_cont, v_cont = _dequantize_fp8_contiguous_kv(
                        k_cont,
                        v_cont,
                        self.kv_cache_dtype,
                        _kv_scales(layer)[0],
                        _kv_scales(layer)[1],
                    )
                    ref_out = self.flash_attn_func(
                        query[start:end].unsqueeze(0),
                        k_cont.unsqueeze(0),
                        v_cont.unsqueeze(0),
                        causal=True,
                        softmax_scale=self.scale,
                    )
                    diff = (out_seq - ref_out).abs()
                    nan_count = int(torch.isnan(out_seq).sum().item())
                    logger.warning(
                        "FLASH_ATTN_V100 debug prefix compare: "
                        "query_len=%d seq_len=%d max_diff=%.8f mean_diff=%.8f "
                        "nan_count=%d q_absmax=%.6f k_absmax=%.6f v_absmax=%.6f "
                        "kv_cache_shape=%s key_shape=%s key_stride=%s "
                        "value_stride=%s key_contig=%s value_contig=%s",
                        end - start,
                        seq_len,
                        float(diff.max().item()),
                        float(diff.mean().item()),
                        nan_count,
                        float(query[start:end].abs().max().item()),
                        float(k_cont.abs().max().item()),
                        float(v_cont.abs().max().item()),
                        tuple(kv_cache.shape),
                        tuple(key_cache.shape),
                        tuple(key_cache.stride()),
                        tuple(value_cache.stride()),
                        str(key_cache.is_contiguous()),
                        str(value_cache.is_contiguous()),
                    )
                    if nan_count > 0:
                        dump_path = (
                            f"/tmp/flash_v100_prefill_nan_dump_pid{os.getpid()}.pt"
                        )
                        torch.save(
                            {
                                "query": query[start:end].detach().cpu(),
                                "key_cache": key_cache.detach().cpu(),
                                "value_cache": value_cache.detach().cpu(),
                                "block_table": attn_metadata.block_table[
                                    i:i + 1].detach().cpu(),
                                "seq_lens": attn_metadata.seq_lens[
                                    i:i + 1].detach().cpu(),
                                "k_cont": k_cont.detach().cpu(),
                                "v_cont": v_cont.detach().cpu(),
                                "out_seq": out_seq.detach().cpu(),
                                "ref_out": ref_out.detach().cpu(),
                            },
                            dump_path,
                        )
                        logger.warning(
                            "FLASH_ATTN_V100 saved failing prefix prefill dump to %s",
                            dump_path,
                        )
                    _logged_prefill_compare = True
            else:
                seq_len = int(seq_lens[i].item())
                k_cont, v_cont = _extract_contiguous_kv_from_paged_cache(
                    kv_cache=kv_cache,
                    block_table=attn_metadata.block_table[i:i + 1],
                    seq_lens=attn_metadata.seq_lens[i:i + 1],
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    block_size=block_size,
                    total_tokens=seq_len,
                )
                k_cont, v_cont = _dequantize_fp8_contiguous_kv(
                    k_cont,
                    v_cont,
                    self.kv_cache_dtype,
                    _kv_scales(layer)[0],
                    _kv_scales(layer)[1],
                )

                out_seq = self.flash_attn_func(
                    query[start:end].unsqueeze(0),
                    k_cont.unsqueeze(0),
                    v_cont.unsqueeze(0),
                    causal=True,
                    softmax_scale=self.scale,
                )
            # [fa2_sm70 patch] Copy ONLY if the op did not write our buffer itself. The pointer test
            # is what makes this safe for every op version: an op that ignores `out` returns its own
            # tensor and is copied exactly as before.
            if out_seq.data_ptr() != out_slice.data_ptr():
                out_slice.copy_(out_seq.squeeze(0))

        return output


class FlashAttnV100Backend(TritonAttentionBackend):
    """Flash Attention V100 Backend."""

    # Keep vLLM unified KV cache update path.
    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_impl_cls():
        return FlashAttnV100Impl

    @staticmethod
    def get_builder_cls():
        return FlashAttnV100MetadataBuilder

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN_V100"

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        # Keep this aligned with the dense prefill kernel dispatch table.
        return [64, 128, 256]
