# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, overload

from vllm.distributed.kv_events import KVCacheEvent
from vllm.logger import init_logger
from vllm.v1.core import fa2sm70_tail
import os as _os_f3

# [18.10] F3: без резерва блоков спекуляции GDN для запросов с max_tokens=1. Умолчание 0.
_NO_SPEC_MAXTOK1 = _os_f3.environ.get("FA2SM70_NO_SPEC_BLOCKS_MAXTOK1", "0") == "1"
if _NO_SPEC_MAXTOK1:
    print(f"[fa2_sm70 18.10] NO_SPEC_BLOCKS_MAXTOK1=1 ({__file__})", flush=True)
from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request

logger = init_logger(__name__)

# Счётчик промахов хвост-кэша: [по длине, по токенам]. Печать по степеням двойки.
_fa2sm70_счёт = [0, 0]


@dataclass
class KVCacheBlocks:
    """
    The allocation result of KVCacheManager, work as the interface between
    Scheduler and KVCacheManager, to hide KVCacheManager's internal data
    structure from the Scheduler.
    """

    blocks: tuple[Sequence[KVCacheBlock], ...]
    """
    `blocks[i][j]` refers to the i-th kv_cache_group
    and the j-th block of tokens.We don't use block of
    tokens as the outer dimension because it assumes all
    kv_cache_groups have the same number of blocks, which is true for now but
    will be broken if we want to give different block_size to different
    kv_cache_groups in the future.

    Each single type KVCacheBlocks could be represented as:
    - list[KVCacheBlock] for more than one KVCacheBlock
    - an empty tuple for requests without KVCacheBlock
      (a precomputed KVCacheBlocks is in KVCacheManager to avoid GC overhead)
    """

    def __add__(self, other: "KVCacheBlocks") -> "KVCacheBlocks":
        """Adds two KVCacheBlocks instances."""
        return KVCacheBlocks(
            tuple(
                list(itertools.chain(blk1, blk2))
                for blk1, blk2 in zip(self.blocks, other.blocks)
            )
        )

    @overload
    def get_block_ids(
        self,
        allow_none: Literal[False] = False,
    ) -> tuple[list[int], ...]: ...

    @overload
    def get_block_ids(
        self,
        allow_none: Literal[True] = True,
    ) -> tuple[list[int], ...] | None: ...

    def get_block_ids(
        self,
        allow_none: bool = False,
    ) -> tuple[list[int], ...] | None:
        """
        Converts the KVCacheBlocks instance to block_ids.

        Returns:
            tuple[list[int], ...]: A tuple of lists where:
                - the outer tuple corresponds to KV cache groups
                - each inner list contains the block_ids of the blocks in that
                  group
        """
        if allow_none and all(len(group) == 0 for group in self.blocks):
            return None
        return tuple([blk.block_id for blk in group] for group in self.blocks)

    def get_unhashed_block_ids(self) -> list[int]:
        """Get block_ids of unhashed blocks from KVCacheBlocks instance."""
        assert len(self.blocks) == 1, "Only one group is supported"
        return [block.block_id for block in self.blocks[0] if block.block_hash is None]

    def new_empty(self) -> "KVCacheBlocks":
        """
        Creates a new KVCacheBlocks instance with no blocks.
        """
        return KVCacheBlocks(tuple(() for _ in range(len(self.blocks))))


class KVCacheManager:
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        hash_block_size: int,
        enable_caching: bool = True,
        use_eagle: bool = False,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ) -> None:
        self.max_model_len = max_model_len

        self.enable_caching = enable_caching
        self.use_eagle = use_eagle
        self.log_stats = log_stats
        self.metrics_collector = metrics_collector
        # FIXME: make prefix cache stats conditional on log_stats. We still need
        # this comment because when the log stats is enabled there are still
        # potential configs we could expose in the future.
        self.prefix_cache_stats = PrefixCacheStats() if log_stats else None

        self.coordinator = get_kv_cache_coordinator(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            use_eagle=self.use_eagle,
            enable_caching=self.enable_caching,
            enable_kv_cache_events=enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=self.metrics_collector,
        )
        self.num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
        self.block_pool = self.coordinator.block_pool
        self.kv_cache_config = kv_cache_config

        # Pre-constructed KVCacheBlocks with no blocks, callers should use this
        # via create_kv_cache_blocks instead of creating new ones to avoid GC
        # overhead.
        #
        # We use nested tuples to ensure the empty KVCacheBlocks is immutable.
        self.empty_kv_cache_blocks = KVCacheBlocks(
            tuple(() for _ in range(self.num_kv_cache_groups))
        )

        # НЕПРЕРЫВНЫЙ ХВОСТ. Размер блока берётся ТОТ ЖЕ, которым хэшируются блоки: координатор
        # утверждает их совпадение (assert hash_block_size == self.block_size), поэтому смешение
        # двух разных размеров здесь невозможно по построению.
        self._fa2sm70_block_size = hash_block_size
        # Запись, ИЗЪЯТАЯ из реестра под конкретный запрос. Реестр владел одной ссылкой на блоки;
        # эта ссылка теперь наша, и её обязан отпустить ровно один путь -- либо после того как
        # запрос сам взял блоки (allocate_slots), либо при освобождении запроса, если он так и не
        # был запланирован. Иначе блок не вернётся в пул НИКОГДА.
        self._fa2sm70_pending: dict[str, fa2sm70_tail.TailEntry] = {}

    @property
    def usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """
        pools = getattr(self.coordinator, "block_pools", None)
        if pools and getattr(self.coordinator, "split_pools", False):
            return max(p.get_usage() for p in pools)
        return self.block_pool.get_usage()

    def make_prefix_cache_stats(self) -> PrefixCacheStats | None:
        """Get (and reset) the prefix cache stats.

        Returns:
            The current prefix caching stats, or None if logging is disabled.
        """
        if not self.log_stats:
            return None
        stats = self.prefix_cache_stats
        self.prefix_cache_stats = PrefixCacheStats()
        return stats

    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
        """Get the computed (cached) blocks for the request.
        Note that the computed blocks must be full.

        Args:
            request: The request to get the computed blocks.

        Returns:
            A tuple containing:
                - A list of blocks that are computed for the request.
                - The number of computed tokens.
        """
        # We skip finding the prefix cache hit when prefix caching is
        # disabled or the request is marked as skipping kv cache read
        # (which happens when the request requires prompt logprobs
        # or calls a pooling model with all pooling).
        if not self.enable_caching or request.skip_reading_prefix_cache:
            return self.empty_kv_cache_blocks, 0

        # NOTE: When all tokens hit the cache, we must recompute the last token
        # to obtain logits. Thus, set max_cache_hit_length to prompt_length - 1.
        # This can trigger recomputation of an entire block, rather than just
        # the single last token, because allocate_slots() requires
        # num_computed_tokens to be block-size aligned. Removing this limitation
        # could slightly improve performance in the future.
        max_cache_hit_length = request.num_tokens - 1
        computed_blocks, num_new_computed_tokens = (
            self.coordinator.find_longest_cache_hit(
                request.block_hashes, max_cache_hit_length
            )
        )

        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.record(
                num_tokens=request.num_tokens,
                num_hits=num_new_computed_tokens,
                preempted=request.num_preemptions > 0,
            )

        if fa2sm70_tail.ENABLED:
            computed_blocks, num_new_computed_tokens = self._fa2sm70_try_extend(
                request, computed_blocks, num_new_computed_tokens
            )

        return self.create_kv_cache_blocks(computed_blocks), num_new_computed_tokens

    # ------------------------------------------------------------------ ХВОСТ

    def _fa2sm70_try_extend(
        self,
        request: Request,
        computed: tuple[list[KVCacheBlock], ...],
        num_computed: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        """Продлить попадание неполным блоком, переданным предыдущим ходом диалога.

        Обычный поиск заканчивается на границе ЦЕЛОГО блока (здесь 1568 токенов). Если предыдущий
        запрос оставил хвост ровно с этой границы, и токены совпали -- забираем его и объявляем
        посчитанным ТОЧНОЕ число токенов, а не кратное блоку.
        """
        if not self.enable_caching:
            return computed, num_computed
        bs = self._fa2sm70_block_size
        nfull = num_computed // bs
        if num_computed % bs != 0:
            # Уже невыровнено -- значит хвост подцеплен кем-то ещё; второй раз не продлеваем.
            return computed, num_computed
        parent = request.block_hashes[nfull - 1] if nfull > 0 else None
        pool = self.block_pool
        # Тот же запас, что у них: последний токен обязан считаться заново ради логитов.
        max_len = request.num_tokens - 1
        for n in pool.fa2sm70_tails.tail_lengths(parent):
            end = nfull * bs + n
            if end > max_len or end <= num_computed:
                # ПЕЧАТЬ ПО СТЕПЕНЯМ ДВОЙКИ, А НЕ НА КАЖДУЮ ИТЕРАЦИЮ. Это горячий цикл
                # ПЛАНИРОВЩИКА: он крутится тысячи раз в секунду, и одна строка на итерацию
                # забивает канал вывода. Замерено 20.08: 1 313 177 строк из 1 313 716 в логе
                # за 40 минут, EngineCore 39 % CPU при ПУСТОЙ очереди, новые запросы до движка
                # не доходили вовсе -- сервер выглядел живым (/health 200), а генерация висела.
                _fa2sm70_счёт[0] += 1
                _c = _fa2sm70_счёт[0]
                if _c & (_c - 1) == 0:
                    logger.info(
                        "[fa2_sm70 хвост] мимо по ДЛИНЕ: end=%d, посчитано=%d, предел=%d "
                        "(таких промахов всего %d)",
                        end, num_computed, max_len, _c,
                    )
                continue
            mine = request.all_token_ids[nfull * bs : end]
            key = fa2sm70_tail.make_tail_key(parent, mine)
            entry = pool.fa2sm70_take_tail(key)
            if entry is None:
                # НЕ «просто мимо»: печатаем ПОЗИЦИЮ первого расхождения токенов. Хэш говорит
                # только «не то», а нам нужно знать, ЧТО именно разошлось -- шаблон диалога,
                # длина или содержимое.
                theirs = pool.fa2sm70_tails.tokens_of(parent, n)
                _fa2sm70_счёт[1] += 1
                _c2 = _fa2sm70_счёт[1]
                if theirs is not None and _c2 & (_c2 - 1) == 0:
                    lim = min(len(mine), len(theirs))
                    d = next((i for i in range(lim) if mine[i] != theirs[i]), lim)
                    logger.info(
                        "[fa2_sm70 хвост] мимо по ТОКЕНАМ: совпало %d из %d/%d, "
                        "первое расхождение на %d (мой %s, записан %s)",
                        d, len(mine), len(theirs), nfull * bs + d,
                        mine[d] if d < len(mine) else None,
                        theirs[d] if d < len(theirs) else None,
                    )
                continue
            if len(entry.blocks_by_group) != len(computed):
                # Состав групп изменился -- запись не наша, вернуть ссылку и идти дальше.
                pool._fa2sm70_release_tail(entry)
                continue
            extended = tuple(
                list(computed[i]) + list(entry.blocks_by_group[i])
                for i in range(len(computed))
            )
            if fa2sm70_tail.POISON:
                # ОТРИЦАТЕЛЬНЫЙ КОНТРОЛЬ. Объявляем посчитанным на четыре токена больше, чем есть.
                # Эти четыре не будут вычислены ни разу: у внимания останутся чужие слоты, у GDN --
                # дыра в цепи состояния. Вывод ОБЯЗАН испортиться. Если он не испортился, значит
                # восстановленный хвост ни на что не влияет, и «совпало» в основном опыте не
                # доказывало ничего.
                end += 4
            self._fa2sm70_pending[request.request_id] = entry
            # СЧЁТЧИК МАРШРУТА пишется ПО ФАКТУ работы. Без него «вывод совпал» доказывал бы лишь
            # то, что мы сравнили прежний путь сам с собой.
            logger.info(
                "[fa2_sm70 хвост] ПОДХВАЧЕН: +%d ток (было %d, стало %d), реестр %s",
                end - num_computed, num_computed, end, pool.fa2sm70_tails.stats,
            )
            return extended, end
        return computed, num_computed

    def _fa2sm70_settle(self, request_id: str) -> None:
        """Отпустить ссылку реестра ПОСЛЕ того, как блоки взял сам запрос.

        Порядок обязателен: сначала запрос делает touch (ref 1 -> 2), потом мы снимаем свою (2 -> 1).
        Обратный порядок на миг обнулил бы счётчик, и блок ушёл бы в очередь свободных.
        """
        entry = self._fa2sm70_pending.pop(request_id, None)
        if entry is not None:
            self.block_pool._fa2sm70_release_tail(entry)

    def _fa2sm70_register_tail(self, request: Request) -> None:
        """Запомнить хвост завершившегося запроса: неполный блок KV + состояние GDN на его конце."""
        if not self.enable_caching:
            return
        bs = self._fa2sm70_block_size
        # ДЛИНА БЕРЁТСЯ ПО ФАКТИЧЕСКИ ПОСЧИТАННОМУ, а не по числу токенов заявки. У завершившегося
        # запроса последний токен уже выбран, но его KV и состояние GDN в кэш НЕ записаны; у
        # вытесненного посчитано и того меньше. Хвост, объявленный длиннее посчитанного, вернул бы
        # блок, не соответствующий токенам, -- и это была бы ТИХАЯ неверность, а не падение.
        total = request.num_computed_tokens
        nfull = total // bs
        tail_len = total - nfull * bs
        if tail_len <= 0 or nfull > len(request.block_hashes):
            # Ровно на границе (запоминать нечего) либо хэшей меньше, чем целых блоков.
            return
        parent = request.block_hashes[nfull - 1] if nfull > 0 else None
        tail_ids = request.all_token_ids[nfull * bs : total]
        key = fa2sm70_tail.make_tail_key(parent, tail_ids)
        blocks_by_group: list[list[KVCacheBlock]] = []
        for mgr in self.coordinator.single_type_managers:
            req_blocks = mgr.req_to_blocks.get(request.request_id)
            if req_blocks is None or len(req_blocks) <= nfull:
                return
            blk = req_blocks[nfull]
            if blk is None or blk.is_null:
                return
            blocks_by_group.append([blk])
        logger.info(
            "[fa2_sm70 хвост] ЗАПОМНЕН: %d ток хвоста при %d посчитанных", tail_len, total
        )
        self.block_pool.fa2sm70_hold_tail(
            fa2sm70_tail.TailEntry(
                key, total, tuple(blocks_by_group), request.request_id, tail_ids
            )
        )

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        num_lookahead_tokens: int = 0,
        num_external_computed_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
    ) -> KVCacheBlocks | None:
        """Add slots for a request with new tokens to append.

        Args:
            request: The request to allocate slots.
            num_new_tokens: The number of new tokens to be allocated and computed.
            num_new_computed_tokens: The number of new computed tokens just
                hitting the prefix caching, excluding external tokens.
            new_computed_blocks: The cached blocks for the above new computed
                tokens, grouped as a tuple by kv cache groups.
            num_lookahead_tokens: The number of speculative tokens to allocate.
                This is used by spec decode proposers with kv-cache such
                as eagle.
            num_external_computed_tokens: The number of tokens that their
                KV caches are not cached by vLLM but cached by the connector.
            delay_cache_blocks: Whether to skip caching the blocks. This is
                used by P/D when allocating blocks used in a KV transfer
                which will complete in a future step.
            num_encoder_tokens: The number of encoder tokens to allocate for
                cross-attention in encoder-decoder models(e.g., Whisper).
                For decoder-only models, this should be 0.

        Blocks layout:
        ```
        ----------------------------------------------------------------------
        | < comp > | < new_comp > | < ext_comp >  | < new >  | < lookahead > |
        ----------------------------------------------------------------------
                                                  |   < to be computed >     |
        ----------------------------------------------------------------------
                                  |            < to be allocated >           |
        ----------------------------------------------------------------------
                                  | < to be cached (roughly, |
                                  | details below)>          |
        ----------------------------------------------------------------------
        | Prefix-cached tokens from either vLLM   |
        | or connector. Can be safely removed if  |
        | they are outside sliding window.        |
        ----------------------------------------------------------------------
        |   < cached by vLLM >    | not cached by |
                                  | vLLM, but     |
        | ref_cnt  | ref_cnt not  | cached by     |
        | increased| increased yet| connector     |
        ----------------------------------------------------------------------
        ```

        Abbrivations:

        ```
        comp      = request.num_computed_tokens
        new_comp  = num_new_computed_tokens
                  = len(new_computed_blocks) * block_size
        ext_comp  = num_external_computed_tokens, cached by the connector
        new       = num_new_tokens, including unverified draft tokens
        lookahead = num_lookahead_tokens
        ```

        NOTE: for new tokens which include both verified and unverified draft
        tokens, we only cache the verified tokens (by capping the number at
        `request.num_tokens`).

        The allocation has three stages:
        - Free unnecessary blocks in `comp` and check
           if we have sufficient free blocks (return None if not).
        - Handle prefix tokens (`comp + new_comp + ext_comp`):
            - Free unnecessary blocks (e.g. outside sliding window)
            - Allocate new blocks for `ext_comp` tokens inside
              sliding window
        - Allocate new blocks for tokens to be computed (`new + lookahead`)

        Returns:
            A list of new allocated blocks.
        """
        # When loading KV data asynchronously, we may have zero new tokens to
        # compute while still allocating slots for externally computed tokens.
        if _NO_SPEC_MAXTOK1 and getattr(request, "max_tokens", None) == 1:
            # [18.10, F3] декода у такого запроса не будет -- блоки спекуляции GDN не резервировать
            for _mgr in self.coordinator.single_type_managers:
                _бс = getattr(_mgr, "_без_спец", None)
                if _бс is not None:
                    _бс.add(request.request_id)
        if num_new_tokens == 0 and num_external_computed_tokens == 0:
            raise ValueError(
                "num_new_tokens must be greater than 0 when there are no "
                "external computed tokens"
            )

        if new_computed_blocks is not None:
            new_computed_block_list = new_computed_blocks.blocks
        else:
            new_computed_block_list = self.empty_kv_cache_blocks.blocks

        # The number of computed tokens is the number of computed tokens plus
        # the new prefix caching hits
        num_local_computed_tokens = (
            request.num_computed_tokens + num_new_computed_tokens
        )
        total_computed_tokens = min(
            num_local_computed_tokens + num_external_computed_tokens,
            self.max_model_len,
        )
        num_tokens_main_model = total_computed_tokens + num_new_tokens
        num_tokens_need_slot = min(
            num_tokens_main_model + num_lookahead_tokens,
            self.max_model_len,
        )

        # Free the blocks that are skipped during the attention computation
        # (e.g., tokens outside the sliding window).
        # We can do this even if we cannot schedule this request due to
        # insufficient free blocks.
        # Should call this function before allocating new blocks to reduce
        # the number of evicted blocks.
        self.coordinator.remove_skipped_blocks(
            request.request_id, total_computed_tokens
        )

        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=num_tokens_need_slot,
            new_computed_blocks=new_computed_block_list,
            num_encoder_tokens=num_encoder_tokens,
            total_computed_tokens=num_local_computed_tokens
            + num_external_computed_tokens,
            num_tokens_main_model=num_tokens_main_model,
        )

        # [FA2/SM70] В раздельном режиме сумма по разным пулам ничего не значит: место
        # должно найтись В КАЖДОМ пуле отдельно (координатор помнит разрез по группам).
        if not self.coordinator.has_free_blocks_for():
            # Cannot allocate new blocks
            return None

        if (
            new_computed_block_list is not self.empty_kv_cache_blocks.blocks
            or num_external_computed_tokens > 0
        ):
            # Append the new computed blocks to the request blocks until now to
            # avoid the case where the new blocks cannot be allocated.
            self.coordinator.allocate_new_computed_blocks(
                request_id=request.request_id,
                new_computed_blocks=new_computed_block_list,
                num_local_computed_tokens=num_local_computed_tokens,
                num_external_computed_tokens=num_external_computed_tokens,
            )

        if fa2sm70_tail.ENABLED and request.request_id in self._fa2sm70_pending:
            # Запрос уже сделал touch внутри allocate_new_computed_blocks -- снимаем ссылку реестра.
            # И правим учёт: последний из взятых блоков НЕПОЛОН, значит он ещё НЕ захэширован, и
            # считать его «уже закэшированным» нельзя -- иначе он не попадёт в кэш, когда дозаполнится.
            for mgr in self.coordinator.single_type_managers:
                n = mgr.num_cached_block.get(request.request_id)
                if n:
                    mgr.num_cached_block[request.request_id] = n - 1
            self._fa2sm70_settle(request.request_id)

        new_blocks = self.coordinator.allocate_new_blocks(
            request.request_id,
            num_tokens_need_slot,
            num_tokens_main_model,
            num_encoder_tokens,
        )

        # P/D: delay caching blocks if we have to recv from
        # remote. Update state for locally cached blocks.
        if not self.enable_caching or delay_cache_blocks:
            return self.create_kv_cache_blocks(new_blocks)

        # NOTE(woosuk): We want to commit (cache) up to num_local_computed_tokens
        # + num_external_computed_tokens + num_new_tokens, but must exclude
        # "non-committable" tokens (e.g., draft tokens that could be rejected).
        # Therefore, we cap the number at `request.num_tokens`, ensuring only
        # "finalized" tokens are cached.
        num_tokens_to_cache = min(
            total_computed_tokens + num_new_tokens,
            request.num_tokens,
        )
        self.coordinator.cache_blocks(request, num_tokens_to_cache)

        return self.create_kv_cache_blocks(new_blocks)

    def free(self, request: Request) -> None:
        """Free the blocks allocated for the request.
        We free the blocks in reverse order so that the tail blocks are evicted
        first when caching is enabled.

        Args:
            request: The request to free the blocks.
        """
        if fa2sm70_tail.ENABLED:
            # Ссылка, взятая под этот запрос, но так и не отданная (запрос не был запланирован).
            self._fa2sm70_settle(request.request_id)
            self._fa2sm70_register_tail(request)
        self.coordinator.free(request.request_id)

    def remove_skipped_blocks(
        self, request_id: str, total_computed_tokens: int
    ) -> None:
        """Remove the blocks that are no longer needed from `blocks` and replace
        the removed blocks with null_block.

        Args:
            request_id: The request ID.
            total_computed_tokens: The total number of computed tokens, including
                local computed tokens and external computed tokens.
        """
        self.coordinator.remove_skipped_blocks(request_id, total_computed_tokens)

    def evict_blocks(self, block_ids: set[int]) -> None:
        """evict blocks from the prefix cache by their block IDs.

        Args:
            block_ids: Set of block IDs to evict from cache.
        """
        self.block_pool.evict_blocks(block_ids)

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalidate prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        pools = getattr(self.coordinator, "block_pools", None)
        if pools and getattr(self.coordinator, "split_pools", False):
            if not all(p.reset_prefix_cache() for p in pools):
                return False
            return True
        if not self.block_pool.reset_prefix_cache():
            return False
        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.reset = True
        return True

    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
        """Calculate the number of common prefix blocks for each kv cache group.

        The function selects a running request and iterates through its blocks.
        A block is considered a common prefix block if ALL requests with
        allocated KV cache share it (i.e., ref_cnt equals the number of entries
        in req_to_blocks).

        NOTE(woosuk): The number of requests with allocated KV cache is **greater
        than or equal to** the number of requests scheduled in the current step.
        This is because having allocated KV cache only indicates that:
        1. The request has not yet finished, and
        2. The request holds its blocks unfreed.

        While all scheduled requests must have allocated KV cache, the inverse
        is not necessarily true. There may be requests with allocated KV cache
        that are not scheduled in the current step.

        This can result in an edge case where the number of common prefix blocks
        is 0, even though all scheduled requests share a common prefix. This
        occurs because there may be unscheduled requests that do not share the
        common prefix. Currently, this case cannot be easily detected, so the
        function returns 0 in such cases.

        Args:
            running_request_id: The request ID of any running request, used to
                identify the common prefix blocks.

        Returns:
            list[int]: The number of common prefix blocks for each kv cache
            group.
        """
        return self.coordinator.get_num_common_prefix_blocks(running_request_id)

    def new_step_starts(self) -> None:
        """Начало шага планировщика."""
        self.coordinator.new_step_starts()

    def take_events(self) -> list[KVCacheEvent]:
        """Take the KV cache events from the block pool.

        Returns:
            A list of KV cache events.
        """
        return self.block_pool.take_events()

    def get_blocks(self, request_id: str) -> KVCacheBlocks:
        """Get the blocks of a request."""
        return self.create_kv_cache_blocks(self.coordinator.get_blocks(request_id))

    def get_block_ids(self, request_id: str) -> tuple[list[int], ...]:
        """Get the block ids of a request."""
        return self.get_blocks(request_id).get_block_ids()

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """Cache the blocks for the request, if enabled.

        Args:
            request: The request to cache the blocks.
            num_computed_tokens: The number of computed tokens, including tokens
                that are already cached and tokens to be cached.
        """
        if self.enable_caching:
            self.coordinator.cache_blocks(request, num_computed_tokens)

    def create_kv_cache_blocks(
        self, blocks: tuple[list[KVCacheBlock], ...]
    ) -> KVCacheBlocks:
        # Only create new KVCacheBlocks for non-empty blocks
        return KVCacheBlocks(blocks) if any(blocks) else self.empty_kv_cache_blocks
