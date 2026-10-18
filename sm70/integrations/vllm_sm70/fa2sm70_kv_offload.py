# SPDX-License-Identifier: Apache-2.0
"""ВЫГРУЗКА KV В ОЗУ (fa2_sm70): второй уровень префикс-кэша в закреплённой памяти хоста.

ЗАЧЕМ. Пул GPU у боевого -- 32 блока по 16384 токена на группу, и под нагрузкой Clef блоки
состояния GDN вытесняются раньше, чем их кто-то переиспользует (разбор 15.10: 0 попаданий из
35.7 млн токенов). Блок, ушедший из GPU, здесь не теряется: копия всех его байтов лежит в ОЗУ, и
попадание в неё возвращает 16384 токена за ~0.05 с вместо секунд префилла.

ВКЛЮЧЕНИЕ (оба рычага читает пускач и этот модуль):
  FA2SM70_KV_OFFLOAD=1      пускач добавляет --kv-transfer-config с этим классом и
                            --no-disable-hybrid-kv-cache-manager (без HMA модель с GDN не идёт);
  FA2SM70_KV_OFFLOAD_GB=16  сколько ОЗУ закрепить НА ВЕСЬ СЕРВЕР (делится поровну между рангами).
Отладка: FA2SM70_KV_OFFLOAD_SYNC=1 -- копии в ОЗУ на основном потоке (без перекрытия);
FA2SM70_KV_OFFLOAD_LOG_S=60 -- период строки учёта в журнале планировщика;
FA2SM70_KV_OFFLOAD_TRACE=1 -- строка на каждое попадание и запись.

ПОЧЕМУ НЕ ГОТОВЫЙ OffloadingConnector. Он не SupportsHMA (с ним vLLM выключает гибридный менеджер,
а без него GDN не работает), требует одну страницу на все группы и копирует блок по виду тензора.
У нас вид врёт: int8-слой внимания режется `_carve` на данные K+V и две ОТДЕЛЬНЫЕ таблицы шкал в
конце буфера, копия по stride(0) взяла бы хвост чужого блока и не взяла бы шкал. Плюс побочные
таблицы средних/размахов ключей разреженного отбора (_KBAR/_DEVP) и шкалы int16 состояния GDN --
их в kv_caches нет вовсе.

ЕДИНИЦА. Слот ОЗУ = один блок (16384 токена) ВСЕХ групп: для каждого слоя внимания -- K+V, шкалы
K, шкалы V, строки _KBAR и _DEVP этого блока; для каждого слоя GDN -- страница [conv|ssm] и строка
шкал int16. Ключ слота -- хэш блока движка (цепной: он же покрывает весь префикс, соль кэша и
хэши медиа). На ранг 323.6 МиБ на блок (опись 16.10).

ПОПАДАНИЕ. Синхронное (load_kv_async=False): в get_num_new_matched_tokens считаем, сколько
блоков сразу за локальным попаданием лежит в ОЗУ, движок выделяет под них блоки, а воркер копирует
их H2D в начале шага ДО прохода модели. Внимание грузится во все m блоков, GDN -- только в
последний (режим 'all': состояние нужно лишь на конце префикса; движок сам кладёт в группу GDN
null-блоки до n-1 и один настоящий на n-1 -- ровно как при локальном попадании).

ЗАПИСЬ (сквозная). Блок, ДОПИСАННЫЙ шагом S (и целиком лежащий в промпте), копируется D2H в
НАЧАЛЕ шага S+1, а не в wait_for_save шага S: черновик MTP пишет свой KV и средние ключей ПОСЛЕ
прохода цели (вне окна коннектора), а состояние GDN конца блока -- итог шага. В начале S+1 всё это
уже записано (порядок потока), а писать в эти блоки до прохода S+1 некому. Копия идёт на побочном
потоке; основной ждёт её, только если планировщик видит, что какой-то из сохраняемых блоков в этом
же шаге выдан заново (флаг conflict) или есть загрузки; иначе -- в начале следующего шага (перекрытие
не больше одного шага). Хэш становится видимым для поиска со следующего планирования.

Декодные блоки (с ответом модели) НЕ сохраняются: состояние GDN у границы при спекуляции переходит
между колонками (GRAN_SRC/DST), и мгновенный снимок там не гарантирован. Следующий ход диалога
несёт их в промпте и сохранит сам.
"""

from __future__ import annotations

import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

# Имя под «vllm.»: модуль лежит вне пакета vllm, и логгер с его собственным именем не наследует
# обработчики vLLM -- строки [kv-offload] молча пропадали (стенд 18.10).
logger = init_logger(f"vllm.{__name__}")

GB = float(os.environ.get("FA2SM70_KV_OFFLOAD_GB", "16"))
SYNC = os.environ.get("FA2SM70_KV_OFFLOAD_SYNC", "0") == "1"
LOG_S = float(os.environ.get("FA2SM70_KV_OFFLOAD_LOG_S", "60"))
# Строка на каждое попадание и каждую запись (гейт стенда опознаёт по ней путь запроса).
TRACE = os.environ.get("FA2SM70_KV_OFFLOAD_TRACE", "0") == "1"

# Подменяется тестами (CPU-обвязка без настоящего бэкенда).
_BACKEND = None


def _backend():
    global _BACKEND
    if _BACKEND is None:
        from vllm.v1.attention.backends import fa2_sm70 as be

        _BACKEND = be
    return _BACKEND


def _block_size(kv_cache_config: "KVCacheConfig") -> int:
    sizes = {int(g.kv_cache_spec.block_size) for g in kv_cache_config.kv_cache_groups}
    if len(sizes) != 1:
        raise ValueError(
            f"[kv-offload] блоки групп разного размера {sorted(sizes)}: слот ОЗУ = один блок всех "
            "групп, нужен общий размер (раздельные пулы FA2SM70_SPLIT_POOLS=1 его и дают)")
    return sizes.pop()


def est_slot_bytes(kv_cache_config: "KVCacheConfig") -> int:
    """Оценка байтов слота НА РАНГ по описанию групп -- одна и та же у планировщика и воркера.

    Страница слоя из описания + 1/16 на побочные таблицы (_KBAR+_DEVP у int8-внимания -- 3.1 %
    страницы, шкалы int16 у GDN -- 0.2 %; при 1/32 запас был бы ~8 КиБ на слой) + 4 КиБ на слой.
    Воркер сверяет с точным числом и падает громко, если оценка мала: иначе слотов в ОЗУ окажется
    больше бюджета. Закрепляется точное число (слотов x точный слот), оно меньше бюджета.
    """
    total = 0
    for g in kv_cache_config.kv_cache_groups:
        pg = int(g.kv_cache_spec.page_size_bytes)
        total += len(g.layer_names) * (pg + pg // 16 + 4096)
    return total


def num_slots(kv_cache_config: "KVCacheConfig", world: int) -> int:
    return max(0, int(GB * (1 << 30)) // (max(1, world) * est_slot_bytes(kv_cache_config)))


@dataclass
class Fa2Sm70OffloadMeta(KVConnectorMetadata):
    # (слот, номер блока по группам; -1 = эту группу не трогать)
    saves: list[tuple[int, tuple[int, ...]]] = field(default_factory=list)
    loads: list[tuple[int, tuple[int, ...]]] = field(default_factory=list)
    # какой-то сохраняемый блок в этом шаге выдан заново -> основной поток ждёт копию до прохода
    conflict: bool = False


# ---------------------------------------------------------------------------------------------
# Планировщик
# ---------------------------------------------------------------------------------------------
class _SchedSide:
    def __init__(self, vllm_config: "VllmConfig", kv_cache_config: "KVCacheConfig"):
        groups = kv_cache_config.kv_cache_groups
        self.ng = len(groups)
        self.is_mamba = [isinstance(g.kv_cache_spec, MambaSpec) for g in groups]
        for g in groups:
            if not isinstance(g.kv_cache_spec, (MambaSpec, AttentionSpec)):
                raise ValueError(f"[kv-offload] группа {type(g.kv_cache_spec).__name__} не поддержана")
        self.B = _block_size(kv_cache_config)
        pc = vllm_config.parallel_config
        hash_bs = (int(vllm_config.cache_config.block_size)
                   * int(getattr(pc, "decode_context_parallel_size", 1) or 1)
                   * int(getattr(pc, "prefill_context_parallel_size", 1) or 1))
        if self.B % hash_bs:
            raise ValueError(f"[kv-offload] блок {self.B} не кратен шагу хэша {hash_bs}")
        self.ratio = self.B // hash_bs
        sc = vllm_config.speculative_config
        use_eagle = bool(sc is not None and sc.use_eagle())
        # Как координатор (kv_cache_coordinator.py, FA2SM70_EAGLE_SBROS): при 1 попадание со
        # спекуляцией короче на блок -- внешнее попадание обязано вести себя так же.
        self.drop = use_eagle and os.environ.get("FA2SM70_EAGLE_SBROS", "1") != "0"
        self.world = int(pc.world_size)
        self.n_slots = num_slots(kv_cache_config, self.world)
        self.lru: OrderedDict[Any, int] = OrderedDict()   # хэш -> слот, старые первыми
        self.free: list[int] = list(range(self.n_slots - 1, -1, -1))
        self.reqs: dict[str, "Request"] = {}
        self.blocks: dict[str, list[list[int]]] = {}
        self.offer: dict[str, tuple[int, list[tuple[Any, int]]]] = {}
        self.pending: list[tuple[Any, tuple[int, ...]]] = []   # дописаны прошлым шагом
        self.step_loads: list[tuple[int, tuple[int, ...]]] = []
        self.locked: set[int] = set()
        self.st = dict(lookups=0, hit_reqs=0, hit_blocks=0, saved=0, evicted=0, conflict=0,
                       nospace=0, skip_null=0)
        self._t_log = time.monotonic()
        est = est_slot_bytes(kv_cache_config)
        logger.info(
            "[kv-offload] планировщик: слотов %d (по %.1f МиБ на ранг x %d рангов, бюджет %.1f ГиБ), "
            "ёмкость %d токенов, блок %d, групп %d (GDN: %s), сброс блока при спекуляции: %s",
            self.n_slots, est / 2**20, self.world, GB, self.n_slots * self.B, self.B, self.ng,
            self.is_mamba, self.drop)

    # --- поиск -------------------------------------------------------------------------------
    def _hash(self, request: "Request", i: int):
        hi = (i + 1) * self.ratio - 1
        hs = request.block_hashes
        return hs[hi] if hi < len(hs) else None

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int):
        self.offer.pop(request.request_id, None)
        if self.n_slots == 0 or num_computed_tokens % self.B:
            return 0, False
        self.st["lookups"] += 1
        L = num_computed_tokens // self.B
        maxb = (request.num_tokens - 1) // self.B
        chain: list[tuple[Any, int]] = []
        i = L
        while i < maxb:
            h = self._hash(request, i)
            s = self.lru.get(h) if h is not None else None
            if s is None:
                break
            chain.append((h, s))
            i += 1
        if self.drop and chain:
            chain.pop()
        if not chain:
            return 0, False
        self.offer[request.request_id] = (L, chain)
        return len(chain) * self.B, False

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        rid = request.request_id
        self.reqs[rid] = request
        off = self.offer.pop(rid, None)
        if num_external_tokens <= 0 or off is None:
            return
        L, chain = off
        m = num_external_tokens // self.B
        if m != len(chain) or num_external_tokens % self.B:
            raise RuntimeError(f"[kv-offload] движок принял {num_external_tokens} внешних токенов, "
                               f"предложено {len(chain) * self.B}")
        ids = blocks.get_block_ids()
        last = L + m - 1
        for k, (h, s) in enumerate(chain):
            i = L + k
            per = []
            for g in range(self.ng):
                if self.is_mamba[g] and i != last:
                    per.append(-1)
                    continue
                b = ids[g][i] if i < len(ids[g]) else 0
                if b <= 0:
                    raise RuntimeError(f"[kv-offload] под внешний блок {i} группы {g} выдан "
                                       f"null-блок (список {len(ids[g])})")
                per.append(b)
            self.step_loads.append((s, tuple(per)))
            self.locked.add(s)
            self.lru.move_to_end(h)
        self.st["hit_reqs"] += 1
        self.st["hit_blocks"] += m
        if TRACE:
            logger.info("[kv-offload] ПОПАДАНИЕ %s: локально %d блоков, из ОЗУ %d (слоты %s)",
                        rid, L, m, [s for _, s in chain])

    # --- слоты -------------------------------------------------------------------------------
    def _take_slot(self):
        if self.free:
            return self.free.pop()
        for h, s in self.lru.items():
            if s not in self.locked:
                del self.lru[h]
                self.st["evicted"] += 1
                return s
        return None

    # --- шаг ---------------------------------------------------------------------------------
    def build_connector_meta(self, so: "SchedulerOutput") -> Fa2Sm70OffloadMeta:
        fresh: set[tuple[int, int]] = set()
        before: dict[str, int] = {}
        for nr in so.scheduled_new_reqs:
            self.blocks[nr.req_id] = [list(x) for x in nr.block_ids]
            before[nr.req_id] = nr.num_computed_tokens
            for g, x in enumerate(nr.block_ids):
                fresh.update((g, b) for b in x)
        cr = so.scheduled_cached_reqs
        resumed = getattr(cr, "resumed_req_ids", set()) or set()
        for k, rid in enumerate(cr.req_ids):
            before[rid] = cr.num_computed_tokens[k]
            nb = cr.new_block_ids[k]
            if nb is None:
                continue
            for g, x in enumerate(nb):
                fresh.update((g, b) for b in x)
            cur = self.blocks.get(rid)
            if rid in resumed or cur is None:
                self.blocks[rid] = [list(x) for x in nb]
            else:
                for g, x in enumerate(nb):
                    cur[g].extend(x)
        for _, per in self.step_loads:
            fresh.update((g, b) for g, b in enumerate(per) if b > 0)

        # 1) дописанные прошлым шагом -> копия в начале этого
        meta = Fa2Sm70OffloadMeta(loads=self.step_loads)
        for h, per in self.pending:
            if h in self.lru:
                self.lru.move_to_end(h)
                continue
            s = self._take_slot()
            if s is None:
                self.st["nospace"] += 1
                continue
            self.lru[h] = s
            self.locked.add(s)
            meta.saves.append((s, per))
            if not meta.conflict and any((g, b) in fresh for g, b in enumerate(per)):
                meta.conflict = True
        if meta.conflict:
            self.st["conflict"] += 1
        self.st["saved"] += len(meta.saves)
        if TRACE and meta.saves:
            logger.info("[kv-offload] ЗАПИСЬ %d блоков в слоты %s%s", len(meta.saves),
                        [sl for sl, _ in meta.saves], " (conflict)" if meta.conflict else "")
        self.pending = []

        # 2) что допишет ЭТОТ шаг (только промпт) -> копия в начале следующего
        seen = set()
        B = self.B
        for rid, n in so.num_scheduled_tokens.items():
            req = self.reqs.get(rid)
            blk = self.blocks.get(rid)
            c0 = before.get(rid)
            if req is None or blk is None or c0 is None:
                continue
            end = min(c0 + n, req.num_prompt_tokens)
            for i in range(c0 // B, end // B):
                if (i + 1) * B <= c0:
                    continue
                h = self._hash(req, i)
                if h is None:
                    break
                if h in self.lru or h in seen:
                    continue
                per = tuple(blk[g][i] if i < len(blk[g]) else 0 for g in range(self.ng))
                if any(b <= 0 for b in per):
                    self.st["skip_null"] += 1
                    continue
                seen.add(h)
                self.pending.append((h, per))

        self.step_loads = []
        self.locked = set()
        now = time.monotonic()
        if now - self._t_log >= LOG_S:
            self._t_log = now
            logger.info("[kv-offload] слотов занято %d/%d; %s", len(self.lru), self.n_slots,
                        " ".join(f"{k}={v}" for k, v in self.st.items()))
        return meta

    def request_finished(self, request: "Request"):
        rid = request.request_id
        self.reqs.pop(rid, None)
        self.blocks.pop(rid, None)
        self.offer.pop(rid, None)
        return False, None


# ---------------------------------------------------------------------------------------------
# Воркер
# ---------------------------------------------------------------------------------------------
@dataclass
class _Rng:
    """Диапазон байтов блока j: [base + j*stride, +length) в сыром буфере `u8`."""
    u8: torch.Tensor
    base: int
    stride: int
    length: int
    nb: int
    host: int = 0       # смещение внутри слота


@dataclass
class _Lazy:
    """Таблица, которой при регистрации может ещё не быть (_KBAR/_DEVP, шкалы int16)."""
    kind: str           # "kbar" | "devp" | "scale"
    key: Any            # пул p (dict) или имя слоя GDN
    rows: int           # строк на блок
    row_bytes: int
    host: int = 0
    tab: Any = None     # разрешённая таблица (адрес постоянен: словари бэкенда не заменяют её)

    @property
    def length(self) -> int:
        return self.rows * self.row_bytes


def _u8(t: torch.Tensor) -> tuple[torch.Tensor, int]:
    st = t.untyped_storage()
    u8 = torch.empty(0, dtype=torch.uint8, device=t.device).set_(st, 0, (st.nbytes(),), (1,))
    return u8, st.data_ptr()


def _rng_of(t: torch.Tensor, u8: torch.Tensor, sp: int) -> _Rng:
    """Блочный тензор [nb, ...]: блок j -- непрерывный кусок t[j]."""
    if not t[0].is_contiguous():
        raise RuntimeError(f"[kv-offload] блок тензора {tuple(t.shape)} не непрерывен")
    es = t.element_size()
    return _Rng(u8, t.data_ptr() - sp, t.stride(0) * es, t[0].numel() * es, int(t.shape[0]))


def _merge(rs: list[_Rng]) -> list[_Rng]:
    out: list[_Rng] = []
    for r in rs:
        p = out[-1] if out else None
        if (p is not None and p.u8.data_ptr() == r.u8.data_ptr() and p.stride == r.stride
                and p.base + p.length == r.base and p.nb == r.nb):
            p.length += r.length
        else:
            out.append(_Rng(r.u8, r.base, r.stride, r.length, r.nb))
    return out


class _WorkerSide:
    def __init__(self, vllm_config: "VllmConfig", kv_cache_config: "KVCacheConfig"):
        self.cfg = vllm_config
        self.kvc = kv_cache_config
        self.B = _block_size(kv_cache_config)
        self.n_slots = num_slots(kv_cache_config, int(vllm_config.parallel_config.world_size))
        self.est = est_slot_bytes(kv_cache_config)
        self.plan: list[list[Any]] = []
        self.slot_bytes = 0
        self.host: torch.Tensor | None = None
        self._registered = False
        self.side = None
        self.prev_ev = None
        self.kv_caches: dict[str, Any] = {}
        self.st = dict(saves=0, loads=0, waits=0)

    # --- раскладка ---------------------------------------------------------------------------
    def _attn_plan(self, name: str, kv: torch.Tensor) -> list[Any]:
        be = _backend()
        if kv.dim() != 5 or int(kv.shape[1]) != 2:
            raise RuntimeError(f"[kv-offload] слой {name}: неожиданная форма {tuple(kv.shape)}")
        last = int(kv.shape[-1])
        if kv.dtype in (torch.int8, torch.uint8) and (last - 4) in (64, 128, 256, 512):
            fmt = be._I8
            if kv.storage_offset() != 0:
                raise RuntimeError(f"[kv-offload] слой {name}: int8-пул со смещением в буфере -- "
                                   "_carve режет от начала буфера, раскладка не та")
        elif kv.dtype in (torch.float16, torch.bfloat16):
            fmt = be._FP16
        else:
            raise RuntimeError(f"[kv-offload] слой {name}: формат {kv.dtype}/{last} не поддержан")
        p = be._carve(kv, fmt)
        u8, sp = _u8(kv)
        rs = [_rng_of(p["k"], u8, sp), _rng_of(p["v"], u8, sp)]
        nb, bs, hkv = int(kv.shape[0]), int(kv.shape[2]), int(kv.shape[3])
        for key in ("ks", "vs"):
            t = p.get(key)
            if t is not None:
                n = bs * hkv
                rs.append(_Rng(u8, t.data_ptr() - sp, n * t.element_size(),
                               n * t.element_size(), nb))
        plan: list[Any] = _merge(rs)
        if (getattr(be, "_SPARSE_DEC", False) or getattr(be, "_SPARSE_POOL", False)):
            sb = int(be._SPARSE_B)
            if bs % sb == 0:
                d = int(p["d"])
                rows, rb = bs // sb, hkv * d * 2
                plan.append(_Lazy("kbar", p, rows, rb))
                if float(getattr(be, "_SPARSE_GAMMA", 0.0)) > 0:
                    plan.append(_Lazy("devp", p, rows, rb))
        return plan

    def _gdn_plan(self, name: str, tensors) -> list[Any]:
        if not isinstance(tensors, (list, tuple)) or not tensors:
            raise RuntimeError(f"[kv-offload] слой GDN {name}: ожидался список [conv, ssm]")
        u8, sp = _u8(tensors[0])
        rs = []
        for t in tensors:
            if t.untyped_storage().data_ptr() != sp:
                raise RuntimeError(f"[kv-offload] слой GDN {name}: conv и ssm в разных буферах")
            rs.append(_rng_of(t, u8, sp))
        plan: list[Any] = _merge(rs)
        ssm = tensors[-1]
        if ssm.dtype == torch.int16:
            plan.append(_Lazy("scale", name, 1, 0))   # размер строки -- при первом обращении
        return plan

    def register_kv_caches(self, kv_caches: dict[str, Any]):
        self.kv_caches = kv_caches
        extents = []
        self.plan = []
        for gi, g in enumerate(self.kvc.kv_cache_groups):
            gp: list[Any] = []
            for name in g.layer_names:
                if name not in kv_caches:
                    raise RuntimeError(f"[kv-offload] слоя {name} нет среди kv_caches")
                if isinstance(g.kv_cache_spec, MambaSpec):
                    lp = self._gdn_plan(name, kv_caches[name])
                else:
                    lp = self._attn_plan(name, kv_caches[name])
                for r in lp:
                    if isinstance(r, _Rng):
                        a = r.u8.data_ptr() + r.base
                        extents.append((a, a + (r.nb - 1) * r.stride + r.length, name))
                gp.extend(lp)
            self.plan.append(gp)
        # Слои разных групп в ОДНОЙ памяти (обычный гибридный пул) копировать по номерам блоков
        # своей группы нельзя -- это одна и та же страница под двумя номерами.
        extents.sort()
        for (a0, a1, n0), (b0, b1, n1) in zip(extents, extents[1:]):
            if b0 < a1 and n0 != n1:
                raise RuntimeError(f"[kv-offload] слои {n0} и {n1} делят память -- нужен "
                                   "FA2SM70_SPLIT_POOLS=1")
        self._layout_host()

    def _scale_tab(self, name: str):
        t = getattr(self.kv_caches[name][-1], "fa2sm70_scales", None)
        if t is None:
            lay = self.cfg.compilation_config.static_forward_context.get(name)
            kvl = getattr(lay, "kv_cache", None) if lay is not None else None
            if kvl:
                e = kvl[0]
                if isinstance(e, (list, tuple)) and len(e) > 1:
                    t = getattr(e[1], "fa2sm70_scales", None)
        return t

    def _layout_host(self):
        off = 0
        for gp in self.plan:
            for r in gp:
                if isinstance(r, _Lazy) and r.kind == "scale":
                    # Шкалы int16 [блоков, HV, D1/32, 4] fp32 заводятся в capture_model, то есть
                    # ПОСЛЕ регистрации -- размер строки берём по той же формуле, что их создаёт,
                    # и сверяем с живой таблицей при первом обращении (_resolve).
                    import fa2sm70_gdn_i16 as i16
                    shp = i16.форма_масштабов(tuple(self.kv_caches[r.key][-1].shape))
                    n = 1
                    for x in shp[1:]:
                        n *= int(x)
                    r.row_bytes = n * 4
                r.host = off
                off += r.length
        self.slot_bytes = off
        if off > self.est:
            raise RuntimeError(f"[kv-offload] слот {off} Б больше оценки {self.est} Б -- "
                               "поправить est_slot_bytes, иначе бюджет ОЗУ будет превышен")

    def _resolve(self):
        """Ленивые таблицы -- один раз, на ОСНОВНОМ потоке (_kbar_pool может завести таблицу)."""
        be = _backend()
        for gp in self.plan:
            for r in gp:
                if not isinstance(r, _Lazy) or r.tab is not None:
                    continue
                if r.kind == "kbar":
                    t = be._kbar_pool(r.key, int(be._SPARSE_B))
                elif r.kind == "devp":
                    t = be._dev_pool(r.key, int(be._SPARSE_B))
                else:
                    t = self._scale_tab(r.key)
                if t is None:
                    raise RuntimeError(f"[kv-offload] таблица {r.kind} не найдена "
                                       f"({r.key if r.kind == 'scale' else 'пул внимания'}) -- "
                                       "копия блока была бы неполной")
                rb = t[0].numel() * t.element_size()          # одна строка таблицы
                if not t.is_contiguous() or rb != r.row_bytes:
                    raise RuntimeError(f"[kv-offload] таблица {r.kind}: строка {rb} Б, ожидалось "
                                       f"{r.row_bytes} Б (форма {tuple(t.shape)})")
                r.tab = t.view(-1).view(torch.uint8)
        self._registered = True

    def _ensure_host(self):
        if self.host is not None or self.n_slots == 0:
            return
        n = self.n_slots * self.slot_bytes
        t0 = time.monotonic()
        # Не torch pin_memory: кэширующий распределитель хоста округляет до степени двойки
        # (8.5 ГиБ -> 16 ГиБ закреплённых). Обычная память + cudaHostRegister закрепляет ровно n.
        h = torch.empty(n, dtype=torch.uint8)
        rc = torch.cuda.cudart().cudaHostRegister(h.data_ptr(), n, 0)
        if int(rc) != 0:
            raise RuntimeError(f"[kv-offload] cudaHostRegister({n}) = {rc}")
        self.host = h
        self.side = torch.cuda.Stream()
        logger.info("[kv-offload] ранг: слотов %d по %.1f МиБ (оценка %.1f), закреплено %.2f ГиБ "
                    "за %.1f с, pinned=%s", self.n_slots, self.slot_bytes / 2**20,
                    self.est / 2**20, n / 2**30, time.monotonic() - t0, h.is_pinned())

    # --- копии -------------------------------------------------------------------------------
    def _copy(self, slot: int, per: tuple[int, ...], to_host: bool):
        assert self.host is not None
        base = slot * self.slot_bytes
        for g, blk in enumerate(per):
            if blk < 0:
                continue
            for r in self.plan[g]:
                hv = self.host[base + r.host: base + r.host + r.length]
                if isinstance(r, _Rng):
                    if blk >= r.nb:
                        raise RuntimeError(f"[kv-offload] блок {blk} вне пула {r.nb}")
                    o = r.base + blk * r.stride
                    gv = r.u8[o: o + r.length]
                else:
                    o = blk * r.length
                    if o + r.length > r.tab.numel():
                        raise RuntimeError(f"[kv-offload] блок {blk} вне таблицы {r.kind}")
                    gv = r.tab[o: o + r.length]
                if to_host:
                    hv.copy_(gv, non_blocking=True)
                else:
                    gv.copy_(hv, non_blocking=True)

    def start_load_kv(self, meta: Fa2Sm70OffloadMeta):
        if not meta.saves and not meta.loads and self.prev_ev is None:
            return
        self._ensure_host()
        if self.host is None:
            return
        main = torch.cuda.current_stream()
        if self.prev_ev is not None:
            main.wait_event(self.prev_ev)
            self.prev_ev = None
        if not self._registered:
            self._resolve()
        if meta.saves:
            if SYNC:
                for s, per in meta.saves:
                    self._copy(s, per, to_host=True)
            else:
                ev0 = torch.cuda.Event()
                ev0.record(main)
                self.side.wait_event(ev0)
                with torch.cuda.stream(self.side):
                    for s, per in meta.saves:
                        self._copy(s, per, to_host=True)
                ev = torch.cuda.Event()
                ev.record(self.side)
                if meta.conflict or meta.loads:
                    main.wait_event(ev)
                    self.st["waits"] += 1
                else:
                    self.prev_ev = ev
            self.st["saves"] += len(meta.saves)
        for s, per in meta.loads:
            self._copy(s, per, to_host=False)
        self.st["loads"] += len(meta.loads)

    def shutdown(self):
        if self.host is not None and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaHostUnregister(self.host.data_ptr())
            except Exception:
                pass
        self.host = None


# ---------------------------------------------------------------------------------------------
# Коннектор
# ---------------------------------------------------------------------------------------------
class Fa2Sm70OffloadConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole,
                 kv_cache_config: "KVCacheConfig | None" = None):
        super().__init__(vllm_config, role, kv_cache_config)
        if kv_cache_config is None:
            raise ValueError("[kv-offload] нужен kv_cache_config (новая подпись коннектора)")
        self._s = _SchedSide(vllm_config, kv_cache_config) if role == KVConnectorRole.SCHEDULER else None
        self._w = _WorkerSide(vllm_config, kv_cache_config) if role == KVConnectorRole.WORKER else None

    # воркер
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self._w is not None
        self._w.register_kv_caches(kv_caches)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        assert self._w is not None
        meta = self._get_connector_metadata()
        assert isinstance(meta, Fa2Sm70OffloadMeta)
        self._w.start_load_kv(meta)

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: Any,
                      **kwargs: Any) -> None:
        return

    def wait_for_save(self):
        return

    def shutdown(self):
        if self._w is not None:
            self._w.shutdown()

    # планировщик
    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int):
        assert self._s is not None
        return self._s.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        assert self._s is not None
        self._s.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(self, scheduler_output: "SchedulerOutput") -> KVConnectorMetadata:
        assert self._s is not None
        return self._s.build_connector_meta(scheduler_output)

    def request_finished(self, request: "Request", block_ids: list[int]):
        assert self._s is not None
        return self._s.request_finished(request)

    def request_finished_all_groups(self, request: "Request", block_ids: tuple[list[int], ...]):
        assert self._s is not None
        return self._s.request_finished(request)
