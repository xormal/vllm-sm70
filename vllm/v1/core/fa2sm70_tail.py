# SPDX-License-Identifier: Apache-2.0
"""НЕПРЕРЫВНЫЙ ХВОСТ (fa2_sm70): передача неполного последнего блока следующему ходу диалога.

ЗАЧЕМ. Префикс-кэш vLLM переиспользует только ЦЕЛЫЕ блоки, а на гибридной сети блок навязан
частным «страница состояния GDN / байты KV на токен» и равен 1568 токенам. Поэтому КАЖДЫЙ ход
диалога пересчитывает неполный хвост -- замерено 604..1468 токенов, матожидание 784, цена
~1.25 мс/токен, то есть 0.8-1.8 с на ход. Работа не медленная, она ЛИШНЯЯ.

ЗАМЫСЕЛ. Хвост не РАЗДЕЛЯЕТСЯ, а ПЕРЕДАЁТСЯ. Законный потребитель ровно один -- следующий ход того
же диалога, чей текст есть продолжение предыдущего. При завершении запроса запоминаем ключ хвоста;
следующий запрос, чьи токены совпали, ЗАБИРАЕТ запись (она тут же изымается), и владелец, который
дописывает в блок, остаётся единственным. Гонка двух запросов за один хвост разрешается тем, что
второй записи не находит и считает как раньше: корректность от этого не зависит, только скорость.

ПОЧЕМУ НЕ РАЗДЕЛЯТЬ. Совпадение хэша означает совпадение первых L токенов, поэтому ЧТЕНИЕ разделять
было бы безопасно. Небезопасна ЗАПИСЬ: два владельца дописывали бы разные токены в одни и те же
слоты после позиции L и читали бы чужое. Передача снимает это без копирования при записи.

Модуль намеренно не знает ни про torch, ни про пул блоков: только ключи, длины и учёт. Всё, что
трогает блоки, живёт на стороне вызывающего.
"""

from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Any

# Гейт. Умолчание ВЫКЛЮЧЕНО: боевой сервер идёт из закреплённого слепка и не должен зависеть от
# незавершённой правки. Читается ОДИН раз на процесс -- значит A/B делается разными подъёмами,
# а не переключением на лету (см. грабли «гейт в static const»).
ENABLED: bool = os.environ.get("FA2SM70_TAIL_CACHE") == "1"

# Сколько диалогов помним. Каждая запись держит ОДИН индекс блока, а он на этой сети стоит
# 48.2 МиБ (12.2 внимание по 16 слоям + 36.0 состояние GDN по 48 слоям). Умолчание 8 -> 0.38 ГиБ.
CAPACITY: int = int(os.environ.get("FA2SM70_TAIL_SLOTS", "8"))

# Отрицательный контроль: намеренно испортить восстановление, чтобы доказать СИЛУ проверки.
POISON: bool = os.environ.get("FA2SM70_TAIL_POISON") == "1"


class TailKey(tuple):
    """Ключ хвоста: (хэш последнего ПОЛНОГО блока или None, длина хвоста, хэш хвостовых токенов).

    Длина входит в ключ ЯВНО. Без неё два разных хвоста с общим началом дали бы один ключ, и мы
    вернули бы чужое состояние -- дефект класса «ключ по адресу вместо содержимого», который в этом
    проекте уже стоил суток.
    """

    __slots__ = ()


def _default_hash(obj: Any) -> bytes:
    """ДЕТЕРМИНИРОВАННЫЙ хэш. Встроенный `hash()` рандомизирован на процесс, и ключ, построенный на
    нём, разъехался бы между воркерами и между подъёмами -- дефект, который не падает, а молча не
    попадает."""
    return hashlib.sha256(repr(obj).encode("utf-8")).digest()


def make_tail_key(
    parent_block_hash: Any,
    tail_token_ids: Sequence[int],
    hash_fn: Callable[[Any], bytes] | None = None,
) -> TailKey:
    """Ключ по СОДЕРЖИМОМУ: родительский хэш + сами токены хвоста + их число."""
    fn = hash_fn or _default_hash
    n = len(tail_token_ids)
    digest = fn((parent_block_hash, tuple(tail_token_ids), ("fa2sm70_tail", n)))
    return TailKey((parent_block_hash, n, digest))


class TailEntry:
    """Запись хвоста. `blocks_by_group` -- по одному списку блоков на группу KV-кэша.

    `num_tokens` -- ТОЧНОЕ число токенов, которому соответствуют и содержимое блока внимания, и
    состояние GDN. Именно оно станет `num_computed_tokens` у забравшего, то есть здесь мы выражаем
    «сколько посчитано» в ТОКЕНАХ, а не в блоках -- снимаемое допущение vLLM.
    """

    __slots__ = ("key", "num_tokens", "blocks_by_group", "req_id", "tail_token_ids")

    def __init__(
        self,
        key: TailKey,
        num_tokens: int,
        blocks_by_group: tuple[list[Any], ...],
        req_id: str,
        tail_token_ids: Sequence[int] = (),
    ) -> None:
        # Сами токены хвоста держим РЯДОМ С ХЭШЕМ: при несовпадении они дают позицию первого
        # расхождения, а хэш даёт только «не то». Тысяча int -- ничто против 48 МиБ блока.
        self.tail_token_ids = tuple(tail_token_ids)
        self.key = key
        self.num_tokens = num_tokens
        self.blocks_by_group = blocks_by_group
        self.req_id = req_id

    def __repr__(self) -> str:  # pragma: no cover -- только для журнала
        return f"TailEntry(len={self.num_tokens}, req={self.req_id})"


class TailRegistry:
    """Реестр хвостов с вытеснением по давности. Ничего не освобождает сам.

    Освобождение блоков -- забота вызывающего: он передаёт `on_evict`, который вернёт блоки в пул.
    Так модуль остаётся проверяемым без пула и без карты.
    """

    def __init__(
        self,
        capacity: int = CAPACITY,
        on_evict: Callable[[TailEntry], None] | None = None,
    ) -> None:
        self.capacity = max(0, capacity)
        self._on_evict = on_evict
        self._entries: OrderedDict[TailKey, TailEntry] = OrderedDict()
        # Счётчики маршрута. Пишутся ПО ФАКТУ работы -- наличие модуля доказательством не является.
        self.stats: dict[str, int] = {
            "registered": 0,
            "hit": 0,
            "miss": 0,
            "evicted": 0,
            "replaced": 0,
        }

    def __len__(self) -> int:
        return len(self._entries)

    def register(self, entry: TailEntry) -> TailEntry | None:
        """Запомнить хвост. Возвращает ВЫТЕСНЕННУЮ запись (её блоки вызывающий обязан освободить).

        При нулевой ёмкости запись не принимается и немедленно возвращается назад: вызывающий
        освободит блоки тем же путём, что и при вытеснении. Молча терять блоки нельзя -- это утечка
        пула, которая проявится через часы как «нет свободных блоков».
        """
        if self.capacity == 0:
            return entry

        old = self._entries.pop(entry.key, None)
        if old is not None:
            self.stats["replaced"] += 1

        self._entries[entry.key] = entry
        self.stats["registered"] += 1

        if old is None and len(self._entries) > self.capacity:
            _, old = self._entries.popitem(last=False)
            self.stats["evicted"] += 1

        if old is not None and self._on_evict is not None:
            self._on_evict(old)
            return None
        return old

    def take(self, key: TailKey) -> TailEntry | None:
        """Найти и ИЗЪЯТЬ запись: у хвоста остаётся единственный владелец.

        Именно изъятие, а не просмотр, делает передачу безопасной без копирования при записи.
        """
        entry = self._entries.pop(key, None)
        if entry is None:
            self.stats["miss"] += 1
            return None
        self.stats["hit"] += 1
        return entry

    def tail_lengths(self, parent_block_hash: Any) -> list[int]:
        """Длины хвостов, записанных под этим родителем -- от ДЛИННОГО к короткому.

        Длина в ключ входит явно, поэтому перебирать приходится только реально записанные варианты
        (их единицы), а не все возможные.
        """
        return sorted(
            {k[1] for k in self._entries if k[0] == parent_block_hash}, reverse=True
        )

    def tokens_of(self, parent_block_hash: Any, n: int) -> tuple[int, ...] | None:
        """Токены записанного хвоста -- только для диагностики расхождения."""
        for k, e in self._entries.items():
            if k[0] == parent_block_hash and k[1] == n:
                return e.tail_token_ids
        return None

    def drop(self, key: TailKey) -> TailEntry | None:
        """Изъять запись, не считая это попаданием (например, при сбросе префикс-кэша)."""
        return self._entries.pop(key, None)

    def clear(self) -> list[TailEntry]:
        """Опустошить реестр, вернув все записи вызывающему на освобождение."""
        out = list(self._entries.values())
        self._entries.clear()
        return out
