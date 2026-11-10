#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Prism history identities and acknowledged edit planning."""

import asyncio
import copy
import json
from dataclasses import dataclass, field
from typing import cast
from uuid import uuid4

from rimelabs_sdk import RimeInputError
from rimelabs_sdk import realtime as r

from pipecat.processors.aggregators.async_tool_messages import parse_message
from pipecat.processors.aggregators.llm_context import LLMStandardMessage
from pipecat.services.rime.realtime._adapter import history_items
from pipecat.utils.string import TextPartForConcatenation, concatenate_aggregated_text


class PrismHistoryUpdateError(RuntimeError):
    """A rejected, partial, or uncertain provider history edit.

    Attributes:
        status: Whether the edit was rejected, partially applied, or uncertain.
        recoverable: Whether independent operations can continue.
        confirmed_context: Detached provider projection, or None when unknown.
    """

    def __init__(self, status: str, confirmed_context: list | None):
        """Describe an edit without exposing conversation text in its message."""
        super().__init__(f"Prism history update {status}")
        self.status = status
        self.recoverable = status != "uncertain"
        self.confirmed_context = copy.deepcopy(confirmed_context)


@dataclass
class Record:
    """One local context record and its separate server identities."""

    message: dict
    server_ids: list[str] = field(default_factory=list)
    bound: bool = True
    identity: str = field(default_factory=lambda: uuid4().hex)
    accepted: dict = field(init=False)
    variants: list[dict] = field(default_factory=list)
    texts: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self.accepted = copy.deepcopy(self.message)
        if len(self.server_ids) == 1 and self.message.get("role") == "assistant":
            if isinstance(self.message.get("content"), str):
                self.texts[self.server_ids[0]] = self.message["content"]

    def matches(self, message: dict, *, shared: bool = False) -> bool:
        """Match tool progress aliases only when omitted provider records are preserved."""
        return _equivalent(self.accepted, message) or (
            (shared or self.accepted.get("role") != "tool")
            and any(_equivalent(variant, message) for variant in self.variants)
        )


class History:
    """Own conversion and mutations for one SDK conversation.

    Explicit context replacements match provider items by content and order.
    Transcript aggregation never changes provider history. Edits have no rollback.
    """

    def __init__(self, session: r.RealtimeSession):
        """Bind the ledger to its conversation."""
        self.session = session
        self.records: list[Record] = []
        self.active_inputs: set[str] = set()
        self.active_responses: set[str] = set()
        self.pending_calls: set[str] = set()
        self.response_epoch = 0
        self._inputs: dict[str, Record] = {}
        self.lock = asyncio.Lock()
        self._writes = 0

    def reserve_input(self, item_id: str) -> Record:
        """Reserve provider speech order before its transcript arrives."""
        if item_id not in self._inputs:
            if len(self._inputs) >= 4096:
                raise RuntimeError("Prism input identity retention limit exceeded")
            record = Record({"role": "user", "content": ""}, [item_id], bound=False)
            self.records.append(record)
            self._inputs[item_id] = record
        return self._inputs[item_id]

    def transcript(self, item_id: str, text: str) -> None:
        """Record a provider transcript without changing the aggregator's context."""
        record = self.reserve_input(item_id)
        record.message["content"] = text
        record.accepted = copy.deepcopy(record.message)

    def add_live(self, message: dict, server_ids: list[str]) -> None:
        """Track provider items until normal aggregation records them locally."""
        self.records.append(Record(message, server_ids, bound=False))

    async def sync(self, messages: list, *, shared: bool = False) -> None:
        """Apply an explicit context snapshot while retaining matching provider items.

        Identical replacements cannot be distinguished from retained messages.
        """
        known_calls = {
            call["id"] for record in self.records for call in record.accepted.get("tool_calls", [])
        }
        filtered = []
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("Prism does not support opaque provider history")
            envelope = parse_message(cast(LLMStandardMessage, message))
            if envelope and envelope.tool_call_id in known_calls:
                if envelope.kind == "final":
                    content = envelope.result or "COMPLETED"
                    filtered.append(
                        {
                            "role": "tool",
                            "tool_call_id": envelope.tool_call_id,
                            "content": "CANCELLED" if content.startswith("CANCELLED") else content,
                        }
                    )
                continue
            if message.get("role") == "tool" and (
                message.get("tool_call_id") in self.pending_calls
                or (
                    message.get("tool_call_id") in known_calls
                    and message.get("content") == "IN_PROGRESS"
                )
            ):
                continue
            filtered.append(message)
        matches = []
        used = set()
        for message in filtered:
            self._coalesce(message)
            match = next(
                (
                    record
                    for record in self.records
                    if record.identity not in used and record.matches(message, shared=shared)
                ),
                None,
            )
            matches.append(match)
            if match is not None:
                used.add(match.identity)
                match.message = message
                match.bound = True
        if shared:
            # An ordinary context is a view. Only an append may introduce new items.
            first_new = next(
                (i for i, record in enumerate(matches) if record is None), len(matches)
            )
            if any(record is not None for record in matches[first_new:]):
                raise PrismHistoryUpdateError("rejected", self.projection())
            if first_new < len(matches):
                await self.edit(filtered[first_new:], append=True)
            return
        bound = list(self.records)
        retained = [record for record in matches if record is not None]
        if len(retained) == len(filtered) and bound == retained:
            return
        prefix = len(retained)
        if bound == retained and all(record is not None for record in matches[:prefix]):
            await self.edit(filtered[prefix:], append=True)
            return
        base = tuple(record.message for record in bound)
        indices = {record.identity: index for index, record in enumerate(bound)}
        await self.edit(
            filtered,
            append=False,
            base=base,
            retained=tuple(
                indices[record.identity] if record is not None else None for record in matches
            ),
        )

    def _coalesce(self, message: dict) -> None:
        """Match consecutive live text items merged by the standard aggregator."""
        role = message.get("role")
        if role not in ("user", "assistant") or not isinstance(message.get("content"), str):
            return
        group = []
        for record in tuple(self.records):
            if (
                record.bound
                or record.accepted.get("role") != role
                or not record.message.get("content")
                or record.accepted.get("tool_calls")
            ):
                group = []
                continue
            group.append(record)
            text = concatenate_aggregated_text(
                [
                    TextPartForConcatenation(
                        part.accepted["content"], includes_inter_part_spaces=False
                    )
                    for part in group
                ]
            )
            generated = concatenate_aggregated_text(
                [
                    TextPartForConcatenation(
                        part.message["content"], includes_inter_part_spaces=False
                    )
                    for part in group
                ]
            )
            if len(group) > 1 and message["content"] in (text, generated):
                first = group[0]
                first.texts = {key: value for part in group for key, value in part.texts.items()}
                first.message = copy.deepcopy(message)
                first.accepted = {**copy.deepcopy(message), "content": text}
                if generated != text:
                    first.variants.append({**copy.deepcopy(message), "content": generated})
                first.server_ids = [item_id for part in group for item_id in part.server_ids]
                removed = {part.identity for part in group[1:]}
                self.records[:] = [part for part in self.records if part.identity not in removed]
                return
            if not any(message["content"].startswith(value) for value in (text, generated)):
                group = [record]

    def projection(self) -> list:
        """Return a detached projection of confirmed provider records."""
        return copy.deepcopy([record.accepted for record in self.records])

    def tool_variant(self, call_id: str, output: str) -> None:
        """Match the application's latest result to an already recorded progress pair."""
        for record in self.records:
            if record.accepted.get("tool_call_id") == call_id:
                record.variants.append({**record.accepted, "content": output})
                record.variants = record.variants[-8:]
                return

    def truncate(self, event: r.ItemTruncated) -> None:
        """Update confirmed retained text without replacing application context."""
        if event.kept_text is None or event.content_index != 0:
            return
        for record in self.records:
            if event.item_id in record.server_ids and record.accepted.get("role") == "assistant":
                record.variants.append(copy.deepcopy(record.accepted))
                record.variants = record.variants[-8:]
                record.texts[event.item_id] = event.kept_text
                retained = concatenate_aggregated_text(
                    [
                        TextPartForConcatenation(
                            record.texts[item_id], includes_inter_part_spaces=False
                        )
                        for item_id in record.server_ids
                        if item_id in record.texts
                    ]
                )
                record.accepted = {**record.accepted, "content": retained}
                return

    async def edit(
        self, messages: list, *, append: bool, base: tuple | None = None, retained: tuple = ()
    ) -> None:
        """Apply edits and report confirmed state after a recoverable refusal."""
        async with self.lock:
            self._writes = 0
            try:
                await self._edit(messages, append=append, base=base, retained=retained)
            except Exception as error:
                known = isinstance(error, (ValueError, TypeError, RimeInputError)) or (
                    isinstance(error, r.RimeRealtimeError) and error.fault.scope == "event"
                )
                if known and self._writes:
                    try:
                        await self._repair()
                    except Exception:
                        known = False
                status = ("partial" if self._writes else "rejected") if known else "uncertain"
                raise PrismHistoryUpdateError(
                    status, self.projection() if known else None
                ) from error

    async def _repair(self) -> None:
        """Read confirmed order once; active omissions cannot establish a replacement."""
        epoch = self.response_epoch
        snapshot = await self.session.get_history()
        if self.active_responses or self.pending_calls or epoch != self.response_epoch:
            raise RuntimeError("Cannot establish completed history order during active work")
        records = []
        for item in snapshot:
            if isinstance(item, r.HistoryMessage):
                message = {"role": item.role, "content": item.text}
            elif isinstance(item, r.HistoryFunctionCall):
                message = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": item.call_id,
                            "type": "function",
                            "function": {"name": item.name, "arguments": item.arguments},
                        }
                    ],
                }
            else:
                message = {"role": "tool", "tool_call_id": item.call_id, "content": item.output}
            records.append(Record(message, [item.id] if item.id else []))
        self.records = records
        self._inputs.clear()

    async def _edit(
        self, messages: list, *, append: bool, base: tuple | None = None, retained: tuple = ()
    ) -> None:
        """Validate every item before applying append, retained-ID delete, or rebuild.

        New arrivals are outside ``base`` and stay in the ledger. A rebuild is
        rejected while a reply or tool call is active.
        """
        items = history_items(messages)
        existing = {id(record.message): record for record in self.records}
        base_ids = {id(message) for message in base} if base is not None else None
        old = [
            record for record in self.records if base_ids is None or id(record.message) in base_ids
        ]
        retained_records = (
            [existing.get(id(base[index])) if index is not None else None for index in retained]
            if retained and base is not None
            else [None] * len(messages)
        )
        retained_records = [
            record if record and record.matches(message) else None
            for record, message in zip(retained_records, messages)
        ]
        retained_ids = [record.identity for record in retained_records if record is not None]
        retained_id_set = set(retained_ids)
        retained_order = [record.identity for record in old if record.identity in retained_id_set]
        can_delete = not append and all(retained_records) and retained_order == retained_ids
        rebuild = not append and not can_delete
        if rebuild and (self.active_responses or self.pending_calls):
            raise ValueError("Prism history rebuild requires no active replies or calls")
        remove = (
            old
            if rebuild
            else [record for record in old if record.identity not in retained_id_set]
            if can_delete
            else []
        )
        deleted_calls = {
            call["id"] for record in remove for call in record.message.get("tool_calls", [])
        }
        remove.extend(
            record
            for record in self.records
            if record not in remove
            and record.message.get("role") == "tool"
            and record.message.get("tool_call_id") in deleted_calls
        )
        removed_ids = {item_id for record in remove for item_id in record.server_ids}
        if removed_ids & self.active_inputs or any(
            call.get("id") in self.pending_calls
            for record in remove
            for call in record.message.get("tool_calls", [])
        ):
            raise ValueError("Cannot remove active inputs or pending calls")
        epoch = self.response_epoch
        deleted_calls = {
            call["id"] for record in remove for call in record.message.get("tool_calls", [])
        }
        for record in remove:
            if (
                record.message.get("role") == "tool"
                and record.message.get("tool_call_id") in deleted_calls
            ):
                continue
            for item_id in tuple(record.server_ids):
                self._check_epoch(epoch, rebuild)
                await self.session.delete_item(item_id)
                self._writes += 1
                record.server_ids.remove(item_id)
                if not record.server_ids:
                    self.records[:] = [value for value in self.records if value is not record]
                self._inputs.pop(item_id, None)
            for call in record.accepted.get("tool_calls", []):
                self.records[:] = [
                    value
                    for value in self.records
                    if value.accepted.get("tool_call_id") != call["id"]
                ]
        if can_delete:
            for record, message in zip(retained_records, messages):
                assert record is not None
                record.message = message
        else:
            anchor = "root" if rebuild and self.records else None
            offset = 0
            insertion = 0 if rebuild else len(self.records)
            for message in messages:
                converted = history_items_for_record(message, items, offset)
                offset += len(converted)
                record = Record(message)
                for item in converted:
                    self._check_epoch(epoch, rebuild)
                    ref = await self.session.create_item(
                        item,
                        previous_item_id=anchor
                        if not isinstance(item, r.HistoryFunctionResult)
                        else None,
                    )
                    self._writes += 1
                    if anchor is not None:
                        anchor = ref.item_id
                    record.server_ids.append(ref.item_id)
                    if len(record.server_ids) == 1:
                        self.records.insert(insertion, record)
                        insertion += 1
        self._check_epoch(epoch, rebuild)

    def _check_epoch(self, epoch: int, rebuild: bool) -> None:
        if rebuild and self.response_epoch != epoch:
            raise ValueError("A reply started during the history rebuild")


def history_items_for_record(message: dict, items: list, offset: int) -> list:
    """Take the already-validated SDK items belonging to one context dictionary."""
    calls = message.get("tool_calls", [])
    has_text = message.get("content") is not None and (bool(message.get("content")) or not calls)
    count = 1 if message.get("role") == "tool" else len(calls) + has_text
    return items[offset : offset + count]


def _equivalent(left: dict, right: dict) -> bool:
    """Ignore JSON argument formatting and absent empty assistant content."""

    def canonical(message):
        value = copy.deepcopy(message)
        if value.get("tool_calls"):
            if not value.get("content"):
                value.pop("content", None)
            for call in value["tool_calls"]:
                call.setdefault("type", "function")
                arguments = call["function"].get("arguments", "{}")
                try:
                    call["function"]["arguments"] = json.loads(arguments)
                except (ValueError, TypeError):
                    pass
        return value

    return canonical(left) == canonical(right)
