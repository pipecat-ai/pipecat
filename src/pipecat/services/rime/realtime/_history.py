#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Prism history identities and acknowledged edit planning."""

import copy
import time
from dataclasses import dataclass, field
from uuid import uuid4

from rimelabs_sdk import realtime as r

from pipecat.adapters.llm.services.rime_realtime_adapter import history_items


class PartialHistoryError(RuntimeError):
    """Some history mutations committed before an edit failed."""


@dataclass
class Record:
    """One local context record and its separate server identities."""

    message: dict
    server_ids: list[str] = field(default_factory=list)
    imported: bool = True
    created_at: float = field(default_factory=time.monotonic)
    identity: str = field(default_factory=lambda: uuid4().hex)
    accepted: dict = field(init=False)

    def __post_init__(self):
        self.accepted = copy.deepcopy(self.message)


class History:
    """Own conversion and mutations for one SDK conversation.

    Records retain the original context dictionaries. Replacements with no object
    identity require a rebuild, including replacements with identical text.
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

    def reserve_input(self, item_id: str, context) -> Record:
        """Reserve speech order before its transcript arrives."""
        if item_id not in self._inputs:
            if len(self._inputs) >= 4096:
                raise RuntimeError("Prism input identity retention limit exceeded")
            message = {"role": "user", "content": ""}
            record = Record(message, [item_id], imported=False)
            self.records.append(record)
            self._inputs[item_id] = record
            context.add_message(message)
        return self._inputs[item_id]

    def transcript(self, item_id: str, text: str, context) -> None:
        """Update the reserved user record without adding a second message."""
        record = self.reserve_input(item_id, context)
        record.message["content"] = text
        record.accepted = copy.deepcopy(record.message)

    def add_live(self, message: dict, server_ids: list[str], context) -> None:
        """Record a server-created item in the Pipecat conversation."""
        self.records.append(Record(message, server_ids, imported=False))
        context.add_message(message)

    def edit_base(self, base: tuple, captured_at: float) -> tuple:
        """Include preceding accepted mutations while retaining later live arrivals."""
        identities = {id(message) for message in base}
        return (
            *base,
            *(
                record.message
                for record in self.records
                if (record.imported or record.created_at <= captured_at)
                and id(record.message) not in identities
            ),
        )

    def matches(self, messages: list, base: tuple, captured_at: float) -> bool:
        """Check both dictionary identity and accepted content for a context snapshot."""
        return (
            len(base) <= len(self.records)
            and all(record.created_at > captured_at for record in self.records[len(base) :])
            and all(
                record.message is message and record.accepted == value
                for record, message, value in zip(self.records, base, messages)
            )
        )

    def restore(self, context) -> None:
        """Restore accepted records before submitting an external context replacement."""
        for record in self.records:
            if record.message != record.accepted:
                record.message.clear()
                record.message.update(copy.deepcopy(record.accepted))
        context.set_messages([record.message for record in self.records])

    async def edit(
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
            record if record and record.accepted == message else None
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
        removed_records = {record.identity for record in remove}
        removed_ids = {item_id for record in remove for item_id in record.server_ids}
        if removed_ids & self.active_inputs or any(
            call.get("id") in self.pending_calls
            for record in remove
            for call in record.message.get("tool_calls", [])
        ):
            raise ValueError("Cannot remove active inputs or pending calls")
        epoch = self.response_epoch
        changed = 0
        try:
            deleted_calls = {
                call["id"] for record in remove for call in record.message.get("tool_calls", [])
            }
            for record in remove:
                if (
                    record.message.get("role") == "tool"
                    and record.message.get("tool_call_id") in deleted_calls
                ):
                    continue
                for item_id in record.server_ids:
                    self._check_epoch(epoch, rebuild)
                    await self.session.delete_item(item_id)
                    changed += 1
            new_records = []
            if not can_delete:
                preserved = [
                    record for record in self.records if record.identity not in removed_records
                ]
                anchor = "root" if rebuild and preserved else None
                offset = 0
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
                        if anchor is not None:
                            anchor = ref.item_id
                        changed += 1
                        record.server_ids.append(ref.item_id)
                    new_records.append(record)
            self._check_epoch(epoch, rebuild)
        except Exception as error:
            if changed:
                raise PartialHistoryError("Prism history edit partially applied") from error
            raise
        self.records[:] = [
            record for record in self.records if record.identity not in removed_records
        ]
        if append:
            self.records.extend(new_records)
        elif can_delete:
            for record, message in zip(retained_records, messages):
                assert record is not None
                record.message = message
        else:
            self.records[:] = [*new_records, *self.records]

    def _check_epoch(self, epoch: int, rebuild: bool) -> None:
        if rebuild and self.response_epoch != epoch:
            raise ValueError("A reply started during the history rebuild")

    def completed_messages(self) -> list[dict]:
        """Return completed records suitable for an explicit restart."""
        if self.pending_calls or self.active_responses:
            raise ValueError("Cannot restart with pending calls or active replies")
        messages = [
            copy.deepcopy(record.message)
            for record in self.records
            if record.message.get("content") or record.message.get("tool_calls")
        ]
        history_items(messages)
        return messages


def history_items_for_record(message: dict, items: list, offset: int) -> list:
    """Take the already-validated SDK items belonging to one context dictionary."""
    calls = message.get("tool_calls", [])
    has_text = message.get("content") is not None and (bool(message.get("content")) or not calls)
    count = 1 if message.get("role") == "tool" else len(calls) + has_text
    return items[offset : offset + count]
