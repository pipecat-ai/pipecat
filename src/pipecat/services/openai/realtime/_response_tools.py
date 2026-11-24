#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Response-scoped client tool detection for the OpenAI Realtime wire protocol."""

from collections import OrderedDict
from dataclasses import dataclass, field
from itertools import chain
from typing import Any


@dataclass
class _Response:
    items: set[str] = field(default_factory=set)
    calls: set[str | None] = field(default_factory=set)


class RealtimeResponseTools:
    """Track the active responses and their output items on one WebSocket."""

    def __init__(self):
        self._active: dict[str | None, _Response] = {}
        # A cancelled or completed response may still have deltas in flight.
        self._recent: OrderedDict[str | None, _Response] = OrderedDict()

    def _response(self, event: Any) -> _Response | None:
        if (response_id := getattr(event, "response_id", None)) is not None:
            return self._active.get(response_id) or self._recent.get(response_id)
        item_id = getattr(event, "item_id", None)
        call_id = getattr(event, "call_id", None)
        if item_id is not None or call_id is not None:
            for response in chain(self._active.values(), reversed(self._recent.values())):
                if item_id in response.items or (call_id is not None and call_id in response.calls):
                    return response
            return None
        # Some compatible servers omit the IDs on argument and text deltas.
        return next(iter(self._active.values())) if len(self._active) == 1 else None

    def observe(self, event: Any) -> None:
        """Record wire boundaries and client function calls, including overlapping responses."""
        if event.type in ("response.created", "response.done"):
            response = event.response
            response_id = response.get("id") if isinstance(response, dict) else response.id
            if event.type == "response.created":
                self._active[response_id] = _Response()
            elif (finished := self._active.pop(response_id, None)) is not None:
                self._recent[response_id] = finished
                if len(self._recent) > 64:
                    self._recent.popitem(last=False)
            return

        if event.type in ("response.output_item.added", "response.output_item.done"):
            item_id = event.item.id
            has_tool = event.item.type == "function_call"
            call_id = event.item.call_id
        elif event.type in (
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
        ):
            item_id = getattr(event, "item_id", None)
            call_id = event.call_id
            has_tool = True
        else:
            return

        response = self._response(event)
        if response is None:
            return
        if item_id is not None:
            response.items.add(item_id)
        if has_tool:
            response.calls.add(call_id)

    def has_tool_call(self, event: Any) -> bool:
        """Whether a client function has appeared in the response that owns this text."""
        response = self._response(event)
        return response is not None and bool(response.calls)
