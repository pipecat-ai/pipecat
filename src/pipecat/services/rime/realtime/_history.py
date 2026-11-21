#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Append-only Prism history writes."""

import asyncio

from rimelabs_sdk import RimeInputError
from rimelabs_sdk import realtime as r

from pipecat.services.rime.realtime._adapter import history_items


class PrismHistoryUpdateError(RuntimeError):
    """A rejected, partially applied, or uncertain history append.

    Attributes:
        status: ``"rejected"`` when nothing was written, ``"partial"`` when some
            items were written, or ``"uncertain"`` when the outcome is unknown.
        recoverable: Whether the conversation can continue.
    """

    def __init__(self, status: str):
        """Describe an append without exposing conversation text in its message."""
        super().__init__(f"Prism history update {status}")
        self.status = status
        self.recoverable = status != "uncertain"


class History:
    """Append context messages to one Prism conversation's history.

    Each append is validated as a whole before anything is written. Writes are
    serialized with tool result submission through :attr:`lock`.
    """

    def __init__(self, session: r.RealtimeSession):
        """Bind the history to its conversation."""
        self.session = session
        self.lock = asyncio.Lock()

    async def append(self, messages: list) -> None:
        """Write messages to the end of the provider history, in order."""
        async with self.lock:
            writes = 0
            try:
                for item in history_items(messages):
                    await self.session.create_item(item)
                    writes += 1
            except Exception as error:
                known = isinstance(error, (ValueError, TypeError, RimeInputError)) or (
                    isinstance(error, r.RimeRealtimeError) and error.fault.scope == "event"
                )
                status = ("partial" if writes else "rejected") if known else "uncertain"
                raise PrismHistoryUpdateError(status) from error
