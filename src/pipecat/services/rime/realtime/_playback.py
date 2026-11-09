#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Ordered zero-played receipts for conversations that discard all audio."""

import asyncio

from rimelabs_sdk import realtime as r


class DiscardPlayback:
    """Report completed discarded responses in their creation order."""

    def __init__(self, sdk: r.RealtimeSession):
        self._sdk = sdk
        self._pending: dict[str, tuple[r.ResponseRef, r.ResponseHold, bool]] = {}
        self._lock = asyncio.Lock()

    def start(self, response: r.ResponseRef) -> None:
        """Keep a response until its receipt can follow all earlier output."""
        if len(self._pending) >= 128:
            raise RuntimeError("Prism pending discard receipt limit exceeded")
        self._pending[response.response_id] = response, self._sdk.hold_response(response), False

    def end(self, response: r.ResponseRef) -> None:
        """Record completion before another reply can start admission."""
        entry = self._pending.get(response.response_id)
        if entry is None:
            return
        self._pending[response.response_id] = entry[0], entry[1], True

    async def settle(self) -> None:
        """Send receipts once this and all earlier responses have ended."""
        async with self._lock:
            while self._pending:
                key, (ref, hold, ended) = next(iter(self._pending.items()))
                if not ended:
                    break
                try:
                    await self._sdk.report_playback(r.PlaybackFinished(response=ref, played_ms=0))
                finally:
                    self._pending.pop(key, None)
                    hold.release()

    def close(self) -> None:
        """Release pending references without claiming playback completion."""
        for _, hold, _ in self._pending.values():
            hold.release()
        self._pending.clear()
