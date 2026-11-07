#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the STTService keepalive."""

import asyncio

import pytest
from websockets.exceptions import ConnectionClosedError

from pipecat.services.stt_service import STTService
from pipecat.utils.asyncio.task_manager import TaskManager

KEEPALIVE_INTERVAL = 0.01


class KeepaliveSTTService(STTService):
    """Minimal STT service that records its keepalives."""

    def __init__(self, **kwargs):
        super().__init__(
            keepalive_timeout=KEEPALIVE_INTERVAL,
            keepalive_interval=KEEPALIVE_INTERVAL,
            **kwargs,
        )
        self.keepalives = 0
        self.failures_left = 0

    async def run_stt(self, audio: bytes):
        yield None

    async def _send_keepalive(self, silence: bytes):
        if self.failures_left:
            self.failures_left -= 1
            raise ConnectionClosedError(None, None)
        self.keepalives += 1


@pytest.mark.asyncio
async def test_keepalive_survives_failed_send():
    """A failed send is logged and later keepalives still go out."""
    service = KeepaliveSTTService()
    service._task_manager = TaskManager()
    service._sample_rate = 16000
    service.failures_left = 2
    service._create_keepalive_task()
    try:
        async with asyncio.timeout(1):
            while service.keepalives < 2:
                await asyncio.sleep(KEEPALIVE_INTERVAL)
        assert service.failures_left == 0
        assert not service._keepalive_task.done()
    finally:
        await service._cancel_keepalive_task()
