#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Websocket output transports pace each audio chunk for its duration."""

import unittest
from unittest.mock import AsyncMock

from pipecat.transports.websocket.client import (
    WebsocketClientOutputTransport,
    WebsocketClientParams,
)
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketOutputTransport,
    FastAPIWebsocketParams,
)
from pipecat.transports.websocket.server import (
    SingleClientWebsocketServerParams,
    SingleClientWebsocketServerTransport,
)
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.frame_processor_helpers import frame_processor_setup

_SAMPLE_RATE = 16000
_CHUNKS = 4


def _expected_interval(sample_rate: int, channels: int, chunks: int = _CHUNKS) -> float:
    """Seconds of 16-bit audio in one output chunk."""
    audio_bytes_10ms = int(sample_rate / 100) * channels * 2
    return (audio_bytes_10ms * chunks) / (sample_rate * channels * 2)


class TestWebsocketSendInterval(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._task_manager = TaskManager()

    def _setup(self, sample_rate: int):
        return frame_processor_setup(self._task_manager, audio_out_sample_rate=sample_rate)

    async def test_server_mono_interval_matches_the_chunk(self):
        output = SingleClientWebsocketServerTransport(
            SingleClientWebsocketServerParams(
                audio_out_enabled=True,
                audio_out_sample_rate=_SAMPLE_RATE,
                audio_out_channels=1,
                audio_out_10ms_chunks=_CHUNKS,
            )
        ).output()
        await output.setup(self._setup(_SAMPLE_RATE))

        self.assertEqual(output.audio_chunk_size, 1280)
        self.assertAlmostEqual(output._send_interval, _expected_interval(_SAMPLE_RATE, 1))

    async def test_server_stereo_interval_is_not_doubled(self):
        output = SingleClientWebsocketServerTransport(
            SingleClientWebsocketServerParams(
                audio_out_enabled=True,
                audio_out_sample_rate=_SAMPLE_RATE,
                audio_out_channels=2,
                audio_out_10ms_chunks=_CHUNKS,
            )
        ).output()
        await output.setup(self._setup(_SAMPLE_RATE))

        # 40ms of stereo is twice the bytes of mono, and the same amount of time.
        self.assertEqual(output.audio_chunk_size, 2560)
        self.assertAlmostEqual(output._send_interval, 0.04)
        self.assertAlmostEqual(output._send_interval, _expected_interval(_SAMPLE_RATE, 2))

    async def test_fastapi_and_client_use_the_same_interval(self):
        for output_cls, params_cls in (
            (FastAPIWebsocketOutputTransport, FastAPIWebsocketParams),
            (WebsocketClientOutputTransport, WebsocketClientParams),
        ):
            params = params_cls(
                audio_out_enabled=True,
                audio_out_sample_rate=48000,
                audio_out_channels=2,
                audio_out_10ms_chunks=_CHUNKS,
            )
            output = output_cls(AsyncMock(), AsyncMock(), params)
            await output.setup(self._setup(48000))
            self.assertAlmostEqual(
                output._send_interval, _expected_interval(48000, 2), msg=output_cls.__name__
            )
