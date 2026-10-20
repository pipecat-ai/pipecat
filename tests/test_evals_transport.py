#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the eval transport's per-connection flags, over its server and over an accepted WebSocket."""

import asyncio
import types
import unittest
from unittest.mock import AsyncMock, MagicMock

from pipecat.evals.fastapi_transport import (
    EvalFastAPIWebsocketInputTransport,
    EvalFastAPIWebsocketOutputTransport,
    EvalFastAPIWebsocketTransport,
)
from pipecat.evals.serializer import EvalConnectionFlags, EvalSerializer
from pipecat.evals.transport import (
    CAPTURE_AUDIO_QUERY_PARAM,
    CAPTURE_IMAGES_QUERY_PARAM,
    SKIP_TTS_QUERY_PARAM,
    EvalTransport,
    EvalTransportParams,
    _query_flag,
)
from pipecat.frames.frames import LLMConfigureOutputFrame


def _ws(path=None, request_path=None):
    """A minimal stand-in for a websockets connection object."""
    request = types.SimpleNamespace(path=request_path) if request_path is not None else None
    return types.SimpleNamespace(path=path, request=request)


class TestQueryFlag(unittest.TestCase):
    def test_true_via_legacy_path(self):
        self.assertTrue(_query_flag(_ws(path="/?skip_tts=true"), SKIP_TTS_QUERY_PARAM))

    def test_true_via_request_path(self):
        self.assertTrue(
            _query_flag(_ws(path=None, request_path="/?skip_tts=1"), SKIP_TTS_QUERY_PARAM)
        )

    def test_accepts_yes_and_mixed_case(self):
        self.assertTrue(_query_flag(_ws(path="/?skip_tts=YES"), SKIP_TTS_QUERY_PARAM))

    def test_capture_audio_flag(self):
        self.assertTrue(
            _query_flag(_ws(path="/?capture_bot_audio=true"), CAPTURE_AUDIO_QUERY_PARAM)
        )
        self.assertFalse(_query_flag(_ws(path="/?skip_tts=true"), CAPTURE_AUDIO_QUERY_PARAM))

    def test_capture_images_flag(self):
        self.assertTrue(
            _query_flag(_ws(path="/?capture_bot_images=true"), CAPTURE_IMAGES_QUERY_PARAM)
        )
        self.assertFalse(
            _query_flag(_ws(path="/?capture_bot_audio=true"), CAPTURE_IMAGES_QUERY_PARAM)
        )

    def test_false_when_absent(self):
        self.assertFalse(_query_flag(_ws(path="/"), SKIP_TTS_QUERY_PARAM))

    def test_false_when_falsey_value(self):
        self.assertFalse(_query_flag(_ws(path="/?skip_tts=false"), SKIP_TTS_QUERY_PARAM))

    def test_false_when_no_path_at_all(self):
        self.assertFalse(_query_flag(_ws(), SKIP_TTS_QUERY_PARAM))


class TestConnectionOutputSettings(unittest.IsolatedAsyncioTestCase):
    async def test_audio_connection_reenables_tts_without_query_flag(self):
        await self._assert_tts_resets_between_connections("/")

    async def test_audio_connection_reenables_tts_with_false_query_flag(self):
        await self._assert_tts_resets_between_connections("/?skip_tts=false")

    async def _assert_tts_resets_between_connections(self, audio_path: str):
        with self.assertWarns(DeprecationWarning):
            transport = EvalTransport(params=EvalTransportParams())
        self.addAsyncCleanup(transport.cleanup)
        input_transport = transport.input()
        push_frame = AsyncMock()
        input_transport.push_frame = push_frame
        transport.output().set_client_connection = AsyncMock()
        greeting_settings = asyncio.Queue()

        @transport.event_handler("on_client_connected")
        async def on_connected(transport, websocket):
            settings = [
                call.args[0].skip_tts
                for call in push_frame.await_args_list
                if isinstance(call.args[0], LLMConfigureOutputFrame)
            ]
            greeting_settings.put_nowait(settings[-1])

        observed = []
        for path in ("/?skip_tts=true", audio_path, "/?skip_tts=true"):
            await transport._on_client_connected(_ws(path=path))
            observed.append(await asyncio.wait_for(greeting_settings.get(), timeout=1))

        # Each greeting must see its own session's output setting.
        self.assertEqual(observed, [True, False, True])


class TestEvalFastAPIWebsocketTransport(unittest.IsolatedAsyncioTestCase):
    def _transport(self, flags: EvalConnectionFlags) -> EvalFastAPIWebsocketTransport:
        params = EvalTransportParams(serializer=EvalSerializer())
        return EvalFastAPIWebsocketTransport(MagicMock(), params, flags)

    def test_uses_the_eval_input_and_output(self):
        transport = self._transport(EvalConnectionFlags())
        self.assertIsInstance(transport.input(), EvalFastAPIWebsocketInputTransport)
        self.assertIsInstance(transport.output(), EvalFastAPIWebsocketOutputTransport)

    def test_capture_flags_reach_the_serializer(self):
        transport = self._transport(
            EvalConnectionFlags(capture_bot_audio=True, capture_bot_images=True)
        )
        serializer = transport._params.serializer
        self.assertTrue(serializer._capture_audio)
        self.assertTrue(serializer._capture_images)

    async def test_skip_tts_is_configured_before_on_client_connected(self):
        self.assertEqual(await self._greeting_skip_tts(True), [True])
        self.assertEqual(await self._greeting_skip_tts(False), [False])

    async def _greeting_skip_tts(self, skip_tts: bool) -> list[bool]:
        """The skip_tts settings pushed by the time ``on_client_connected`` runs."""
        transport = self._transport(EvalConnectionFlags(skip_tts=skip_tts))
        push_frame = AsyncMock()
        transport.input().push_frame = push_frame
        greeting_settings = asyncio.Queue()

        @transport.event_handler("on_client_connected")
        async def on_connected(transport, websocket):
            greeting_settings.put_nowait(
                [
                    call.args[0].skip_tts
                    for call in push_frame.await_args_list
                    if isinstance(call.args[0], LLMConfigureOutputFrame)
                ]
            )

        await transport._on_client_connected(MagicMock())
        return await asyncio.wait_for(greeting_settings.get(), timeout=1)


if __name__ == "__main__":
    unittest.main()
