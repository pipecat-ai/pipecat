#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import json
import unittest

from websockets.protocol import State

from pipecat.frames.frames import (
    AggregatedTextFrame,
    AggregatedTextProgressFrame,
    AggregationType,
    TTSTextFrame,
)
from pipecat.services.inworld.tts import InworldHttpTTSService, InworldTTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.context.aggregated_frame_sequencer import AggregatedFrameSequencer


class _FakeWebSocket:
    """Minimal stand-in for the Inworld websocket that records sends."""

    def __init__(self):
        self.state = State.OPEN
        self.sent: list[dict] = []

    async def send(self, data: str):
        self.sent.append(json.loads(data))


class _FakeHttpResponse:
    """Minimal aiohttp response stand-in; the 400 makes run_tts bail after posting."""

    status = 400

    async def text(self):
        return "rejected"

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeHttpSession:
    """Records the JSON payload of each POST."""

    def __init__(self):
        self.payloads: list[dict] = []

    def post(self, url, json=None, headers=None):
        self.payloads.append(json)
        return _FakeHttpResponse()


class TestInworldUpdateSettingsRotatesContextGracefully(unittest.IsolatedAsyncioTestCase):
    """A runtime settings change (language, voice, model, ...) re-mints the turn
    context on the same live websocket instead of reconnecting. Reconnecting used
    to cancel the receive task before the old context's close_context message
    could be acknowledged, dropping trailing audio and the transcript prefix.
    """

    async def _service_with_pending_prefix(self, old_ctx: str):
        service = InworldTTSService.__new__(InworldTTSService)
        service._name = "InworldTTSService#0"
        service._settings = InworldTTSService.Settings(
            model="inworld-tts-2",
            voice="Ashley",
            language=Language.EN,
            speaking_rate=None,
            temperature=None,
            delivery_mode=None,
        )
        # Applying a settings delta reports the service usable again.
        service._is_usable = True

        # Real streaming sequencer with a mid-sentence prefix pending on the turn ctx.
        seq = AggregatedFrameSequencer(name=service._name, streaming=True)
        service._aggregated_frame_sequencer = seq
        service._turn_context_id = old_ctx
        for token in ("Hi", " there"):
            frame = AggregatedTextFrame(token, AggregationType.SENTENCE, raw_text=token)
            await seq.register_spoken(frame, old_ctx, token, append_to_context=True)
        assert seq._slots == []  # nothing promoted — sentence has no boundary yet

        pushed: list = []

        async def fake_push(frames, context_id):
            pushed.extend(frames)

        async def fake_flush(context_id=None):
            service._flushed = context_id

        reconnects: list[str] = []

        async def fake_connect():
            reconnects.append("connect")

        async def fake_disconnect():
            reconnects.append("disconnect")

        async def noop_report_error(*args, **kwargs):
            pass

        service._flushed = None
        service._push_sequencer_frames = fake_push
        service.flush_audio = fake_flush
        service.audio_context_available = lambda context_id: True
        service.create_context_id = lambda: "ctx-new"
        service._connect = fake_connect
        service._disconnect = fake_disconnect
        service._reconnects = reconnects
        service._report_error = noop_report_error

        service._websocket = _FakeWebSocket()
        service._sent_context_ids = {old_ctx}
        service._cumulative_time = 5.0
        service._generation_end_time = 3.0

        return service, seq, pushed

    async def test_language_change_does_not_reconnect(self):
        old_ctx = "ctx-old"
        service, _seq, _pushed = await self._service_with_pending_prefix(old_ctx)

        await service._update_settings(InworldTTSService.Settings(language=Language.ES))

        self.assertEqual(service._reconnects, [])

    async def test_language_change_flushes_and_closes_old_context(self):
        old_ctx = "ctx-old"
        service, _seq, _pushed = await self._service_with_pending_prefix(old_ctx)

        await service._update_settings(InworldTTSService.Settings(language=Language.ES))

        self.assertEqual(service._flushed, old_ctx)
        close_messages = [
            m
            for m in service._websocket.sent
            if m.get("contextId") == old_ctx and "close_context" in m
        ]
        self.assertEqual(len(close_messages), 1)
        self.assertNotIn(old_ctx, service._sent_context_ids)

    async def test_language_change_rotates_turn_context_id(self):
        old_ctx = "ctx-old"
        service, _seq, _pushed = await self._service_with_pending_prefix(old_ctx)

        await service._update_settings(InworldTTSService.Settings(language=Language.ES))

        self.assertEqual(service._turn_context_id, "ctx-new")

    async def test_language_change_finalizes_pending_prefix(self):
        old_ctx = "ctx-old"
        service, seq, pushed = await self._service_with_pending_prefix(old_ctx)

        await service._update_settings(InworldTTSService.Settings(language=Language.ES))

        # The old context's pending sentence was force-promoted into a real slot.
        self.assertEqual([s.frame.text for s in seq._slots], ["Hi there"])
        self.assertEqual(seq._slots[0].context_id, old_ctx)
        self.assertTrue(
            any(isinstance(f, AggregatedTextFrame) and f.text == "Hi there" for f in pushed)
        )

        # A word-timestamp for the flushed prefix (arriving on the OLD context during
        # playout) still finds the promoted slot and emits a progress frame.
        result = seq.process_word("Hi", pts=10, context_id=old_ctx)
        self.assertTrue(any(isinstance(f, TTSTextFrame) and f.text == "Hi" for f in result))
        progress = [f for f in result if isinstance(f, AggregatedTextProgressFrame)]
        self.assertEqual(len(progress), 1)
        self.assertEqual(progress[0].accumulated_text, "Hi")

    async def test_next_context_create_carries_new_language(self):
        old_ctx = "ctx-old"
        service, _seq, _pushed = await self._service_with_pending_prefix(old_ctx)
        service._audio_encoding = "PCM"
        service._audio_sample_rate = 24000
        service._apply_text_normalization = None
        service._auto_mode = True
        service._timestamp_transport_strategy = "ASYNC"
        service._buffer_settings = {"maxBufferDelayMs": None, "bufferCharThreshold": None}
        service._timestamp_type = "WORD"

        await service._update_settings(InworldTTSService.Settings(language=Language.ES))
        await service._send_context(service._turn_context_id)

        create_messages = [m for m in service._websocket.sent if "create" in m]
        self.assertEqual(len(create_messages), 1)
        self.assertEqual(create_messages[0]["contextId"], "ctx-new")
        self.assertEqual(create_messages[0]["create"]["language"], "es-ES")

    async def test_generation_timing_reset_on_rotate(self):
        old_ctx = "ctx-old"
        service, _seq, _pushed = await self._service_with_pending_prefix(old_ctx)

        await service._update_settings(InworldTTSService.Settings(language=Language.ES))

        self.assertEqual(service._cumulative_time, 0.0)
        self.assertEqual(service._generation_end_time, 0.0)

    async def test_empty_delta_is_a_noop(self):
        old_ctx = "ctx-old"
        service, seq, pushed = await self._service_with_pending_prefix(old_ctx)

        await service._update_settings(InworldTTSService.Settings())

        self.assertEqual(seq._slots, [])  # still pending, not promoted
        self.assertEqual(pushed, [])
        self.assertIsNone(service._flushed)
        self.assertEqual(service._turn_context_id, old_ctx)
        self.assertEqual(service._reconnects, [])
        self.assertEqual(service._websocket.sent, [])

    async def test_language_change_while_idle_does_not_flush(self):
        # Between turns there's no turn context to finalize or flush; the new
        # language just applies to whichever context opens next.
        old_ctx = "ctx-old"
        service, seq, pushed = await self._service_with_pending_prefix(old_ctx)
        service._turn_context_id = None

        await service._update_settings(InworldTTSService.Settings(language=Language.ES))

        self.assertIsNone(service._flushed)
        self.assertEqual(pushed, [])
        self.assertEqual(service._websocket.sent, [])
        self.assertEqual(service._reconnects, [])


class TestInworldHttpLanguageChange(unittest.IsolatedAsyncioTestCase):
    """InworldHttpTTSService already reads settings fresh per request, so a
    language change applies to the very next call with no reconnect concept.
    """

    async def test_language_change_applies_to_next_request(self):
        session = _FakeHttpSession()
        service = InworldHttpTTSService(
            api_key="test-key",
            aiohttp_session=session,
            settings=InworldHttpTTSService.Settings(voice="Ashley", language=Language.EN),
        )

        async for _ in service.run_tts("Hello", "ctx-1"):
            pass
        self.assertEqual(session.payloads[0]["language"], "en-US")

        await service._update_settings(InworldHttpTTSService.Settings(language=Language.ES))

        async for _ in service.run_tts("Hola", "ctx-2"):
            pass
        self.assertEqual(session.payloads[1]["language"], "es-ES")
