#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import io
import json
import unittest
import wave
from unittest.mock import AsyncMock, patch

from websockets.protocol import State

from pipecat.frames.frames import ErrorFrame, TTSAudioRawFrame, TTSStoppedFrame
from pipecat.services.rumik import tts as rumik_tts
from pipecat.services.rumik.tts import RumikHttpTTSService, RumikTTSService, RumikTTSSettings
from pipecat.services.settings import NOT_GIVEN


def _make_wav(pcm: bytes, *, sample_rate: int = 24000, channels: int = 1) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buffer.getvalue()


class _FakeHttpResponse:
    def __init__(self, *, status=200, body=b"", text=""):
        self.status = status
        self._body = body
        self._text = text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def read(self):
        return self._body

    async def json(self):
        return json.loads(self._body.decode("utf-8"))

    async def text(self):
        return self._text


class _FakeHttpClientSession:
    def __init__(self, response: _FakeHttpResponse):
        self.response = response
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


class _FakeMintClientSession:
    instances = []

    def __init__(self, *_, **__):
        self.calls = []
        _FakeMintClientSession.instances.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        body = {
            "ws_url": "ws://example.test/ws/tts",
            "token": "sess-token",
            "request_id": "request-1",
        }
        return _FakeHttpResponse(body=json.dumps(body).encode("utf-8"))


class _FakeWebSocket:
    def __init__(self, messages=None):
        self.state = State.OPEN
        self.sent = []
        self.closed = False
        self.messages = list(messages or [])

    async def send(self, message):
        self.sent.append(json.loads(message))

    async def close(self):
        self.closed = True
        self.state = State.CLOSED

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.messages:
            raise StopAsyncIteration
        message = self.messages.pop(0)
        if isinstance(message, Exception):
            raise message
        return message


class RumikTTSSettingsTests(unittest.TestCase):
    def test_package_exports_tts_services(self):
        import pipecat.services.rumik as rumik

        self.assertIs(rumik.RumikHttpTTSService, RumikHttpTTSService)
        self.assertIs(rumik.RumikTTSService, RumikTTSService)
        self.assertIs(rumik.RumikTTSSettings, RumikTTSSettings)

    def test_settings_are_sparse_by_default(self):
        settings = RumikTTSSettings()

        self.assertIs(settings.model, NOT_GIVEN)
        self.assertIs(settings.voice, NOT_GIVEN)
        self.assertIs(settings.language, NOT_GIVEN)
        self.assertIs(settings.description, NOT_GIVEN)
        self.assertIs(settings.f0_up_key, NOT_GIVEN)
        self.assertIs(settings.temperature, NOT_GIVEN)
        self.assertIs(settings.top_p, NOT_GIVEN)
        self.assertIs(settings.top_k, NOT_GIVEN)
        self.assertIs(settings.repetition_penalty, NOT_GIVEN)
        self.assertIs(settings.max_new_tokens, NOT_GIVEN)

    def test_sparse_delta_does_not_reset_existing_store_fields(self):
        settings = RumikTTSSettings(
            model="custom",
            voice="speaker_1",
            language=None,
            description="warm",
            f0_up_key=0,
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.2,
            max_new_tokens=2048,
        )

        changed = settings.apply_update(RumikTTSSettings(voice="speaker_2"))

        self.assertEqual(changed, {"voice": "speaker_1"})
        self.assertEqual(settings.model, "custom")
        self.assertEqual(settings.description, "warm")
        self.assertEqual(settings.temperature, 0.7)
        self.assertEqual(settings.voice, "speaker_2")


class RumikHttpTTSServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_http_request_uses_rumik_tts_endpoint_headers_and_payload(self):
        pcm = b"\x01\x00\x02\x00"
        session = _FakeHttpClientSession(_FakeHttpResponse(body=_make_wav(pcm)))
        service = RumikHttpTTSService(
            api_key="test-api-key",
            gateway_url="https://example.test/",
            aiohttp_session=session,
            settings=RumikHttpTTSService.Settings(
                model="mulberry",
                voice="speaker_2",
                description="warm",
                f0_up_key=-1,
                temperature=0.7,
                top_p=0.9,
                top_k=32,
                repetition_penalty=1.1,
                max_new_tokens=1024,
            ),
        )

        frames = [frame async for frame in service.run_tts(" hello   world ", "ctx-1")]

        self.assertEqual(len(frames), 1)
        self.assertIsInstance(frames[0], TTSAudioRawFrame)
        url, kwargs = session.calls[0]
        self.assertEqual(url, "https://example.test/v1/tts")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-api-key")
        self.assertEqual(
            kwargs["json"],
            {
                "text": "hello world",
                "model": "mulberry",
                "description": "warm",
                "speaker": "speaker_2",
                "f0_up_key": -1,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 32,
                "repetition_penalty": 1.1,
                "max_new_tokens": 1024,
            },
        )

    async def test_wav_response_is_converted_to_raw_pcm_frame(self):
        pcm = b"\x01\x00\x02\x00\x03\x00\x04\x00"
        session = _FakeHttpClientSession(_FakeHttpResponse(body=_make_wav(pcm)))
        service = RumikHttpTTSService(
            api_key="key",
            gateway_url="https://example.test",
            aiohttp_session=session,
        )

        frames = [frame async for frame in service.run_tts("hello", "ctx-1")]

        self.assertEqual(len(frames), 1)
        frame = frames[0]
        self.assertIsInstance(frame, TTSAudioRawFrame)
        self.assertEqual(frame.audio, pcm)
        self.assertEqual(frame.sample_rate, 24000)
        self.assertEqual(frame.num_channels, 1)
        self.assertEqual(frame.context_id, "ctx-1")

    async def test_invalid_wav_contract_yields_error_frame(self):
        session = _FakeHttpClientSession(
            _FakeHttpResponse(body=_make_wav(b"\x01\x00", sample_rate=16000))
        )
        service = RumikHttpTTSService(
            api_key="key",
            gateway_url="https://example.test",
            aiohttp_session=session,
        )

        frames = [frame async for frame in service.run_tts("hello", "ctx-1")]

        self.assertEqual(len(frames), 1)
        self.assertIsInstance(frames[0], ErrorFrame)
        self.assertIn("Expected 24000 Hz WAV", frames[0].error)

    async def test_http_error_statuses_yield_error_frame(self):
        for status in (401, 402, 429, 502):
            with self.subTest(status=status):
                session = _FakeHttpClientSession(
                    _FakeHttpResponse(status=status, text="provider error")
                )
                service = RumikHttpTTSService(
                    api_key="key",
                    gateway_url="https://example.test",
                    aiohttp_session=session,
                )

                frames = [frame async for frame in service.run_tts("hello", "ctx-1")]

                self.assertEqual(len(frames), 1)
                self.assertIsInstance(frames[0], ErrorFrame)
                self.assertIn(f"HTTP {status}", frames[0].error)


class RumikTTSServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_constructor_uses_rumik_defaults_and_full_response_aggregation(self):
        service = RumikTTSService(api_key="key", gateway_url="https://example.test/")

        self.assertEqual(service._gateway_url, "https://example.test")
        self.assertEqual(service._init_sample_rate, 24000)
        self.assertEqual(service._settings.model, "muga")
        self.assertIsNone(service._settings.voice)
        self.assertIsNone(service._settings.description)
        self.assertIsNone(service._settings.f0_up_key)
        self.assertEqual(service._settings.temperature, 0.6)
        self.assertEqual(service._settings.top_p, 0.95)
        self.assertEqual(service._settings.top_k, 50)
        self.assertEqual(service._settings.repetition_penalty, 1.2)

        aggregations = [
            aggregation async for aggregation in service._text_aggregator.aggregate("Hello. ")
        ]
        self.assertEqual(aggregations, [])
        flushed = await service._text_aggregator.flush()
        self.assertEqual(flushed.text, "Hello.")

    async def test_constructor_rejects_unsupported_sample_rate(self):
        with self.assertRaisesRegex(ValueError, "24000"):
            RumikTTSService(
                api_key="key",
                gateway_url="https://example.test",
                sample_rate=16000,
            )

    async def test_mint_websocket_session_uses_rumik_api_contract(self):
        _FakeMintClientSession.instances = []
        service = RumikTTSService(
            api_key="test-api-key",
            gateway_url="https://example.test/",
            settings=RumikTTSService.Settings(model="mulberry"),
        )

        with patch.object(rumik_tts.aiohttp, "ClientSession", _FakeMintClientSession):
            response = await service._mint_websocket_session()

        self.assertEqual(response["request_id"], "request-1")
        session = _FakeMintClientSession.instances[0]
        url, kwargs = session.calls[0]
        self.assertEqual(url, "https://example.test/v1/tts/ws-connect")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-api-key")
        self.assertEqual(kwargs["json"], {"text": "init", "model": "mulberry"})

    async def test_connect_websocket_uses_minted_token_url(self):
        service = RumikTTSService(api_key="key", gateway_url="https://example.test")
        websocket = _FakeWebSocket()

        with (
            patch.object(
                service,
                "_mint_websocket_session",
                AsyncMock(
                    return_value={
                        "ws_url": "ws://example.test/ws/tts",
                        "token": "sess-token",
                        "request_id": "request-1",
                    }
                ),
            ),
            patch.object(rumik_tts, "websocket_connect", AsyncMock(return_value=websocket)) as ws,
        ):
            await service._connect_websocket()

        self.assertIs(service._websocket, websocket)
        self.assertEqual(ws.call_args.args[0], "ws://example.test/ws/tts?token=sess-token")

    async def test_run_tts_sends_text_payload_and_holds_request_lock(self):
        service = RumikTTSService(
            api_key="key",
            gateway_url="https://example.test",
            settings=RumikTTSService.Settings(
                model="mulberry",
                voice="speaker_3",
                description="calm narrator",
                f0_up_key=2,
            ),
        )
        service._websocket = _FakeWebSocket()

        frames = [frame async for frame in service.run_tts(" hello   world ", "ctx-1")]

        self.assertEqual(frames, [None])
        self.assertEqual(
            service._websocket.sent,
            [
                {
                    "text": "hello world",
                    "description": "calm narrator",
                    "speaker": "speaker_3",
                    "f0_up_key": 2,
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "top_k": 50,
                    "repetition_penalty": 1.2,
                    "max_new_tokens": 2048,
                }
            ],
        )
        self.assertEqual(service._active_context_id, "ctx-1")
        self.assertTrue(service._synthesis_lock.locked())

        service.append_to_audio_context = AsyncMock()
        service.remove_audio_context = AsyncMock()
        await service._finish_active_context()

        self.assertIsNone(service._active_context_id)
        self.assertFalse(service._synthesis_lock.locked())

    async def test_binary_audio_and_done_are_appended_to_active_context(self):
        service = RumikTTSService(api_key="key", gateway_url="https://example.test")
        service._sample_rate = 24000
        service._active_context_id = "ctx-1"
        service._websocket = _FakeWebSocket([b"\x01\x00", json.dumps({"type": "done"})])
        service.append_to_audio_context = AsyncMock()
        service.remove_audio_context = AsyncMock()

        await service._receive_messages()

        frames = [call.args[1] for call in service.append_to_audio_context.mock_calls]
        self.assertIsInstance(frames[0], TTSAudioRawFrame)
        self.assertEqual(frames[0].audio, b"\x01\x00")
        self.assertEqual(frames[0].sample_rate, 24000)
        self.assertEqual(frames[0].num_channels, 1)
        self.assertIsInstance(frames[1], TTSStoppedFrame)
        service.remove_audio_context.assert_awaited_once_with("ctx-1")

    async def test_error_control_frame_appends_error_then_stop(self):
        service = RumikTTSService(api_key="key", gateway_url="https://example.test")
        service._active_context_id = "ctx-1"
        service._websocket = _FakeWebSocket([json.dumps({"type": "error", "message": "bad"})])
        service.append_to_audio_context = AsyncMock()
        service.remove_audio_context = AsyncMock()

        await service._receive_messages()

        appended_frames = [call.args[1] for call in service.append_to_audio_context.mock_calls]
        self.assertIsInstance(appended_frames[0], ErrorFrame)
        self.assertIsInstance(appended_frames[1], TTSStoppedFrame)
        service.remove_audio_context.assert_awaited_once_with("ctx-1")

    async def test_disconnect_sends_close_frame(self):
        service = RumikTTSService(api_key="key", gateway_url="https://example.test")
        websocket = _FakeWebSocket()
        service._websocket = websocket

        await service._disconnect_websocket()

        self.assertEqual(websocket.sent, [{"type": "close"}])
        self.assertTrue(websocket.closed)
        self.assertIsNone(service._websocket)

    async def test_interruption_restarts_socket_and_clears_active_context_lock(self):
        service = RumikTTSService(api_key="key", gateway_url="https://example.test")
        await service._synthesis_lock.acquire()
        service._active_context_id = "ctx-1"
        service._bot_speaking = True
        service._disconnect = AsyncMock()
        service._connect = AsyncMock()

        await service.on_audio_context_interrupted("ctx-1")

        service._disconnect.assert_awaited_once_with(clear_active_context=False)
        service._connect.assert_awaited_once()
        self.assertIsNone(service._active_context_id)
        self.assertFalse(service._synthesis_lock.locked())
        self.assertFalse(service._bot_speaking)


if __name__ == "__main__":
    unittest.main()
