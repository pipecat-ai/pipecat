#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
import json

import pytest
from websockets.protocol import State

from pipecat.frames.frames import TTSAudioRawFrame, TTSStoppedFrame
from pipecat.services.palabra.tts import (
    REGION_URLS,
    PalabraTTSService,
    language_to_palabra_tts_language,
)
from pipecat.transcriptions.language import Language


class _FakeWebsocket:
    def __init__(self, messages=(), *, state=State.OPEN):
        self._messages = list(messages)
        self.state = state
        self.sent = []

    async def send(self, payload):
        self.sent.append(json.loads(payload))

    async def close(self):
        self.state = State.CLOSED

    def __aiter__(self):
        return self._iter_messages()

    async def _iter_messages(self):
        for message in self._messages:
            yield message


def _service(monkeypatch=None, messages=(), contexts=(), **kwargs):
    """Build a service with a fake socket; ``contexts`` are the open audio contexts."""
    service = PalabraTTSService(api_key="test-key", **kwargs)
    service._sample_rate = 24000
    service._websocket = _FakeWebsocket(messages)

    appended = []
    removed = []
    errors = []
    open_contexts = set(contexts)

    async def fake_append(context_id, frame):
        appended.append((context_id, frame))

    async def fake_remove(context_id):
        removed.append(context_id)
        open_contexts.discard(context_id)

    async def fake_push_error(error_msg, **kwargs):
        errors.append(error_msg)

    async def fake_noop(*args, **kwargs):
        pass

    if monkeypatch:
        monkeypatch.setattr(service, "append_to_audio_context", fake_append)
        monkeypatch.setattr(service, "remove_audio_context", fake_remove)
        monkeypatch.setattr(service, "audio_context_available", lambda c: c in open_contexts)
        monkeypatch.setattr(service, "get_audio_contexts", lambda: list(open_contexts))
        monkeypatch.setattr(service, "push_error", fake_push_error)
        monkeypatch.setattr(service, "stop_ttfb_metrics", fake_noop)
        monkeypatch.setattr(service, "stop_all_metrics", fake_noop)
        monkeypatch.setattr(service, "start_tts_usage_metrics", fake_noop)
    return service, appended, removed, errors


def _audio_chunk(generation_id: str, audio: bytes | None, last_chunk: bool = False) -> str:
    data = {"generation_id": generation_id, "last_chunk": last_chunk}
    if audio is not None:
        data["audio"] = base64.b64encode(audio).decode()
    return json.dumps({"message_type": "audio_chunk", "data": data})


def test_language_code_is_the_lowercased_language_value():
    assert language_to_palabra_tts_language(Language.EN) == "en"
    assert language_to_palabra_tts_language(Language.EN_US) == "en-us"
    assert language_to_palabra_tts_language(Language.ES_MX) == "es-mx"
    assert language_to_palabra_tts_language(Language.PT_BR) == "pt-br"
    assert language_to_palabra_tts_language(Language.FR_FR) == "fr-fr"
    assert language_to_palabra_tts_language(Language.ES_419) == "es-419"


def test_region_selects_endpoint():
    assert PalabraTTSService(api_key="k")._url == REGION_URLS["eu"]
    assert PalabraTTSService(api_key="k", region="us")._url == REGION_URLS["us"]
    assert PalabraTTSService(api_key="k", url="wss://custom")._url == "wss://custom"
    with pytest.raises(ValueError):
        PalabraTTSService(api_key="k", region="mars")


def test_init_message_reflects_settings():
    service, *_ = _service(
        settings=PalabraTTSService.Settings(
            language=Language.ES_ES, voice="default_high", speed=1.2, deaccent_strength=0.5
        )
    )

    assert service._build_init_msg() == {
        "type": "init",
        "language": "es-es",
        "model": "auto",
        "voice_options": {"voice_id": "default_high", "speed": 1.2, "deaccent_strength": 0.5},
        "output": {"format": "pcm", "sample_rate": 24000},
    }


@pytest.mark.asyncio
async def test_run_tts_sends_one_generation_per_sentence(monkeypatch):
    service, *_ = _service(monkeypatch)

    async for _ in service.run_tts("Hello there.", "ctx-1"):
        pass

    sent = service._websocket.sent
    assert len(sent) == 1
    assert sent[0]["type"] == "text"
    assert sent[0]["text"] == "Hello there."
    assert sent[0]["is_eos"] is True
    assert sent[0]["voice_options"] == {"voice_id": "default_low"}
    assert service._generations == {sent[0]["generation_id"]: "ctx-1"}


@pytest.mark.asyncio
async def test_audio_is_routed_to_the_generation_context(monkeypatch):
    service, appended, removed, _ = _service(monkeypatch, contexts=["ctx-1"])
    generation_id = service._next_generation_id("ctx-1")
    service._websocket = _FakeWebsocket(
        [
            _audio_chunk(generation_id, b"\x01\x02"),
            _audio_chunk("stale", b"\x09\x09"),
            _audio_chunk(generation_id, None, last_chunk=True),
        ]
    )

    await service._receive_messages()

    assert len(appended) == 1
    context_id, frame = appended[0]
    assert context_id == "ctx-1"
    assert isinstance(frame, TTSAudioRawFrame)
    assert frame.audio == b"\x01\x02"
    assert frame.sample_rate == 24000
    assert frame.context_id == "ctx-1"
    # The context stays open: its text was not flushed yet.
    assert removed == []
    assert service._generations == {}


@pytest.mark.asyncio
async def test_context_closes_when_flushed_and_last_generation_ends(monkeypatch):
    service, appended, removed, _ = _service(monkeypatch, contexts=["ctx-1"])
    gen_a = service._next_generation_id("ctx-1")
    gen_b = service._next_generation_id("ctx-1")
    await service.flush_audio("ctx-1")
    assert removed == []

    service._websocket = _FakeWebsocket(
        [
            _audio_chunk(gen_a, None, last_chunk=True),
            _audio_chunk(gen_b, b"\x01", last_chunk=False),
            _audio_chunk(gen_b, None, last_chunk=True),
        ]
    )
    await service._receive_messages()

    assert removed == ["ctx-1"]
    assert isinstance(appended[-1][1], TTSStoppedFrame)
    assert appended[-1][1].context_id == "ctx-1"
    assert "ctx-1" not in service._flushed_contexts


@pytest.mark.asyncio
async def test_flush_with_nothing_pending_closes_immediately(monkeypatch):
    service, appended, removed, _ = _service(monkeypatch, contexts=["ctx-1"])

    await service.flush_audio("ctx-1")

    assert removed == ["ctx-1"]
    assert isinstance(appended[0][1], TTSStoppedFrame)


@pytest.mark.asyncio
async def test_interruption_cancels_and_forgets_generations(monkeypatch):
    service, *_ = _service(monkeypatch, contexts=["ctx-1"])
    service._next_generation_id("ctx-1")
    service._flushed_contexts.add("ctx-1")

    await service.on_audio_context_interrupted("ctx-1")

    assert service._websocket.sent == [{"type": "cancel"}]
    assert service._generations == {}
    assert service._flushed_contexts == set()


@pytest.mark.asyncio
async def test_session_kept_error_is_reported_without_closing_contexts(monkeypatch):
    service, appended, removed, errors = _service(
        monkeypatch,
        messages=[
            json.dumps(
                {"message_type": "error", "data": {"code": "RATE_LIMIT_EXCEEDED", "desc": "slow"}}
            )
        ],
        contexts=["ctx-1"],
    )
    service._next_generation_id("ctx-1")

    await service._receive_messages()

    assert errors == ["Palabra TTS error RATE_LIMIT_EXCEEDED: slow"]
    assert removed == []
    assert len(service._generations) == 1


@pytest.mark.asyncio
async def test_fatal_error_closes_open_contexts(monkeypatch):
    service, appended, removed, errors = _service(
        monkeypatch,
        messages=[
            json.dumps({"message_type": "error", "data": {"code": "SERVER_ERROR", "desc": "boom"}})
        ],
        contexts=["ctx-1"],
    )
    service._next_generation_id("ctx-1")

    await service._receive_messages()

    assert errors == ["Palabra TTS error SERVER_ERROR: boom"]
    assert removed == ["ctx-1"]
    assert isinstance(appended[0][1], TTSStoppedFrame)
    assert service._generations == {}


@pytest.mark.asyncio
async def test_language_change_reconnects_but_voice_change_does_not(monkeypatch):
    service, *_ = _service(monkeypatch)
    service._is_usable = True
    reconnects = []

    async def fake_disconnect():
        reconnects.append("disconnect")

    async def fake_connect():
        reconnects.append("connect")

    monkeypatch.setattr(service, "_disconnect", fake_disconnect)
    monkeypatch.setattr(service, "_connect", fake_connect)

    await service._update_settings(PalabraTTSService.Settings(voice="default_high", speed=0.9))
    assert reconnects == []
    assert service._voice_options() == {"voice_id": "default_high", "speed": 0.9}

    await service._update_settings(PalabraTTSService.Settings(language=Language.RU))
    assert reconnects == ["disconnect", "connect"]
    assert service._settings.language == "ru"
