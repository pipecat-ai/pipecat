#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for MistralSTTService language hint forwarding."""

import asyncio
import types

import pytest
from loguru import logger

stt_mod = pytest.importorskip("pipecat.services.mistral.stt", exc_type=ImportError)

from pipecat.transcriptions.language import Language

LANG_MODEL = "voxtral-mini-transcribe-realtime-3"


@pytest.fixture
def warnings():
    messages = []
    sink_id = logger.add(messages.append, level="WARNING")
    yield messages
    logger.remove(sink_id)


class _FakeConnection:
    is_closed = False

    async def events(self):
        await asyncio.Event().wait()
        yield


def _make_service(**settings):
    service = stt_mod.MistralSTTService(
        api_key="test",
        settings=stt_mod.MistralSTTSettings(**settings),
    )
    service._receive_task = None
    service.create_task = lambda coro, name=None: coro.close()
    return service


@pytest.mark.asyncio
async def test_language_forwarded_when_sdk_and_model_support_it():
    captured = {}

    async def connect(model, audio_format=None, target_streaming_delay_ms=None, language=None):
        captured.update({"model": model, "language": language})
        return _FakeConnection()

    service = _make_service(model=LANG_MODEL, language=Language.FR_FR)
    realtime = types.SimpleNamespace(connect=connect)
    service._client = types.SimpleNamespace(audio=types.SimpleNamespace(realtime=realtime))

    await service._connect()

    # BCP-47 hint "fr-FR" is normalized to the base ISO-639-1 code.
    assert captured["language"] == "fr"
    assert captured["model"] == LANG_MODEL


@pytest.mark.asyncio
async def test_language_not_sent_to_unsupported_model(warnings):
    captured = {}

    async def connect(model, audio_format=None, target_streaming_delay_ms=None, **kwargs):
        captured.update(kwargs)
        return _FakeConnection()

    # The default 2602 model does not accept a session language hint.
    service = _make_service(language=Language.FR)
    realtime = types.SimpleNamespace(connect=connect)
    service._client = types.SimpleNamespace(audio=types.SimpleNamespace(realtime=realtime))

    await service._connect()

    assert "language" not in captured
    assert any("does not support a language hint" in str(m) for m in warnings)


@pytest.mark.asyncio
async def test_language_skipped_when_sdk_lacks_support(warnings):
    captured = {}

    async def connect(model, audio_format=None, target_streaming_delay_ms=None):
        captured["model"] = model
        return _FakeConnection()

    service = _make_service(model=LANG_MODEL, language=Language.FR_FR)
    realtime = types.SimpleNamespace(connect=connect)
    service._client = types.SimpleNamespace(audio=types.SimpleNamespace(realtime=realtime))

    await service._connect()

    assert captured["model"] == LANG_MODEL
    assert any("mistralai>=3.7.0" in str(m) for m in warnings)


@pytest.mark.asyncio
async def test_no_language_no_warning(warnings):
    captured = {}

    async def connect(model, **kwargs):
        captured.update(kwargs)
        return _FakeConnection()

    service = _make_service()
    realtime = types.SimpleNamespace(connect=connect)
    service._client = types.SimpleNamespace(audio=types.SimpleNamespace(realtime=realtime))

    await service._connect()

    assert "language" not in captured
    assert not any("mistralai" in str(m) or "language" in str(m) for m in warnings)
