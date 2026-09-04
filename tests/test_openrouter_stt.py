#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for OpenRouterSTTService."""

import base64
import json
import unittest

import pytest
from aiohttp import web

from pipecat.frames.frames import ErrorFrame, TranscriptionFrame
from pipecat.services.openrouter.stt import (
    OPENROUTER_DEFAULT_STT_MODEL,
    OPENROUTER_STT_BASE_URL,
    OpenRouterSTTService,
    OpenRouterSTTSettings,
)
from pipecat.services.settings import is_given
from pipecat.transcriptions.language import Language


# ---------------------------------------------------------------------------
# Settings contract
# ---------------------------------------------------------------------------


def test_settings_delta_defaults_are_not_given():
    """Empty OpenRouterSTTSettings must have all fields set to NOT_GIVEN."""
    from dataclasses import fields

    s = OpenRouterSTTSettings()
    for f in fields(s):
        if f.name == "extra":
            continue
        val = getattr(s, f.name)
        assert not is_given(val), (
            f"OpenRouterSTTSettings.{f.name} defaults to {val!r}, expected NOT_GIVEN"
        )


def test_service_settings_complete_after_init():
    """After construction, _settings must have no NOT_GIVEN values."""
    from dataclasses import fields

    svc = OpenRouterSTTService(api_key="test-key")
    for f in fields(svc._settings):
        if f.name == "extra":
            continue
        val = getattr(svc._settings, f.name)
        assert is_given(val), (
            f"OpenRouterSTTService._settings.{f.name} is NOT_GIVEN after construction"
        )


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_default_model():
    svc = OpenRouterSTTService(api_key="test-key")
    assert svc._settings.model == OPENROUTER_DEFAULT_STT_MODEL


def test_default_language():
    svc = OpenRouterSTTService(api_key="test-key")
    assert svc._settings.language == Language.EN


def test_settings_override_model_and_language():
    svc = OpenRouterSTTService(
        api_key="test-key",
        settings=OpenRouterSTTService.Settings(
            model="openai/whisper-large-v3",
            language=Language.FR,
        ),
    )
    assert svc._settings.model == "openai/whisper-large-v3"
    assert svc._settings.language == Language.FR


# ---------------------------------------------------------------------------
# language_to_service_language
# ---------------------------------------------------------------------------


def test_language_to_service_language_strips_region():
    svc = OpenRouterSTTService(api_key="test-key")
    assert svc.language_to_service_language(Language.EN) == "en"
    assert svc.language_to_service_language(Language.FR) == "fr"
    assert svc.language_to_service_language(Language.ZH) == "zh"


# ---------------------------------------------------------------------------
# run_stt — live aiohttp test server
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_stt_success(aiohttp_client):
    """A 200 response with text should yield a TranscriptionFrame."""
    received_body = {}

    async def handler(request):
        received_body.update(await request.json())
        return web.Response(
            content_type="application/json",
            body=json.dumps({"text": "Hello from OpenRouter."}),
        )

    app = web.Application()
    app.router.add_post("/audio/transcriptions", handler)
    client = await aiohttp_client(app)
    base_url = str(client.make_url("/")).rstrip("/")

    svc = OpenRouterSTTService(api_key="test-key", base_url=base_url)
    svc._user_id = "user-1"

    frames = []
    async for frame in svc.run_stt(b"\x00" * 256):
        frames.append(frame)

    assert len(frames) == 1
    assert isinstance(frames[0], TranscriptionFrame)
    assert frames[0].text == "Hello from OpenRouter."


@pytest.mark.asyncio
async def test_run_stt_request_body(aiohttp_client):
    """run_stt should send model, base64 audio, format, and language."""
    received_body = {}

    async def handler(request):
        received_body.update(await request.json())
        return web.Response(
            content_type="application/json",
            body=json.dumps({"text": "test"}),
        )

    app = web.Application()
    app.router.add_post("/audio/transcriptions", handler)
    client = await aiohttp_client(app)
    base_url = str(client.make_url("/")).rstrip("/")

    audio = b"\x01\x02\x03\x04" * 64
    svc = OpenRouterSTTService(api_key="test-key", base_url=base_url)
    svc._user_id = "user-1"

    async for _ in svc.run_stt(audio):
        pass

    assert received_body["model"] == OPENROUTER_DEFAULT_STT_MODEL
    assert received_body["input_audio"]["data"] == base64.b64encode(audio).decode("utf-8")
    assert received_body["input_audio"]["format"] == "wav"
    assert received_body["language"] == "en"


@pytest.mark.asyncio
async def test_run_stt_optional_params_forwarded(aiohttp_client):
    """temperature and prompt should be included in the request when set."""
    received_body = {}

    async def handler(request):
        received_body.update(await request.json())
        return web.Response(
            content_type="application/json",
            body=json.dumps({"text": "test"}),
        )

    app = web.Application()
    app.router.add_post("/audio/transcriptions", handler)
    client = await aiohttp_client(app)
    base_url = str(client.make_url("/")).rstrip("/")

    svc = OpenRouterSTTService(
        api_key="test-key",
        base_url=base_url,
        settings=OpenRouterSTTService.Settings(temperature=0.2, prompt="Pipecat"),
    )
    svc._user_id = "user-1"

    async for _ in svc.run_stt(b"\x00" * 64):
        pass

    assert received_body["temperature"] == 0.2
    assert received_body["prompt"] == "Pipecat"


@pytest.mark.asyncio
async def test_run_stt_error_status_yields_error_frame(aiohttp_client):
    """A non-200 response should yield an ErrorFrame."""

    async def handler(request):
        return web.Response(status=400, text="bad request")

    app = web.Application()
    app.router.add_post("/audio/transcriptions", handler)
    client = await aiohttp_client(app)
    base_url = str(client.make_url("/")).rstrip("/")

    svc = OpenRouterSTTService(api_key="test-key", base_url=base_url)
    svc._user_id = "user-1"

    frames = []
    async for frame in svc.run_stt(b"\x00" * 64):
        frames.append(frame)

    assert len(frames) == 1
    assert isinstance(frames[0], ErrorFrame)


@pytest.mark.asyncio
async def test_run_stt_empty_transcript_yields_no_frame(aiohttp_client):
    """An empty transcript should produce no output frames."""

    async def handler(request):
        return web.Response(
            content_type="application/json",
            body=json.dumps({"text": "   "}),
        )

    app = web.Application()
    app.router.add_post("/audio/transcriptions", handler)
    client = await aiohttp_client(app)
    base_url = str(client.make_url("/")).rstrip("/")

    svc = OpenRouterSTTService(api_key="test-key", base_url=base_url)
    svc._user_id = "user-1"

    frames = []
    async for frame in svc.run_stt(b"\x00" * 64):
        frames.append(frame)

    assert frames == []


if __name__ == "__main__":
    unittest.main()
