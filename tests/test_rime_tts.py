#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import pytest

from pipecat.services.rime.tts import (
    RimeHttpTTSService,
    RimeNonJsonTTSService,
    RimeTTSService,
)


def test_coda_sampling_params_are_included_in_websocket_params():
    service = RimeTTSService(
        api_key="test-api-key",
        settings=RimeTTSService.Settings(
            model="coda",
            voice="luna",
            repetition_penalty=1.1,
            temperature=0.5,
            top_p=0.9,
            timeScaleFactor=1.2,
        ),
    )

    params = service._build_ws_params()

    assert params["modelId"] == "coda"
    assert params["speaker"] == "luna"
    assert params["repetition_penalty"] == 1.1
    assert params["temperature"] == 0.5
    assert params["top_p"] == 0.9
    assert params["timeScaleFactor"] == 1.2


def test_non_json_service_defaults_to_coda_without_a_voice():
    with pytest.warns(DeprecationWarning, match="RimeNonJsonTTSService"):
        service = RimeNonJsonTTSService(api_key="test-api-key")

    assert service._settings.model == "coda"
    assert service._settings.voice is None
    assert service._url == "wss://users.rime.ai/ws"


class _ErrorResponse:
    status = 400

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return False


class _CapturingSession:
    def __init__(self):
        self.payload = None
        self.headers = None

    def post(self, url, *, json, headers):
        self.payload = json
        self.headers = headers
        return _ErrorResponse()


@pytest.mark.asyncio
async def test_coda_sampling_params_are_included_in_http_payload():
    session = _CapturingSession()
    service = RimeHttpTTSService(
        api_key="test-api-key",
        aiohttp_session=session,
        sample_rate=24000,
        settings=RimeHttpTTSService.Settings(
            model="coda",
            voice="luna",
            repetition_penalty=1.1,
            temperature=0.5,
            top_p=0.9,
            timeScaleFactor=1.2,
        ),
    )

    _ = [frame async for frame in service.run_tts("Hello", "context")]

    assert session.payload["modelId"] == "coda"
    assert session.payload["speaker"] == "luna"
    assert session.payload["repetition_penalty"] == 1.1
    assert session.payload["temperature"] == 0.5
    assert session.payload["top_p"] == 0.9
    assert session.payload["timeScaleFactor"] == 1.2
    assert session.headers["Accept"] == "audio/pcm"


@pytest.mark.asyncio
async def test_mist_text_settings_are_included_in_http_payload():
    session = _CapturingSession()
    service = RimeHttpTTSService(
        api_key="test-api-key",
        aiohttp_session=session,
        sample_rate=24000,
        settings=RimeHttpTTSService.Settings(
            model="mistv2",
            voice="luna",
            pauseBetweenBrackets=True,
            phonemizeBetweenBrackets=True,
            noTextNormalization=True,
        ),
    )

    _ = [frame async for frame in service.run_tts("Hello", "context")]

    assert session.payload["modelId"] == "mistv2"
    assert session.payload["pauseBetweenBrackets"] is True
    assert session.payload["phonemizeBetweenBrackets"] is True
    assert session.payload["noTextNormalization"] is True
