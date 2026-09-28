#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from websockets.protocol import State

from pipecat.frames.frames import (
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.blynt.stt import (
    BlyntSessionContext,
    BlyntSTTOptions,
    BlyntSTTService,
    DeclaredValues,
    Fact,
    STTLanguages,
)
from pipecat.transcriptions.language import Language


def _options(**kwargs) -> BlyntSTTOptions:
    return BlyntSTTOptions(deployment_id="deploy123", api_key="secret", **kwargs)


def _service(**kwargs) -> BlyntSTTService:
    service = BlyntSTTService(options=_options(**kwargs))
    service._websocket = SimpleNamespace(state=State.OPEN, send=AsyncMock())
    service.push_frame = AsyncMock()
    service.push_error = AsyncMock()
    service.start_ttfb_metrics = AsyncMock()
    service.stop_ttfb_metrics = AsyncMock()
    service.start_processing_metrics = AsyncMock()
    service.stop_processing_metrics = AsyncMock()
    return service


def _sent_events(service: BlyntSTTService) -> list[dict]:
    return [json.loads(call.args[0]) for call in service._websocket.send.await_args_list]


def test_blynt_options_build_the_deployment_url_and_auth_header():
    options = _options()

    assert options.get_ws_url() == "wss://api.blynt.ai/api/v1/deployments/deploy123/ws"
    assert options.get_headers() == {"Authorization": "Bearer secret"}
    assert options.language_code == "fr"
    assert options.session_context is None


@pytest.mark.parametrize(
    "base_url, expected",
    [
        ("https://eu.blynt.ai/", "wss://eu.blynt.ai"),
        ("http://localhost:8000", "ws://localhost:8000"),
        ("localhost:8000", "wss://localhost:8000"),
    ],
)
def test_blynt_options_normalize_the_base_url_scheme(base_url, expected):
    options = _options(base_url=base_url)

    assert options.get_ws_url() == f"{expected}/api/v1/deployments/deploy123/ws"


def test_blynt_options_fall_back_to_the_environment(monkeypatch):
    monkeypatch.setenv("BLYNT_DEPLOYMENT_ID", "env-deploy")
    monkeypatch.setenv("BLYNT_API_KEY", "env-key")

    options = BlyntSTTOptions()

    assert options.deployment_id == "env-deploy"
    assert options.api_key == "env-key"


@pytest.mark.parametrize("missing", ["BLYNT_DEPLOYMENT_ID", "BLYNT_API_KEY"])
def test_blynt_options_require_a_deployment_id_and_an_api_key(monkeypatch, missing):
    monkeypatch.setenv("BLYNT_DEPLOYMENT_ID", "env-deploy")
    monkeypatch.setenv("BLYNT_API_KEY", "env-key")
    monkeypatch.delenv(missing)

    with pytest.raises(ValueError, match=missing):
        BlyntSTTOptions()


def test_blynt_options_reject_an_unsupported_language():
    with pytest.raises(ValueError):
        _options(language="fr-FR")


def test_blynt_session_context_payload():
    assert BlyntSessionContext().to_payload() is None

    context = BlyntSessionContext(
        facts=[Fact(name="domaine", value="immatriculation")],
        hints=[DeclaredValues(values=["plaque", "AA-123-BB"])],
    )

    assert context.to_payload() == {
        "facts": [{"name": "domaine", "value": "immatriculation"}],
        "hints": [{"type": "values", "values": ["plaque", "AA-123-BB"]}],
    }


def test_blynt_settings_carry_the_options_language():
    service = BlyntSTTService(options=_options(language=STTLanguages.EN))

    assert service._settings.language == Language.EN
    assert service._settings.model is None


@pytest.mark.asyncio
async def test_blynt_start_session_includes_the_session_context():
    context = BlyntSessionContext(facts=[Fact(name="domaine", value="immatriculation")])
    service = _service(language=STTLanguages.EN, session_context=context)

    await service._send_start_session()

    assert _sent_events(service) == [
        {
            "type": "start_session",
            "language": "en",
            "turn_taking_mode": "manual",
            "sessionContext": {
                "facts": [{"name": "domaine", "value": "immatriculation"}],
                "hints": [],
            },
        }
    ]


@pytest.mark.asyncio
async def test_blynt_vad_frames_delimit_turns():
    service = _service()

    await service.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await service.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

    assert _sent_events(service) == [{"type": "start_turn"}, {"type": "end_turn"}]


@pytest.mark.asyncio
async def test_blynt_a_partial_pushes_an_interim_transcription():
    service = _service()

    await service._process_response({"type": "turn_partial", "transcript": "bonjour"})

    frame = service.push_frame.await_args.args[0]
    assert isinstance(frame, InterimTranscriptionFrame)
    assert frame.text == "bonjour"
    assert frame.language == Language.FR


@pytest.mark.asyncio
async def test_blynt_a_turn_end_pushes_a_final_transcription():
    service = _service()

    await service._process_response({"type": "turn_ended", "transcript": "bonjour à tous"})

    frame = service.push_frame.await_args.args[0]
    assert isinstance(frame, TranscriptionFrame)
    assert frame.text == "bonjour à tous"
    service.stop_processing_metrics.assert_awaited_once()


@pytest.mark.asyncio
async def test_blynt_a_false_interruption_pushes_nothing():
    service = _service()

    await service._process_response(
        {"type": "turn_ended", "transcript": None, "kind": "false_interruption"}
    )

    service.push_frame.assert_not_awaited()


@pytest.mark.asyncio
async def test_blynt_a_server_error_is_pushed_upstream():
    service = _service()

    await service._process_response({"type": "error", "message": "bad deployment"})

    service.push_error.assert_awaited_once_with(error_msg="bad deployment")
