#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for AWS Transcribe automatic language identification."""

from unittest.mock import AsyncMock
from urllib.parse import parse_qsl, urlsplit

import pytest

from pipecat.frames.frames import InterimTranscriptionFrame, TranscriptionFrame
from pipecat.services.aws import stt as aws_stt
from pipecat.services.aws.stt import AWSTranscribeSTTService
from pipecat.services.aws.utils import get_presigned_url
from pipecat.transcriptions.language import Language
from pipecat.utils.errors import ErrorCategory

Settings = AWSTranscribeSTTService.Settings

_CREDENTIALS = {"access_key": "key-id", "secret_key": "secret", "session_token": None}


class _FakeWebsocket:
    def __init__(self, messages):
        self._messages = messages

    def __aiter__(self):
        return self._iter_messages()

    async def _iter_messages(self):
        for message in self._messages:
            yield message


def _service(**settings) -> AWSTranscribeSTTService:
    return AWSTranscribeSTTService(
        api_key="secret",
        aws_access_key_id="key-id",
        region="us-east-1",
        settings=Settings(**settings),
    )


def _connected_service(**settings) -> AWSTranscribeSTTService:
    service = _service(**settings)
    service._websocket = object()
    service._disconnect = AsyncMock()
    service._connect = AsyncMock()
    service.push_error = AsyncMock()
    return service


def _query(url: str) -> list[tuple[str, str]]:
    return parse_qsl(urlsplit(url).query)


def test_presigned_url_identifies_language_among_options():
    url = get_presigned_url(
        region="us-east-1",
        credentials=_CREDENTIALS,
        identify_language=True,
        language_options=["en-US", "es-US"],
        preferred_language="es-US",
    )

    params = dict(_query(url))
    assert params["identify-language"] == "true"
    assert params["language-options"] == "en-US,es-US"
    assert params["preferred-language"] == "es-US"
    assert "language-code" not in params
    assert "identify-multiple-languages" not in params


def test_presigned_url_keeps_signed_parameters_in_name_order():
    url = get_presigned_url(
        region="us-east-1",
        credentials=_CREDENTIALS,
        identify_multiple_languages=True,
        language_options=["en-US", "es-US"],
        preferred_language="en-US",
        show_speaker_label=True,
    )

    # SigV4 signs the query with its parameters sorted; the signature comes last.
    names = [name for name, _ in _query(url)][:-1]
    assert names == sorted(names)


def test_configured_language_is_sent_without_language_options():
    service = _service(language=Language.ES)

    assert service._language_params() == {"language_code": "es-ES"}


@pytest.mark.parametrize(
    ("settings", "expected"),
    [
        (
            dict(language_options=[Language.EN, Language.ES]),
            {
                "identify_language": True,
                "identify_multiple_languages": False,
                "language_options": ["en-US", "es-ES"],
                "preferred_language": None,
            },
        ),
        (
            dict(
                language_options=[Language.EN, Language.ES_US],
                identify_multiple_languages=True,
                preferred_language=Language.ES_US,
            ),
            {
                "identify_language": False,
                "identify_multiple_languages": True,
                "language_options": ["en-US", "es-US"],
                "preferred_language": "es-US",
            },
        ),
        (
            # Language.EN and Language.EN_US both resolve to en-US.
            dict(language_options=[Language.EN, Language.EN_US, Language.FR]),
            {
                "identify_language": True,
                "identify_multiple_languages": False,
                "language_options": ["en-US", "fr-FR"],
                "preferred_language": None,
            },
        ),
    ],
)
def test_language_options_replace_the_configured_language(settings, expected):
    service = _service(language=Language.DE, **settings)

    assert service._language_params() == expected


@pytest.mark.parametrize(
    ("settings", "match"),
    [
        (dict(identify_multiple_languages=True), "identify_multiple_languages requires"),
        (dict(preferred_language=Language.EN), "preferred_language requires"),
        (dict(language_options=[Language.EN]), "at least two"),
        (dict(language_options=[Language.EN, Language.EN_US]), "at least two"),
        (
            dict(language_options=[Language.EN, Language.ES], preferred_language=Language.FR),
            "must be one of language_options",
        ),
    ],
)
def test_conflicting_settings_fail_at_construction(settings, match):
    with pytest.raises(ValueError, match=match):
        _service(**settings)


def test_preferred_language_matches_options_by_aws_code():
    # The listed Language.EN and the preferred Language.EN_US are both en-US to AWS.
    service = _service(
        language_options=[Language.EN, Language.ES], preferred_language=Language.EN_US
    )

    assert service._language_params()["preferred_language"] == "en-US"


@pytest.mark.asyncio
async def test_runtime_update_turns_language_identification_on_and_reconnects():
    service = _connected_service()

    changed = await service._update_settings(
        Settings(language_options=[Language.EN, Language.ES], preferred_language=Language.ES)
    )

    assert changed.keys() == {"language_options", "preferred_language"}
    service._disconnect.assert_awaited_once()
    service._connect.assert_awaited_once()
    assert service._language_params()["language_options"] == ["en-US", "es-ES"]


@pytest.mark.asyncio
async def test_runtime_update_is_checked_against_the_settings_it_joins():
    service = _connected_service(language_options=[Language.EN, Language.ES])

    changed = await service._update_settings(Settings(preferred_language=Language.ES))

    assert changed.keys() == {"preferred_language"}
    assert service._language_params()["preferred_language"] == "es-ES"


@pytest.mark.asyncio
async def test_conflicting_runtime_update_is_rejected_whole():
    service = _connected_service(
        language_options=[Language.EN, Language.ES], preferred_language=Language.ES
    )

    # Dropping the options while the preferred language is still set conflicts,
    # so the stability change in the same delta is not applied either.
    changed = await service._update_settings(
        Settings(language_options=None, partial_results_stability="low")
    )

    assert changed == {}
    service._disconnect.assert_not_awaited()
    service._connect.assert_not_awaited()
    service.push_error.assert_awaited_once()
    assert "preferred_language requires" in service.push_error.await_args.args[0]
    assert service.push_error.await_args.kwargs["category"] is ErrorCategory.APPLICATION
    assert service._settings.language_options == [Language.EN, Language.ES]
    assert service._settings.partial_results_stability == "high"


@pytest.mark.asyncio
async def test_runtime_update_turns_language_identification_off():
    service = _connected_service(
        language=Language.FR,
        language_options=[Language.EN, Language.ES],
        preferred_language=Language.ES,
    )

    await service._update_settings(Settings(language_options=None, preferred_language=None))

    service._connect.assert_awaited_once()
    assert service._language_params() == {"language_code": "fr-FR"}


def _result(transcript: str, is_partial: bool, language_code: str | None) -> dict:
    result = {"IsPartial": is_partial, "Alternatives": [{"Transcript": transcript}]}
    if language_code:
        result["LanguageCode"] = language_code
    return {"Transcript": {"Results": [result]}}


async def _transcription_frames(service, payloads, monkeypatch):
    pushed = []

    async def fake_push_frame(frame, direction=None):
        pushed.append(frame)

    monkeypatch.setattr(
        aws_stt, "decode_event", lambda message: ({":message-type": "event"}, message)
    )
    service.push_frame = fake_push_frame
    service._websocket = _FakeWebsocket(payloads)

    await service._receive_messages()

    return [
        frame
        for frame in pushed
        if isinstance(frame, (InterimTranscriptionFrame, TranscriptionFrame))
    ]


@pytest.mark.asyncio
async def test_frames_carry_the_identified_language(monkeypatch):
    service = _service(language_options=[Language.EN, Language.ES])

    frames = await _transcription_frames(
        service,
        [_result("Hola", True, "es-US"), _result("Hola.", False, "es-US")],
        monkeypatch,
    )

    assert [(type(f), f.language) for f in frames] == [
        (InterimTranscriptionFrame, "es-US"),
        (TranscriptionFrame, "es-US"),
    ]


@pytest.mark.asyncio
async def test_frames_carry_the_configured_language_without_identification(monkeypatch):
    service = _service(language=Language.FR)

    frames = await _transcription_frames(service, [_result("Bonjour.", False, None)], monkeypatch)

    assert [f.language for f in frames] == ["fr-FR"]
