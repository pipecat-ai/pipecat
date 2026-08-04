#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import json

import pytest
from websockets.protocol import State

from pipecat.services.maya.tts import (
    MAYA_LANGUAGES,
    MAYA_SAMPLE_RATE,
    MayaTTSService,
    language_to_maya_language,
)
from pipecat.transcriptions.language import Language


@pytest.mark.parametrize(
    "language, expected",
    [
        (Language.HI, "hi"),
        (Language.HI_IN, "hi"),
        (Language.BN, "bn"),
        (Language.GU_IN, "gu"),
        (Language.KN, "kn"),
        (Language.ML, "ml"),
        (Language.MR, "mr"),
        (Language.OR, "or"),
        (Language.PA_IN, "pa"),
        (Language.TA, "ta"),
        (Language.TE_IN, "te"),
    ],
)
def test_language_mapping(language: Language, expected: str):
    assert language_to_maya_language(language) == expected


def test_every_english_locale_maps_to_indian_english():
    # `en` is Indian English; Maya has no British or American variant.
    for language in (Language.EN, Language.EN_US, Language.EN_GB, Language.EN_IN):
        assert language_to_maya_language(language) == "en"


def test_an_unverified_language_falls_back_to_its_base_code():
    # Maya is the authority on what it speaks: send the base code and let it
    # answer, so a language Maya adds later needs no change here.
    assert language_to_maya_language(Language.FR) == "fr"
    assert language_to_maya_language(Language.FR_CA) == "fr"
    assert language_to_maya_language(Language.JA) == "ja"


def test_mapped_codes_are_all_accepted_by_maya():
    for language in (Language.HI, Language.BN, Language.TA, Language.TE, Language.EN):
        assert language_to_maya_language(language) in MAYA_LANGUAGES


def test_defaults():
    service = MayaTTSService(api_key="key")
    assert service._settings.voice == "Ananya"
    assert service._settings.model is None
    assert service._url == "wss://tts.mayaresearch.ai/v1/tts/stream"


def test_language_enum_is_stored_as_a_service_string():
    service = MayaTTSService(
        api_key="key",
        settings=MayaTTSService.Settings(voice="Arjun", language=Language.TA_IN),
    )
    assert service._settings.voice == "Arjun"
    assert service._settings.language == "ta"


def test_native_sample_rate():
    assert MAYA_SAMPLE_RATE == 24000


class _FakeWebsocket:
    """Records what the service sends."""

    def __init__(self):
        self.sent = []
        self.state = State.OPEN

    async def send(self, message):
        self.sent.append(json.loads(message))


@pytest.mark.asyncio
async def test_start_frame_selects_v2_and_carries_voice_and_language():
    service = MayaTTSService(
        api_key="key",
        settings=MayaTTSService.Settings(voice="Arjun", language=Language.HI),
    )
    service._websocket = _FakeWebsocket()
    await service._send_start()

    start = service._websocket.sent[0]
    assert start == {"type": "start", "v2": True, "voice": "Arjun", "language": "hi"}


@pytest.mark.asyncio
async def test_start_frame_omits_the_language_when_unset():
    # No language means Maya auto-detects, which suits code-switched text.
    service = MayaTTSService(api_key="key")
    service._websocket = _FakeWebsocket()
    await service._send_start()

    assert "language" not in service._websocket.sent[0]


@pytest.mark.asyncio
async def test_an_unverified_language_string_is_still_sent():
    # The verified tuple documents what's been tested, it doesn't gate the
    # field, so a language Maya adds later works without a release.
    service = MayaTTSService(api_key="key", settings=MayaTTSService.Settings(language="as"))
    service._websocket = _FakeWebsocket()
    await service._send_start()

    assert service._websocket.sent[0]["language"] == "as"


@pytest.mark.asyncio
async def test_flush_closes_the_turn():
    # Maya holds a turn open while sentences arrive with continue=true; an empty
    # frame with continue=false is what makes it emit `end`.
    service = MayaTTSService(api_key="key")
    service._websocket = _FakeWebsocket()
    await service.flush_audio("turn-1")

    assert service._websocket.sent == [
        {"type": "text", "context_id": "turn-1", "text": "", "continue": False}
    ]


@pytest.mark.asyncio
async def test_settings_update_re_announces_the_voice():
    # Maya re-reads voice and language from a `start` sent mid-session, so a
    # settings change has to be pushed to the open socket to take effect.
    service = MayaTTSService(api_key="key", settings=MayaTTSService.Settings(voice="Ananya"))
    service._websocket = _FakeWebsocket()
    await service._update_settings(MayaTTSService.Settings(voice="Arjun"))

    assert service._websocket.sent[-1]["voice"] == "Arjun"


@pytest.mark.asyncio
async def test_interruption_cancels_the_whole_turn():
    service = MayaTTSService(api_key="key")
    service._websocket = _FakeWebsocket()
    await service.on_audio_context_interrupted("turn-1")

    assert {"type": "cancel", "context_id": "turn-1"} in service._websocket.sent
