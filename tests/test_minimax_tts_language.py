#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for MiniMax TTS language name mapping and emotions."""

import warnings

import pytest

from pipecat.services.minimax.tts import MiniMaxHttpTTSService, language_to_minimax_language
from pipecat.transcriptions.language import Language


def test_minimax_base_language_uses_name():
    assert language_to_minimax_language(Language.PT) == "Portuguese"


def test_minimax_regional_variant_uses_base_language_name():
    """MiniMax names languages without regions, so a regional variant uses its base name."""
    assert language_to_minimax_language(Language.PT_BR) == "Portuguese"
    assert language_to_minimax_language(Language.EN_US) == "English"


@pytest.mark.parametrize("emotion", ["calm", "whisper", "happy", "neutral"])
def test_minimax_input_params_accept_documented_emotions(emotion):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        service = MiniMaxHttpTTSService(
            api_key="key",
            group_id="group",
            aiohttp_session=None,
            params=MiniMaxHttpTTSService.InputParams(emotion=emotion),
        )
    assert service._settings.emotion == emotion
