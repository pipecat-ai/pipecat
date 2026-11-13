#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for Inworld TTS language code mapping."""

from pipecat.services.inworld.tts import language_to_inworld_language
from pipecat.transcriptions.language import Language


def test_inworld_base_languages_resolve_to_canonical_regional_tags():
    """Base GA languages should use the regional tags emitted by Inworld Playground."""
    assert language_to_inworld_language(Language.EN) == "en-US"
    assert language_to_inworld_language(Language.RU) == "ru-RU"
    assert language_to_inworld_language(Language.FR) == "fr-FR"
    assert language_to_inworld_language(Language.ZH) == "zh-CN"


def test_inworld_regional_languages_are_preserved():
    """Explicit regional variants should be passed through as supported BCP-47 tags."""
    assert language_to_inworld_language(Language.EN_GB) == "en-GB"
    assert language_to_inworld_language(Language.PT_PT) == "pt-PT"
    assert language_to_inworld_language(Language.RU_RU) == "ru-RU"


def test_inworld_other_languages_are_passed_through_as_bcp47_tags():
    """Languages outside the canonical locale map should keep their BCP-47 enum value."""
    assert language_to_inworld_language(Language.SV_SE) == "sv-SE"
    assert language_to_inworld_language(Language.UK_UA) == "uk-UA"
