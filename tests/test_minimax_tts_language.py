#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for MiniMax TTS language name mapping."""

from pipecat.services.minimax.tts import language_to_minimax_language
from pipecat.transcriptions.language import Language


def test_minimax_base_language_uses_name():
    assert language_to_minimax_language(Language.PT) == "Portuguese"


def test_minimax_regional_variant_uses_base_language_name():
    """MiniMax names languages without regions, so a regional variant uses its base name."""
    assert language_to_minimax_language(Language.PT_BR) == "Portuguese"
    assert language_to_minimax_language(Language.EN_US) == "English"
