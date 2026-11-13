#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for Kokoro TTS language code mapping.

The service module is imported with ``pytest.importorskip`` so the suite is
skipped rather than failing collection when the optional Kokoro dependencies
aren't installed.
"""

import pytest

# The service raises ImportError (not ModuleNotFoundError) when its extra is absent.
pytest.importorskip("pipecat.services.kokoro.tts", exc_type=ImportError)

from pipecat.services.kokoro.tts import language_to_kokoro_language  # noqa: E402
from pipecat.transcriptions.language import Language  # noqa: E402


def test_kokoro_uses_espeak_names_not_iso_codes():
    """Mandarin and French use espeak-ng voice names, which differ from the ISO code.

    espeak-ng has no ``zh`` or bare ``fr`` — a name it doesn't know fails at
    synthesis time rather than falling back.
    """
    assert language_to_kokoro_language(Language.ZH) == "cmn"
    assert language_to_kokoro_language(Language.FR) == "fr-fr"


def test_kokoro_chinese_variants_all_map_to_mandarin():
    """Kokoro's ``zf_``/``zm_`` voices are Mandarin, whatever the script."""
    assert language_to_kokoro_language(Language.ZH_CN) == "cmn"
    assert language_to_kokoro_language(Language.ZH_HK) == "cmn"
    assert language_to_kokoro_language(Language.ZH_TW) == "cmn"
    assert language_to_kokoro_language(Language.CMN) == "cmn"


def test_kokoro_regional_variants_use_their_own_espeak_voice():
    """Regions espeak-ng ships separately keep their own name."""
    assert language_to_kokoro_language(Language.EN_GB) == "en-gb"
    assert language_to_kokoro_language(Language.FR_BE) == "fr-be"
    assert language_to_kokoro_language(Language.FR_CH) == "fr-ch"
    assert language_to_kokoro_language(Language.PT_BR) == "pt-br"
    # espeak-ng has no Canadian French; fr-fr is the nearest it ships.
    assert language_to_kokoro_language(Language.FR_CA) == "fr-fr"


def test_kokoro_base_languages_pass_through():
    assert language_to_kokoro_language(Language.EN) == "en-us"
    assert language_to_kokoro_language(Language.ES) == "es"
    assert language_to_kokoro_language(Language.HI) == "hi"
    assert language_to_kokoro_language(Language.IT) == "it"
    assert language_to_kokoro_language(Language.JA) == "ja"
    assert language_to_kokoro_language(Language.PT) == "pt"
