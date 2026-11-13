#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for KokoroTTSService voice validation at construction.

The service module is imported with ``pytest.importorskip`` so the suite is
skipped when the kokoro extra is not installed. The model and voices file are
replaced with stand-ins so nothing is downloaded.
"""

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("pipecat.services.kokoro.tts", exc_type=ImportError)

from pipecat.services.kokoro.tts import KokoroTTSService  # noqa: E402


def _build(**settings):
    with (
        patch("pipecat.services.kokoro.tts._ensure_model_files"),
        patch("pipecat.services.kokoro.tts.Kokoro") as mock_kokoro,
    ):
        mock_kokoro.return_value = MagicMock(voices={"af_heart": object()})
        return KokoroTTSService(settings=KokoroTTSService.Settings(**settings))


def test_a_known_voice_constructs():
    assert _build(voice="af_heart")._settings.voice == "af_heart"


@pytest.mark.parametrize("voice", [None, ""], ids=["none", "blank"])
def test_missing_or_blank_voice_is_rejected(voice):
    with pytest.raises(ValueError, match="Kokoro TTS voice must be specified"):
        _build(voice=voice)


def test_unknown_voice_is_rejected_at_construction():
    with pytest.raises(ValueError, match="'af_nope' is not in the voices file"):
        _build(voice="af_nope")
