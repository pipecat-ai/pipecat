#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for XTTSService voice validation at construction."""

from unittest.mock import MagicMock

import pytest

from pipecat.services.xtts.tts import XTTSService


def _build(**settings):
    return XTTSService(
        base_url="http://localhost:8000",
        aiohttp_session=MagicMock(),
        settings=XTTSService.Settings(**settings),
    )


def test_a_voice_constructs():
    assert _build(voice="Claribel Dervla")._settings.voice == "Claribel Dervla"


@pytest.mark.parametrize("voice", [None, ""], ids=["none", "blank"])
def test_missing_or_blank_voice_is_rejected(voice):
    with pytest.raises(ValueError, match="XTTS voice must be specified"):
        _build(voice=voice)
