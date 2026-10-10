#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for SmallestSTTService's Pulse 2.0 model support."""

import pytest

from pipecat.services.smallest.stt import SmallestSTTModel, SmallestSTTService
from pipecat.services.stt_service import STTSettings
from pipecat.transcriptions.language import Language


def test_default_model_is_pulse():
    service = SmallestSTTService(api_key="test-key")
    assert service._settings.model == SmallestSTTModel.PULSE.value


def test_pulse_2_defaults_to_english():
    service = SmallestSTTService(
        api_key="test-key",
        settings=SmallestSTTService.Settings(model=SmallestSTTModel.PULSE_2.value),
    )
    assert service._settings.language == "en"


def test_pulse_2_rejects_non_english_language_at_construction():
    with pytest.raises(ValueError, match="English-only"):
        SmallestSTTService(
            api_key="test-key",
            settings=SmallestSTTService.Settings(
                model=SmallestSTTModel.PULSE_2.value, language=Language.HI
            ),
        )


def test_pulse_rejects_english_only_restriction_does_not_apply():
    """Non-English languages are fine on the default `pulse` model."""
    service = SmallestSTTService(
        api_key="test-key",
        settings=SmallestSTTService.Settings(language=Language.HI),
    )
    assert service._settings.language == "hi"


@pytest.mark.asyncio
async def test_switching_to_pulse_2_with_non_english_language_raises():
    service = SmallestSTTService(api_key="test-key")

    with pytest.raises(ValueError, match="English-only"):
        await service._update_settings(
            STTSettings(model=SmallestSTTModel.PULSE_2.value, language=Language.HI)
        )


def test_emotion_and_gender_detection_default_on():
    service = SmallestSTTService(api_key="test-key")
    assert service._settings.emotion_detection is True
    assert service._settings.gender_detection is True
