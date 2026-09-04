#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for which TTS services report a processing-time metric."""

import pytest

from pipecat.services.cartesia.tts import CartesiaHttpTTSService, CartesiaTTSService
from pipecat.services.deepgram.tts import DeepgramHttpTTSService, DeepgramTTSService
from pipecat.services.elevenlabs.tts import ElevenLabsHttpTTSService, ElevenLabsTTSService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.services.tts_service import TTSService, WebsocketTTSService

# Services whose run_tts completes synthesis before returning.
SYNCHRONOUS = [
    CartesiaHttpTTSService,
    DeepgramHttpTTSService,
    ElevenLabsHttpTTSService,
    OpenAITTSService,
]

# Services whose run_tts sends the text and returns, leaving audio to arrive
# on the receive task.
HANDS_OFF = [
    CartesiaTTSService,
    DeepgramTTSService,
    ElevenLabsTTSService,
]


@pytest.mark.parametrize("cls", SYNCHRONOUS)
def test_synchronous_services_report_processing_metrics(cls):
    service = cls.__new__(cls)
    assert service.supports_processing_metrics


@pytest.mark.parametrize("cls", HANDS_OFF)
def test_handoff_services_do_not_report_processing_metrics(cls):
    service = cls.__new__(cls)
    assert not service.supports_processing_metrics


def test_default_is_to_report():
    # A service that synthesizes inside run_tts is the common case, so the base
    # class opts in and the handoff services opt out.
    assert TTSService.supports_processing_metrics.fget(None)
    assert not WebsocketTTSService.supports_processing_metrics.fget(None)
