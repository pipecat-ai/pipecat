#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for Gemini TTS in expressive conversation."""

from unittest.mock import AsyncMock, MagicMock, patch

import google.genai as genai
import pytest

from pipecat.adapters.tts.base_expression_adapter import BaseExpressionAdapter
from pipecat.adapters.tts.services.gemini_adapter import Gemini38ExpressionAdapter
from pipecat.expressive.expression import DeliveryScope, ExpressionDelivery
from pipecat.frames.frames import LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame
from pipecat.pipeline.worker import PipelineParams
from pipecat.services.google.tts import GeminiTTSService
from pipecat.tests.utils import run_test

requires_speech_metadata = pytest.mark.skipif(
    not hasattr(genai.types, "SpeechMetadata"),
    reason="Gemini 3.8 TTS request building requires google-genai >= 2.25.0",
)


class TestGemini38ExpressionAdapter:
    def test_performs_every_universal_delivery_between_sentences(self):
        capabilities = Gemini38ExpressionAdapter().capabilities()
        assert capabilities.allows_free_text_delivery
        assert not capabilities.allows_free_text_sounds
        assert capabilities.delivery_scope == DeliveryScope.SENTENCE
        assert capabilities.emphasis
        assert capabilities.sound_name("clear throat") == "throat-clearing"
        # Delivery is free text, so a name that isn't universal is performed as written,
        # while sounds are a fixed list.
        assert capabilities.delivery_name("british accent") == "british accent"
        assert capabilities.sound_name("beatbox") is None

    def test_delivery_is_a_request_style(self):
        adapter = Gemini38ExpressionAdapter()
        assert adapter.request_settings(["calm", "slow"]) == {"style": "calm, speaking slowly"}
        assert adapter.request_settings(["emphasis"]) == {}
        assert adapter.request_settings([]) == {}
        assert adapter.format_text("Hi.", ["calm"], None) == "Hi."

    def test_emphasis_is_written_in_capitals(self):
        assert Gemini38ExpressionAdapter().format_text("now", ["emphasis"], None) == "NOW"

    def test_sounds_and_pauses(self):
        adapter = Gemini38ExpressionAdapter()
        assert adapter.format_sound("laugh") == " <laugh> "
        assert adapter.format_pause(600) == " <short pause> "
        assert adapter.format_pause(1500) == " <long pause> "


@patch("google.genai.Client")
@pytest.mark.parametrize(
    "model, use_genai, adapter",
    [
        ("gemini-3.8-flash-lite-tts", True, Gemini38ExpressionAdapter),
        ("gemini-3.1-flash-tts-preview", True, BaseExpressionAdapter),
        ("gemini-3.8-flash-tts", False, BaseExpressionAdapter),
    ],
)
def test_expression_adapter_follows_the_model_and_backend(
    mock_genai_client_class, model, use_genai, adapter
):
    with patch("pipecat.services.google.tts.GoogleBaseTTSService._create_client"):
        service = GeminiTTSService(
            api_key="key", use_genai=use_genai, settings=GeminiTTSService.Settings(model=model)
        )
    assert type(service.get_expression_adapter()) is adapter


@requires_speech_metadata
@pytest.mark.asyncio
@patch("google.genai.Client")
async def test_each_request_adds_its_delivery_to_the_prompt(mock_genai_client_class):
    mock_client = MagicMock()
    mock_genai_client_class.return_value = mock_client
    mock_stream = AsyncMock()
    mock_client.aio.models.generate_content_stream = mock_stream

    async def no_audio(*args, **kwargs):
        return
        yield

    mock_stream.side_effect = no_audio

    tts = GeminiTTSService(
        api_key="key",
        settings=GeminiTTSService.Settings(
            model="gemini-3.8-flash-tts", voice="Kore", prompt="A friendly host"
        ),
    )
    await run_test(
        tts,
        frames_to_send=[
            LLMFullResponseStartFrame(),
            LLMTextFrame("<<sympathetic>> I'm sorry. [[sigh]] It's <<slow: A B 7>>."),
            LLMTextFrame(" <<neutral>> ^^Anything^^ else?"),
            LLMFullResponseEndFrame(),
        ],
        pipeline_params=PipelineParams(enable_expressive_conversation=True),
    )

    parts = [call.kwargs["contents"][0].parts[0] for call in mock_stream.call_args_list]
    assert [(part.text, part.speech_metadata.style) for part in parts] == [
        ("I'm sorry.", "A friendly host; sympathetic"),
        ("<sigh> It's A B 7.", "A friendly host; sympathetic, speaking slowly"),
        ("ANYTHING else?", "A friendly host"),
    ]
