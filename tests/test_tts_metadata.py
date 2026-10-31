#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the metadata a TTSService broadcasts at start."""

from collections.abc import AsyncGenerator

import pytest

from pipecat.adapters.tts.base_expression_adapter import BaseExpressionAdapter
from pipecat.expressive.expression import DeliveryScope, ExpressionCapabilities, ExpressionTerm
from pipecat.frames.frames import Frame, TTSMetadataFrame, TTSUpdateSettingsFrame
from pipecat.pipeline.worker import PipelineParams
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.tests.utils import run_test

CAPABILITIES = ExpressionCapabilities(
    sounds=[ExpressionTerm("laugh", "Something is genuinely funny.")],
    delivery=[ExpressionTerm("cheerful"), ExpressionTerm("whisper")],
    pause_range_ms=(100, 3000),
    emphasis=True,
    delivery_scope=DeliveryScope.SENTENCE,
)


EXPRESSIVE = PipelineParams(enable_expressive_conversation=True)


class DeclaringAdapter(BaseExpressionAdapter):
    def __init__(self, capabilities: ExpressionCapabilities | None):
        self._capabilities = capabilities

    def capabilities(self) -> ExpressionCapabilities | None:
        return self._capabilities


class SilentTTSService(TTSService):
    """A TTS service that never speaks, declaring the given capabilities."""

    def __init__(self, capabilities: ExpressionCapabilities | None = None, **kwargs):
        super().__init__(sample_rate=16000, **kwargs)
        self._declaring_adapter = DeclaringAdapter(capabilities)

    def get_expression_adapter(self) -> BaseExpressionAdapter:
        return self._declaring_adapter

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        yield  # pragma: no cover


@pytest.mark.asyncio
async def test_metadata_is_broadcast_both_ways_at_start():
    """The LLM sits upstream of the TTS, so the metadata goes both ways."""
    tts = SilentTTSService()
    down, up = await run_test(
        tts,
        frames_to_send=[],
        expected_down_frames=[TTSMetadataFrame],
        expected_up_frames=[TTSMetadataFrame],
        pipeline_params=EXPRESSIVE,
    )

    assert down[0].service_name == tts.name
    assert up[0].service_name == tts.name


@pytest.mark.asyncio
async def test_a_service_declares_no_expression_by_default():
    down, _ = await run_test(SilentTTSService(), frames_to_send=[], pipeline_params=EXPRESSIVE)

    assert down[0].expression is None


@pytest.mark.asyncio
async def test_a_service_broadcasts_the_expression_it_declares():
    down, _ = await run_test(
        SilentTTSService(CAPABILITIES), frames_to_send=[], pipeline_params=EXPRESSIVE
    )

    assert down[0].expression == CAPABILITIES


@pytest.mark.asyncio
async def test_nothing_is_broadcast_without_expressive_conversation():
    down, up = await run_test(SilentTTSService(CAPABILITIES), frames_to_send=[])

    assert not [f for f in down + up if isinstance(f, TTSMetadataFrame)]


class ModelTTSService(SilentTTSService):
    """A TTS service that performs expression only on its "expressive" model."""

    def get_expression_adapter(self) -> BaseExpressionAdapter:
        if self._settings.model == "expressive":
            return self._declaring_adapter
        return DeclaringAdapter(None)


@pytest.mark.asyncio
async def test_a_model_change_that_changes_the_expression_is_broadcast():
    down, up = await run_test(
        ModelTTSService(CAPABILITIES),
        frames_to_send=[TTSUpdateSettingsFrame(delta=TTSSettings(model="expressive"))],
        pipeline_params=EXPRESSIVE,
    )

    metadata = [f for f in up if isinstance(f, TTSMetadataFrame)]
    assert [f.expression for f in metadata] == [None, CAPABILITIES]
    assert [f.expression for f in down if isinstance(f, TTSMetadataFrame)] == [None, CAPABILITIES]


@pytest.mark.asyncio
async def test_a_change_that_keeps_the_expression_is_not_broadcast():
    _, up = await run_test(
        ModelTTSService(CAPABILITIES),
        frames_to_send=[TTSUpdateSettingsFrame(delta=TTSSettings(model="plain"))],
        pipeline_params=EXPRESSIVE,
    )

    assert len([f for f in up if isinstance(f, TTSMetadataFrame)]) == 1
