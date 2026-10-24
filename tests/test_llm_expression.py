#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the expression an LLM service is taught by the TTS service it feeds."""

import pytest
from loguru import logger

from pipecat.expressive.expression import ExpressionCapabilities, ExpressionTerm
from pipecat.expressive.instructions import expression_instructions
from pipecat.expressive.perception import PerceptionCapabilities, PerceptionTerm
from pipecat.expressive.policy import ExpressionGuidance, ExpressionPolicy
from pipecat.frames.frames import (
    Frame,
    ManuallySwitchServiceFrame,
    PerceptionMetadataFrame,
    TTSMetadataFrame,
)
from pipecat.pipeline.llm_switcher import LLMSwitcher
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.tests.utils import SleepFrame, run_test

EXPRESSIVE = PipelineParams(enable_expressive_conversation=True)

CALM_HAPPY = ExpressionCapabilities(
    delivery=[ExpressionTerm("calm"), ExpressionTerm("happy")], pause_range_ms=(100, 3000)
)
LAUGHTER = ExpressionCapabilities(sounds=[ExpressionTerm("laughter")])


def _llm(name: str | None = None) -> OpenAILLMService:
    return OpenAILLMService(
        api_key="test", name=name, settings=OpenAILLMService.Settings(system_instruction="APP")
    )


def _expressive(policy: ExpressionPolicy) -> PipelineParams:
    return PipelineParams(enable_expressive_conversation=True, expression_policy=policy)


def _metadata(expression: ExpressionCapabilities | None) -> TTSMetadataFrame:
    return TTSMetadataFrame(service_name="tts", expression=expression)


async def _send_up(llm: OpenAILLMService, *frames: Frame, params=EXPRESSIVE):
    """Send frames up to the LLM, as a TTS service downstream of it would."""
    _, up = await run_test(
        llm,
        frames_to_send=list(frames),
        frames_to_send_direction=FrameDirection.UPSTREAM,
        pipeline_params=params,
    )
    return up


@pytest.mark.asyncio
async def test_the_tts_expression_is_taught_after_the_prompt():
    llm = _llm()

    up = await _send_up(llm, _metadata(CALM_HAPPY))

    assert llm._settings.system_instruction == f"APP\n\n{expression_instructions(CALM_HAPPY)}"
    assert [f for f in up if isinstance(f, TTSMetadataFrame)]


@pytest.mark.asyncio
async def test_nothing_is_taught_without_expressive_conversation():
    llm = _llm()

    await _send_up(llm, _metadata(CALM_HAPPY), params=None)

    assert llm._settings.system_instruction == "APP"


@pytest.mark.asyncio
async def test_the_latest_tts_replaces_what_is_taught():
    llm = _llm()

    await _send_up(llm, _metadata(CALM_HAPPY), _metadata(LAUGHTER))

    assert llm._settings.system_instruction == f"APP\n\n{expression_instructions(LAUGHTER)}"


@pytest.mark.asyncio
async def test_a_tts_that_performs_nothing_is_taught_nothing():
    llm = _llm()

    await _send_up(llm, _metadata(CALM_HAPPY), _metadata(None))

    assert llm._settings.system_instruction == "APP"


@pytest.mark.asyncio
async def test_the_policy_shapes_what_is_taught():
    policy = ExpressionPolicy(guidance=[ExpressionGuidance.MATCH_STYLE], delivery=["calm"])
    llm = _llm()

    await _send_up(llm, _metadata(CALM_HAPPY), params=_expressive(policy))

    assert llm._settings.system_instruction == (
        f"APP\n\n{expression_instructions(CALM_HAPPY, policy)}"
    )


@pytest.mark.asyncio
async def test_every_llm_in_a_switcher_is_taught_the_policy():
    policy = ExpressionPolicy(guidance=[ExpressionGuidance.MATCH_STYLE])
    first, second = _llm("first"), _llm("second")

    await run_test(
        LLMSwitcher(llms=[first, second]),
        frames_to_send=[
            ManuallySwitchServiceFrame(service=second),
            SleepFrame(),
            _metadata(CALM_HAPPY),
        ],
        pipeline_params=_expressive(policy),
    )

    assert second._settings.system_instruction == (
        f"APP\n\n{expression_instructions(CALM_HAPPY, policy)}"
    )


@pytest.mark.asyncio
async def test_policy_tags_the_tts_cant_perform_are_warned_about():
    policy = ExpressionPolicy(delivery=["calm", "relieved"], sounds=["sighs"])
    llm = _llm()
    warnings: list[str] = []
    handler_id = logger.add(lambda message: warnings.append(str(message)), level="WARNING")
    try:
        await _send_up(llm, _metadata(CALM_HAPPY), params=_expressive(policy))
    finally:
        logger.remove(handler_id)

    assert any("can't perform <<relieved>>, [[sighs]]" in w for w in warnings)


def test_a_policy_without_expressive_conversation_is_warned_about():
    warnings: list[str] = []
    handler_id = logger.add(lambda message: warnings.append(str(message)), level="WARNING")
    try:
        PipelineWorker(_llm(), params=PipelineParams(expression_policy=ExpressionPolicy()))
    finally:
        logger.remove(handler_id)

    assert any("expression_policy has no effect" in w for w in warnings)


TONE = PerceptionCapabilities(
    [PerceptionTerm("tone", labels=["positive", "neutral", "frustrated"])]
)


@pytest.mark.asyncio
async def test_the_perception_upstream_is_taught_with_the_tts_expression():
    llm = _llm()

    await run_test(
        llm,
        frames_to_send=[PerceptionMetadataFrame(service_name="tone", perception=TONE)],
        pipeline_params=EXPRESSIVE,
    )
    await _send_up(llm, _metadata(CALM_HAPPY))

    assert llm._settings.system_instruction == (
        f"APP\n\n{expression_instructions(CALM_HAPPY, perception=TONE)}"
    )


@pytest.mark.asyncio
async def test_perception_is_not_taught_without_expressive_conversation():
    llm = _llm()

    await run_test(
        llm, frames_to_send=[PerceptionMetadataFrame(service_name="tone", perception=TONE)]
    )

    assert llm._settings.system_instruction == "APP"
