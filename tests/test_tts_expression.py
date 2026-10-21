#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for how a TTSService reads expression tags in expressive conversation."""

import pytest

from pipecat.frames.frames import (
    AggregatedTextFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TextType,
    TTSSpeakFrame,
    TTSTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMAssistantAggregator
from pipecat.services.tts_service import TextAggregationMode, TTSService
from pipecat.tests.utils import SleepFrame, run_test
from tests.test_tts_frame_ordering import (
    MockHttpPushTextTTSService,
    _MockWordTimestampHttpTTSService,
)

EXPRESSIVE = PipelineParams(enable_expressive_conversation=True)

REPLY = ["<<cheerful>> It's [[la", "ugh]] *back up* now. [[pause 400]] Bye!"]
WRITTEN = "<<cheerful>> It's [[laugh]] *back up* now. [[pause 400]] Bye!"

SERVICES = [MockHttpPushTextTTSService, _MockWordTimestampHttpTTSService]


def response(*chunks: str) -> list[Frame]:
    return [
        LLMFullResponseStartFrame(),
        *[LLMTextFrame(c) for c in chunks],
        LLMFullResponseEndFrame(),
    ]


def record_requests(tts: TTSService) -> list[str]:
    requests: list[str] = []

    @tts.event_handler("on_tts_request")
    async def on_tts_request(tts, context_id, text):
        requests.append(text)

    return requests


async def context_after(tts: TTSService, frames: list[Frame]) -> str:
    context = LLMContext()
    await run_test(
        Pipeline([tts, LLMAssistantAggregator(context)]),
        frames_to_send=frames,
        pipeline_params=EXPRESSIVE,
    )
    return context.messages[-1]["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("service", SERVICES)
async def test_the_tts_is_sent_only_the_words(service):
    tts = service()
    requests = record_requests(tts)
    await run_test(tts, frames_to_send=response(*REPLY), pipeline_params=EXPRESSIVE)

    assert requests == ["It's back up now.", "Bye!"]


@pytest.mark.asyncio
@pytest.mark.parametrize("service", SERVICES)
async def test_the_context_keeps_the_reply_as_written(service):
    assert await context_after(service(), response(*REPLY)) == WRITTEN


@pytest.mark.asyncio
async def test_words_carry_their_modes_and_events_come_in_the_order_written():
    down, _ = await run_test(
        MockHttpPushTextTTSService(), frames_to_send=response(*REPLY), pipeline_params=EXPRESSIVE
    )

    # What reaches the context and clients: events, and words as they are spoken.
    written = [
        (f.text_type, f.text, f.modes)
        for f in down
        if isinstance(f, TTSTextFrame) or (isinstance(f, AggregatedTextFrame) and f.skip_tts)
    ]
    assert written == [
        (TextType.SENTENCE, "It's", ["cheerful"]),
        (TextType.EXPRESSION_EVENT, "laugh", None),
        (TextType.SENTENCE, "back up", ["cheerful", "emphasis"]),
        (TextType.SENTENCE, "now.", ["cheerful"]),
        (TextType.EXPRESSION_PAUSE, "400", None),
        (TextType.SENTENCE, "Bye!", ["cheerful"]),
    ]


@pytest.mark.asyncio
async def test_each_change_of_modes_is_a_segment_of_its_own():
    down, _ = await run_test(
        _MockWordTimestampHttpTTSService(),
        frames_to_send=response("<<calm>> It's *back up* now."),
        pipeline_params=EXPRESSIVE,
    )

    segments = [
        (f.text, f.modes, f.raw_text)
        for f in down
        if type(f) is AggregatedTextFrame and f.will_be_spoken
    ]
    assert segments == [
        ("It's", ["calm"], "<<calm>> It's"),
        ("back up", ["calm", "emphasis"], "*back up*"),
        ("now.", ["calm"], "now."),
    ]


@pytest.mark.asyncio
async def test_tags_split_across_tokens_are_read_in_token_mode():
    tts = MockHttpPushTextTTSService(text_aggregation_mode=TextAggregationMode.TOKEN)
    requests = record_requests(tts)
    context = await context_after(tts, response("<<chee", "rful>> It's", " [[la", "ugh]] fine."))

    assert "".join(requests) == "It's  fine."
    assert context == "<<cheerful>> It's [[laugh]] fine."


@pytest.mark.asyncio
@pytest.mark.parametrize("service", SERVICES)
async def test_a_tts_speak_frame_is_read_too(service):
    tts = service()
    requests = record_requests(tts)
    context = await context_after(tts, [TTSSpeakFrame("<<cheerful>> Welcome back!")])

    assert requests == ["Welcome back!"]
    assert context == "<<cheerful>> Welcome back!"


@pytest.mark.asyncio
async def test_text_aggregated_upstream_is_read_too():
    tts = MockHttpPushTextTTSService()
    requests = record_requests(tts)
    sentence = AggregatedTextFrame("<<calm>> One moment.", TextType.SENTENCE)
    context = await context_after(
        tts, [LLMFullResponseStartFrame(), sentence, LLMFullResponseEndFrame()]
    )

    assert requests == ["One moment."]
    assert context == "<<calm>> One moment."


@pytest.mark.asyncio
async def test_a_tag_cut_off_by_an_interruption_is_never_spoken():
    tts = MockHttpPushTextTTSService(text_aggregation_mode=TextAggregationMode.TOKEN)
    requests = record_requests(tts)
    await run_test(
        tts,
        frames_to_send=[
            LLMFullResponseStartFrame(),
            LLMTextFrame("Sure "),
            LLMTextFrame("[[la"),
            SleepFrame(),
            InterruptionFrame(),
            *response("ugh]] Okay."),
        ],
        pipeline_params=EXPRESSIVE,
    )

    assert requests == ["Sure ", "ugh]] Okay."]


@pytest.mark.asyncio
async def test_tags_are_spoken_as_written_without_expressive_conversation():
    tts = MockHttpPushTextTTSService()
    requests = record_requests(tts)
    await run_test(tts, frames_to_send=response("<<cheerful>> Hi!"))

    assert requests == ["<<cheerful>> Hi!"]
