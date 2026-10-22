#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for how a TTSService reads expression tags in expressive conversation."""

import re
from collections.abc import AsyncGenerator

import pytest
from loguru import logger

from pipecat.expressive.adapter import ExpressionAdapter
from pipecat.expressive.expression import ExpressionCapabilities, ExpressionTerm, ModeScope
from pipecat.expressive.parser import Event, Pause, Text
from pipecat.expressive.tts_expression import TTSExpression, _segment_frames
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
from pipecat.processors.aggregators.llm_text_processor import LLMTextProcessor
from pipecat.services.tts_service import TextAggregationMode, TTSService
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.utils.text.transforms.strip_markdown import strip_markdown
from tests.test_tts_frame_ordering import (
    MockHttpPushTextTTSService,
    _MockWordTimestampHttpTTSService,
)

EXPRESSIVE = PipelineParams(enable_expressive_conversation=True)

REPLY = ["<<cheerful>> It's [[la", "ugh]] ^back up^ now. [[pause 400]] Bye!"]
WRITTEN = "<<cheerful>> It's [[laugh]] ^back up^ now. [[pause 400]] Bye!"


class BracketAdapter(ExpressionAdapter):
    """Renders modes as {modes} where they change, events as <name>, pauses as <Nms>."""

    def capabilities(self):
        return ExpressionCapabilities(
            events=[ExpressionTerm("laugh")],
            modes=[ExpressionTerm("cheerful"), ExpressionTerm("calm")],
            pause_range_ms=(100, 1000),
            emphasis=True,
        )

    def render_text(self, text, modes, previous_modes):
        changed = modes != (previous_modes or [])
        return f"{{{','.join(modes)}}}{text}" if changed else text

    def render_event(self, name):
        return f"<{name}>"

    def render_pause(self, duration_ms):
        return f"<{duration_ms}ms>"


class PushTextTTSService(MockHttpPushTextTTSService):
    expression_adapter_class = BracketAdapter


class WordTimestampTTSService(_MockWordTimestampHttpTTSService):
    """Reports timestamps for the words it speaks, not for its markup."""

    expression_adapter_class = BracketAdapter

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        async for frame in super().run_tts(re.sub(r"\{[^}]*\}|<[^>]*>", " ", text), context_id):
            yield frame


SERVICES = [PushTextTTSService, WordTimestampTTSService]


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
async def test_the_tts_is_sent_the_words_with_their_rendered_expression(service):
    tts = service()
    requests = record_requests(tts)
    await run_test(tts, frames_to_send=response(*REPLY), pipeline_params=EXPRESSIVE)

    assert requests == [
        "{cheerful}It's <laugh> {cheerful,emphasis}back up {cheerful}now.",
        "<400ms>{cheerful}Bye!",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("service", SERVICES)
async def test_the_context_keeps_the_reply_as_written(service):
    assert await context_after(service(), response(*REPLY)) == WRITTEN


@pytest.mark.asyncio
async def test_words_carry_their_modes_and_events_come_in_the_order_written():
    down, _ = await run_test(
        PushTextTTSService(), frames_to_send=response(*REPLY), pipeline_params=EXPRESSIVE
    )

    # What reaches the context and clients: words and events, as they are spoken.
    written = [(f.text_type, f.text, f.modes) for f in down if isinstance(f, TTSTextFrame)]
    assert written == [
        (TextType.SENTENCE, "It's", ["cheerful"]),
        (TextType.EXPRESSION_EVENT, "laugh", None),
        (TextType.SENTENCE, "back up", ["cheerful", "emphasis"]),
        (TextType.SENTENCE, "now.", ["cheerful"]),
        (TextType.EXPRESSION_PAUSE, "400", None),
        (TextType.SENTENCE, "Bye!", ["cheerful"]),
    ]


@pytest.mark.asyncio
async def test_word_timestamps_report_words_and_events_in_order():
    down, _ = await run_test(
        WordTimestampTTSService(),
        frames_to_send=response(*REPLY),
        pipeline_params=EXPRESSIVE,
    )

    words = [f.text for f in down if isinstance(f, TTSTextFrame)]
    assert words == ["It's", "<laugh>", "back", "up", "now.", "<400ms>", "Bye!"]


@pytest.mark.asyncio
async def test_each_change_of_modes_is_a_segment_of_its_own():
    down, _ = await run_test(
        WordTimestampTTSService(),
        frames_to_send=response("<<calm>> It's ^back up^ now."),
        pipeline_params=EXPRESSIVE,
    )

    segments = [
        (f.text, f.modes, f.raw_text)
        for f in down
        if type(f) is AggregatedTextFrame and f.will_be_spoken
    ]
    assert segments == [
        ("It's", ["calm"], "<<calm>> It's"),
        ("back up", ["calm", "emphasis"], "^back up^"),
        ("now.", ["calm"], "now."),
    ]


@pytest.mark.asyncio
async def test_tags_split_across_tokens_are_read_in_token_mode():
    tts = PushTextTTSService(text_aggregation_mode=TextAggregationMode.TOKEN)
    requests = record_requests(tts)
    context = await context_after(tts, response("<<chee", "rful>> It's", " [[la", "ugh]] fine."))

    assert "".join(requests) == "{cheerful}It's <laugh> fine."
    assert context == "<<cheerful>> It's [[laugh]] fine."


@pytest.mark.asyncio
@pytest.mark.parametrize("service", SERVICES)
async def test_a_tts_speak_frame_is_read_too(service):
    tts = service()
    requests = record_requests(tts)
    context = await context_after(tts, [TTSSpeakFrame("<<cheerful>> Welcome back!")])

    assert requests == ["{cheerful}Welcome back!"]
    assert context == "<<cheerful>> Welcome back!"


@pytest.mark.asyncio
async def test_text_aggregated_upstream_is_read_too():
    tts = PushTextTTSService()
    requests = record_requests(tts)
    sentence = AggregatedTextFrame("<<calm>> One moment.", TextType.SENTENCE)
    context = await context_after(
        tts, [LLMFullResponseStartFrame(), sentence, LLMFullResponseEndFrame()]
    )

    assert requests == ["{calm}One moment."]
    assert context == "<<calm>> One moment."


@pytest.mark.asyncio
async def test_a_tag_cut_off_by_an_interruption_is_never_spoken():
    tts = PushTextTTSService(text_aggregation_mode=TextAggregationMode.TOKEN)
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


@pytest.mark.asyncio
async def test_a_service_without_an_adapter_speaks_only_the_words():
    tts = MockHttpPushTextTTSService()
    requests = record_requests(tts)
    down, _ = await run_test(tts, frames_to_send=response(*REPLY), pipeline_params=EXPRESSIVE)

    assert requests == ["It's back up now.", "Bye!"]
    # What clients see matches what was spoken: no modes, and no events.
    assert [(f.text, f.modes) for f in down if isinstance(f, TTSTextFrame)] == [
        ("It's", []),
        ("back up", []),
        ("now.", []),
        ("Bye!", []),
    ]
    assert not any(
        isinstance(f, AggregatedTextFrame) and f.text_type == TextType.EXPRESSION_EVENT
        for f in down
    )
    context = await context_after(MockHttpPushTextTTSService(), response(*REPLY))
    assert context == "<<cheerful>> It's ^back up^ now. Bye!"


@pytest.mark.asyncio
async def test_a_mode_the_service_cant_perform_is_left_out():
    tts = PushTextTTSService()
    requests = record_requests(tts)
    down, _ = await run_test(
        tts, frames_to_send=response("<<whisper calm>> Shh."), pipeline_params=EXPRESSIVE
    )

    assert requests == ["{calm}Shh."]
    assert [f.modes for f in down if isinstance(f, TTSTextFrame)] == [["calm"]]


@pytest.mark.asyncio
async def test_each_unperformed_tag_is_warned_about_as_written():
    warnings = []
    sink = logger.add(lambda message: warnings.append(message.record["message"]), level="WARNING")
    try:
        await run_test(
            MockHttpPushTextTTSService(),
            frames_to_send=response(*REPLY),
            pipeline_params=EXPRESSIVE,
        )
    finally:
        logger.remove(sink)

    unperformed = [
        w.split("can't perform ")[1].split(",")[0] for w in warnings if "can't perform" in w
    ]
    assert unperformed == ["<<cheerful>>", "[[laugh]]", "^emphasis^", "[[pause 400]]"]


@pytest.mark.asyncio
async def test_an_event_the_service_cant_perform_is_left_out():
    tts = PushTextTTSService()
    requests = record_requests(tts)
    context = await context_after(tts, response("Ha [[snort]] okay."))

    assert requests == ["Ha okay."]
    assert context == "Ha okay."


@pytest.mark.asyncio
async def test_a_pause_is_kept_within_the_services_range():
    tts = PushTextTTSService()
    requests = record_requests(tts)
    await run_test(
        tts, frames_to_send=response("One. [[pause 5000]] Two."), pipeline_params=EXPRESSIVE
    )

    assert requests == ["One.", "<1000ms>Two."]


class StyleAdapter(ExpressionAdapter):
    """For a service that takes modes with each request."""

    def capabilities(self):
        return ExpressionCapabilities(
            modes=[ExpressionTerm("calm"), ExpressionTerm("cheerful")],
            mode_scope=ModeScope.SENTENCE,
        )


class StyleTTSService(MockHttpPushTextTTSService):
    """Records the modes each request is spoken in, as a style."""

    expression_adapter_class = StyleAdapter

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.styles: list[tuple[str, str]] = []

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        self.styles.append((text, " ".join(self.expression_modes) or "plain"))
        async for frame in super().run_tts(text, context_id):
            yield frame


@pytest.mark.asyncio
async def test_each_request_is_given_the_modes_it_is_spoken_in():
    tts = StyleTTSService()
    await run_test(
        tts,
        frames_to_send=[
            *response("One. <<calm>> Two. Three. <<neutral>> Four. <<cheerful>> Five."),
            *response("Six."),
        ],
        pipeline_params=EXPRESSIVE,
    )

    assert tts.styles == [
        ("One.", "plain"),
        ("Two.", "calm"),
        ("Three.", "calm"),
        ("Four.", "plain"),
        ("Five.", "cheerful"),
        ("Six.", "plain"),
    ]


@pytest.mark.asyncio
async def test_each_sentence_states_its_modes():
    # Not every service keeps modes from one request to the next.
    tts = PushTextTTSService()
    requests = record_requests(tts)
    await run_test(
        tts,
        frames_to_send=response("<<calm>> One. Two. <<neutral>> Three."),
        pipeline_params=EXPRESSIVE,
    )

    assert requests == ["{calm}One.", "{calm}Two.", "Three."]


@pytest.mark.asyncio
async def test_streamed_tokens_state_modes_only_where_they_change():
    tts = PushTextTTSService(text_aggregation_mode=TextAggregationMode.TOKEN)
    requests = record_requests(tts)
    await run_test(
        tts,
        frames_to_send=response("<<calm>> One.", " Two."),
        pipeline_params=EXPRESSIVE,
    )

    assert requests == ["{calm}One.", " Two."]


@pytest.mark.asyncio
async def test_each_response_starts_from_the_baseline_when_streaming_tokens():
    tts = PushTextTTSService(text_aggregation_mode=TextAggregationMode.TOKEN)
    requests = record_requests(tts)
    await run_test(
        tts,
        # The first response ends in text the parser holds until its end.
        frames_to_send=[*response("<<calm>> One ^two"), *response("<<calm>> Three.")],
        pipeline_params=EXPRESSIVE,
    )

    assert requests[-1] == "{calm}Three."


@pytest.mark.asyncio
async def test_a_request_is_spoken_in_the_modes_its_words_start_in():
    tts = StyleTTSService()
    down, _ = await run_test(
        tts,
        frames_to_send=response("The code is <<calm>> four two. Anything else?"),
        pipeline_params=EXPRESSIVE,
    )

    # The change takes effect from the next request, as a mode lasts until the next tag.
    assert tts.styles == [("The code is four two.", "plain"), ("Anything else?", "calm")]
    assert [(f.text, f.modes) for f in down if isinstance(f, TTSTextFrame)] == [
        ("The code is", []),
        ("four two.", []),
        ("Anything else?", ["calm"]),
    ]


class LearnedAdapter(ExpressionAdapter):
    """An adapter whose vocabulary is learned after it is created, as at start."""

    def __init__(self):
        self.events: list[str] = []

    def capabilities(self):
        return ExpressionCapabilities(events=[ExpressionTerm(name) for name in self.events])


def test_the_parser_knows_capabilities_learned_before_first_use():
    adapter = LearnedAdapter()
    expression = TTSExpression(lambda: adapter, owner="tts")
    adapter.events = ["laugh"]

    # A declared event keeps its kind in mode brackets.
    segments = expression.feed("<<laugh>> Hi.") + expression.flush()

    assert isinstance(segments[0], Event) and segments[0].name == "laugh"
    assert isinstance(segments[1], Text)


def test_pieces_leave_out_what_the_service_cant_perform():
    expression = TTSExpression(LearnedAdapter, owner="tts")
    segments = expression.read_whole("<<calm>> Hi. [[laugh]]")

    pieces = expression.pieces(
        AggregatedTextFrame("", TextType.SENTENCE),
        segments,
        streaming_tokens=False,
        trim=lambda text: text.strip() or None,
    )

    assert [(frame.text, frame.modes, text) for frame, text in pieces] == [("Hi.", [], "Hi.")]


class MarkupAdapter(ExpressionAdapter):
    """Renders tags as bracketed markup, which its service reports back with its words."""

    def capabilities(self):
        return ExpressionCapabilities(
            events=[ExpressionTerm("laugh")],
            modes=[ExpressionTerm("calm")],
            pause_range_ms=(100, 1000),
        )

    def render_text(self, text, modes, previous_modes):
        changed = modes and modes != (previous_modes or [])
        return f"[{' '.join(modes)}] {text}" if changed else text

    def render_event(self, name):
        return f" [{name}] "

    def render_pause(self, duration_ms):
        return " [pause] "


class ReportingTTSService(_MockWordTimestampHttpTTSService):
    """Reports each token it is sent, markup included, as ElevenLabs' alignment does."""

    expression_adapter_class = MarkupAdapter


async def spoken_words(tts: TTSService, text: str) -> tuple[list[tuple[str, int]], list[str]]:
    warnings = []
    sink = logger.add(lambda message: warnings.append(message.record["message"]), level="WARNING")
    try:
        down, _ = await run_test(tts, frames_to_send=response(text), pipeline_params=EXPRESSIVE)
    finally:
        logger.remove(sink)
    return [(f.text, f.pts) for f in down if isinstance(f, TTSTextFrame)], warnings


@pytest.mark.asyncio
async def test_an_event_is_timed_by_the_markup_its_service_reports():
    words, warnings = await spoken_words(ReportingTTSService(), "It's [[laugh]] back.")

    # A word carries what the TTS was sent: the event's markup.
    assert [text for text, _ in words] == ["It's", "[laugh]", "back."]
    its, laugh, back = (pts for _, pts in words)
    assert its < laugh < back
    assert not [w for w in warnings if "Dropping word" in w]


@pytest.mark.asyncio
async def test_echoed_mode_markup_is_not_spoken():
    words, warnings = await spoken_words(ReportingTTSService(), "<<calm>> It's [[laugh]] back.")

    assert [text for text, _ in words] == ["It's", "[laugh]", "back."]
    assert not [w for w in warnings if "Dropping word" in w]


@pytest.mark.asyncio
async def test_an_unreported_pause_leaves_the_next_event_its_own_time():
    class UnreportedPauseService(ReportingTTSService):
        async def run_tts(self, text, context_id):
            async for frame in super().run_tts(text.replace("[pause]", ""), context_id):
                yield frame

    words, _ = await spoken_words(UnreportedPauseService(), "It's [[pause 400]] [[laugh]] back.")

    assert [text for text, _ in words] == ["It's", "[pause]", "[laugh]", "back."]
    its, pause, laugh, back = (pts for _, pts in words)
    # The pause is completed by what comes after it, the laugh by its own markup.
    assert its < pause <= laugh < back


@pytest.mark.asyncio
@pytest.mark.parametrize("service", SERVICES)
async def test_punctuation_after_emphasis_stays_with_it_in_the_context(service):
    context = await context_after(service(), response("It's ^back^! ^Wow^, nice."))

    assert context == "It's ^back^! ^Wow^, nice."


@pytest.mark.asyncio
async def test_expression_survives_markdown_stripped_before_the_tts():
    tts = PushTextTTSService()
    requests = record_requests(tts)
    await run_test(
        Pipeline([LLMTextProcessor(text_transforms=[("*", strip_markdown)]), tts]),
        frames_to_send=response(WRITTEN),
        pipeline_params=EXPRESSIVE,
    )

    assert "".join(requests) == (
        "{cheerful}It's <laugh> {cheerful,emphasis}back up {cheerful}now.<400ms>{cheerful}Bye!"
    )


@pytest.mark.asyncio
async def test_spacing_between_pieces_is_sent_as_written():
    tts = MockHttpPushTextTTSService()
    requests = record_requests(tts)
    await run_test(
        tts, frames_to_send=response("That is ^very^ ^good^ news."), pipeline_params=EXPRESSIVE
    )

    assert requests == ["That is very good news."]


@pytest.mark.asyncio
async def test_spacing_between_pieces_is_sent_as_written_when_streaming_tokens():
    tts = MockHttpPushTextTTSService(text_aggregation_mode=TextAggregationMode.TOKEN)
    requests = record_requests(tts)
    await run_test(tts, frames_to_send=response("^Wow^ that's great."), pipeline_params=EXPRESSIVE)

    assert "".join(requests) == "Wow that's great."


@pytest.mark.asyncio
async def test_a_response_after_a_tts_speak_frame_states_its_modes():
    tts = PushTextTTSService(text_aggregation_mode=TextAggregationMode.TOKEN)
    requests = record_requests(tts)
    await run_test(
        tts,
        frames_to_send=[TTSSpeakFrame("<<calm>> Hi."), *response("<<calm>> Three.")],
        pipeline_params=EXPRESSIVE,
    )

    assert requests == ["{calm}Hi.", "{calm}Three."]


@pytest.mark.asyncio
async def test_a_patterns_match_is_spoken_as_it_is():
    """Tags are read in the text an LLM writes its reply in, not in a pattern's match."""
    tts = PushTextTTSService()
    requests = record_requests(tts)
    spelled = AggregatedTextFrame("ABC", "spell", raw_text="<spell>ABC</spell>")
    context = await context_after(
        tts, [LLMFullResponseStartFrame(), spelled, LLMFullResponseEndFrame()]
    )

    assert requests == ["ABC"]
    assert context == "<spell>ABC</spell>"


# The frames a unit's segments become.

SEGMENTS = [
    Text(" It's ", ("calm",), "<<calm>> It's "),
    Event("laugh", "[[laugh]]"),
    Text(" fine. ", ("calm",), " fine. "),
    Pause(400, "[[pause 400]]"),
]


def test_words_carry_their_modes_and_events_have_text_types_of_their_own():
    pieces = _segment_frames(SEGMENTS, TextType.SENTENCE, trim=True)

    assert [(f.text_type, f.text, f.modes, bool(f.skip_tts)) for f, _ in pieces] == [
        (TextType.SENTENCE, "It's", ["calm"], False),
        (TextType.EXPRESSION_EVENT, "laugh", None, False),
        (TextType.SENTENCE, "fine.", ["calm"], False),
        (TextType.EXPRESSION_PAUSE, "400", None, False),
    ]
    assert [f.raw_text for f, _ in pieces] == [
        "<<calm>> It's",
        "[[laugh]]",
        "fine.",
        "[[pause 400]]",
    ]


def test_the_text_to_speak_keeps_the_spaces_between_the_words():
    pieces = _segment_frames(SEGMENTS, TextType.SENTENCE, trim=True)

    assert [text for _, text in pieces] == ["It's ", "", " fine.", ""]


def test_tokens_are_left_as_they_are():
    pieces = _segment_frames(SEGMENTS, TextType.TOKEN, trim=False)

    assert [(f.text, text) for f, text in pieces] == [
        (" It's ", " It's "),
        ("laugh", ""),
        (" fine. ", " fine. "),
        ("400", ""),
    ]
