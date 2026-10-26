#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for how a TTSService reads expression tags in expressive conversation."""

import asyncio
import re
from collections.abc import AsyncGenerator

import pytest
from loguru import logger

from pipecat.adapters.tts.base_expression_adapter import BaseExpressionAdapter
from pipecat.expressive.expression import DeliveryScope, ExpressionCapabilities, ExpressionTerm
from pipecat.frames.frames import (
    AggregatedTextFrame,
    AggregatedTextProgressFrame,
    Frame,
    InterruptionFrame,
    LLMContextAssistantTurnFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TextType,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMAssistantAggregator
from pipecat.processors.aggregators.llm_text_processor import LLMTextProcessor
from pipecat.services.tts_service import TextAggregationMode, TTSService
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.utils.text.skip_tags_aggregator import SkipTagsAggregator
from pipecat.utils.text.transforms.strip_markdown import strip_markdown
from tests.test_tts_frame_ordering import (
    MockHttpPushTextTTSService,
    _MockWordTimestampHttpTTSService,
)

EXPRESSIVE = PipelineParams(enable_expressive_conversation=True)

REPLY = ["<<cheerful>> It's [[la", "ugh]] ^^back up^^ now. [[pause 400]] Bye!"]
WRITTEN = "<<cheerful>> It's [[laugh]] ^^back up^^ now. [[pause 400]] Bye!"


class BracketAdapter(BaseExpressionAdapter):
    """Formats delivery as <delivery> where it changes, sounds as <name>, pauses as <Nms>, set apart."""

    def capabilities(self):
        return ExpressionCapabilities(
            sounds=[ExpressionTerm("laugh")],
            delivery=[ExpressionTerm("cheerful"), ExpressionTerm("calm")],
            pause_range_ms=(100, 1000),
            emphasis=True,
        )

    def format_text(self, text, delivery, previous_delivery):
        changed = delivery != (previous_delivery or [])
        return f"<{','.join(delivery)}> {text}" if changed and delivery else text

    def format_sound(self, name):
        return f" <{name}> "

    def format_pause(self, duration_ms):
        return f" <{duration_ms}ms> "


class PushTextTTSService(MockHttpPushTextTTSService):
    expression_adapter_class = BracketAdapter


class WordTimestampTTSService(_MockWordTimestampHttpTTSService):
    """Reports timestamps for the words it speaks, not for its markup."""

    expression_adapter_class = BracketAdapter

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        # Markup is set apart from the words, which keep their punctuation.
        words = re.sub(r" ([!?.,;:])", r"\1", re.sub(r"<[^>]*>", " ", text))
        async for frame in super().run_tts(words, context_id):
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
async def test_the_tts_is_sent_the_words_with_their_formatted_expression(service):
    tts = service()
    requests = record_requests(tts)
    await run_test(tts, frames_to_send=response(*REPLY), pipeline_params=EXPRESSIVE)

    assert requests == [
        "<cheerful> It's <laugh> <cheerful,emphasis> back up <cheerful> now.",
        "<400ms> <cheerful> Bye!",
    ]


@pytest.mark.asyncio
async def test_apps_get_the_words_while_the_context_keeps_the_tags():
    context = LLMContext()
    aggregator = LLMAssistantAggregator(context)
    stopped = []

    @aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message):
        stopped.append(message.content)

    down, _ = await run_test(
        Pipeline([WordTimestampTTSService(), aggregator]),
        frames_to_send=response(*REPLY),
        pipeline_params=EXPRESSIVE,
    )

    assert context.messages[-1]["content"] == WRITTEN
    assert stopped == ["It's back up now. Bye!"]
    assert [f.text for f in down if isinstance(f, LLMContextAssistantTurnFrame)] == [
        "It's back up now. Bye!"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("service", SERVICES)
async def test_the_context_keeps_the_reply_as_written(service):
    assert await context_after(service(), response(*REPLY)) == WRITTEN


@pytest.mark.asyncio
async def test_without_word_timestamps_a_sentence_carries_its_delivery_and_sounds():
    down, _ = await run_test(
        PushTextTTSService(), frames_to_send=response(*REPLY), pipeline_params=EXPRESSIVE
    )

    # Each sentence is reported whole, in the delivery it starts in, after its sounds.
    written = [(f.text_type, f.text, f.delivery) for f in down if isinstance(f, TTSTextFrame)]
    assert written == [
        (TextType.EXPRESSION_SOUND, "laugh", None),
        (TextType.SENTENCE, "It's back up now.", ["cheerful"]),
        (TextType.EXPRESSION_PAUSE, "400", None),
        (TextType.SENTENCE, "Bye!", ["cheerful"]),
    ]


@pytest.mark.asyncio
async def test_word_timestamps_report_each_words_delivery_and_the_sounds_between():
    down, _ = await run_test(
        WordTimestampTTSService(),
        frames_to_send=response(*REPLY),
        pipeline_params=EXPRESSIVE,
    )

    words = [(f.text_type, f.text, f.delivery) for f in down if isinstance(f, TTSTextFrame)]
    assert words == [
        (TextType.WORD, "It's", ["cheerful"]),
        (TextType.EXPRESSION_SOUND, "laugh", None),
        (TextType.WORD, "back", ["cheerful", "emphasis"]),
        (TextType.WORD, "up", ["cheerful", "emphasis"]),
        (TextType.WORD, "now.", ["cheerful"]),
        (TextType.EXPRESSION_PAUSE, "400", None),
        (TextType.WORD, "Bye!", ["cheerful"]),
    ]


@pytest.mark.asyncio
async def test_a_sentence_is_one_segment_shown_without_its_tags():
    down, _ = await run_test(
        WordTimestampTTSService(),
        frames_to_send=response("<<calm>> It's ^^back up^^ now."),
        pipeline_params=EXPRESSIVE,
    )

    segments = [
        (f.text, f.raw_text) for f in down if type(f) is AggregatedTextFrame and f.will_be_spoken
    ]
    assert segments == [("It's back up now.", "<<calm>> It's ^^back up^^ now.")]
    # Its progress carries the delivery of each word as it is spoken.
    progress = [
        (f.accumulated_text, f.delivery) for f in down if isinstance(f, AggregatedTextProgressFrame)
    ]
    assert progress == [
        ("It's", ["calm"]),
        ("It's back", ["calm", "emphasis"]),
        ("It's back up", ["calm", "emphasis"]),
        ("It's back up now.", ["calm"]),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("service", SERVICES)
@pytest.mark.parametrize(
    "mode, chunks",
    [
        (TextAggregationMode.TOKEN, ("<<chee", "rful>>", " It's", " fine.")),
        (TextAggregationMode.TOKEN, ("It's", " fine.", " [[laugh]]")),
        (TextAggregationMode.TOKEN, ("It's", " fine.", " <<calm>>")),
        (TextAggregationMode.SENTENCE, ("It's fine. Bye. [[laugh]]",)),
        (TextAggregationMode.SENTENCE, ("It's fine. Bye. [[snort]]",)),
        (TextAggregationMode.SENTENCE, ("It's fine. Bye. <<calm>>",)),
    ],
)
async def test_a_unit_that_is_only_tags_reaches_the_context(service, mode, chunks):
    # A tag with no words of its own, performed or not, is text the LLM wrote.
    tts = service(text_aggregation_mode=mode)

    assert await context_after(tts, response(*chunks)) == "".join(chunks)


@pytest.mark.asyncio
@pytest.mark.parametrize("service", SERVICES)
@pytest.mark.parametrize("mode", [TextAggregationMode.SENTENCE, TextAggregationMode.TOKEN])
async def test_a_reply_that_is_only_tags_is_recorded_and_ended(service, mode):
    # It speaks nothing, so no audio context ends it: the reply's end does.
    tts = service(text_aggregation_mode=mode)
    context = LLMContext()
    await run_test(
        Pipeline([tts, LLMAssistantAggregator(context)]),
        frames_to_send=[*response("<<calm>>"), SleepFrame(sleep=0.1), *response("Hi there.")],
        pipeline_params=EXPRESSIVE,
    )

    assert [m["content"] for m in context.messages] == ["<<calm>>", "Hi there."]
    assert not tts._tts_contexts


@pytest.mark.asyncio
@pytest.mark.parametrize("service", SERVICES)
@pytest.mark.parametrize(
    "first, first_request", [("\n", "Hi"), (" ", "Hi"), ("<<cheerful>>", "<cheerful> Hi")]
)
async def test_whitespace_before_anything_is_said_isnt_sent_in_token_mode(
    service, first, first_request
):
    # As without expressive conversation: nothing has been said for it to separate.
    tts = service(text_aggregation_mode=TextAggregationMode.TOKEN)
    requests = record_requests(tts)

    await run_test(
        tts, frames_to_send=response(first, " ", "Hi", " there."), pipeline_params=EXPRESSIVE
    )

    assert requests == [first_request, " there."]


@pytest.mark.asyncio
@pytest.mark.parametrize("service", SERVICES)
async def test_a_reply_of_only_whitespace_says_nothing_in_token_mode(service):
    tts = service(text_aggregation_mode=TextAggregationMode.TOKEN)
    requests = record_requests(tts)

    down, _ = await run_test(
        tts, frames_to_send=response("\n", "\n\n", " "), pipeline_params=EXPRESSIVE
    )

    assert requests == []
    assert not [f for f in down if isinstance(f, TTSStartedFrame)]


class SpellingTTSService(PushTextTTSService):
    """Keeps its own tags whole, as Cartesia keeps <spell> tags."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._text_aggregator = SkipTagsAggregator(
            [("<spell>", "</spell>")], aggregation_type=self._text_aggregation_mode
        )


@pytest.mark.asyncio
async def test_a_service_keeps_its_own_tags_whole_too():
    tts = SpellingTTSService()
    requests = record_requests(tts)

    await run_test(
        tts,
        frames_to_send=response("Your code is <spell>A. B. C.</spell> okay? <<calm>> Bye."),
        pipeline_params=EXPRESSIVE,
    )

    assert requests == ["Your code is <spell>A. B. C.</spell> okay?", "<calm> Bye."]


@pytest.mark.asyncio
async def test_tags_split_across_tokens_are_read_in_token_mode():
    tts = PushTextTTSService(text_aggregation_mode=TextAggregationMode.TOKEN)
    requests = record_requests(tts)
    context = await context_after(tts, response("<<chee", "rful>> It's", " [[la", "ugh]] fine."))

    assert "".join(requests) == "<cheerful> It's <laugh> fine."
    assert context == "<<cheerful>> It's [[laugh]] fine."


@pytest.mark.asyncio
@pytest.mark.parametrize("service", SERVICES)
async def test_a_tts_speak_frame_is_read_too(service):
    tts = service()
    requests = record_requests(tts)
    context = await context_after(tts, [TTSSpeakFrame("<<cheerful>> Welcome back!")])

    assert requests == ["<cheerful> Welcome back!"]
    assert context == "<<cheerful>> Welcome back!"


@pytest.mark.asyncio
async def test_text_aggregated_upstream_is_read_too():
    tts = PushTextTTSService()
    requests = record_requests(tts)
    sentence = AggregatedTextFrame("<<calm>> One moment.", TextType.SENTENCE)
    context = await context_after(
        tts, [LLMFullResponseStartFrame(), sentence, LLMFullResponseEndFrame()]
    )

    assert requests == ["<calm> One moment."]
    assert context == "<<calm>> One moment."


@pytest.mark.asyncio
async def test_a_service_that_ends_inputs_in_a_space_gets_one():
    # A service whose inputs are joined verbatim needs the space, or one
    # sentence's last word runs into the next one's first.
    tts = PushTextTTSService(append_trailing_space=True)
    requests = record_requests(tts)
    await run_test(tts, frames_to_send=response(*REPLY), pipeline_params=EXPRESSIVE)

    assert all(request.endswith(" ") for request in requests)


class UnreportingTTSService(_MockWordTimestampHttpTTSService):
    """Never reports the last two words of a request, as a provider can drop events."""

    expression_adapter_class = BracketAdapter

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        words = re.sub(r" ([!?.,;:])", r"\1", re.sub(r"<[^>]*>", " ", text)).split()
        self._word_times = [(word, i * 0.1) for i, word in enumerate(words[:-2])]
        async for frame in super().run_tts(text, context_id):
            yield frame


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "written",
    ["<<calm>> I'm so sorry your order is late.", "<<calm>> Your code is <<slow: A B 7>>, okay?"],
)
async def test_words_the_service_never_reports_reach_the_context_as_written(written):
    assert await context_after(UnreportingTTSService(), response(written)) == written


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
    # What clients see matches what was spoken: no delivery, and no sounds.
    assert [(f.text, f.delivery) for f in down if isinstance(f, TTSTextFrame)] == [
        ("It's back up now.", []),
        ("Bye!", []),
    ]
    context = await context_after(MockHttpPushTextTTSService(), response(*REPLY))
    assert context == "<<cheerful>> It's [[laugh]] ^^back up^^ now. [[pause 400]] Bye!"


@pytest.mark.asyncio
async def test_a_delivery_the_service_cant_perform_is_left_out():
    tts = PushTextTTSService()
    requests = record_requests(tts)
    down, _ = await run_test(
        tts, frames_to_send=response("<<whisper>> Shh."), pipeline_params=EXPRESSIVE
    )

    assert requests == ["Shh."]
    assert [f.delivery for f in down if isinstance(f, TTSTextFrame)] == [[]]


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
    assert unperformed == ["<<cheerful>>", "[[laugh]]", "^^emphasis^^", "[[pause 400]]"]


@pytest.mark.asyncio
async def test_a_sound_the_service_cant_perform_is_left_out():
    tts = PushTextTTSService()
    requests = record_requests(tts)
    context = await context_after(tts, response("Ha [[snort]] okay."))

    assert requests == ["Ha okay."]
    # The context keeps the reply as written.
    assert context == "Ha [[snort]] okay."


@pytest.mark.asyncio
async def test_a_pause_is_kept_within_the_services_range():
    tts = PushTextTTSService()
    requests = record_requests(tts)
    await run_test(
        tts, frames_to_send=response("One. [[pause 5000]] Two."), pipeline_params=EXPRESSIVE
    )

    assert requests == ["One.", "<1000ms> Two."]


class StyleAdapter(BaseExpressionAdapter):
    """For a service that takes delivery with each request."""

    def capabilities(self):
        return ExpressionCapabilities(
            delivery=[ExpressionTerm("calm"), ExpressionTerm("cheerful")],
            delivery_scope=DeliveryScope.SENTENCE,
        )

    def request_settings(self, delivery):
        return {"style": " ".join(delivery)} if delivery else {}


class StyleTTSService(MockHttpPushTextTTSService):
    """Records the delivery each request is spoken in, as a style."""

    expression_adapter_class = StyleAdapter

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.styles: list[tuple[str, str]] = []

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        self.styles.append((text, self.expression_settings.get("style", "plain")))
        async for frame in super().run_tts(text, context_id):
            yield frame


@pytest.mark.asyncio
async def test_each_request_is_given_the_delivery_it_is_spoken_in():
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
async def test_each_sentence_states_its_delivery():
    # Not every service keeps delivery from one request to the next.
    tts = PushTextTTSService()
    requests = record_requests(tts)
    await run_test(
        tts,
        frames_to_send=response("<<calm>> One. Two. <<neutral>> Three."),
        pipeline_params=EXPRESSIVE,
    )

    assert requests == ["<calm> One.", "<calm> Two.", "Three."]


@pytest.mark.asyncio
async def test_streamed_tokens_state_delivery_only_where_it_changes():
    tts = PushTextTTSService(text_aggregation_mode=TextAggregationMode.TOKEN)
    requests = record_requests(tts)
    await run_test(
        tts,
        frames_to_send=response("<<calm>> One.", " Two."),
        pipeline_params=EXPRESSIVE,
    )

    assert requests == ["<calm> One.", " Two."]


@pytest.mark.asyncio
async def test_each_response_starts_from_the_baseline_when_streaming_tokens():
    tts = PushTextTTSService(text_aggregation_mode=TextAggregationMode.TOKEN)
    requests = record_requests(tts)
    await run_test(
        tts,
        # The first response ends in text the parser holds until its end.
        frames_to_send=[*response("<<calm>> One ^^two"), *response("<<calm>> Three.")],
        pipeline_params=EXPRESSIVE,
    )

    assert requests[-1] == "<calm> Three."


@pytest.mark.asyncio
async def test_a_span_sets_the_delivery_of_its_whole_request():
    tts = StyleTTSService()
    await run_test(
        tts,
        frames_to_send=response("Your code is <<calm: A B 7>>. Anything else?"),
        pipeline_params=EXPRESSIVE,
    )

    assert tts.styles == [("Your code is A B 7.", "calm"), ("Anything else?", "plain")]


@pytest.mark.asyncio
async def test_a_request_is_spoken_in_the_delivery_its_words_start_in():
    tts = StyleTTSService()
    down, _ = await run_test(
        tts,
        frames_to_send=response("The code is <<calm>> four two. Anything else?"),
        pipeline_params=EXPRESSIVE,
    )

    # The change takes effect from the next request, as a delivery lasts until the next tag.
    assert tts.styles == [("The code is four two.", "plain"), ("Anything else?", "calm")]
    assert [(f.text, f.delivery) for f in down if isinstance(f, TTSTextFrame)] == [
        ("The code is four two.", []),
        ("Anything else?", ["calm"]),
    ]


class MarkupAdapter(BaseExpressionAdapter):
    """Formats tags as bracketed markup, which its service reports back with its words."""

    def capabilities(self):
        return ExpressionCapabilities(
            sounds=[ExpressionTerm("laugh")],
            delivery=[ExpressionTerm("calm")],
            pause_range_ms=(100, 1000),
        )

    def format_text(self, text, delivery, previous_delivery):
        changed = delivery and delivery != (previous_delivery or [])
        return f"[{' '.join(delivery)}] {text}" if changed else text

    def format_sound(self, name):
        return f" [{name}] "

    def format_pause(self, duration_ms):
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
async def test_a_sound_is_timed_by_the_markup_its_service_reports():
    words, warnings = await spoken_words(ReportingTTSService(), "It's [[laugh]] back.")

    # The markup the service reports times the sound.
    assert [text for text, _ in words] == ["It's", "laugh", "back."]
    its, laugh, back = (pts for _, pts in words)
    assert its < laugh < back
    assert not [w for w in warnings if "Dropping word" in w]


@pytest.mark.asyncio
async def test_a_sound_its_service_reports_is_reported_as_performed():
    down, _ = await run_test(
        ReportingTTSService(),
        frames_to_send=response("It's [[laugh]] back."),
        pipeline_params=EXPRESSIVE,
    )

    sound = next(
        f for f in down if isinstance(f, TTSTextFrame) and f.text_type == TextType.EXPRESSION_SOUND
    )
    # Clients are told a sound was performed when it is spoken, in a segment of its own.
    assert sound.will_be_spoken
    assert sound.segment_id is None


@pytest.mark.asyncio
async def test_reported_delivery_markup_is_not_spoken():
    words, warnings = await spoken_words(ReportingTTSService(), "<<calm>> It's [[laugh]] back.")

    assert [text for text, _ in words] == ["It's", "laugh", "back."]
    assert not [w for w in warnings if "Dropping word" in w]


@pytest.mark.asyncio
@pytest.mark.parametrize("service", [ReportingTTSService, WordTimestampTTSService])
@pytest.mark.parametrize(
    "text, spoken",
    [
        (
            "<<calm>> It's <<neutral: A B 7>>, okay.",
            [("It's", ["calm"]), ("A", []), ("B", []), ("7,", []), ("okay.", ["calm"])],
        ),
        (
            "Email <<calm: docs@acme.com>> today.",
            [("Email", []), ("docs@acme.com", ["calm"]), ("today.", [])],
        ),
    ],
)
async def test_a_spans_words_are_spoken_in_its_delivery(service, text, spoken):
    down, _ = await run_test(service(), frames_to_send=response(text), pipeline_params=EXPRESSIVE)

    assert [(f.text, f.delivery) for f in down if isinstance(f, TTSTextFrame)] == spoken
    assert await context_after(service(), response(text)) == text


@pytest.mark.asyncio
@pytest.mark.parametrize("service", [ReportingTTSService, WordTimestampTTSService])
@pytest.mark.parametrize(
    "text, spoken",
    [
        ("Hi [[laugh]], there.", ["Hi,", "laugh", "there."]),
        ("It's ^^back^^ [[laugh]]! Bye.", ["It's", "back!", "laugh", "Bye."]),
        ("Really [[laugh]]? Yes.", ["Really?", "laugh", "Yes."]),
        ("Wait [[pause 300]]. Okay.", ["Wait.", "300", "Okay."]),
        ("See you [[laugh]]", ["See", "you", "laugh"]),
    ],
)
async def test_a_sound_before_punctuation_is_reported_after_the_words_it_follows(
    service, text, spoken
):
    words, warnings = await spoken_words(service(), text)

    assert [word for word, _ in words] == spoken
    assert not [w for w in warnings if "Dropping word" in w]
    assert await context_after(service(), response(text)) == text


@pytest.mark.asyncio
@pytest.mark.parametrize("service", SERVICES)
async def test_punctuation_after_emphasis_stays_with_it_in_the_context(service):
    context = await context_after(service(), response("It's ^^back^^! ^^Wow^^, nice."))

    assert context == "It's ^^back^^! ^^Wow^^, nice."


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
        "<cheerful> It's <laugh> <cheerful,emphasis> back up <cheerful> now.<400ms> <cheerful> Bye!"
    )


@pytest.mark.asyncio
async def test_spacing_between_pieces_is_sent_as_written():
    tts = MockHttpPushTextTTSService()
    requests = record_requests(tts)
    await run_test(
        tts, frames_to_send=response("That is ^^very^^ ^^good^^ news."), pipeline_params=EXPRESSIVE
    )

    assert requests == ["That is very good news."]


@pytest.mark.asyncio
async def test_spacing_between_pieces_is_sent_as_written_when_streaming_tokens():
    tts = MockHttpPushTextTTSService(text_aggregation_mode=TextAggregationMode.TOKEN)
    requests = record_requests(tts)
    await run_test(
        tts, frames_to_send=response("^^Wow^^ that's great."), pipeline_params=EXPRESSIVE
    )

    assert "".join(requests) == "Wow that's great."


@pytest.mark.asyncio
async def test_a_response_after_a_tts_speak_frame_states_its_delivery():
    tts = PushTextTTSService(text_aggregation_mode=TextAggregationMode.TOKEN)
    requests = record_requests(tts)
    await run_test(
        tts,
        frames_to_send=[TTSSpeakFrame("<<calm>> Hi."), *response("<<calm>> Three.")],
        pipeline_params=EXPRESSIVE,
    )

    assert requests == ["<calm> Hi.", "<calm> Three."]


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


class StreamingTTSService(TTSService):
    """Streams each word's audio and timestamp after the request, as a WebSocket service does."""

    expression_adapter_class = BracketAdapter

    def __init__(self):
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            push_text_frames=False,
            stop_frame_timeout_s=0.2,
            sample_rate=16000,
        )

    def can_generate_metrics(self) -> bool:
        return False

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        words = re.sub(r" ([!?.,;:])", r"\1", re.sub(r"<[^>]*>", " ", text)).split()
        self.create_task(self._stream(words, context_id))
        yield None

    async def _stream(self, words: list[str], context_id: str):
        for i, word in enumerate(words):
            await asyncio.sleep(0.05)
            audio = TTSAudioRawFrame(b"\x00" * 320, 16000, 1, context_id=context_id)
            await self.append_to_audio_context(context_id, audio)
            await self.add_word_timestamps([(word, i * 0.05)], context_id=context_id)


@pytest.mark.asyncio
async def test_uninterruptible_speech_keeps_its_delivery_through_an_interruption():
    speech = TTSSpeakFrame("<<calm>> One two three four five six.")
    speech.interruptible = False
    down, _ = await run_test(
        StreamingTTSService(),
        frames_to_send=[speech, SleepFrame(sleep=0.15), InterruptionFrame(), SleepFrame(sleep=0.8)],
        pipeline_params=EXPRESSIVE,
    )

    words = [(f.text, f.delivery) for f in down if isinstance(f, TTSTextFrame)]
    assert words == [(word, ["calm"]) for word in ["One", "two", "three", "four", "five", "six."]]
