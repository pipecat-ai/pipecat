#
# Copyright (c) 2024–2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the text flow tracer: the real processors, traced over a response."""

from pathlib import Path

import pytest
import yaml

from pipecat.textflow import (
    AggregatorSpec,
    ChunkingSpec,
    LLMTextProcessorSpec,
    PatternSpec,
    RTVISpec,
    TextFlowRunError,
    TextFlowSetup,
    TransformSpec,
    TTSSpec,
    run_textflow,
)
from pipecat.textflow.capture import TextFlowCaptureError, capture_bot, mirror_setup
from pipecat.textflow.mocks import chunk_text, tokenize
from pipecat.textflow.setup import TokenizationSpec, presets
from pipecat.textflow.transforms import build_transform, registered_transforms

CODE_HELPER_TEXT = (
    "Your card is <card>1234-5678-9012-3456</card>. Run <code>npm install</code> to start."
)
BOT_PATH = Path(__file__).parent / "textflow_bot.py"


def _rows(trace):
    return [(s.text_type, s.llm_text, s.segment_text, s.tts_text) for s in trace.segments]


@pytest.mark.asyncio
async def test_code_helper_reproduces_the_architecture_table():
    """The six segments of the word-tracking README, with their three texts."""
    trace = await run_textflow(TextFlowSetup.preset("code-helper"), CODE_HELPER_TEXT)

    assert _rows(trace) == [
        ("sentence", "Your card is ", "Your card is", "Your card is"),
        (
            "credit_card",
            "<card>1234-5678-9012-3456</card>",
            "1234-5678-9012-3456",
            "<spell>1234-5678-9012-3456</spell>",
        ),
        ("sentence", ".", ".", "."),
        ("sentence", " Run ", "Run", "Run"),
        ("code", "<code>npm install</code>", "npm install", None),
        ("sentence", " to start.", "to start.", "to start."),
    ]
    assert [s.outcome for s in trace.segments] == ["spoken"] * 4 + ["skipped", "spoken"]
    assert [r["text"] for r in trace.provider_requests] == [
        "Your card is",
        "<spell>1234-5678-9012-3456</spell>",
        ".",
        "Run",
        "to start.",
    ]


@pytest.mark.asyncio
async def test_code_helper_consumers():
    """The context keeps the tags, the client gets the redaction, the code block is unspoken."""
    trace = await run_textflow(TextFlowSetup.preset("code-helper"), CODE_HELPER_TEXT)

    [message] = trace.context_messages
    assert message["role"] == "assistant"
    assert "<card>1234-5678-9012-3456</card>" in message["content"]
    assert "<code>npm install</code>" in message["content"]

    card = trace.segments[1]
    assert {m["data"]["text"] for m in card.rtvi_messages} == {"XXXX-XXXX-XXXX-3456"}
    assert card.rtvi_messages[-1]["data"]["spoken_status"] == "completed"
    assert card.context_text == "<card>1234-5678-9012-3456</card>"
    assert [w.llm_text for w in card.words] == ["<card>1234-5678-9012-3456</card>"]

    code = trace.segments[4]
    assert code.events[0].stage == "tts" and code.events[0].kind == "skip"
    assert code.context_text == "<code>npm install</code>"
    assert not code.words


@pytest.mark.asyncio
async def test_words_split_the_segment_text_exactly():
    """Every progress split rebuilds the segment text, as the README guarantees."""
    trace = await run_textflow(TextFlowSetup.preset("plain"), "Hello there. How are you today?")

    for segment in trace.segments:
        assert segment.words
        for word in segment.words:
            assert (word.accumulated or "") + (word.remaining or "") == segment.segment_text
        assert segment.words[-1].remaining == ""
        statuses = [m["data"]["spoken_status"] for m in segment.rtvi_messages]
        assert statuses[-1] == "completed"


@pytest.mark.asyncio
async def test_bot_llm_and_tts_messages_are_off_unless_asked_for():
    """The client gets bot-output alone by default; the LLM and TTS lifecycle and
    deprecated text messages only when the setup turns them on."""
    trace = await run_textflow(TextFlowSetup(), "Hello there.")
    types = {m["type"] for m in trace.rtvi_messages}
    assert not any(t.startswith("bot-llm") or t.startswith("bot-tts") for t in types)
    assert "bot-output" in types

    setup = TextFlowSetup(rtvi=RTVISpec(bot_llm=True, bot_tts=True))
    trace = await run_textflow(setup, "Hello there.")
    types = {m["type"] for m in trace.rtvi_messages}
    assert {"bot-llm-started", "bot-llm-stopped", "bot-tts-started", "bot-tts-stopped"} <= types
    tokens = [m["data"]["text"] for m in trace.rtvi_messages if m["type"] == "bot-llm-text"]
    assert "".join(tokens) == "Hello there."
    spoken = [m["data"]["text"] for m in trace.rtvi_messages if m["type"] == "bot-tts-text"]
    assert spoken == ["Hello", "there."]


@pytest.mark.asyncio
async def test_llm_text_processor_transform_reaches_every_consumer():
    """A rewrite before SPLIT 1 changes the context, the client and the provider alike."""
    setup = TextFlowSetup(
        llm_text_processor=LLMTextProcessorSpec(transforms=[TransformSpec(name="upper")]),
    )
    trace = await run_textflow(setup, "Hello there.")

    [segment] = trace.segments
    assert segment.events[0].stage == "llm_text_processor"
    assert (segment.llm_text, segment.segment_text, segment.tts_text) == ("HELLO THERE.",) * 3
    assert trace.context_messages[0]["content"] == "HELLO THERE."
    assert segment.rtvi_messages[-1]["data"]["text"] == "HELLO THERE."


@pytest.mark.asyncio
async def test_tts_transform_reaches_the_provider_only():
    """A rewrite at SPLIT 2 is heard but never recorded or displayed."""
    setup = TextFlowSetup(tts=TTSSpec(transforms=[TransformSpec(name="upper")]))
    trace = await run_textflow(setup, "Hello there.")

    [segment] = trace.segments
    assert segment.events[0].stage == "tts"
    assert (segment.llm_text, segment.segment_text) == ("Hello there.", "Hello there.")
    assert segment.tts_text == "HELLO THERE."
    assert trace.context_messages[0]["content"] == "Hello there."
    assert segment.rtvi_messages[-1]["data"]["text"] == "Hello there."


@pytest.mark.asyncio
async def test_transform_that_leaves_nothing_drops_the_segment():
    setup = TextFlowSetup(
        llm_text_processor=LLMTextProcessorSpec(transforms=[TransformSpec(name="drop")]),
    )
    trace = await run_textflow(setup, "Hello there.")

    [segment] = trace.segments
    assert segment.outcome == "dropped"
    assert segment.segment_id is None
    assert trace.context_messages == []
    assert trace.provider_requests == []


@pytest.mark.asyncio
async def test_failing_transform_is_reported():
    setup = TextFlowSetup(tts=TTSSpec(transforms=[TransformSpec(name="fail")]))
    trace = await run_textflow(setup, "Hello there.")

    [segment] = trace.segments
    assert segment.outcome == "dropped"
    assert any(e.kind == "error" for e in segment.events)
    assert trace.errors and "transform failed on purpose" in trace.errors[0]
    assert trace.provider_requests == []


@pytest.mark.asyncio
async def test_no_word_timestamps_reports_the_segment_once():
    """Without word events the client gets one completed message and no progress."""
    setup = TextFlowSetup(tts=TTSSpec(word_timestamps=False))
    trace = await run_textflow(setup, "Hello there.")

    [segment] = trace.segments
    assert [m["data"]["spoken_status"] for m in segment.rtvi_messages] == ["new", "completed"]
    assert [w.accumulated for w in segment.words] == [None]
    assert trace.context_messages[0]["content"] == "Hello there."


@pytest.mark.asyncio
async def test_tts_aggregates_without_an_llm_text_processor():
    setup = TextFlowSetup(llm_text_processor=None)
    trace = await run_textflow(setup, "Hello there. How are you?")

    assert [s.produced_by for s in trace.segments] == ["tts", "tts"]
    assert [s.segment_text for s in trace.segments] == ["Hello there.", "How are you?"]
    assert trace.context_messages[0]["content"] == "Hello there. How are you?"


@pytest.mark.asyncio
async def test_pattern_pair_setup_round_trips_through_yaml():
    setup = TextFlowSetup(
        name="tagged",
        llm_text_processor=LLMTextProcessorSpec(
            aggregator=AggregatorSpec(
                kind="pattern_pair",
                patterns=[PatternSpec(type="code", start="<code>", end="</code>")],
            )
        ),
        tts=TTSSpec(skip_text_types=["code"]),
        rtvi=RTVISpec(transforms=[TransformSpec(text_type="code", name="upper")]),
    )
    assert TextFlowSetup.model_validate(yaml.safe_load(setup.to_yaml())) == setup


@pytest.mark.asyncio
async def test_every_preset_runs():
    for preset in presets():
        trace = await run_textflow(preset)
        assert trace.segments, preset.name
        assert not trace.errors, preset.name


def test_chunking():
    text = "Hello there. Bye"
    assert chunk_text(text, ChunkingSpec(style="whole")) == [text]
    assert chunk_text(text, ChunkingSpec(style="words")) == ["Hello ", "there. ", "Bye"]
    assert chunk_text(text, ChunkingSpec(style="characters", size=6)) == [
        "Hello ",
        "there.",
        " Bye",
    ]
    assert chunk_text(text, ChunkingSpec(style="custom", chunks=["Hel", "lo"])) == ["Hel", "lo"]


def test_tokenization():
    assert tokenize("Hi there.", TokenizationSpec(gap_s=0.5)) == [("Hi", 0.0), ("there.", 0.5)]
    assert [w for w, _ in tokenize("Hi there.", TokenizationSpec(style="split_punctuation"))] == [
        "Hi",
        "there",
        ".",
    ]
    assert [w for w, _ in tokenize("Hi there.", TokenizationSpec(drop_words=["Hi"]))] == ["there."]


@pytest.mark.asyncio
async def test_registered_transforms_build_and_run():
    for entry in registered_transforms():
        transform = build_transform(entry.name)
        if entry.name == "fail":
            with pytest.raises(RuntimeError):
                await transform("Hello", "sentence")
        else:
            assert isinstance(await transform("Hello", "sentence"), str)


@pytest.mark.asyncio
async def test_steps_show_the_buffer_filling_and_cutting():
    """Each chunk lands in the aggregator before the cut it causes, and the buffer is shown."""
    trace = await run_textflow(TextFlowSetup.preset("code-helper"), CODE_HELPER_TEXT)

    steps = trace.timeline
    arrivals = [i for i, e in enumerate(steps) if e.label == "LLMTextFrame"]
    appends = [i for i, e in enumerate(steps) if e.kind == "aggregate" and e.label == "append"]
    assert len(arrivals) == len(appends) == len(trace.chunks)
    assert all(a < b for a, b in zip(arrivals, appends))
    assert [steps[i].data["chunk"] for i in appends] == trace.chunks
    assert steps[appends[2]].data["buffer"] == "Your card is "

    first_cut = next(e for e in steps if e.kind == "aggregate" and e.label == "cut")
    assert first_cut.data == {
        "action": "cut",
        "buffer": "<card>",
        "text": "Your card is",
        "text_type": "sentence",
    }
    # The cut comes after the chunk that completed the sentence, and the frame it
    # became reaches the TTS before the next chunk arrives.
    cut_at = steps.index(first_cut)
    handed = next(
        i
        for i, e in enumerate(steps)
        if e.label == "AggregatedTextFrame" and e.destination == "tts"
    )
    assert appends[3] < cut_at < handed < arrivals[4]
    assert steps[handed].data["text"] == "Your card is"

    flush = next(e for e in steps if e.label == "flush")
    assert flush.data["text"] == "to start."


@pytest.mark.asyncio
async def test_steps_place_stage_actions_and_releases_in_order():
    trace = await run_textflow(TextFlowSetup.preset("code-helper"), CODE_HELPER_TEXT)

    steps = trace.timeline
    wrap = next(e for e in steps if e.kind == "stage" and e.label == "transform wrap")
    assert wrap.segment_id == trace.segments[1].segment_id
    assert wrap.seq == trace.segments[1].events[0].seq
    request = next(e for e in steps if e.kind == "provider" and e.segment_id == wrap.segment_id)
    assert steps.index(wrap) < steps.index(request)
    assert request.data["text"] == "<spell>1234-5678-9012-3456</spell>"

    skip = next(e for e in steps if e.kind == "stage" and e.label == "skip")
    assert skip.segment_id == trace.segments[4].segment_id

    # Words reach the transport at once and leave it in timestamp order, on time.
    held = [e for e in steps if e.label == "TTSTextFrame" and e.destination == "transport"]
    released = [e for e in steps if e.label == "TTSTextFrame" and e.destination == "context"]
    assert [e.data["text"] for e in held] == [e.data["text"] for e in released]
    assert all(h.t <= r.t for h, r in zip(held, released))
    assert [r.data["pts_s"] for r in released] == sorted(r.data["pts_s"] for r in released)
    assert released[0].data["context_so_far"] == "Your"
    assert released[-1].data["context_so_far"] == trace.context_messages[0]["content"]

    commit = next(e for e in steps if e.label == "LLMContextAssistantTurnFrame")
    assert commit.source == "context"
    assert sum(e.label == "LLMContextAssistantTurnFrame" for e in steps) == 1
    assert sum(e.label == "BotStartedSpeakingFrame" for e in steps) == 1


@pytest.mark.asyncio
async def test_capture_bot_finds_the_text_path():
    """The bot's own processors come back, with what the trace will not run noted."""
    async with await capture_bot(BOT_PATH) as captured:
        _check_capture(captured)


def _check_capture(captured):
    assert captured.summary.pipeline == [
        "RTVIProcessor",
        "FrameProcessor",
        "LLMUserAggregator",
        "OpenAILLMService",
        "LLMTextProcessor",
        "PassThrough",
        "SilentTTS",
        "FrameProcessor",
        "LLMAssistantAggregator",
    ]
    assert captured.llm_text_processor is not None
    assert captured.tts is not None and type(captured.tts).__name__ == "SilentTTS"
    assert captured.rtvi_params is not None and captured.rtvi_params.skip_text_types == ["code"]
    assert any("PassThrough" in note for note in captured.summary.notes)

    setup = mirror_setup(captured)
    assert setup.bot == str(BOT_PATH.resolve())
    assert setup.name == "textflow_bot"
    assert setup.llm_text_processor is not None
    assert setup.llm_text_processor.aggregator.kind == "pattern_pair"
    assert [p.type for p in setup.llm_text_processor.aggregator.patterns] == ["code"]
    assert [t.name for t in setup.llm_text_processor.transforms] == ["replace_text"]
    assert setup.tts.skip_text_types == ["code"]
    assert [t.name for t in setup.tts.transforms] == ["shout"]
    assert setup.tts.word_timestamps is False
    assert setup.rtvi.enabled and setup.rtvi.skip_text_types == ["code"]


@pytest.mark.asyncio
async def test_bot_setup_runs_the_bots_own_objects():
    """A stand-in run with ``bot`` set uses the bot's aggregator, transforms and RTVI parameters."""
    setup = TextFlowSetup(bot=str(BOT_PATH), tts=TTSSpec(word_timestamps=True, live=False))
    trace = await run_textflow(setup, "Dr. Smith wrote <code>x = 1</code>. Bye.")

    assert trace.bot is not None and "PassThrough" in trace.bot.pipeline
    assert trace.setup.bot == str(BOT_PATH.resolve())
    assert trace.setup.tts.word_timestamps is True
    assert trace.setup.tts.live is False

    first = trace.segments[0]
    assert first.events[0].stage == "llm_text_processor" and first.events[0].name == "replace_text"
    assert first.llm_text == "Doctor Smith wrote "
    assert trace.provider_requests[0]["text"] == "DOCTOR SMITH WROTE"
    assert "Doctor Smith" in trace.context_messages[0]["content"]

    code = next(s for s in trace.segments if s.text_type == "code")
    assert code.outcome == "skipped"
    assert code.rtvi_messages == []


@pytest.mark.asyncio
async def test_live_tts_runs_the_bots_own_service():
    """With a bot the TTS service runs itself by default and its provider is called."""
    setup = TextFlowSetup(bot=str(BOT_PATH))
    trace = await run_textflow(setup, "Dr. Smith wrote <code>x = 1</code>. Bye.")

    assert trace.setup.tts.live is True
    assert trace.bot is not None and trace.bot.tts_service == "SilentTTS"
    assert [r["text"] for r in trace.provider_requests] == ["DOCTOR SMITH WROTE", ".", "BYE."]
    first = trace.segments[0]
    assert first.outcome == "spoken"
    assert [e.name for e in first.events if e.stage == "tts"] == ["shout"]
    # SilentTTS reports no word timestamps, so each segment reaches the context whole.
    assert [w.text for w in first.words] == ["Doctor Smith wrote"]
    assert "Doctor Smith wrote" in trace.context_messages[0]["content"]
    assert trace.errors == []


@pytest.mark.asyncio
async def test_live_tts_needs_a_bot():
    with pytest.raises(TextFlowRunError, match="needs a bot"):
        await run_textflow(TextFlowSetup(tts=TTSSpec(live=True)), "Hi.")


@pytest.mark.asyncio
async def test_capture_holds_the_bot_until_closed():
    """``bot()`` waits at its runner's ``run()`` and returns once the capture is closed."""
    captured = await capture_bot(BOT_PATH)
    hold = captured._hold
    assert hold is not None and hold.task is not None and not hold.task.done()
    assert await captured.close() is None
    assert hold.task.done()
    assert await captured.close() is None


@pytest.mark.asyncio
async def test_capture_reports_a_file_without_a_bot(tmp_path):
    path = tmp_path / "nobot.py"
    path.write_text("x = 1\n")
    with pytest.raises(TextFlowCaptureError, match="no bot"):
        await capture_bot(path)
    with pytest.raises(TextFlowCaptureError, match="does not exist"):
        await capture_bot(tmp_path / "missing.py")
