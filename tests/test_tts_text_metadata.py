#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for carrying text metadata through a TTS service (``text_metadata_keys``)."""

from collections.abc import AsyncGenerator

import pytest

from pipecat.frames.frames import (
    AggregatedTextFrame,
    AggregatedTextProgressFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSTextFrame,
)
from pipecat.services.tts_service import TextAggregationMode, TTSService
from pipecat.tests.utils import run_test
from pipecat.utils.context.aggregated_frame_sequencer import AggregatedFrameSequencer
from pipecat.utils.text.base_text_aggregator import TextType

_FAKE_AUDIO = b"\x00\x01" * 320
_SAMPLE_RATE = 16000


class _WordTimestampTTSService(TTSService):
    """Synchronous TTS that reports each word of its input as a word timestamp."""

    def __init__(self, **kwargs):
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            push_text_frames=False,
            sample_rate=_SAMPLE_RATE,
            **kwargs,
        )

    def can_generate_metrics(self) -> bool:
        return False

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        await self.add_word_timestamps(
            [(w, i * 0.1) for i, w in enumerate(text.split())], context_id=context_id
        )
        yield TTSAudioRawFrame(
            audio=_FAKE_AUDIO, sample_rate=_SAMPLE_RATE, num_channels=1, context_id=context_id
        )


class _PushTextTTSService(TTSService):
    """Synchronous TTS without word timestamps: the base class pushes its text."""

    def __init__(self, **kwargs):
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            push_text_frames=True,
            sample_rate=_SAMPLE_RATE,
            **kwargs,
        )

    def can_generate_metrics(self) -> bool:
        return False

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        yield TTSAudioRawFrame(
            audio=_FAKE_AUDIO, sample_rate=_SAMPLE_RATE, num_channels=1, context_id=context_id
        )


def _tagged(frame: Frame, **metadata) -> Frame:
    frame.metadata.update(metadata)
    return frame


def _response(*chunks: tuple[str, dict]) -> list[Frame]:
    """An LLM response streamed as (text, metadata) chunks."""
    return [
        LLMFullResponseStartFrame(),
        *(_tagged(LLMTextFrame(text), **metadata) for text, metadata in chunks),
        LLMFullResponseEndFrame(),
    ]


def _sentences(frames: list[Frame]) -> list[tuple[str, dict]]:
    return [
        (f.text, f.metadata) for f in frames if type(f) is AggregatedTextFrame and f.will_be_spoken
    ]


def _words(frames: list[Frame]) -> list[tuple[str, dict]]:
    return [(f.text, f.metadata) for f in frames if isinstance(f, TTSTextFrame)]


M1 = {"source": {"message_id": "m1"}}
M2 = {"source": {"message_id": "m2"}}


# ---------------------------------------------------------------------------
# AggregatedFrameSequencer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sequencer_word_and_progress_frames_carry_the_slot_metadata():
    sequencer = AggregatedFrameSequencer()
    source = AggregatedTextFrame("Hello there.", TextType.SENTENCE)
    await sequencer.register_spoken(source, "ctx", source.text, append_to_context=True, metadata=M1)

    frames = sequencer.process_word("Hello", 1, "ctx")
    frames += sequencer.process_word("there.", 2, "ctx")

    assert _words(frames) == [("Hello", M1), ("there.", M1)]
    progress = [f for f in frames if isinstance(f, AggregatedTextProgressFrame)]
    assert len(progress) == 2
    assert all(f.metadata == M1 for f in progress)


@pytest.mark.asyncio
async def test_sequencer_without_metadata_leaves_frames_untouched():
    sequencer = AggregatedFrameSequencer()
    source = AggregatedTextFrame("Hello there.", TextType.SENTENCE)
    source.metadata["source"] = "not-selected"
    await sequencer.register_spoken(source, "ctx", source.text, append_to_context=True)

    frames = sequencer.process_word("Hello", 1, "ctx")
    frames += sequencer.process_word("there.", 2, "ctx")

    assert _words(frames) == [("Hello", {}), ("there.", {})]


@pytest.mark.asyncio
async def test_sequencer_force_completed_text_carries_the_slot_metadata():
    sequencer = AggregatedFrameSequencer()
    source = AggregatedTextFrame("Hello there.", TextType.SENTENCE)
    await sequencer.register_spoken(source, "ctx", source.text, append_to_context=True, metadata=M1)

    frames = sequencer.process_word("Hello", 1, "ctx")
    frames += sequencer.force_complete("ctx", 2)

    assert _words(frames) == [("Hello", M1), ("there.", M1)]


@pytest.mark.asyncio
async def test_sequencer_streaming_ends_the_sentence_where_the_source_changes():
    sequencer = AggregatedFrameSequencer(streaming=True)

    async def token(text: str, metadata: dict) -> list[Frame]:
        frame = AggregatedTextFrame(text, TextType.TOKEN)
        return await sequencer.register_spoken(
            frame, "ctx", text, append_to_context=True, metadata=metadata
        )

    frames = await token("Let me", M1)
    frames += await token(" check", M1)
    frames += await token(" Your order", M2)
    frames += await token(" shipped.", M2)
    frames += await sequencer.finalize("ctx")

    sentences = [(f.text, f.metadata) for f in frames if type(f) is AggregatedTextFrame]
    assert sentences == [("Let me check", M1), (" Your order shipped.", M2)]

    words = []
    for i, word in enumerate(["Let", "me", "check", "Your", "order", "shipped."]):
        words += sequencer.process_word(word, i, "ctx")
    assert _words(words) == [
        ("Let", M1),
        ("me", M1),
        ("check", M1),
        ("Your", M2),
        ("order", M2),
        ("shipped.", M2),
    ]


# ---------------------------------------------------------------------------
# TTSService
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sentences_and_words_carry_their_source_metadata():
    tts = _WordTimestampTTSService(text_metadata_keys=["source"])
    down, _ = await run_test(
        tts,
        frames_to_send=_response(("Hello ", M1), ("there. ", M1), ("How are ", M2), ("you?", M2)),
    )

    assert _sentences(down) == [("Hello there.", M1), ("How are you?", M2)]
    assert _words(down) == [
        ("Hello", M1),
        ("there.", M1),
        ("How", M2),
        ("are", M2),
        ("you?", M2),
    ]


@pytest.mark.asyncio
async def test_a_sentence_never_mixes_two_sources():
    tts = _WordTimestampTTSService(text_metadata_keys=["source"])
    down, _ = await run_test(
        tts, frames_to_send=_response(("One moment", M1), (" Your order shipped.", M2))
    )

    assert _sentences(down) == [("One moment", M1), ("Your order shipped.", M2)]
    assert [m for _, m in _words(down)] == [M1, M1, M2, M2, M2]


@pytest.mark.asyncio
async def test_metadata_is_not_carried_by_default():
    tts = _WordTimestampTTSService()
    down, _ = await run_test(tts, frames_to_send=_response(("Hello ", M1), ("there.", M2)))

    # Without text_metadata_keys, text from both sources is one sentence, as before.
    assert _sentences(down) == [("Hello there.", {})]
    assert all(metadata == {} for _, metadata in _words(down))


@pytest.mark.asyncio
async def test_only_the_selected_keys_are_carried():
    tts = _WordTimestampTTSService(text_metadata_keys=["source"])
    frames = _response(("Hello there.", {**M1, "private": "x"}))
    down, _ = await run_test(tts, frames_to_send=frames)

    assert _sentences(down) == [("Hello there.", M1)]
    assert _words(down) == [("Hello", M1), ("there.", M1)]


@pytest.mark.asyncio
async def test_services_without_word_timestamps_carry_metadata():
    tts = _PushTextTTSService(text_metadata_keys=["source"])
    down, _ = await run_test(
        tts, frames_to_send=_response(("Hello there. ", M1), ("How are you?", M2))
    )

    assert _words(down) == [("Hello there.", M1), ("How are you?", M2)]


@pytest.mark.asyncio
async def test_aggregated_text_frames_from_the_app_carry_metadata():
    tts = _WordTimestampTTSService(text_metadata_keys=["source"])
    source = _tagged(AggregatedTextFrame("Hello there.", TextType.SENTENCE), **M1)
    down, _ = await run_test(tts, frames_to_send=[source])

    assert _words(down) == [("Hello", M1), ("there.", M1)]


@pytest.mark.asyncio
async def test_speak_frames_carry_their_own_metadata():
    tts = _WordTimestampTTSService(text_metadata_keys=["source"])
    down, _ = await run_test(
        tts,
        frames_to_send=[
            LLMFullResponseStartFrame(),
            _tagged(LLMTextFrame("Sure. "), **M1),
            _tagged(TTSSpeakFrame("Please hold."), **M2),
            _tagged(LLMTextFrame("Done."), **M1),
            LLMFullResponseEndFrame(),
        ],
    )

    words = dict(_words(down))
    assert words["Please"] == M2 and words["hold."] == M2
    assert words["Sure."] == M1 and words["Done."] == M1


@pytest.mark.asyncio
async def test_unspoken_llm_text_carries_metadata():
    tts = _WordTimestampTTSService(text_metadata_keys=["source"])
    frames = _response(("Hello there. ", M1), ("How", M2), (" are you?", M2))
    for frame in frames:
        frame.skip_tts = True
    down, _ = await run_test(tts, frames_to_send=frames)

    sentences = [(f.text, f.metadata) for f in down if type(f) is AggregatedTextFrame]
    assert sentences == [("Hello there.", M1), ("How are you?", M2)]


@pytest.mark.asyncio
async def test_token_mode_sentences_and_words_carry_their_source_metadata():
    tts = _WordTimestampTTSService(
        text_aggregation_mode=TextAggregationMode.TOKEN, text_metadata_keys=["source"]
    )
    down, _ = await run_test(
        tts,
        frames_to_send=_response(
            ("Hello", M1), (" there.", M1), (" How", M2), (" are", M2), (" you?", M2)
        ),
    )

    assert [(text.strip(), m) for text, m in _sentences(down)] == [
        ("Hello there.", M1),
        ("How are you?", M2),
    ]
    assert _words(down) == [
        ("Hello", M1),
        ("there.", M1),
        ("How", M2),
        ("are", M2),
        ("you?", M2),
    ]
