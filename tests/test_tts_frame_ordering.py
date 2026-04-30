#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for frame ordering across TTS service types.

Covers three patterns:
- HTTP TTS services (e.g. CartesiaHttpTTSService): yield audio frames synchronously.
- WebSocket TTS services without pause (e.g. CartesiaTTSService): deliver audio via
  append_to_audio_context from a background receive loop, no frame-processing pause.
- WebSocket TTS services with pause (e.g. ElevenLabsTTSService): same delivery
  mechanism, but pause downstream frame processing while audio is in flight.

For all three patterns we verify:
    AggregatedTextFrame → TTSStartedFrame → TTSAudioRawFrame (1+) → TTSStoppedFrame → FooFrame

repeated for each TTSSpeakFrame, with no cross-group contamination.

Also covers LLM response flow with push_text_frames=True (non-word-timestamp TTS):
verifies TTSTextFrame ordering relative to LLMFullResponseEndFrame.

Also covers smart-text / WordCompletionTracker features:
- Skipped frames (skip_aggregator_types) held until preceding spoken slots complete.
- raw_text on AggregatedTextFrame propagated as spans to TTSTextFrames.
- Overflow: a single TTS word straddling two AggregatedTextFrame boundaries produces
  two correctly-attributed TTSTextFrames.
- Force-complete safety net: skipped frames flush even when TTS drops word timestamps.
"""

import asyncio
import unittest
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass

import pytest

from pipecat.frames.frames import (
    AggregatedTextFrame,
    DataFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TextFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.tests.utils import run_test
from pipecat.utils.text.base_text_aggregator import AggregationType

# ---------------------------------------------------------------------------
# Test-only frame
# ---------------------------------------------------------------------------

_FAKE_AUDIO = b"\x00\x01" * 320  # 320 bytes of silence
_SAMPLE_RATE = 16000


@dataclass
class FooFrame(DataFrame):
    """Marker frame used to verify relative ordering against TTS audio frames."""

    label: str = ""


# ---------------------------------------------------------------------------
# Mock TTS services
# ---------------------------------------------------------------------------


class MockHttpTTSService(TTSService):
    """Simulates an HTTP TTS service (e.g. CartesiaHttpTTSService).

    Audio frames are yielded synchronously from run_tts(), so the audio context
    is fully populated before the next downstream frame is processed.
    TTSStoppedFrame is appended by the base class in on_turn_context_completed()
    once it detects _is_yielding_frames_synchronously is True.
    """

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
        yield TTSAudioRawFrame(
            audio=_FAKE_AUDIO,
            sample_rate=_SAMPLE_RATE,
            num_channels=1,
            context_id=context_id,
        )


class MockHttpPushTextTTSService(TTSService):
    """Simulates an HTTP TTS service with push_text_frames=True.

    Used to test that LLMFullResponseEndFrame is emitted after all TTSTextFrames
    when the TTS service generates text frames itself (non-word-timestamp mode).
    """

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
            audio=_FAKE_AUDIO,
            sample_rate=_SAMPLE_RATE,
            num_channels=1,
            context_id=context_id,
        )


class MockWebSocketTTSService(TTSService):
    """Simulates a WebSocket TTS service without frame-processing pause (e.g. CartesiaTTSService).

    run_tts() is an empty async generator (signals async delivery). A background
    task appends audio frames and the TTSStoppedFrame to the audio context after a
    short delay, mimicking real WebSocket receive-loop behaviour.
    pause_frame_processing=False means downstream frames (FooFrame) are NOT held.
    """

    def __init__(self, **kwargs):
        super().__init__(
            push_start_frame=True,
            push_text_frames=False,
            pause_frame_processing=False,
            sample_rate=_SAMPLE_RATE,
            **kwargs,
        )

    def can_generate_metrics(self) -> bool:
        return False

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        async def _deliver_audio():
            await asyncio.sleep(0.01)
            await self.append_to_audio_context(
                context_id,
                TTSAudioRawFrame(
                    audio=_FAKE_AUDIO,
                    sample_rate=_SAMPLE_RATE,
                    num_channels=1,
                    context_id=context_id,
                ),
            )
            await self.append_to_audio_context(context_id, TTSStoppedFrame(context_id=context_id))
            await self.remove_audio_context(context_id)

        self.create_task(_deliver_audio(), name=f"mock_ws_deliver_{context_id}")
        if False:
            yield  # make this an async generator without yielding anything


class MockWebSocketPauseTTSService(TTSService):
    """Simulates a WebSocket TTS service WITH frame-processing pause (e.g. ElevenLabsTTSService).

    Identical to MockWebSocketTTSService except pause_frame_processing=True.
    on_audio_context_completed() resumes downstream processing once the full
    audio context has been pushed, guaranteeing FooFrame arrives after TTSStoppedFrame.
    """

    def __init__(self, **kwargs):
        super().__init__(
            push_start_frame=True,
            push_text_frames=False,
            pause_frame_processing=True,
            sample_rate=_SAMPLE_RATE,
            **kwargs,
        )

    def can_generate_metrics(self) -> bool:
        return False

    async def on_audio_context_completed(self, context_id: str):
        # Resume frame processing after the audio context is fully played out.
        await self._maybe_resume_frame_processing()

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        async def _deliver_audio():
            await asyncio.sleep(0.01)
            await self.append_to_audio_context(
                context_id,
                TTSAudioRawFrame(
                    audio=_FAKE_AUDIO,
                    sample_rate=_SAMPLE_RATE,
                    num_channels=1,
                    context_id=context_id,
                ),
            )
            await self.append_to_audio_context(context_id, TTSStoppedFrame(context_id=context_id))
            await self.remove_audio_context(context_id)

        self.create_task(_deliver_audio(), name=f"mock_ws_pause_deliver_{context_id}")
        if False:
            yield


class _MockWordTimestampHttpTTSService(TTSService):
    """HTTP-style TTS: yields audio synchronously, calls add_word_timestamps first.

    ``word_times`` pins the exact tokens and their timestamps.  When omitted the
    service splits the input text on spaces, assigning 0.1 s gaps.
    """

    def __init__(
        self,
        includes_inter_frame_spaces: bool = False,
        word_times: list[tuple[str, float]] | None = None,
        **kwargs,
    ):
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            push_text_frames=False,
            sample_rate=_SAMPLE_RATE,
            **kwargs,
        )
        self._includes_inter_frame_spaces = includes_inter_frame_spaces
        self._word_times = word_times

    def can_generate_metrics(self) -> bool:
        return False

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        word_times = self._word_times or [(w, i * 0.1) for i, w in enumerate(text.split())]
        await self.add_word_timestamps(
            word_times,
            context_id=context_id,
            includes_inter_frame_spaces=self._includes_inter_frame_spaces,
        )
        yield TTSAudioRawFrame(
            audio=_FAKE_AUDIO,
            sample_rate=_SAMPLE_RATE,
            num_channels=1,
            context_id=context_id,
        )


class _MockWordTimestampWSTTSService(TTSService):
    """WebSocket-style TTS: delivers audio asynchronously via the audio context.

    Word timestamps are enqueued as ``_WordTimestampEntry`` items (audio context
    already exists at call time) and processed by ``_handle_audio_context`` in
    playback order.

    ``word_times`` pins the exact tokens and their timestamps.  When omitted the
    service splits the input text on spaces, assigning 0.1 s gaps.
    """

    def __init__(
        self,
        includes_inter_frame_spaces: bool = False,
        word_times: list[tuple[str, float]] | None = None,
        **kwargs,
    ):
        super().__init__(
            push_start_frame=True,
            push_text_frames=False,
            pause_frame_processing=False,
            sample_rate=_SAMPLE_RATE,
            **kwargs,
        )
        self._includes_inter_frame_spaces = includes_inter_frame_spaces
        self._word_times = word_times

    def can_generate_metrics(self) -> bool:
        return False

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        async def _deliver():
            await asyncio.sleep(0.01)
            word_times = self._word_times or [(w, i * 0.1) for i, w in enumerate(text.split())]
            await self.add_word_timestamps(
                word_times,
                context_id=context_id,
                includes_inter_frame_spaces=self._includes_inter_frame_spaces,
            )
            await self.append_to_audio_context(
                context_id,
                TTSAudioRawFrame(
                    audio=_FAKE_AUDIO,
                    sample_rate=_SAMPLE_RATE,
                    num_channels=1,
                    context_id=context_id,
                ),
            )
            await self.append_to_audio_context(context_id, TTSStoppedFrame(context_id=context_id))
            await self.remove_audio_context(context_id)

        self.create_task(_deliver(), name=f"mock_ws_word_deliver_{context_id}")
        if False:
            yield


# ---------------------------------------------------------------------------
# Assertion helper
# ---------------------------------------------------------------------------


def _assert_group_ordering(
    down_frames: Sequence[Frame],
    expected_groups: list[tuple[str, str]],
) -> None:
    """Assert two (or more) TTS+FooFrame groups are in strict order.

    Args:
        down_frames: All downstream frames received by the test sink.
        expected_groups: List of (tts_text, foo_label) pairs, one per TTSSpeakFrame.
            tts_text is unused in assertions today but included for readability.
    """
    relevant = [
        f
        for f in down_frames
        if isinstance(
            f, (AggregatedTextFrame, TTSStartedFrame, TTSAudioRawFrame, TTSStoppedFrame, FooFrame)
        )
    ]

    # Locate the FooFrames that delimit groups.
    foo_indices = [i for i, f in enumerate(relevant) if isinstance(f, FooFrame)]
    assert len(foo_indices) == len(expected_groups), (
        f"Expected {len(expected_groups)} FooFrames, got {len(foo_indices)}.\n"
        f"Relevant frames: {[type(f).__name__ for f in relevant]}"
    )

    # Build groups: everything up to and including each FooFrame.
    groups: list[list[Frame]] = []
    prev = 0
    for idx in foo_indices:
        groups.append(relevant[prev : idx + 1])
        prev = idx + 1

    for group, (_, foo_label) in zip(groups, expected_groups):
        types = [type(f) for f in group]
        type_names = [t.__name__ for t in types]

        assert AggregatedTextFrame in types, (
            f"Group {foo_label!r}: missing AggregatedTextFrame. Got: {type_names}"
        )
        assert TTSStartedFrame in types, (
            f"Group {foo_label!r}: missing TTSStartedFrame. Got: {type_names}"
        )
        assert TTSAudioRawFrame in types, (
            f"Group {foo_label!r}: missing TTSAudioRawFrame. Got: {type_names}"
        )
        assert TTSStoppedFrame in types, (
            f"Group {foo_label!r}: missing TTSStoppedFrame. Got: {type_names}"
        )

        started_idx = types.index(TTSStartedFrame)
        stopped_idx = types.index(TTSStoppedFrame)
        foo_idx = types.index(FooFrame)

        assert started_idx < stopped_idx, (
            f"Group {foo_label!r}: TTSStartedFrame (pos {started_idx}) must precede "
            f"TTSStoppedFrame (pos {stopped_idx}). Got: {type_names}"
        )
        assert stopped_idx < foo_idx, (
            f"Group {foo_label!r}: TTSStoppedFrame (pos {stopped_idx}) must precede "
            f"FooFrame (pos {foo_idx}). Got: {type_names}"
        )

        # All frames between TTSStartedFrame and TTSStoppedFrame must be audio.
        mid_types = types[started_idx + 1 : stopped_idx]
        for t in mid_types:
            assert t in (TTSAudioRawFrame, TTSTextFrame), (
                f"Group {foo_label!r}: unexpected frame {t.__name__!r} between "
                f"TTSStartedFrame and TTSStoppedFrame. Got: {type_names}"
            )

        # Check the FooFrame label.
        actual_label = group[foo_idx].label
        assert actual_label == foo_label, (
            f"Expected FooFrame(label={foo_label!r}), got label={actual_label!r}"
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

_GROUPS = [("test 1", "1"), ("test 2", "2")]


def _make_frames_no_sleep() -> list[Frame]:
    """Return two TTSSpeakFrame+FooFrame pairs sent back-to-back.

    Only correct for services that pause downstream processing until the audio
    context is fully consumed (pause_frame_processing=True + on_audio_context_completed).
    """
    return [
        TTSSpeakFrame(text="test 1", append_to_context=False),
        FooFrame(label="1"),
        TTSSpeakFrame(text="test 2", append_to_context=False),
        FooFrame(label="2"),
    ]


def _print_frames_received(frames_received) -> None:
    print("FRAMES RECEIVED:")
    for frame in frames_received[0]:
        print(frame.name)


@pytest.mark.asyncio
async def test_http_tts_frame_ordering():
    """HTTP TTS services yield audio synchronously."""
    tts = MockHttpTTSService()
    frames_received = await run_test(tts, frames_to_send=_make_frames_no_sleep())

    # only for debugging
    _print_frames_received(frames_received)

    _assert_group_ordering(frames_received[0], _GROUPS)


@pytest.mark.asyncio
async def test_websocket_tts_no_pause_frame_ordering():
    """WebSocket TTS services without pause_frame_processing."""
    tts = MockWebSocketTTSService()
    frames_received = await run_test(tts, frames_to_send=_make_frames_no_sleep())
    _assert_group_ordering(frames_received[0], _GROUPS)


@pytest.mark.asyncio
async def test_websocket_tts_with_pause_frame_ordering():
    """WebSocket TTS services with pause_frame_processing=True."""
    tts = MockWebSocketPauseTTSService()
    frames_received = await run_test(tts, frames_to_send=_make_frames_no_sleep())
    _assert_group_ordering(frames_received[0], _GROUPS)


@pytest.mark.asyncio
async def test_http_push_text_llm_response_end_after_tts_text():
    """LLMFullResponseEndFrame must arrive after all TTSTextFrames.

    Simulates an LLM response producing multiple sentences through an HTTP TTS
    service with push_text_frames=True.  Each sentence is sent as a separate
    TextFrame terminated by a period so the sentence aggregator flushes it.
    The final sentence is flushed by the LLMFullResponseEndFrame itself.

    Expected downstream ordering:
        LLMFullResponseStartFrame
        ... TTSTextFrame (per sentence) ...
        LLMFullResponseEndFrame          ← must come AFTER all TTSTextFrames
    """
    tts = MockHttpPushTextTTSService()

    # Two sentences: the first ends with a period (triggers aggregator flush),
    # the second does NOT (will be flushed by LLMFullResponseEndFrame).
    frames_to_send = [
        LLMFullResponseStartFrame(),
        TextFrame(text="Hello there. "),
        TextFrame(text="How are you?"),
        LLMFullResponseEndFrame(),
    ]
    frames_received = await run_test(tts, frames_to_send=frames_to_send)
    down = frames_received[0]

    # Collect relevant frame types for ordering check.
    relevant = [
        f
        for f in down
        if isinstance(f, (LLMFullResponseStartFrame, TTSTextFrame, LLMFullResponseEndFrame))
    ]
    type_names = [type(f).__name__ for f in relevant]

    # There should be exactly one LLMFullResponseStartFrame, 2 TTSTextFrames, 1 LLMFullResponseEndFrame.
    tts_text_frames = [f for f in relevant if isinstance(f, TTSTextFrame)]
    end_frames = [f for f in relevant if isinstance(f, LLMFullResponseEndFrame)]
    start_frames = [f for f in relevant if isinstance(f, LLMFullResponseStartFrame)]

    assert len(start_frames) == 1, (
        f"Expected 1 LLMFullResponseStartFrame, got {len(start_frames)}: {type_names}"
    )
    assert len(tts_text_frames) == 2, (
        f"Expected 2 TTSTextFrames, got {len(tts_text_frames)}: {type_names}"
    )
    assert len(end_frames) == 1, (
        f"Expected 1 LLMFullResponseEndFrame, got {len(end_frames)}: {type_names}"
    )

    # The critical check: LLMFullResponseEndFrame must come after ALL TTSTextFrames.
    end_idx = relevant.index(end_frames[0])
    last_tts_text_idx = max(relevant.index(f) for f in tts_text_frames)

    assert last_tts_text_idx < end_idx, (
        f"LLMFullResponseEndFrame (pos {end_idx}) must come after the last "
        f"TTSTextFrame (pos {last_tts_text_idx}). Got: {type_names}"
    )


@pytest.mark.asyncio
async def test_http_word_timestamps_verbatim_tokens():
    """HTTP path: text, PTS order, flag, and text-before-audio are all verified.

    Word timestamps arrive in the audio context queue before the audio frame.
    _handle_audio_context caches them, then flushes when the first audio frame
    arrives (start_word_timestamps), so TTSTextFrames must be emitted before
    the TTSAudioRawFrame in the downstream sequence.
    """
    word_times = [("hello", 0.0), ("world", 0.2)]
    tts = _MockWordTimestampHttpTTSService(
        includes_inter_frame_spaces=True,
        word_times=word_times,
    )
    frames_received = await run_test(
        tts,
        frames_to_send=[TTSSpeakFrame(text="hello world", append_to_context=False)],
    )
    down = frames_received[0]
    tts_text_frames = [f for f in down if isinstance(f, TTSTextFrame)]
    audio_frames = [f for f in down if isinstance(f, TTSAudioRawFrame)]

    assert [f.text for f in tts_text_frames] == ["hello", "world"]
    assert all(f.includes_inter_frame_spaces is True for f in tts_text_frames)

    pts_values = [f.pts for f in tts_text_frames]
    assert pts_values == sorted(pts_values) and len(set(pts_values)) == len(pts_values), (
        f"PTS values must be strictly increasing, got {pts_values}"
    )

    # TTSTextFrames must precede the audio frame (they are flushed from cache
    # at the moment the first audio chunk sets the timestamp baseline).
    last_text_idx = max(down.index(f) for f in tts_text_frames)
    first_audio_idx = down.index(audio_frames[0])
    assert last_text_idx < first_audio_idx, (
        "TTSTextFrames must appear before TTSAudioRawFrame in the downstream sequence"
    )


@pytest.mark.asyncio
async def test_http_word_timestamps_punctuation_tokens():
    """Verbatim punctuation tokens are preserved with flag=True; default flag is False.

    Models the Inworld API scenario: the TTS returns tokens exactly as sent.
    Space placement rule:
      - word-follows-word: space is the leading char of the next word (e.g. " world")
      - word-follows-punctuation: space is the trailing char of the punctuation token
        (e.g. "! "), so the following word token carries no leading space.
    The flag must reach every frame and the text must not be modified.
    Also acts as a regression guard that flag=False is the default.
    """
    verbatim_tokens = [
        ("hello", 0.0),
        (" world", 0.15),
        ("! ", 0.3),
        ("How", 0.45),
        (" are", 0.6),
        (" you", 0.75),
        ("?", 0.9),
    ]
    expected_texts = ["hello", " world", "! ", "How", " are", " you", "?"]

    # With flag=True: all tokens verbatim, all frames carry the flag.
    tts_ifs = _MockWordTimestampHttpTTSService(
        includes_inter_frame_spaces=True,
        word_times=verbatim_tokens,
    )
    frames_ifs = await run_test(
        tts_ifs,
        frames_to_send=[TTSSpeakFrame(text="hello world! How are you?", append_to_context=False)],
    )
    text_frames_ifs = [f for f in frames_ifs[0] if isinstance(f, TTSTextFrame)]
    assert [f.text for f in text_frames_ifs] == expected_texts, (
        "Verbatim tokens must not be modified"
    )
    assert all(f.includes_inter_frame_spaces is True for f in text_frames_ifs)

    # With flag=False (default): same tokens, flag must be False on every frame.
    tts_plain = _MockWordTimestampHttpTTSService(
        word_times=verbatim_tokens,
    )
    frames_plain = await run_test(
        tts_plain,
        frames_to_send=[TTSSpeakFrame(text="hello world! How are you?", append_to_context=False)],
    )
    text_frames_plain = [f for f in frames_plain[0] if isinstance(f, TTSTextFrame)]
    expected_texts = ["hello", "world", "!", "How", "are", "you", "?"]
    assert [f.text for f in text_frames_plain] == expected_texts
    assert all(f.includes_inter_frame_spaces is False for f in text_frames_plain)


@pytest.mark.asyncio
async def test_websocket_word_timestamps_verbatim_tokens():
    """WebSocket path: _WordTimestampEntry carries verbatim text, PTS, and flag.

    Unlike the HTTP path the word timestamps are sent asynchronously from a
    background task.  They arrive before the audio frame and are cached until
    start_word_timestamps() fires, so the same text-before-audio ordering
    property must hold.
    """
    word_times = [("hello", 0.0), ("world", 0.2)]
    tts = _MockWordTimestampWSTTSService(
        includes_inter_frame_spaces=True,
        word_times=word_times,
    )
    frames_received = await run_test(
        tts,
        frames_to_send=[TTSSpeakFrame(text="hello world", append_to_context=False)],
    )
    down = frames_received[0]
    tts_text_frames = [f for f in down if isinstance(f, TTSTextFrame)]
    audio_frames = [f for f in down if isinstance(f, TTSAudioRawFrame)]

    assert [f.text for f in tts_text_frames] == ["hello", "world"]
    assert all(f.includes_inter_frame_spaces is True for f in tts_text_frames)

    pts_values = [f.pts for f in tts_text_frames]
    assert pts_values == sorted(pts_values) and len(set(pts_values)) == len(pts_values), (
        f"PTS values must be strictly increasing, got {pts_values}"
    )

    last_text_idx = max(down.index(f) for f in tts_text_frames)
    first_audio_idx = down.index(audio_frames[0])
    assert last_text_idx < first_audio_idx, (
        "TTSTextFrames must appear before TTSAudioRawFrame in the downstream sequence"
    )


@pytest.mark.asyncio
async def test_websocket_word_timestamps_punctuation_tokens():
    """WebSocket path: verbatim punctuation tokens reach TTSTextFrame unchanged."""
    verbatim_tokens = [
        ("hello", 0.0),
        (" world", 0.15),
        ("! ", 0.3),
        ("How", 0.45),
        (" are", 0.6),
        (" you", 0.75),
        ("?", 0.9),
    ]
    tts = _MockWordTimestampWSTTSService(
        includes_inter_frame_spaces=True,
        word_times=verbatim_tokens,
    )
    frames_received = await run_test(
        tts,
        frames_to_send=[TTSSpeakFrame(text="hello world! How are you?", append_to_context=False)],
    )
    text_frames = [f for f in frames_received[0] if isinstance(f, TTSTextFrame)]
    assert [f.text for f in text_frames] == ["hello", " world", "! ", "How", " are", " you", "?"], (
        "Verbatim tokens must not be modified"
    )
    assert all(f.includes_inter_frame_spaces is True for f in text_frames)


# ---------------------------------------------------------------------------
# Per-call word-timestamp mock (for overflow tests)
# ---------------------------------------------------------------------------


class _MockPerCallWordTimestampHttpTTSService(TTSService):
    """HTTP-style TTS where each run_tts() call consumes its own word-time list.

    Designed for tests that need different word tokens per sentence. The
    ``word_times_per_call`` list is consumed in order; an empty inner list means
    no word-timestamp events are emitted for that call.
    """

    def __init__(
        self,
        word_times_per_call: list[list[tuple[str, float]]],
        **kwargs,
    ):
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            push_text_frames=False,
            sample_rate=_SAMPLE_RATE,
            **kwargs,
        )
        self._word_times_queue = list(word_times_per_call)

    def can_generate_metrics(self) -> bool:
        return False

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        word_times = self._word_times_queue.pop(0) if self._word_times_queue else []
        if word_times:
            await self.add_word_timestamps(word_times, context_id=context_id)
        yield TTSAudioRawFrame(
            audio=_FAKE_AUDIO,
            sample_rate=_SAMPLE_RATE,
            num_channels=1,
            context_id=context_id,
        )


# ---------------------------------------------------------------------------
# Tests: skipped frame ordering (skip_aggregator_types)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_http_skipped_frame_waits_for_spoken_words():
    """Skipped frames are held until the preceding spoken slot's word timestamps
    are all processed, then flushed in order (HTTP / synchronous audio path).

    Sequence sent:
        AggregatedTextFrame("hello world", SENTENCE)  — spoken; yields 2 TTSTextFrames
        AggregatedTextFrame("some code", "code")       — in skip_aggregator_types; must wait

    Expected downstream order:
        TTSTextFrame("hello")
        TTSTextFrame("world")
        AggregatedTextFrame("some code", append_to_context=True)
    """
    tts = _MockWordTimestampHttpTTSService(skip_aggregator_types=["code"])
    frames_received = await run_test(
        tts,
        frames_to_send=[
            AggregatedTextFrame("hello world", AggregationType.SENTENCE),
            AggregatedTextFrame("some code", "code"),
        ],
    )
    down = frames_received[0]

    word_frames = [f for f in down if isinstance(f, TTSTextFrame)]
    skipped = [f for f in down if isinstance(f, AggregatedTextFrame) and f.text == "some code"]

    assert [f.text for f in word_frames] == ["hello", "world"]
    assert len(skipped) == 1
    assert skipped[0].append_to_context is True

    last_word_idx = max(down.index(f) for f in word_frames)
    skipped_idx = down.index(skipped[0])
    assert skipped_idx > last_word_idx, (
        f"Skipped frame (pos {skipped_idx}) must appear after last word frame (pos {last_word_idx})"
    )


@pytest.mark.asyncio
async def test_ws_skipped_frame_waits_for_spoken_words():
    """Same ordering guarantee on the WebSocket / async audio delivery path.

    Because audio is delivered from a background task after asyncio.sleep(), the
    skipped frame arrives at _push_frame_respecting_previous_aggregated_frame
    *before* the spoken slot's word timestamps have been processed, directly
    exercising the hold-and-flush path.
    """
    tts = _MockWordTimestampWSTTSService(skip_aggregator_types=["code"])
    frames_received = await run_test(
        tts,
        frames_to_send=[
            AggregatedTextFrame("hello world", AggregationType.SENTENCE),
            AggregatedTextFrame("some code", "code"),
        ],
    )
    down = frames_received[0]

    word_frames = [f for f in down if isinstance(f, TTSTextFrame)]
    skipped = [f for f in down if isinstance(f, AggregatedTextFrame) and f.text == "some code"]

    assert [f.text for f in word_frames] == ["hello", "world"]
    assert len(skipped) == 1
    assert skipped[0].append_to_context is True

    last_word_idx = max(down.index(f) for f in word_frames)
    skipped_idx = down.index(skipped[0])
    assert skipped_idx > last_word_idx, (
        f"Skipped frame (pos {skipped_idx}) must appear after last word frame (pos {last_word_idx})"
    )


@pytest.mark.asyncio
async def test_skipped_frame_before_spoken_emits_immediately():
    """A skipped frame with no preceding spoken slot is emitted right away.

    Sequence:
        AggregatedTextFrame("some code", "code")       — no spoken slot before it → emits now
        AggregatedTextFrame("hello world", SENTENCE)   — spoken; TTSTextFrames follow

    Expected: AggregatedTextFrame("some code") appears *before* TTSTextFrame("hello").
    """
    tts = _MockWordTimestampHttpTTSService(skip_aggregator_types=["code"])
    frames_received = await run_test(
        tts,
        frames_to_send=[
            AggregatedTextFrame("some code", "code"),
            AggregatedTextFrame("hello world", AggregationType.SENTENCE),
        ],
    )
    down = frames_received[0]

    word_frames = [f for f in down if isinstance(f, TTSTextFrame)]
    skipped = [f for f in down if isinstance(f, AggregatedTextFrame) and f.text == "some code"]

    assert len(skipped) == 1
    assert skipped[0].append_to_context is True
    assert len(word_frames) >= 1

    skipped_idx = down.index(skipped[0])
    first_word_idx = down.index(word_frames[0])
    assert skipped_idx < first_word_idx, (
        f"Skipped frame (pos {skipped_idx}) must appear before first word frame (pos {first_word_idx})"
    )


@pytest.mark.asyncio
async def test_skipped_frame_flushed_when_word_timestamps_incomplete():
    """Force-complete path: skipped frame still emits when the TTS drops word timestamps.

    Only one of the two expected tokens ("hello") is returned. The spoken slot never
    reaches its expected character count through the normal path. When
    on_audio_context_done fires it force-completes any remaining spoken slots and
    flushes the waiting skipped frame.
    """
    tts = _MockWordTimestampHttpTTSService(
        word_times=[("hello", 0.0)],  # "world" is never sent
        skip_aggregator_types=["code"],
    )
    frames_received = await run_test(
        tts,
        frames_to_send=[
            AggregatedTextFrame("hello world", AggregationType.SENTENCE),
            AggregatedTextFrame("some code", "code"),
        ],
    )
    down = frames_received[0]

    skipped = [f for f in down if isinstance(f, AggregatedTextFrame) and f.text == "some code"]
    assert len(skipped) == 1, "Skipped frame must be flushed via force-complete safety net"
    assert skipped[0].append_to_context is True


# ---------------------------------------------------------------------------
# Tests: raw_text propagation through WordCompletionTracker
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_raw_text_propagated_to_tts_text_frames():
    """raw_text on AggregatedTextFrame is split across TTSTextFrames by the tracker.

    The frame carries raw_text="<card>4111 1111</card>" while the TTS-prepared
    text is "4111 1111". The WordCompletionTracker advances a cursor through the
    raw text in step with incoming word tokens, so each TTSTextFrame receives the
    exact raw span it represents.

    Expected (trailing whitespace stripped because includes_inter_frame_spaces=False):
        TTSTextFrame("4111").raw_text == "<card>4111"
        TTSTextFrame("1111").raw_text == "1111</card>"
    """
    tts = _MockWordTimestampHttpTTSService()
    frames_received = await run_test(
        tts,
        frames_to_send=[
            AggregatedTextFrame(
                "4111 1111", AggregationType.SENTENCE, raw_text="<card>4111 1111</card>"
            )
        ],
    )
    word_frames = [f for f in frames_received[0] if isinstance(f, TTSTextFrame)]

    assert [f.text for f in word_frames] == ["4111", "1111"]
    # get_raw_consumed() strips trailing whitespace when includes_inter_frame_spaces=False
    assert word_frames[0].raw_text == "<card>4111"
    assert word_frames[1].raw_text == "1111</card>"


# ---------------------------------------------------------------------------
# Tests: overflow — TTS word spanning two AggregatedTextFrame boundaries
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_overflow_word_spanning_two_aggregated_frames():
    """A single TTS token straddling two AggregatedTextFrame boundaries produces
    two correctly-attributed TTSTextFrames.

    Setup:
        Frame 1: AggregatedTextFrame("abc", SENTENCE)
        Frame 2: AggregatedTextFrame("def", SENTENCE)

    The TTS for frame 1 returns the single token "abcdef", which overshoots
    frame 1 by three characters. _emit_overflow_word splits it:
        TTSTextFrame("abc")  — frame 1's portion (context_id = ctx1)
        TTSTextFrame("def")  — overflow attributed to frame 2 (context_id = ctx2)

    Frame 2 receives no word-timestamp events because the overflow already
    consumed its expected text.
    """
    tts = _MockPerCallWordTimestampHttpTTSService(
        word_times_per_call=[
            [("abcdef", 0.0)],  # frame 1: single token spanning both frames
            [],  # frame 2: no word timestamps (overflow already covered it)
        ]
    )
    frames_received = await run_test(
        tts,
        frames_to_send=[
            AggregatedTextFrame("abc", AggregationType.SENTENCE),
            AggregatedTextFrame("def", AggregationType.SENTENCE),
        ],
    )
    word_frames = [f for f in frames_received[0] if isinstance(f, TTSTextFrame)]

    assert [f.text for f in word_frames] == ["abc", "def"], (
        f"Expected ['abc', 'def'] but got {[f.text for f in word_frames]}"
    )
    assert word_frames[0].context_id != word_frames[1].context_id, (
        "Overflow TTSTextFrame must carry frame 2's context_id, not frame 1's"
    )


# ---------------------------------------------------------------------------
# Per-call word-timestamp mock for WebSocket path (for force-complete tests)
# ---------------------------------------------------------------------------


class _MockPerCallWordTimestampWSTTSService(TTSService):
    """WebSocket-style TTS where each run_tts() call consumes its own word-time list.

    Mirrors _MockPerCallWordTimestampHttpTTSService but uses the async audio-context
    delivery pattern so it exercises _handle_audio_context (the WebSocket path).
    An empty inner list means no word-timestamp events are emitted for that call.
    """

    def __init__(
        self,
        word_times_per_call: list[list[tuple[str, float]]],
        **kwargs,
    ):
        super().__init__(
            push_start_frame=True,
            push_text_frames=False,
            pause_frame_processing=False,
            sample_rate=_SAMPLE_RATE,
            **kwargs,
        )
        self._word_times_queue = list(word_times_per_call)

    def can_generate_metrics(self) -> bool:
        return False

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        word_times = self._word_times_queue.pop(0) if self._word_times_queue else []

        async def _deliver():
            await asyncio.sleep(0.01)
            if word_times:
                await self.add_word_timestamps(word_times, context_id=context_id)
            await self.append_to_audio_context(
                context_id,
                TTSAudioRawFrame(
                    audio=_FAKE_AUDIO,
                    sample_rate=_SAMPLE_RATE,
                    num_channels=1,
                    context_id=context_id,
                ),
            )
            await self.append_to_audio_context(context_id, TTSStoppedFrame(context_id=context_id))
            await self.remove_audio_context(context_id)

        self.create_task(_deliver(), name=f"mock_ws_per_call_deliver_{context_id}")
        if False:
            yield


# ---------------------------------------------------------------------------
# Tests: _force_complete_spoken_slots — TTSTextFrame emission for dropped timestamps
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_http_force_complete_partial_timestamps_emits_remaining_text():
    """_force_complete_spoken_slots emits a TTSTextFrame for the unspoken word suffix.

    Only the first token ("hello") is delivered as a word-timestamp event; "world"
    is never sent.  When the audio context ends _force_complete_spoken_slots fires,
    reads get_remaining_text() from the tracker, and emits TTSTextFrame("world").

    Expected TTSTextFrames in order: ["hello", "world"].
    """
    tts = _MockWordTimestampHttpTTSService(word_times=[("hello", 0.0)])
    frames_received = await run_test(
        tts,
        frames_to_send=[TTSSpeakFrame(text="hello world", append_to_context=False)],
    )
    word_frames = [f for f in frames_received[0] if isinstance(f, TTSTextFrame)]

    assert [f.text for f in word_frames] == ["hello", "world"], (
        f"Expected ['hello', 'world'] but got {[f.text for f in word_frames]}"
    )


@pytest.mark.asyncio
async def test_http_force_complete_no_timestamps_emits_full_text():
    """_force_complete_spoken_slots emits the full text when no word timestamps arrive.

    No word-timestamp events are sent for "hello world".  The slot remains incomplete
    when the audio context ends; force-complete reads the full remaining text from the
    tracker and emits TTSTextFrame("hello world").
    """
    tts = _MockPerCallWordTimestampHttpTTSService(word_times_per_call=[[]])
    frames_received = await run_test(
        tts,
        frames_to_send=[TTSSpeakFrame(text="hello world", append_to_context=False)],
    )
    word_frames = [f for f in frames_received[0] if isinstance(f, TTSTextFrame)]

    assert len(word_frames) == 1, (
        f"Expected exactly 1 TTSTextFrame, got {len(word_frames)}: {[f.text for f in word_frames]}"
    )
    assert word_frames[0].text == "hello world", (
        f"Expected TTSTextFrame('hello world'), got {word_frames[0].text!r}"
    )


@pytest.mark.asyncio
async def test_http_force_complete_raw_text_propagated():
    """force-complete carries the correct raw_text span on the emitted TTSTextFrame.

    AggregatedTextFrame carries raw_text="<card>4111 1111</card>".  Only "4111" arrives
    as a word-timestamp; "1111" is force-completed.

    Expected:
        TTSTextFrame("4111").raw_text == "<card>4111"    — from normal word path
        TTSTextFrame("1111").raw_text == "1111</card>"   — from force-complete path
    """
    tts = _MockPerCallWordTimestampHttpTTSService(word_times_per_call=[[("4111", 0.0)]])
    frames_received = await run_test(
        tts,
        frames_to_send=[
            AggregatedTextFrame(
                "4111 1111", AggregationType.SENTENCE, raw_text="<card>4111 1111</card>"
            )
        ],
    )
    word_frames = [f for f in frames_received[0] if isinstance(f, TTSTextFrame)]

    assert [f.text for f in word_frames] == ["4111", "1111"], (
        f"Expected ['4111', '1111'] but got {[f.text for f in word_frames]}"
    )
    assert word_frames[0].raw_text == "<card>4111", (
        f"Expected raw_text '<card>4111' on first frame, got {word_frames[0].raw_text!r}"
    )
    assert word_frames[1].raw_text == "1111</card>", (
        f"Expected raw_text '1111</card>' on force-complete frame, got {word_frames[1].raw_text!r}"
    )


@pytest.mark.asyncio
async def test_ws_force_complete_partial_timestamps_emits_remaining_text():
    """WebSocket path: _force_complete_spoken_slots emits TTSTextFrame for dropped token.

    Mirrors test_http_force_complete_partial_timestamps_emits_remaining_text on the
    async audio delivery path to confirm force-complete fires correctly from
    _handle_audio_context when TTSStoppedFrame arrives before all word timestamps.
    """
    tts = _MockWordTimestampWSTTSService(word_times=[("hello", 0.0)])
    frames_received = await run_test(
        tts,
        frames_to_send=[TTSSpeakFrame(text="hello world", append_to_context=False)],
    )
    word_frames = [f for f in frames_received[0] if isinstance(f, TTSTextFrame)]

    assert [f.text for f in word_frames] == ["hello", "world"], (
        f"Expected ['hello', 'world'] but got {[f.text for f in word_frames]}"
    )


@pytest.mark.asyncio
async def test_ws_force_complete_no_timestamps_emits_full_text():
    """WebSocket path: full text emitted as single TTSTextFrame when no timestamps arrive."""
    tts = _MockPerCallWordTimestampWSTTSService(word_times_per_call=[[]])
    frames_received = await run_test(
        tts,
        frames_to_send=[TTSSpeakFrame(text="hello world", append_to_context=False)],
    )
    word_frames = [f for f in frames_received[0] if isinstance(f, TTSTextFrame)]

    assert len(word_frames) == 1, (
        f"Expected exactly 1 TTSTextFrame, got {len(word_frames)}: {[f.text for f in word_frames]}"
    )
    assert word_frames[0].text == "hello world", (
        f"Expected TTSTextFrame('hello world'), got {word_frames[0].text!r}"
    )


if __name__ == "__main__":
    unittest.main()
