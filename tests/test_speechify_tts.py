#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for Speechify TTS speech-mark and Server-Sent Events handling."""

import pytest

from pipecat.services.speechify.tts import (
    _output_format_from_sample_rate,
    _parse_sse_event,
    _SpeechMarkAccumulator,
    language_to_speechify_language,
)
from pipecat.transcriptions.language import Language

TEXT = "Hello world."


def word_times(*chunks, text=TEXT, time_offset=0.0):
    """Feed chunks of speech marks through an accumulator and collect its words."""
    accumulator = _SpeechMarkAccumulator(text, time_offset)
    collected = [word for chunk in chunks for word in accumulator.add(chunk)]
    return collected + accumulator.flush()


def mark(value, start, end, start_time=0, end_time=0):
    """Build a word speech mark."""
    return {
        "type": "word",
        "value": value,
        "start": start,
        "end": end,
        "start_time": start_time,
        "end_time": end_time,
    }


# The speech marks Speechify streams for TEXT.
STREAMING_MARKS = [
    mark("Hello", 0, 5, start_time=0, end_time=469),
    mark("world.", 6, 12, start_time=469, end_time=1152),
]


class TestSpeechMarks:
    """Speech marks become (word, seconds) pairs and an utterance end time."""

    def test_marks_convert_milliseconds_to_seconds(self):
        assert word_times(STREAMING_MARKS) == [("Hello", 0.0), ("world.", 0.469)]

    def test_time_offset_shifts_words(self):
        """The offset sequences words across the utterances of one turn."""
        assert word_times(STREAMING_MARKS, time_offset=2.5) == [
            ("Hello", 2.5),
            ("world.", pytest.approx(2.969)),
        ]

    def test_end_time_tracks_the_last_mark_and_ignores_the_offset(self):
        accumulator = _SpeechMarkAccumulator(TEXT, time_offset=2.5)
        accumulator.add(STREAMING_MARKS)

        assert accumulator.end_time == pytest.approx(1.152)

    def test_non_word_marks_are_skipped(self):
        marks = [
            {"type": "sentence", "start": 0, "end": 12, "start_time": 0, "end_time": 1152},
            STREAMING_MARKS[0],
        ]

        assert word_times(marks) == [("Hello", 0.0)]

    @pytest.mark.parametrize("marks", [None, [], [{}], [{"type": "word", "value": ""}]])
    def test_empty_marks_produce_nothing(self, marks):
        assert word_times(marks) == []


class TestWordRecovery:
    """Words come from the synthesized text, not from the mark's normalized value."""

    def test_normalized_value_is_replaced_by_the_original_spelling(self):
        """Speechify flattens typographic apostrophes; the offsets recover them."""
        assert word_times([mark("you'd", 0, 5)], text="you’d like") == [("you’d", 0.0)]

    @pytest.mark.parametrize(
        "unusable",
        [
            {"value": "world."},  # No offsets at all.
            {"start": 6, "end": 99, "value": "world."},  # End past the text.
            {"start": 6, "end": 6, "value": "world."},  # Empty span.
            {"start": "6", "end": "12", "value": "world."},  # Offsets not integers.
        ],
    )
    def test_unusable_offsets_fall_back_to_the_value(self, unusable):
        assert word_times([{"type": "word", **unusable}]) == [("world.", 0.0)]


class TestSplitWords:
    """Marks that abut in the text are one word, even across chunk boundaries."""

    # Speechify marks "text-to-speech" as five separate marks.
    HYPHENATED = "I like text-to-speech systems."

    def test_abutting_marks_join_into_one_word(self):
        marks = [
            mark("text", 7, 11),
            mark("-", 11, 12),
            mark("to", 12, 14),
            mark("-", 14, 15),
            mark("speech", 15, 21),
            mark("systems.", 22, 30),
        ]

        assert word_times(marks, text=self.HYPHENATED) == [
            ("text-to-speech", 0.0),
            ("systems.", 0.0),
        ]

    def test_a_split_word_is_joined_across_chunks(self):
        """A run of abutting marks can straddle two speech.chunk events."""
        assert word_times(
            [mark("text", 7, 11), mark("-", 11, 12), mark("to", 12, 14), mark("-", 14, 15)],
            [mark("speech", 15, 21), mark("systems.", 22, 30)],
            text=self.HYPHENATED,
        ) == [("text-to-speech", 0.0), ("systems.", 0.0)]

    def test_a_joined_word_keeps_the_first_marks_timestamp(self):
        marks = [mark("text", 7, 11, start_time=500), mark("-", 11, 12, start_time=900)]

        assert word_times(marks, text=self.HYPHENATED) == [("text-", 0.5)]

    def test_a_word_is_released_as_soon_as_the_text_shows_it_cannot_continue(self):
        """Only words the text can still extend are held back for the next chunk."""
        accumulator = _SpeechMarkAccumulator(self.HYPHENATED)

        assert accumulator.add([mark("I", 0, 1), mark("like", 2, 6)]) == [
            ("I", 0.0),
            ("like", 0.0),
        ]

    def test_a_word_left_pending_at_the_end_of_the_stream_is_flushed(self):
        """The text ends mid-run when the trailing mark is not the final character."""
        accumulator = _SpeechMarkAccumulator(self.HYPHENATED)
        accumulator.add([mark("text", 7, 11), mark("-", 11, 12)])

        assert accumulator.flush() == [("text-", 0.0)]
        assert accumulator.flush() == []


class TestSSEParsing:
    """Server-Sent Events blocks decode to an event name and payload."""

    def test_chunk_with_audio_and_marks(self):
        event = _parse_sse_event(
            'event: speech.chunk\ndata: {"audio":"SUQzBAA=","speech_marks":[]}'
        )

        assert event == ("speech.chunk", {"audio": "SUQzBAA=", "speech_marks": []})

    def test_done_event(self):
        event = _parse_sse_event(
            'event: speech.done\ndata: {"billable_characters_count":40,"audio_duration_ms":4350}'
        )

        assert event == (
            "speech.done",
            {"billable_characters_count": 40, "audio_duration_ms": 4350},
        )

    def test_unknown_event_name_is_preserved_for_the_caller_to_ignore(self):
        event = _parse_sse_event("event: speech.something-new\ndata: {}")

        assert event == ("speech.something-new", {})

    def test_multi_line_data_is_joined(self):
        event = _parse_sse_event('event: speech.done\ndata: {"audio_duration_ms":\ndata: 4350}')

        assert event == ("speech.done", {"audio_duration_ms": 4350})

    def test_comment_lines_are_ignored(self):
        event = _parse_sse_event(": keep-alive\nevent: speech.done\ndata: {}")

        assert event == ("speech.done", {})

    @pytest.mark.parametrize(
        "block",
        [
            "event: speech.done",  # No data field.
            'data: {"audio":',  # Truncated JSON.
            "",
        ],
    )
    def test_undecodable_blocks_yield_nothing(self, block):
        assert _parse_sse_event(block) is None


class TestOutputFormat:
    """Sample rates map to Speechify's PCM output formats."""

    @pytest.mark.parametrize("sample_rate", [8000, 16000, 22050, 24000, 44100, 48000])
    def test_supported_sample_rates_pass_through(self, sample_rate):
        assert _output_format_from_sample_rate(sample_rate) == (f"pcm_{sample_rate}", sample_rate)

    def test_unsupported_sample_rate_falls_back(self):
        """Speechify has no pcm_32000, so the caller is told the real synthesis rate."""
        assert _output_format_from_sample_rate(32000) == ("pcm_24000", 24000)


class TestLanguageMapping:
    """Languages resolve to the regional tags Speechify documents."""

    def test_base_languages_resolve_to_supported_regional_tags(self):
        assert language_to_speechify_language(Language.EN) == "en-US"
        assert language_to_speechify_language(Language.PT) == "pt-BR"
        assert language_to_speechify_language(Language.DE) == "de-DE"

    def test_regional_variants_are_preserved(self):
        assert language_to_speechify_language(Language.ES_MX) == "es-MX"
        assert language_to_speechify_language(Language.EN_GB) == "en-GB"

    def test_unmapped_languages_pass_through_as_bcp47_tags(self):
        assert language_to_speechify_language(Language.JA_JP) == "ja-JP"
