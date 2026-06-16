#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for AggregatedFrameSequencer.

All methods on the sequencer are synchronous and return lists of frames,
so no async machinery is needed here.

Test groups:
- register_skipped: immediate flush vs. blocked by a preceding spoken slot
- register_spoken / complete_spoken_slot: push_text_frames=True path
- flush: pts propagation, transport_destination, stops at incomplete spoken slot
- process_word: normal, completing, passthrough, raw_text propagation
- process_word overflow: single token spanning two slot boundaries
- process_word force-complete via belongs_here failure
- force_complete: remaining text emission, raw_text, corrupt raw discard, slot ordering
- clear: resets all state
"""

import unittest

from pipecat.frames.frames import (
    AggregatedTextFrame,
    AggregatedTextProgressFrame,
    AggregationType,
    TTSTextFrame,
)
from pipecat.utils.context.aggregated_frame_sequencer import AggregatedFrameSequencer
from pipecat.utils.context.word_completion_tracker import WordCompletionTracker
from pipecat.utils.string import TextPartForConcatenation, concatenate_aggregated_text

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seq() -> AggregatedFrameSequencer:
    return AggregatedFrameSequencer(name="test")


def _spoken_frame(text: str) -> AggregatedTextFrame:
    return AggregatedTextFrame(text, AggregationType.SENTENCE)


def _skipped_frame(text: str) -> AggregatedTextFrame:
    return AggregatedTextFrame(text, "code")


def _tracker(tts_text: str, llm_text: str | None = None) -> WordCompletionTracker:
    return WordCompletionTracker(tts_text, llm_text=llm_text)


# ---------------------------------------------------------------------------
# register_skipped
# ---------------------------------------------------------------------------


class TestRegisterSkipped(unittest.TestCase):
    def test_emits_immediately_with_empty_queue(self):
        seq = _seq()
        frame = _skipped_frame("code block")
        result = seq.register_skipped(frame, "ctx1", None)
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], frame)

    def test_sets_append_to_context_true(self):
        seq = _seq()
        frame = _skipped_frame("code")
        seq.register_skipped(frame, "ctx1", None)
        self.assertTrue(frame.append_to_context)

    def test_sets_context_id_on_frame(self):
        seq = _seq()
        frame = _skipped_frame("code")
        seq.register_skipped(frame, "ctx42", None)
        self.assertEqual(frame.context_id, "ctx42")

    def test_sets_transport_destination(self):
        seq = _seq()
        frame = _skipped_frame("code")
        result = seq.register_skipped(frame, "ctx1", "dest-A")
        self.assertEqual(result[0].transport_destination, "dest-A")

    def test_blocked_by_incomplete_spoken_slot(self):
        seq = _seq()
        seq.register_spoken(_spoken_frame("hello world"), "ctx1", _tracker("hello world"), True)
        result = seq.register_skipped(_skipped_frame("code"), "ctx2", None)
        self.assertEqual(result, [])

    def test_emits_immediately_after_already_complete_spoken_slot(self):
        seq = _seq()
        seq.register_spoken(_spoken_frame("hi"), "ctx1", tracker=None, append_to_context=True)
        seq.complete_spoken_slot()
        result = seq.register_skipped(_skipped_frame("code"), "ctx2", None)
        self.assertEqual(len(result), 1)

    def test_multiple_skipped_before_any_spoken_all_emit(self):
        seq = _seq()
        r1 = seq.register_skipped(_skipped_frame("code1"), "ctx1", None)
        r2 = seq.register_skipped(_skipped_frame("code2"), "ctx2", None)
        self.assertEqual(len(r1), 1)
        self.assertEqual(len(r2), 1)


# ---------------------------------------------------------------------------
# register_spoken / complete_spoken_slot  (push_text_frames=True path)
# ---------------------------------------------------------------------------


class TestCompleteSpokenSlot(unittest.TestCase):
    def test_noop_with_empty_queue(self):
        seq = _seq()
        self.assertEqual(seq.complete_spoken_slot(), [])

    def test_marks_slot_complete_and_flushes_skipped(self):
        seq = _seq()
        seq.register_spoken(_spoken_frame("hello"), "ctx1", tracker=None, append_to_context=True)
        skipped = _skipped_frame("code")
        seq.register_skipped(skipped, "ctx2", None)  # blocked

        result = seq.complete_spoken_slot()
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], skipped)
        self.assertTrue(skipped.append_to_context)

    def test_only_first_pending_slot_is_marked(self):
        seq = _seq()
        seq.register_spoken(_spoken_frame("one"), "ctx1", tracker=None, append_to_context=True)
        seq.register_spoken(_spoken_frame("two"), "ctx2", tracker=None, append_to_context=True)
        skipped = _skipped_frame("code")
        seq.register_skipped(skipped, "ctx3", None)

        # ctx2 still blocks the skipped frame
        result = seq.complete_spoken_slot()
        self.assertEqual(result, [])

    def test_skipped_flushes_after_all_preceding_spoken_complete(self):
        seq = _seq()
        seq.register_spoken(_spoken_frame("one"), "ctx1", tracker=None, append_to_context=True)
        seq.register_spoken(_spoken_frame("two"), "ctx2", tracker=None, append_to_context=True)
        skipped = _skipped_frame("code")
        seq.register_skipped(skipped, "ctx3", None)

        seq.complete_spoken_slot()  # completes ctx1
        result = seq.complete_spoken_slot()  # completes ctx2 → flush skipped
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], skipped)


# ---------------------------------------------------------------------------
# flush
# ---------------------------------------------------------------------------


class TestFlush(unittest.TestCase):
    def test_empty_queue_returns_empty(self):
        self.assertEqual(_seq().flush(), [])

    def test_stops_at_incomplete_spoken_slot(self):
        seq = _seq()
        seq.register_spoken(_spoken_frame("hello"), "ctx1", tracker=None, append_to_context=True)
        seq.register_skipped(_skipped_frame("code"), "ctx2", None)
        self.assertEqual(seq.flush(), [])

    def test_last_word_pts_assigned_to_skipped_frame(self):
        seq = _seq()
        seq.register_spoken(_spoken_frame("hello"), "ctx1", _tracker("hello"), True)
        skipped = _skipped_frame("code")
        seq.register_skipped(skipped, "ctx2", None)

        # process_word("hello") completes the spoken slot and calls flush(last_word_pts=77)
        result = seq.process_word("hello", pts=77, context_id="ctx1")
        flushed = [f for f in result if isinstance(f, AggregatedTextFrame) and f.text == "code"]
        self.assertEqual(len(flushed), 1)
        self.assertEqual(flushed[0].pts, 77)

    def test_complete_spoken_slots_are_swept(self):
        seq = _seq()
        seq.register_spoken(_spoken_frame("hello"), "ctx1", tracker=None, append_to_context=True)
        seq.complete_spoken_slot()
        # Queue should be empty after sweeping the complete spoken slot
        self.assertEqual(seq._slots, [])


# ---------------------------------------------------------------------------
# process_word — basic
# ---------------------------------------------------------------------------


class TestProcessWordBasic(unittest.TestCase):
    def _seq_with_spoken(self, text: str, ctx: str = "ctx1", append: bool = True):
        seq = _seq()
        seq.register_spoken(_spoken_frame(text), ctx, _tracker(text), append)
        return seq

    def test_returns_tts_text_frame(self):
        seq = self._seq_with_spoken("hello")
        result = seq.process_word("hello", pts=100, context_id="ctx1")
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], TTSTextFrame)
        self.assertIsInstance(result[1], AggregatedTextProgressFrame)

    def test_frame_text_and_pts(self):
        seq = self._seq_with_spoken("hello")
        result = seq.process_word("hello", pts=100, context_id="ctx1")
        self.assertEqual(result[0].text, "hello")
        self.assertEqual(result[0].pts, 100)

    def test_frame_context_id(self):
        seq = self._seq_with_spoken("hello", ctx="ctx99")
        result = seq.process_word("hello", pts=1, context_id="ctx99")
        self.assertEqual(result[0].context_id, "ctx99")

    def test_append_to_context_true(self):
        seq = self._seq_with_spoken("hello", append=True)
        result = seq.process_word("hello", pts=1, context_id="ctx1")
        self.assertTrue(result[0].append_to_context)

    def test_append_to_context_false(self):
        seq = self._seq_with_spoken("hello", append=False)
        result = seq.process_word("hello", pts=1, context_id="ctx1")
        self.assertFalse(result[0].append_to_context)

    def test_non_completing_word_does_not_flush_skipped(self):
        seq = self._seq_with_spoken("hello world")
        seq.register_skipped(_skipped_frame("code"), "ctx2", None)
        result = seq.process_word("hello", pts=10, context_id="ctx1")
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], TTSTextFrame)
        self.assertIsInstance(result[1], AggregatedTextProgressFrame)

    def test_completing_word_flushes_blocked_skipped_frame(self):
        seq = self._seq_with_spoken("hello")
        skipped = _skipped_frame("code")
        seq.register_skipped(skipped, "ctx2", None)
        result = seq.process_word("hello", pts=50, context_id="ctx1")
        self.assertEqual(len(result), 3)
        self.assertIsInstance(result[0], TTSTextFrame)
        self.assertIsInstance(result[1], AggregatedTextProgressFrame)
        self.assertIs(result[2], skipped)

    def test_last_of_multiple_words_flushes_skipped(self):
        seq = self._seq_with_spoken("hello world")
        skipped = _skipped_frame("code")
        seq.register_skipped(skipped, "ctx2", None)
        seq.process_word("hello", pts=10, context_id="ctx1")
        result = seq.process_word("world", pts=20, context_id="ctx1")
        self.assertTrue(any(f is skipped for f in result))

    def test_none_context_emits_passthrough(self):
        # Services without audio contexts pass context_id=None and rely on the
        # passthrough path even when no slot is registered.
        seq = _seq()
        result = seq.process_word("hello", pts=1, context_id=None)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], TTSTextFrame)
        self.assertEqual(result[0].text, "hello")
        self.assertIsNone(result[0].context_id)

    def test_none_context_passthrough_appends_to_context(self):
        seq = _seq()
        result = seq.process_word("hello", pts=1, context_id=None)
        self.assertTrue(result[0].append_to_context)

    def test_unknown_context_is_dropped(self):
        # A real (non-None) context_id that was never registered is stale and must
        # be dropped rather than emitted into the current turn's transcript.
        seq = _seq()
        result = seq.process_word("hello", pts=1, context_id="ctx-unknown")
        self.assertEqual(result, [])

    def test_unrecognised_word_emits_passthrough(self):
        seq = _seq()
        seq.register_spoken(_spoken_frame("hello world"), "ctx1", _tracker("hello world"), True)
        # "zzz" doesn't belong to "hello world" and there is no next slot
        result = seq.process_word("zzz", pts=5, context_id="ctx1")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, "zzz")


# ---------------------------------------------------------------------------
# process_word — raw_text propagation
# ---------------------------------------------------------------------------


class TestProcessWordRawText(unittest.TestCase):
    def test_raw_text_split_across_word_frames(self):
        seq = _seq()
        seq.register_spoken(
            _spoken_frame("4111 1111"),
            "ctx1",
            WordCompletionTracker("4111 1111", llm_text="<card>4111 1111</card>"),
            append_to_context=True,
        )
        r1 = seq.process_word("4111", pts=10, context_id="ctx1")
        r2 = seq.process_word("1111", pts=20, context_id="ctx1")
        self.assertEqual(r1[0].raw_text, "<card>4111")
        last_word_frames = [f for f in r2 if isinstance(f, TTSTextFrame)]
        self.assertEqual(last_word_frames[0].raw_text, "1111</card>")

    def test_raw_text_none_when_no_llm_text(self):
        seq = _seq()
        seq.register_spoken(_spoken_frame("hello"), "ctx1", _tracker("hello"), True)
        result = seq.process_word("hello", pts=1, context_id="ctx1")
        self.assertIsNone(result[0].raw_text)


# ---------------------------------------------------------------------------
# process_word — overflow (single token spanning two slots)
# ---------------------------------------------------------------------------


class TestProcessWordOverflow(unittest.TestCase):
    def test_overflow_produces_two_tts_text_frames(self):
        seq = _seq()
        seq.register_spoken(_spoken_frame("abc"), "ctx1", _tracker("abc"), True)
        seq.register_spoken(_spoken_frame("def"), "ctx2", _tracker("def"), True)

        result = seq.process_word("abcdef", pts=100, context_id="ctx1")
        word_frames = [f for f in result if isinstance(f, TTSTextFrame)]
        self.assertEqual(len(word_frames), 2)
        self.assertEqual(word_frames[0].text, "abc")
        self.assertEqual(word_frames[1].text, "def")

    def test_overflow_assigns_correct_context_ids(self):
        seq = _seq()
        seq.register_spoken(_spoken_frame("abc"), "ctx1", _tracker("abc"), True)
        seq.register_spoken(_spoken_frame("def"), "ctx2", _tracker("def"), True)

        result = seq.process_word("abcdef", pts=100, context_id="ctx1")
        word_frames = [f for f in result if isinstance(f, TTSTextFrame)]
        self.assertEqual(word_frames[0].context_id, "ctx1")
        self.assertEqual(word_frames[1].context_id, "ctx2")

    def test_overflow_completing_next_slot_flushes_skipped(self):
        seq = _seq()
        seq.register_spoken(_spoken_frame("abc"), "ctx1", _tracker("abc"), True)
        seq.register_spoken(_spoken_frame("def"), "ctx2", _tracker("def"), True)
        skipped = _skipped_frame("code")
        seq.register_skipped(skipped, "ctx3", None)  # blocked behind ctx2

        result = seq.process_word("abcdef", pts=100, context_id="ctx1")
        self.assertTrue(any(f is skipped for f in result))

    def test_overflow_not_completing_next_slot_does_not_flush_skipped(self):
        seq = _seq()
        seq.register_spoken(_spoken_frame("abc"), "ctx1", _tracker("abc"), True)
        seq.register_spoken(_spoken_frame("def ghi"), "ctx2", _tracker("def ghi"), True)
        skipped = _skipped_frame("code")
        seq.register_skipped(skipped, "ctx3", None)

        # "abcdef" overflows: "def" goes to ctx2, but ctx2 still expects " ghi"
        result = seq.process_word("abcdef", pts=100, context_id="ctx1")
        self.assertFalse(any(f is skipped for f in result))


# ---------------------------------------------------------------------------
# process_word — force-complete via word_belongs_here failure
# ---------------------------------------------------------------------------


class TestProcessWordForcesComplete(unittest.TestCase):
    def test_word_for_next_slot_force_completes_current(self):
        """When a word belongs to the next slot but not the current, the current
        slot is force-completed and the word is routed to the next slot."""
        seq = _seq()
        seq.register_spoken(_spoken_frame("hello"), "ctx1", _tracker("hello"), True)
        seq.register_spoken(_spoken_frame("world"), "ctx2", _tracker("world"), True)

        # "world" doesn't belong to ctx1 but belongs to ctx2
        result = seq.process_word("world", pts=50, context_id="ctx2")
        word_frames = [f for f in result if isinstance(f, TTSTextFrame)]
        texts = {f.text for f in word_frames}
        self.assertIn("world", texts)

    def test_force_complete_then_overflow_flushes_skipped(self):
        seq = _seq()
        seq.register_spoken(_spoken_frame("hello"), "ctx1", _tracker("hello"), True)
        seq.register_spoken(_spoken_frame("world"), "ctx2", _tracker("world"), True)
        skipped = _skipped_frame("code")
        seq.register_skipped(skipped, "ctx3", None)

        # "world" force-completes ctx1 and completes ctx2 via overflow
        result = seq.process_word("world", pts=50, context_id="ctx2")
        self.assertTrue(any(f is skipped for f in result))


# ---------------------------------------------------------------------------
# force_complete
# ---------------------------------------------------------------------------


class TestForceComplete(unittest.TestCase):
    def test_emits_remaining_text_when_word_dropped(self):
        seq = _seq()
        seq.register_spoken(_spoken_frame("hello world"), "ctx1", _tracker("hello world"), True)
        seq.process_word("hello", pts=10, context_id="ctx1")  # "world" never arrives

        result = seq.force_complete(last_word_pts=50)
        tts_frames = [f for f in result if isinstance(f, TTSTextFrame)]
        self.assertEqual(len(tts_frames), 1)
        self.assertEqual(tts_frames[0].text, "world")
        self.assertEqual(tts_frames[0].pts, 50)

    def test_emits_full_text_when_no_words_arrived(self):
        seq = _seq()
        seq.register_spoken(_spoken_frame("hello world"), "ctx1", _tracker("hello world"), True)

        result = seq.force_complete(last_word_pts=0)
        tts_frames = [f for f in result if isinstance(f, TTSTextFrame)]
        self.assertEqual(len(tts_frames), 1)
        self.assertEqual(tts_frames[0].text, "hello world")

    def test_already_complete_slot_emits_nothing(self):
        seq = _seq()
        seq.register_spoken(_spoken_frame("hi"), "ctx1", _tracker("hi"), True)
        seq.process_word("hi", pts=5, context_id="ctx1")  # completes normally

        result = seq.force_complete(last_word_pts=10)
        self.assertEqual(result, [])

    def test_flushes_skipped_frames_after_completing(self):
        seq = _seq()
        seq.register_spoken(_spoken_frame("hello"), "ctx1", _tracker("hello"), True)
        skipped = _skipped_frame("code")
        seq.register_skipped(skipped, "ctx2", None)

        result = seq.force_complete(last_word_pts=20)
        self.assertTrue(any(f is skipped for f in result))
        self.assertTrue(skipped.append_to_context)

    def test_propagates_raw_text(self):
        seq = _seq()
        seq.register_spoken(
            _spoken_frame("4111 1111"),
            "ctx1",
            WordCompletionTracker("4111 1111", llm_text="<card>4111 1111</card>"),
            append_to_context=True,
        )
        seq.process_word("4111", pts=10, context_id="ctx1")  # "1111" never arrives

        result = seq.force_complete(last_word_pts=20)
        tts_frames = [f for f in result if isinstance(f, TTSTextFrame)]
        self.assertEqual(tts_frames[0].text, "1111")
        self.assertEqual(tts_frames[0].raw_text, "1111</card>")

    def test_discards_corrupt_raw_remaining(self):
        """raw_remaining is discarded when it does not contain remaining_text."""
        seq = _seq()
        # "abc" normalized ≠ "xyz" normalized — any remaining won't be in raw_remaining
        seq.register_spoken(
            _spoken_frame("abc"),
            "ctx1",
            WordCompletionTracker("abc", llm_text="xyz"),
            append_to_context=True,
        )
        result = seq.force_complete(last_word_pts=0)
        tts_frames = [f for f in result if isinstance(f, TTSTextFrame)]
        self.assertEqual(len(tts_frames), 1)
        self.assertEqual(tts_frames[0].text, "abc")
        self.assertIsNone(tts_frames[0].raw_text)  # discarded due to corruption

    def test_slot_without_tracker_just_marks_complete_and_flushes(self):
        seq = _seq()
        seq.register_spoken(_spoken_frame("hello"), "ctx1", tracker=None, append_to_context=True)
        skipped = _skipped_frame("code")
        seq.register_skipped(skipped, "ctx2", None)

        result = seq.force_complete(last_word_pts=0)
        tts_frames = [f for f in result if isinstance(f, TTSTextFrame)]
        self.assertEqual(tts_frames, [])  # no tracker → no word frame
        self.assertTrue(any(f is skipped for f in result))

    def test_multiple_incomplete_slots_all_emitted(self):
        seq = _seq()
        seq.register_spoken(_spoken_frame("hello"), "ctx1", _tracker("hello"), True)
        seq.register_spoken(_spoken_frame("world"), "ctx2", _tracker("world"), True)

        result = seq.force_complete(last_word_pts=0)
        tts_frames = [f for f in result if isinstance(f, TTSTextFrame)]
        texts = {f.text for f in tts_frames}
        self.assertIn("hello", texts)
        self.assertIn("world", texts)


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


class TestClear(unittest.TestCase):
    def test_clears_slots(self):
        seq = _seq()
        seq.register_spoken(_spoken_frame("hello"), "ctx1", _tracker("hello"), True)
        seq.register_skipped(_skipped_frame("code"), "ctx2", None)
        seq.clear()
        self.assertEqual(seq._slots, [])

    def test_clears_context_map(self):
        seq = _seq()
        seq.register_spoken(_spoken_frame("hello"), "ctx1", _tracker("hello"), True)
        seq.clear()
        self.assertEqual(seq._context_append_to_context, {})

    def test_after_clear_skipped_emits_immediately(self):
        seq = _seq()
        seq.register_spoken(_spoken_frame("hello"), "ctx1", _tracker("hello"), True)
        seq.clear()
        frame = _skipped_frame("code")
        result = seq.register_skipped(frame, "ctx2", None)
        self.assertEqual(len(result), 1)

    def test_after_clear_process_word_drops_stale_word(self):
        seq = _seq()
        seq.register_spoken(_spoken_frame("hello"), "ctx1", _tracker("hello"), True)
        seq.clear()
        # ctx1 was wiped by clear(); a delayed word for it is stale and dropped.
        result = seq.process_word("hello", pts=1, context_id="ctx1")
        self.assertEqual(result, [])

    def test_stale_words_do_not_corrupt_next_turn_transcript(self):
        # Regression for #4750: after an interruption clears context A and a new
        # context B is registered, delayed word-timestamps for A must not interleave
        # into B's transcript.
        seq = _seq()
        # Turn A starts, then is interrupted (clear wipes its slot + context map).
        seq.register_spoken(
            _spoken_frame("I just wanted to follow up"), "ctxA", _tracker("I"), True
        )
        seq.clear()
        # Turn B (the voicemail message) is registered.
        seq.register_spoken(_spoken_frame("Hello"), "ctxB", _tracker("Hello"), True)
        # Delayed words for the dead context A arrive — every one must be dropped.
        for stale in ("I", "just", "wanted", "to", "follow", "up"):
            self.assertEqual(seq.process_word(stale, pts=1, context_id="ctxA"), [])
        # Context B's own word still flows normally and appends to the transcript.
        result = seq.process_word("Hello", pts=2, context_id="ctxB")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].text, "Hello")
        self.assertEqual(result[0].context_id, "ctxB")
        self.assertTrue(result[0].append_to_context)


# ---------------------------------------------------------------------------
# CJK languages — Korean, Japanese, Chinese
# ---------------------------------------------------------------------------


class TestCJKLanguages(unittest.TestCase):
    """Sequencer behaviour for CJK language scenarios.

    Korean: Cartesia returns each word as a separate timestamp event (one word
    per process_word call).  Japanese/Chinese: Cartesia merges all characters
    in one timestamp message into a single combined token before calling
    process_word.
    """

    # --- Korean ---

    def test_korean_word_by_word_completes_slot_and_flushes_skipped(self):
        """Korean words fed one at a time complete the spoken slot and unblock a skipped frame."""
        seq = _seq()
        sentence = "저는 여러분의 AI 어시스턴트입니다."
        words = ["저는", "여러분의", "AI", "어시스턴트입니다."]
        seq.register_spoken(_spoken_frame(sentence), "ctx1", _tracker(sentence), True)
        skipped = _skipped_frame("[code]")
        seq.register_skipped(skipped, "ctx2", None)

        # Skipped stays blocked until the last word arrives
        for word in words[:-1]:
            partial = seq.process_word(word, pts=100, context_id="ctx1")
            self.assertFalse(any(f is skipped for f in partial))

        result = seq.process_word(words[-1], pts=200, context_id="ctx1")
        self.assertTrue(any(f is skipped for f in result))

    def test_korean_force_complete_emits_correct_remaining_text(self):
        """After one Korean word, force_complete emits the correct unspoken suffix."""
        seq = _seq()
        sentence = "저는 여러분의 AI 어시스턴트입니다."
        seq.register_spoken(_spoken_frame(sentence), "ctx1", _tracker(sentence), True)
        seq.process_word("저는", pts=10, context_id="ctx1")

        result = seq.force_complete(last_word_pts=50)
        tts_frames = [f for f in result if isinstance(f, TTSTextFrame)]
        self.assertEqual(len(tts_frames), 1)
        self.assertEqual(tts_frames[0].text, "여러분의 AI 어시스턴트입니다.")
        self.assertEqual(tts_frames[0].pts, 50)

    # --- Japanese ---

    def test_japanese_combined_groups_complete_spoken_slot(self):
        """Two Cartesia-style combined Japanese groups complete the slot and flush skipped."""
        seq = _seq()
        sentence = "こんにちは、私はあなたの"
        seq.register_spoken(_spoken_frame(sentence), "ctx1", _tracker(sentence), True)
        skipped = _skipped_frame("[skipped]")
        seq.register_skipped(skipped, "ctx2", None)

        r1 = seq.process_word("こんにちは、私", pts=100, context_id="ctx1")
        self.assertFalse(any(f is skipped for f in r1))

        r2 = seq.process_word("はあなたの", pts=200, context_id="ctx1")
        self.assertTrue(any(f is skipped for f in r2))

    def test_japanese_force_complete_emits_remaining_chars(self):
        """After the first Japanese combined group, force_complete emits the rest."""
        seq = _seq()
        sentence = "こんにちは、私はあなたの"
        seq.register_spoken(_spoken_frame(sentence), "ctx1", _tracker(sentence), True)
        seq.process_word("こんにちは、私", pts=10, context_id="ctx1")

        result = seq.force_complete(last_word_pts=50)
        tts_frames = [f for f in result if isinstance(f, TTSTextFrame)]
        self.assertEqual(len(tts_frames), 1)
        self.assertEqual(tts_frames[0].text, "はあなたの")

    # --- Chinese ---

    def test_chinese_combined_groups_complete_spoken_slot(self):
        """Two Cartesia-style combined Chinese groups complete the slot and flush skipped."""
        seq = _seq()
        sentence = "你好，我是你的智能"
        seq.register_spoken(_spoken_frame(sentence), "ctx1", _tracker(sentence), True)
        skipped = _skipped_frame("[skipped]")
        seq.register_skipped(skipped, "ctx2", None)

        r1 = seq.process_word("你好，我是", pts=100, context_id="ctx1")
        self.assertFalse(any(f is skipped for f in r1))

        r2 = seq.process_word("你的智能", pts=200, context_id="ctx1")
        self.assertTrue(any(f is skipped for f in r2))

    def test_chinese_force_complete_emits_remaining_chars(self):
        """After the first Chinese combined group, force_complete emits the rest."""
        seq = _seq()
        sentence = "你好，我是你的智能"
        seq.register_spoken(_spoken_frame(sentence), "ctx1", _tracker(sentence), True)
        seq.process_word("你好，我是", pts=10, context_id="ctx1")

        result = seq.force_complete(last_word_pts=50)
        tts_frames = [f for f in result if isinstance(f, TTSTextFrame)]
        self.assertEqual(len(tts_frames), 1)
        self.assertEqual(tts_frames[0].text, "你的智能")


# ---------------------------------------------------------------------------
# CJK context assembly — includes_inter_frame_spaces propagation
# ---------------------------------------------------------------------------


class TestCJKContextAssembly(unittest.TestCase):
    """CJK word-timestamp chunks assembled into assistant context must not include extra spaces.

    Regression: _build_word_frame always created TTSTextFrame(includes_inter_frame_spaces=False).
    The context aggregator then injected a space between every consecutive word frame,
    producing "です。 何か" instead of "です。何か" for Japanese/Chinese.

    The fix adds set_includes_inter_frame_spaces(context_id, True) so _build_word_frame
    stamps the correct flag on each frame and the context aggregator skips the space.
    """

    @staticmethod
    def _assemble_context(frames) -> str:
        """Reassemble TTSTextFrames the same way the context aggregator does."""
        parts = [
            TextPartForConcatenation(
                f.text,
                includes_inter_part_spaces=f.includes_inter_frame_spaces,
            )
            for f in frames
            if isinstance(f, TTSTextFrame)
        ]
        return concatenate_aggregated_text(parts)

    def test_japanese_chunks_no_space_in_context(self):
        """Japanese ElevenLabs-style word chunks must concatenate without an extra space."""
        seq = _seq()
        sentence = "どんなことでも気軽に相談してくださいね。"
        seq.register_spoken(
            _spoken_frame(sentence),
            "ctx1",
            _tracker(sentence),
            True,
            includes_inter_frame_spaces=True,
        )

        r1 = seq.process_word("どんなことでも気", pts=100, context_id="ctx1")
        r2 = seq.process_word("軽に相談してくださいね。", pts=200, context_id="ctx1")

        context_text = self._assemble_context(r1 + r2)
        self.assertEqual(
            context_text,
            "どんなことでも気軽に相談してくださいね。",
            "Japanese CJK chunks must not be separated by a space in context",
        )

    def test_chinese_chunks_no_space_in_context(self):
        """Chinese ElevenLabs-style word chunks must concatenate without an extra space."""
        seq = _seq()
        sentence = "你好，我是你的智能助手。"
        seq.register_spoken(
            _spoken_frame(sentence),
            "ctx1",
            _tracker(sentence),
            True,
            includes_inter_frame_spaces=True,
        )

        r1 = seq.process_word("你好，我是", pts=100, context_id="ctx1")
        r2 = seq.process_word("你的智能助手。", pts=200, context_id="ctx1")

        context_text = self._assemble_context(r1 + r2)
        self.assertEqual(
            context_text,
            "你好，我是你的智能助手。",
            "Chinese CJK chunks must not be separated by a space in context",
        )

    def test_english_words_still_have_spaces_in_context(self):
        """Non-CJK (English) word tokens must still be joined with spaces."""
        seq = _seq()
        sentence = "Hello world."
        seq.register_spoken(_spoken_frame(sentence), "ctx1", _tracker(sentence), True)

        r1 = seq.process_word("Hello", pts=100, context_id="ctx1")
        r2 = seq.process_word("world.", pts=200, context_id="ctx1")

        context_text = self._assemble_context(r1 + r2)
        self.assertEqual(context_text, "Hello world.")

    def test_force_complete_cjk_frame_has_flag(self):
        """force_complete for a CJK slot must also produce a frame with the flag set."""
        seq = _seq()
        sentence = "こんにちは、私はあなたの"
        seq.register_spoken(
            _spoken_frame(sentence),
            "ctx1",
            _tracker(sentence),
            True,
            includes_inter_frame_spaces=True,
        )
        seq.process_word("こんにちは、私", pts=10, context_id="ctx1")

        result = seq.force_complete(last_word_pts=50)
        tts_frames = [f for f in result if isinstance(f, TTSTextFrame)]
        self.assertEqual(len(tts_frames), 1)
        self.assertTrue(
            tts_frames[0].includes_inter_frame_spaces,
            "force_complete must propagate includes_inter_frame_spaces for CJK slots",
        )


# ---------------------------------------------------------------------------
# AggregatedTextProgressFrame emission
# ---------------------------------------------------------------------------


class TestAggregatedTextProgressFrame(unittest.TestCase):
    def _seq_with_spoken(self, text: str, ctx: str = "ctx1") -> AggregatedFrameSequencer:
        seq = _seq()
        frame = _spoken_frame(text)
        seq.register_spoken(frame, ctx, _tracker(text), append_to_context=True)
        return seq, frame

    def test_progress_frame_emitted_alongside_word_frame(self):
        seq, source = self._seq_with_spoken("hello")
        result = seq.process_word("hello", pts=100, context_id="ctx1")
        progress = [f for f in result if isinstance(f, AggregatedTextProgressFrame)]
        self.assertEqual(len(progress), 1)
        p = progress[0]
        self.assertEqual(p.text, "hello")
        self.assertEqual(p.aggregated_by, AggregationType.SENTENCE)
        self.assertEqual(p.accumulated_text, "hello")
        self.assertEqual(p.remaining_text, "")
        self.assertEqual(p.context_id, "ctx1")
        self.assertEqual(p.segment_id, source.id)
        self.assertEqual(p.pts, 100)

    def test_progress_accumulated_and_remaining_mid_slot(self):
        seq, _ = self._seq_with_spoken("hello world")
        result = seq.process_word("hello", pts=10, context_id="ctx1")
        progress = [f for f in result if isinstance(f, AggregatedTextProgressFrame)]
        self.assertEqual(len(progress), 1)
        self.assertEqual(progress[0].accumulated_text, "hello")
        self.assertEqual(progress[0].remaining_text, " world")

    def test_no_progress_frame_for_passthrough(self):
        seq = _seq()
        result = seq.process_word("hello", pts=1, context_id="ctx-unknown")
        progress = [f for f in result if isinstance(f, AggregatedTextProgressFrame)]
        self.assertEqual(progress, [])

    def test_progress_uses_user_facing_text_not_tts_text(self):
        """accumulated/remaining in the progress frame come from user_facing_text, not tts_text."""
        seq = _seq()
        frame = _spoken_frame("4111 1111 1111 1111")
        tracker = WordCompletionTracker(
            "<spell>4111 1111 1111 1111</spell>",
            llm_text="<card>4111 1111 1111 1111</card>",
            user_facing_text="4111 1111 1111 1111",
        )
        seq.register_spoken(frame, "ctx1", tracker, append_to_context=True)
        result = seq.process_word("4111", pts=10, context_id="ctx1")
        progress = [f for f in result if isinstance(f, AggregatedTextProgressFrame)]
        self.assertEqual(len(progress), 1)
        p = progress[0]
        # user_facing_text has no SSML tags
        self.assertEqual(p.accumulated_text, "4111")
        self.assertEqual(p.remaining_text, " 1111 1111 1111")
        # Sanity: tts accumulated includes the opening tag and would be different
        self.assertNotEqual(p.accumulated_text, tracker.get_accumulated_tts_text())

    def test_card_scenario_word_by_word(self):
        """Progress accumulated/remaining track user_facing_text through all four digit groups."""
        seq = _seq()
        frame = _spoken_frame("4111 1111 1111 1111")
        tracker = WordCompletionTracker(
            "<spell>4111 1111 1111 1111</spell>",
            llm_text="<card>4111 1111 1111 1111</card>",
            user_facing_text="4111 1111 1111 1111",
        )
        seq.register_spoken(frame, "ctx1", tracker, append_to_context=True)

        steps = [
            ("4111", "4111", " 1111 1111 1111"),
            ("1111", "4111 1111", " 1111 1111"),
            ("1111", "4111 1111 1111", " 1111"),
            ("1111", "4111 1111 1111 1111", ""),
        ]
        for word, exp_acc, exp_rem in steps:
            result = seq.process_word(word, pts=10, context_id="ctx1")
            progress = [f for f in result if isinstance(f, AggregatedTextProgressFrame)]
            self.assertEqual(len(progress), 1, f"expected 1 progress frame after '{word}'")
            self.assertEqual(progress[0].accumulated_text, exp_acc)
            self.assertEqual(progress[0].remaining_text, exp_rem)


# ---------------------------------------------------------------------------
# CJK includes_inter_frame_spaces: per-call arg must reach the emitted frame
# even when register_spoken did not set the flag on the slot.
#
# Regression: process_word used slot.includes_inter_frame_spaces exclusively
# when an active slot existed, ignoring the per-call includes_inter_frame_spaces
# argument.  tts_service.py calls register_spoken() without setting this flag,
# so it always defaulted to False even when add_word_timestamps was called
# with includes_inter_frame_spaces=True (as ElevenLabsTTSService does for CJK).
# ---------------------------------------------------------------------------


class TestCJKProcessWordFlagPropagation(unittest.TestCase):
    """process_word must propagate includes_inter_frame_spaces to TTSTextFrame.

    These tests simulate the tts_service.py code path: register_spoken is called
    without includes_inter_frame_spaces (the default False), and then process_word
    is called with includes_inter_frame_spaces=True (as add_word_timestamps does
    for ElevenLabs CJK languages).

    The flag must reach the emitted TTSTextFrame so the context aggregator knows
    not to inject spaces between consecutive CJK word tokens.
    """

    @staticmethod
    def _assemble_context(frames) -> str:
        parts = [
            TextPartForConcatenation(
                f.text,
                includes_inter_part_spaces=f.includes_inter_frame_spaces,
            )
            for f in frames
            if isinstance(f, TTSTextFrame)
        ]
        return concatenate_aggregated_text(parts)

    def test_process_word_flag_reaches_frame_when_slot_has_no_flag(self):
        """includes_inter_frame_spaces=True on process_word must stamp the emitted frame.

        register_spoken is called without includes_inter_frame_spaces (simulating
        tts_service.py), then process_word is called with includes_inter_frame_spaces=True
        (simulating add_word_timestamps for ElevenLabs CJK).  The frame must carry
        includes_inter_frame_spaces=True.
        """
        seq = _seq()
        sentence = "どんなことでも気軽に話しかけてくださいね。"
        # tts_service.py does NOT pass includes_inter_frame_spaces to register_spoken
        seq.register_spoken(_spoken_frame(sentence), "ctx1", _tracker(sentence), True)

        # add_word_timestamps passes includes_inter_frame_spaces=True for CJK
        result = seq.process_word(
            "どんなことでも気", pts=100, context_id="ctx1", includes_inter_frame_spaces=True
        )

        tts_frames = [f for f in result if isinstance(f, TTSTextFrame)]
        self.assertEqual(len(tts_frames), 1)
        self.assertTrue(
            tts_frames[0].includes_inter_frame_spaces,
            "TTSTextFrame must carry includes_inter_frame_spaces=True when process_word "
            "is called with that flag, even if register_spoken did not set it on the slot",
        )

    def test_cjk_two_chunks_no_space_when_slot_has_no_flag(self):
        """Two CJK chunks must concatenate without a space when process_word carries the flag.

        Matches the ElevenLabs runtime: register_spoken gets no flag; both
        process_word calls get includes_inter_frame_spaces=True.  Context assembly
        must produce '気軽に' not '気 軽に'.
        """
        seq = _seq()
        sentence = "どんなことでも気軽に話しかけてくださいね。"
        seq.register_spoken(_spoken_frame(sentence), "ctx1", _tracker(sentence), True)

        r1 = seq.process_word(
            "どんなことでも気", pts=100, context_id="ctx1", includes_inter_frame_spaces=True
        )
        r2 = seq.process_word(
            "軽に話しかけてくださいね。",
            pts=200,
            context_id="ctx1",
            includes_inter_frame_spaces=True,
        )

        assembled = self._assemble_context(r1 + r2)
        self.assertEqual(
            assembled,
            "どんなことでも気軽に話しかけてくださいね。",
            "CJK word chunks must concatenate without spaces when process_word "
            "carries includes_inter_frame_spaces=True",
        )

    def test_force_complete_cjk_flag_when_slot_has_no_flag(self):
        """force_complete must also carry the flag for CJK slots registered without it.

        When TTS drops the final token, force_complete emits the remainder.  The
        flag must still reach that frame so the context assembler doesn't add a space.
        """
        seq = _seq()
        sentence = "どんなことでも気軽に話しかけてくださいね。"
        seq.register_spoken(_spoken_frame(sentence), "ctx1", _tracker(sentence), True)

        # First chunk arrives with the flag via process_word
        seq.process_word(
            "どんなことでも気", pts=100, context_id="ctx1", includes_inter_frame_spaces=True
        )
        # Second chunk is dropped by TTS — force_complete emits the remainder
        result = seq.force_complete(last_word_pts=200)
        tts_frames = [f for f in result if isinstance(f, TTSTextFrame)]

        self.assertEqual(len(tts_frames), 1)
        self.assertTrue(
            tts_frames[0].includes_inter_frame_spaces,
            "force_complete must propagate includes_inter_frame_spaces for CJK slots "
            "even when register_spoken did not set the flag",
        )


if __name__ == "__main__":
    unittest.main()
