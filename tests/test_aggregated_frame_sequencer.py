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

from pipecat.frames.frames import AggregatedTextFrame, AggregationType, TTSTextFrame
from pipecat.utils.context.aggregated_frame_sequencer import AggregatedFrameSequencer
from pipecat.utils.context.word_completion_tracker import WordCompletionTracker

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
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], TTSTextFrame)

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
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], TTSTextFrame)

    def test_completing_word_flushes_blocked_skipped_frame(self):
        seq = self._seq_with_spoken("hello")
        skipped = _skipped_frame("code")
        seq.register_skipped(skipped, "ctx2", None)
        result = seq.process_word("hello", pts=50, context_id="ctx1")
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], TTSTextFrame)
        self.assertIs(result[1], skipped)

    def test_last_of_multiple_words_flushes_skipped(self):
        seq = self._seq_with_spoken("hello world")
        skipped = _skipped_frame("code")
        seq.register_skipped(skipped, "ctx2", None)
        seq.process_word("hello", pts=10, context_id="ctx1")
        result = seq.process_word("world", pts=20, context_id="ctx1")
        self.assertTrue(any(f is skipped for f in result))

    def test_no_active_slot_emits_passthrough(self):
        seq = _seq()
        result = seq.process_word("hello", pts=1, context_id="ctx-unknown")
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], TTSTextFrame)
        self.assertEqual(result[0].text, "hello")
        self.assertEqual(result[0].context_id, "ctx-unknown")

    def test_passthrough_uses_default_append_to_context_true(self):
        seq = _seq()
        result = seq.process_word("hello", pts=1, context_id="ctx-unknown")
        self.assertTrue(result[0].append_to_context)

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

    def test_after_clear_process_word_uses_passthrough(self):
        seq = _seq()
        seq.register_spoken(_spoken_frame("hello"), "ctx1", _tracker("hello"), True)
        seq.clear()
        result = seq.process_word("hello", pts=1, context_id="ctx1")
        # No active slot after clear → passthrough
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, "hello")


if __name__ == "__main__":
    unittest.main()
