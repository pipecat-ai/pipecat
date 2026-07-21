#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for AggregatedFrameSequencer.

register_spoken is async (it may drive an async text aggregator when the
sequencer streams tokens), so most test classes here use
unittest.IsolatedAsyncioTestCase. Every other method remains synchronous.

Test groups:
- register_skipped: immediate flush vs. blocked by a preceding spoken slot
- register_spoken / complete_spoken_slot: push_text_frames=True path (build_tracker=False)
- flush: pts propagation, transport_destination, stops at incomplete spoken slot
- process_word: normal, completing, passthrough, raw_text propagation
- process_word overflow: single token spanning two slot boundaries
- process_word force-complete via belongs_here failure
- force_complete: remaining text emission, raw_text, corrupt raw discard, slot ordering
- clear: resets all state
- register_spoken streaming: token-by-token sentence accumulation and promotion
- register_spoken buffered words: words arriving before a pending sentence promotes
- register_skipped forces finalize: streamed pending sentence forced by a skipped frame
- finalize: end-of-turn forced promotion
- clear resets streaming state: pending accumulation and buffered words
"""

import unittest

from pipecat.frames.frames import (
    AggregatedTextFrame,
    AggregatedTextProgressFrame,
    AggregationType,
    TTSTextFrame,
)
from pipecat.utils.context.aggregated_frame_sequencer import (
    AggregatedFrameSequencer,
    _ParallelSentenceAggregator,
)
from pipecat.utils.string import TextPartForConcatenation, concatenate_aggregated_text

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seq(streaming: bool = False) -> AggregatedFrameSequencer:
    return AggregatedFrameSequencer(name="test", streaming=streaming)


def _spoken_frame(text: str, raw_text: str | None = None) -> AggregatedTextFrame:
    return AggregatedTextFrame(text, AggregationType.SENTENCE, raw_text=raw_text)


def _skipped_frame(text: str) -> AggregatedTextFrame:
    return AggregatedTextFrame(text, "code")


# ---------------------------------------------------------------------------
# register_skipped
# ---------------------------------------------------------------------------


class TestRegisterSkipped(unittest.IsolatedAsyncioTestCase):
    async def test_emits_immediately_with_empty_queue(self):
        seq = _seq()
        frame = _skipped_frame("code block")
        result = await seq.register_skipped(frame, "ctx1", None)
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], frame)

    async def test_sets_append_to_context_true(self):
        seq = _seq()
        frame = _skipped_frame("code")
        await seq.register_skipped(frame, "ctx1", None)
        self.assertTrue(frame.append_to_context)

    async def test_sets_context_id_on_frame(self):
        seq = _seq()
        frame = _skipped_frame("code")
        await seq.register_skipped(frame, "ctx42", None)
        self.assertEqual(frame.context_id, "ctx42")

    async def test_sets_transport_destination(self):
        seq = _seq()
        frame = _skipped_frame("code")
        result = await seq.register_skipped(frame, "ctx1", "dest-A")
        self.assertEqual(result[0].transport_destination, "dest-A")

    async def test_blocked_by_incomplete_spoken_slot(self):
        seq = _seq()
        await seq.register_spoken(_spoken_frame("hello world"), "ctx1", "hello world", True)
        result = await seq.register_skipped(_skipped_frame("code"), "ctx2", None)
        self.assertEqual(result, [])

    async def test_emits_immediately_after_already_complete_spoken_slot(self):
        seq = _seq()
        await seq.register_spoken(
            _spoken_frame("hi"), "ctx1", "hi", append_to_context=True, build_tracker=False
        )
        seq.complete_spoken_slot()
        result = await seq.register_skipped(_skipped_frame("code"), "ctx2", None)
        self.assertEqual(len(result), 1)

    async def test_multiple_skipped_before_any_spoken_all_emit(self):
        seq = _seq()
        r1 = await seq.register_skipped(_skipped_frame("code1"), "ctx1", None)
        r2 = await seq.register_skipped(_skipped_frame("code2"), "ctx2", None)
        self.assertEqual(len(r1), 1)
        self.assertEqual(len(r2), 1)


# ---------------------------------------------------------------------------
# register_spoken / complete_spoken_slot  (push_text_frames=True path)
# ---------------------------------------------------------------------------


class TestCompleteSpokenSlot(unittest.IsolatedAsyncioTestCase):
    def test_noop_with_empty_queue(self):
        seq = _seq()
        self.assertEqual(seq.complete_spoken_slot(), [])

    async def test_marks_slot_complete_and_flushes_skipped(self):
        seq = _seq()
        await seq.register_spoken(
            _spoken_frame("hello"), "ctx1", "hello", append_to_context=True, build_tracker=False
        )
        skipped = _skipped_frame("code")
        await seq.register_skipped(skipped, "ctx2", None)  # blocked

        result = seq.complete_spoken_slot()
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], skipped)
        self.assertTrue(skipped.append_to_context)

    async def test_only_first_pending_slot_is_marked(self):
        seq = _seq()
        await seq.register_spoken(
            _spoken_frame("one"), "ctx1", "one", append_to_context=True, build_tracker=False
        )
        await seq.register_spoken(
            _spoken_frame("two"), "ctx2", "two", append_to_context=True, build_tracker=False
        )
        skipped = _skipped_frame("code")
        await seq.register_skipped(skipped, "ctx3", None)

        # ctx2 still blocks the skipped frame
        result = seq.complete_spoken_slot()
        self.assertEqual(result, [])

    async def test_skipped_flushes_after_all_preceding_spoken_complete(self):
        seq = _seq()
        await seq.register_spoken(
            _spoken_frame("one"), "ctx1", "one", append_to_context=True, build_tracker=False
        )
        await seq.register_spoken(
            _spoken_frame("two"), "ctx2", "two", append_to_context=True, build_tracker=False
        )
        skipped = _skipped_frame("code")
        await seq.register_skipped(skipped, "ctx3", None)

        seq.complete_spoken_slot()  # completes ctx1
        result = seq.complete_spoken_slot()  # completes ctx2 → flush skipped
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], skipped)


# ---------------------------------------------------------------------------
# flush
# ---------------------------------------------------------------------------


class TestFlush(unittest.IsolatedAsyncioTestCase):
    def test_empty_queue_returns_empty(self):
        self.assertEqual(_seq().flush(), [])

    async def test_stops_at_incomplete_spoken_slot(self):
        seq = _seq()
        await seq.register_spoken(
            _spoken_frame("hello"), "ctx1", "hello", append_to_context=True, build_tracker=False
        )
        await seq.register_skipped(_skipped_frame("code"), "ctx2", None)
        self.assertEqual(seq.flush(), [])

    async def test_last_word_pts_assigned_to_skipped_frame(self):
        seq = _seq()
        await seq.register_spoken(_spoken_frame("hello"), "ctx1", "hello", True)
        skipped = _skipped_frame("code")
        await seq.register_skipped(skipped, "ctx2", None)

        # process_word("hello") completes the spoken slot and calls flush(last_word_pts=77)
        result = seq.process_word("hello", pts=77, context_id="ctx1")
        flushed = [f for f in result if isinstance(f, AggregatedTextFrame) and f.text == "code"]
        self.assertEqual(len(flushed), 1)
        self.assertEqual(flushed[0].pts, 77)

    async def test_complete_spoken_slots_are_swept(self):
        seq = _seq()
        await seq.register_spoken(
            _spoken_frame("hello"), "ctx1", "hello", append_to_context=True, build_tracker=False
        )
        seq.complete_spoken_slot()
        # Queue should be empty after sweeping the complete spoken slot
        self.assertEqual(seq._slots, [])


# ---------------------------------------------------------------------------
# process_word — basic
# ---------------------------------------------------------------------------


class TestProcessWordBasic(unittest.IsolatedAsyncioTestCase):
    async def _seq_with_spoken(self, text: str, ctx: str = "ctx1", append: bool = True):
        seq = _seq()
        await seq.register_spoken(_spoken_frame(text), ctx, text, append)
        return seq

    async def test_returns_tts_text_frame(self):
        seq = await self._seq_with_spoken("hello")
        result = seq.process_word("hello", pts=100, context_id="ctx1")
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], TTSTextFrame)
        self.assertIsInstance(result[1], AggregatedTextProgressFrame)

    async def test_frame_text_and_pts(self):
        seq = await self._seq_with_spoken("hello")
        result = seq.process_word("hello", pts=100, context_id="ctx1")
        self.assertEqual(result[0].text, "hello")
        self.assertEqual(result[0].pts, 100)

    async def test_frame_context_id(self):
        seq = await self._seq_with_spoken("hello", ctx="ctx99")
        result = seq.process_word("hello", pts=1, context_id="ctx99")
        self.assertEqual(result[0].context_id, "ctx99")

    async def test_append_to_context_true(self):
        seq = await self._seq_with_spoken("hello", append=True)
        result = seq.process_word("hello", pts=1, context_id="ctx1")
        self.assertTrue(result[0].append_to_context)

    async def test_append_to_context_false(self):
        seq = await self._seq_with_spoken("hello", append=False)
        result = seq.process_word("hello", pts=1, context_id="ctx1")
        self.assertFalse(result[0].append_to_context)

    async def test_non_completing_word_does_not_flush_skipped(self):
        seq = await self._seq_with_spoken("hello world")
        await seq.register_skipped(_skipped_frame("code"), "ctx2", None)
        result = seq.process_word("hello", pts=10, context_id="ctx1")
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], TTSTextFrame)
        self.assertIsInstance(result[1], AggregatedTextProgressFrame)

    async def test_completing_word_flushes_blocked_skipped_frame(self):
        seq = await self._seq_with_spoken("hello")
        skipped = _skipped_frame("code")
        await seq.register_skipped(skipped, "ctx2", None)
        result = seq.process_word("hello", pts=50, context_id="ctx1")
        self.assertEqual(len(result), 3)
        self.assertIsInstance(result[0], TTSTextFrame)
        self.assertIsInstance(result[1], AggregatedTextProgressFrame)
        self.assertIs(result[2], skipped)

    async def test_last_of_multiple_words_flushes_skipped(self):
        seq = await self._seq_with_spoken("hello world")
        skipped = _skipped_frame("code")
        await seq.register_skipped(skipped, "ctx2", None)
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

    async def test_unrecognised_word_emits_passthrough(self):
        seq = _seq()
        await seq.register_spoken(_spoken_frame("hello world"), "ctx1", "hello world", True)
        # "zzz" doesn't belong to "hello world" and there is no next slot
        result = seq.process_word("zzz", pts=5, context_id="ctx1")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, "zzz")


# ---------------------------------------------------------------------------
# process_word — raw_text propagation
# ---------------------------------------------------------------------------


class TestProcessWordRawText(unittest.IsolatedAsyncioTestCase):
    async def test_raw_text_split_across_word_frames(self):
        seq = _seq()
        await seq.register_spoken(
            _spoken_frame("4111 1111", raw_text="<card>4111 1111</card>"),
            "ctx1",
            "4111 1111",
            append_to_context=True,
        )
        r1 = seq.process_word("4111", pts=10, context_id="ctx1")
        r2 = seq.process_word("1111", pts=20, context_id="ctx1")
        self.assertEqual(r1[0].raw_text, "<card>4111")
        last_word_frames = [f for f in r2 if isinstance(f, TTSTextFrame)]
        self.assertEqual(last_word_frames[0].raw_text, "1111</card>")

    async def test_raw_text_defaults_to_frame_text_when_no_raw_text(self):
        """llm_text is always derived from frame.raw_text or frame.text — never None."""
        seq = _seq()
        await seq.register_spoken(_spoken_frame("hello"), "ctx1", "hello", True)
        result = seq.process_word("hello", pts=1, context_id="ctx1")
        self.assertEqual(result[0].raw_text, "hello")


# ---------------------------------------------------------------------------
# process_word — overflow (single token spanning two slots)
# ---------------------------------------------------------------------------


class TestProcessWordOverflow(unittest.IsolatedAsyncioTestCase):
    async def test_overflow_produces_two_tts_text_frames(self):
        seq = _seq()
        await seq.register_spoken(_spoken_frame("abc"), "ctx1", "abc", True)
        await seq.register_spoken(_spoken_frame("def"), "ctx2", "def", True)

        result = seq.process_word("abcdef", pts=100, context_id="ctx1")
        word_frames = [f for f in result if isinstance(f, TTSTextFrame)]
        self.assertEqual(len(word_frames), 2)
        self.assertEqual(word_frames[0].text, "abc")
        self.assertEqual(word_frames[1].text, "def")

    async def test_overflow_assigns_correct_context_ids(self):
        seq = _seq()
        await seq.register_spoken(_spoken_frame("abc"), "ctx1", "abc", True)
        await seq.register_spoken(_spoken_frame("def"), "ctx2", "def", True)

        result = seq.process_word("abcdef", pts=100, context_id="ctx1")
        word_frames = [f for f in result if isinstance(f, TTSTextFrame)]
        self.assertEqual(word_frames[0].context_id, "ctx1")
        self.assertEqual(word_frames[1].context_id, "ctx2")

    async def test_overflow_completing_next_slot_flushes_skipped(self):
        seq = _seq()
        await seq.register_spoken(_spoken_frame("abc"), "ctx1", "abc", True)
        await seq.register_spoken(_spoken_frame("def"), "ctx2", "def", True)
        skipped = _skipped_frame("code")
        await seq.register_skipped(skipped, "ctx3", None)  # blocked behind ctx2

        result = seq.process_word("abcdef", pts=100, context_id="ctx1")
        self.assertTrue(any(f is skipped for f in result))

    async def test_overflow_not_completing_next_slot_does_not_flush_skipped(self):
        seq = _seq()
        await seq.register_spoken(_spoken_frame("abc"), "ctx1", "abc", True)
        await seq.register_spoken(_spoken_frame("def ghi"), "ctx2", "def ghi", True)
        skipped = _skipped_frame("code")
        await seq.register_skipped(skipped, "ctx3", None)

        # "abcdef" overflows: "def" goes to ctx2, but ctx2 still expects " ghi"
        result = seq.process_word("abcdef", pts=100, context_id="ctx1")
        self.assertFalse(any(f is skipped for f in result))


# ---------------------------------------------------------------------------
# process_word — force-complete via word_belongs_here failure
# ---------------------------------------------------------------------------


class TestProcessWordForcesComplete(unittest.IsolatedAsyncioTestCase):
    async def test_word_for_next_slot_force_completes_current(self):
        """When a word belongs to the next slot but not the current, the current
        slot is force-completed and the word is routed to the next slot."""
        seq = _seq()
        await seq.register_spoken(_spoken_frame("hello"), "ctx1", "hello", True)
        await seq.register_spoken(_spoken_frame("world"), "ctx2", "world", True)

        # "world" doesn't belong to ctx1 but belongs to ctx2
        result = seq.process_word("world", pts=50, context_id="ctx2")
        word_frames = [f for f in result if isinstance(f, TTSTextFrame)]
        texts = {f.text for f in word_frames}
        self.assertIn("world", texts)

    async def test_force_complete_then_overflow_flushes_skipped(self):
        seq = _seq()
        await seq.register_spoken(_spoken_frame("hello"), "ctx1", "hello", True)
        await seq.register_spoken(_spoken_frame("world"), "ctx2", "world", True)
        skipped = _skipped_frame("code")
        await seq.register_skipped(skipped, "ctx3", None)

        # "world" force-completes ctx1 and completes ctx2 via overflow
        result = seq.process_word("world", pts=50, context_id="ctx2")
        self.assertTrue(any(f is skipped for f in result))

    async def test_whitespace_slot_force_complete_skips_emission(self):
        """When a whitespace-only slot is force-completed, get_word_for_frame()
        returns an empty string for it, so no frame should be emitted for that
        slot."""
        seq = _seq()
        await seq.register_spoken(_spoken_frame(" "), "ctx1", " ", True)
        await seq.register_spoken(_spoken_frame("World"), "ctx2", "World", True)

        # Word for ctx2 arrives, forcing ctx1 (whitespace) to complete
        result = seq.process_word("World", pts=10, context_id="ctx2")
        word_frames = [f for f in result if isinstance(f, TTSTextFrame)]

        # Should only emit ONE frame with ctx2, not a duplicate with ctx1
        self.assertEqual(len(word_frames), 1)
        self.assertEqual(word_frames[0].text, "World")
        self.assertEqual(word_frames[0].context_id, "ctx2")


# ---------------------------------------------------------------------------
# force_complete
# ---------------------------------------------------------------------------


class TestForceComplete(unittest.IsolatedAsyncioTestCase):
    async def test_emits_remaining_text_when_word_dropped(self):
        seq = _seq()
        await seq.register_spoken(_spoken_frame("hello world"), "ctx1", "hello world", True)
        seq.process_word("hello", pts=10, context_id="ctx1")  # "world" never arrives

        result = seq.force_complete(last_word_pts=50)
        tts_frames = [f for f in result if isinstance(f, TTSTextFrame)]
        self.assertEqual(len(tts_frames), 1)
        self.assertEqual(tts_frames[0].text, "world")
        self.assertEqual(tts_frames[0].pts, 50)

    async def test_emits_full_text_when_no_words_arrived(self):
        seq = _seq()
        await seq.register_spoken(_spoken_frame("hello world"), "ctx1", "hello world", True)

        result = seq.force_complete(last_word_pts=0)
        tts_frames = [f for f in result if isinstance(f, TTSTextFrame)]
        self.assertEqual(len(tts_frames), 1)
        self.assertEqual(tts_frames[0].text, "hello world")

    async def test_already_complete_slot_emits_nothing(self):
        seq = _seq()
        await seq.register_spoken(_spoken_frame("hi"), "ctx1", "hi", True)
        seq.process_word("hi", pts=5, context_id="ctx1")  # completes normally

        result = seq.force_complete(last_word_pts=10)
        self.assertEqual(result, [])

    async def test_flushes_skipped_frames_after_completing(self):
        seq = _seq()
        await seq.register_spoken(_spoken_frame("hello"), "ctx1", "hello", True)
        skipped = _skipped_frame("code")
        await seq.register_skipped(skipped, "ctx2", None)

        result = seq.force_complete(last_word_pts=20)
        self.assertTrue(any(f is skipped for f in result))
        self.assertTrue(skipped.append_to_context)

    async def test_propagates_raw_text(self):
        seq = _seq()
        await seq.register_spoken(
            _spoken_frame("4111 1111", raw_text="<card>4111 1111</card>"),
            "ctx1",
            "4111 1111",
            append_to_context=True,
        )
        seq.process_word("4111", pts=10, context_id="ctx1")  # "1111" never arrives

        result = seq.force_complete(last_word_pts=20)
        tts_frames = [f for f in result if isinstance(f, TTSTextFrame)]
        self.assertEqual(tts_frames[0].text, "1111")
        self.assertEqual(tts_frames[0].raw_text, "1111</card>")

    async def test_discards_corrupt_raw_remaining(self):
        """raw_remaining is discarded when it does not contain remaining_text."""
        seq = _seq()
        # "abc" normalized ≠ "xyz" normalized — any remaining won't be in raw_remaining
        await seq.register_spoken(
            _spoken_frame("abc", raw_text="xyz"),
            "ctx1",
            "abc",
            append_to_context=True,
        )
        result = seq.force_complete(last_word_pts=0)
        tts_frames = [f for f in result if isinstance(f, TTSTextFrame)]
        self.assertEqual(len(tts_frames), 1)
        self.assertEqual(tts_frames[0].text, "abc")
        self.assertIsNone(tts_frames[0].raw_text)  # discarded due to corruption

    async def test_slot_without_tracker_just_marks_complete_and_flushes(self):
        seq = _seq()
        await seq.register_spoken(
            _spoken_frame("hello"), "ctx1", "hello", append_to_context=True, build_tracker=False
        )
        skipped = _skipped_frame("code")
        await seq.register_skipped(skipped, "ctx2", None)

        result = seq.force_complete(last_word_pts=0)
        tts_frames = [f for f in result if isinstance(f, TTSTextFrame)]
        self.assertEqual(tts_frames, [])  # no tracker → no word frame
        self.assertTrue(any(f is skipped for f in result))

    async def test_multiple_incomplete_slots_all_emitted(self):
        seq = _seq()
        await seq.register_spoken(_spoken_frame("hello"), "ctx1", "hello", True)
        await seq.register_spoken(_spoken_frame("world"), "ctx2", "world", True)

        result = seq.force_complete(last_word_pts=0)
        tts_frames = [f for f in result if isinstance(f, TTSTextFrame)]
        texts = {f.text for f in tts_frames}
        self.assertIn("hello", texts)
        self.assertIn("world", texts)


# ---------------------------------------------------------------------------
# Late words after force_complete (back-to-back turn double-emit guard)
# ---------------------------------------------------------------------------


class TestLateWordsAfterForceComplete(unittest.IsolatedAsyncioTestCase):
    """Late word-timestamps arriving after context teardown must not re-emit the turn.

    Regression for the back-to-back (superseding) double-emit seen with word-timestamp
    services (e.g. Cartesia) in non-streaming SENTENCE mode: on two LLM responses with
    no user turn between, the second turn's async word timestamps land *after*
    force_complete() has already emitted the slots' remaining text and emptied the queue.
    Without a guard those late words fall through to the passthrough branch and emit the
    turn's text a second time, even though the audio was synthesized once.
    """

    CTX = "ctx-B"
    SENTENCES = ["I'm sorry to hear that.", "Please tell me your PIN."]

    async def _register(self, seq):
        for s in self.SENTENCES:
            await seq.register_spoken(_spoken_frame(s), self.CTX, s, append_to_context=True)

    @staticmethod
    def _joined(frames):
        return "".join(f.text for f in frames if isinstance(f, TTSTextFrame)).replace(" ", "")

    async def test_force_complete_then_late_words_emit_turn_once(self):
        seq = _seq()
        await self._register(seq)

        emitted = seq.force_complete(last_word_pts=1000)  # teardown wins the race
        pts = 2000
        for s in self.SENTENCES:
            for w in s.split():
                emitted += seq.process_word(w, pts, self.CTX)  # late words
                pts += 100

        marker = "".join(self.SENTENCES).replace(" ", "")
        self.assertEqual(
            self._joined(emitted).count(marker),
            1,
            "turn text was emitted more than once: late words after force_complete "
            "were re-emitted as passthrough",
        )

    async def test_late_words_after_teardown_are_dropped(self):
        seq = _seq()
        await self._register(seq)
        seq.force_complete(last_word_pts=1000)  # empties the queue, records the context

        for w in "I'm sorry to hear that.".split():
            self.assertEqual(seq.process_word(w, pts=2000, context_id=self.CTX), [])

    async def test_normal_completion_then_force_complete_emits_once(self):
        """Regression guard: words that complete slots before teardown emit exactly once."""
        seq = _seq()
        await self._register(seq)

        emitted = []
        pts = 100
        for s in self.SENTENCES:
            for w in s.split():
                emitted += seq.process_word(w, pts, self.CTX)
                pts += 100
        emitted += seq.force_complete(last_word_pts=pts)  # nothing left to complete

        marker = "".join(self.SENTENCES).replace(" ", "")
        self.assertEqual(self._joined(emitted).count(marker), 1)

    async def test_new_slot_for_context_reenables_words(self):
        """Re-registering a slot under a force-completed context clears the drop flag.

        A word for that context is only stale until the context is used again; once a
        fresh slot is registered under the same id, its words must flow normally.
        """
        seq = _seq()
        await seq.register_spoken(_spoken_frame("hello"), self.CTX, "hello", True)
        seq.force_complete(last_word_pts=10)  # records self.CTX as force-completed
        # Same context id reused for a new turn.
        await seq.register_spoken(_spoken_frame("world"), self.CTX, "world", True)

        result = seq.process_word("world", pts=20, context_id=self.CTX)
        word_frames = [f for f in result if isinstance(f, TTSTextFrame)]
        self.assertEqual([f.text for f in word_frames], ["world"])


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


class TestClear(unittest.IsolatedAsyncioTestCase):
    async def test_clears_slots(self):
        seq = _seq()
        await seq.register_spoken(_spoken_frame("hello"), "ctx1", "hello", True)
        await seq.register_skipped(_skipped_frame("code"), "ctx2", None)
        seq.clear()
        self.assertEqual(seq._slots, [])

    async def test_clears_context_map(self):
        seq = _seq()
        await seq.register_spoken(_spoken_frame("hello"), "ctx1", "hello", True)
        seq.clear()
        self.assertEqual(seq._context_append_to_context, {})

    async def test_after_clear_skipped_emits_immediately(self):
        seq = _seq()
        await seq.register_spoken(_spoken_frame("hello"), "ctx1", "hello", True)
        seq.clear()
        frame = _skipped_frame("code")
        result = await seq.register_skipped(frame, "ctx2", None)
        self.assertEqual(len(result), 1)

    async def test_after_clear_process_word_drops_stale_word(self):
        seq = _seq()
        await seq.register_spoken(_spoken_frame("hello"), "ctx1", "hello", True)
        seq.clear()
        # ctx1 was wiped by clear(); a delayed word for it is stale and dropped.
        result = seq.process_word("hello", pts=1, context_id="ctx1")
        self.assertEqual(result, [])

    async def test_stale_words_do_not_corrupt_next_turn_transcript(self):
        # Regression for #4750: after an interruption clears context A and a new
        # context B is registered, delayed word-timestamps for A must not interleave
        # into B's transcript.
        seq = _seq()
        # Turn A starts, then is interrupted (clear wipes its slot + context map).
        await seq.register_spoken(
            _spoken_frame("I just wanted to follow up"), "ctxA", "I just wanted to follow up", True
        )
        seq.clear()
        # Turn B (the voicemail message) is registered.
        await seq.register_spoken(_spoken_frame("Hello"), "ctxB", "Hello", True)
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


class TestCJKLanguages(unittest.IsolatedAsyncioTestCase):
    """Sequencer behaviour for CJK language scenarios.

    Korean: Cartesia returns each word as a separate timestamp event (one word
    per process_word call).  Japanese/Chinese: Cartesia merges all characters
    in one timestamp message into a single combined token before calling
    process_word.
    """

    # --- Korean ---

    async def test_korean_word_by_word_completes_slot_and_flushes_skipped(self):
        """Korean words fed one at a time complete the spoken slot and unblock a skipped frame."""
        seq = _seq()
        sentence = "저는 여러분의 AI 어시스턴트입니다."
        words = ["저는", "여러분의", "AI", "어시스턴트입니다."]
        await seq.register_spoken(_spoken_frame(sentence), "ctx1", sentence, True)
        skipped = _skipped_frame("[code]")
        await seq.register_skipped(skipped, "ctx2", None)

        # Skipped stays blocked until the last word arrives
        for word in words[:-1]:
            partial = seq.process_word(word, pts=100, context_id="ctx1")
            self.assertFalse(any(f is skipped for f in partial))

        result = seq.process_word(words[-1], pts=200, context_id="ctx1")
        self.assertTrue(any(f is skipped for f in result))

    async def test_korean_force_complete_emits_correct_remaining_text(self):
        """After one Korean word, force_complete emits the correct unspoken suffix."""
        seq = _seq()
        sentence = "저는 여러분의 AI 어시스턴트입니다."
        await seq.register_spoken(_spoken_frame(sentence), "ctx1", sentence, True)
        seq.process_word("저는", pts=10, context_id="ctx1")

        result = seq.force_complete(last_word_pts=50)
        tts_frames = [f for f in result if isinstance(f, TTSTextFrame)]
        self.assertEqual(len(tts_frames), 1)
        self.assertEqual(tts_frames[0].text, "여러분의 AI 어시스턴트입니다.")
        self.assertEqual(tts_frames[0].pts, 50)

    # --- Japanese ---

    async def test_japanese_combined_groups_complete_spoken_slot(self):
        """Two Cartesia-style combined Japanese groups complete the slot and flush skipped."""
        seq = _seq()
        sentence = "こんにちは、私はあなたの"
        await seq.register_spoken(_spoken_frame(sentence), "ctx1", sentence, True)
        skipped = _skipped_frame("[skipped]")
        await seq.register_skipped(skipped, "ctx2", None)

        r1 = seq.process_word("こんにちは、私", pts=100, context_id="ctx1")
        self.assertFalse(any(f is skipped for f in r1))

        r2 = seq.process_word("はあなたの", pts=200, context_id="ctx1")
        self.assertTrue(any(f is skipped for f in r2))

    async def test_japanese_force_complete_emits_remaining_chars(self):
        """After the first Japanese combined group, force_complete emits the rest."""
        seq = _seq()
        sentence = "こんにちは、私はあなたの"
        await seq.register_spoken(_spoken_frame(sentence), "ctx1", sentence, True)
        seq.process_word("こんにちは、私", pts=10, context_id="ctx1")

        result = seq.force_complete(last_word_pts=50)
        tts_frames = [f for f in result if isinstance(f, TTSTextFrame)]
        self.assertEqual(len(tts_frames), 1)
        self.assertEqual(tts_frames[0].text, "はあなたの")

    # --- Chinese ---

    async def test_chinese_combined_groups_complete_spoken_slot(self):
        """Two Cartesia-style combined Chinese groups complete the slot and flush skipped."""
        seq = _seq()
        sentence = "你好，我是你的智能"
        await seq.register_spoken(_spoken_frame(sentence), "ctx1", sentence, True)
        skipped = _skipped_frame("[skipped]")
        await seq.register_skipped(skipped, "ctx2", None)

        r1 = seq.process_word("你好，我是", pts=100, context_id="ctx1")
        self.assertFalse(any(f is skipped for f in r1))

        r2 = seq.process_word("你的智能", pts=200, context_id="ctx1")
        self.assertTrue(any(f is skipped for f in r2))

    async def test_chinese_force_complete_emits_remaining_chars(self):
        """After the first Chinese combined group, force_complete emits the rest."""
        seq = _seq()
        sentence = "你好，我是你的智能"
        await seq.register_spoken(_spoken_frame(sentence), "ctx1", sentence, True)
        seq.process_word("你好，我是", pts=10, context_id="ctx1")

        result = seq.force_complete(last_word_pts=50)
        tts_frames = [f for f in result if isinstance(f, TTSTextFrame)]
        self.assertEqual(len(tts_frames), 1)
        self.assertEqual(tts_frames[0].text, "你的智能")


# ---------------------------------------------------------------------------
# CJK context assembly — includes_inter_frame_spaces propagation
# ---------------------------------------------------------------------------


class TestCJKContextAssembly(unittest.IsolatedAsyncioTestCase):
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

    async def test_japanese_chunks_no_space_in_context(self):
        """Japanese ElevenLabs-style word chunks must concatenate without an extra space."""
        seq = _seq()
        sentence = "どんなことでも気軽に相談してくださいね。"
        await seq.register_spoken(
            _spoken_frame(sentence),
            "ctx1",
            sentence,
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

    async def test_chinese_chunks_no_space_in_context(self):
        """Chinese ElevenLabs-style word chunks must concatenate without an extra space."""
        seq = _seq()
        sentence = "你好，我是你的智能助手。"
        await seq.register_spoken(
            _spoken_frame(sentence),
            "ctx1",
            sentence,
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

    async def test_english_words_still_have_spaces_in_context(self):
        """Non-CJK (English) word tokens must still be joined with spaces."""
        seq = _seq()
        sentence = "Hello world."
        await seq.register_spoken(_spoken_frame(sentence), "ctx1", sentence, True)

        r1 = seq.process_word("Hello", pts=100, context_id="ctx1")
        r2 = seq.process_word("world.", pts=200, context_id="ctx1")

        context_text = self._assemble_context(r1 + r2)
        self.assertEqual(context_text, "Hello world.")

    async def test_force_complete_cjk_frame_has_flag(self):
        """force_complete for a CJK slot must also produce a frame with the flag set."""
        seq = _seq()
        sentence = "こんにちは、私はあなたの"
        await seq.register_spoken(
            _spoken_frame(sentence),
            "ctx1",
            sentence,
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


class TestAggregatedTextProgressFrame(unittest.IsolatedAsyncioTestCase):
    async def _seq_with_spoken(self, text: str, ctx: str = "ctx1") -> AggregatedFrameSequencer:
        seq = _seq()
        frame = _spoken_frame(text)
        await seq.register_spoken(frame, ctx, text, append_to_context=True)
        return seq, frame

    async def test_progress_frame_emitted_alongside_word_frame(self):
        seq, source = await self._seq_with_spoken("hello")
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

    async def test_progress_accumulated_and_remaining_mid_slot(self):
        seq, _ = await self._seq_with_spoken("hello world")
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

    async def test_progress_uses_user_facing_text_not_tts_text(self):
        """accumulated/remaining in the progress frame come from user_facing_text, not tts_text."""
        seq = _seq()
        frame = _spoken_frame("4111 1111 1111 1111", raw_text="<card>4111 1111 1111 1111</card>")
        await seq.register_spoken(
            frame, "ctx1", "<spell>4111 1111 1111 1111</spell>", append_to_context=True
        )
        result = seq.process_word("4111", pts=10, context_id="ctx1")
        progress = [f for f in result if isinstance(f, AggregatedTextProgressFrame)]
        self.assertEqual(len(progress), 1)
        p = progress[0]
        # user_facing_text has no SSML tags
        self.assertEqual(p.accumulated_text, "4111")
        self.assertEqual(p.remaining_text, " 1111 1111 1111")

    async def test_french_space_before_question_mark_word_by_word(self):
        """Trailing space-separated punctuation ("Comment ça va ?") completes the slot.

        French typography sets terminal ``?`` off from its word with a space, and
        the TTS emits that ``?`` as its own word-timestamp event. The slot must
        stay open until the ``?`` arrives, and the final progress frame must carry
        the full sentence as ``accumulated_text`` with an empty ``remaining_text``
        -- otherwise RTVI clients committing captions on the completed sentence
        would drop the ``?``.
        """
        seq = _seq()
        frame = _spoken_frame("Comment ça va ?")
        await seq.register_spoken(frame, "ctx1", "Comment ça va ?", append_to_context=True)

        steps = [
            ("Comment", "Comment", " ça va ?"),
            ("ça", "Comment ça", " va ?"),
            ("va", "Comment ça va", " ?"),
            ("?", "Comment ça va ?", ""),
        ]
        for word, exp_acc, exp_rem in steps:
            result = seq.process_word(word, pts=10, context_id="ctx1")
            word_frames = [f for f in result if isinstance(f, TTSTextFrame)]
            self.assertEqual([f.text for f in word_frames], [word])
            progress = [f for f in result if isinstance(f, AggregatedTextProgressFrame)]
            self.assertEqual(len(progress), 1, f"expected 1 progress frame after '{word}'")
            self.assertEqual(progress[0].accumulated_text, exp_acc)
            self.assertEqual(progress[0].remaining_text, exp_rem)

        # The "va" (last alnum) word must NOT complete the slot; only "?" does.
        self.assertEqual(seq._slots, [])

    async def test_french_same_sentence_twice_word_by_word(self):
        """Two consecutive identical French sentences each complete on their own ``?``.

        Reproduces the double ``TTSSpeakFrame("Comment ça va ?")`` scenario: both
        sentences occupy their own slot/context and stream in word by word. Each
        must complete only when its trailing ``?`` arrives, and every progress
        frame must be attributed to the sentence's own context -- the second
        sentence's words must not leak into the first slot or vice versa.
        """
        seq = _seq()
        sentence = "Comment ça va ?"
        await seq.register_spoken(_spoken_frame(sentence), "ctx1", sentence, True)
        await seq.register_spoken(_spoken_frame(sentence), "ctx2", sentence, True)

        steps = [
            ("Comment", "Comment", " ça va ?"),
            ("ça", "Comment ça", " va ?"),
            ("va", "Comment ça va", " ?"),
            ("?", "Comment ça va ?", ""),
        ]

        # First sentence: consumes ctx1, ctx2 stays queued behind it.
        for word, exp_acc, exp_rem in steps:
            result = seq.process_word(word, pts=10, context_id="ctx1")
            word_frames = [f for f in result if isinstance(f, TTSTextFrame)]
            self.assertEqual([f.text for f in word_frames], [word])
            self.assertTrue(all(f.context_id == "ctx1" for f in word_frames))
            progress = [f for f in result if isinstance(f, AggregatedTextProgressFrame)]
            self.assertEqual(len(progress), 1)
            self.assertEqual(progress[0].context_id, "ctx1")
            self.assertEqual(progress[0].accumulated_text, exp_acc)
            self.assertEqual(progress[0].remaining_text, exp_rem)

        # ctx1 is complete and swept; ctx2 is now the active slot.
        self.assertEqual(len(seq._slots), 1)

        # Second sentence: consumes ctx2 and empties the queue.
        for word, exp_acc, exp_rem in steps:
            result = seq.process_word(word, pts=20, context_id="ctx2")
            word_frames = [f for f in result if isinstance(f, TTSTextFrame)]
            self.assertEqual([f.text for f in word_frames], [word])
            self.assertTrue(all(f.context_id == "ctx2" for f in word_frames))
            progress = [f for f in result if isinstance(f, AggregatedTextProgressFrame)]
            self.assertEqual(len(progress), 1)
            self.assertEqual(progress[0].context_id, "ctx2")
            self.assertEqual(progress[0].accumulated_text, exp_acc)
            self.assertEqual(progress[0].remaining_text, exp_rem)

        self.assertEqual(seq._slots, [])

    async def test_card_scenario_word_by_word(self):
        """Progress accumulated/remaining track user_facing_text through all four digit groups."""
        seq = _seq()
        frame = _spoken_frame("4111 1111 1111 1111", raw_text="<card>4111 1111 1111 1111</card>")
        await seq.register_spoken(
            frame, "ctx1", "<spell>4111 1111 1111 1111</spell>", append_to_context=True
        )

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


class TestCJKProcessWordFlagPropagation(unittest.IsolatedAsyncioTestCase):
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

    async def test_process_word_flag_reaches_frame_when_slot_has_no_flag(self):
        """includes_inter_frame_spaces=True on process_word must stamp the emitted frame.

        register_spoken is called without includes_inter_frame_spaces (simulating
        tts_service.py), then process_word is called with includes_inter_frame_spaces=True
        (simulating add_word_timestamps for ElevenLabs CJK).  The frame must carry
        includes_inter_frame_spaces=True.
        """
        seq = _seq()
        sentence = "どんなことでも気軽に話しかけてくださいね。"
        # tts_service.py does NOT pass includes_inter_frame_spaces to register_spoken
        await seq.register_spoken(_spoken_frame(sentence), "ctx1", sentence, True)

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

    async def test_cjk_two_chunks_no_space_when_slot_has_no_flag(self):
        """Two CJK chunks must concatenate without a space when process_word carries the flag.

        Matches the ElevenLabs runtime: register_spoken gets no flag; both
        process_word calls get includes_inter_frame_spaces=True.  Context assembly
        must produce '気軽に' not '気 軽に'.
        """
        seq = _seq()
        sentence = "どんなことでも気軽に話しかけてくださいね。"
        await seq.register_spoken(_spoken_frame(sentence), "ctx1", sentence, True)

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

    async def test_force_complete_cjk_flag_when_slot_has_no_flag(self):
        """force_complete must also carry the flag for CJK slots registered without it.

        When TTS drops the final token, force_complete emits the remainder.  The
        flag must still reach that frame so the context assembler doesn't add a space.
        """
        seq = _seq()
        sentence = "どんなことでも気軽に話しかけてくださいね。"
        await seq.register_spoken(_spoken_frame(sentence), "ctx1", sentence, True)

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


# ---------------------------------------------------------------------------
# Voice-formatting: sequencer behaviour when transforms change alnum content
#
# Simulates the billing message scenario:
#   user_facing: "Your balance is $5, due on 3/15. Call us at 555-1234."
#   tts (post-transform): "Your balance is five dollars, due on 3/15. Call us at 555-1234."
#
# Expected sequencer behaviour:
#   - "five"    → append_to_context=False (mid transformed segment, suppress)
#   - "dollars," → append_to_context=True (segment completes, raw_text="$5,")
#   - All other words → append_to_context=True (unchanged segments)
# ---------------------------------------------------------------------------

_BILL_UF = "Your balance is $5, due on 3/15. Call us at 555-1234."
_BILL_TTS = "Your balance is five dollars, due on 3/15. Call us at 555-1234."
_BILL_WORDS = [
    "Your",
    "balance",
    "is",
    "five",
    "dollars,",
    "due",
    "on",
    "3/15.",
    "Call",
    "us",
    "at",
    "555-1234.",
]


class TestVoiceFormattingTransforms(unittest.IsolatedAsyncioTestCase):
    """Sequencer correctly handles transform-aware trackers for billing messages."""

    async def _setup(self):
        seq = AggregatedFrameSequencer(name="test-billing")
        source = AggregatedTextFrame(_BILL_UF, AggregationType.SENTENCE, raw_text=_BILL_UF)
        await seq.register_spoken(source, "ctx1", _BILL_TTS, append_to_context=True)
        return seq, source

    def _advance(self, seq, *words):
        for word in words:
            seq.process_word(word, pts=10, context_id="ctx1")

    def _word_frames(self, frames):
        return [f for f in frames if isinstance(f, TTSTextFrame)]

    def _progress_frames(self, frames):
        return [f for f in frames if isinstance(f, AggregatedTextProgressFrame)]

    # --- append_to_context ---

    async def test_pre_transform_words_append_to_context(self):
        seq, _ = await self._setup()
        for word in ("Your", "balance", "is"):
            result = seq.process_word(word, pts=10, context_id="ctx1")
            wf = self._word_frames(result)
            self.assertTrue(wf[0].append_to_context, f"word '{word}' should append to context")

    async def test_mid_transform_word_suppressed(self):
        """`five` is mid-segment: append_to_context must be False."""
        seq, _ = await self._setup()
        self._advance(seq, "Your", "balance", "is")
        result = seq.process_word("five", pts=20, context_id="ctx1")
        wf = self._word_frames(result)
        self.assertEqual(len(wf), 1)
        self.assertFalse(wf[0].append_to_context)

    async def test_completing_transform_word_appends_to_context(self):
        """`dollars,` completes the segment: append_to_context must be True."""
        seq, _ = await self._setup()
        self._advance(seq, "Your", "balance", "is", "five")
        result = seq.process_word("dollars,", pts=30, context_id="ctx1")
        wf = self._word_frames(result)
        self.assertEqual(len(wf), 1)
        self.assertTrue(wf[0].append_to_context)

    async def test_post_transform_words_append_to_context(self):
        seq, _ = await self._setup()
        self._advance(seq, "Your", "balance", "is", "five", "dollars,")
        result = seq.process_word("due", pts=40, context_id="ctx1")
        wf = self._word_frames(result)
        self.assertTrue(wf[0].append_to_context)

    # --- raw_text / llm_consumed ---

    async def test_mid_transform_word_raw_text_none(self):
        """`five` is mid-segment: raw_text must be None so it is not written to context."""
        seq, _ = await self._setup()
        self._advance(seq, "Your", "balance", "is")
        result = seq.process_word("five", pts=20, context_id="ctx1")
        wf = self._word_frames(result)
        self.assertIsNone(wf[0].raw_text)

    async def test_completing_transform_word_raw_text_is_original(self):
        """`dollars,` completing the segment must carry `$5,` as raw_text."""
        seq, _ = await self._setup()
        self._advance(seq, "Your", "balance", "is", "five")
        result = seq.process_word("dollars,", pts=30, context_id="ctx1")
        wf = self._word_frames(result)
        self.assertEqual(wf[0].raw_text, "$5,")

    # --- AggregatedTextProgressFrame ---

    async def test_no_progress_frame_emitted_mid_transform(self):
        """No AggregatedTextProgressFrame is emitted for mid-segment words ('five').

        The user_facing_pos is held during a transformed segment, so emitting a
        progress frame with identical accumulated/remaining text would be redundant.
        """
        seq, _ = await self._setup()
        self._advance(seq, "Your", "balance", "is")
        result = seq.process_word("five", pts=20, context_id="ctx1")
        self.assertEqual(self._progress_frames(result), [])

    async def test_progress_accumulated_jumps_after_transform_completes(self):
        """After 'dollars,' completes, accumulated_text must include 'Your balance is $5,'."""
        seq, _ = await self._setup()
        self._advance(seq, "Your", "balance", "is", "five")
        result = seq.process_word("dollars,", pts=30, context_id="ctx1")
        pf = self._progress_frames(result)
        self.assertEqual(len(pf), 1)
        self.assertEqual(pf[0].accumulated_text, "Your balance is $5,")

    async def test_progress_remaining_after_transform_completes(self):
        """After 'dollars,', remaining_text starts with 'due on 3/15.'"""
        seq, _ = await self._setup()
        self._advance(seq, "Your", "balance", "is", "five")
        result = seq.process_word("dollars,", pts=30, context_id="ctx1")
        pf = self._progress_frames(result)
        self.assertTrue(pf[0].remaining_text.startswith(" due on 3/15."), pf[0].remaining_text)

    async def test_progress_full_sentence_completion(self):
        """After all words the slot is complete and the queue is empty."""
        seq, _ = await self._setup()
        for word in _BILL_WORDS:
            seq.process_word(word, pts=10, context_id="ctx1")
        self.assertEqual(seq._slots, [])

    # --- force_complete during transform ---

    async def test_force_complete_mid_transform_emits_remaining_tts_text(self):
        """force_complete when mid-segment emits remaining tts_text (not user_facing_text)."""
        seq, _ = await self._setup()
        self._advance(seq, "Your", "balance", "is", "five")
        # "dollars," never arrives — force complete
        result = seq.force_complete(last_word_pts=99)
        wf = self._word_frames(result)
        self.assertEqual(len(wf), 1)
        # Remaining tts_text starts with "dollars," (the unexpanded portion)
        self.assertIn("dollars,", wf[0].text)

    async def test_force_complete_after_transform_emits_remaining_unchanged_text(self):
        """force_complete after the transform emits remaining tts_text correctly."""
        seq, _ = await self._setup()
        self._advance(seq, "Your", "balance", "is", "five", "dollars,", "due", "on")
        result = seq.force_complete(last_word_pts=99)
        wf = self._word_frames(result)
        self.assertEqual(len(wf), 1)
        self.assertIn("3/15.", wf[0].text)
        self.assertIn("555-1234.", wf[0].text)


# ---------------------------------------------------------------------------
# register_spoken — streaming (TOKEN mode): token-by-token sentence accumulation
#
# In streaming mode the sequencer owns a _ParallelSentenceAggregator;
# register_spoken is called once per token and only registers a real slot once a
# sentence boundary is confirmed (which happens when the NEXT sentence's first
# token arrives, or via finalize() at end of turn).
# ---------------------------------------------------------------------------


async def _stream(seq, ctx, *tokens):
    """Feed same-text tokens through streaming register_spoken; return pushed frames."""
    frames = []
    for t in tokens:
        frames += await seq.register_spoken(_spoken_frame(t), ctx, t, append_to_context=True)
    return frames


class TestRegisterSpokenStreaming(unittest.IsolatedAsyncioTestCase):
    async def test_non_terminal_tokens_do_not_promote(self):
        seq = _seq(streaming=True)
        await _stream(seq, "ctx1", "Hi", " there")
        self.assertEqual(seq._slots, [])

    async def test_terminal_token_alone_does_not_promote(self):
        # Needs the next sentence's lookahead token before the boundary confirms.
        seq = _seq(streaming=True)
        await _stream(seq, "ctx1", "Hi", " there", "!")
        self.assertEqual(seq._slots, [])

    async def test_next_sentence_token_promotes_clean_first_sentence(self):
        seq = _seq(streaming=True)
        # The lookahead token " How" confirms "Hi there!" but is NOT folded in.
        await _stream(seq, "ctx1", "Hi", " there", "!", " How")
        self.assertEqual(len(seq._slots), 1)
        slot = seq._slots[0]
        self.assertIsNotNone(slot.tracker)
        self.assertEqual(slot.frame.aggregated_by, AggregationType.SENTENCE)
        self.assertEqual(slot.frame.text, "Hi there!")

    async def test_promoted_slot_processes_words_normally(self):
        seq = _seq(streaming=True)
        await _stream(seq, "ctx1", "Hi", " there", "!", " How")  # promotes "Hi there!"
        result = seq.process_word("Hi", pts=10, context_id="ctx1")
        result += seq.process_word("there!", pts=20, context_id="ctx1")
        word_frames = [f for f in result if isinstance(f, TTSTextFrame)]
        progress = [f for f in result if isinstance(f, AggregatedTextProgressFrame)]
        self.assertEqual([f.text for f in word_frames], ["Hi", "there!"])
        self.assertEqual(progress[-1].accumulated_text, "Hi there!")
        self.assertEqual(progress[-1].remaining_text, "")

    async def test_slot_ifs_not_set_from_streaming_path(self):
        # The spacing bug: LLM tokens carry includes_inter_frame_spaces=True, but
        # that must NOT propagate to the promoted slot / word frames — English
        # word timestamps must still be joined with spaces in the context.
        seq = _seq(streaming=True)
        await _stream(seq, "ctx1", "Hi", " there", "!", " How")
        self.assertFalse(seq._slots[0].includes_inter_frame_spaces)
        r1 = seq.process_word("Hi", pts=10, context_id="ctx1")
        r2 = seq.process_word("there!", pts=20, context_id="ctx1")
        parts = [
            TextPartForConcatenation(
                f.text, includes_inter_part_spaces=f.includes_inter_frame_spaces
            )
            for f in r1 + r2
            if isinstance(f, TTSTextFrame)
        ]
        self.assertEqual(concatenate_aggregated_text(parts), "Hi there!")

    async def test_multi_sentence_promotes_each_cleanly(self):
        seq = _seq(streaming=True)
        # Two sentences streamed; the first promotes when the second starts.
        await _stream(seq, "ctx1", "Hi", " there", "!", " How", " are", " you", "?")
        self.assertEqual([s.frame.text for s in seq._slots], ["Hi there!"])
        # The trailing sentence promotes on finalize.
        await seq.finalize()
        self.assertEqual([s.frame.text for s in seq._slots], ["Hi there!", " How are you?"])

    async def test_transformed_tts_text_preserved_through_promotion(self):
        seq = _seq(streaming=True)
        # tts differs from user-facing (a simulated transform): "$5" -> "five dollars".
        for tts, txt in [("five dollars", "$5"), (".", "."), (" Ok", " Ok")]:
            await seq.register_spoken(_spoken_frame(txt), "ctx1", tts, append_to_context=True)
        self.assertEqual(len(seq._slots), 1)
        slot = seq._slots[0]
        self.assertEqual(slot.frame.text, "$5.")  # user-facing unaffected
        result = seq.process_word("five", pts=10, context_id="ctx1")
        word_frames = [f for f in result if isinstance(f, TTSTextFrame)]
        self.assertEqual(word_frames[0].text, "five")

    async def test_no_tracker_registers_each_token_immediately(self):
        # streaming + build_tracker=False (push_text_frames=True): per-token slots.
        seq = _seq(streaming=True)
        await seq.register_spoken(
            _spoken_frame("Hi"), "ctx1", "Hi", append_to_context=True, build_tracker=False
        )
        self.assertEqual(len(seq._slots), 1)
        self.assertIsNone(seq._slots[0].tracker)
        await seq.register_spoken(
            _spoken_frame(" there"), "ctx1", " there", append_to_context=True, build_tracker=False
        )
        self.assertEqual(len(seq._slots), 2)


# ---------------------------------------------------------------------------
# register_spoken — streaming: words buffered until a pending sentence promotes
# ---------------------------------------------------------------------------


class TestRegisterSpokenBufferedWords(unittest.IsolatedAsyncioTestCase):
    async def test_word_for_pending_sentence_is_buffered(self):
        seq = _seq(streaming=True)
        await _stream(seq, "ctx1", "Hi", " there")  # nothing promoted yet
        result = seq.process_word("Hi", pts=10, context_id="ctx1")
        self.assertEqual(result, [])
        self.assertEqual(len(seq._buffered_words), 1)

    async def test_buffered_word_replayed_once_boundary_confirmed(self):
        seq = _seq(streaming=True)
        await _stream(seq, "ctx1", "Hi", " there", "!")  # pending, not promoted
        # Word arrives before promotion -> buffered.
        self.assertEqual(seq.process_word("Hi", pts=10, context_id="ctx1"), [])
        # The next sentence's token promotes "Hi there!"; buffered word replays.
        result = await _stream(seq, "ctx1", " How")
        word_frames = [f for f in result if isinstance(f, TTSTextFrame)]
        self.assertEqual([f.text for f in word_frames], ["Hi"])

    async def test_word_still_unmatched_after_one_promotion_is_rebuffered(self):
        seq = _seq(streaming=True)
        await _stream(seq, "ctx1", "Hi", " there", "!", " How")  # promotes "Hi there!"
        self.assertEqual(len(seq._slots), 1)
        # A word for the second (not-yet-promoted) sentence is buffered.
        self.assertEqual(seq.process_word("How", pts=5, context_id="ctx1"), [])
        self.assertEqual(len(seq._buffered_words), 1)
        # First sentence completes via its own words.
        seq.process_word("Hi", pts=10, context_id="ctx1")
        seq.process_word("there!", pts=20, context_id="ctx1")
        self.assertEqual(seq._slots, [])
        # Second sentence promotes; the buffered "How" now matches.
        result = await seq.finalize()
        word_frames = [f for f in result if isinstance(f, TTSTextFrame)]
        self.assertTrue(any(f.text == "How" for f in word_frames))

    async def test_non_streaming_sequencer_keeps_passthrough_path(self):
        seq = _seq(streaming=False)
        await seq.register_spoken(_spoken_frame("hello world"), "ctx1", "hello world", True)
        result = seq.process_word("zzz", pts=5, context_id="ctx1")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, "zzz")
        self.assertEqual(seq._buffered_words, [])


# ---------------------------------------------------------------------------
# register_skipped — finalizes a pending streamed sentence first
# ---------------------------------------------------------------------------


class TestRegisterSkippedForcesFinalize(unittest.IsolatedAsyncioTestCase):
    async def test_pending_sentence_promoted_before_skipped_slot(self):
        seq = _seq(streaming=True)
        await _stream(seq, "ctx1", "Hi", " there")  # pending, no boundary
        skipped = _skipped_frame("code")
        await seq.register_skipped(skipped, "ctx2", None)
        self.assertEqual(len(seq._slots), 2)
        self.assertTrue(seq._slots[0].spoken)
        self.assertEqual(seq._slots[0].frame.text, "Hi there")
        self.assertFalse(seq._slots[1].spoken)
        self.assertIs(seq._slots[1].frame, skipped)

    async def test_register_skipped_with_nothing_pending_behaves_as_today(self):
        seq = _seq(streaming=True)
        frame = _skipped_frame("code")
        result = await seq.register_skipped(frame, "ctx1", None)
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], frame)

    async def test_skipped_frame_stays_blocked_until_finalized_sentence_completes(self):
        seq = _seq(streaming=True)
        await _stream(seq, "ctx1", "Hi", " there")
        skipped = _skipped_frame("code")
        result = await seq.register_skipped(skipped, "ctx1", None)
        # The forced finalize emits the sentence frame, but the skipped frame
        # stays blocked behind the not-yet-spoken sentence slot.
        self.assertFalse(any(f is skipped for f in result))
        result = seq.process_word("Hi", pts=10, context_id="ctx1")
        result += seq.process_word("there", pts=20, context_id="ctx1")
        self.assertTrue(any(f is skipped for f in result))


# ---------------------------------------------------------------------------
# finalize — end-of-turn forced promotion
# ---------------------------------------------------------------------------


class TestFinalizeEndOfTurn(unittest.IsolatedAsyncioTestCase):
    async def test_finalize_promotes_pending_sentence_with_no_terminal_punctuation(self):
        seq = _seq(streaming=True)
        await _stream(seq, "ctx1", "Hi", " there")
        self.assertEqual(seq._slots, [])
        result = await seq.finalize()
        # The promoted sentence frame is emitted (the "new" announcement).
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], AggregatedTextFrame)
        self.assertEqual(result[0].text, "Hi there")
        self.assertTrue(result[0].will_be_spoken)
        self.assertEqual(len(seq._slots), 1)
        self.assertEqual(seq._slots[0].frame.text, "Hi there")

    async def test_finalize_with_nothing_pending_is_a_noop(self):
        seq = _seq(streaming=True)
        self.assertEqual(await seq.finalize(), [])
        self.assertEqual(seq._slots, [])

    async def test_finalize_on_non_streaming_is_a_noop(self):
        seq = _seq(streaming=False)
        self.assertEqual(await seq.finalize(), [])

    async def test_finalize_does_not_create_slot_for_whitespace_only_pending(self):
        seq = _seq(streaming=True)
        await _stream(seq, "ctx1", "   ")
        result = await seq.finalize()
        self.assertEqual(result, [])
        self.assertEqual(seq._slots, [])

    async def test_finalize_then_processing_words_drains_the_slot(self):
        seq = _seq(streaming=True)
        await _stream(seq, "ctx1", "Hi", " there")
        await seq.finalize()
        seq.process_word("Hi", pts=10, context_id="ctx1")
        seq.process_word("there", pts=20, context_id="ctx1")
        self.assertEqual(seq._slots, [])


# ---------------------------------------------------------------------------
# clear — resets streaming-specific state
# ---------------------------------------------------------------------------


class TestClearResetsStreamingState(unittest.IsolatedAsyncioTestCase):
    async def test_clear_empties_pending_aggregator(self):
        seq = _seq(streaming=True)
        await _stream(seq, "ctx1", "Hi", " there")
        seq.clear()
        # A fresh aggregator: finalize now yields nothing.
        self.assertEqual(await seq.finalize(), [])
        self.assertIsNone(seq._streaming_slot_meta)

    async def test_clear_empties_buffered_words(self):
        seq = _seq(streaming=True)
        await _stream(seq, "ctx1", "Hi", " there")
        seq.process_word("Hi", pts=10, context_id="ctx1")  # buffered
        self.assertEqual(len(seq._buffered_words), 1)
        seq.clear()
        self.assertEqual(seq._buffered_words, [])

    async def test_sequencer_behaves_fresh_after_clear(self):
        seq = _seq(streaming=True)
        await _stream(seq, "ctx1", "Hi")
        seq.clear()
        await _stream(seq, "ctx1", "Bye", "!", " Ok")
        self.assertEqual(len(seq._slots), 1)
        self.assertEqual(seq._slots[0].frame.text, "Bye!")


# ---------------------------------------------------------------------------
# _ParallelSentenceAggregator — the sequencer's internal token→sentence grouper
#
# Emits a sentence only once the NEXT sentence's first token supplies the
# lookahead (no over-grouping), and keeps the three channels token-aligned.
# ---------------------------------------------------------------------------


async def _agg_feed(agg, *tokens):
    """Feed same-text tokens through the aggregator, collecting emitted sentences.

    Each token is used for all three channels. Returns the list of
    (tts, llm, user) tuples emitted across the whole run.
    """
    out = []
    for t in tokens:
        async for s in agg.aggregate(t, t, t):
            out.append((s.tts_text, s.llm_text, s.user_facing_text))
    return out


class TestParallelSentenceAggregator(unittest.IsolatedAsyncioTestCase):
    async def test_no_boundary_yields_nothing(self):
        agg = _ParallelSentenceAggregator()
        self.assertEqual(await _agg_feed(agg, "Hi", " there"), [])

    async def test_terminal_token_alone_does_not_emit(self):
        # A sentence-ending token needs lookahead (the next sentence's first
        # token) before it is confirmed, so "!" alone emits nothing.
        agg = _ParallelSentenceAggregator()
        self.assertEqual(await _agg_feed(agg, "Hi", " there", "!"), [])

    async def test_next_sentence_token_confirms_first_sentence_only(self):
        # The lookahead token (" How") confirms "Hi there!" — but must NOT be
        # folded into it (that was the over-grouping bug that broke progress).
        agg = _ParallelSentenceAggregator()
        out = await _agg_feed(agg, "Hi", " there", "!", " How")
        self.assertEqual([e[0] for e in out], ["Hi there!"])

    async def test_multi_sentence_stream_yields_each_cleanly(self):
        agg = _ParallelSentenceAggregator()
        emitted = await _agg_feed(agg, "Hi", " there", "!", " How", " are", " you", "?")
        # Only the first sentence is confirmed mid-stream; the second waits.
        self.assertEqual([e[0] for e in emitted], ["Hi there!"])
        # The trailing sentence comes out on flush.
        f = await agg.flush()
        self.assertIsNotNone(f)
        self.assertEqual(f.tts_text, " How are you?")

    async def test_flush_emits_trailing_partial_sentence(self):
        agg = _ParallelSentenceAggregator()
        await _agg_feed(agg, "Just", " a", " fragment")
        f = await agg.flush()
        self.assertIsNotNone(f)
        self.assertEqual(f.user_facing_text, "Just a fragment")

    async def test_flush_returns_none_when_empty(self):
        self.assertIsNone(await _ParallelSentenceAggregator().flush())

    async def test_flush_returns_none_for_whitespace_only(self):
        agg = _ParallelSentenceAggregator()
        await _agg_feed(agg, "   ")
        self.assertIsNone(await agg.flush())

    async def test_handle_interruption_resets(self):
        agg = _ParallelSentenceAggregator()
        await _agg_feed(agg, "Hi", " there")
        await agg.handle_interruption()
        self.assertIsNone(await agg.flush())
        # Behaves fresh afterwards.
        out = await _agg_feed(agg, "Bye", "!", " Next")
        self.assertEqual([e[0] for e in out], ["Bye!"])

    async def test_channels_stay_token_aligned_under_transform(self):
        # tts_text differs from user-facing/llm (a simulated transform), but
        # because emission is token-aligned (whole tokens only), the three
        # channels correspond to the same sentence span.
        agg = _ParallelSentenceAggregator()
        out = []
        # (tts, llm, user) per token: "$5" is spoken as "five dollars".
        triples = [
            ("five dollars", "$5", "$5"),
            (".", ".", "."),
            (" Thanks", " Thanks", " Thanks"),
        ]
        for tts, llm, user in triples:
            async for s in agg.aggregate(tts, llm, user):
                out.append((s.tts_text, s.llm_text, s.user_facing_text))
        self.assertEqual(out, [("five dollars.", "$5.", "$5.")])

    async def test_coarse_chunk_straddling_boundary_splits_inside_token(self):
        # A coarse chunk carries the tail of one sentence and the head of the
        # next ("Hey" then " there! I'm ..."). The boundary must be sliced
        # inside the chunk so "Hey there!" is emitted whole — not split into a
        # bare "Hey" with " there!" wrongly carried into the next sentence.
        agg = _ParallelSentenceAggregator()
        out = await _agg_feed(agg, "Hey", " there! I'm your friendly assistant, here to")
        self.assertEqual([e[0] for e in out], ["Hey there!"])
        # The remainder (post-boundary head of the chunk) stays buffered.
        f = await agg.flush()
        self.assertIsNotNone(f)
        self.assertEqual(f.tts_text, " I'm your friendly assistant, here to")

    async def test_single_chunk_with_two_boundaries_splits_both(self):
        # One chunk containing two full sentence endings plus the lookahead of a
        # third confirms both completed sentences at once, sliced cleanly.
        agg = _ParallelSentenceAggregator()
        out = await _agg_feed(agg, "One. Two. Three")
        self.assertEqual([e[0] for e in out], ["One.", " Two."])
        f = await agg.flush()
        self.assertIsNotNone(f)
        self.assertEqual(f.tts_text, " Three")

    async def test_straddling_split_keeps_all_three_channels_equal(self):
        # On the identical-channel path the sliced sentence is reported the same
        # in every channel, so downstream user/llm/tts stay in agreement.
        agg = _ParallelSentenceAggregator()
        out = []
        async for s in agg.aggregate("Hi", "Hi", "Hi"):
            out.append(s)
        async for s in agg.aggregate(" there! Next", " there! Next", " there! Next"):
            out.append(s)
        self.assertEqual(len(out), 1)
        self.assertEqual(
            (out[0].tts_text, out[0].llm_text, out[0].user_facing_text),
            ("Hi there!", "Hi there!", "Hi there!"),
        )

    async def test_divergent_channel_straddle_falls_back_to_token_boundary(self):
        # When a transform has diverged the channels, a straddling chunk cannot be
        # split at a shared offset, so emission falls back to the token boundary:
        # the text accumulated before the chunk is emitted whole.
        agg = _ParallelSentenceAggregator()
        out = []
        triples = [
            ("five dollars", "$5", "$5"),
            (" spent! More", " spent! More", " spent! More"),
        ]
        for tts, llm, user in triples:
            async for s in agg.aggregate(tts, llm, user):
                out.append((s.tts_text, s.llm_text, s.user_facing_text))
        self.assertEqual(out, [("five dollars", "$5", "$5")])

    async def test_divergent_cut_does_not_poison_later_sentences(self):
        # After a divergent-channel token-boundary cut whose triggering token
        # straddled a boundary, the inner aggregator has consumed past that boundary
        # while the whole token stays buffered here. Alignment must NOT be falsely
        # restored: a later identical-channel chunk must still be grouped correctly
        # instead of being sliced against a stale buffer (which used to emit a
        # spurious partial-sentence anchor and drop a real sentence).
        agg = _ParallelSentenceAggregator()
        out = []
        # "$5" is spoken as "five dollars" (length-changing transform diverges the
        # channels); " spent! More" then straddles a sentence boundary.
        triples = [
            ("five dollars", "$5", "$5"),
            (" spent! More", " spent! More", " spent! More"),
            (" stuff.", " stuff.", " stuff."),
            (" X", " X", " X"),
        ]
        for tts, llm, user in triples:
            async for s in agg.aggregate(tts, llm, user):
                out.append((s.tts_text, s.llm_text, s.user_facing_text))
        # Token-boundary cuts only — no spurious " spent!" anchor, and " More stuff."
        # is not lost (it rides out with the second emitted unit).
        self.assertEqual(
            [e[0] for e in out],
            ["five dollars", " spent! More stuff."],
        )

    async def test_alignment_recovers_after_divergent_cut(self):
        # Once a divergent run resolves back to identical channels at a token edge,
        # the fast path must re-engage so a later straddling chunk is again sliced
        # inside the token rather than cut coarsely at the token boundary.
        agg = _ParallelSentenceAggregator()
        out = []
        triples = [
            ("five dollars", "$5", "$5"),  # diverge the channels
            (" spent!", " spent!", " spent!"),  # completes the diverged sentence
            (" Buy", " Buy", " Buy"),  # confirms the cut; buffer realigns to " Buy"
            (" more! Now", " more! Now", " more! Now"),  # fast path slices inside token
        ]
        for tts, llm, user in triples:
            async for s in agg.aggregate(tts, llm, user):
                out.append((s.tts_text, s.llm_text, s.user_facing_text))
        self.assertEqual(
            [e[0] for e in out],
            ["five dollars spent!", " Buy more!"],
        )


if __name__ == "__main__":
    unittest.main()
