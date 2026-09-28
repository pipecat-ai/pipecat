#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.frames.frames import (
    AggregatedTextFrame,
    AggregatedTextProgressFrame,
    AggregationType,
    TTSTextFrame,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.soniox.tts import SonioxTTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.context.aggregated_frame_sequencer import AggregatedFrameSequencer
from pipecat.utils.context.word_completion_tracker import WordCompletionTracker
from pipecat.utils.string import TextPartForConcatenation, concatenate_aggregated_text


def _service(language: str | None) -> SonioxTTSService:
    service = SonioxTTSService.__new__(SonioxTTSService)
    service._name = "SonioxTTSService#0"
    service._settings = TTSSettings(language=language)
    service._partials = {}
    return service


def _timestamps(text: str, start: float = 0.0) -> dict:
    chars = list(text)
    return {
        "characters": chars,
        "character_start_times_seconds": [round(start + i * 0.1, 1) for i in range(len(chars))],
    }


def test_soniox_english_characters_assemble_into_words():
    service = _service("en")

    assert service._to_word_times("s1", _timestamps("Hi world. ")) == [
        ("Hi", 0.0),
        ("world.", 0.3),
    ]


def test_soniox_english_partial_word_carries_across_messages():
    service = _service("en")

    assert service._to_word_times("s1", _timestamps("Hel")) == []
    assert service._to_word_times("s1", _timestamps("lo you ", start=0.3)) == [
        ("Hello", 0.0),
        ("you", 0.6),
    ]


def test_soniox_english_final_partial_word_is_buffered_for_terminated():
    service = _service("en")

    assert service._to_word_times("s1", _timestamps("Hi you")) == [("Hi", 0.0)]
    # The receive loop flushes this buffered word when Soniox sends `terminated`.
    assert service._partials["s1"] == ("you", 0.3)


def test_soniox_streams_buffer_partial_words_independently():
    service = _service("en")

    service._to_word_times("s1", _timestamps("Hel"))
    service._to_word_times("s2", _timestamps("wor", start=1.0))

    assert service._partials["s1"] == ("Hel", 0.0)
    assert service._partials["s2"] == ("wor", 1.0)

    assert service._to_word_times("s1", _timestamps("lo ", start=0.3)) == [("Hello", 0.0)]
    assert service._to_word_times("s2", _timestamps("ld ", start=1.3)) == [("world", 1.0)]


def test_soniox_timestamp_length_mismatch_returns_empty():
    service = _service("en")

    assert (
        service._to_word_times(
            "s1", {"characters": ["H", "i"], "character_start_times_seconds": [0.0]}
        )
        == []
    )
    assert "s1" not in service._partials


def test_soniox_japanese_timestamps_emit_per_character():
    service = _service("ja")

    assert service._to_word_times("s1", _timestamps("こんにちは、私")) == [
        ("こ", 0.0),
        ("ん", 0.1),
        ("に", 0.2),
        ("ち", 0.3),
        ("は", 0.4),
        ("私", 0.6),
    ]
    assert "s1" not in service._partials


def test_soniox_chinese_timestamps_emit_per_character():
    service = _service("zh")

    assert service._to_word_times("s1", _timestamps("你好，世界。")) == [
        ("你", 0.0),
        ("好", 0.1),
        ("世", 0.3),
        ("界", 0.4),
    ]


def test_soniox_japanese_tokens_concatenate_without_spaces():
    service = _service("ja")
    tokens = service._to_word_times("s1", _timestamps("こんにちは、私"))

    includes_inter_frame_spaces = service._is_chinese_or_japanese_language()
    assert (
        concatenate_aggregated_text(
            [
                TextPartForConcatenation(
                    word, includes_inter_part_spaces=includes_inter_frame_spaces
                )
                for word, _start in tokens
            ]
        )
        == "こんにちは私"
    )


def test_soniox_japanese_punctuation_recovered_by_word_tracker():
    # CJK tokens drop punctuation (isalnum filter), which is safe: the frame
    # sequencer's WordCompletionTracker matches tokens by alphanumeric content
    # only and commits spans of the original text, sweeping adjacent punctuation
    # into each consumed span.
    text = "こんにちは、私はAIです。"
    service = _service("ja")
    tokens = service._to_word_times("s1", _timestamps(text))

    tracker = WordCompletionTracker(text)
    complete = False
    for word, _start in tokens:
        complete = tracker.add_word_and_check_complete(word)

    assert complete
    assert tracker.get_accumulated_user_facing_text() == text


class TestSonioxUpdateSettingsFinalizesOldContext(unittest.IsolatedAsyncioTestCase):
    """A mid-reply voice/model/language/speed change re-mints the turn context. The
    old context's still-pending sentence must be finalized first, or the
    already-heard prefix's word-timestamps land on no slot and drop from the
    transcript.
    """

    async def _service_with_pending_prefix(self, old_ctx: str):
        service = SonioxTTSService.__new__(SonioxTTSService)
        service._name = "SonioxTTSService#0"
        service._settings = SonioxTTSService.Settings(
            model="tts-rt-v2",
            voice="Adrian",
            language=Language.EN,
            speed=None,
        )
        # Applying a settings delta reports the service usable again.
        service._is_usable = True
        # Real streaming sequencer with a mid-sentence prefix pending on the turn ctx.
        seq = AggregatedFrameSequencer(name=service._name, streaming=True)
        service._aggregated_frame_sequencer = seq
        service._turn_context_id = old_ctx
        for token in ("Hi", " there"):
            frame = AggregatedTextFrame(token, AggregationType.SENTENCE, raw_text=token)
            await seq.register_spoken(frame, old_ctx, token, append_to_context=True)
        assert seq._slots == []  # nothing promoted — sentence has no boundary yet

        pushed: list = []

        async def fake_push(frames, context_id):
            pushed.extend(frames)

        async def fake_flush(context_id=None):
            service._flushed = context_id

        service._flushed = None
        service._push_sequencer_frames = fake_push
        service.flush_audio = fake_flush
        service.audio_context_available = lambda context_id: True
        service.create_context_id = lambda: "ctx-new"
        return service, seq, pushed

    async def test_voice_change_finalizes_and_rescues_prefix(self):
        old_ctx = "ctx-old"
        service, seq, pushed = await self._service_with_pending_prefix(old_ctx)

        await service._update_settings(SonioxTTSService.Settings(voice="Emma"))

        # The old context's pending sentence was force-promoted into a real slot.
        self.assertEqual([s.frame.text for s in seq._slots], ["Hi there"])
        self.assertEqual(seq._slots[0].context_id, old_ctx)
        self.assertTrue(
            any(isinstance(f, AggregatedTextFrame) and f.text == "Hi there" for f in pushed)
        )
        self.assertEqual(service._flushed, old_ctx)
        self.assertEqual(service._turn_context_id, "ctx-new")

        # A word-timestamp for the flushed prefix (on the OLD context) still finds
        # the promoted slot and emits a progress frame.
        result = seq.process_word("Hi", pts=10, context_id=old_ctx)
        self.assertTrue(any(isinstance(f, TTSTextFrame) and f.text == "Hi" for f in result))
        progress = [f for f in result if isinstance(f, AggregatedTextProgressFrame)]
        self.assertEqual(len(progress), 1)
        self.assertEqual(progress[0].accumulated_text, "Hi")

    async def test_speed_change_also_finalizes(self):
        # Soniox additionally re-mints on a speed change, so it must finalize too.
        old_ctx = "ctx-old"
        service, seq, pushed = await self._service_with_pending_prefix(old_ctx)

        await service._update_settings(SonioxTTSService.Settings(speed=1.2))

        self.assertEqual([s.frame.text for s in seq._slots], ["Hi there"])
        self.assertEqual(service._turn_context_id, "ctx-new")
