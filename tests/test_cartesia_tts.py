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
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.settings import TTSSettings
from pipecat.transcriptions.language import Language
from pipecat.utils.context.aggregated_frame_sequencer import AggregatedFrameSequencer
from pipecat.utils.context.word_completion_tracker import WordCompletionTracker
from pipecat.utils.string import TextPartForConcatenation, concatenate_aggregated_text


def _service(language: str) -> CartesiaTTSService:
    service = CartesiaTTSService.__new__(CartesiaTTSService)
    service._settings = TTSSettings(language=language)
    return service


def _process_word_timestamps(
    words: list[str], starts: list[float], language: str
) -> list[tuple[str, float]]:
    return _service(language)._normalize_word_timestamps(words, starts)


def _concatenate_processed_timestamps(
    timestamp_groups: list[tuple[list[str], list[float]]], language: str
) -> str:
    service = _service(language)
    text_parts = []
    for words, starts in timestamp_groups:
        processed_timestamps = service._normalize_word_timestamps(words, starts)
        includes_inter_frame_spaces = service._word_timestamps_include_inter_frame_spaces()
        text_parts.extend(
            TextPartForConcatenation(
                word,
                includes_inter_part_spaces=includes_inter_frame_spaces,
            )
            for word, _timestamp in processed_timestamps
        )
    return concatenate_aggregated_text(text_parts)


def test_cartesia_chinese_word_timestamps_join_without_spaces():
    assert _process_word_timestamps(
        words=["你", "好", "。"],
        starts=[0.0, 0.1, 0.2],
        language="zh",
    ) == [("你好。", 0.0)]


def test_cartesia_japanese_word_timestamps_join_without_spaces():
    assert _process_word_timestamps(
        words=["こ", "ん", "に", "ち", "は", "。"],
        starts=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        language="ja",
    ) == [("こんにちは。", 0.0)]


def test_cartesia_korean_word_timestamps_preserve_words_and_timestamps():
    assert _process_word_timestamps(
        words=["안녕하세요", "반갑습니다"],
        starts=[0.0, 0.2],
        language="ko",
    ) == [("안녕하세요", 0.0), ("반갑습니다", 0.2)]


def test_cartesia_korean_word_timestamps_do_not_join_latin_and_hangul():
    assert _process_word_timestamps(
        words=["AI", "어시스턴트입니다."],
        starts=[3.7026982, 4.1999383],
        language="ko",
    ) == [("AI", 3.7026982), ("어시스턴트입니다.", 4.1999383)]


def test_cartesia_japanese_timestamp_groups_reassemble_without_spaces():
    assert (
        _concatenate_processed_timestamps(
            [
                (["こ", "ん", "に", "ち", "は", "、", "私"], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
                (["は", "あ", "な", "た", "の"], [1.0, 1.1, 1.2, 1.3, 1.4]),
            ],
            language="ja",
        )
        == "こんにちは、私はあなたの"
    )


def test_cartesia_chinese_timestamp_groups_reassemble_without_spaces():
    assert (
        _concatenate_processed_timestamps(
            [
                (["你", "好", "，", "我", "是"], [0.1, 0.2, 0.3, 0.4, 0.5]),
                (["你", "的", "智", "能"], [1.0, 1.1, 1.2, 1.3]),
            ],
            language="zh",
        )
        == "你好，我是你的智能"
    )


def test_cartesia_korean_timestamp_groups_reassemble_with_spaces():
    assert (
        _concatenate_processed_timestamps(
            [
                (["저는"], [1.6]),
                (["여러분의"], [1.8]),
                (["AI", "어시스턴트입니다."], [3.7, 4.2]),
            ],
            language="ko",
        )
        == "저는 여러분의 AI 어시스턴트입니다."
    )


def test_cartesia_spell_tag_keeps_its_word_attached_to_following_punctuation():
    assert _process_word_timestamps(
        words=["<spell>1234</spell>."],
        starts=[0.0],
        language="en",
    ) == [("1234.", 0.0)]


def test_cartesia_tag_between_two_words_keeps_them_separated():
    assert _process_word_timestamps(
        words=["to<spell>1234</spell>"],
        starts=[0.0],
        language="en",
    ) == [("to 1234", 0.0)]


def test_cartesia_tag_only_token_is_dropped():
    assert (
        _process_word_timestamps(
            words=['<break time="80ms"/>'],
            starts=[0.0],
            language="en",
        )
        == []
    )


def test_cartesia_spell_token_matches_the_text_sent_for_synthesis():
    """Every normalized token has to be recognised by the word tracker.

    A token the tracker cannot place force-completes the slot, which emits all the
    text left unspoken — synthesis tags included — as one TTSTextFrame, ending the
    turn's word-level tracking.
    """
    text = "Hello, I love to <spell>1234</spell>."
    tracker = WordCompletionTracker(text, llm_text=text, user_facing_text=text)

    for word, _ in _process_word_timestamps(
        words=["Hello,", "I", "love", "to", "<spell>1234</spell>."],
        starts=[0.0, 0.1, 0.2, 0.3, 0.4],
        language="en",
    ):
        assert tracker.word_belongs_here(word), f"{word!r} was not recognised"
        tracker.add_word_and_check_complete(word)

    assert tracker.is_complete
    assert tracker.get_accumulated_user_facing_text() == text


class TestCartesiaUpdateSettingsFinalizesOldContext(unittest.IsolatedAsyncioTestCase):
    """A mid-reply voice/model/language change re-mints the turn context. The old
    context's still-pending sentence must be finalized first, or the already-heard
    prefix's word-timestamps land on no slot and drop out of the transcript.
    """

    async def _service_with_pending_prefix(self, old_ctx: str):
        service = CartesiaTTSService.__new__(CartesiaTTSService)
        service._name = "CartesiaTTSService#0"
        service._settings = CartesiaTTSService.Settings(
            model="sonic-3.5",
            voice="voiceA",
            language=Language.EN,
            generation_config=None,
            pronunciation_dict_id=None,
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

        # Stub I/O so _update_settings exercises the finalize/flush/re-mint logic
        # without a websocket. Capture frames the finalize pushes.
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

        await service._update_settings(CartesiaTTSService.Settings(voice="voiceB"))

        # The old context's pending sentence was force-promoted into a real slot.
        self.assertEqual([s.frame.text for s in seq._slots], ["Hi there"])
        self.assertEqual(seq._slots[0].context_id, old_ctx)
        # The finalize pushed the promoted sentence anchor downstream.
        self.assertTrue(
            any(isinstance(f, AggregatedTextFrame) and f.text == "Hi there" for f in pushed)
        )
        # The context was flushed and the turn context re-minted afterwards.
        self.assertEqual(service._flushed, old_ctx)
        self.assertEqual(service._turn_context_id, "ctx-new")

        # A word-timestamp for the flushed prefix (arriving on the OLD context during
        # playout) still finds the promoted slot and emits a progress frame.
        result = seq.process_word("Hi", pts=10, context_id=old_ctx)
        self.assertTrue(any(isinstance(f, TTSTextFrame) and f.text == "Hi" for f in result))
        progress = [f for f in result if isinstance(f, AggregatedTextProgressFrame)]
        self.assertEqual(len(progress), 1)
        self.assertEqual(progress[0].accumulated_text, "Hi")

    async def test_non_remint_change_does_not_finalize(self):
        # A change that does not re-mint the context (e.g. pronunciation_dict_id)
        # must NOT finalize — that would prematurely promote and mis-segment the
        # ongoing reply.
        old_ctx = "ctx-old"
        service, seq, pushed = await self._service_with_pending_prefix(old_ctx)

        await service._update_settings(CartesiaTTSService.Settings(pronunciation_dict_id="dict-1"))

        self.assertEqual(seq._slots, [])  # still pending, not promoted
        self.assertEqual(pushed, [])
        self.assertIsNone(service._flushed)
        self.assertEqual(service._turn_context_id, old_ctx)
