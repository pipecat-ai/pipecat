#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import json
import unittest
from unittest.mock import AsyncMock, MagicMock

import pytest

from pipecat.adapters.tts.services.cartesia_adapter import (
    CartesiaExpressionAdapter,
)
from pipecat.expressive.expression import DeliveryScope
from pipecat.frames.frames import (
    AggregatedTextFrame,
    AggregatedTextProgressFrame,
    LLMAssistantPushAggregationFrame,
    TextType,
    TTSStartedFrame,
    TTSTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMAssistantAggregator
from pipecat.services.cartesia.tts import (
    CartesiaHttpTTSService,
    CartesiaTTSService,
    GenerationConfig,
    _generation_config,
    language_to_cartesia_language,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TextAggregationMode, TTSContext
from pipecat.tests.utils import run_test
from pipecat.transcriptions.language import Language
from pipecat.utils.context.aggregated_frame_sequencer import AggregatedFrameSequencer
from pipecat.utils.context.word_completion_tracker import WordCompletionTracker
from pipecat.utils.context.word_timestamp_tracker import WordTimestamp
from pipecat.utils.string import TextPartForConcatenation, concatenate_aggregated_text


def _service(language: str) -> CartesiaTTSService:
    service = CartesiaTTSService.__new__(CartesiaTTSService)
    service._settings = TTSSettings(language=language)
    return service


@pytest.mark.parametrize(
    "language, expected",
    [
        (Language.ES, "es"),
        (Language.ES_MX, "es-MX"),
        (Language.EN_GB, "en-GB"),
        (Language.ES_AR, "es-AR"),
        (Language.ZH_TW, "zh"),
        (Language.ES_BO, "es"),
        (Language.DE_CH, "de"),
        (Language.NB, "no"),
        (Language.NB_NO, "no-NO"),
        (Language.FIL, "tl"),
        (Language.FIL_PH, "tl-PH"),
    ],
)
def test_language_to_cartesia_language_keeps_supported_locales(language, expected):
    assert language_to_cartesia_language(language) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("service_cls", [CartesiaTTSService, CartesiaHttpTTSService])
@pytest.mark.parametrize("model", ["sonic-3.6", "sonic-3"])
@pytest.mark.parametrize(
    "language, expected",
    [(Language.ES_MX, "es-MX"), ("en-GB", "en-GB"), (Language.DE_CH, "de"), (None, None)],
)
@pytest.mark.parametrize("update", [False, True])
async def test_requests_send_the_mapped_language(service_cls, model, language, expected, update):
    service = service_cls(
        api_key="test",
        settings=service_cls.Settings(
            model=model, voice="voice", language="en" if update else language
        ),
    )
    if update:
        await service._update_settings(service.Settings(language=language))
    if service_cls is CartesiaTTSService:
        payload = json.loads(service._build_msg(text="Hello", context_id="ctx"))
    else:
        response = MagicMock(status=200, read=AsyncMock(return_value=b"\x00\x00"))
        service._session = MagicMock()
        service._session.post.return_value.__aenter__.return_value = response
        _ = [frame async for frame in service.run_tts("Hello", "ctx")]
        payload = service._session.post.call_args.kwargs["json"]

    assert payload.get("language") == expected
    assert "locale" not in payload


@pytest.mark.asyncio
@pytest.mark.parametrize("service_cls", [CartesiaTTSService, CartesiaHttpTTSService])
@pytest.mark.parametrize("accent, expected", [("british", "british"), (None, None)])
async def test_requests_send_the_accent_only_when_set(service_cls, accent, expected):
    service = service_cls(
        api_key="test",
        settings=service_cls.Settings(voice="voice", accent=accent),
    )
    if service_cls is CartesiaTTSService:
        payload = json.loads(service._build_msg(text="Hello", context_id="ctx"))
    else:
        response = MagicMock(status=200, read=AsyncMock(return_value=b"\x00\x00"))
        service._session = MagicMock()
        service._session.post.return_value.__aenter__.return_value = response
        _ = [frame async for frame in service.run_tts("Hello", "ctx")]
        payload = service._session.post.call_args.kwargs["json"]

    assert payload.get("accent") == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("service_cls", [CartesiaTTSService, CartesiaHttpTTSService])
@pytest.mark.parametrize("normalization, expected", [("off", "off"), (None, None)])
async def test_requests_send_the_normalization_only_when_set(service_cls, normalization, expected):
    service = service_cls(
        api_key="test",
        settings=service_cls.Settings(voice="voice", normalization=normalization),
    )
    if service_cls is CartesiaTTSService:
        payload = json.loads(service._build_msg(text="Hello", context_id="ctx"))
    else:
        response = MagicMock(status=200, read=AsyncMock(return_value=b"\x00\x00"))
        service._session = MagicMock()
        service._session.post.return_value.__aenter__.return_value = response
        _ = [frame async for frame in service.run_tts("Hello", "ctx")]
        payload = service._session.post.call_args.kwargs["json"]

    assert payload.get("normalization") == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "language, delta, expected, rotated",
    [
        ("en-US", {"language": "en-GB"}, "en-GB", True),
        ("en-US", {"language": Language.ES_MX}, "es-MX", True),
        ("en-US", {"language": Language.DE_CH}, "de", True),
        ("en-US", {"language": "en-US"}, "en-US", False),
    ],
)
async def test_locale_update_rotates_context(language, delta, expected, rotated):
    service = CartesiaTTSService(
        api_key="test",
        settings=CartesiaTTSService.Settings(model="sonic-3.6", voice="voice", language=language),
    )
    service._turn_context_id = "old"
    service.flush_audio = AsyncMock()
    service.audio_context_available = lambda context_id: True

    await service._update_settings(service.Settings(**delta))

    assert (service._turn_context_id != "old") is rotated
    if rotated:
        service.flush_audio.assert_awaited_once_with(context_id="old")
    else:
        service.flush_audio.assert_not_awaited()
    payload = json.loads(service._build_msg(text="Hello", context_id=service._turn_context_id))
    assert payload["language"] == expected


def _process_word_timestamps(
    words: list[str], starts: list[float], language: str
) -> list[tuple[str, float]]:
    return _service(language)._normalize_word_timestamps(words, starts)


def test_word_timestamps_keep_their_ends():
    """Word ends Cartesia reports go along with their words, tags stripped."""
    service = _service("en")
    assert service._normalize_word_timestamps(
        ["Hello", '<break time="1s"/>', "world"], [0.0, 0.4, 1.4], [0.4, 1.4, 1.8]
    ) == [("Hello", 0.0, 0.4), ("world", 1.4, 1.8)]
    assert service._normalize_word_timestamps(["Hello"], [0.0]) == [("Hello", 0.0)]


async def _queued_word_timestamps(language, batches) -> list[WordTimestamp]:
    service = _service(language)
    service._websocket = MagicMock()
    service._websocket.__aiter__.return_value = [
        json.dumps({"type": "timestamps", "context_id": "ctx1", "word_timestamps": batch})
        for batch in batches
    ]
    service.audio_context_available = MagicMock(return_value=True)
    service.append_to_audio_context = AsyncMock()
    await service._process_messages()

    return [call.args[1] for call in service.append_to_audio_context.await_args_list]


async def _assert_timestamp_transcript(
    language: str, words: list[WordTimestamp], sentence: str, *, streaming: bool = False
) -> list[TTSTextFrame]:
    service = CartesiaTTSService(
        api_key="test",
        settings=CartesiaTTSService.Settings(voice="voice", language=language),
        text_aggregation_mode=TextAggregationMode.TOKEN
        if streaming
        else TextAggregationMode.SENTENCE,
    )
    service._tts_contexts["ctx1"] = TTSContext()
    service._push_sequencer_frames = AsyncMock()
    seq = service._aggregated_frame_sequencer
    for text in list(sentence) if streaming else [sentence]:
        frame = AggregatedTextFrame(text, TextType.TOKEN if streaming else TextType.SENTENCE)
        await service._register_spoken_text(frame, text, "ctx1")
    await seq.finalize("ctx1")
    frames = [
        frame
        for w in words
        for frame in seq.process_word(
            w.word, int(w.timestamp * 1e9), w.context_id, w.includes_inter_frame_spaces
        )
        if isinstance(frame, TTSTextFrame)
    ]
    frames.extend(f for f in seq.force_complete("ctx1", 0) if isinstance(f, TTSTextFrame))
    context = LLMContext()
    await run_test(
        LLMAssistantAggregator(context),
        frames_to_send=[TTSStartedFrame(), *frames, LLMAssistantPushAggregationFrame()],
    )
    assert context.messages == [{"role": "assistant", "content": sentence}]
    return frames


@pytest.mark.asyncio
@pytest.mark.parametrize("language", ["th", "th-TH"])
async def test_thai_timestamps_preserve_spacing_and_marks_in_the_transcript(language):
    batches = [
        {
            "words": ["ส", "ว", "ั", "ส<spell>ด</spell>", "ี"],
            "start": [0, 1, 2, 3, 4],
            "end": [1, 2, 3, 4, 5],
        },
        {
            "words": ["  ", '<break time="80ms"/>', "ค", "่", "ะ\t"],
            "start": [5, 6, 7, 8, 9],
            "end": [6, 7, 8, 9, 10],
        },
        {"words": [" คุ", "ณ"], "start": [10, 11]},
    ]
    words = await _queued_word_timestamps(language, batches)
    assert [w.word for w in words] == ["ส", "ว", "ั", "สด", "ี", "  ", "ค", "่", "ะ\t", " คุ", "ณ"]
    assert [w.timestamp for w in words] == [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11]
    assert [w.end for w in words] == [1, 2, 3, 4, 5, 6, 8, 9, 10, None, None]

    sentence = "สวัสดี  ค่ะ\t คุณ"
    frames = await _assert_timestamp_transcript(language, words, sentence)
    assert "".join(frame.text for frame in frames) == sentence


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "language,batches,sentence",
    [
        ("en", [["Please", "spell"], ["<spell>1234</spell>."]], "Please spell 1234."),
        ("ja", [["で", "す"], ["。"]], "です。"),
        ("zh", [["你", "好"], ["。"]], "你好。"),
    ],
)
async def test_timestamp_spacing_in_the_assistant_transcript(language, batches, sentence):
    words = await _queued_word_timestamps(
        language, [{"words": group, "start": [0.0] * len(group)} for group in batches]
    )
    await _assert_timestamp_transcript(language, words, sentence)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_thai_tokens_can_split_a_mark_or_cross_a_sentence_boundary(streaming):
    tokens = ["สวัสด", "ี ค่ะ", "! ขอที่อย", "ู่"]
    words = await _queued_word_timestamps(
        "th", [{"words": [token], "start": [i]} for i, token in enumerate(tokens)]
    )
    sentence = "".join(tokens)
    frames = await _assert_timestamp_transcript("th", words, sentence, streaming=streaming)
    assert "".join(frame.text for frame in frames) == sentence


def test_final_thai_mark_precedes_an_unspoken_synthesis_tag():
    tracker = WordCompletionTracker("สวัสดี<break/>", llm_text="สวัสดี", literal_word_timestamps=True)
    assert not tracker.add_word_and_check_complete("สวัสด")
    assert tracker.add_word_and_check_complete("ี")
    assert tracker.get_llm_consumed() == "สวัสดี"


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


def test_cartesia_chinese_word_timestamps_keep_one_entry_per_character():
    assert _process_word_timestamps(
        words=["你", "好", "。"],
        starts=[0.0, 0.1, 0.2],
        language="zh",
    ) == [("你", 0.0), ("好", 0.1), ("。", 0.2)]


def test_cartesia_japanese_word_timestamps_keep_one_entry_per_character():
    assert _process_word_timestamps(
        words=["こ", "ん", "に", "ち", "は", "。"],
        starts=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        language="ja",
    ) == [
        ("こ", 0.0),
        ("ん", 0.1),
        ("に", 0.2),
        ("ち", 0.3),
        ("は", 0.4),
        ("。", 0.5),
    ]


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


def test_cartesia_laughter_token_is_kept():
    """It marks when the laughter sound is performed."""
    assert _process_word_timestamps(
        words=["ancient.", "[laughter]", "Let"],
        starts=[0.0, 0.5, 1.0],
        language="en",
    ) == [("ancient.", 0.0), ("[laughter]", 0.5), ("Let", 1.0)]


class TestCartesiaExpressionAdapter:
    def test_capabilities(self):
        capabilities = CartesiaExpressionAdapter().capabilities()
        delivery = [term.name for term in capabilities.delivery]
        assert "excited" in delivery
        assert "neutral" not in delivery
        assert [term.name for term in capabilities.sounds] == ["laughter"]
        assert capabilities.delivery_scope == DeliveryScope.SENTENCE
        assert not capabilities.emphasis

    def test_emotion_formatted_where_it_changes(self):
        adapter = CartesiaExpressionAdapter()
        assert adapter.format_text("Hi.", ["excited"], None) == '<emotion value="excited"/> Hi.'
        assert adapter.format_text("Hi.", ["excited"], ["excited"]) == "Hi."
        assert adapter.format_text("Hi.", ["sad"], ["excited"]) == '<emotion value="sad"/> Hi.'
        assert adapter.format_text("Hi.", [], None) == "Hi."

    def test_return_to_baseline_formats_neutral(self):
        adapter = CartesiaExpressionAdapter()
        assert adapter.format_text("Hi.", [], ["sad"]) == '<emotion value="neutral"/> Hi.'

    def test_of_several_delivery_names_the_first_is_formatted(self):
        adapter = CartesiaExpressionAdapter()
        assert adapter.format_text("Hi.", ["sad", "calm"], None) == '<emotion value="sad"/> Hi.'
        assert adapter.format_text("Hi.", ["sad", "tired"], ["sad", "calm"]) == "Hi."

    def test_slow_and_fast_set_the_speed_not_the_emotion(self):
        adapter = CartesiaExpressionAdapter()
        assert {"slow", "fast"} <= {term.name for term in adapter.capabilities().delivery}
        assert adapter.format_text("Hi.", ["slow", "sad"], None) == '<emotion value="sad"/> Hi.'
        assert adapter.format_text("Hi.", ["sad", "slow"], ["sad"]) == "Hi."
        assert adapter.request_settings(["sad", "slow"]) == {"speed": 0.7}
        assert adapter.request_settings(["fast"]) == {"speed": 1.3}
        assert adapter.request_settings(["sad"]) == {}

    def test_sounds_and_pauses(self):
        adapter = CartesiaExpressionAdapter()
        assert adapter.format_sound("laughter") == " [laughter] "
        assert adapter.format_pause(600) == ' <break time="600ms"/> '


def test_a_request_in_a_speed_delivery_states_its_speed():
    assert _generation_config(None, {"speed": 0.7}) == {"speed": 0.7}
    assert _generation_config(GenerationConfig(volume=1.2), {"speed": 1.3}) == {
        "volume": 1.2,
        "speed": 1.3,
    }
    # Without a speed delivery, the configured speed is kept.
    assert _generation_config(GenerationConfig(speed=0.9), {}) == {"speed": 0.9}
    assert _generation_config(None, {}) is None


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
        service._expression_formatter = None
        # Real streaming sequencer with a mid-sentence prefix pending on the turn ctx.
        seq = AggregatedFrameSequencer(name=service._name, streaming=True)
        service._aggregated_frame_sequencer = seq
        service._turn_context_id = old_ctx
        for token in ("Hi", " there"):
            frame = AggregatedTextFrame(token, TextType.SENTENCE, raw_text=token)
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


class TestCartesiaCJKPunctuationTokens(unittest.IsolatedAsyncioTestCase):
    """Cartesia reports a Japanese sentence's final mark as its own token. Each mark
    must reach the conversation context exactly once, whether it arrives with the
    character before it or opens the next timestamps message.
    """

    async def _queued_words(self, batches: list[list[str]]) -> list[WordTimestamp]:
        return await _queued_word_timestamps(
            "ja", [{"words": words, "start": [0.0] * len(words)} for words in batches]
        )

    async def test_marks_attach_to_the_preceding_character(self):
        queued = await self._queued_words([["で", "す", "。"], ["歯", "科"]])
        self.assertEqual([w.word for w in queued], ["で", "す。", "歯", "科"])
        self.assertTrue(all(w.includes_inter_frame_spaces for w in queued))

    async def test_context_matches_llm_text(self):
        sentences = ["です。", "歯科ですか？"]
        queued = await self._queued_words(
            [["で", "す", "。"], ["歯", "科"], ["で", "す"], ["か"], ["？"]]
        )

        seq = AggregatedFrameSequencer()
        for sentence in sentences:
            frame = AggregatedTextFrame(sentence, TextType.SENTENCE, raw_text=sentence)
            await seq.register_spoken(
                frame, "ctx1", sentence, append_to_context=True, includes_inter_frame_spaces=True
            )
        frames = []
        for i, w in enumerate(queued):
            frames.extend(
                seq.process_word(w.word, (i + 1) * 10, w.context_id, w.includes_inter_frame_spaces)
            )

        context_text = concatenate_aggregated_text(
            [
                TextPartForConcatenation(
                    f.raw_text or f.text, includes_inter_part_spaces=f.includes_inter_frame_spaces
                )
                for f in frames
                if isinstance(f, TTSTextFrame) and f.append_to_context
            ]
        )
        self.assertEqual(context_text, "".join(sentences))


def test_spell_tag_aggregator_keeps_the_settings_language():
    service = CartesiaTTSService(
        api_key="test-api-key",
        settings=CartesiaTTSService.Settings(voice="voice", language=Language.FR),
    )

    assert service._text_aggregator.language == "fr"
