#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import string

import pytest

from pipecat.services.cartesia.tts import (
    NON_SPEECH_CHARACTERS,
    CartesiaTTSService,
)
from pipecat.services.settings import TTSSettings
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


def _configured_service(**kwargs) -> CartesiaTTSService:
    return CartesiaTTSService(
        api_key="test-key",
        settings=CartesiaTTSService.Settings(voice="test-voice"),
        **kwargs,
    )


def test_cartesia_strips_leading_punctuation_by_default():
    assert _configured_service()._leading_non_speech_characters == NON_SPEECH_CHARACTERS


def test_cartesia_leading_non_speech_characters_can_be_overridden():
    service = _configured_service(leading_non_speech_characters="")
    assert service._leading_non_speech_characters == ""


@pytest.mark.parametrize(
    "text, expected_skipped",
    [
        # An LLM resuming an interrupted reply opens with these; alone they normalize to
        # an empty transcript, which the API rejects.
        ("...", True),
        ("…", True),
        ('"...', True),
        ("--", True),
        (" ... ", True),
        # Anything carrying speech is left untouched, punctuation included.
        ("  ... hello", False),
        ("Perfetto...", False),
        ("hello", False),
        # Symbols are voiced as words, so they must survive.
        ("$5 a month", False),
        ("& company", False),
        ("50% off", False),
    ],
)
def test_cartesia_skips_only_chunks_with_nothing_to_voice(text: str, expected_skipped: bool):
    skipped = not text.strip(string.whitespace + NON_SPEECH_CHARACTERS)
    assert skipped is expected_skipped
