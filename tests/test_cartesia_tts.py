#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from pipecat.services.cartesia.utils import process_word_timestamps_for_language


def test_cartesia_cjk_word_timestamps_join_chinese_without_spaces():
    assert process_word_timestamps_for_language(
        words=["你", "好", "。"],
        starts=[0.0, 0.1, 0.2],
        language="zh",
    ) == [("你好。", 0.0)]


def test_cartesia_cjk_word_timestamps_join_japanese_without_spaces():
    assert process_word_timestamps_for_language(
        words=["こ", "ん", "に", "ち", "は", "。"],
        starts=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        language="ja",
    ) == [("こんにちは。", 0.0)]


def test_cartesia_korean_word_timestamps_join_with_spaces():
    assert process_word_timestamps_for_language(
        words=["안녕하세요", "반갑습니다"],
        starts=[0.0, 0.2],
        language="ko",
    ) == [("안녕하세요 반갑습니다", 0.0)]
