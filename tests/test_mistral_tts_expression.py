#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for Mistral TTS in expressive conversation."""

import pytest

from pipecat.adapters.tts.base_expression_adapter import BaseExpressionAdapter
from pipecat.adapters.tts.services.mistral_adapter import MistralExpressionAdapter


class TestMistralExpressionAdapter:
    def test_capabilities(self):
        capabilities = MistralExpressionAdapter().capabilities()
        assert not capabilities.allows_free_text_sounds
        assert not capabilities.allows_free_text_delivery
        assert not capabilities.emphasis
        assert capabilities.pause_range_ms is None
        assert capabilities.delivery_name("sympathetic") == "warm"
        assert capabilities.sound_name("clear throat") == "throat clearing"

    def test_tags_where_emotion_or_pace_changes(self):
        adapter = MistralExpressionAdapter()
        assert adapter.format_text("Hi.", ["happy"], None) == "[emotion=happy] Hi."
        assert adapter.format_text("Hi.", ["happy"], ["happy"]) == "Hi."
        assert adapter.format_text("A B 7", ["calm", "slow"], ["calm"]) == "[pace=slow] A B 7"
        assert adapter.format_text("Hi.", ["calm", "fast"], None) == (
            "[emotion=calm][pace=fast] Hi."
        )

    def test_a_return_to_the_normal_pace_writes_moderate(self):
        adapter = MistralExpressionAdapter()
        assert adapter.format_text(", thanks.", ["calm"], ["calm", "slow"]) == (
            "[pace=moderate] , thanks."
        )

    def test_a_return_to_the_baseline_writes_no_emotion(self):
        assert MistralExpressionAdapter().format_text("Bye.", [], ["happy"]) == "Bye."

    def test_one_emotion_at_a_time(self):
        adapter = MistralExpressionAdapter()
        assert adapter.format_text("Hi.", ["happy", "warm"], None) == "[emotion=happy] Hi."

    def test_sounds(self):
        assert MistralExpressionAdapter().format_sound("laugh hard") == "[laugh hard]"


@pytest.mark.parametrize(
    "model, adapter",
    [
        ("voxtral-mini-tts-3-0", MistralExpressionAdapter),
        ("voxtral-mini-tts-3", MistralExpressionAdapter),
        ("voxtral-mini-tts-2603", BaseExpressionAdapter),
    ],
)
def test_expression_adapter_follows_the_model(model, adapter):
    tts = pytest.importorskip("pipecat.services.mistral.tts", exc_type=ImportError)
    service = tts.MistralTTSService(
        api_key="key", settings=tts.MistralTTSService.Settings(model=model)
    )
    assert type(service.get_expression_adapter()) is adapter
